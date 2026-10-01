"""Tests for the code-state / environment digests in the drag-session provenance sidecar."""

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def dsb():
    # the harness pops QT_QPA_PLATFORM at import (native backend); keep the test session's value
    saved = os.environ.get("QT_QPA_PLATFORM")
    try:
        from paper_benchmarks import drag_session_benchmark
    finally:
        if saved is not None:
            os.environ["QT_QPA_PLATFORM"] = saved
    return drag_session_benchmark


def _git(root, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=root, check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Committed repo with one tracked file in each measured tree and one outside."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    root = tmp_path / "repo"
    for rel in ("src/pkg.py", "paper_benchmarks/bench.py", "other/notes.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_clean_repo(dsb, repo):
    s = dsb._code_state(repo)
    assert re.fullmatch(r"[0-9a-f]{40}", s["base_commit"])
    assert s["dirty"] is False
    assert s["tracked_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    assert s["untracked"] == {}
    assert HEX64.match(s["combined_sha256"])


def test_dirty_repo_digests(dsb, repo):
    clean = dsb._code_state(repo)
    (repo / "src/pkg.py").write_text("# edited\n")
    (repo / "paper_benchmarks/new.py").write_bytes(b"x = 1\n")
    (repo / "other/untracked.py").write_text("ignored: outside src/ and paper_benchmarks/\n")
    s = dsb._code_state(repo)
    assert s["base_commit"] == clean["base_commit"]
    assert s["dirty"] is True
    assert HEX64.match(s["tracked_diff_sha256"]) and s["tracked_diff_sha256"] != clean["tracked_diff_sha256"]
    assert s["untracked"] == {"paper_benchmarks/new.py": hashlib.sha256(b"x = 1\n").hexdigest()}
    # combined digest follows the documented formula and tracks untracked content
    expected = "\n".join([s["tracked_diff_sha256"], f"paper_benchmarks/new.py {s['untracked']['paper_benchmarks/new.py']}"])
    assert s["combined_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    (repo / "paper_benchmarks/new.py").write_bytes(b"x = 2\n")
    assert dsb._code_state(repo)["combined_sha256"] != s["combined_sha256"]
    # a change outside the measured trees does not alter the identity
    (repo / "other/notes.py").write_text("# edited too\n")
    assert dsb._code_state(repo)["tracked_diff_sha256"] == s["tracked_diff_sha256"]


def test_not_a_repo_never_raises(dsb, tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    s = dsb._code_state(tmp_path)
    for k in ("base_commit", "dirty", "tracked_diff_sha256", "untracked", "combined_sha256"):
        assert s[k] is None, k


def test_env_freeze_sha256(dsb, monkeypatch):
    digest = dsb._env_freeze_sha256()
    if digest is None:
        pytest.skip("pip freeze unavailable in this environment")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # as the harness does
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, check=True, env=env).stdout
    assert digest == hashlib.sha256(out).hexdigest()
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    assert dsb._env_freeze_sha256() is None


def test_sidecar_records_start_end_and_unchanged(dsb, repo, tmp_path, monkeypatch, qapp):
    monkeypatch.setattr(dsb, "_code_state", functools.partial(dsb._code_state, repo))
    monkeypatch.setattr(dsb, "_env_freeze_sha256", lambda: "0" * 64)
    args = argparse.Namespace(tool="modern", legacy_sync="full", edit_view="api", drag_protocol="random3d",
                              tri_backend="default", thumbnails="off", ns="10", ortho_follow=False,
                              warmup=0, pacing="sleep", replicate=0, drags=1)
    csv_path = str(tmp_path / "run.csv")

    dsb._write_provenance(args, csv_path, "t0", None, dsb._code_state())
    meta = json.loads((tmp_path / "run.meta.json").read_text())
    assert meta["code_state"] == meta["code_state_end"]
    assert meta["code_state_unchanged"] is True
    assert HEX64.match(meta["env_freeze_sha256"])

    start = dsb._code_state()
    (repo / "src/pkg.py").write_text("# edited while measuring\n")
    dsb._write_provenance(args, csv_path, "t0", None, start)
    meta = json.loads((tmp_path / "run.meta.json").read_text())
    assert meta["code_state"]["combined_sha256"] != meta["code_state_end"]["combined_sha256"]
    assert meta["code_state_unchanged"] is False

    # unknown identity is never reported as unchanged
    dsb._write_provenance(args, csv_path, "t0", None, None)
    assert json.loads((tmp_path / "run.meta.json").read_text())["code_state_unchanged"] is False
