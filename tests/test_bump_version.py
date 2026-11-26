"""Tests for scripts/bump_version.py on temporary copies of the release files."""

from __future__ import annotations

import configparser
import importlib.util
import re
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "bump_version.py"
RELEASE_FILES = (
    "setup.cfg",
    "CITATION.cff",
    "docs/conf.py",
    "docs/changelog.rst",
    "README.md",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("bump_version", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _copy_release_tree(tmp_path: Path, mod) -> tuple[Path, str, str]:
    """Copy the release files and return ``(root, current_version, next_version)``.

    The checkout is either between releases (the changelog has the
    "(unreleased)" section of the next version, which is then the target) or at
    a release commit (it has none). In the second case a section for the next
    patch version is added to the copy, so the tests do not depend on the
    release state of the checkout.
    """
    for rel in RELEASE_FILES:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)
    current = mod.read_current_version(tmp_path)
    changelog = tmp_path / "docs" / "changelog.rst"
    text = changelog.read_text(encoding="utf-8")
    unreleased = re.search(r"^v(\d+\.\d+\.\d+) \(unreleased\)$", text, flags=re.M)
    if unreleased:
        target = unreleased.group(1)
    else:
        major, minor, patch = (int(part) for part in current.split("."))
        target = f"{major}.{minor}.{patch + 1}"
        header = f"v{target} (unreleased)"
        title = "Changelog\n=========\n\n"
        assert text.startswith(title)
        text = title + f"{header}\n{'-' * len(header)}\n\n" + text[len(title):]
        changelog.write_text(text, encoding="utf-8")
    return tmp_path, current, target


def test_bump_version_dry_run_writes_nothing(tmp_path, capsys):
    mod = _load_module()
    root, _current, target = _copy_release_tree(tmp_path, mod)
    before = {rel: (root / rel).read_text(encoding="utf-8") for rel in RELEASE_FILES}

    rc = mod.main([target, "--date", "2026-11-25", "--dry-run", "--root", str(root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would write setup.cfg" in out
    assert "would write CITATION.cff" in out
    assert "would write docs/conf.py" in out
    assert "would write docs/changelog.rst" in out
    assert "would write README.md" in out
    assert "15 README links" in out

    after = {rel: (root / rel).read_text(encoding="utf-8") for rel in RELEASE_FILES}
    assert after == before


def test_bump_version_updates_all_five_files_consistently(tmp_path):
    mod = _load_module()
    root, _current, target = _copy_release_tree(tmp_path, mod)

    rc = mod.main([target, "--date", "2026-11-25", "--root", str(root)])
    assert rc == 0

    cfg = configparser.ConfigParser()
    cfg.read(root / "setup.cfg")
    assert cfg["metadata"]["version"] == target

    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {target}\n" in cff
    assert "date-released: 2026-11-25\n" in cff

    conf = (root / "docs/conf.py").read_text(encoding="utf-8")
    assert f"release = '{target}'\n\n" in conf

    changelog = (root / "docs/changelog.rst").read_text(encoding="utf-8")
    header = f"v{target} (2026-11-25)"
    assert "(unreleased)" not in changelog
    assert f"{header}\n{'-' * len(header)}\n" in changelog

    readme = (root / "README.md").read_text(encoding="utf-8")
    pins = re.findall(
        r"(?:https://raw\.githubusercontent\.com/N-Friederich/napari-turbobox"
        r"|https://github\.com/N-Friederich/napari-turbobox/(?:blob|tree))/(v\d+\.\d+\.\d+)/",
        readme,
    )
    assert len(pins) == 15
    assert set(pins) == {f"v{target}"}


@pytest.mark.parametrize("bad_version", ["0.2.0", "current", "v0.1.1", "0.1"])
def test_bump_version_rejects_invalid_or_reserved_versions(tmp_path, bad_version):
    mod = _load_module()
    root, current, _target = _copy_release_tree(tmp_path, mod)
    if bad_version == "current":
        bad_version = current
    before = {rel: (root / rel).read_text(encoding="utf-8") for rel in RELEASE_FILES}

    rc = mod.main([bad_version, "--date", "2026-11-25", "--root", str(root)])
    assert rc == 1

    after = {rel: (root / rel).read_text(encoding="utf-8") for rel in RELEASE_FILES}
    assert after == before
