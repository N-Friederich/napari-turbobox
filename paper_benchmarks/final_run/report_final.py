"""Numbers for the README and the paper from the final benchmark run.

Aggregates paper_results/{main,large_n,tutorial_full} with the figure
script's own ``aggregate()`` (bootstrap generator re-seeded before every
directory, as in a fresh process), adds the large-N and full-reassignment
ratios, the interactive limits and the memory/resolution re-runs, and writes
paper_results/RESULTS.md and paper_results/results_numbers.json. TB_RESULTS selects another
results directory with the same layout (including environment/pip_freeze_benchmark_env.txt);
its runs must also come from the reference environment, or the checks below stop the script.

Before anything is aggregated or written, every stage is checked against the
design of run_final.sh and run_post.sh: all runs and replicates present,
every configuration row present once and complete, the drag count and pacing
of the stage, one protocol, one code state for all stages, the recorded
environment freeze, AC power and normal priority. Any problem stops the
script with exit code 1, and RESULTS.md and results_numbers.json are left as
they are.

Run from the repository root:  PYTHONPATH=src:. python paper_benchmarks/final_run/report_final.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import paper_benchmarks.make_final_figures as mff

V2 = Path(os.environ.get("TB_RESULTS", "paper_results"))  # another directory with the same layout
MAIN = os.environ.get("TB_MAIN", "main")  # main_prelim for the preliminary run
LARGE = os.environ.get("TB_LARGE", "large_n")  # large_n_prelim for the preliminary run
MAIN_DRAGS = int(os.environ.get("TB_MAIN_DRAGS", "50"))  # as MAIN_DRAGS in run_final.sh
LARGE_DRAGS = int(os.environ.get("TB_LARGE_DRAGS", "50"))  # as LARGE_DRAGS in run_final.sh
THR = mff.THR
TB, BB = "modern", "legacy"


def agg(d):
    mff._rng = np.random.default_rng(0)
    S, sp = mff.aggregate(V2 / d, V2 / d)
    return S, sp


def ratio_ci(leg, mod):
    mff._rng = np.random.default_rng(0)
    return mff._ratio_ci(leg, mod)


def run_medians(d, tool, panel, n, v):
    rows = []
    for f in sorted((V2 / d).glob("drag_session_*_rep*.csv")):
        if "dist" in f.name:
            continue
        df = pd.read_csv(f)
        g = df[(df.tool == tool) & (df.panel == panel) & (df.n_boxes == n) & (df.n_viewers.astype(str) == v)]
        rows.extend(g.sub_median_ms.tolist())
    return np.asarray(rows, float)


def interactive(r):
    return bool(r.median_ms <= THR and r.p95_ms <= THR and r.p99_ms <= THR)


def fmt(r):
    return f"{r.median_ms:.1f} [{r.ci_lo:.1f}–{r.ci_hi:.1f}]"


SAME_ACROSS_TOOLS = ("edit_view", "drag_protocol", "ortho_follow", "warmup", "pacing", "env_freeze_sha256",
                     "napari", "napari_bbox", "python", "qt_version", "triangulation_backend", "drags", "ns")
# napari-bbox triangulates with its own code, so the resolved mesher differs between the tools by construction
SAME_PER_TOOL = {TB: ("thumbnails", "triangulation_mesher"), BB: ("legacy_sync", "triangulation_mesher")}


# Expected design of every drag stage (run_final.sh): replicates per tool, drags per configuration,
# pacing, napari-bbox synchronization, edit path, and the (panel, N, V) rows of every per-run CSV.
V_A, V_B = "3+3D", ("1", "2", "4", "8")
N_MAIN, N_LARGE, N_SLEEP = (10, 50, 100, 200, 500, 1000), (2000, 5000, 10000), (10, 100, 1000, 10000)
MOVES = 60  # mouse moves per drag (N_SUB in drag_session_benchmark.py)
DUMP = {("a", 100, V_A), ("b", 100, "4")}  # configurations with every update saved (DUMP in the harness)


def _grid(ns, panel_b):
    rows = {("a", n, V_A) for n in ns}
    if panel_b:
        rows |= {("b", 100, v) for v in V_B}
    return rows


def _stage(reps, drags, pacing, sync, edit, grid):
    return {"reps": reps, "drags": drags, "pacing": pacing, "sync": sync, "edit": edit, "grid": grid}


STAGES = {
    MAIN: _stage({TB: 5, BB: 5}, MAIN_DRAGS, "spin", "per_box_quiet", "mouse", _grid(N_MAIN, True)),
    LARGE: _stage({TB: 5, BB: 5}, LARGE_DRAGS, "spin", "per_box_quiet", "mouse", _grid(N_LARGE, False)),
    "sleep_pacing": _stage({TB: 3, BB: 3}, 10, "sleep", "per_box_quiet", "mouse", _grid(N_SLEEP, True)),
    "sleep_pacing_spin": _stage({TB: 3, BB: 3}, 10, "spin", "per_box_quiet", "mouse", _grid(N_SLEEP, True)),
    "tutorial_full": _stage({BB: 3}, 10, "spin", "full", "api", _grid(N_MAIN, False)),
}
PROTOCOL = {"drag_protocol": "xy_visible", "ortho_follow": True, "warmup": 1, "thumbnails": "off"}  # every run
STATS = ("sub_median_ms", "sub_p95_ms", "sub_p99_ms", "commit_median_ms")
FREEZE = V2 / "environment" / "pip_freeze_benchmark_env.txt"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_problems(name, m, codes, states):
    """Same code state at start and end, reference environment, AC power and nice 0."""
    out = []
    start = (m.get("code_state") or {}).get("combined_sha256")
    end = (m.get("code_state_end") or {}).get("combined_sha256")
    if start is None or start != end:
        out.append(f"{name}: code state at the start ({start}) and at the end ({end}) differ or are missing")
    codes.add(start)
    if m.get("env_differs_from_reference") != {}:
        out.append(f"{name}: environment differs from the reference: {m.get('env_differs_from_reference')}")
    for when, st in states.items():
        st = m.get(st)
        if not isinstance(st, dict):
            out.append(f"{name}: no process state ({when})")
            continue
        if st.get("nice") != 0:
            out.append(f"{name}: nice = {st.get('nice')} ({when})")
        if "AC Power" not in str(st.get("power")):
            out.append(f"{name}: not on AC power ({when}): {st.get('power')}")
    return out


def _missing_values(name, df, cols):
    absent = [c for c in cols if c not in df]
    if absent:
        return [f"{name}: columns missing: {absent}"]
    bad = int((~np.isfinite(df[list(cols)].astype(float))).to_numpy().sum())
    return [f"{name}: {bad} missing or non-finite values in {list(cols)}"] if bad else []


def _csv_problems(csv, m, exp):
    """One row per expected configuration, the stage's drag count, no missing statistic."""
    df, name = pd.read_csv(csv), csv.name
    out = _missing_values(name, df, STATS)
    rows = list(zip(df.panel, df.n_boxes.astype(int), df.n_viewers.astype(str)))
    if len(rows) != len(set(rows)):
        out.append(f"{name}: duplicate configuration rows")
    if exp["grid"] - set(rows):
        out.append(f"{name}: configurations missing: {sorted(exp['grid'] - set(rows))}")
    if set(rows) - exp["grid"]:
        out.append(f"{name}: unexpected configurations: {sorted(set(rows) - exp['grid'])}")
    if set(df.tool) != {m.get("tool")} or set(df.replicate) != {m.get("replicate")}:
        out.append(f"{name}: tool or replicate column does not match the sidecar")
    if not ((df.n_drags == exp["drags"]).all() and (df.n_sub == exp["drags"] * MOVES).all()):
        out.append(f"{name}: n_drags / n_sub are not {exp['drags']} / {exp['drags'] * MOVES} in every row")
    return out


def _dist_problems(d, csv, m, exp):
    """The saved per-update times (DUMP configurations of the stage): all drags of every run."""
    want = {f"{p}_N{n}_V{v}" for p, n, v in DUMP & exp["grid"]}
    dist = d / csv.name.replace("drag_session_", "drag_session_dist_", 1)
    if not want:
        return [f"{dist.name}: not expected in this stage"] if dist.exists() else []
    if not dist.is_file():
        return [f"{dist.name}: missing"]
    g = pd.read_csv(dist)
    out = _missing_values(dist.name, g, ("sub_ms",))
    counts = g.groupby("config").size().to_dict()
    if set(counts) != want or any(c != exp["drags"] * MOVES for c in counts.values()):
        out.append(f"{dist.name}: updates per configuration {counts}, expected {exp['drags'] * MOVES} for {sorted(want)}")
    if set(g.tool) != {m.get("tool")} or set(g.replicate) != {m.get("replicate")}:
        out.append(f"{dist.name}: tool or replicate column does not match the sidecar")
    return out


def validate(stage, codes, freeze):
    """Check one drag stage against its expected design. Returns (problems, runs per tool)."""
    exp, d = STAGES[stage], V2 / stage
    k = {TB: 0, BB: 0}
    csvs = [f for f in sorted(d.glob("drag_session_*_rep*.csv")) if "dist" not in f.name]
    metas = {f.name[: -len(".meta.json")]: json.loads(f.read_text()) for f in sorted(d.glob("drag_session_*.meta.json"))}
    if not metas:
        return [f"{stage}: no runs"], k
    problems = [f"{f.name}: no .meta.json" for f in csvs if f.stem not in metas]
    ns_a = ",".join(str(n) for n in sorted(n for p, n, _ in exp["grid"] if p == "a"))
    for name, m in metas.items():
        problems += _run_problems(name, m, codes, {"start": "process_state_start", "end": "process_state_end"})
        if not m.get("code_state_unchanged"):
            problems.append(f"{name}: code changed during the run")
        if m.get("env_freeze_sha256") != freeze:
            problems.append(f"{name}: env_freeze_sha256 is not the sha256 of {FREEZE}")
        want = dict(PROTOCOL, drags=exp["drags"], pacing=exp["pacing"], edit_view=exp["edit"], ns=ns_a)
        if m.get("tool") == BB:
            want["legacy_sync"] = exp["sync"]
        problems += [f"{name}: {key} = {m.get(key)!r}, expected {val!r}" for key, val in want.items() if m.get(key) != val]
        csv = d / str(m.get("csv"))
        if not csv.is_file():
            problems.append(f"{name}: CSV {m.get('csv')} missing")
            continue
        problems += _csv_problems(csv, m, exp) + _dist_problems(d, csv, m, exp)
    ms = list(metas.values())
    for key in SAME_ACROSS_TOOLS:
        vals = {json.dumps(m.get(key)) for m in ms}
        if len(vals) != 1:
            problems.append(f"{stage}: {key} differs between runs: {sorted(vals)}")
    for tool in (TB, BB):
        tm = [m for m in ms if m.get("tool") == tool]
        for key in SAME_PER_TOOL[tool]:
            vals = {json.dumps(m.get(key)) for m in tm}
            if len(vals) > 1:
                problems.append(f"{stage}: {tool}: {key} differs between runs: {sorted(vals)}")
        reps, expected = sorted(m.get("replicate") for m in tm), list(range(exp["reps"].get(tool, 0)))
        if reps != expected:
            problems.append(f"{stage}: {tool} replicates {reps}, expected {expected}")
        k[tool] = len(reps)
    return problems, k


def _read_csv(path, problems):
    if not path.is_file():
        problems.append(f"{path}: missing")
        return None
    return pd.read_csv(path)


def validate_post(codes):
    """Memory (panel c) and image size (panel f), run_post.sh: complete tables, code state, figure copies."""
    problems = []
    metas = {"memory": (V2 / "memory" / "benchmark_memory.meta.json", {}),
             "resolution": (V2 / "resolution" / "benchmark_resolution.meta.json",
                            {"thumbnails": "off", "pacing": "spin", "edit_view": "mouse", "ortho_follow": True,
                             "warmup": 1, "drags_per_size": 12})}
    for name, (path, want) in metas.items():
        if not path.is_file():
            problems.append(f"{path}: missing")
            continue
        m = json.loads(path.read_text())
        problems += _run_problems(name, m, codes, {"run": "process_state"})
        problems += [f"{name}: {key} = {m.get(key)!r}, expected {val!r}" for key, val in want.items() if m.get(key) != val]
    mem = _read_csv(V2 / "memory" / "benchmark_memory.csv", problems)
    if mem is not None:
        rows = list(zip(mem.plugin, mem.n_boxes.astype(int)))
        want = {(p, n) for p in ("napari-turbobox", "napari-bbox") for n in N_MAIN + N_LARGE}
        if len(rows) != len(set(rows)) or set(rows) != want:
            problems.append(f"benchmark_memory.csv: {len(rows)} rows; missing {sorted(want - set(rows))}, "
                            f"unexpected {sorted(set(rows) - want)}")
        problems += _missing_values("benchmark_memory.csv", mem, ("memory_increase_mb",))
    ds = _read_csv(V2 / "memory" / "benchmark_memory_datastructure.csv", problems)
    if ds is not None:
        for tool, cols in ((TB, ("store_bytes", "index_bytes", "store_plus_index")), (BB, ("total_alloc_bytes",))):
            t = ds[ds.tool == tool]
            if sorted(t.n_boxes.astype(int)) != [100, 1000, 10000]:
                problems.append(f"benchmark_memory_datastructure.csv: {tool} rows N = {sorted(t.n_boxes)}")
            problems += _missing_values(f"benchmark_memory_datastructure.csv ({tool})", t, cols)
    res = _read_csv(V2 / "resolution" / "benchmark_resolution.csv", problems)
    if res is not None:
        sizes = sorted(res.image_xy.astype(int))
        if sizes != [128, 256, 512, 1024, 2048] or set(res.n_boxes) != {100} or set(res.n_viewers) != {4}:
            problems.append(f"benchmark_resolution.csv: image sizes {sizes}, N {set(res.n_boxes)}, V {set(res.n_viewers)}")
        problems += _missing_values("benchmark_resolution.csv", res, ("sub_median_ms", "sub_p95_ms"))
    # make_final_figures.py reads copies of these files from the main directory
    for src in (V2 / "memory" / "benchmark_memory.csv", V2 / "memory" / "benchmark_memory_datastructure.csv",
                V2 / "resolution" / "benchmark_resolution.csv"):
        copy = V2 / MAIN / src.name
        if src.is_file() and (not copy.is_file() or copy.read_bytes() != src.read_bytes()):
            problems.append(f"{copy}: missing or different from {src}")
    return problems


out, md = {}, ["# Final benchmark results", ""]
md += ["## Validation (expected runs, configurations and protocol of every stage; one code state for all)", ""]
out["validation"] = {}
failed, codes = [], set()
freeze = _sha256(FREEZE) if FREEZE.is_file() else None
if freeze is None:
    failed.append(f"{FREEZE}: missing")
# The published freeze had two lines with private paths replaced after the run. freeze_redaction.json
# maps its digest to the digest of the file as it was during the run, which is what the sidecars record.
REDACTION = FREEZE.with_name("freeze_redaction.json")
if freeze is not None and REDACTION.is_file():
    redaction = json.loads(REDACTION.read_text())
    if freeze == redaction.get("original_sha256"):
        pass  # the unredacted file itself
    elif freeze == redaction.get("published_sha256") and redaction.get("original_sha256"):
        freeze = redaction["original_sha256"]
    else:
        failed.append(f"{REDACTION}: {FREEZE} matches neither original_sha256 nor published_sha256")
for stage, exp in STAGES.items():
    problems, k = validate(stage, codes, freeze)
    failed += problems
    out["validation"][stage] = {"k": k, "problems": problems}
    md.append(f"- {stage}: k = {k}; {exp['drags']} drags per configuration, pacing {exp['pacing']}; OK")
problems = validate_post(codes)
failed += problems
out["validation"]["memory_resolution"] = {"problems": problems}
md.append("- memory (2 tools, 9 N; store and index at 3 N) and image size (5 sizes): complete; OK")
if len(codes) != 1 or None in codes:
    failed.append(f"code states differ between runs: {sorted(map(str, codes))}")
if failed:
    print(f"Validation failed ({len(failed)} problems); RESULTS.md and results_numbers.json not written:",
          file=sys.stderr)
    print("\n".join("  " + p for p in failed), file=sys.stderr)
    sys.exit(1)
code = next(iter(codes))
out["validation"]["code_state"], out["validation"]["env_freeze_sha256"] = code, freeze
md.append(f"- every run: code state {code[:12]} at start and end, environment freeze {freeze[:12]} "
          "(drag runs), AC power, nice 0")
md.append("")
md.append("Median of the k per-run medians, ms [bootstrap 95 % interval; with k <= 5 runs it equals the range of the "
          "run medians]. p95/p99: median over runs. Speed-up: ratio of the two medians [bootstrap interval]. "
          "Interactive: median, p95, p99 <= 33 ms. k per tool is given in each section.")
md.append("")

S, sp = agg(MAIN)
a = S[S.panel == "a"]
ks = lambda D, tool: sorted(set(int(k) for k in D[D.tool == tool].k))
md += [f"## Main run, panel a (XY editable + XZ/YZ + 3D); k = {ks(a, TB)} (TurboBox), {ks(a, BB)} (napari-bbox)", "",
       "| N | TurboBox | p95 / p99 | 3D update | napari-bbox per-box | p95 / p99 | 3D update | speed-up [CI] |",
       "|---:|---|---|---|---|---|---|---|"]
out["main_a"] = []
for n in sorted(a.n_boxes.unique()):
    m = a[(a.tool == TB) & (a.n_boxes == n)].iloc[0]
    g = a[(a.tool == BB) & (a.n_boxes == n)].iloc[0]
    s = sp[sp.n_boxes == n]
    s = s.iloc[0] if len(s) else None
    stxt = f"{s.speedup:.1f}× [{s.sp_lo:.1f}–{s.sp_hi:.1f}]" if s is not None else "–"
    md.append(f"| {n} | {fmt(m)} | {m.p95_ms:.1f} / {m.p99_ms:.1f} | {m.commit_median_ms:.1f} | {fmt(g)} | "
              f"{g.p95_ms:.1f} / {g.p99_ms:.1f} | {g.commit_median_ms:.1f} | {stxt} |")
    out["main_a"].append(dict(n=int(n), tb=m[["median_ms", "ci_lo", "ci_hi", "p95_ms", "p99_ms", "commit_median_ms", "k"]].to_dict(),
                              bb=g[["median_ms", "ci_lo", "ci_hi", "p95_ms", "p99_ms", "commit_median_ms", "k"]].to_dict(),
                              speedup=None if s is None else s[["speedup", "sp_lo", "sp_hi"]].to_dict(),
                              tb_interactive=interactive(m), bb_interactive=interactive(g)))

b = S[S.panel == "b"].copy()
b["V"] = pd.to_numeric(b.n_viewers, errors="coerce")
md += ["", f"## Main run, panel b (V identical XY views, N = 100, no 3D); k = {ks(b, TB)} / {ks(b, BB)}", "",
       "| V | TurboBox | p95 / p99 | napari-bbox per-box | p95 / p99 | ratio of medians [CI] |", "|---:|---|---|---|---|---|"]
out["main_b"] = []
for v in sorted(b.V.dropna().unique()):
    m = b[(b.tool == TB) & (b.V == v)].iloc[0]
    g = b[(b.tool == BB) & (b.V == v)].iloc[0]
    r, lo, hi = ratio_ci(run_medians(MAIN, BB, "b", 100, str(int(v))), run_medians(MAIN, TB, "b", 100, str(int(v))))
    md.append(f"| {int(v)} | {fmt(m)} | {m.p95_ms:.1f} / {m.p99_ms:.1f} | {fmt(g)} | {g.p95_ms:.1f} / {g.p99_ms:.1f} | "
              f"{g.median_ms / m.median_ms:.1f}× [{lo:.1f}–{hi:.1f}] |")
    out["main_b"].append(dict(v=int(v), tb=float(m.median_ms), bb=float(g.median_ms), ratio=float(g.median_ms / m.median_ms),
                              ratio_boot=[r, lo, hi], tb_p99=float(m.p99_ms), bb_p99=float(g.p99_ms)))

limits = {}
for tool in (TB, BB):
    ok = [int(r.n_boxes) for _, r in a[a.tool == tool].sort_values("n_boxes").iterrows() if interactive(r)]
    limits[tool] = ok

large = None
L = None
if any((V2 / LARGE).glob("drag_session_*_rep*.csv")):
    L, _ = agg(LARGE)
    la = L[L.panel == "a"]
    md += ["", f"## Large-N run, panel a; k = {ks(la, TB)} / {ks(la, BB)}", "",
           "| N | TurboBox | p95 / p99 | 3D update | napari-bbox per-box | p95 / p99 | 3D update | ratio of medians [CI] |",
           "|---:|---|---|---|---|---|---|---|"]
    large = []
    for n in sorted(la.n_boxes.unique()):
        mm = la[(la.tool == TB) & (la.n_boxes == n)]
        gg = la[(la.tool == BB) & (la.n_boxes == n)]
        if mm.empty or gg.empty:
            continue
        m, g = mm.iloc[0], gg.iloc[0]
        r, lo, hi = ratio_ci(run_medians(LARGE, BB, "a", n, "3+3D"), run_medians(LARGE, TB, "a", n, "3+3D"))
        md.append(f"| {n} | {fmt(m)} | {m.p95_ms:.1f} / {m.p99_ms:.1f} | {m.commit_median_ms:.1f} | {fmt(g)} | "
                  f"{g.p95_ms:.1f} / {g.p99_ms:.1f} | {g.commit_median_ms:.1f} | {g.median_ms / m.median_ms:.1f}× [{lo:.1f}–{hi:.1f}] |")
        large.append(dict(n=int(n), tb=float(m.median_ms), tb_ci=[float(m.ci_lo), float(m.ci_hi)], tb_p95=float(m.p95_ms),
                          tb_p99=float(m.p99_ms), tb_commit=float(m.commit_median_ms), bb=float(g.median_ms),
                          bb_ci=[float(g.ci_lo), float(g.ci_hi)], bb_p95=float(g.p95_ms), bb_p99=float(g.p99_ms),
                          bb_commit=float(g.commit_median_ms), ratio=float(g.median_ms / m.median_ms),
                          ratio_boot=[r, lo, hi], k_tb=int(m.k), k_bb=int(g.k)))
        for tool, row in ((TB, m), (BB, g)):
            if interactive(row):
                limits[tool].append(int(n))
    out["large_n"] = large

if any((V2 / "tutorial_full").glob("drag_session_*_rep*.csv")):
    T, _ = agg("tutorial_full")
    ta = T[(T.panel == "a") & (T.tool == BB)]
    md += ["", f"## Full-reassignment baseline (napari's multiple-viewer example pattern); k = {ks(ta, BB)}", "",
           "| N | napari-bbox full | p95 / p99 | 3D update | TurboBox (main) | ratio of medians |", "|---:|---|---|---|---|---|"]
    out["tutorial_full"] = []
    for n in sorted(ta.n_boxes.unique()):
        g = ta[ta.n_boxes == n].iloc[0]
        m = a[(a.tool == TB) & (a.n_boxes == n)].iloc[0]
        md.append(f"| {n} | {fmt(g)} | {g.p95_ms:.1f} / {g.p99_ms:.1f} | {g.commit_median_ms:.1f} | {m.median_ms:.1f} | "
                  f"{g.median_ms / m.median_ms:.1f}× |")
        out["tutorial_full"].append(dict(n=int(n), bb_full=float(g.median_ms), ci=[float(g.ci_lo), float(g.ci_hi)],
                                         p99=float(g.p99_ms), tb_main=float(m.median_ms),
                                         ratio=float(g.median_ms / m.median_ms), k=int(g.k)))

if any((V2 / "sleep_pacing").glob("drag_session_*_rep*.csv")):
    # Pacing sensitivity: the same protocol with idle waiting (time.sleep) between moves instead of
    # busy-waiting; compared with the busy-waiting medians of the main (N <= 1,000) and large-N runs.
    P, _ = agg("sleep_pacing")
    spin = {}
    if any((V2 / "sleep_pacing_spin").glob("drag_session_*_rep*.csv")):
        # matched control: the same 10 drags with busy-waiting, run alternately with the sleep runs
        C, _ = agg("sleep_pacing_spin")
        spin_src = "matched busy-waiting control (sleep_pacing_spin, same drags)"
        sources = (("sleep_pacing_spin", C),)
    else:
        spin_src = "main and large-N runs (not matched: different drag counts and times)"
        sources = ((MAIN, S), (LARGE, L if large is not None else None))
    for d, D in sources:
        if D is None:
            continue
        for _, r in D.iterrows():
            spin[(r.tool, r.panel, int(r.n_boxes), str(r.n_viewers))] = r
    md += ["", f"Busy-waiting reference: {spin_src}. A configuration whose busy-waiting median exceeds 33.3 ms "
           "never waits between moves, so its sleep row is a drift control, not a pacing comparison."]
    md += ["", f"## Pacing sensitivity: idle waiting (sleep) vs busy-waiting (spin); k = {ks(P, TB)} / {ks(P, BB)} "
           f"(10 drags per run; busy-waiting reference: {spin_src})", "",
           "| panel | N | V | TurboBox sleep | TurboBox spin | napari-bbox sleep | napari-bbox spin | "
           "speed-up sleep [CI] | speed-up spin | TB p99 sleep / spin | BB p99 sleep / spin |",
           "|---|---:|---:|---|---|---|---|---|---|---|---|"]
    out["sleep_pacing"] = []
    for (panel, n, v), grp in P.groupby(["panel", "n_boxes", "n_viewers"]):
        mm, gg = grp[grp.tool == TB], grp[grp.tool == BB]
        if mm.empty or gg.empty:
            continue
        m, g = mm.iloc[0], gg.iloc[0]
        ms, gs = spin.get((TB, panel, int(n), str(v))), spin.get((BB, panel, int(n), str(v)))
        r, lo, hi = ratio_ci(run_medians("sleep_pacing", BB, panel, n, str(v)), run_medians("sleep_pacing", TB, panel, n, str(v)))
        sp_spin = f"{gs.median_ms / ms.median_ms:.1f}×" if ms is not None and gs is not None else "–"
        md.append(f"| {panel} | {int(n)} | {v} | {fmt(m)} | {'–' if ms is None else f'{ms.median_ms:.1f}'} | {fmt(g)} | "
                  f"{'–' if gs is None else f'{gs.median_ms:.1f}'} | {g.median_ms / m.median_ms:.1f}× [{lo:.1f}–{hi:.1f}] | "
                  f"{sp_spin} | {m.p99_ms:.1f} / {'–' if ms is None else f'{ms.p99_ms:.1f}'} | "
                  f"{g.p99_ms:.1f} / {'–' if gs is None else f'{gs.p99_ms:.1f}'} |")
        out["sleep_pacing"].append(dict(
            panel=panel, n=int(n), v=str(v), tb_sleep=float(m.median_ms), bb_sleep=float(g.median_ms),
            tb_spin=None if ms is None else float(ms.median_ms), bb_spin=None if gs is None else float(gs.median_ms),
            ratio_sleep=float(g.median_ms / m.median_ms), ratio_sleep_boot=[r, lo, hi],
            ratio_spin=None if ms is None or gs is None else float(gs.median_ms / ms.median_ms),
            tb_p99_sleep=float(m.p99_ms), bb_p99_sleep=float(g.p99_ms),
            tb_interactive_sleep=interactive(m), bb_interactive_sleep=interactive(g)))

md += ["", "## Interactive limits (median, p95, p99 <= 33 ms), panel a", ""]
for tool, name in ((TB, "TurboBox"), (BB, "napari-bbox per-box")):
    ns = sorted(limits[tool])
    md.append(f"- {name}: interactive at N = {ns}")
out["interactive_ns"] = {k: sorted(v) for k, v in limits.items()}

mem = V2 / "memory" / "benchmark_memory.csv"
if mem.exists():
    M = pd.read_csv(mem)
    md += ["", "## Memory (tracemalloc, one hidden 2D viewer, napari 0.7.0; MiB)", "",
           "| N | TurboBox | napari-bbox | ratio |", "|---:|---|---|---|"]
    out["memory"] = []
    for n in sorted(M.n_boxes.unique()):
        t = M[(M.plugin == "napari-turbobox") & (M.n_boxes == n)].memory_increase_mb
        l = M[(M.plugin == "napari-bbox") & (M.n_boxes == n)].memory_increase_mb
        if len(t) and len(l):
            md.append(f"| {n} | {t.iloc[0]:.2f} | {l.iloc[0]:.2f} | {l.iloc[0] / t.iloc[0]:.1f}× |")
            out["memory"].append(dict(n=int(n), tb=float(t.iloc[0]), bb=float(l.iloc[0])))
    ds = V2 / "memory" / "benchmark_memory_datastructure.csv"
    if ds.exists():
        D = pd.read_csv(ds)
        md.append("")
        md.append("Store + one spatial index (TurboBox), bytes: " + ", ".join(
            f"N={int(r.n_boxes)}: {int(r.store_plus_index)}" for _, r in D[D.tool == "modern"].iterrows()))

res = V2 / "resolution" / "benchmark_resolution.csv"
if res.exists():
    R = pd.read_csv(res)
    md += ["", "## Image size (TurboBox, N = 100, 4 XY/YZ/XZ views, no 3D)", "",
           "| image px/side | median ms | p95 ms |", "|---:|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {int(r.image_xy)} | {r.sub_median_ms:.1f} | {r.sub_p95_ms:.1f} |")
    out["resolution"] = R[["image_xy", "sub_median_ms", "sub_p95_ms"]].to_dict("records")

metas = sorted(V2.glob("*/drag_session_*.meta.json"))
walls = {}
for f in metas:
    j = json.loads(f.read_text())
    stage = f.parent.name
    s, e = pd.Timestamp(j["started"]), pd.Timestamp(j["finished"])
    lo, hi = walls.get(stage, (s, e))
    walls[stage] = (min(lo, s), max(hi, e))
md += ["", "## Wall time per stage (first start to last finish)", ""]
for stage, (s, e) in walls.items():
    md.append(f"- {stage}: {s:%H:%M}–{e:%H:%M} ({(e - s).total_seconds() / 3600:.2f} h)")
out["wall"] = {k: [str(v[0]), str(v[1])] for k, v in walls.items()}

(V2 / "RESULTS.md").write_text("\n".join(md) + "\n")
(V2 / "results_numbers.json").write_text(json.dumps(out, indent=1, default=float))
print("\n".join(md))
