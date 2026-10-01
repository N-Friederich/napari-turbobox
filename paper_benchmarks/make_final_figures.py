"""Aggregate the k=5 drag-session run and build the manuscript figures.

Single, self-contained figure+stats step. Reads the tracked per-replicate CSVs
written by ``drag_session_benchmark.py`` (plus the memory and resolution CSVs),
aggregates them at the **replicate level** (independent unit = run; one aggregate
per run = its median per-sub-update latency), and emits both the manuscript
figures and the tidy summary/speedup tables.

Statistics (independent unit = replicate run; NO per-frame pseudoreplication):
median of the per-run medians with a percentile-bootstrap 95 % interval over the
run medians (with k = 5 runs this interval equals their range), p95/p99 as the
median of the per-run percentiles, median end-of-drag 3D update, and the
speed-up (napari-bbox / TurboBox) as the ratio of the two medians, with the
2.5-97.5 percentile interval of bootstrap ratios of resampled run medians.
No significance test.

Inputs (all under ``--results``, default ``paper_results/main``):
    drag_session_<tag>_rep<r>.csv         per-run summary (panels a + b)
    drag_session_dist_<tag>_rep<r>.csv    every sub-update time (panel e)
    benchmark_memory.csv                  memory added by the box layer (panel c)
    benchmark_memory_datastructure.csv    store+index data-structure bytes (panel c)
    benchmark_resolution.csv              update time vs image size (panel f)
Optional ``--large-n DIR``: per-run CSVs of a large-N run (panel a only); its
configurations are added to panels (a) and (d) with open markers, and its own
drag_session_summary.csv and drag_session_speedup.csv are (re)written in DIR.

Outputs (written under ``--out``, default ``benchmark_results/figures``):
    drag_session_summary.csv, drag_session_speedup.csv   derived tidy tables
    Figure3_performance.{png,pdf}   one 6-panel composite:
        (a) N-sweep, four-viewer layout  (b) view scaling  (c) memory
        (d) speed-up  (e) update-time distribution  (f) image size

Labels: CPU-side time per update (ms) in real, hidden napari viewers:
synchronization and per-view geometry update, no rendering (thumbnails are
redrawn once at the end of a drag). 33 ms (30 Hz) budget marked.
"""
import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THR = 33.0
# Consistent two-colour code across EVERY panel so the reader learns it once:
#   TurboBox  = KIT-Blue  RGB (70,100,170)
#   napari-bbox = Brown   RGB (167,130,46)
# Multiple curves within ONE tool are distinguished by line style/alpha, not new hues.
# Brown-vs-blue differs in both hue and lightness (colour-blind safe); line style is redundant.
C_TB, C_BB = "#4664AA", "#A7822E"
C_NEUTRAL = "#333333"  # for tool-agnostic quantities (e.g. the speed-up ratio)
YL = r"CPU-side update time ($\mathrm{ms}$)"

FIG_WIDTH = 7.08  # 180 mm (standard full 2-column width in Bioinformatics)
FIG_HEIGHT = 4.70  # Compact height for 2x3 subplots

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "legend.title_fontsize": 7.2,
    "legend.borderpad": 0.25,
    "legend.labelspacing": 0.2,
    "legend.handletextpad": 0.4,
    "legend.handlelength": 1.2,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "axes.linewidth": 0.75,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "pdf.fonttype": 42,  # TrueType fonts for vector export (required for OUP/LaTeX)
    "ps.fonttype": 42,
})

# Deterministic bootstrap (fixed seed -> reproducible CIs across re-runs).
_rng = np.random.default_rng(0)


# --------------------------------------------------------------------------
# Replicate-level aggregation (independent unit = run; k=5)
# --------------------------------------------------------------------------
def _boot_ci(v, fn=np.median, n=10000):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    bs = [fn(_rng.choice(v, len(v), True)) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _ratio_ci(leg, mod, n=10000):
    """Speed-up = ratio of the medians of the run medians, with a percentile-bootstrap interval."""
    leg, mod = np.asarray(leg, float), np.asarray(mod, float)
    bs = [np.median(_rng.choice(leg, len(leg), True)) / np.median(_rng.choice(mod, len(mod), True))
          for _ in range(n)]
    return float(np.median(leg) / np.median(mod)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def aggregate(results_dir: Path, out_dir: Path):
    """Aggregate per-replicate CSVs to the summary + speedup tables (and persist them)."""
    files = [f for f in sorted(glob.glob(str(results_dir / "drag_session_*_rep*.csv")))
             if "dist" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(
            f"No drag_session_<tool>_rep<r>.csv under {results_dir}. "
            "Run drag_session_benchmark.py for both tools, replicates 0..4, first."
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    agg, rows = {}, []
    for (tool, panel, n, v), g in df.groupby(["tool", "panel", "n_boxes", "n_viewers"], dropna=False):
        rm = g["sub_median_ms"].values  # one median per replicate run (k=5)
        lo, hi = _boot_ci(rm)
        rec = {"tool": tool, "panel": panel, "n_boxes": n, "n_viewers": v, "k": len(g),
                   "median_ms": float(np.median(rm)), "ci_lo": lo, "ci_hi": hi,
                   "p95_ms": float(np.median(g["sub_p95_ms"])), "p99_ms": float(np.median(g["sub_p99_ms"])),
                   "commit_median_ms": float(np.median(g["commit_median_ms"]))}
        rows.append(rec)
        agg[(tool, panel, n, str(v))] = (rec, rm)
    S = pd.DataFrame(rows)

    sp_rows = []
    for n in sorted(S[S.panel == "a"].n_boxes.unique()):
        km, kl = ("modern", "a", n, "3+3D"), ("legacy", "a", n, "3+3D")
        if km not in agg or kl not in agg:
            continue
        (_, mr), (_, lr) = agg[km], agg[kl]
        sp, slo, shi = _ratio_ci(lr, mr)
        sp_rows.append({"n_boxes": n, "speedup": sp, "sp_lo": slo, "sp_hi": shi})
    sp = pd.DataFrame(sp_rows, columns=["n_boxes", "speedup", "sp_lo", "sp_hi"])

    out_dir.mkdir(parents=True, exist_ok=True)
    S.to_csv(out_dir / "drag_session_summary.csv", index=False)
    sp.to_csv(out_dir / "drag_session_speedup.csv", index=False)
    return S, sp


# --------------------------------------------------------------------------
# Figure panels
# --------------------------------------------------------------------------
def _thr(ax, fs=6.8, ha="right", x=0.99, y=THR, va="bottom"):
    # Dotted real-time line; label sits in a clear zone with a semi-transparent
    # white backing box so it never muddies a curve where the two cross.
    ax.axhline(THR, color="crimson", lw=1.0, ls=":", zorder=1)
    pad = " "
    ax.text(x, y, f"{pad}" + r"$33\,\mathrm{ms}\ (30\,\mathrm{Hz})$" + f"{pad}", color="crimson", va=va, ha=ha,
            fontsize=fs, transform=ax.get_yaxis_transform(), zorder=4,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.5})


def _log_limits(values, pad=1.5):
    """Decade-aligned log-axis limits that keep every plotted value visible."""
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x) and x > 0], float)
    lo = 10 ** np.floor(np.log10(v.min() / pad))
    hi = 10 ** np.ceil(np.log10(max(v.max(), THR) * pad))
    return lo, hi


def _set_panel_label(ax, label):
    """Place (a)-(f) panel label in the upper left corner outside the axes."""
    ax.text(-0.15, 1.07, label, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", va="top", ha="right")


def panel_a(ax, S, L=None):
    # Top-to-bottom legend order matching curve values at rightmost edge:
    # 1. napari-bbox (editing)
    # 2. napari-bbox (end-of-drag update: 3D view, thumbnails)
    # 3. TurboBox (end-of-drag update)
    # 4. TurboBox (editing)
    _set_panel_label(ax, "(a)")
    a = S[S.panel == "a"].sort_values("n_boxes")
    m, g = a[a.tool == "modern"], a[a.tool == "legacy"]
    h_bb_edit, = ax.plot(g.n_boxes, g.median_ms, "-s", color=C_BB, ms=4, label="napari-bbox (editing)")
    h_bb_3d, = ax.plot(g.n_boxes, g.commit_median_ms, "--^", color=C_BB, ms=3.5, mfc="white", alpha=0.6,
                       label="napari-bbox (end of drag)")
    h_tb_3d, = ax.plot(m.n_boxes, m.commit_median_ms, "--D", color=C_TB, ms=3.5, mfc="white", alpha=0.6,
                       label="TurboBox (end of drag)")
    h_tb_edit, = ax.plot(m.n_boxes, m.median_ms, "-o", color=C_TB, ms=4, label="TurboBox (editing)")
    vals = [m.median_ms, g.median_ms, m.commit_median_ms, g.commit_median_ms]
    if L is not None and not L.empty:
        # Large-N run (fewer drags per run): open markers, joined to the main curves by dotted lines.
        la = L[L.panel == "a"].sort_values("n_boxes")
        for tool, df, col, mk in (("legacy", g, C_BB, "s"), ("modern", m, C_TB, "o")):
            x = la[la.tool == tool]
            if x.empty:
                continue
            xs = np.r_[df.n_boxes.values[-1:], x.n_boxes.values]
            ax.plot(xs, np.r_[df.median_ms.values[-1:], x.median_ms.values], ":", color=col, lw=1.0)
            ax.plot(x.n_boxes, x.median_ms, mk, color=col, ms=4, mfc="white")
            vals.append(x.median_ms)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*_log_limits(np.concatenate([np.asarray(v, float) for v in vals])))
    _thr(ax, ha="left", x=0.03)  # park at left, where both curves sit below 33 ms (clear)
    ax.set_xlabel(r"Number of boxes ($N$)")
    ax.set_ylabel(YL)
    ax.set_title(r"Editing ($3\times 2\mathrm{D} + 3\mathrm{D}$)", pad=4)
    ax.legend(handles=[h_bb_edit, h_bb_3d, h_tb_3d, h_tb_edit], loc="upper left", framealpha=0.85)
    ax.grid(True, which="both", alpha=0.3)


def panel_b(ax, S):
    # Top-to-bottom legend order matching curve values at rightmost edge:
    # 1. napari-bbox
    # 2. TurboBox
    _set_panel_label(ax, "(b)")
    b = S[S.panel == "b"].copy()
    b["n_viewers"] = pd.to_numeric(b["n_viewers"], errors="coerce")
    b = b.dropna(subset=["n_viewers"]).sort_values("n_viewers")
    m, g = b[b.tool == "modern"], b[b.tool == "legacy"]
    h_bb = ax.errorbar(g.n_viewers, g.median_ms, yerr=[g.median_ms - g.ci_lo, g.ci_hi - g.median_ms],
                       fmt="-s", color=C_BB, ms=4, capsize=2, label="napari-bbox")
    h_tb = ax.errorbar(m.n_viewers, m.median_ms, yerr=[m.median_ms - m.ci_lo, m.ci_hi - m.median_ms],
                       fmt="-o", color=C_TB, ms=4, capsize=2, label="TurboBox")
    for _, rm in m.iterrows():
        rg = g[g.n_viewers == rm.n_viewers]
        if not rg.empty:
            sp = rg.iloc[0].median_ms / rm.median_ms
            if rm.n_viewers == 1:
                off, ha, va = (3, 4), "left", "bottom"
            elif rm.n_viewers == 4:
                off, ha, va = (0, -10), "center", "top"
            elif rm.n_viewers == 8:
                off, ha, va = (-4, 5), "right", "bottom"
            else:
                off, ha, va = (0, 5), "center", "bottom"
            ax.annotate(f"{sp:.1f}" + r"$\times$", (rm.n_viewers, rm.median_ms), textcoords="offset points",
                        xytext=off, fontsize=6.8, color=C_NEUTRAL, ha=ha, va=va)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(*_log_limits(np.r_[m.ci_lo.values, g.ci_hi.values, m.median_ms.values, g.median_ms.values]))
    _thr(ax, ha="left", x=0.03)  # left side is clear at 33 ms
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel(r"Synchronized $2\mathrm{D}$ views ($V$), $N = 10^2$")
    ax.set_ylabel(YL)
    ax.set_title(r"View scaling ($V$, $N = 10^2$)", pad=4)
    ax.legend(handles=[h_bb, h_tb], loc="lower right", framealpha=0.85)  # keeps the 33 ms label (left) free
    ax.grid(True, which="both", alpha=0.3)


def panel_c(ax, mem, si):
    # Top-to-bottom legend order matching curve values at rightmost edge:
    # 1. napari-bbox (added allocations)
    # 2. TurboBox (added allocations)
    # 3. TurboBox (store+index)
    # The first two curves are the traced allocations added by loading the boxes into an
    # existing viewer and layer (memory_benchmark.py), not the total memory of the process.
    _set_panel_label(ax, "(c)")
    tb = mem[mem.plugin == "napari-turbobox"].sort_values("n_boxes")
    bb = mem[mem.plugin == "napari-bbox"].sort_values("n_boxes")
    h_bb, = ax.plot(bb.n_boxes, bb.memory_increase_mb, "-s", color=C_BB, ms=4,
                    label="napari-bbox (added allocations)")
    h_tb, = ax.plot(tb.n_boxes, tb.memory_increase_mb, "-o", color=C_TB, ms=4,
                    label="TurboBox (added allocations)")
    handles = [h_bb, h_tb]
    vals = [bb.memory_increase_mb.values, tb.memory_increase_mb.values]
    if si is not None:
        sim = si[si.tool == "modern"].sort_values("n_boxes")
        h_si, = ax.plot(sim.n_boxes, sim.store_plus_index / 1024 / 1024, "--D", color=C_TB, ms=3.5, mfc="white",
                        alpha=0.7, label="TurboBox (store+index)")
        handles.append(h_si)
        vals.append(sim.store_plus_index.values / 1024 / 1024)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = _log_limits(np.concatenate(vals))
    ax.set_ylim(lo, hi * 10)  # one extra decade on top, so the legend does not cover the curves
    ax.set_xlabel(r"Number of boxes ($N$)")
    ax.set_ylabel(r"Memory added ($\mathrm{MiB}$)")
    ax.set_title("Memory footprint", pad=4)
    ax.legend(handles=handles, loc="upper left", framealpha=0.85)
    ax.grid(True, which="both", alpha=0.3)


def panel_d(ax, sp, sp_large=None):
    _set_panel_label(ax, "(d)")
    if sp.empty:
        return
    sp = sp.sort_values("n_boxes")
    ax.errorbar(sp.n_boxes, sp.speedup, yerr=[sp.speedup - sp.sp_lo, sp.sp_hi - sp.speedup],
                fmt="-o", color=C_NEUTRAL, ms=4, capsize=2.5)
    if sp_large is not None and not sp_large.empty:
        sl = sp_large.sort_values("n_boxes")
        ax.plot(np.r_[sp.n_boxes.values[-1:], sl.n_boxes.values], np.r_[sp.speedup.values[-1:], sl.speedup.values],
                ":", color=C_NEUTRAL, lw=1.0)
        ax.errorbar(sl.n_boxes, sl.speedup, yerr=[sl.speedup - sl.sp_lo, sl.sp_hi - sl.speedup],
                    fmt="o", color=C_NEUTRAL, mfc="white", ms=4, capsize=2.5)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.text(0.98, 1, "equal speed ", color="gray", va="bottom", ha="right", fontsize=6.5,
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel(r"Number of boxes ($N$)")
    ax.set_ylabel(r"Speed-up ($\times$ faster)")
    ax.set_title(r"Speed-up factor", pad=4)
    ax.grid(True, which="both", alpha=0.3)


def panel_resolution(ax, res):
    """(f) Per-update response time vs background image size (resolution independence)."""
    _set_panel_label(ax, "(f)")
    res_path = res / "benchmark_resolution.csv"
    if not res_path.exists():
        return
    r = pd.read_csv(res_path).sort_values("image_xy")
    ax.plot(r.image_xy, r.sub_median_ms, "-o", color=C_TB, ms=4, label="TurboBox")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, max(42, r.sub_median_ms.max() * 1.3))
    _thr(ax, ha="right", x=0.98)  # linear axis: park label on right
    ax.set_xticks(r.image_xy.values)
    ax.set_xticklabels([str(int(x)) for x in r.image_xy.values])
    ax.set_xlabel(r"Image dimension ($\mathrm{px}/\mathrm{side}$)")
    ax.set_ylabel(YL)
    ax.set_title(r"Image size ($N = 10^2$)", pad=4)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="paper_results/main", help="dir with the input CSVs")
    ap.add_argument("--out", default="benchmark_results/figures", help="dir for figures + derived tables")
    ap.add_argument("--large-n", default=None,
                    help="dir with per-run CSVs of a large-N run, added to panels (a) and (d) with open markers")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    S, sp = aggregate(res, out)
    L = sp_large = None
    if a.large_n:
        L, sp_large = aggregate(Path(a.large_n), Path(a.large_n))

    # ONE 6-panel performance composite (no separate supplement):
    # (a) realistic latency  (b) view scaling  (c) memory
    # (d) speed-up           (e) distribution   (f) resolution
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(FIG_WIDTH, FIG_HEIGHT), layout="constrained")
    panel_a(ax[0, 0], S, L)
    panel_b(ax[0, 1], S)
    try:
        mem = pd.read_csv(res / "benchmark_memory.csv")
        ds_path = res / "benchmark_memory_datastructure.csv"
        si = pd.read_csv(ds_path) if ds_path.exists() else None
        panel_c(ax[0, 2], mem, si)
    except FileNotFoundError:
        ax[0, 2].text(0.5, 0.5, "memory data pending", ha="center")
    panel_d(ax[1, 0], sp, sp_large)
    # Panel (e): every measured update of every run. The harness already runs one
    # untimed warm-up drag per configuration and does not save it.
    dist_files = sorted(res.glob("drag_session_dist_*.csv"))
    if dist_files:
        d = pd.concat([pd.read_csv(f) for f in dist_files], ignore_index=True)
        _violin(ax[1, 1], d, "a_N100", r"Latency distribution ($N = 10^2$)")
    panel_resolution(ax[1, 2], res)

    for ext in ("png", "pdf"):
        out_file = out / f"Figure3_performance.{ext}"
        fig.savefig(out_file, dpi=300)
        print(f"wrote {out_file}")

    plt.close(fig)


def _violin(ax, d, cfg_key, title):
    """Spread of per-update response times for one config: TurboBox vs napari-bbox."""
    _set_panel_label(ax, "(e)")
    data, colors, ticks = [], [], []
    cfgs = [c for c in sorted(d.config.unique()) if cfg_key in c]
    cfg = cfgs[0] if cfgs else None
    for tool, col, name in (("modern", C_TB, "TurboBox"), ("legacy", C_BB, "napari-bbox")):
        vals = d[(d.config == cfg) & (d.tool == tool)].sub_ms.values if cfg else []
        if len(vals):
            data.append(vals)
            colors.append(col)
            ticks.append(name)
    if data:
        vp = ax.violinplot(data, showmedians=True, widths=0.7)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c)
            body.set_edgecolor(c)
            body.set_alpha(0.55)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in vp:
                vp[part].set_color(C_NEUTRAL)
                vp[part].set_linewidth(1.0)
        ax.set_xticks(range(1, len(ticks) + 1))
        ax.set_xticklabels(ticks, fontsize=7.2)
        ax.set_yscale("log")
        ax.set_ylim(*_log_limits(np.concatenate(data), pad=1.2))
        _thr(ax, ha="right", x=0.98)
    ax.set_ylabel(YL)
    ax.set_title(title, pad=4)
    ax.grid(True, which="both", axis="y", alpha=0.3)


if __name__ == "__main__":
    main()

