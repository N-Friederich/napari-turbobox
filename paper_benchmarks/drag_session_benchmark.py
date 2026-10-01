"""Drag-session benchmark: update times of a box drag in synchronized napari viewers.

Source of the latency numbers in the paper (Figure 3a, b, d and e). One process
runs one tool (``--tool modern``: napari-TurboBox, ``--tool legacy``:
napari-bbox) for one replicate, on real napari viewers with hidden windows.
Both tools are driven through :mod:`paper_benchmarks.sync_adapter`, so their
views are updated at the same points: the 2D views on every update, the 3D view
once per drag, at the release.

  (a) four-viewer layout: editable XY view, read-only YZ and XZ views and one
      3D view; N boxes from ``--ns`` (default 10, 50, 100, 200, 500, 1000).
  (b) view scaling: V identical XY views at N = 100, V in {1, 2, 4, 8}.

Each drag moves or resizes one box in 60 sub-updates on a 30 Hz schedule; the
waiting time between sub-updates (``--pacing``) is not part of the measured
times. ``--drags`` (default 50) drags per configuration follow ``--warmup``
untimed drags. The README (section "Benchmark: update propagation") gives the
protocol of the reported run and all its commands, for example::

    python -m paper_benchmarks.drag_session_benchmark --drag-protocol xy_visible --ortho-follow \
        --warmup 1 --pacing spin --edit-view mouse --tool modern --thumbnails off \
        --replicate 0 --out benchmark_results/main

Outputs per invocation (``<tag>`` names the tool and the protocol options),
read by ``make_final_figures.py``:
    drag_session_<tag>_rep<r>.csv        one row per configuration: median, 95th and
                                         99th percentile of the sub-update times and
                                         the median end-of-drag time
    drag_session_dist_<tag>_rep<r>.csv   every sub-update time of panel a at N = 100
                                         and panel b at V = 4
    drag_session_<tag>_rep<r>.meta.json  versions, Qt binding, code state, environment
                                         and process state of the run
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time

import numpy as np

os.environ.pop("QT_QPA_PLATFORM", None)  # native backend; never force offscreen
from qtpy.QtWidgets import QApplication

# The environment in which the published numbers were measured (full pip freeze:
# paper_results/environment/pip_freeze_benchmark_env.txt). Runs in another
# environment work, but their timings are not comparable one to one.
REFERENCE_ENV = {
    "python": "3.12.13",
    "napari": "0.7.0",
    "napari-bbox": "0.1.1",
    "numpy": "2.4.3",
    "vispy": "0.16.1",
    "bermuda": "0.1.7",
    "PyQt6": "6.10.2",
    "platform": "macOS arm64 (Apple M2 Pro)",
}

IMAGE_SHAPE = (60, 256, 256)
N_SUB = 60
TARGET_HZ = 30
# orthogonal plane cycling: (dims.order, sliced-axis) -> XY, YZ, XZ
_ORIENT = [((0, 1, 2), 0), ((2, 0, 1), 2), ((1, 0, 2), 1)]
# configs whose full per-sub-update distribution is dumped (panel, N, vlabel), for Figure 3e
DUMP = {("a", 100, "3+3D"), ("b", 100, "4")}


def make_boxes(n, image_shape, seed=42):
    """Deterministic initial boxes (same seed -> identical layout for both tools)."""
    rng = np.random.default_rng(seed)
    size = np.array([18, 18, 18])
    bounds = np.array(image_shape) - size
    mins = np.stack([rng.integers(0, bounds[d], size=n) for d in range(3)], axis=1).astype(float)
    return np.stack([mins, mins + size], axis=1)


def _config_2d(v, image_shape, i, orthogonal):
    """Configure a 2D viewer: orthogonal -> cycle XY/YZ/XZ; else all identical XY."""
    v.add_image(np.zeros(image_shape, np.uint8), name="bg")
    order, sax = _ORIENT[i % 3] if orthogonal else _ORIENT[0]
    v.dims.ndisplay = 2
    v.dims.order = order
    v.dims.set_point(sax, image_shape[sax] // 2)


def setup_modern(n, V, image_shape, orthogonal, with_3d):
    """Build the modern napari-turbobox shared-store setup.

    Returns ``(adapter, viewers, layers, initial_boxes)``. The 3D viewer (if any)
    is auto-assigned ``sync_mode="on_commit"`` by ``create_synchronized_bbox_layers``
    (it reads ndisplay), i.e. deferred to drag-release.
    """
    import napari

    from napari_turbobox import create_synchronized_bbox_layers
    from paper_benchmarks.sync_adapter import TurboBoxAdapter, TurboBoxMouseAdapter

    viewers = [napari.Viewer(show=False) for _ in range(V)]
    for i, v in enumerate(viewers):
        _config_2d(v, image_shape, i, orthogonal)
    if with_3d:
        v3 = napari.Viewer(show=False)
        v3.add_image(np.zeros(image_shape, np.uint8), name="bg")
        v3.dims.ndisplay = 3
        viewers.append(v3)
    layers = create_synchronized_bbox_layers(
        main_viewer=viewers[0], sub_viewers=viewers[1:], image_shape=image_shape
    )
    layers[0].add_boxes(make_boxes(n, image_shape))
    if THUMBNAILS == "on":
        for layer in layers:
            layer.suppress_thumbnail_in_session = False
    adapter = TurboBoxMouseAdapter(layers[0]) if EDIT_VIEW == "mouse" else TurboBoxAdapter(layers[0])
    return adapter, viewers, layers, layers[0]._bbox_store.data.copy()


class _LegacyShim:
    """Adapt a napari-bbox ``BoundingBoxLayer`` to the ``(N, 2, 3)`` min/max convention.

    napari-bbox's ``.data`` getter returns per-box ``(8, 3)`` corner arrays, while
    :class:`SyncAdapter` and the modern store work in ``(N, 2, 3)`` min/max form.
    The getter converts corners -> min/max; the setter passes ``(N, 2, 3)`` straight
    through (which napari-bbox accepts). The per-box min/max is cheap and does not
    bias timing.
    """

    def __init__(self, layer):
        self._layer = layer

    @property
    def data(self):
        out = [[np.asarray(c, float).min(0), np.asarray(c, float).max(0)] for c in self._layer.data]
        return np.array(out, float) if out else np.empty((0, 2, 3))

    @data.setter
    def data(self, v):
        self._layer.data = np.asarray(v, float)


THUMBNAILS = "off"  # modern: "off" = skip thumbnails inside edit sessions (default), "on" = rasterize every update
TRI_BACKEND = "default"  # "default" (napari settings) or "pure_python" (forced)
DRAG_PROTOCOL = "random3d"  # "random3d" (records) or "xy_visible" (box visible in XY, y/x moves)
LEGACY_SYNC = "full"  # "full" (layer.data per viewer), "per_box" (public remove/add), "per_box_nothumb",
# "per_box_quiet" (per_box_nothumb with napari data events deferred to the release, as napari-bbox's own drag)
EDIT_VIEW = "api"  # "api": the edited view is updated like the others; "mouse": both tools drag the
# box with the mouse in the edited (first) view, through napari's mouse dispatch and their own handlers
ORTHO_FOLLOW = False  # panel a: before each drag (untimed), slice the XZ/YZ views through the dragged box
WARMUP = 0  # untimed warm-up drags per configuration (state restored afterwards)
PACING = "sleep"  # "sleep" (idle between sub-edits) or "spin" (busy-wait; same 30 Hz schedule)


def setup_legacy(n, V, image_shape, orthogonal, with_3d):
    """Build the legacy napari-bbox setup (no shared store: V independent layers).

    The 3D viewer (if any) is registered with ``ndisplay=3`` so
    :class:`NapariBboxAdapter` coalesces it to drag-release -- mirroring the
    modern ``on_commit`` deferral so the comparison stays fair.
    """
    import napari
    import napari_bbox  # noqa: F401  (registers Viewer.add_bounding_boxes)

    from paper_benchmarks.sync_adapter import (
        NapariBboxAdapter,
        NapariBboxMouseAdapter,
        NapariBboxPerBoxAdapter,
    )

    boxes = make_boxes(n, image_shape)
    viewers, shims, ndisplays = [], [], []
    for i in range(V):
        v = napari.Viewer(show=False)
        _config_2d(v, image_shape, i, orthogonal)
        lyr = v.add_bounding_boxes(ndim=3, edge_color="cyan")  # napari-bbox's own entry point
        s = _LegacyShim(lyr)
        s.data = boxes
        viewers.append(v)
        shims.append(s)
        ndisplays.append(2)
    if with_3d:
        v = napari.Viewer(show=False)
        v.add_image(np.zeros(image_shape, np.uint8), name="bg")
        v.dims.ndisplay = 3
        lyr = v.add_bounding_boxes(ndim=3, edge_color="cyan")  # napari-bbox's own entry point
        s = _LegacyShim(lyr)
        s.data = boxes
        viewers.append(v)
        shims.append(s)
        ndisplays.append(3)
    if EDIT_VIEW == "mouse":
        # the other live views follow per box, thumbnails deferred
        assert LEGACY_SYNC in ("per_box_nothumb", "per_box_quiet"), (
            "--edit-view mouse needs --legacy-sync per_box_nothumb or per_box_quiet")
        adapter = NapariBboxMouseAdapter([s._layer for s in shims], ndisplays=ndisplays,
                                         quiet=LEGACY_SYNC == "per_box_quiet")
    elif LEGACY_SYNC in ("per_box", "per_box_nothumb", "per_box_quiet"):
        adapter = NapariBboxPerBoxAdapter(
            [s._layer for s in shims],
            ndisplays=ndisplays,
            block_thumbnails=LEGACY_SYNC in ("per_box_nothumb", "per_box_quiet"),
            quiet=LEGACY_SYNC == "per_box_quiet",
        )
    else:
        adapter = NapariBboxAdapter(shims, ndisplays=ndisplays)
    return adapter, viewers, shims, shims[0].data.copy()


def measure_drag_session(adapter, drag_idx, n_boxes, current, image_shape,
                         n_sub=N_SUB, target_hz=TARGET_HZ, before_drag=None):
    """Run one drag (60 sub-updates @ 30 Hz, then commit) and return per-update + commit latency.

    ``current`` is mutated in place so the next drag starts where this one ended.
    The drag target is clamped once, before interpolation, so every interpolated
    sub-update of the (convex, axis-aligned) box stays in-bounds without per-step
    spring-back that would distort the trajectory. The 30 Hz pacing sleep is
    excluded from the measured per-update latency.
    """
    rng = np.random.default_rng(seed=drag_idx)
    box_idx = int(rng.integers(0, n_boxes))
    op = "translate" if drag_idx % 2 == 0 else "resize"
    moving_axes = np.ones(3)
    if DRAG_PROTOCOL == "xy_visible":
        # A user drags a box they see in the editing XY view (slice z = Z/2),
        # in the view's plane: pick among boxes crossing that slice, move y/x only.
        z = image_shape[0] // 2
        visible = np.flatnonzero((current[:, 0, 0] <= z) & (current[:, 1, 0] >= z))
        if len(visible):
            box_idx = int(visible[rng.integers(0, len(visible))])
        moving_axes = np.array([0.0, 1.0, 1.0])
    start = current[box_idx].copy()
    bounds = np.array(image_shape, float) - 1
    if op == "translate":
        delta = rng.integers(-20, 21, size=3).astype(float) * moving_axes
        size = start[1] - start[0]
        end_min = np.clip(start[0] + delta, 0.0, bounds - size)
        end_box = np.stack([end_min, end_min + size])
    else:
        delta = rng.integers(-15, 16, size=3).astype(float) * moving_axes
        end_min = np.clip(start[0], 0.0, bounds - 1.0)
        end_max = np.clip(start[1] + delta, end_min + 1.0, bounds)
        end_box = np.stack([end_min, end_max])

    if before_drag is not None:  # untimed: e.g. move the orthogonal views onto the box
        before_drag(box_idx, start)
    prepare = getattr(adapter, "prepare_drag", None)
    if prepare is not None:  # untimed mouse press, for an adapter that drives a drag handler
        prepare(box_idx, start, end_box)
        QApplication.processEvents()

    subs = []
    nominal = 1.0 / target_hz
    session_start = time.perf_counter()
    next_target = session_start
    adapter.begin_edit_session()
    for s in range(n_sub):
        t = s / (n_sub - 1) if n_sub > 1 else 0.0
        interp = start + t * (end_box - start)
        now = time.perf_counter()
        if now < next_target:  # 30 Hz real pacing (waiting time excluded from latency below)
            if PACING == "spin":
                while time.perf_counter() < next_target:
                    pass
            else:
                time.sleep(next_target - now)
        t0 = time.perf_counter()
        adapter.update_box(box_idx, interp)
        QApplication.processEvents()
        subs.append((time.perf_counter() - t0) * 1000)
        next_target = session_start + (s + 1) * nominal
    tc = time.perf_counter()
    adapter.end_edit_session()
    QApplication.processEvents()
    commit_ms = (time.perf_counter() - tc) * 1000
    current[box_idx] = end_box
    return subs, commit_ms


def run_condition(tool, n, V, orthogonal, with_3d, n_drags, image_shape=IMAGE_SHAPE):
    """Run ``n_drags`` drags for one (tool, N, V, layout) condition; return (all sub_ms, commit_ms list)."""
    setup = setup_modern if tool == "modern" else setup_legacy
    adapter, viewers, _layers, current = setup(n, V, image_shape, orthogonal, with_3d)
    if TRI_BACKEND == "pure_python":
        # Viewer construction re-applies the settings backend, so force it after
        # setup (module-level switch, never via settings, which auto-save to disk)
        # and rebuild existing TurboBox shapes under it.
        from napari.utils.triangulation_backend import TriangulationBackend, set_backend

        set_backend(TriangulationBackend.pure_python)
        if tool == "modern":
            for layer in _layers:
                layer._sync_shapes_from_bboxes()
    follow = None
    if ORTHO_FOLLOW and orthogonal:
        def follow(box_idx, box):
            centre = np.asarray(box, float).mean(axis=0)
            for i, v in enumerate(viewers[1:V], start=1):
                _order, sax = _ORIENT[i % 3]
                v.dims.set_point(sax, round(centre[sax]))
            for _ in range(3):
                QApplication.processEvents()
    for w in range(WARMUP):
        # Untimed warm-up drag on its own trajectory; the moved box is put back
        # afterwards, so the timed drags are identical with and without warm-up.
        before = current.copy()
        measure_drag_session(adapter, 10_000 + w, n, current, image_shape, before_drag=follow)
        changed = np.flatnonzero(np.any(current != before, axis=(1, 2)))
        adapter.begin_edit_session()
        for idx in changed:
            adapter.update_box(int(idx), before[idx])
        adapter.end_edit_session()
        current[changed] = before[changed]
        QApplication.processEvents()
    gc.collect()
    subs_all, commits = [], []
    for d in range(n_drags):
        s, c = measure_drag_session(adapter, d, n, current, image_shape, before_drag=follow)
        subs_all.extend(s)
        commits.append(c)
    for v in viewers:
        v.close()
    return subs_all, commits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", required=True, choices=["modern", "legacy"])
    ap.add_argument("--replicate", type=int, required=True)
    ap.add_argument("--out", default="benchmark_results")
    ap.add_argument("--drags", type=int, default=50)
    ap.add_argument("--legacy-sync", choices=["full", "per_box", "per_box_nothumb", "per_box_quiet"], default="full")
    ap.add_argument("--drag-protocol", choices=["random3d", "xy_visible"], default="random3d")
    ap.add_argument("--tri-backend", choices=["default", "pure_python"], default="default")
    ap.add_argument("--panels", default="ab", help="which panels to run: a, b or ab")
    ap.add_argument("--ns", default="10,50,100,200,500,1000", help="panel-a box counts")
    ap.add_argument("--thumbnails", choices=["off", "on"], default="off", help="modern only")
    ap.add_argument("--ortho-follow", action="store_true",
                    help="panel a: slice the XZ/YZ views through the dragged box before each drag")
    ap.add_argument("--warmup", type=int, default=0, help="untimed warm-up drags per configuration")
    ap.add_argument("--pacing", choices=["sleep", "spin"], default="sleep")
    ap.add_argument("--edit-view", choices=["api", "mouse"], default="api",
                    help="mouse: drag in the first view through napari's mouse dispatch (xy_visible only)")
    args = ap.parse_args()
    global LEGACY_SYNC, DRAG_PROTOCOL, TRI_BACKEND, THUMBNAILS, ORTHO_FOLLOW, WARMUP, PACING, EDIT_VIEW
    LEGACY_SYNC = args.legacy_sync
    DRAG_PROTOCOL = args.drag_protocol
    TRI_BACKEND = args.tri_backend
    THUMBNAILS = args.thumbnails
    ORTHO_FOLLOW = args.ortho_follow
    WARMUP = args.warmup
    PACING = args.pacing
    EDIT_VIEW = args.edit_view
    if EDIT_VIEW == "mouse" and DRAG_PROTOCOL != "xy_visible":
        ap.error("--edit-view mouse drags boxes visible in the XY view: use --drag-protocol xy_visible")
    try:
        _app = QApplication.instance() or QApplication([])
    except Exception as e:
        print(f"FATAL: no Qt/Viewer available: {e}")
        sys.exit(2)

    rows, dist = [], []
    t_start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env_diffs = _warn_if_environment_differs()
    state_start = _process_state()
    code_start = _code_state()  # before any measurement; re-checked in _write_provenance

    def record(panel, n, V, orthogonal, with_3d, vlabel):
        t0 = time.perf_counter()
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        subs, commits = run_condition(args.tool, n, V, orthogonal, with_3d, args.drags)
        rows.append({"tool": args.tool, "replicate": args.replicate, "panel": panel, "n_boxes": n,
                         "n_viewers": vlabel, "n_drags": len(commits), "n_sub": len(subs),
                         "sub_median_ms": float(np.median(subs)),
                         "sub_p95_ms": float(np.percentile(subs, 95)),
                         "sub_p99_ms": float(np.percentile(subs, 99)),
                         "commit_median_ms": float(np.median(commits)), "started": started})
        if (panel, n, vlabel) in DUMP:
            for x in subs:
                dist.append({"tool": args.tool, "replicate": args.replicate,
                                 "config": f"{panel}_N{n}_V{vlabel}", "sub_ms": x})
        print(f"[{args.tool} rep{args.replicate}] {panel} N={n:>4} V={vlabel}: "
              f"med={np.median(subs):7.2f} p95={np.percentile(subs, 95):7.2f} "
              f"p99={np.percentile(subs, 99):8.2f} commit={np.median(commits):8.2f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # (a) realistic: orthogonal 3x2D + 1x3D (deferred), N-sweep
    if "a" in args.panels:
        for n in (int(x) for x in args.ns.split(",")):
            record("a", n, 3, orthogonal=True, with_3d=True, vlabel="3+3D")
    # (b) V-scaling: identical XY viewers at N=100
    if "b" in args.panels:
        for V in (1, 2, 4, 8):
            record("b", 100, V, orthogonal=False, with_3d=False, vlabel=str(V))

    os.makedirs(args.out, exist_ok=True)
    tag = args.tool if args.legacy_sync == "full" or args.tool == "modern" else f"{args.tool}-{args.legacy_sync}"
    if args.drag_protocol != "random3d":
        tag = f"{tag}-{args.drag_protocol}"
    if args.tri_backend != "default":
        tag = f"{tag}-{args.tri_backend}"
    if args.tool == "modern" and args.thumbnails != "off":
        tag = f"{tag}-thumbs_{args.thumbnails}"
    if args.ortho_follow:
        tag = f"{tag}-follow"
    if args.warmup:
        tag = f"{tag}-warm{args.warmup}"
    if args.pacing != "sleep":
        tag = f"{tag}-{args.pacing}"
    if args.edit_view != "api":
        tag = f"{tag}-{args.edit_view}"
    p = os.path.join(args.out, f"drag_session_{tag}_rep{args.replicate}.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if dist:
        pd_path = os.path.join(args.out, f"drag_session_dist_{tag}_rep{args.replicate}.csv")
        with open(pd_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dist[0].keys()))
            w.writeheader()
            w.writerows(dist)
    _write_provenance(args, p, t_start, state_start, code_start, env_diffs)
    print(f"wrote {p}")


def _environment_differences():
    """Return ``{name: (reference, found)}`` for every entry that differs from REFERENCE_ENV."""
    import platform
    from importlib import metadata

    found = {"python": platform.python_version()}
    for dist in ("napari", "napari-bbox", "numpy", "vispy", "bermuda", "PyQt6"):
        try:
            found[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            found[dist] = "not installed"
    is_ref_platform = platform.system() == "Darwin" and platform.machine() == "arm64"
    found["platform"] = REFERENCE_ENV["platform"] if is_ref_platform else f"{platform.system()} {platform.machine()}"
    return {k: (v, found[k]) for k, v in REFERENCE_ENV.items() if found.get(k) != v}


def _warn_if_environment_differs():
    diffs = _environment_differences()
    if diffs:
        lines = [f"  {k}: benchmarked with {ref}, found {got}" for k, (ref, got) in diffs.items()]
        print("WARNING: this environment differs from the one the published numbers were measured in:\n"
              + "\n".join(lines)
              + "\n  The run is valid, but its timings are not directly comparable. The reference\n"
              "  environment is listed in paper_results/environment/pip_freeze_benchmark_env.txt.",
              file=sys.stderr, flush=True)
    return diffs


def _process_state():
    """Scheduling and power state of this process (for the provenance sidecar)."""
    import subprocess

    def _run(*cmd):
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except Exception:
            return None

    batt = _run("pmset", "-g", "batt")
    return {"nice": os.nice(0), "ps": _run("ps", "-o", "pri=,stat=", "-p", str(os.getpid())),
            "power": batt.splitlines()[0] if batt else None, "loadavg": list(os.getloadavg())}


# measured code: plugin source + harness. The explicit diff options keep user git config (color,
# external diff, textconv, renames, algorithm, prefixes) out of the digest; --binary makes binary
# changes part of it (not "Binary files differ").
_CODE_PATHS = ("src", "paper_benchmarks")
_DIFF_CMD = ("diff", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames", "--diff-algorithm=myers",
             "--src-prefix=a/", "--dst-prefix=b/", "--binary", "HEAD", "--", *_CODE_PATHS)


def _code_state(repo_root=None, timeout=10):
    """Content identity of the measured code, so a dirty run is bound to one exact tree.

    ``tracked_diff_sha256`` hashes the bytes of ``git <_DIFF_CMD>``, ``untracked`` maps each
    untracked, non-ignored file under ``_CODE_PATHS`` to its SHA256, and ``combined_sha256`` is
    sha256(utf-8 of tracked_diff_sha256 + "\\n<path> <sha256>" per untracked file, sorted by path).
    Read-only and guarded: every git call has a timeout, a failure leaves its fields None.
    """
    import hashlib
    import subprocess

    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")  # never take index.lock mid-run

    def _git(*a, cwd):
        try:
            return subprocess.run(["git", *a], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  timeout=timeout, check=True).stdout
        except Exception:
            return None

    state = {"base_commit": None, "dirty": None, "tracked_diff_sha256": None, "untracked": None,
                 "combined_sha256": None, "tracked_diff_cmd": "git " + " ".join(_DIFF_CMD)}
    try:
        top = _git("rev-parse", "--show-toplevel", cwd=repo_root or os.path.dirname(os.path.abspath(__file__)))
        if top is None:
            return state
        root = os.fsdecode(top.strip())
        head = _git("rev-parse", "HEAD", cwd=root)
        state["base_commit"] = head.decode().strip() if head else None
        diff = _git(*_DIFF_CMD, cwd=root)
        if diff is not None:
            state["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        others = _git("ls-files", "--others", "--exclude-standard", "-z", "--", *_CODE_PATHS, cwd=root)
        if others is not None:
            untracked = {}
            for p in sorted(os.fsdecode(x) for x in others.split(b"\0") if x):
                try:
                    with open(os.path.join(root, p), "rb") as f:
                        untracked[p] = hashlib.sha256(f.read()).hexdigest()
                except OSError:  # vanished or unreadable between listing and hashing
                    untracked[p] = None
            state["untracked"] = untracked
        if diff is not None and others is not None:
            state["dirty"] = bool(diff) or bool(state["untracked"])
            lines = [state["tracked_diff_sha256"], *(f"{p} {h}" for p, h in state["untracked"].items())]
            state["combined_sha256"] = hashlib.sha256(os.fsencode("\n".join(lines))).hexdigest()
    except Exception:  # provenance must never abort a run; unknown fields stay None
        pass
    return state


def _env_freeze_sha256(timeout=60):
    """SHA256 of the stdout bytes of ``<sys.executable> -m pip freeze``; None on any failure."""
    import hashlib
    import subprocess

    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout,
                             check=True, env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                                                  "PIP_DISABLE_PIP_VERSION_CHECK": "1"}).stdout
    except Exception:
        return None
    return hashlib.sha256(out).hexdigest()


def _write_provenance(args, csv_path, t_start, state_start=None, code_start=None, env_diffs=None):
    """Sidecar JSON so every record carries versions, host, commit, code/env digests and timestamps."""
    import json
    import platform
    import subprocess
    from importlib import metadata

    import napari
    from qtpy import API_NAME, QT_VERSION

    def _ver(dist):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None

    def _git(*a):
        try:
            return subprocess.check_output(
                ["git", *a], cwd=os.path.dirname(os.path.abspath(__file__)), text=True
            ).strip()
        except Exception:
            return None

    backend = backend_resolved = None
    try:
        from napari.layers.shapes._shapes_models import Polygon
        from napari.layers.shapes._shapes_models import shape as _shape_mod

        backend = repr(_shape_mod.TRIANGULATION_BACKEND)
        backend_resolved = Polygon._set_meshes.__name__  # e.g. _set_meshes_compiled_bermuda
    except Exception:
        pass
    plugin_file = None
    try:
        mod = __import__("napari_turbobox" if args.tool == "modern" else "napari_bbox")
        plugin_file = getattr(mod, "__file__", None)
    except Exception:
        pass
    # True only if the code identity is known at both ends and identical (unknown -> False)
    code_end = _code_state()
    code_unchanged = bool(code_start and code_start["combined_sha256"] is not None
                          and all(code_start[k] == code_end[k] for k in ("base_commit", "combined_sha256")))
    meta = {
        "csv": os.path.basename(csv_path), "tool": args.tool, "legacy_sync": args.legacy_sync, "edit_view": args.edit_view,
        "drag_protocol": args.drag_protocol, "tri_backend_requested": args.tri_backend, "thumbnails": args.thumbnails, "ns": args.ns,
        "ortho_follow": args.ortho_follow, "warmup": args.warmup, "pacing": args.pacing,
        "replicate": args.replicate, "drags": args.drags,
        "started": t_start, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "harness_commit": _git("rev-parse", "HEAD"), "harness_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "harness_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "code_state": code_start, "code_state_end": code_end, "code_state_unchanged": code_unchanged,
        "env_freeze_sha256": _env_freeze_sha256(),
        "python": sys.version.split()[0], "napari": napari.__version__, "numpy": np.__version__,
        "vispy": _ver("vispy"), "qt_api": API_NAME, "qt_version": QT_VERSION,
        "qt_platform": QApplication.instance().platformName(), "triangulation_backend": backend,
        "triangulation_mesher": backend_resolved, "pyqt_or_pyside": _ver("PyQt6") or _ver("PyQt5") or _ver("PySide6"),
        "napari_turbobox": _ver("napari-turbobox"), "napari_bbox": _ver("napari-bbox"), "plugin_file": plugin_file,
        "host": platform.node(), "machine": platform.machine(), "os": platform.platform(),
        "gc_enabled": gc.isenabled(), "gc_threshold": gc.get_threshold(),
        "process_state_start": state_start, "process_state_end": _process_state(),
        "reference_env": REFERENCE_ENV,
        "env_differs_from_reference": {k: list(v) for k, v in (env_diffs or {}).items()},
    }
    with open(csv_path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
