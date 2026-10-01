# Benchmark: update propagation

This page describes the benchmark of the napari-TurboBox Application Note in full: what is
measured, the baselines, the protocol, the results, the provenance of every run and how to
reproduce it. The README gives a short summary.


# What is measured

The benchmark measures how long one mouse move of a drag takes to reach all
live views: the time from the move until every live 2D view has updated its
shapes and Qt has processed the pending events (`QApplication.processEvents()`
returns). This is the CPU-side update time of a warmed process, not the
end-to-end display latency.

- Both tools are dragged with the mouse in the editable XY view: the press,
  every move and the release go through napari's own mouse dispatch
  (`mouse_press_callbacks`, `mouse_move_callbacks`,
  `mouse_release_callbacks`) into each plugin's own drag handler
  (`--edit-view mouse`). Only the hit test at the press is fixed, so that both
  tools drag the same box. The press is not timed; the release is timed
  separately (below). In both tools the dragged box stays selected and its
  highlight is redrawn on every move. The events are dispatched at napari's
  layer level on hidden canvases, so the canvas-level cursor and status-bar
  updates of a physical mouse are not included for either tool.
- The viewers are real napari viewers with hidden windows
  (`napari.Viewer(show=False)`, native Qt platform). No frame is rendered
  during the measurement, so paint, rasterization and buffer swaps are excluded
  for both tools.
- Layer thumbnails are not redrawn during a drag but once at its end, as both
  tools' own mouse drags do (TurboBox `--thumbnails off`, its default;
  napari-bbox's synchronized views with its public `block_thumbnail_update()`).
- The 3D view is updated once per drag, at the release, for both tools. The
  end-of-drag update at the release (3D view, thumbnails, for TurboBox also
  the undo entry) is reported separately (`commit_median_ms`) and is not part
  of the per-update times.
- The waiting time between moves (30 Hz schedule) is excluded and spent
  busy-waiting (`--pacing spin`): without rendering, an idle process would
  otherwise resume from processor idle states before every move, which a
  rendering napari session does not. A sensitivity run with idle waiting
  (`--pacing sleep`, k = 3 × 10 drags, panel a at N = 10 / 100 / 1,000 / 10,000
  and panel b), each replicate followed by the same drags with busy-waiting as
  a matched control, is written to `paper_results/sleep_pacing/` and
  `paper_results/sleep_pacing_spin/` and compared in
  [`paper_results/RESULTS.md`](https://github.com/N-Friederich/napari-turbobox/blob/v0.1.0/paper_results/RESULTS.md) (section "Pacing sensitivity"). With idle
  waiting, TurboBox's median update time is 8.0–8.5 ms in the four-viewer
  layout (5.3–9.2 ms for 1 to 8 views in panel b), and the speed-ups shrink to
  1.1–2.3× at N = 10 to 1,000 and 19× at N = 10,000 (matched busy-waiting
  control: 2.4–10.4× and 80×). At the sampled counts the interactive
  classification does not change: TurboBox is interactive at all four (99th
  percentile at most 13.3 ms), napari-bbox at N = 10, 100 and 1,000. N = 2,000
  and 5,000 were not run with idle waiting.

# Baselines

napari-bbox has no multi-viewer synchronization, so the harness
(`paper_benchmarks/sync_adapter.py`) keeps one napari-bbox layer per viewer.
The edited XY view is updated by napari-bbox's own drag handler; the other
views follow every move:

- **per-box, quiet** (main baseline, `--legacy-sync per_box_quiet`):
  napari-bbox 0.1.1's public per-box methods (`selected_data`,
  `remove_selected`, `add`) replace the edited box in every other 2D layer.
  napari's layer data events are held back during the drag
  (`layer.events.data.blocker()`) and emitted once per layer at the release, as
  napari-bbox's own drag does; otherwise every remove and add makes the viewer
  recompute the layer extents. The box is selected by adding it to the empty
  selection set, which skips the selection setter's style and display updates;
  its style is passed to `add()` explicitly. Together this roughly halves
  napari-bbox's update time compared with `--legacy-sync per_box_nothumb`. The
  napari-bbox layers are created with napari-bbox's own entry point
  (`viewer.add_bounding_boxes`). The 3D layer receives the final box at the
  release.
- **full reassignment** (`--legacy-sync full --edit-view api`): the whole box
  array is assigned to every live 2D layer, including the edited one, on every
  update (`layer.data = boxes`), the mechanism of napari's multiple-viewer
  example (which, driven by napari-bbox's data events, would update the other
  views only at the start and end of a drag); the 3D layer is reassigned at the
  release.

# Protocol

- Volume 60 × 256 × 256 voxels (blank image), boxes of 18 × 18 × 18 voxels at
  seeded random positions (seed 42), identical for both tools.
- Drag protocol `xy_visible`: each drag picks one box that crosses the editing
  XY slice (z = 30) and moves it in y and x only, alternately translating it by
  up to ±20 voxels and resizing it (moving its max corner) by up to ±15 voxels
  per axis, in 60 linearly interpolated sub-edits on a 30 Hz schedule. The drag sequence is seeded by
  the drag index and is identical for both tools. Each configuration starts
  with one untimed warm-up drag (`--warmup 1`), after which the boxes are
  restored; it is not saved.
- Panel a (four-viewer layout): editable XY view, read-only YZ and XZ views
  (all live) and one 3D view (updated at the end of each drag), with
  N = 10, 50, 100, 200, 500 and 1,000 boxes.
- Panel b (view scaling): V = 1, 2, 4 and 8 identical XY views at N = 100, no
  3D view.
- Main run: 50 drags × 60 sub-edits per configuration, k = 5 replicate
  processes per tool, the two tools interleaved per replicate.
- Large-N run: panel a with N = 2,000, 5,000 and 10,000, 50 drags per
  configuration, k = 5.
- Full-reassignment baseline: panel a with N <= 1,000, 10 drags, k = 3
  (napari-bbox only).
- Statistics: per run, the median, 95th and 99th percentile of all sub-edit
  times; per configuration, the median of the k run medians with a 95 %
  percentile-bootstrap interval over the run medians (10,000 resamples; for
  one tool's median with k = 3 or 5 runs, this interval equals the range of
  the run medians), and the median of the k per-run 95th and 99th
  percentiles. The speed-up is the ratio of the two medians, with a 95%
  percentile-bootstrap interval from 10,000 ratios of resampled run medians
  (`drag_session_speedup.csv`, N <= 1,000); this interval is not the range of
  the run medians. A configuration counts as interactive when its median and
  its 95th and 99th percentiles (each the median over the k runs) are all at
  most 33 ms, the 30 Hz CPU update budget marked in the figure.
- Environment (one environment for both tools): Python 3.12.13, napari 0.7.0,
  napari-bbox 0.1.1, PyQt6 6.10.2, NumPy 2.4.3, vispy 0.16.1, bermuda 0.1.7;
  Apple M2 Pro, macOS 27.0, native Cocoa Qt platform, display attached,
  viewer windows hidden. Every run also writes a `.meta.json` file with the
  package versions, Qt binding, triangulation backend, git commit and
  timestamps. Exact environment file:
  `paper_results/environment/pip_freeze_benchmark_env.txt`. A local path and a
  private repository in it were redacted after the run. The sha256 of the file as it was during
  the run is `5d3c862b…` (see `paper_results/environment/README.md`). Every
  drag-run sidecar records this digest and the code state
  at the start and end of the run; the memory and image-size sidecars record
  the start and end code states and the reference-environment check. Code
  state of the final run (25 Sep 2026, 13:00–19:42): `code_state.combined_sha256`
  `a8ad8214…` at the start and end of every run, `src/napari_turbobox/layer.py`
  sha256 `c14ade67…`. The measured `src/`, `paper_benchmarks/`, `tests/` and run
  scripts are archived in `paper_results/provenance/benchmarked_code_state_20260925.tgz`
  (sha256 `64706c5a…`, without bytecode caches and build metadata). The absolute
  repository path in two run scripts was redacted, before that the sha256 was
  `fd9e2469…`. The raw snapshot (sha256 `e2e2a41d…`) is kept by the authors.
  The measured code was never committed. It was the working tree on top of
  commit 1b2ba33 of the private development repository, and
  `paper_results/provenance/code_state_vs_1b2ba33.patch` is its diff against
  that commit. The code-state digest covers `src/` and `paper_benchmarks/`,
  and the code-state and `layer.py` hashes above belong to that archive. The plugin code in this
  repository differs from it in four functions (see `docs/changelog.rst`):
  - In `TurboBoxLayer._incremental_shape_update`, a new `except` branch marks
    the view for a rebuild when the in-place edit raises. This function runs
    on every timed TurboBox update, but the new branch runs only on an error.
  - `UndoStack.get_undo_description` now returns the label of the action that
    undo reverts.
  - `TurboBoxLayer.undo` reads that label before undoing and logs it;
    `redo` changed only in its log message.
- Other changes since the final run: comments and docstrings in `src/` were
  revised. `tests/test_undo.py` and `tests/test_incremental_enter_leave.py`
  gained tests. `test_undo.py` now expects the corrected undo label, and
  `test_incremental_enter_leave.py` also checks that the mesh vertices are
  finite (except in the zero-extent test, where napari's own triangulation
  produces NaN for flat boxes).
  `paper_benchmarks/make_final_figures.py` changed in the panel (c) legend
  and y-range and the y-axis label ("CPU-side update time", panels a, b, e
  and f) of Figure 3 and in its docstrings and comments, and panel (e)
  now plots all 50 measured drags per run (before, it left out the first one).
  Its `--update-paper-figures` option was removed, and its default input and
  output directories are now `paper_results/main` and
  `benchmark_results/figures`. In `drag_session_benchmark.py`, the module
  docstring, the path of the environment file in one comment and one warning
  message, one further comment and the default `--out` (now
  `benchmark_results`) changed, and the process-state record skips
  `os.nice` and `os.getloadavg` on Windows, where they do not exist; in `memory_benchmark.py`, the module docstring
  and two comments. `sync_adapter.py`
  is unchanged. Five scripts of an earlier benchmark design that the final run
  did not use (`run_benchmarks.sh`, `resolution_benchmark.py`,
  `generate_paper_screenshots.py`, `generate_gui_screenshots.py`,
  `interactive_celegans_multiview.py`) and `src/requirements/legacy.txt` were
  removed. The run scripts moved to `paper_benchmarks/final_run/`
  (`run_final_v2.sh` is now `run_final.sh`). `run_final.sh` and `run_post.sh`
  now find the repository from their own location, require the Python in `BB`,
  take the results directory from `OUT`, write their logs to `benchmark_logs/`,
  and `run_final.sh` records the code state against `HEAD` and the pip freeze
  of `BB` in `OUT`; `run_final.sh` takes its stop file from `benchmark_logs/STOP`
  and moves invalid runs to `OUT/_invalid/`. `with_lock.py` (not in the archive,
  like `sleep_check.py`) keeps its lock file in `benchmark_logs/`. `memory_final.py` and
  `resolution_final.py` changed only in their default output directory and the
  script path they record (`resolution_final.py` also in one docstring line).
  `report_final.py`, which is not in the archive, gained the checks against
  the design described under Reproducing, reads `paper_results/` and accepts
  the redacted environment file through
  `paper_results/environment/freeze_redaction.json`.
  Otherwise only docstrings, comments and descriptive strings changed (six test
  docstrings and two test comments, the fixture's provenance text, the
  `src/requirements/modern.txt` header, the control panel's window title, now
  "napari-TurboBox" instead of "nD Bounding Boxes"). The plugin manifest
  `src/napari_turbobox/napari.yaml` gained the categories Annotation and
  Visualization, and the tutorial helper `show_path` keeps local paths out of
  the printed output. The sidecars of the reported run
  still name the former locations of the run scripts and of the benchmark
  environment (under `records/` in the development checkout). Host names and
  absolute paths of the benchmark machine were redacted in the sidecars, in the
  environment file and in two archived run scripts before publication (see
  `paper_results/environment/README.md`). The logs of the reported run
  (driver, power and lock logs) are kept by the authors.

# Results

Median time per update in ms [95 % interval], panel a (four-viewer layout).

| N boxes | napari-TurboBox | napari-bbox 0.1.1, per-box quiet | Speed-up |
|---:|---|---|---|
| 10 | 1.6 [1.6–1.7] | 3.9 [3.9–3.9] | 2.4× [2.4–2.4] |
| 100 | 1.7 [1.7–1.7] | 5.3 [5.2–5.3] | 3.2× [3.1–3.2] |
| 1,000 | 1.7 [1.7–1.7] | 18.0 [17.8–18.2] | 10.5× [10.4–10.6] |
| 10,000 * | 2.0 [2.0–2.0] | 158.7 [155.6–159.1] | 79.1× [77.5–80.2] |

\* Large-N run (N = 2,000–10,000), same protocol: 5 runs × 50 drags.

Full-reassignment baseline (napari-bbox 0.1.1, k = 3, 10 drags, updates written without mouse events), compared with
TurboBox's median from the main run:

| N boxes | napari-bbox, full reassignment | Ratio to TurboBox |
|---:|---|---|
| 10 | 23.1 [23.1–23.2] | 14× |
| 100 | 112.2 [112.0–112.4] | 67× |
| 1,000 | 908.6 [907.8–909.3] | 531× |

Largest tested N at which the median, 95th and 99th percentile stay within 33 ms:
napari-TurboBox 10,000 (the largest N tested; 99th percentile 4.4 ms), napari-bbox (per-box)
1,000 (at 2,000: median 32.4 ms, 95th/99th percentile 33.5/35.1 ms). The 95th/99th percentiles,
the end-of-drag 3D update times and the view-scaling results (panel b) are in
`paper_results/main/drag_session_summary.csv`.

# Reproducing

Create one environment for both tools. The versions above were used for the
reported run; pin them to reproduce it exactly.

```bash
pip install "napari[all]==0.7.0" "napari-bbox==0.1.1" pandas matplotlib scipy
```

Run from the repository root (TurboBox is imported from `src/`, as in the
reported run; the scripts in `paper_benchmarks/final_run/` also import
`paper_benchmarks` from the root). These are the commands of the final run
script. They write to `$OUT` (here `benchmark_results/`, which git ignores),
so the reported files in `paper_results/` stay unchanged. Run them in bash: zsh, the default shell on macOS, does not split
unquoted variables such as `$H` into words, so the commands fail there. Start
`bash` first.

```bash
export PYTHONPATH="$PWD/src:$PWD"
OUT=benchmark_results
H="-m paper_benchmarks.drag_session_benchmark --drag-protocol xy_visible --ortho-follow --warmup 1 --pacing spin --edit-view mouse"
TB="--tool modern --thumbnails off"
LG="--tool legacy --legacy-sync per_box_quiet"

# 1) Large N: panel a, N = 2,000 / 5,000 / 10,000, k = 5 x 50 drags
for r in 0 1 2 3 4; do
  python $H $TB --panels a --ns 2000,5000,10000 --replicate $r --out $OUT/large_n
  python $H $LG --panels a --ns 2000,5000,10000 --replicate $r --out $OUT/large_n
done

# 2) Main run: panels a and b, k = 5 x 50 drags, tools interleaved per replicate
for r in 0 1 2 3 4; do
  python $H $TB --replicate $r --out $OUT/main
  python $H $LG --replicate $r --out $OUT/main
done

# 3) Pacing sensitivity: idle waiting between moves, k = 3 x 10 drags, each replicate followed by the
#    same drags with busy-waiting as a matched control
for r in 0 1 2; do
  python ${H/--pacing spin/--pacing sleep} $TB --ns 10,100,1000,10000 --drags 10 --replicate $r --out $OUT/sleep_pacing
  python ${H/--pacing spin/--pacing sleep} $LG --ns 10,100,1000,10000 --drags 10 --replicate $r --out $OUT/sleep_pacing
  python $H $TB --ns 10,100,1000,10000 --drags 10 --replicate $r --out $OUT/sleep_pacing_spin
  python $H $LG --ns 10,100,1000,10000 --drags 10 --replicate $r --out $OUT/sleep_pacing_spin
done

# 4) Full-reassignment baseline: panel a, N <= 1,000, k = 3 x 10 drags
for r in 0 1 2; do
  python -m paper_benchmarks.drag_session_benchmark --drag-protocol xy_visible --ortho-follow --warmup 1 \
      --pacing spin --tool legacy --legacy-sync full --edit-view api --panels a --drags 10 \
      --replicate $r --out $OUT/tutorial_full
done
```

`paper_benchmarks/final_run/run_final.sh` runs these blocks with
`TB_THUMBS=off BB_SYNC=per_box_quiet PACING=spin EDIT_VIEW=mouse` (stages
`large main sleep post tut`); it waits
for AC power, keeps the machine awake, serializes all processes with a lock
and checks the power log after every process. It runs on macOS only (`pmset`,
`caffeinate`), finds the repository from its own location, writes to `OUT`
(default: `benchmark_results/`; a relative path is taken from the repository
root) and its logs to `benchmark_logs/`, skips runs whose result file already
exists, and records the code state and the pip freeze of the environment in
`OUT`. `BB` must name the Python of the benchmark environment, for example
`BB=$(which python) TB_THUMBS=off BB_SYNC=per_box_quiet PACING=spin EDIT_VIEW=mouse paper_benchmarks/final_run/run_final.sh`.
On other systems, use the commands above.

Reported wall time on the machine above: large N 1.8 h, main run 2.9 h, pacing sensitivity with its
busy-waiting control 0.9 h, memory and image size 4 min, full-reassignment baseline 1.0 h (6.7 h in total).

Each run writes `drag_session_<tag>_rep<r>.csv` (one row per configuration:
per-run median, 95th and 99th percentile of the sub-edit times and the median
end-of-drag 3D update time), a `.meta.json` provenance file and, for panel a
at N = 100 and panel b at V = 4, `drag_session_dist_<tag>_rep<r>.csv` with every
sub-edit time. The tags are `modern-xy_visible-follow-warm1-spin-mouse`,
`legacy-per_box_quiet-xy_visible-follow-warm1-spin-mouse` and
`legacy-xy_visible-follow-warm1-spin`; the runs with idle waiting
(`sleep_pacing/`) have the same tags without `-spin`.

The memory panel (c) and the image-size panel (f) come from two shorter
measurements in the same environment, run after blocks 1 to 3: memory for
both tools (`paper_benchmarks/final_run/memory_final.py`, which runs
`paper_benchmarks/memory_benchmark.py` once per plugin) and image size for
TurboBox (`paper_benchmarks/final_run/resolution_final.py`, the drag session
of the harness at five image sizes with the same drag protocol and thumbnail
setting). `paper_benchmarks/final_run/run_post.sh` runs both:

```bash
python paper_benchmarks/final_run/memory_final.py $OUT/memory
TB_THUMBS=off PACING=spin EDIT_VIEW=mouse python paper_benchmarks/final_run/resolution_final.py $OUT/resolution
```

Aggregation and figure: the command below writes `drag_session_summary.csv`,
`drag_session_speedup.csv` and `Figure3_performance.{png,pdf}` into `--out`.
It reads `benchmark_memory*.csv` and `benchmark_resolution.csv` from the
results directory, so they are copied there first. `--large-n` adds the
large-N configurations to panels (a) and (d) as open markers and also
(re)writes the large-N run's own `drag_session_summary.csv` and
`drag_session_speedup.csv` in its directory. With
`paper_results` in place of `$OUT`, the same command rebuilds the reported
tables and figure from the reported per-run files (the memory and image-size
files are already in `paper_results/main/`).

```bash
cp $OUT/memory/benchmark_memory*.csv $OUT/resolution/benchmark_resolution.csv $OUT/main/
python -m paper_benchmarks.make_final_figures --results $OUT/main --out $OUT/main \
    --large-n $OUT/large_n
```

All numbers of the tables above (including the large-N and full-reassignment
ratios and the interactive limits) are collected by
`paper_benchmarks/final_run/report_final.py` into [`paper_results/RESULTS.md`](https://github.com/N-Friederich/napari-turbobox/blob/v0.1.0/paper_results/RESULTS.md) and
`paper_results/results_numbers.json`. It aggregates each directory with the
same `aggregate()` function and re-seeds the bootstrap before each directory,
as a fresh process would. Before that it checks every stage against the design
above (all runs, replicates and configurations, drag counts, pacing, one code
state for all runs, the environment digest, AC power) and stops without
writing anything if a value is missing or differs. `TB_RESULTS=$OUT` checks
and aggregates a re-run instead; its runs must also come from the reference
environment.

```bash
python paper_benchmarks/final_run/report_final.py
```

For a quick functional check, run a short smoke configuration, for example
`python -m paper_benchmarks.drag_session_benchmark --drag-protocol xy_visible --edit-view mouse --tool modern --panels a --ns 10,100 --drags 2 --replicate 0 --out smoke_out`.

Timings depend on hardware, operating system and display; re-running the
aggregation on the same per-run CSVs reproduces the tables exactly.
