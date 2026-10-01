# Benchmark environment (final run, 25 Sep 2026)

`pip_freeze_benchmark_env.txt` is the `pip freeze` of the one environment in
which both tools were benchmarked (Python 3.12.13, macOS 27.0, Apple M2 Pro).

- napari 0.7.0, PyQt6 6.10.2, NumPy 2.4.3, vispy 0.16.1, bermuda 0.1.7.
- napari-bbox 0.1.1: the unmodified PyPI sdist, installed with
  `pip install --no-deps` from a local copy. That is why the freeze shows a
  `file://` URL instead of a version pin; `pip install napari-bbox==0.1.1`
  installs the same code.
- napari-TurboBox: imported from the `src/` of the measured code state
  (`PYTHONPATH=src`), not installed. That code state is archived in
  `paper_results/provenance/benchmarked_code_state_20260925.tgz`.

Every drag run in `paper_results/` has a `.meta.json` sidecar (its
`drag_session_dist_*` file belongs to the same run) with the versions, Qt
binding, triangulation backend, git commit, dirty flag and timestamps of that
run; the memory and image-size directories have one sidecar each. Four points
need a note:

- `napari_turbobox` (`0.2.0`) is read from the package metadata that an
  earlier editable install left next to the imported source
  (`src/napari_turbobox.egg-info`); TurboBox itself is not installed in the
  environment. 0.2.0 was the version string of the development repository.
  The code was first released publicly as version 0.1.0. The code that ran is
  the archived `src/` named in `plugin_file`. The released `src/` differs from
  it in four functions and in the control panel's window title. Only the error
  branch added to `_incremental_shape_update` lies on the timed path, and it
  runs only when the in-place edit raises (see the README).
- `thumbnails` applies to TurboBox only. napari-bbox's thumbnail behaviour
  follows from `legacy_sync`: with `per_box` and `full`, its layers update
  their thumbnails on every write (napari's default).
- `harness_commit` is `1b2ba33` with `harness_dirty: true`: the run used an
  uncommitted working tree on top of commit 1b2ba33 of the development
  repository. That repository and its branch `perf/incremental-sync`
  (`harness_branch`) are private. This public repository starts from a single
  commit. The measured code itself is archived in
  `paper_results/provenance/benchmarked_code_state_20260925.tgz`, and its diff
  against 1b2ba33 is `paper_results/provenance/code_state_vs_1b2ba33.patch`.
  The archive is the complete record of the measured code. The patch ties it to
  1b2ba33 for the authors. The README describes how the published code differs
  from it.
- Two lines of the freeze cannot be installed from this file with
  `pip install -r`: napari-bbox (the local `file://` copy described above) and
  an editable install of an unrelated, unpublished plugin that the benchmark
  does not use. `src/requirements/modern.txt` lists installable pins.

## Redactions after the run

Host names and absolute paths of the benchmark machine were removed before
the records were published. Nothing else changed.

- Sidecars: `host` reads `<redacted>`, and `plugin_file` is given relative to
  the root of the development checkout. For napari-bbox runs it points into its
  `records/` directory, which is not published.
- Freeze: line 82 (the local directory of the napari-bbox sdist) and line 86
  (an editable install from a private repository) were replaced by
  `<redacted>`. The sidecars and
  `results_numbers.json` keep the digest of the file as it was during the run
  (`5d3c862b…`). `freeze_redaction.json` gives that digest, the digest of the
  published file (`11a8b0e3…`) and the two line numbers.
  `paper_benchmarks/final_run/report_final.py` accepts the published file
  through this record.
- Code archive: in `records/TB-PERF-01/run_final_v2.sh` and `run_post.sh`, the
  absolute repository path in `R=` was replaced by `<redacted>`. All other
  files are byte-identical. The archive was packed again with Python's
  `tarfile` with the same member order, names, modes, owners and times. Its
  sha256 changed from `fd9e2469…` to `64706c5a…`.

The unredacted files are kept by the authors.
