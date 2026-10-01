"""Panel (c): memory of both tools on one napari version (the final benchmark environment).

memory_benchmark picks the plugin from the napari version, so the choice is forced per run;
the napari version passed to it is the one actually installed. Writes
<out>/benchmark_memory.csv, <out>/benchmark_memory_datastructure.csv and <out>/benchmark_memory.meta.json.
"""
import json
import sys
import time
from pathlib import Path

import napari

import paper_benchmarks.drag_session_benchmark as dsb
from paper_benchmarks import memory_benchmark as mb

out = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/memory")
out.mkdir(parents=True, exist_ok=True)
env_diffs = dsb._warn_if_environment_differs()
code_start = dsb._code_state()
t_start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
for plugin in ("napari-turbobox", "napari-bbox"):
    mb.get_environment_info = lambda p=plugin: {"napari_version": napari.__version__, "plugin_name": p}
    mb.run_benchmark(output_dir=str(out))
meta = {"script": "paper_benchmarks/final_run/memory_final.py", "started": t_start,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "napari": napari.__version__,
        "code_state": code_start, "code_state_end": dsb._code_state(),
        "env_differs_from_reference": {k: list(v) for k, v in env_diffs.items()},
        "process_state": dsb._process_state()}
(out / "benchmark_memory.meta.json").write_text(json.dumps(meta, indent=1, default=str))
