"""Panel (f): TurboBox update time vs background image size, with the final run's protocol.

Same settings as run_final.sh (TB_THUMBS, PACING, EDIT_VIEW from the environment; xy_visible drags,
XZ/YZ views following the dragged box, one untimed warm-up drag). N = 100 boxes, four 2D
views (XY editable, YZ, XZ, XY), no 3D view, 12 drags per image size, sizes 128..2048 px/side.
Writes <out>/benchmark_resolution.csv (image_xy, n_boxes, n_viewers, sub_median_ms, sub_p95_ms)
and <out>/benchmark_resolution.meta.json.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import paper_benchmarks.drag_session_benchmark as dsb

dsb.DRAG_PROTOCOL = "xy_visible"
dsb.THUMBNAILS = os.environ["TB_THUMBS"]
dsb.PACING = os.environ["PACING"]
dsb.ORTHO_FOLLOW = True
dsb.WARMUP = 1
dsb.EDIT_VIEW = os.environ.get("EDIT_VIEW", "api")
out = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/resolution")
out.mkdir(parents=True, exist_ok=True)

from qtpy.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
env_diffs = dsb._warn_if_environment_differs()
code_start = dsb._code_state()
t_start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
rows = []
for xy in (128, 256, 512, 1024, 2048):
    subs, _ = dsb.run_condition("modern", 100, 4, True, False, 12, image_shape=(100, xy, xy))
    rows.append(dict(image_xy=xy, n_boxes=100, n_viewers=4, sub_median_ms=float(np.median(subs)),
                     sub_p95_ms=float(np.percentile(subs, 95))))
    print(f"  image={xy:>4}: median={rows[-1]['sub_median_ms']:6.2f} p95={rows[-1]['sub_p95_ms']:6.2f} ms", flush=True)
with open(out / "benchmark_resolution.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
code_end = dsb._code_state()
meta = dict(script="paper_benchmarks/final_run/resolution_final.py", started=t_start,
            finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), thumbnails=dsb.THUMBNAILS, pacing=dsb.PACING,
            ortho_follow=True, warmup=1, edit_view=dsb.EDIT_VIEW, drags_per_size=12, code_state=code_start, code_state_end=code_end,
            env_differs_from_reference={k: list(v) for k, v in env_diffs.items()},
            process_state=dsb._process_state())
(out / "benchmark_resolution.meta.json").write_text(json.dumps(meta, indent=1, default=str))
print(f"wrote {out / 'benchmark_resolution.csv'}")
