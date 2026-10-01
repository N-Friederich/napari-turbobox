<p align="center">
  <img src="https://raw.githubusercontent.com/N-Friederich/napari-turbobox/v0.1.0/docs/logo-256.png" width="128" alt="napari-TurboBox logo: a see-through 3D box with three speed lines">
</p>

# napari-TurboBox

napari-TurboBox is a napari plugin for annotating 2D and 3D images with
axis-aligned bounding boxes (AABBs). All boxes of an annotation live in one
shared store. One editable box layer and any number of read-only box layers in
other napari viewers (for example two orthogonal slice views and a 3D view)
display that store: every edit is written once and reaches all views, the 2D
views on every edit and the 3D views when the edit ends. Mouse edits keep each
box axis-aligned and, when the image shape is known, inside the image. The
boxes can be exported as per-slice COCO or YOLO annotations, for example to
train 2D detectors or to use as per-slice box prompts, or in a native
NumPy/JSON format.

![Four napari viewers of one C. elegans volume share one box store: XY is editable, XZ, YZ and 3D are read-only. A box that has slipped off its nucleus is dragged back onto it in XY; XZ follows every move, YZ shows the box when it arrives, and the 3D view follows at the release. Its corner is pulled out until the box fits the nucleus. The XY view steps through z and draws only the boxes on each slice. Cmd+Z undoes the fit in every view and Cmd+Shift+Z redoes it. The 3D view is turned.](https://raw.githubusercontent.com/N-Friederich/napari-turbobox/v0.1.0/docs/demo.gif)

A real napari session, recorded by [`docs/make_demo.py`](https://github.com/N-Friederich/napari-turbobox/blob/v0.1.0/docs/make_demo.py):
every edit is a Qt mouse or key event on the XY canvas, and the 3D view is
turned by a mouse drag on its canvas. Before it starts, the script sets the
cameras and slices and moves one box off its nucleus and shrinks it. The
pointer, the dashed line that marks the XY slice in XZ and YZ, and the cyan
outline of the edited box are drawn on top. Data: *C. elegans* nuclei, Long et
al. (2022), doi:[10.5281/zenodo.5942575](https://doi.org/10.5281/zenodo.5942575),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see [Application example](#application-example-c-elegans-nuclei)).

Distribution name `napari-turbobox`, import name `napari_turbobox`, licence Apache 2.0.
Architecture and API documentation: [`docs/`](https://github.com/N-Friederich/napari-turbobox/tree/v0.1.0/docs) (Sphinx sources).

## Features

- **One shared store.** `BBoxDataStore` holds all boxes as one `(N, 2, D)`
  `float32` array (`[min corner, max corner]` per box). The layers of a
  synchronized layout all read this store; the store notifies them when it
  changes.
- **Synchronized multi-view layouts.** `create_synchronized_bbox_layers` builds
  one editable layer in a main viewer and read-only layers in any number of
  other viewers. Layers in 2D viewers update on every edit, including every
  mouse-move step of a drag. Layers in viewers that show 3D when the layers are
  created update once, when the edit session (for example a mouse drag) ends.
- **Slice-aware drawing.** A 2D view draws only the boxes that intersect its
  current slice. Editing a single box updates, adds, or removes that box's
  shape in each view when the incremental path's checks pass; otherwise it
  rebuilds the view. A slice change or a bulk edit (add, delete, import, undo)
  rebuilds the view. A 3D view draws every box as a wireframe.
- **Constrained AABB editing with the mouse.** In the editable layer, dragging a
  box moves it. Shift-dragging, or grabbing a box within 20 % of its size from
  an edge, resizes it: on each displayed axis, the side nearer to the grab
  point follows the mouse. Only the displayed axes change, and every axis keeps
  an extent of at least one voxel. When the layer knows the image shape, a
  moved box stops at the image border and resized coordinates are clamped to
  `[0, shape - 1]`. When boxes overlap, the box with the smallest displayed
  area under the cursor is grabbed. napari's Transform mode and the
  non-rectangle drawing tools (ellipse, line, path, polygon, lasso) are
  switched back to Select mode.
- **napari's own tools work on single boxes.** napari's *Add Rectangle* adds
  one box. With a known image shape, a rectangle drawn on a slice is extruded
  along the hidden axis through the full image depth (`0` to `shape - 1`); a
  click without dragging
  adds nothing. napari's *Delete selected shapes* removes exactly the selected
  box(es). Read-only views ignore edits.
- **Undo/redo.** The box layer binds Ctrl+Z / Ctrl+Y and Cmd+Z / Cmd+Shift+Z
  (active while the box layer is selected). On macOS, Cmd+Y is napari's own
  shortcut for switching between 2D and 3D display, so redo there with
  Cmd+Shift+Z or Ctrl+Y. The editable layer keeps up to 50 snapshots of all
  boxes by default (adjustable from 1 to 200 in the control panel). Undo and redo reach
  all synchronized views.
- **Control panel** (napari widget *BBox Control Panel*): add a box layer, draw
  a box, add a box that fits the image or the current viewport, remove all
  boxes, list, select and delete boxes, export and import.
- **Export**: native `.npy` / `.json`, COCO JSON (one 2D annotation per slice),
  YOLO labels (one text file per slice). The same converters are available in
  Python (`napari_turbobox.export`). See [Export conventions](#export-conventions).
  Boxes store coordinates only: the control panel exports all boxes as one
  class, and the Python converters accept an optional class per box
  (`category_ids`).
- **Import** of native `.npy` / `.json` box files; imported boxes are validated
  and appended to the existing boxes. COCO and YOLO files cannot be imported.

The npe2 manifest contributes one widget (*BBox Control Panel*) and no file
readers or writers; images are opened with napari's own readers.

## Installation

Requirements: Python >= 3.10 and napari >= 0.6.6, < 0.10 with a Qt binding
(PyQt5, PyQt6 or PySide6). napari-turbobox itself depends only on napari,
NumPy and qtpy.

```bash
pip install "napari-turbobox[all]"   # napari-TurboBox + napari with a Qt binding
```

The `all` extra installs `napari[all]>=0.6.6,<0.10`; if napari is
already installed, `pip install napari-turbobox` is enough.

`napari[all]` installs a Qt binding (PyQt5 with napari 0.6.6, PyQt6 with
napari 0.7.0 and 0.9.1) and napari's optional compiled triangulation backend
(`bermuda`). The napari Shapes layers that napari-TurboBox builds on use that
backend when it is installed; napari-TurboBox does not require it. Without any
Qt binding, napari cannot open a viewer.

From a checkout of this repository (editable install with the test
dependencies):

```bash
pip install -e ".[all,testing]"
pytest
```

Other extras: `tutorials` (JupyterLab and the file readers the notebooks use),
`benchmark` (napari-bbox 0.1.1, pandas, matplotlib for `paper_benchmarks/`),
`dev` (tests, ruff, npe2, build).

The tests build real napari viewers and therefore need a display (on Linux CI
they run under Xvfb).

Tested on macOS with:

| napari | Python | Qt binding |
|---|---|---|
| 0.6.6 | 3.12 | PyQt6 |
| 0.7.0 | 3.10 | PyQt5, PyQt6 |
| 0.7.0 | 3.12 | PyQt6 |
| 0.9.1 | 3.12 | PyQt6 |

The full test suite passes in each of these environments (with napari 0.7.0
and Python 3.10 once per Qt binding). Tests that need the example data or the
optional packages napari-bbox or hypothesis are skipped when these are
missing, and four older widget tests are always skipped because they crash
in headless runs. On napari 0.6.6 one test is an expected
failure: napari's Shapes layer cannot switch a 2D layer back from 3D display
(see [Limitations](#limitations)). PySide6 was last tested with napari 0.7.0
at commit 1b2ba33 of the private development repository, before the final
benchmark.

## Quick start

### In the napari GUI

1. Open a 2D or 3D image in napari.
2. In napari's **Plugins** menu, choose **BBox Control Panel** (listed with
   the plugin's display name in parentheses).
3. Click **Add Bounding Box Layer**. The new layer takes its shape from the
   selected image layer (or the top-most image layer), and also its scale and
   translation when the image has as many dimensions as the viewer. Without an
   image layer, boxes are not clamped to image bounds.
4. Click **Draw New BBox (Interactive)**. This switches the layer to napari's
   *Add Rectangle* mode. Drag a rectangle on the current slice; the box spans
   the full depth of the hidden axis.
5. Switch the layer to napari's *Select* mode. Drag a box to move it;
   Shift-drag it, or grab it near an edge, to resize it.
6. Delete boxes with napari's *Delete selected shapes*, or select a box in the
   panel's list and click **Delete Selected**. **Remove All Bounding Boxes**
   clears the layer. Ctrl+Z / Ctrl+Y (or Cmd+Z / Cmd+Shift+Z on macOS) undo
   and redo.
7. **Export BBoxes...** writes native `.npy`/`.json`, a COCO JSON file or a
   directory of YOLO label files. **Import BBoxes...** appends boxes from a
   native `.npy`/`.json` file.

**Fit Current Image Layer** adds one box that covers the whole image.
**Fit Current Viewport** adds a box over the middle half of the displayed axes
that is one slice thick along the hidden axes.

The panel can also be opened from Python:
`viewer.window.add_plugin_dock_widget("napari-turbobox", "BBox Control Panel")`.

![The XY viewer window: napari's layer controls and layer list on the left, the canvas with the selected box in the middle, and the BBox Control Panel on the right with the target layer, the box count, the quick-create buttons, the box list, import and export, and the advanced settings.](https://raw.githubusercontent.com/N-Friederich/napari-turbobox/v0.1.0/docs/screenshot-panel.png)

The panel manages the box layers of one viewer. Synchronized multi-view layouts
are set up with the Python API below.

### Python API: four synchronized viewers

```python
import napari
import numpy as np

from napari_turbobox import create_synchronized_bbox_layers

volume = np.random.default_rng(0).random((60, 256, 256)).astype(np.float32)  # (z, y, x)

xy, yz, xz, v3d = (napari.Viewer(title=t) for t in ("XY (editable)", "YZ", "XZ", "3D"))
for viewer in (xy, yz, xz, v3d):
    viewer.add_image(volume, name="volume")
yz.dims.order = (2, 0, 1)  # displays (z, y); slices along x
xz.dims.order = (1, 0, 2)  # displays (z, x); slices along y
v3d.dims.ndisplay = 3      # set BEFORE creating the layers (see Limitations)

layers = create_synchronized_bbox_layers(
    main_viewer=xy,
    sub_viewers=[yz, xz, v3d],
    image_shape=volume.shape,
    edge_color="cyan",
)
# Boxes are [[z_min, y_min, x_min], [z_max, y_max, x_max]] in voxel indices.
layers[0].add_boxes(np.array([[[10, 40, 40], [30, 90, 110]]], dtype=float))

xy.window.add_plugin_dock_widget("napari-turbobox", "BBox Control Panel")
napari.run()
```

Signature: `create_synchronized_bbox_layers(main_viewer, sub_viewers,
bbox_data=None, image_shape=None, sync_mode="live", **layer_kwargs)`. It returns
one layer per viewer: `layers[0]` is the editable layer in `main_viewer`, the
others are read-only. `layer_kwargs` (for example `edge_color`, `edge_width`,
`scale`) are passed to every `TurboBoxLayer`. Each layer's update mode
is chosen from its viewer's display mode at creation (2D viewer: every edit;
3D viewer: end of each edit session). `sync_mode="snapshot"` instead creates
independent, unsynchronized layers.

Before validation, `add_boxes` and the `bounding_boxes` setter expand a flat
hidden axis. With `image_shape` set, that axis spans the full image depth (`0`
to `shape - 1`). Validation then checks image bounds and the minimum extent:
with `image_shape` set, every coordinate must lie in `[0, shape - 1]`, and on
every axis `max - min >= 1`; invalid boxes raise `ValueError`. Boxes passed as
`bbox_data`, and writes made directly to the store (`layer.store`), are not
validated, so prefer `add_boxes`. `layer.bounding_boxes` returns a copy of all
boxes.

Export from Python:

```python
from napari_turbobox.export import boxes_to_coco, boxes_to_yolo

boxes = layers[0].bounding_boxes                   # (N, 2, 3), float32
coco = boxes_to_coco(boxes, volume.shape)          # dict: images, annotations, categories
yolo = boxes_to_yolo(boxes, volume.shape)          # {"slice_0010.txt": "0 0.292969 0.253906 0.273438 0.195312\n", ...}
```

## Export conventions

Box coordinates are in napari axis order, `(z, y, x)` for 3D and `(y, x)` for
2D, in the image's data (voxel index) coordinates. COCO and YOLO are 2D
formats, so 3D boxes are projected (`napari_turbobox/export.py`):

- **Per slice** (default; the only mode offered by the control panel): a 3D
  box yields one 2D annotation on every integer slice from `ceil(z_min)` to
  `floor(z_max)` inclusive, all with the same x/y box. A box thinner than one
  slice (`ceil(z_min) > floor(z_max)`) yields one annotation on slice
  `round((z_min + z_max) / 2)`. With a known image shape, slice indices are
  clamped to `[0, Z - 1]`.
- **Max projection** (Python API only, `mode="max_projection"`): each box yields
  one annotation of its x/y footprint on a single image.
- **Axes and size**: x is the last axis, y the second-to-last. The width is
  `x_max - x_min` and the height `y_max - y_min` (the drawn extent). For a box
  whose min and max are the first and last voxel index of an object, this is
  one pixel less than the object's pixel count. Image width and height are the
  last two axes of the image shape; if no image shape is available, they are
  derived from the box extents and a warning is issued. Coordinates outside
  the image are clamped (with a warning).
- **COCO JSON**: `bbox = [x_min, y_min, width, height]` in pixels,
  `area = width * height`, `iscrowd = 0`, annotation ids from 1. There is one
  `images` entry per slice that has at least one annotation: `id` = slice index,
  `file_name = "slice_NNNN.png"` (four-digit zero-padded slice index), `width`,
  `height`. The exporter writes no image files; you supply the slice images
  under these names. The default category is `{"id": 1, "name": "object"}`; with
  `category_ids` (0-based) the COCO `category_id` is `class_id + 1`.
- **YOLO**: one `slice_NNNN.txt` per slice that has at least one box, one line
  `class x_center y_center width height` per box, normalized by the image width
  and height, six decimals. The class defaults to `0`; `category_ids` are used
  as given.
- **2D boxes** go to a single image: `image.png` (COCO) / `image.txt` (YOLO).
- **Native**: `.npy` stores the `(N, 2, D)` `float32` array, `.json` the same
  array as nested lists. Import reads both formats, requires
  `D = layer.ndim`, and appends the boxes to the layer.

## Tutorials

The Jupyter notebooks in [`tutorials/`](https://github.com/N-Friederich/napari-turbobox/tree/v0.1.0/tutorials) run as they are: each
application notebook downloads one public volume (verified against its
published checksum), derives one box per object, loads the boxes into the
synchronized four-viewer layout, replays a mouse drag, and exports COCO and
YOLO files.

```bash
pip install -e ".[all,tutorials]"
jupyter lab tutorials/
```

| Notebook | Modality, data | Boxes |
|---|---|---:|
| [`00_napari_turbobox_api`](https://github.com/N-Friederich/napari-turbobox/blob/v0.1.0/tutorials/00_napari_turbobox_api.ipynb) | the Python API on a synthetic volume | – |
| [`01_confocal_celegans_nuclei`](https://github.com/N-Friederich/napari-turbobox/blob/v0.1.0/tutorials/01_confocal_celegans_nuclei.ipynb) | confocal, *C. elegans* nuclei (Zenodo 5942575) | 555 |
| [`02_electron_microscopy_nucmm_z`](https://github.com/N-Friederich/napari-turbobox/blob/v0.1.0/tutorials/02_electron_microscopy_nucmm_z.ipynb) | serial-section EM, zebrafish nuclei (NucMM-Z) | 606 |
| [`03_micro_ct_nucmm_m`](https://github.com/N-Friederich/napari-turbobox/blob/v0.1.0/tutorials/03_micro_ct_nucmm_m.ipynb) | X-ray micro-CT, mouse cortex nuclei (NucMM-M) | 282 |
| [`04_light_sheet_cellseg3d_mouse_brain`](https://github.com/N-Friederich/napari-turbobox/blob/v0.1.0/tutorials/04_light_sheet_cellseg3d_mouse_brain.ipynb) | light-sheet, cleared mouse brain nuclei (CellSeg3D) | 631 |
| [`05_instant_sim_polii_clusters`](https://github.com/N-Friederich/napari-turbobox/blob/v0.1.0/tutorials/05_instant_sim_polii_clusters.ipynb) | instant-SIM, nuclei and RNA Pol II clusters (Hajiabadi et al. 2022), boxes from a threshold segmentation | 38 |

Each notebook's counts and a screenshot of the four views are in
[`tutorials/results/`](https://github.com/N-Friederich/napari-turbobox/tree/v0.1.0/tutorials/results).

## Application example: C. elegans nuclei

`examples/celegans_workflow.py` applies napari-TurboBox to a public
fluorescence microscopy volume of *C. elegans* nuclei:

- **Dataset**: "3D nuclei instance segmentation dataset of fluorescence
  microscopy volumes of C. elegans" by Long, Peng, Liu, Kim, Myers, Kainmueller
  and Weigert, Zenodo, doi:[10.5281/zenodo.5942575](https://doi.org/10.5281/zenodo.5942575),
  licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Archive `c_elegans_nuclei.zip`, 84,403,531 bytes, MD5
  `7d616d91c4a6cce75a7d4dd37d7e666a`. The data are not part of this
  repository. If you use the data, the record asks you to cite Long et al.
  (2009), *Nat. Methods* 6:667–672 (raw data); the curated masks are from
  Hirsch and Kainmueller (2020), MIDL, and the train/val/test split follows
  Weigert et al. (2020), WACV. The README demo shows a cropped stretch of the
  first training volume with drawn annotations (pointer, slice line, outline).
- **Volume**: the first training volume,
  `c_elegans_nuclei/train/images/C18G1_2L1_1.tif` (140 × 140 × 1244 voxels,
  `uint8`), with its instance masks
  `c_elegans_nuclei/train/masks/C18G1_2L1_1.tif` (555 labelled nuclei).
- **Workflow**: one 3D box per nucleus from the ground-truth masks (minimum and
  maximum voxel index of each label on every axis), loaded into the
  four-viewer layout (editable XY view; read-only YZ, XZ and 3D views) and
  exported as per-slice COCO and YOLO annotations.

```bash
python examples/celegans_workflow.py          # download (once), derive boxes, export
python examples/celegans_workflow.py --gui    # open the four-viewer layout instead
```

The script downloads the archive once into `--data`, else `$TURBOBOX_DATA`,
else `~/.cache/napari-turbobox`, checks its size and MD5 and extracts only this
volume and its masks; exports go to `--out`. It uses `tifffile` and `scipy`,
which napari installs. `--gui` opens the control panel through the plugin
manifest, so napari-TurboBox must be installed (`pip install -e .`).
`tests/test_real_data_workflow.py` (marked `slow`) runs the script and compares
the boxes, the counts and hashes of the COCO and YOLO files with
`tests/fixtures/celegans_volume001_expected.json`; it is skipped when the data
are not available locally, unless `TURBOBOX_ALLOW_DOWNLOAD=1` is set.

With boxes derived this way, all 555 boxes pass the plugin's validation
unchanged, and the per-slice export contains 7,262 COCO annotations on 64
slices (64 COCO images, 64 YOLO files).

The four views of this volume with all 555 boxes, taken at the end of the demo
recording (the cyan outline marks the box that was dragged onto its nucleus
and fitted to it there):

![XY, XZ and YZ views of a stretch of the C. elegans larva with the boxes drawn on each view's slice, and the 3D view with every box as a wireframe.](https://raw.githubusercontent.com/N-Friederich/napari-turbobox/v0.1.0/docs/screenshot-views.png)

## Generic workflow: from instance labels to boxes

`examples/labels_to_boxes.py` shows the same steps for your own data: an
instance label image (background 0, one integer per object) is turned into one
axis-aligned box per label, the boxes are loaded into a box layer for review
and correction, and the result is exported with the conventions above.

## Benchmark: update propagation

### What is measured

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

### Baselines

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

### Protocol

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

### Results

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

### Reproducing

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

## Limitations

- Boxes have 2 or 3 spatial axes (`D = 2` or `3`). Time and channel axes are
  not supported.
- The depth of a box (its extent along the axis hidden in the editing view)
  cannot be dragged in the read-only orthogonal views, and new rectangles span
  the full depth. To set it with the mouse, reorient the editable viewer so that
  it displays the depth axis (for example `viewer.dims.order = (1, 0, 2)` for an
  XZ view) and drag or Shift-drag the box there; or set it through the API
  (`layer.bounding_boxes = ...`) or by importing a box file.
- Clamping to the image needs the image shape (`image_shape`, or an image layer
  when the layer is added from the control panel). Without it, only the
  minimum extent of one voxel per axis is enforced, and a rectangle drawn with
  Add Rectangle is not extruded through the full depth.
- A layer's update mode (every edit vs. end of edit session) is decided from
  its viewer's display mode when the layer is created. If a viewer is switched
  between 2D and 3D later, set `layer.sync_mode` (`"live"` or `"on_commit"`)
  yourself.
- The benchmark excludes rendering because the viewers are hidden. Mouse
  events are dispatched at the layer level into each tool's own drag handler,
  including selection-highlight updates; canvas-level cursor and status-bar
  updates are excluded. End-of-drag times include the 3D view update,
  thumbnails, and, for TurboBox, the undo snapshot.
- Boxes carry no class label. Per-box classes can be passed to the Python
  converters (`category_ids`) but are not stored with the boxes, and COCO or
  YOLO files cannot be imported.
- A 2D box layer (`D = 2`) keeps showing its rectangles when the viewer is
  switched to 3D display on napari >= 0.7. On napari 0.6.x, napari's Shapes
  layer cannot be switched back from 3D display in that case (plain Shapes has
  the same limitation).
- Each undo snapshot is a full copy of all boxes.
- The data store is not thread-safe; modify boxes from the main (Qt) thread.

## Repository layout

- `src/napari_turbobox/`: the plugin (store, layer, multi-view factory, control
  panel, export, spatial index, undo)
- `tests/`: test suite (pytest, pytest-qt)
- `examples/`: `celegans_workflow.py` and `labels_to_boxes.py` (see above) and
  `multi_viewer_demo.py` (three synchronized viewers on a synthetic volume)
- `docs/`: the logo (`logo.svg`; `logo-small.svg` for 32 px and below,
  `logo-mono.svg` in one colour, `logo-256.png`), the README demo (`demo.gif`,
  `screenshot-*.png`) and its recorder `make_demo.py`, and the Sphinx sources:
  architecture (store, update path, spatial index, export), API reference,
  contributing guide, changelog
- `paper_benchmarks/`: benchmark harness, synchronization adapters and
  aggregation/figure script; `paper_benchmarks/final_run/`: the scripts that
  ran the final benchmark and collect its numbers
- `paper_results/`: per-run records, summary tables, Figure 3 and the
  environment and code-state files of the benchmark described above
- `tutorials/`: the tutorial notebooks and their recorded results

## Citation

If you use napari-TurboBox, please cite it using the metadata in
`CITATION.cff` (GitHub's "Cite this repository" button reads this file).
An Application Note on napari-TurboBox is in preparation.

## Licence

Apache License 2.0. See `LICENSE` and `NOTICE`.
