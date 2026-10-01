<p align="center">
  <img src="https://raw.githubusercontent.com/N-Friederich/napari-turbobox/v0.1.0/docs/logo-256.png" width="128" alt="napari-TurboBox logo: a see-through 3D box with three speed lines">
</p>

# napari-TurboBox

[![tests](https://github.com/N-Friederich/napari-turbobox/actions/workflows/test.yml/badge.svg)](https://github.com/N-Friederich/napari-turbobox/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/napari-turbobox.svg)](https://pypi.org/project/napari-turbobox/)
[![Python versions](https://img.shields.io/pypi/pyversions/napari-turbobox.svg)](https://pypi.org/project/napari-turbobox/)
[![License](https://img.shields.io/github/license/N-Friederich/napari-turbobox)](https://github.com/N-Friederich/napari-turbobox/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/N-Friederich/napari-turbobox/graph/badge.svg)](https://codecov.io/gh/N-Friederich/napari-turbobox)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-turbobox)](https://napari-hub.org/plugins/napari-turbobox)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

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
(see [Good to know](#good-to-know)). PySide6 was last tested with napari 0.7.0
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
v3d.dims.ndisplay = 3      # set BEFORE creating the layers (see Good to know)

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

## Benchmark

napari-TurboBox was compared with napari-bbox 0.1.1 in real napari viewers.
Both tools replay the same seeded mouse drags in the editable XY view of the
four-viewer layout (XY editable, XZ and YZ read-only, 3D view updated at the
end of each drag). One update is one mouse move propagated to all live 2D
views. The table gives the median time per update over five runs. No frame is
drawn during the measurement.

| N boxes | napari-TurboBox (ms) | napari-bbox (ms) | Speed-up |
|---:|---:|---:|---:|
| 10 | 1.6 | 3.9 | 2.4× |
| 100 | 1.7 | 5.3 | 3.2× |
| 1,000 | 1.7 | 18.0 | 10.5× |
| 10,000 | 2.0 | 158.7 | 79× |

napari-TurboBox stays within the 33 ms budget (30 Hz) up to the largest tested
count of 10,000 boxes, napari-bbox up to about 1,000.

![Figure 3 of the Application Note: update time versus number of boxes and views, memory, speed-up, distributions and image size.](https://raw.githubusercontent.com/N-Friederich/napari-turbobox/v0.1.0/paper_results/main/Figure3_performance.png)

The full protocol, the controls (idle waiting, full reassignment, memory and
image size), the provenance of every run and the commands to reproduce it are
in [`docs/benchmark.md`](https://github.com/N-Friederich/napari-turbobox/blob/main/docs/benchmark.md). All numbers with their
confidence intervals are in
[`paper_results/RESULTS.md`](https://github.com/N-Friederich/napari-turbobox/blob/main/paper_results/RESULTS.md).

## Good to know

- **Dimensions.** Boxes have two or three spatial axes. Time and channel axes
  are not part of a box.
- **Setting the depth.** New rectangles span the full depth, so most boxes only
  need trimming. The read-only orthogonal views show the depth but do not edit
  it. To trim it with the mouse, show the depth axis in the editable viewer (for
  example `viewer.dims.order = (1, 0, 2)` for an XZ view) and drag or
  Shift-drag the box there. The API (`layer.bounding_boxes = ...`) and box
  import set it as well.
- **Image shape.** With the image shape (`image_shape`, or an image layer when
  the box layer is added from the control panel), mouse edits are clamped to
  the image and new rectangles are extruded through the full depth. Without it,
  boxes keep a minimum extent of one voxel per axis.
- **Switching between 2D and 3D.** A layer chooses its update mode (every edit
  or end of the edit) from its viewer's display when it is created. After
  switching a viewer between 2D and 3D, set `layer.sync_mode` to `"live"` or
  `"on_commit"`.
- **Class labels.** Boxes carry no class label. Per-box classes can be passed
  to the Python converters (`category_ids`). COCO and YOLO files can be
  exported but not imported.
- **napari 0.6.x.** Switching a 2D image between 2D and 3D display fails in
  napari 0.6.x for any Shapes layer. Use napari 0.7 or later if you need this.
- **Undo.** Each undo step stores a copy of all boxes.
- **Threads.** Change boxes from the main (Qt) thread.

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
- `docs/benchmark.md`: benchmark protocol, provenance and reproduction
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
