"""Turn an instance label volume into napari-TurboBox boxes and export them.

This is the generic application workflow. Any 2D or 3D image with a matching
instance label array (0 = background, one positive integer per object), from
any modality, becomes one axis-aligned box per object. The boxes are either
exported in the formats the BBox Control Panel writes, or opened in the
synchronized multi-view layout for review and correction.

Usage::

    python examples/labels_to_boxes.py IMAGE LABELS [--out DIR] [--category-name NAME]
    python examples/labels_to_boxes.py IMAGE LABELS --gui

IMAGE and LABELS are ``.tif``/``.tiff`` (read with tifffile) or ``.npy`` files
with the same shape once length-1 axes are dropped: ``(Z, Y, X)`` or
``(Y, X)``, in napari axis order. Time and channel axes are not supported.

Box convention (the invariant the plugin enforces): inclusive voxel index
ranges ``[min, max]`` per axis with ``0 <= min``, ``max <= shape - 1`` and
``max - min >= 1``. ``min`` is the first and ``max`` the last voxel index of
the object (``scipy.ndimage.find_objects`` stop - 1). An object one voxel
thick along an axis is widened to ``max = min + 1`` (``min = max - 1`` at the
upper image border).

Files written to ``--out``:

- ``boxes.npy``: ``(N, 2, D)`` float32 boxes, the same format as the panel's
  native export; "Import BBoxes..." in the panel appends them to a layer.
- ``label_ids.npy``: the instance label of each box (row i of ``boxes.npy``).
- ``coco_per_slice.json``: COCO detection file. A 3D box gives one 2D box on
  every integer z slice from ceil(z_min) to floor(z_max). Images are named
  ``slice_NNNN.png`` (``image.png`` for 2D); you supply those images.
- ``yolo/slice_NNNN.txt`` (``image.txt`` for 2D): the same 2D boxes as YOLO
  labels (class 0).

COCO/YOLO width and height are ``max - min``: an object covering columns
10..15 gets width 5. Add 1 if your detector expects voxel counts.

The export runs without a viewer window, but importing ``napari_turbobox``
still imports napari and needs a Qt binding (PyQt5, PyQt6 or PySide6).
``--gui`` opens the panel through the plugin manifest, so napari-turbobox must
be installed (``pip install -e .``), not only on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_volume(path: str | Path) -> np.ndarray:
    """Read a ``.tif``/``.tiff`` (tifffile) or ``.npy`` file and drop length-1 axes."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile

        data = tifffile.imread(path)
    elif suffix == ".npy":
        data = np.load(path)
    else:
        raise ValueError(f"{path}: expected a .tif, .tiff or .npy file")
    return np.squeeze(data)


def labels_to_boxes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(boxes, label_ids)``: one inclusive ``(min, max)`` box per label.

    ``boxes`` has shape ``(N, 2, D)`` (float), ordered by ascending label id.
    Label ids should be reasonably dense: ``find_objects`` keeps one entry per
    id up to the largest id.
    """
    from scipy import ndimage

    labels = np.asarray(labels)
    if labels.ndim not in (2, 3) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"labels must be a 2D or 3D integer array; got {labels.ndim}D {labels.dtype}"
        )
    upper = np.asarray(labels.shape, dtype=float) - 1
    if np.any(upper < 1):
        raise ValueError(f"every axis needs at least 2 voxels; got shape {labels.shape}")
    boxes, ids = [], []
    for label_id, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:  # this label id does not occur
            continue
        mins = np.array([s.start for s in slices], dtype=float)
        maxs = np.array([s.stop - 1 for s in slices], dtype=float)  # last voxel, inclusive
        maxs = np.minimum(np.maximum(maxs, mins + 1), upper)  # extent >= 1, inside image
        mins = np.minimum(mins, maxs - 1)  # one-voxel object at the upper border
        boxes.append([mins, maxs])
        ids.append(label_id)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, labels.ndim)
    return boxes, np.asarray(ids, dtype=np.int64)


def export_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, ...],
    out_dir: str | Path,
    label_ids: np.ndarray | None = None,
    category_name: str | None = None,
) -> dict[str, int]:
    """Write ``boxes.npy``, ``coco_per_slice.json`` and ``yolo/``; return the counts."""
    from napari_turbobox import BBoxDataStore
    from napari_turbobox.export import boxes_to_coco, boxes_to_yolo

    # The store every TurboBox layer reads from. It holds float32, so boxes.npy
    # has the dtype of the panel's native export.
    store = BBoxDataStore(initial_data=boxes)
    data = store.data
    shape = tuple(int(s) for s in image_shape)
    out = Path(out_dir)
    (out / "yolo").mkdir(parents=True, exist_ok=True)
    np.save(out / "boxes.npy", data)
    if label_ids is not None:
        np.save(out / "label_ids.npy", np.asarray(label_ids))
    cats = None if category_name is None else [0] * len(data)
    names = None if category_name is None else {0: category_name}
    coco = boxes_to_coco(data, shape, category_ids=cats, category_names=names)
    (out / "coco_per_slice.json").write_text(json.dumps(coco))
    yolo = boxes_to_yolo(data, shape, category_ids=cats)
    for name, text in yolo.items():
        (out / "yolo" / name).write_text(text)
    return {
        "boxes": len(data),
        "coco_images": len(coco["images"]),
        "coco_annotations": len(coco["annotations"]),
        "yolo_files": len(yolo),
    }


def open_multiview(image: np.ndarray, boxes: np.ndarray):
    """Show image and boxes: XY (editable), XZ and YZ (read-only), 3D (read-only).

    Returns ``(viewers, layers)``; call ``napari.run()`` afterwards.
    """
    import napari

    from napari_turbobox import create_synchronized_bbox_layers

    shape = tuple(int(s) for s in image.shape)
    viewers = []
    if image.ndim == 2:
        viewers.append(napari.Viewer(title="TurboBox (editable)"))
        viewers[0].add_image(image, name="image")
    else:
        # (title, dims.order, sliced axis); napari displays the last two axes of the order
        planes = [
            ("XY (editable)", (0, 1, 2), 0),
            ("XZ (read-only)", (1, 0, 2), 1),
            ("YZ (read-only)", (2, 0, 1), 2),
        ]
        for title, order, axis in planes:
            viewer = napari.Viewer(title=title)
            viewer.add_image(image, name="image")
            viewer.dims.order = order
            viewer.dims.set_point(axis, shape[axis] // 2)
            viewers.append(viewer)
        viewer_3d = napari.Viewer(title="3D (read-only)")
        viewer_3d.add_image(image, name="image")
        viewer_3d.dims.ndisplay = 3
        viewers.append(viewer_3d)
    # Each layer's sync mode is read from its viewer's ndisplay when the layers
    # are created, so the viewers are configured first: the 2D views follow every
    # change, including each step of a drag; the 3D view updates once per drag,
    # when the mouse is released.
    layers = create_synchronized_bbox_layers(
        viewers[0], viewers[1:], image_shape=shape, ndim=image.ndim
    )
    layers[0].add_boxes(boxes)  # raises ValueError if a box breaks the invariant
    viewers[0].window.add_plugin_dock_widget("napari-turbobox", "BBox Control Panel")
    return viewers, layers


def print_counts(counts: dict[str, int], out_dir: str | Path) -> None:
    print(
        f"{counts['boxes']} boxes -> {Path(out_dir) / 'boxes.npy'}\n"
        f"COCO: {counts['coco_images']} images, {counts['coco_annotations']} annotations"
        f" -> {Path(out_dir) / 'coco_per_slice.json'}\n"
        f"YOLO: {counts['yolo_files']} label files -> {Path(out_dir) / 'yolo'}"
    )


def main(argv: list[str] | None = None) -> dict[str, int] | None:
    parser = argparse.ArgumentParser(description="Instance labels -> TurboBox boxes -> exports.")
    parser.add_argument("image", help="image volume (.tif, .tiff or .npy)")
    parser.add_argument("labels", help="instance label volume of the same shape")
    parser.add_argument("--out", default="turbobox_export", help="output directory")
    parser.add_argument("--category-name", default=None, help="COCO category name")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="export files (default)")
    mode.add_argument("--gui", action="store_true", help="open napari instead of exporting")
    args = parser.parse_args(argv)

    image, labels = load_volume(args.image), load_volume(args.labels)
    if image.shape != labels.shape:
        parser.error(f"shape mismatch: image {image.shape}, labels {labels.shape}")
    boxes, ids = labels_to_boxes(labels)
    print(f"{len(boxes)} boxes (one per label), volume shape {labels.shape}")
    if args.gui:
        import napari

        _session = open_multiview(image, boxes)  # keep references while napari runs
        napari.run()
        return None
    counts = export_boxes(boxes, image.shape, args.out, ids, args.category_name)
    print_counts(counts, args.out)
    return counts


if __name__ == "__main__":
    main()
