"""Convert TurboBox boxes to the COCO and YOLO 2D detection formats.

COCO and YOLO describe 2D boxes on single images. TurboBox boxes have 2 or 3
axes, and a 3D box usually spans several z slices, so the caller chooses how a
3D box becomes 2D annotations:

- ``mode="per_slice"`` (default): one 2D annotation on every integer slice k
  with ``z_min <= k <= z_max``, that is ``ceil(z_min)`` to ``floor(z_max)``.
  All of them have the box's ``(x, y, w, h)`` and class. This fits per-slice
  prompting: the 3D box gives a 2D box prompt for each slice, a model that
  takes 2D box prompts (for example nnInteractive) segments slice by slice,
  and the masks are stacked back into 3D. A box too thin to contain an
  integer z (for example z from 4.2 to 4.6) still gets one annotation, on
  slice ``round((z_min + z_max) / 2)``. With a known image shape, slice
  indices are clamped to ``[0, Z - 1]`` without a warning.
- ``mode="max_projection"``: one annotation of the box's ``(x, y, w, h)``
  footprint on a single image.

2D boxes always give one annotation each on a single image.

Coordinate convention
---------------------
Boxes use napari axis order: ``(z, y, x)`` in 3D, ``(y, x)`` in 2D. COCO/YOLO
``x`` is the last axis (column) and ``y`` the second to last (row). COCO
``bbox`` is ``[x, y, w, h]``, the top-left corner and the size in pixels.
A YOLO line is ``class x_center y_center w h``, normalized to ``[0, 1]``.
Width and height are ``x_max - x_min`` and ``y_max - y_min`` of the stored
box. No pixel is added for the upper corner.

x and y outside the image are clamped to ``[0, W]`` and ``[0, H]``, with one
warning for all clamped boxes. Such boxes should not exist: when the layer
knows the image shape, its ``add_boxes`` and ``bounding_boxes`` setter reject
them and mouse edits are clamped. Writes made directly to the
``BBoxDataStore`` are not checked, though, so an out-of-range box here points
to a problem upstream, and the warning makes it visible.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _as_minmax(boxes: np.ndarray) -> np.ndarray:
    """Return a float ``(N, 2, D)`` array, row 0 the per-axis minimum, row 1 the maximum.

    Corners given in either order are sorted per axis. Empty input gives shape
    ``(0, 2, D)``, with D = 0 unless the input already has three dimensions.
    Any other shape raises ValueError.
    """
    arr = np.asarray(boxes, dtype=float)
    if arr.size == 0:
        d = arr.shape[-1] if arr.ndim == 3 else 0
        return arr.reshape(0, 2, d)
    if arr.ndim != 3 or arr.shape[1] != 2:
        raise ValueError(f"boxes must have shape (N, 2, D); got {arr.shape}")
    mins = np.minimum(arr[:, 0], arr[:, 1])
    maxs = np.maximum(arr[:, 0], arr[:, 1])
    return np.stack([mins, maxs], axis=1)


def _image_hw(image_shape: Sequence[int] | None, ndim: int) -> tuple[int, int] | None:
    """Return ``(H, W)``, the last two axes of ``image_shape`` rounded to int, or None."""
    if image_shape is None:
        return None
    s = tuple(round(v) for v in image_shape)
    return s[-2], s[-1]


def _slice_indices(box: np.ndarray, mode: str, n_slices: int | None) -> list[int]:
    """Return the slice indices (image ids) that a 3D box is written to.

    ``max_projection`` always gives ``[0]``. ``per_slice`` gives the integer
    slices inside ``[z_min, z_max]``, or the rounded middle slice if there is
    none. With ``n_slices`` set, indices are clamped to ``[0, n_slices - 1]``.
    """
    if mode == "max_projection":
        return [0]
    z_min, z_max = float(box[0, 0]), float(box[1, 0])
    lo, hi = int(np.ceil(z_min)), int(np.floor(z_max))
    idxs = [round((z_min + z_max) / 2.0)] if lo > hi else list(range(lo, hi + 1))
    if n_slices is not None:
        idxs = [min(max(i, 0), n_slices - 1) for i in idxs]
    seen: set[int] = set()
    out: list[int] = []
    for i in idxs:  # clamping can map several indices to one slice: keep each once, in order
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _records(
    boxes: np.ndarray,
    image_shape: Sequence[int] | None,
    mode: str,
    category_ids: Sequence[int] | None,
) -> tuple[list[dict], int | None, int | None, int]:
    """Turn boxes into 2D annotation records, one per box and image.

    Returns ``(records, H, W, ndim)``. Each record is
    ``{"image_id", "x", "y", "w", "h", "cls"}``. ``image_id`` is the slice
    index for 3D boxes in ``per_slice`` mode and 0 otherwise. ``cls`` is the
    box's entry in ``category_ids``, or None. Without boxes, H and W are None.

    Raises ValueError for an unknown ``mode``, for boxes with D other than 2
    or 3, and for ``category_ids`` of the wrong length. Warns (UserWarning)
    when ``image_shape`` is None, because H and W then come from the box
    extents, and when boxes had to be clamped.
    """
    if mode not in ("per_slice", "max_projection"):
        raise ValueError(f"mode must be 'per_slice' or 'max_projection'; got {mode!r}")
    boxes = _as_minmax(boxes)
    n = boxes.shape[0]
    if n == 0:
        return [], None, None, boxes.shape[-1]
    ndim = boxes.shape[-1]
    if ndim not in (2, 3):
        raise ValueError(f"COCO/YOLO export supports 2D or 3D boxes; got D={ndim}")
    if category_ids is not None and len(category_ids) != n:
        raise ValueError(f"category_ids has length {len(category_ids)}, expected {n}")

    hw = _image_hw(image_shape, ndim)
    if hw is None:
        derived_h = int(np.ceil(boxes[:, 1, -2].max())) + 1
        derived_w = int(np.ceil(boxes[:, 1, -1].max())) + 1
        warnings.warn(
            "No image_shape provided; deriving (H, W) from box extents.",
            UserWarning,
            stacklevel=3,
        )
        hw = (derived_h, derived_w)
    h_img, w_img = hw
    n_slices = round(image_shape[0]) if (ndim == 3 and image_shape is not None) else None
    cats = list(category_ids) if category_ids is not None else [None] * n

    records: list[dict] = []
    n_clamped = 0
    for i in range(n):
        box = boxes[i]
        x_min, x_max = float(box[0, -1]), float(box[1, -1])  # last axis = x
        y_min, y_max = float(box[0, -2]), float(box[1, -2])  # 2nd-to-last = y
        cx_min, cx_max = min(max(x_min, 0.0), w_img), min(max(x_max, 0.0), w_img)
        cy_min, cy_max = min(max(y_min, 0.0), h_img), min(max(y_max, 0.0), h_img)
        if (cx_min, cx_max, cy_min, cy_max) != (x_min, x_max, y_min, y_max):
            n_clamped += 1
        x, y = cx_min, cy_min
        w, h = max(cx_max - cx_min, 0.0), max(cy_max - cy_min, 0.0)
        slices = [0] if ndim == 2 else _slice_indices(box, mode, n_slices)
        for s in slices:
            image_id = int(s) if (ndim == 3 and mode == "per_slice") else 0
            records.append({"image_id": image_id, "x": x, "y": y, "w": w, "h": h, "cls": cats[i]})
    if n_clamped:
        warnings.warn(
            f"{n_clamped} box(es) had coordinates outside image bounds and were clamped.",
            UserWarning,
            stacklevel=3,
        )
    return records, h_img, w_img, ndim


def _filename(image_id: int, ndim: int, mode: str, ext: str) -> str:
    if ndim == 3 and mode == "per_slice":
        return f"slice_{image_id:04d}.{ext}"
    return f"image.{ext}"


def boxes_to_coco(
    boxes: np.ndarray,
    image_shape: Sequence[int] | None,
    *,
    mode: str = "per_slice",
    category_ids: Sequence[int] | None = None,
    category_names: dict[int, str] | None = None,
) -> dict:
    """Convert ``(N, 2, D)`` boxes to a COCO detection dict.

    Parameters
    ----------
    boxes : np.ndarray
        ``(N, 2, D)`` boxes with D = 2 or 3, min and max corner in napari
        ``(z, y, x)`` or ``(y, x)`` order.
    image_shape : sequence of int or None
        Shape ``(Z, Y, X)`` or ``(Y, X)`` of the annotated image. Gives the
        size of the ``images`` entries, the clamping bounds and, in 3D, the
        slice range. If None, H and W are derived from the box extents (with
        a warning).
    mode : {"per_slice", "max_projection"}
        How a 3D box becomes 2D annotations (see the module docstring).
    category_ids : sequence of int, optional
        Class of each box, counted from 0. Written as
        ``category_id = class + 1``. If None, every box gets ``category_id`` 1.
    category_names : dict, optional
        Name for a class id (counted from 0). Classes without a name are
        called ``class_<id>``.

    Returns
    -------
    dict
        ``{"images": [...], "annotations": [...], "categories": [...]}``.
        ``images`` lists only the images (slices) that have at least one
        annotation. COCO ``bbox`` is ``[x, y, w, h]``. Without
        ``category_ids`` the only category is ``{"id": 1, "name": "object"}``.

    Raises
    ------
    ValueError
        For an unknown ``mode``, boxes with D other than 2 or 3, or
        ``category_ids`` whose length differs from N.
    """
    records, _h, _w, ndim = _records(boxes, image_shape, mode, category_ids)

    if category_ids is None:
        categories = [{"id": 1, "name": "object"}]
    else:
        names = category_names or {}
        categories = [
            {"id": int(c) + 1, "name": names.get(int(c), f"class_{int(c)}")}
            for c in sorted({int(c) for c in category_ids})
        ]

    images: dict[int, dict] = {}
    annotations: list[dict] = []
    for rec in records:
        iid = rec["image_id"]
        if iid not in images:
            images[iid] = {
                "id": iid,
                "file_name": _filename(iid, ndim, mode, "png"),
                "width": _w,
                "height": _h,
            }
        category_id = 1 if rec["cls"] is None else int(rec["cls"]) + 1
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": iid,
                "category_id": category_id,
                "bbox": [rec["x"], rec["y"], rec["w"], rec["h"]],
                "area": rec["w"] * rec["h"],
                "iscrowd": 0,
            }
        )
    return {"images": list(images.values()), "annotations": annotations, "categories": categories}


def boxes_to_yolo(
    boxes: np.ndarray,
    image_shape: Sequence[int] | None,
    *,
    mode: str = "per_slice",
    category_ids: Sequence[int] | None = None,
) -> dict[str, str]:
    """Convert ``(N, 2, D)`` boxes to the contents of YOLO label files.

    Parameters
    ----------
    boxes, image_shape, mode
        As in :func:`boxes_to_coco`.
    category_ids : sequence of int, optional
        Class of each box, written unchanged as the YOLO class index (YOLO
        counts from 0). If None, every box gets class 0.

    Returns
    -------
    dict of str to str
        ``{file_name: contents}``, one ``.txt`` per image that has at least
        one box: ``slice_NNNN.txt`` per slice, or ``image.txt``. The caller
        writes the files. Each line is ``class x_center y_center w h``,
        normalized to ``[0, 1]`` by the image's W and H. No boxes give ``{}``.

    Raises
    ------
    ValueError
        As in :func:`boxes_to_coco`.
    """
    records, h_img, w_img, ndim = _records(boxes, image_shape, mode, category_ids)
    if not records:
        return {}
    by_image: dict[int, list[dict]] = {}
    for rec in records:
        by_image.setdefault(rec["image_id"], []).append(rec)
    files: dict[str, str] = {}
    for iid, recs in by_image.items():
        lines = []
        for rec in recs:
            cls = 0 if rec["cls"] is None else int(rec["cls"])
            xc = (rec["x"] + rec["w"] / 2.0) / w_img
            yc = (rec["y"] + rec["h"] / 2.0) / h_img
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {rec['w'] / w_img:.6f} {rec['h'] / h_img:.6f}")
        files[_filename(iid, ndim, mode, "txt")] = "\n".join(lines) + "\n"
    return files
