"""Geometry helpers for nD bounding boxes: input normalization and napari path data."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from functools import lru_cache
from itertools import product

import numpy as np

logger = logging.getLogger(__name__)
# paths_from_bbox leaves out edges of this length or shorter (in data units).
MIN_EDGE_LENGTH_DATA = 0.1


def ensure_bbox_array(
    data: Iterable[Sequence[Sequence[float]]] | np.ndarray | None, ndim: int | None = None
) -> np.ndarray:
    """Convert box-like input to a float64 array of shape (N, 2, ndim).

    Accepted input: None or an empty sequence (gives an empty array, with
    ndim 3 unless given), a flat sequence of 2 * ndim values (one box), a 2D
    array of shape (2, ndim) or (2 * N, ndim), or a 3D array of shape
    (N, 2, ndim). Each axis of each box is ordered so that mins <= maxs.

    A 2D input is reshaped to (-1, 2, ndim) without checking its last axis
    against a given ``ndim``: a (4, 3) array with ``ndim=2`` becomes three
    2D boxes.

    Raises
    ------
    ValueError
        If ndim cannot be inferred from a flat input of odd length, the
        input has more than 3 dimensions, a 3D input does not have shape
        (N, 2, ndim), or a flat or 2D input cannot be reshaped to
        (-1, 2, ndim).
    """
    if data is None:
        if ndim is None:
            ndim = 3
        return np.empty((0, 2, ndim), dtype=float)
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        if ndim is None:
            ndim = 3
        return np.empty((0, 2, ndim), dtype=float)
    if arr.ndim == 1:
        if ndim is None:
            if arr.size % 2 != 0:
                raise ValueError("Unable to infer ndim from 1D bbox array.")
            ndim = arr.size // 2
        arr = arr.reshape(1, 2, ndim)
    elif arr.ndim == 2:
        if ndim is None:
            ndim = arr.shape[-1]
        arr = arr.reshape(-1, 2, ndim)
    elif arr.ndim == 3:
        if ndim is None:
            ndim = arr.shape[-1]
    else:
        raise ValueError(f"Expected bbox array with ndim <= 3, got shape {arr.shape}.")
    if arr.shape[-2] != 2:
        raise ValueError(f"Bounding box array must have shape (N, 2, ndim), got {arr.shape}.")
    if ndim is not None and arr.shape[-1] != ndim:
        raise ValueError(f"Bounding box ndim mismatch: expected {ndim}, got {arr.shape[-1]}.")
    mins = np.minimum(arr[:, 0], arr[:, 1])
    maxs = np.maximum(arr[:, 0], arr[:, 1])
    return np.stack([mins, maxs], axis=1)


@lru_cache(maxsize=32)
def _edge_index_template(ndim: int) -> np.ndarray:
    """Return the corner index pairs of the edges of a unit square or cube.

    Corners are numbered as in :func:`corners_from_bbox`. Only ndim 2 and 3
    are supported (ValueError otherwise). The result is cached per ndim.
    """
    if ndim not in (2, 3):
        raise ValueError("Only 2D and 3D bounding boxes are supported.")
    corners = list(product((0, 1), repeat=ndim))
    corner_to_index = {tuple(corner): idx for idx, corner in enumerate(corners)}
    edges: list[tuple[int, int]] = []
    for idx, corner in enumerate(corners):
        for axis in range(ndim):
            neighbour = list(corner)
            neighbour[axis] = 1 - neighbour[axis]
            neighbour = tuple(neighbour)
            neighbour_idx = corner_to_index[neighbour]
            if idx < neighbour_idx:
                edges.append((idx, neighbour_idx))
    return np.asarray(edges, dtype=int)


def corners_from_bbox(bbox: np.ndarray) -> np.ndarray:
    """Return the 2**D corners of one box as a (2**D, D) array.

    The order is that of ``itertools.product((0, 1), repeat=D)``, where 0
    takes the min and 1 the max on each axis.
    """
    mins, maxs = np.asarray(bbox, dtype=float)
    ndim = mins.shape[0]
    offsets = np.array(list(product((0, 1), repeat=ndim)), dtype=float)
    ranges = maxs - mins
    return mins + offsets * ranges


def edges_from_bbox(bbox: np.ndarray) -> np.ndarray:
    """Return the edges of one 2D or 3D box as an (M, 2, D) array (M = 4 or 12)."""
    corners = corners_from_bbox(bbox)
    ndim = corners.shape[-1]
    indices = _edge_index_template(ndim)
    return corners[indices]


def paths_from_bbox(bbox: np.ndarray) -> list[np.ndarray]:
    """Return the edges of one box as separate two-point paths.

    Box coordinates are in the layer's data space, before napari applies
    scale, translate or affine. ``MIN_EDGE_LENGTH_DATA`` is therefore a
    length in data units, not in screen pixels; edges of that length or
    shorter are left out. Only 2D and 3D boxes are supported.

    Parameters
    ----------
    bbox : np.ndarray
        Bounding box array of shape (2, D) specifying [mins, maxs].

    Returns
    -------
    List[np.ndarray]
        The remaining edges, each of shape (2, D). Empty for a box with no
        extent.
    """
    edges = edges_from_bbox(bbox)
    if edges.size == 0:
        return []
    valid_edges = []
    for edge in edges:
        start, end = (edge[0], edge[1])
        length = np.linalg.norm(end - start)
        if length > MIN_EDGE_LENGTH_DATA:
            valid_edges.append(edge)
        else:
            logger.debug("Skipping degenerate edge %s -> %s (length=%.6f)", start, end, length)
    logger.debug("Generated %d valid edges from %d total edges", len(valid_edges), len(edges))
    return valid_edges


# Second name for paths_from_bbox; layer.py imports this one.
path_from_bbox = paths_from_bbox


def paths_from_bboxes(bboxes: np.ndarray) -> list[np.ndarray]:
    """Return the edge paths of all boxes as one flat list.

    Calls :func:`paths_from_bbox` once per box.
    """
    if bboxes.size == 0:
        return []
    all_paths = []
    for bbox in bboxes:
        all_paths.extend(path_from_bbox(bbox))
    return all_paths


def _closed_loop(points: Sequence[np.ndarray]) -> np.ndarray:
    """Stack ``points`` into a (P, D) array that ends at its first point.

    The first point is appended unless the last one already equals it
    (``np.allclose``). Empty input gives a (0, 0) array.
    """
    pts = [np.asarray(pt, dtype=float) for pt in points]
    if not pts:
        return np.empty((0, 0), dtype=float)
    if not np.allclose(pts[0], pts[-1]):
        pts.append(pts[0].copy())
    return np.vstack(pts)


def slice_paths_from_bbox(
    bbox: np.ndarray, dims_displayed: Sequence[int], current_step: Sequence[float]
) -> list[np.ndarray]:
    """Return the outline of one nD box in a 2D slice view.

    The result is empty unless exactly two axes are displayed and the box
    contains ``current_step`` on every hidden axis, with a tolerance of
    1e-6. Otherwise it holds one closed loop: the rectangle on the displayed
    axes, with the hidden coordinates taken from ``current_step``. The loop
    has 5 points, or 4 if the box has zero extent on the first displayed
    axis (its last corner then equals the first). A ``current_step`` shorter
    than ndim is padded with zeros at the front.
    """
    mins, maxs = np.asarray(bbox, dtype=float)
    ndim = mins.shape[0]
    displayed = tuple(int(ax) for ax in dims_displayed)
    if len(displayed) != 2:
        return []
    step = np.asarray(current_step, dtype=float)
    if step.shape[0] < ndim:
        pad_width = ndim - step.shape[0]
        step = np.pad(step, (pad_width, 0), mode="constant", constant_values=0.0)
    hidden_axes = [ax for ax in range(ndim) if ax not in displayed]
    for axis in hidden_axes:
        tolerance = 1e-06
        if step[axis] + tolerance < mins[axis] or step[axis] - tolerance > maxs[axis]:
            return []
    d0, d1 = displayed
    corners = [
        (mins[d0], mins[d1]),
        (mins[d0], maxs[d1]),
        (maxs[d0], maxs[d1]),
        (maxs[d0], mins[d1]),
    ]
    loop_points: list[np.ndarray] = []
    for cx, cy in corners:
        point = step.copy()
        point[d0] = cx
        point[d1] = cy
        loop_points.append(point)
    return [_closed_loop(loop_points)]


def slice_paths_from_bboxes(
    bboxes: np.ndarray, dims_displayed: Sequence[int], current_step: Sequence[float]
) -> list[np.ndarray]:
    """Return the outlines of many boxes in a 2D slice view.

    Gives the loops :func:`slice_paths_from_bbox` gives for each box, with
    the same 1e-6 tolerance, but tests the hidden axes and builds the loop
    coordinates for all boxes at once with NumPy. Every loop has 5 points,
    also for a box with zero extent on the first displayed axis. The result
    is empty unless exactly two axes are displayed.
    """
    if bboxes.size == 0:
        return []
    ndim = bboxes.shape[-1]
    displayed = tuple(int(ax) for ax in dims_displayed)
    if len(displayed) != 2:
        loops: list[np.ndarray] = []
        for bbox in bboxes:
            loops.extend(slice_paths_from_bbox(bbox, dims_displayed, current_step))
        return loops
    step = np.asarray(current_step, dtype=float)
    if step.shape[0] < ndim:
        pad_width = ndim - step.shape[0]
        step = np.pad(step, (pad_width, 0), mode="constant", constant_values=0.0)
    hidden_axes = [ax for ax in range(ndim) if ax not in displayed]
    if not hidden_axes:
        visible_boxes = bboxes
    else:
        mins = bboxes[:, 0, :]
        maxs = bboxes[:, 1, :]
        tolerance = 1e-06
        visible_mask = np.ones(len(bboxes), dtype=bool)
        for axis in hidden_axes:
            axis_visible = (mins[:, axis] - tolerance <= step[axis]) & (
                step[axis] <= maxs[:, axis] + tolerance
            )
            visible_mask &= axis_visible
        visible_boxes = bboxes[visible_mask]
    if visible_boxes.size == 0:
        return []
    d0, d1 = displayed
    rect_mins = visible_boxes[:, 0, [d0, d1]]
    rect_maxs = visible_boxes[:, 1, [d0, d1]]
    n_vis = len(visible_boxes)
    all_pts = np.tile(step, (n_vis, 5, 1))
    all_pts[:, 0, d0] = rect_mins[:, 0]
    all_pts[:, 0, d1] = rect_mins[:, 1]
    all_pts[:, 1, d0] = rect_mins[:, 0]
    all_pts[:, 1, d1] = rect_maxs[:, 1]
    all_pts[:, 2, d0] = rect_maxs[:, 0]
    all_pts[:, 2, d1] = rect_maxs[:, 1]
    all_pts[:, 3, d0] = rect_maxs[:, 0]
    all_pts[:, 3, d1] = rect_mins[:, 1]
    all_pts[:, 4, d0] = rect_mins[:, 0]
    all_pts[:, 4, d1] = rect_mins[:, 1]
    return list(all_pts)


def face_loops_from_bbox(bbox: np.ndarray) -> list[np.ndarray]:
    """Return the six faces of a 3D box as closed loops.

    TurboBoxLayer does not use this; it draws one path per box with
    :func:`wireframe_path_from_bbox`.
    """
    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape[-1] != 3:
        raise ValueError("face_loops_from_bbox only supports 3D boxes.")
    mins, maxs = bbox
    z_loops = [
        [
            (mins[0], mins[1], mins[2]),
            (mins[0], maxs[1], mins[2]),
            (maxs[0], maxs[1], mins[2]),
            (maxs[0], mins[1], mins[2]),
        ],
        [
            (mins[0], mins[1], maxs[2]),
            (mins[0], maxs[1], maxs[2]),
            (maxs[0], maxs[1], maxs[2]),
            (maxs[0], mins[1], maxs[2]),
        ],
    ]
    x_loops = [
        [
            (mins[0], mins[1], mins[2]),
            (mins[0], mins[1], maxs[2]),
            (mins[0], maxs[1], maxs[2]),
            (mins[0], maxs[1], mins[2]),
        ],
        [
            (maxs[0], mins[1], mins[2]),
            (maxs[0], mins[1], maxs[2]),
            (maxs[0], maxs[1], maxs[2]),
            (maxs[0], maxs[1], mins[2]),
        ],
    ]
    y_loops = [
        [
            (mins[0], mins[1], mins[2]),
            (mins[0], mins[1], maxs[2]),
            (maxs[0], mins[1], maxs[2]),
            (maxs[0], mins[1], mins[2]),
        ],
        [
            (mins[0], maxs[1], mins[2]),
            (mins[0], maxs[1], maxs[2]),
            (maxs[0], maxs[1], maxs[2]),
            (maxs[0], maxs[1], mins[2]),
        ],
    ]
    loops = []
    for faces in (z_loops, x_loops, y_loops):
        for face in faces:
            loops.append(_closed_loop(face))
    return loops


# Corner order that draws all 12 edges of a 3D box as one open polyline.
# Corner index = binary (axis0, axis1, axis2), the itertools.product((0, 1),
# repeat=3) order used by corners_from_bbox.
#
#   0-1-3-2-0   face at the axis-0 minimum
#   0-4         edge to the face at the axis-0 maximum
#   4-5-7-6-4   face at the axis-0 maximum
#   4-5 1-3 7-6 are each drawn a second time to reach the three remaining
#               axis-0 edges 1-5, 3-7 and 2-6.
#
# All 8 corners have odd degree (3), so no path draws every edge exactly once;
# at least 3 edges have to be drawn twice, and this order repeats exactly 3.
# Result: 16 vertices and 15 segments covering the 12 distinct edges.
_WIREFRAME_TRAVERSAL = (0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2)


def wireframe_path_from_bbox(bbox: np.ndarray) -> np.ndarray:
    """Return one (16, 3) path that draws all 12 edges of a 3D box.

    The 8 corners are visited in the fixed order ``_WIREFRAME_TRAVERSAL``.
    Every corner of a cube has odd degree (3), so a single open polyline has
    to draw at least 3 edges twice; this order draws exactly 3 twice
    (corner4-corner5, corner1-corner3 and corner6-corner7). The path has 16
    vertices and 15 segments covering the 12 distinct edges.

    With one path per box, the layer has one napari shape per box, as in
    napari-bbox; six face loops would need six shapes per box.

    Parameters
    ----------
    bbox : np.ndarray
        Bounding box of shape (2, 3), i.e. [mins, maxs].

    Returns
    -------
    np.ndarray
        A (16, 3) array of path vertices. Add it to napari with
        ``shape_type='path'``: the polyline is not planar and touches
        itself, so it must not be triangulated as a polygon.

    Raises
    ------
    ValueError
        If the box is not 3D.
    """
    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape[-1] != 3:
        raise ValueError("wireframe_path_from_bbox only supports 3D boxes.")
    corners = corners_from_bbox(bbox)
    return corners[list(_WIREFRAME_TRAVERSAL)]


def face_loops_from_bboxes(bboxes: np.ndarray) -> list[np.ndarray]:
    """Return the face loops of all 3D boxes as one flat list (six per box)."""
    if bboxes.size == 0:
        return []
    loops: list[np.ndarray] = []
    for bbox in bboxes:
        loops.extend(face_loops_from_bbox(bbox))
    return loops


def bbox_from_path(path: np.ndarray) -> np.ndarray | None:
    """Return the (2, D) ``[mins, maxs]`` of the rows of a path without NaN.

    Returns None if the path is empty or every row contains a NaN.
    """
    points = np.asarray(path, dtype=float)
    if points.size == 0:
        return None
    mask = ~np.isnan(points).any(axis=1)
    points = points[mask]
    if points.size == 0:
        return None
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return np.stack([mins, maxs], axis=0)


def extrude_2d_to_3d_box(
    bbox_2d: np.ndarray,
    displayed_axes: Sequence[int],
    current_step: Sequence[float],
    extrusion_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Extend a box drawn in a 2D view along every hidden axis.

    Parameters
    ----------
    bbox_2d : np.ndarray
        A (2, ndim) array ``[mins, maxs]`` in nD space. Its values on the
        hidden axes are replaced.
    displayed_axes : Sequence[int]
        The axes that are currently displayed (typically [1, 2] for y,x in a z-stack).
    current_step : Sequence[float]
        The current position in each dimension.
    extrusion_range : Tuple[float, float] | None
        (min, max) for every hidden axis. If None, each hidden axis spans
        0 to ``max(current_step[axis], 1.0)``, where a missing
        ``current_step`` entry counts as 0.

    Returns
    -------
    np.ndarray
        A new (2, ndim) array. If no axis is hidden, an unchanged copy.
    """
    bbox = np.asarray(bbox_2d, dtype=float).copy()
    mins, maxs = bbox
    ndim = mins.shape[0]
    hidden_axes = [ax for ax in range(ndim) if ax not in displayed_axes]
    if not hidden_axes:
        return bbox
    for axis in hidden_axes:
        if extrusion_range is not None:
            mins[axis] = extrusion_range[0]
            maxs[axis] = extrusion_range[1]
        else:
            current_pos = current_step[axis] if axis < len(current_step) else 0.0
            mins[axis] = 0.0
            maxs[axis] = max(current_pos, 1.0)
    return np.stack([mins, maxs], axis=0)
