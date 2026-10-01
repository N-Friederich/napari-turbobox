import itertools

import numpy as np
import pytest

from napari_turbobox.geometry import (
    MIN_EDGE_LENGTH_DATA,
    _closed_loop,
    _edge_index_template,
    bbox_from_path,
    corners_from_bbox,
    edges_from_bbox,
    ensure_bbox_array,
    extrude_2d_to_3d_box,
    face_loops_from_bbox,
    face_loops_from_bboxes,
    path_from_bbox,
    paths_from_bbox,
    paths_from_bboxes,
    slice_paths_from_bbox,
    slice_paths_from_bboxes,
    wireframe_path_from_bbox,
)


def test_ensure_bbox_array():
    # 1. data is None
    res = ensure_bbox_array(None, ndim=3)
    assert res.shape == (0, 2, 3)
    res_default = ensure_bbox_array(None)
    assert res_default.shape == (0, 2, 3)

    # 2. empty array
    res = ensure_bbox_array([], ndim=4)
    assert res.shape == (0, 2, 4)
    res_default = ensure_bbox_array([])
    assert res_default.shape == (0, 2, 3)

    # 3. 1D array
    res = ensure_bbox_array([0, 0, 10, 10], ndim=2)
    assert res.shape == (1, 2, 2)
    assert np.array_equal(res, [[[0, 0], [10, 10]]])

    # Infer ndim from 1D
    res_infer = ensure_bbox_array([0, 0, 10, 10])
    assert res_infer.shape == (1, 2, 2)

    # Odd size 1D raises error when ndim is None
    with pytest.raises(ValueError, match="Unable to infer ndim"):
        ensure_bbox_array([0, 0, 10])

    # 4. 2D array
    res = ensure_bbox_array([[0, 0], [10, 10]], ndim=2)
    assert res.shape == (1, 2, 2)

    res_infer2 = ensure_bbox_array([[0, 0], [10, 10]])
    assert res_infer2.shape == (1, 2, 2)

    # 5. 3D array
    res = ensure_bbox_array([[[0, 0], [10, 10]]], ndim=2)
    assert res.shape == (1, 2, 2)

    res_infer3 = ensure_bbox_array([[[0, 0], [10, 10]]])
    assert res_infer3.shape == (1, 2, 2)

    # 6. Higher ndim
    with pytest.raises(ValueError, match="Expected bbox array with ndim <= 3"):
        ensure_bbox_array(np.zeros((1, 1, 2, 2)))

    # 7. Wrong shape[-2] != 2
    with pytest.raises(ValueError, match="must have shape"):
        ensure_bbox_array([[[0, 0, 0]]])

    # 8. ndim mismatch
    with pytest.raises(ValueError, match="ndim mismatch"):
        ensure_bbox_array([[[0, 0], [10, 10]]], ndim=3)

    # 9. Sorting min/max order correction
    res_sort = ensure_bbox_array([[[10, 10], [0, 0]]])
    assert np.array_equal(res_sort[0, 0], [0, 0])
    assert np.array_equal(res_sort[0, 1], [10, 10])


def test_edge_index_template():
    # Valid 2D
    edges_2d = _edge_index_template(2)
    assert edges_2d.shape == (4, 2)

    # Valid 3D
    edges_3d = _edge_index_template(3)
    assert edges_3d.shape == (12, 2)

    # Invalid ndim
    with pytest.raises(ValueError, match="Only 2D and 3D"):
        _edge_index_template(4)


def test_corners_from_bbox():
    bbox = np.array([[0, 0], [10, 10]])
    corners = corners_from_bbox(bbox)
    assert corners.shape == (4, 2)
    expected = np.array([[0, 0], [0, 10], [10, 0], [10, 10]], dtype=float)
    # Check that all expected corners are present
    for exp in expected:
        assert any(np.allclose(exp, c) for c in corners)


def test_edges_from_bbox():
    bbox = np.array([[0, 0], [10, 10]])
    edges = edges_from_bbox(bbox)
    assert edges.shape == (4, 2, 2)


def test_paths_from_bbox():
    # Non-degenerate edges
    bbox = np.array([[0, 0], [10, 10]])
    paths = paths_from_bbox(bbox)
    assert len(paths) == 4

    # Degenerate edges
    bbox_deg = np.array([[0, 0], [0, 0.05]])
    paths_deg = paths_from_bbox(bbox_deg)
    assert len(paths_deg) == 0


def test_paths_from_bboxes():
    # Empty
    assert paths_from_bboxes(np.empty((0, 2, 2))) == []

    # Non-empty
    bboxes = np.array([[[0, 0], [10, 10]], [[5, 5], [15, 15]]])
    paths = paths_from_bboxes(bboxes)
    assert len(paths) == 8


def test_closed_loop():
    # Empty
    res = _closed_loop([])
    assert res.shape == (0, 0)

    # Already closed
    pts1 = [np.array([0, 0]), np.array([1, 1]), np.array([0, 0])]
    res1 = _closed_loop(pts1)
    assert len(res1) == 3
    assert np.array_equal(res1[0], res1[-1])

    # Not closed (should close it)
    pts2 = [np.array([0, 0]), np.array([1, 1])]
    res2 = _closed_loop(pts2)
    assert len(res2) == 3
    assert np.array_equal(res2[0], res2[-1])


def test_slice_paths_from_bbox():
    # Invalid displayed dims length != 2
    bbox = np.array([[0, 0, 0], [10, 10, 10]])
    assert slice_paths_from_bbox(bbox, [1], [0, 0, 0]) == []

    # Point on hidden axis within bounds
    # displayed = [1, 2], hidden = [0], step = [5, 0, 0]
    res_in = slice_paths_from_bbox(bbox, [1, 2], [5, 0, 0])
    assert len(res_in) == 1
    assert res_in[0].shape == (5, 3)

    # Point on hidden axis outside bounds (step[0] = 15, mins[0] = 0, maxs[0] = 10)
    res_out = slice_paths_from_bbox(bbox, [1, 2], [15, 0, 0])
    assert len(res_out) == 0

    # Pad step if step size < ndim
    res_pad = slice_paths_from_bbox(bbox, [1, 2], [5])
    assert len(res_pad) == 1


def test_slice_paths_from_bboxes():
    # Empty
    assert slice_paths_from_bboxes(np.empty((0, 2, 2)), [0, 1], [0, 0]) == []

    # Non-empty with display length != 2 (should fallback to loop)
    bbox = np.array([[[0, 0, 0], [10, 10, 10]]])
    res_bad_disp = slice_paths_from_bboxes(bbox, [1], [5, 0, 0])
    assert res_bad_disp == []

    # Without hidden axes (e.g. 2D bboxes)
    bbox_2d = np.array([[[0, 0], [10, 10]]])
    res_2d = slice_paths_from_bboxes(bbox_2d, [0, 1], [5, 5])
    assert len(res_2d) == 1
    assert res_2d[0].shape == (5, 2)

    # With hidden axes
    bbox_3d = np.array([[[0, 0, 0], [10, 10, 10]], [[20, 20, 20], [30, 30, 30]]])
    # step[0] = 5 intersects box 0, not box 1
    res_3d = slice_paths_from_bboxes(bbox_3d, [1, 2], [5, 0, 0])
    assert len(res_3d) == 1
    assert np.allclose(res_3d[0][:, 0], 5)

    # step[0] = 15 intersects neither
    res_3d_empty = slice_paths_from_bboxes(bbox_3d, [1, 2], [15, 0, 0])
    assert len(res_3d_empty) == 0


def test_face_loops_from_bbox():
    # Nd != 3
    bbox_2d = np.array([[0, 0], [10, 10]])
    with pytest.raises(ValueError, match="only supports 3D"):
        face_loops_from_bbox(bbox_2d)

    # Valid 3D
    bbox_3d = np.array([[0, 0, 0], [10, 10, 10]])
    loops = face_loops_from_bbox(bbox_3d)
    assert len(loops) == 6
    for loop in loops:
        assert loop.shape == (5, 3)


def test_face_loops_from_bboxes():
    # Empty
    assert face_loops_from_bboxes(np.empty((0, 2, 3))) == []

    # Non-empty
    bboxes = np.array([[[0, 0, 0], [10, 10, 10]]])
    assert len(face_loops_from_bboxes(bboxes)) == 6


def test_wireframe_path_from_bbox():
    # ndim != 3 is rejected
    with pytest.raises(ValueError, match="only supports 3D"):
        wireframe_path_from_bbox(np.array([[0, 0], [10, 10]]))

    bbox = np.array([[0, 0, 0], [10, 10, 10]], dtype=float)
    path = wireframe_path_from_bbox(bbox)

    # A single (16, 3) open polyline.
    assert path.shape == (16, 3)

    # Every vertex is one of the 8 box corners.
    corners = {tuple(c) for c in corners_from_bbox(bbox)}
    assert all(tuple(v) in corners for v in path)

    # Consecutive vertices form segments; collect the unique ones.
    segments = {frozenset((tuple(a), tuple(b))) for a, b in itertools.pairwise(path)}
    # All 12 axis-aligned box edges are covered (15 segments, 3 retraced).
    assert len(segments) == 12
    # Every segment connects corners differing on exactly one axis.
    for seg in segments:
        p, q = (np.array(x) for x in seg)
        assert int(np.sum(~np.isclose(p, q))) == 1


def test_bbox_from_path():
    # Empty
    assert bbox_from_path(np.empty((0, 2))) is None

    # All nan
    assert bbox_from_path(np.array([[np.nan, np.nan]])) is None

    # Valid
    path = np.array([[0, 1], [10, 20], [np.nan, np.nan]])
    bbox = bbox_from_path(path)
    assert bbox is not None
    assert np.array_equal(bbox, [[0, 1], [10, 20]])


def test_extrude_2d_to_3d_box():
    bbox_2d = np.array([[0, 1, 5], [10, 20, 5]])
    # 1. No hidden axes (displayed = all axes)
    res_no_hidden = extrude_2d_to_3d_box(bbox_2d, [0, 1, 2], [0, 0, 5])
    assert np.array_equal(res_no_hidden, bbox_2d)

    # 2. Hidden axis = 2, explicit extrusion range
    res_range = extrude_2d_to_3d_box(bbox_2d, [0, 1], [0, 0, 5], extrusion_range=(-10, 10))
    assert res_range[0, 2] == -10
    assert res_range[1, 2] == 10

    # 3. Hidden axis = 2, default range
    res_default = extrude_2d_to_3d_box(bbox_2d, [0, 1], [0, 0, 5])
    assert res_default[0, 2] == 0.0
    assert res_default[1, 2] == 5.0

    # 4. Step shorter than ndim
    res_short = extrude_2d_to_3d_box(bbox_2d, [0, 1], [0, 0])
    assert res_short[0, 2] == 0.0
    assert res_short[1, 2] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
