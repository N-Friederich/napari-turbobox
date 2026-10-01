"""A box that enters or leaves a view's slice must update that view like a full rebuild.

``TurboBoxLayer._add_or_remove_shape`` adds or removes the one shape of such a box
instead of re-creating every visible shape. Each scenario runs on two identical
layouts: the layer as shipped, and a reference in which an enter or leave always
takes the full rebuild (``_add_or_remove_shape`` returns False). After every step
both must show the same thing, compared array by array: napari's shape list
(vertices, index arrays, z-order, colours, displayed arrays), the vispy mesh, the
thumbnail, selection, hover, the drawn highlight, extent, text and current style.
Further scenarios: styles that keep the rebuild when the first crossing is an
enter, a filled face (with the default transparent face the thumbnail stays
blank), napari's interaction state, read-only views, and napari raising at five
points inside an enter or leave (``FAILURES``). ``_move_last_shape`` is also checked
on its own against napari's shape list, with shapes whose row blocks differ in size.

The second half covers the in-slice fast path (``_edit_shape``) with napari's
z-order, a one-pixel box, napari's batched updates, napari raising part-way through
an in-place edit, and zero-extent boxes.
"""

import copy
import logging

import napari
import numpy as np
import pytest
from napari.layers import Shapes
from napari.layers.shapes._shape_list import ShapeList
from napari.layers.shapes._shapes_constants import Mode
from napari.layers.shapes._shapes_models import Polygon
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)
from napari.utils.triangulation_backend import TriangulationBackend, get_backend, set_backend
from packaging.version import Version

import napari_turbobox.layer as layer_module
from napari_turbobox import BBoxDataStore, TurboBoxLayer
from napari_turbobox.geometry import slice_paths_from_bbox, wireframe_path_from_bbox
from napari_turbobox.layer import _move_last_shape

SHAPE = (20, 100, 100)
Z, Y, X = 10, 50, 50  # slice of the XY view (z), the XZ view (y) and the YZ view (x)
VIEWS = ("xy", "xz", "yz", "3d")

# Box 2 is the one that gets dragged. It is visible in every view; the other boxes
# sit before and after it in the XZ and YZ views, so it re-enters in the middle.
BOXES = np.array([
    [[2, 45, 10], [12, 60, 25]],    # 0: XY, XZ
    [[0, 5, 45], [8, 20, 60]],      # 1: YZ only
    [[5, 40, 40], [15, 56, 56]],    # 2: XY, XZ, YZ
    [[8, 48, 70], [18, 62, 90]],    # 3: XY, XZ
    [[6, 70, 48], [16, 90, 58]],    # 4: XY, YZ
    [[0, 30, 30], [4, 52, 52]],     # 5: XZ, YZ
    [[12, 5, 5], [19, 20, 20]],     # 6: none of the three slices
], float)
DRAGGED = 2

# Arrays of napari's shape list compared exactly (values, shape and dtype).
LIST_ARRAYS = ("_vertices", "_vertices_index", "_z_index", "_z_order", "_displayed",
               "displayed_vertices", "displayed_vertices_to_shape_num", "displayed_index",
               "_edge_color", "_face_color")
MESH_ARRAYS = ("vertices_index", "triangles_index", "triangles_colors", "triangles_z_order",
               "displayed_triangles_colors", "displayed_triangles_to_shape_index")
# Filled by napari's triangulation. With the compiled backend (bermuda) the edge strip
# of a polygon can start at another corner for the same data, so these are compared
# as the triangles they draw, and exactly under the pure-Python backend.
MESH_GEOMETRY = ("vertices", "vertices_centers", "vertices_offsets", "triangles", "displayed_triangles")


def _triangle_set(vertices, triangles, colors=None):
    """The triangles drawn, as sorted rows of vertex keys (and colour), in no order."""
    vertices = np.asarray(vertices, float)
    triangles = np.asarray(triangles, np.int64).reshape(-1, 3)
    if len(triangles) == 0:
        return np.zeros((0, 4), np.int64)
    corners = vertices[triangles]
    e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    if vertices.shape[1] == 2:
        area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    else:
        area = np.linalg.norm(np.cross(e1, e2), axis=1)
    keep = area > 1e-9
    digits = np.round(vertices * 1000).astype(np.int64) + (1 << 19)
    key = digits[:, 0]
    for j in range(1, digits.shape[1]):
        key = key * (1 << 20) + digits[:, j]
    rows = np.sort(key[triangles[keep]], axis=1)
    if colors is not None:
        c = np.clip(np.round(np.asarray(colors, float)[keep] * 255), 0, 255).astype(np.int64)
        rows = np.column_stack([rows, ((c[:, 0] * 256 + c[:, 1]) * 256 + c[:, 2]) * 256 + c[:, 3]])
    return rows[np.lexsort(rows.T[::-1])]


def _visual(viewer, layer):
    qt_viewer = viewer.window._qt_viewer
    layer_to_visual = getattr(getattr(qt_viewer, "canvas", qt_viewer), "layer_to_visual", None)
    if layer_to_visual is None:
        layer_to_visual = qt_viewer.layer_to_visual
    return layer_to_visual[layer]


def _state(viewer, layer, raw_mesh=False):
    """Everything a view shows or keeps per shape (``raw_mesh``: also the raw mesh arrays)."""
    dv = layer._data_view
    mesh = dv._mesh
    state = {
        "data": [np.asarray(d, float) for d in layer.data],
        "shape_type": list(layer.shape_type),
        "box_ids": None if layer._displayed_box_ids() is None else np.asarray(layer._displayed_box_ids()),
        "edge_width": [float(w) for w in layer.edge_width],
        "selected": sorted(layer.selected_data),
        "value": repr(layer._value),
        "selected_box": getattr(layer, "_selected_box", None),
        "features": len(layer.features),
        "thumbnail": np.asarray(layer.thumbnail),
        "mesh": _triangle_set(mesh.vertices, mesh.triangles, mesh.triangles_colors),
        "mesh_displayed": _triangle_set(mesh.vertices, mesh.displayed_triangles,
                                        mesh.displayed_triangles_colors),
    }
    for name in LIST_ARRAYS:
        state["dv." + name] = np.asarray(getattr(dv, name))
    for name in MESH_ARRAYS + (MESH_GEOMETRY if raw_mesh else ()):
        state["mesh." + name] = np.asarray(getattr(mesh, name))
    mesh_data = _visual(viewer, layer).node.shape_faces.mesh_data
    faces = mesh_data.get_faces()
    state["vispy"] = _triangle_set(mesh_data.get_vertices(), [] if faces is None else faces,
                                   mesh_data.get_face_colors())
    outline, triangles = layer._outline_shapes()
    state["outline"] = None if outline is None else _triangle_set(outline, triangles)
    # The highlight as drawn (hover / selection outline and interaction box).
    node = _visual(viewer, layer).node
    highlight = node.shape_highlights.mesh_data
    vertices, faces = highlight.get_vertices(), highlight.get_faces()
    state["highlight"] = None if vertices is None or faces is None else _triangle_set(vertices, faces)
    lines = getattr(node.highlight_lines, "pos", None)
    state["highlight_lines"] = None if lines is None else np.asarray(lines)
    state["extent"] = np.asarray(layer.extent.data, float)
    state["dims_range"] = repr(viewer.dims.range)
    state["current_style"] = (float(layer.current_edge_width), str(layer.current_edge_color),
                              str(layer.current_face_color))
    try:
        state["text"] = repr(np.asarray(layer._view_text).tolist())
    except Exception as exc:  # napari may not render text for this state
        state["text"] = f"raised {type(exc).__name__}"
    return state


def _assert_same(a, b, where, nan_ok=()):
    """Two ``_state`` dicts are equal. Float arrays must be finite, except the keys in
    ``nan_ok``, where a NaN in both counts as equal (see ``_zero_extent_nan``)."""
    assert a.keys() == b.keys()
    for key in a:
        x, y = a[key], b[key]
        if key == "data":
            assert len(x) == len(y), f"{where}: {key} count {len(x)} != {len(y)}"
            for i, (p, q) in enumerate(zip(x, y)):
                assert np.array_equal(p, q), f"{where}: shape {i} differs"
        elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), f"{where}: {key}"
            assert x.dtype == y.dtype, f"{where}: {key} dtype {x.dtype} != {y.dtype}"
            nan = x.dtype.kind == "f" and key in nan_ok
            if x.dtype.kind == "f" and not nan:
                assert np.isfinite(x).all() and np.isfinite(y).all(), f"{where}: {key} not finite"
            assert x.shape == y.shape and np.array_equal(x, y, equal_nan=nan), f"{where}: {key} differs"
        else:
            assert x == y, f"{where}: {key} {x!r} != {y!r}"


def _assert_consistent(layer, where):
    """napari's derived arrays equal a forced recomputation (nothing left stale)."""
    dv = layer._data_view
    forced = copy.deepcopy(dv)
    with forced.batched_updates():
        forced._update_z_order()
    for name in ("_z_order", "_displayed", "displayed_vertices", "displayed_vertices_to_shape_num",
                 "displayed_index"):
        assert np.array_equal(getattr(dv, name), getattr(forced, name)), f"{where}: stale {name}"
    for name in ("triangles_z_order", "displayed_triangles", "displayed_triangles_colors",
                 "displayed_triangles_to_shape_index"):
        live, fresh = getattr(dv._mesh, name), getattr(forced._mesh, name)
        assert np.array_equal(live, fresh), f"{where}: stale mesh.{name}"


def _assert_matches_store(layer, where):
    """Shape k shows the k-th visible box in index order, and the index maps agree."""
    store = layer.store.data
    if layer._is_3d_view():
        expected = [wireframe_path_from_bbox(b) for b in store]
        ids = np.arange(len(store))
    else:
        point = layer._slice_point_data()
        displayed = tuple(layer._slice_input.displayed)
        hidden = [a for a in range(layer.ndim) if a not in displayed]
        ids = np.array([i for i, b in enumerate(store)
                        if all(b[0, a] - 1e-6 <= point[a] <= b[1, a] + 1e-6 for a in hidden)], np.intp)
        expected = [slice_paths_from_bbox(store[i], displayed, point)[0] for i in ids]
        assert np.array_equal(layer._current_shape_box_ids(), ids), f"{where}: box ids"
        for pos, idx in enumerate(ids):
            assert layer._bbox_index_to_path_index(int(idx)) == pos, f"{where}: box {idx}"
            assert layer._path_index_to_bbox_index(pos) == int(idx), f"{where}: shape {pos}"
    assert len(layer.data) == len(expected), f"{where}: {len(layer.data)} shapes, {len(expected)} boxes"
    for pos, (shown, path) in enumerate(zip(layer.data, expected)):
        assert np.allclose(shown, path, atol=1e-9), f"{where}: shape {pos} (box {ids[pos]})"


class Layout:
    """XY (editable), XZ (editable), YZ (read-only) and 3D (on_commit) views of one store.

    ``reference=True`` makes every enter or leave take the full rebuild. Calls of
    the full rebuild and of ``_add_or_remove_shape`` are counted per view.
    ``layer_kw`` go to every ``TurboBoxLayer`` (e.g. a ``face_color``).
    """

    def __init__(self, make_napari_viewer, boxes=BOXES, reference=False, **layer_kw):
        self.store = BBoxDataStore(initial_data=np.asarray(boxes, float))
        self.viewers, self.layers = [], []
        self.rebuilds = dict.fromkeys(VIEWS, 0)
        self.added_or_removed = dict.fromkeys(VIEWS, 0)
        setups = (((0, 1, 2), 0, Z, True, "live"), ((1, 0, 2), 1, Y, True, "live"),
                  ((2, 0, 1), 2, X, False, "live"), (None, None, None, False, "on_commit"))
        for name, (order, axis, point, interactive, sync_mode) in zip(VIEWS, setups):
            viewer = make_napari_viewer()
            viewer.add_image(np.zeros(SHAPE, np.uint8))
            if order is None:
                viewer.dims.ndisplay = 3
            else:
                viewer.dims.order = order
                viewer.dims.set_point(axis, point)
            layer = TurboBoxLayer(ndim=3, image_shape=SHAPE, bbox_data_store=self.store,
                                  interactive=interactive, sync_mode=sync_mode, **layer_kw)
            viewer.add_layer(layer)
            self._count(name, layer, reference)
            self.viewers.append(viewer)
            self.layers.append(layer)
        self.layers[0].mode = Mode.SELECT
        self.reset_counts()

    def _count(self, name, layer, reference):
        rebuild, add_or_remove = layer._sync_shapes_from_bboxes, layer._add_or_remove_shape

        def counted_rebuild():
            self.rebuilds[name] += 1
            return rebuild()

        def counted_add_or_remove(pos, box_idx, path):
            done = False if reference else add_or_remove(pos, box_idx, path)
            self.added_or_removed[name] += int(done)
            return done

        layer._sync_shapes_from_bboxes = counted_rebuild
        layer._add_or_remove_shape = counted_add_or_remove

    def reset_counts(self):
        for counts in (self.rebuilds, self.added_or_removed):
            for name in counts:
                counts[name] = 0

    def layer(self, name):
        return self.layers[VIEWS.index(name)]

    def states(self, raw_mesh=False):
        return {name: _state(v, lay, raw_mesh) for name, v, lay in zip(VIEWS, self.viewers, self.layers)}


def _check(layout, ref, where, raw_mesh=False, nan_ok=None):
    """The layout equals the reference view by view and matches the store.

    napari's mesh vertices must be finite: the triangle sets in ``_state`` skip
    triangles with a NaN corner, so they alone would not notice. ``nan_ok`` maps a
    view to the state keys that may hold NaN; only the zero-extent test passes it.
    """
    nan_ok = nan_ok or {}
    got, expected = layout.states(raw_mesh), ref.states(raw_mesh)
    for name in VIEWS:
        allowed = nan_ok.get(name, ())
        _assert_same(got[name], expected[name], f"{where} [{name}]", allowed)
        if "mesh.vertices" not in allowed:
            for lay in (layout, ref):
                vertices = lay.layer(name)._data_view._mesh.vertices
                assert np.isfinite(vertices).all(), f"{where} [{name}]: non-finite mesh vertices"
        layer = layout.layer(name)
        _assert_consistent(layer, f"{where} [{name}]")
        if name != "3d" or not layout.store.in_edit_session:
            _assert_matches_store(layer, f"{where} [{name}]")


def _both(make_napari_viewer, boxes=BOXES, **layer_kw):
    return (Layout(make_napari_viewer, boxes, **layer_kw),
            Layout(make_napari_viewer, boxes, reference=True, **layer_kw))


def _update(layouts, idx, box):
    """One store update (no edit session) in every layout."""
    for layout in layouts:
        layout.store.update_box(idx, np.asarray(box, float))


@pytest.fixture
def pure_python_triangulation():
    """Deterministic triangulation; set after the viewers exist (they re-apply the setting)."""
    previous = get_backend()
    yield lambda: set_backend(TriangulationBackend.pure_python)
    set_backend(previous)


def _mouse(layer, kind, position, modifiers=()):
    kw = {"modifiers": list(modifiers), "dims_displayed": [1, 2], "view_direction": None,
          "up_direction": None}
    handler = {"press": mouse_press_callbacks, "move": mouse_move_callbacks,
               "release": mouse_release_callbacks}[kind]
    extra = {"is_dragging": True} if kind == "move" else {}
    handler(layer, read_only_mouse_event(type=f"mouse_{kind}", position=tuple(position), **kw, **extra))


def _drag_through(layouts, points, after_move=None, modifiers=()):
    """Press at points[0], move through the rest, release, in the XY view of every layout."""
    for layout in layouts:
        _mouse(layout.layer("xy"), "press", points[0], modifiers)
    for step, point in enumerate(points[1:]):
        for layout in layouts:
            _mouse(layout.layer("xy"), "move", point, modifiers)
        if after_move is not None:
            after_move(step)
    for layout in layouts:
        _mouse(layout.layer("xy"), "release", points[-1], modifiers)


def _path(start, stop, n):
    return [np.asarray(start, float) + t * (np.asarray(stop, float) - np.asarray(start, float))
            for t in np.linspace(0.0, 1.0, n)]


# The dragged box leaves the XZ (y = 50) and YZ (x = 50) slices and comes back.
OUT_AND_BACK = _path((Z, 48, 48), (Z, 70, 70), 8) + _path((Z, 70, 70), (Z, 46, 47), 8)[1:]


def _store_moves(layout, idx, boxes):
    store = layout.store
    store.begin_edit_session()
    for box in boxes:
        store.update_box(idx, np.asarray(box, float))
        yield
    store.end_edit_session()
    yield


def _run_store_moves(layouts, idx, boxes, where):
    steps = [_store_moves(layout, idx, boxes) for layout in layouts]
    for step in range(len(boxes) + 1):
        for moves in steps:
            next(moves)
        _check(layouts[0], layouts[1], f"{where} step {step}")


def _shifted(box, dz=0.0, dy=0.0, dx=0.0):
    return np.asarray(box, float) + np.array([dz, dy, dx])


def _off_slice(box, axis, at):
    """The box moved along ``axis`` just past the slice ``at`` (inside the image)."""
    box = np.asarray(box, float).copy()
    size = box[1, axis] - box[0, axis]
    low = at + 1 if at + 1 + size <= SHAPE[axis] - 1 else at - 1 - size
    box[:, axis] += low - box[0, axis]
    return box


def _valid(box):
    """Clip a box into the image with an extent of at least 1 on every axis."""
    upper = np.array(SHAPE, float) - 1
    box = np.clip(np.asarray(box, float), 0, upper)
    box[1] = np.minimum(np.maximum(box[1], box[0] + 1), upper)
    box[0] = np.minimum(box[0], box[1] - 1)
    return box


# --------------------------------------------------------------------- enter / leave


@pytest.mark.parametrize("idx", [0, DRAGGED, 3, 5], ids=["first", "middle", "last_in_xz", "yz_xz"])
def test_store_moves_across_every_view(make_napari_viewer, idx):
    """Store edits move one box out of and back into each view's slice."""
    layout, ref = _both(make_napari_viewer)
    _check(layout, ref, "initial")
    box = BOXES[idx]
    off_y = _off_slice(box, 1, Y)
    boxes = [off_y, _off_slice(off_y, 2, X), _off_slice(box, 2, X), box, _off_slice(box, 0, Z), box]
    _run_store_moves((layout, ref), idx, boxes, f"box {idx}")
    assert sum(layout.rebuilds.values()) == 0, layout.rebuilds
    assert sum(layout.added_or_removed.values()) >= 4, layout.added_or_removed
    assert sum(ref.rebuilds.values()) >= 4  # the reference really crossed slices


def test_insert_positions_first_middle_last(make_napari_viewer):
    """An entering box lands at its place in the sorted order: first, middle or last."""
    layout, ref = _both(make_napari_viewer)
    xz = layout.layer("xz")
    assert list(xz._displayed_box_ids()) == [0, DRAGGED, 3, 5]
    for idx, pos in ((0, 0), (DRAGGED, 1), (5, 3)):
        _run_store_moves((layout, ref), idx, [_off_slice(BOXES[idx], 1, Y), BOXES[idx]], f"box {idx}")
        assert xz._bbox_index_to_path_index(idx) == pos
    assert layout.rebuilds["xz"] == 0 and layout.added_or_removed["xz"] == 6


def test_mouse_drag_across_sub_view_slices(make_napari_viewer):
    """A mouse drag in XY takes the box out of the XZ and YZ slices and back:
    no full rebuild anywhere, the box stays selected, every step equals the reference."""
    layout, ref = _both(make_napari_viewer)
    xy = layout.layer("xy")
    seen = {"gone": False}

    def after_move(step):
        _check(layout, ref, f"move {step}")
        assert layout.store.in_edit_session
        ids = xy._displayed_box_ids()
        assert sorted(xy.selected_data) == [list(ids).index(DRAGGED)]
        assert xy._selected_box is not None
        assert xy._outline_shapes()[0] is not None  # the dragged box is outlined
        seen["gone"] |= DRAGGED not in layout.layer("xz")._displayed_box_ids()

    _drag_through((layout, ref), OUT_AND_BACK, after_move)
    _check(layout, ref, "after release")
    assert seen["gone"]
    assert sum(layout.rebuilds.values()) == 0, layout.rebuilds
    assert layout.added_or_removed["xz"] >= 2 and layout.added_or_removed["yz"] >= 2
    assert layout.added_or_removed["xy"] == 0
    assert sorted(xy.selected_data) == [list(xy._displayed_box_ids()).index(DRAGGED)]


def test_enter_leave_raw_arrays_with_pure_python_triangulation(make_napari_viewer,
                                                              pure_python_triangulation):
    """With a deterministic triangulation even the raw mesh arrays equal the rebuild."""
    layout, ref = _both(make_napari_viewer)
    pure_python_triangulation()
    for lay in (layout, ref):
        for layer in lay.layers:
            layer._sync_shapes_from_bboxes()
        lay.reset_counts()
    _check(layout, ref, "initial", raw_mesh=True)
    _drag_through((layout, ref), OUT_AND_BACK,
                  lambda step: _check(layout, ref, f"move {step}", raw_mesh=True))
    _check(layout, ref, "after release", raw_mesh=True)
    for idx in (0, 5):
        box = BOXES[idx]
        _run_store_moves((layout, ref), idx, [_off_slice(box, 1, Y), _off_slice(box, 2, X), box], f"box {idx}")
        _check(layout, ref, f"box {idx}", raw_mesh=True)
    assert sum(layout.rebuilds.values()) == 0
    assert layout.added_or_removed["xz"] >= 4 and layout.added_or_removed["yz"] >= 4


def test_undo_redo_after_enter_leave_drag(make_napari_viewer):
    layout, ref = _both(make_napari_viewer)
    before = layout.store.data.copy()
    _drag_through((layout, ref), _path((Z, 48, 48), (Z, 72, 30), 6))
    after = layout.store.data.copy()
    assert not np.array_equal(before, after)
    for step, expected in (("undo", before), ("redo", after), ("undo", before)):
        for lay in (layout, ref):
            assert getattr(lay.layer("xy"), step)()
        assert np.array_equal(layout.store.data, expected)
        _check(layout, ref, step)
    # and the next drag after undo is incremental again
    layout.reset_counts()
    _drag_through((layout, ref), OUT_AND_BACK)
    _check(layout, ref, "drag after undo")
    assert sum(layout.rebuilds.values()) == 0


def test_3d_view_waits_for_the_commit(make_napari_viewer):
    layout, ref = _both(make_napari_viewer)
    shown_3d = [np.array(d) for d in layout.layer("3d").data]

    def after_move(step):
        now = layout.layer("3d").data
        assert all(np.array_equal(a, b) for a, b in zip(now, shown_3d))

    _drag_through((layout, ref), OUT_AND_BACK[:9], after_move)
    _check(layout, ref, "after release")
    assert not np.array_equal(layout.layer("3d").data[DRAGGED], shown_3d[DRAGGED])


def test_selection_and_hover_reset_like_a_rebuild(make_napari_viewer):
    """A full rebuild clears the view's selection and hover; so does an enter or leave."""
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        xz = lay.layer("xz")
        xz.selected_data = {0, 1}
        xz._value = (1, None)
        xz._set_highlight(force=True)
    _check(layout, ref, "selected")
    _run_store_moves((layout, ref), 4, [_shifted(BOXES[4], dy=-30)], "enter")
    assert layout.added_or_removed["xz"] == 1
    assert layout.layer("xz").selected_data == set()
    for lay in (layout, ref):
        lay.layer("xz").selected_data = {2}
    _run_store_moves((layout, ref), DRAGGED, [_off_slice(BOXES[DRAGGED], 1, Y)], "leave selected")
    assert layout.added_or_removed["xz"] == 2 and layout.rebuilds["xz"] == 0


def test_view_emptied_and_refilled(make_napari_viewer):
    """1 -> 0 and 0 -> 1 shapes take the full rebuild; 1 -> 2 and 2 -> 1 do not."""
    boxes = np.array([[[5, 40, 40], [15, 56, 56]], [[5, 10, 10], [15, 20, 20]]], float)
    layout, ref = _both(make_napari_viewer, boxes)
    assert layout.layer("xz").nshapes == 1
    _run_store_moves((layout, ref), 0, [_off_slice(boxes[0], 1, Y), boxes[0]], "1-0-1")
    assert layout.added_or_removed["xz"] == 0 and layout.rebuilds["xz"] == 2
    layout.reset_counts()
    _run_store_moves((layout, ref), 1, [_shifted(boxes[1], dy=35), boxes[1]], "1-2-1")
    assert layout.added_or_removed["xz"] == 2 and layout.rebuilds["xz"] == 0


def test_move_to_front_in_the_view_keeps_the_full_rebuild(make_napari_viewer):
    """With napari's per-shape z-order the full rebuild re-assigns z by position; an
    enter or leave in that view keeps the rebuild, so the view still equals it."""
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        xz = lay.layer("xz")
        xz.selected_data = {1}
        xz.move_to_front()  # napari refreshes the layer: a full rebuild
    _check(layout, ref, "front")
    layout.reset_counts()
    _drag_through((layout, ref), OUT_AND_BACK, lambda step: _check(layout, ref, f"move {step}"))
    _check(layout, ref, "after release")
    assert layout.added_or_removed["xz"] == 0 and layout.rebuilds["xz"] == 2
    assert layout.added_or_removed["yz"] == 2 and layout.rebuilds["yz"] == 0


def test_move_to_front_in_the_editing_view(make_napari_viewer):
    """move_to_front in XY, then a drag: XY stays on the fast path with its z-order,
    the sub-views add and remove the box."""
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        xy = lay.layer("xy")
        xy.selected_data = {0}
        xy.move_to_front()
        assert len(set(xy._data_view._z_index.tolist())) > 1
    layout.reset_counts()
    _drag_through((layout, ref), OUT_AND_BACK, lambda step: _check(layout, ref, f"move {step}"))
    _check(layout, ref, "after release")
    assert sum(layout.rebuilds.values()) == 0
    assert layout.added_or_removed["xz"] >= 2


@pytest.mark.parametrize("style", ["colour_selected", "new_shape_colour", "edge_width", "features"])
def test_styled_shapes_keep_the_full_rebuild(make_napari_viewer, style):
    """Shapes with their own style: every enter/leave still equals the full rebuild."""
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        xz = lay.layer("xz")
        if style == "colour_selected":
            xz.selected_data = {0}
            xz.current_edge_color = "red"
            xz.selected_data = set()
        elif style == "new_shape_colour":
            xz.current_edge_color = "red"  # nothing selected: only new shapes are red
        elif style == "edge_width":
            xz.edge_width = [1.0] + [2.0] * (xz.nshapes - 1)
        else:
            xz.features = {"score": np.arange(xz.nshapes, dtype=float)}
    _check(layout, ref, "styled")
    layout.reset_counts()
    idx = DRAGGED
    _run_store_moves((layout, ref), idx, [_off_slice(BOXES[idx], 1, Y), BOXES[idx]], style)
    if style in ("new_shape_colour", "edge_width"):
        # new_shape_colour: the removal keeps the uniform colour, the entering shape
        # would be red. edge_width: the removal rebuilds, and the rebuild resets
        # every width to current_edge_width, so the entering shape is added.
        assert layout.added_or_removed["xz"] == 1 and layout.rebuilds["xz"] == 1
    else:
        assert layout.added_or_removed["xz"] == 0 and layout.rebuilds["xz"] == 2


def test_random_store_edits_match_rebuild(make_napari_viewer):
    rng = np.random.default_rng(11)
    size = np.array([6, 18, 18])
    mins = rng.uniform(0, np.array(SHAPE) - 1 - size, size=(25, 3)).round(1)
    boxes = np.stack([mins, mins + size], axis=1)
    # A filled face: with the default transparent one the thumbnails stay blank.
    layout, ref = _both(make_napari_viewer, boxes, face_color="#ff000040")
    for drag in range(8):
        idx = int(rng.integers(len(boxes)))
        start = layout.store.data[idx].astype(float)
        delta = np.round(rng.uniform(-25, 25, 3), 1)
        moves = [_valid(start + t * np.stack([delta, delta])) for t in np.linspace(0.2, 1.0, 5)]
        _run_store_moves((layout, ref), idx, moves, f"drag {drag}")
    assert sum(layout.rebuilds.values()) == 0
    assert sum(layout.added_or_removed.values()) > 5


# ------------------------------------------------ style gates, when the first crossing enters


@pytest.mark.parametrize("style", ["current_width", "current_face", "z_uniform_3", "widths_mixed"])
def test_style_change_then_an_enter_first(make_napari_viewer, style):
    """The view's style changes, and the first enter or leave afterwards is an enter
    (the tests above start with a leave, whose rebuild resets the style). The shape
    a rebuild would add does not share the view's style: the enter keeps the rebuild."""
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        xz = lay.layer("xz")
        if style == "current_width":
            xz.current_edge_width = 5.0  # nothing selected: only new shapes get it
        elif style == "current_face":
            xz.current_face_color = "#ff000080"
        elif style == "z_uniform_3":
            xz.z_index = [3] * xz.nshapes
        else:
            xz.edge_width = [1.0] + [2.0] * (xz.nshapes - 1)
    _check(layout, ref, style)
    layout.reset_counts()
    _update((layout, ref), 4, _shifted(BOXES[4], dy=-30))  # enters XZ at position 3
    _check(layout, ref, f"{style}: enter")
    assert layout.added_or_removed["xz"] == 0 and layout.rebuilds["xz"] == 1


def test_uniform_nonzero_z_index(make_napari_viewer):
    """All shapes at z-index 3: a leave stays in place, an enter keeps the rebuild (which
    gives z 0 to the last position, not to the entering box)."""
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        xz = lay.layer("xz")
        xz.z_index = [3] * xz.nshapes
    _check(layout, ref, "z set")
    layout.reset_counts()
    _update((layout, ref), 0, _off_slice(BOXES[0], 1, Y))
    _check(layout, ref, "leave first")
    assert layout.added_or_removed["xz"] == 1
    _update((layout, ref), 0, BOXES[0])
    _check(layout, ref, "enter first")
    assert layout.rebuilds["xz"] == 1
    _update((layout, ref), 5, _off_slice(BOXES[5], 1, Y))
    _check(layout, ref, "leave last")


def test_manual_text_encoding(make_napari_viewer):
    """Per-shape (manual) text has no ``constant``: enter and leave keep the rebuild."""
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        xz = lay.layer("xz")
        xz.text = {"string": {"array": [f"s{k}" for k in range(xz.nshapes)], "default": "new"},
                   "visible": True}
    _check(layout, ref, "text")
    layout.reset_counts()
    _update((layout, ref), DRAGGED, _off_slice(BOXES[DRAGGED], 1, Y))
    _check(layout, ref, "leave")
    _update((layout, ref), DRAGGED, BOXES[DRAGGED])
    _check(layout, ref, "enter")
    assert layout.added_or_removed["xz"] == 0 and layout.rebuilds["xz"] == 2


def test_current_face_colour_without_selection(make_napari_viewer):
    """New shapes would get another face colour: a leave stays in place, an enter keeps
    the rebuild (which gives the new colour to the last shape, not to the entering box)."""
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        lay.layer("xz").current_face_color = "#ff000080"
    _check(layout, ref, "face set")
    layout.reset_counts()
    _update((layout, ref), DRAGGED, _off_slice(BOXES[DRAGGED], 1, Y))
    _check(layout, ref, "leave")
    assert layout.added_or_removed["xz"] == 1
    _update((layout, ref), DRAGGED, BOXES[DRAGGED])
    _check(layout, ref, "enter")
    assert layout.added_or_removed["xz"] == 1 and layout.rebuilds["xz"] == 1
    # the rebuild gave the red face to the last shape: the colours are not uniform any more
    _update((layout, ref), DRAGGED, _off_slice(BOXES[DRAGGED], 1, Y))
    _check(layout, ref, "leave again")
    assert layout.rebuilds["xz"] == 2


# ------------------------------------------ thumbnail, hover and napari's interaction state


def test_thumbnail_after_enter_and_leave_with_a_filled_face(make_napari_viewer):
    """With the default transparent face the thumbnail stays blank, so no thumbnail
    comparison can fail. With a red face an enter or leave must change it, to the
    rebuild's thumbnail and to napari's own rasterization of what the view holds."""
    layout, ref = _both(make_napari_viewer, face_color="red")
    xz, xz_ref = layout.layer("xz"), ref.layer("xz")
    for where, box in (("leave", _off_slice(BOXES[DRAGGED], 1, Y)), ("enter", BOXES[DRAGGED])):
        before = np.array(xz.thumbnail, copy=True)
        layout.reset_counts()
        _update((layout, ref), DRAGGED, box)
        assert layout.added_or_removed["xz"] == 1 and layout.rebuilds["xz"] == 0
        shown = np.array(xz.thumbnail, copy=True)
        assert not np.array_equal(shown, before), f"{where}: thumbnail unchanged"
        assert np.array_equal(shown, xz_ref.thumbnail), f"{where}: thumbnail differs from the rebuild"
        Shapes._update_thumbnail(xz)  # napari's rasterization, without TurboBox's gate
        assert np.array_equal(shown, xz.thumbnail), f"{where}: thumbnail stale"
        _check(layout, ref, where)


# napari's hover, drag and rubber-band state of a Shapes layer.
INTERACTION = ("_value", "_moving_value", "_is_moving", "_is_selecting", "_drag_box", "_drag_start",
               "_fixed_vertex", "_last_cursor_position")


def _interaction(layer):
    return {name: repr(getattr(layer, name, None)) for name in INTERACTION}


@pytest.mark.parametrize("step", ["leave", "enter"])
def test_hover_then_another_box_crosses(make_napari_viewer, step):
    """A hovered shape (nothing selected), then another box leaves or enters the view:
    hover, napari's interaction state and the drawn highlight are reset like a rebuild."""
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        xz = lay.layer("xz")
        xz._value = (2, None)  # hover over box 3's shape
        xz._set_highlight(force=True)
    _check(layout, ref, "hovered")
    assert layout.layer("xz")._value == (2, None)
    layout.reset_counts()
    if step == "leave":
        _update((layout, ref), 0, _off_slice(BOXES[0], 1, Y))
    else:
        _update((layout, ref), 4, _shifted(BOXES[4], dy=-30))
    assert layout.added_or_removed["xz"] == 1
    assert _interaction(layout.layer("xz")) == _interaction(ref.layer("xz"))
    _check(layout, ref, step)


def test_rubber_band_in_progress_when_a_box_leaves(make_napari_viewer):
    layout, ref = _both(make_napari_viewer)
    for lay in (layout, ref):
        xz = lay.layer("xz")
        xz._is_selecting = True
        xz._drag_start = np.array([1.0, 2.0])
        xz._drag_box = np.array([[1.0, 2.0], [9.0, 60.0]])
        xz._set_highlight(force=True)
    _check(layout, ref, "band")
    _update((layout, ref), DRAGGED, _off_slice(BOXES[DRAGGED], 1, Y))
    assert layout.added_or_removed["xz"] == 1
    assert _interaction(layout.layer("xz")) == _interaction(ref.layer("xz"))
    _check(layout, ref, "left")


def test_read_only_view_stays_read_only(make_napari_viewer):
    """napari's layer controls set ``editable`` on creation and on every display change;
    a read-only view stays read-only through edits, enter/leave and a display toggle,
    and its select tool stays disabled. The editable views follow napari's rule."""
    layout = Layout(make_napari_viewer)
    yz, xz = layout.layer("yz"), layout.layer("xz")
    viewer = layout.viewers[VIEWS.index("yz")]

    def select_button_enabled():
        controls = viewer.window._qt_viewer.controls.widgets[yz]
        return controls.select_button.isEnabled()

    for step, box in (("created", None), ("in-slice edit", _shifted(BOXES[DRAGGED], dz=1)),
                      ("leave", _off_slice(BOXES[DRAGGED], 2, X)), ("enter", BOXES[DRAGGED])):
        if box is not None:
            layout.store.update_box(DRAGGED, box)
        assert not yz.editable and yz.mode == "pan_zoom", step
        assert not select_button_enabled(), step
        assert xz.editable, step
    assert layout.added_or_removed["yz"] == 2 and layout.rebuilds["yz"] == 0
    yz.editable = True
    assert not yz.editable
    viewer.dims.ndisplay = 3
    viewer.dims.ndisplay = 2
    assert not yz.editable and not select_button_enabled()
    xz_viewer = layout.viewers[VIEWS.index("xz")]
    xz_viewer.dims.ndisplay = 3
    assert not xz.editable
    xz_viewer.dims.ndisplay = 2
    assert xz.editable


# ------------------------------------------------ napari raises during an enter or leave


def _extend_then_raise(monkeypatch, dv):
    """napari appends the new shape's arrays, then fails before it appends the shape."""
    original = dv._extend_meshes

    def failing(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated napari failure after _extend_meshes")

    monkeypatch.setattr(dv, "_extend_meshes", failing)


def _add_rejects_a_list(monkeypatch, dv):
    def failing(shape, *args, **kwargs):
        raise TypeError("simulated napari change: add() takes no list")

    monkeypatch.setattr(dv, "add", failing)


def _move_then_raise(monkeypatch, dv):
    """The shape object is moved, then the move fails before the arrays follow."""
    def failing(dv_, pos):
        dv_.shapes.insert(pos, dv_.shapes.pop())
        raise RuntimeError("simulated failure in _move_last_shape")

    monkeypatch.setattr(layer_module, "_move_last_shape", failing)


def _remove_raises(monkeypatch, dv):
    def failing(index, *args, **kwargs):
        raise AttributeError("simulated napari change: ShapeList.remove is gone")

    monkeypatch.setattr(dv, "remove", failing)


def _remove_then_raise(monkeypatch, dv):
    """napari removes the shape, then fails: the colours are one row too long."""
    original = dv.remove

    def failing(index, *args, **kwargs):
        original(index, *args, **kwargs)
        raise RuntimeError("simulated napari failure after ShapeList.remove")

    monkeypatch.setattr(dv, "remove", failing)


FAILURES = {"extend": (_extend_then_raise, True), "add": (_add_rejects_a_list, True),
            "move": (_move_then_raise, True), "remove": (_remove_raises, False),
            "remove_then_raise": (_remove_then_raise, False)}


@pytest.mark.parametrize("failure", list(FAILURES))
def test_napari_failure_during_enter_or_leave(make_napari_viewer, monkeypatch, caplog, failure):
    """napari raises inside an enter or leave of the XZ view: the error reaches the
    store's log, and the next store update, an in-slice edit of another box, rebuilds the
    view, which then equals the reference (colours included). After that enter and leave
    are added and removed in place again."""
    layout, ref = _both(make_napari_viewer, face_color="#00ff0080")
    install, entering = FAILURES[failure]
    idx, box = (4, _shifted(BOXES[4], dy=-30)) if entering else (DRAGGED, _off_slice(BOXES[DRAGGED], 1, Y))
    with monkeypatch.context() as m, caplog.at_level(logging.ERROR):
        install(m, layout.layer("xz")._data_view)
        _update((layout, ref), idx, box)
    assert any("Error notifying listener" in r.getMessage() for r in caplog.records), failure
    assert layout.added_or_removed["xz"] == 0
    layout.reset_counts()
    _update((layout, ref), 0, _shifted(BOXES[0], dx=1))
    assert layout.rebuilds["xz"] == 1 and layout.added_or_removed["xz"] == 0
    _check(layout, ref, f"{failure}: next update")
    layout.reset_counts()
    _update((layout, ref), 5, _off_slice(BOXES[5], 1, Y))
    _update((layout, ref), 5, BOXES[5])
    _check(layout, ref, f"{failure}: leave and enter")
    assert layout.added_or_removed["xz"] == 2 and layout.rebuilds["xz"] == 0


def _twin_2d(make_napari_viewer, ndisplay, reference):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((64, 64), np.uint8))
    layer = TurboBoxLayer(ndim=2, image_shape=(64, 64))
    viewer.add_layer(layer)
    layer.add_boxes([[[5, 5], [20, 30]], [[30, 30], [50, 60]], [[10, 40], [25, 55]]])
    viewer.dims.ndisplay = ndisplay
    calls = {"n": 0}
    original = layer._add_or_remove_shape

    def counted(*args):
        calls["n"] += 1
        return False if reference else original(*args)

    layer._add_or_remove_shape = counted
    return viewer, layer, calls


@pytest.mark.parametrize("ndisplay", [2, 3])
def test_2d_layer_never_adds_or_removes(make_napari_viewer, ndisplay):
    """A 2D layer has no hidden axis: every box is always visible (also in a 3D viewer)."""
    if ndisplay == 3 and Version(napari.__version__) < Version("0.7"):
        pytest.skip("napari < 0.7 cannot draw a 2D Shapes layer in a 3D viewer")
    viewer, layer, calls = _twin_2d(make_napari_viewer, ndisplay, reference=False)
    ref_viewer, ref, _ = _twin_2d(make_napari_viewer, ndisplay, reference=True)
    for lay in (layer, ref):
        lay.store.begin_edit_session()
    for step, box in enumerate(([[6, 7], [22, 33]], [[40, 45], [60, 62]], [[1, 1], [2, 2]])):
        for lay in (layer, ref):
            lay.store.update_box(1, np.asarray(box, float))
        _assert_same(_state(viewer, layer), _state(ref_viewer, ref), f"step {step}")
        _assert_matches_store(layer, f"step {step}")
    for lay in (layer, ref):
        lay.store.end_edit_session()
    _assert_same(_state(viewer, layer), _state(ref_viewer, ref), "end")
    assert calls["n"] == 0


def test_one_upload_per_enter_or_leave_and_store_untouched(make_napari_viewer):
    layout = Layout(make_napari_viewer)
    uploads = []
    layout.layer("xz").events.set_data.connect(lambda event: uploads.append(1))
    before = layout.store.data.copy()
    box = BOXES[DRAGGED]
    layout.store.update_box(DRAGGED, _off_slice(box, 1, Y))  # leaves XZ
    assert layout.added_or_removed["xz"] == 1 and len(uploads) == 1
    layout.store.update_box(DRAGGED, box)  # enters again
    assert layout.added_or_removed["xz"] == 2 and len(uploads) == 2
    assert layout.rebuilds["xz"] == 0
    assert np.array_equal(layout.store.data, before)


def test_layer_without_image_shape_keeps_the_full_rebuild(make_napari_viewer):
    """Without image_shape the layer's extent follows the shapes shown, and the viewer
    re-reads it only on the data events of a rebuild: an enter or leave keeps the
    rebuild, so the extent and the viewer's dims range stay those of a rebuild."""
    views = []
    for reference in (False, True):
        viewer = make_napari_viewer()
        store = BBoxDataStore(initial_data=BOXES.copy())
        layer = TurboBoxLayer(ndim=3, bbox_data_store=store)
        viewer.add_layer(layer)
        viewer.dims.order = (1, 0, 2)
        viewer.dims.set_point(1, Y)
        calls = {"added_or_removed": 0}

        def counted(*args, _original=layer._add_or_remove_shape, _calls=calls, _reference=reference):
            done = False if _reference else _original(*args)
            _calls["added_or_removed"] += int(done)
            return done

        layer._add_or_remove_shape = counted
        views.append((viewer, layer, store, calls))
    (viewer, layer, _, calls), (ref_viewer, ref, _, _) = views
    idx = 3  # the box that reaches furthest in x in the XZ view
    for step, box in enumerate((_off_slice(BOXES[idx], 1, Y), BOXES[idx])):
        for _, _, store, _ in views:
            store.update_box(idx, box)
        assert layer.nshapes == ref.nshapes, step
        assert np.array_equal(layer.extent.data, ref.extent.data), step
        assert viewer.dims.range == ref_viewer.dims.range, step
    assert calls["added_or_removed"] == 0


def _uneven_polygons():
    """Polygons with 3 to 8 corners, so the blocks of every shape have another size."""
    rng = np.random.default_rng(2)
    shapes = []
    for k, n in enumerate((5, 3, 8, 4, 6, 7)):
        angles = np.sort(rng.uniform(0, 2 * np.pi, n))
        points = np.stack([10 + 20 * k + 5 * np.sin(angles), 10 + 5 * np.cos(angles)], axis=1)
        shapes.append(Polygon(points, edge_width=1.0 + k, z_index=0))
    return shapes


def _napari_shape_list(shapes):
    """napari's own ShapeList with the shapes appended one by one (colours per shape)."""
    dv = ShapeList(ndisplay=2)
    for shape in shapes:
        k = shape.edge_width
        dv.add(shape, face_color=np.array([k / 10, 0, 0, 1.0]), edge_color=np.array([0, k / 10, 0, 1.0]))
    return dv


@pytest.mark.parametrize("pos", range(6))
def test_move_last_shape_on_uneven_blocks(pos):
    """``_move_last_shape`` gives the shape list napari builds with the shapes in that
    order. Box rectangles all have blocks of one size, where a wrong shift of the
    block starts would not show; here every block has another size."""
    shapes = _uneven_polygons()
    dv = _napari_shape_list(shapes)
    with dv.batched_updates():
        _move_last_shape(dv, pos)
        dv._update_z_order()
    order = [*shapes[:pos], shapes[-1], *shapes[pos:-1]]
    expected = _napari_shape_list(order)
    assert dv.shapes == order
    # The same Shape objects, so even the triangulation is identical: all arrays exact.
    for owner, names in ((lambda d: d, LIST_ARRAYS), (lambda d: d._mesh, MESH_ARRAYS + MESH_GEOMETRY)):
        for name in names:
            live, ref = getattr(owner(dv), name), getattr(owner(expected), name)
            assert live.dtype == ref.dtype and np.array_equal(live, ref), f"pos {pos}: {name}"


# ------------------------------------------------------------ fast path (_edit_shape)


def _single_view(make_napari_viewer, boxes=BOXES):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros(SHAPE, np.uint8))
    viewer.dims.set_point(0, Z)
    store = BBoxDataStore(initial_data=np.asarray(boxes, float))
    layer = TurboBoxLayer(ndim=3, image_shape=SHAPE, bbox_data_store=store)
    viewer.add_layer(layer)
    return viewer, layer


def test_fast_edit_with_napari_z_order_matches_forced_update(make_napari_viewer):
    """Non-uniform z_index and move_to_front, then fast-path edits: the displayed
    triangles (in z-order) equal a forced ``_update_displayed``."""
    _, layer = _single_view(make_napari_viewer)
    n = layer.nshapes
    layer.z_index = [3, -1, 0, 2][:n] + [0] * max(0, n - 4)
    layer.selected_data = {1}
    layer.move_to_front()
    dv = layer._data_view
    assert not np.array_equal(dv._z_order, np.arange(n))
    fallbacks = {"n": 0}
    original = ShapeList.edit

    def napari_edit(self, *args, **kwargs):
        fallbacks["n"] += 1
        return original(self, *args, **kwargs)

    ShapeList.edit = napari_edit
    try:
        layer.store.begin_edit_session()
        for step, (idx, dy, dx) in enumerate(((DRAGGED, 3, -2), (0, -4, 5), (3, 2, 2), (DRAGGED, -1, 6))):
            box = layer.store.data[idx].astype(float) + np.array([0, dy, dx])
            layer.store.update_box(idx, box)
            _assert_consistent(layer, f"step {step}")
            forced = copy.deepcopy(dv)
            with forced.batched_updates():
                forced._update_displayed()
            assert np.array_equal(dv._mesh.displayed_triangles, forced._mesh.displayed_triangles)
            _assert_matches_store(layer, f"step {step}")
        layer.store.end_edit_session()
    finally:
        ShapeList.edit = original
    assert fallbacks["n"] == 0  # all four edits took the fast path


@pytest.mark.parametrize("backend", ["pure_python", "default"])
@pytest.mark.parametrize("axis", [1, 2])
def test_one_pixel_box_edit_matches_napari_edit(make_napari_viewer, pure_python_triangulation,
                                                axis, backend):
    """Editing a box down to a one-pixel extent (another triangulation) and back:
    the raw shape-list arrays equal napari's own ShapeList.edit of the same shape
    (the triangulated mesh exactly under the pure-Python backend, else as drawn)."""
    _, layer = _single_view(make_napari_viewer)
    if backend == "pure_python":
        pure_python_triangulation()
        layer._sync_shapes_from_bboxes()
    dv = layer._data_view
    pos = layer._bbox_index_to_path_index(DRAGGED)
    start = layer.store.data[DRAGGED].astype(float)
    thin = start.copy()
    thin[1, axis] = thin[0, axis] + 1
    wide = start.copy()
    wide[1, axis] += 3
    layer.store.begin_edit_session()
    for step, box in enumerate((thin, start, thin, wide, start)):
        reference = copy.deepcopy(dv)
        path = slice_paths_from_bbox(box, tuple(layer._slice_input.displayed), layer._slice_point_data())[0]
        reference.edit(pos, np.asarray(path, float))
        layer.store.update_box(DRAGGED, box)
        for name in LIST_ARRAYS:
            live, ref = getattr(dv, name), getattr(reference, name)
            assert live.dtype == ref.dtype and np.array_equal(live, ref), f"{step}: {name}"
        exact = MESH_ARRAYS + (MESH_GEOMETRY if backend == "pure_python" else ())
        for name in exact:
            live, ref = getattr(dv._mesh, name), getattr(reference._mesh, name)
            assert live.dtype == ref.dtype and np.array_equal(live, ref), f"{step}: mesh.{name}"
        for name in MESH_GEOMETRY:
            assert getattr(dv._mesh, name).shape == getattr(reference._mesh, name).shape, name
        for triangles, colors in (("triangles", "triangles_colors"),
                                  ("displayed_triangles", "displayed_triangles_colors")):
            live = _triangle_set(dv._mesh.vertices, getattr(dv._mesh, triangles), getattr(dv._mesh, colors))
            ref = _triangle_set(reference._mesh.vertices, getattr(reference._mesh, triangles),
                                getattr(reference._mesh, colors))
            assert np.array_equal(live, ref), f"{step}: {triangles} draw other triangles"
    layer.store.end_edit_session()


def test_fast_edit_inside_batched_updates_keeps_pending_update(make_napari_viewer):
    """A fast-path edit inside napari's batched_updates, after a slice change there,
    must not drop the display update the slice change scheduled."""
    _, layer = _single_view(make_napari_viewer)
    dv = layer._data_view
    pos = layer._bbox_index_to_path_index(DRAGGED)
    box = layer.store.data[DRAGGED].astype(float) + np.array([0, 2, 3])
    path = slice_paths_from_bbox(box, tuple(layer._slice_input.displayed), layer._slice_point_data())[0]
    assert dv._displayed.all()
    with dv.batched_updates():
        dv.slice_key = np.array([Z + 5.0])  # no shape is on that slice
        layer._edit_shape(pos, np.asarray(path, float))
    assert not dv._displayed.any()
    assert len(dv.displayed_vertices) == 0 and len(dv._mesh.displayed_triangles) == 0
    _assert_consistent(layer, "after batch")


def test_fast_edit_does_not_force_update_displayed(make_napari_viewer):
    """The fast path patches its rows; napari's O(N) _update_displayed never runs."""
    _, layer = _single_view(make_napari_viewer)
    calls = {"forced": 0}
    original = ShapeList._update_displayed

    def counting(self):
        if self._ShapeList__batch_force_call:
            calls["forced"] += 1
        return original(self)

    ShapeList._update_displayed = counting
    try:
        layer.store.begin_edit_session()
        for dy in (1, 2, 3):
            box = layer.store.data[DRAGGED].astype(float) + np.array([0, dy, 0])
            layer.store.update_box(DRAGGED, box)
        layer.store.end_edit_session()
    finally:
        ShapeList._update_displayed = original
    assert calls["forced"] == 0
    assert layer._data_view._ShapeList__update_displayed_called == 0
    _assert_consistent(layer, "after edits")
    _assert_matches_store(layer, "after edits")


@pytest.mark.parametrize("view", ["xy", "3d"])
def test_napari_failure_during_an_in_place_edit(make_napari_viewer, monkeypatch, caplog, view):
    """napari raises inside an in-place edit, after the shape took its new vertices and
    before its mesh vertices did. The error reaches the store's log; the next store
    update, an in-slice edit of another box, rebuilds that view, and every view then
    equals a fresh rebuild. After that edits are applied in place again."""
    layout, fresh = Layout(make_napari_viewer), Layout(make_napari_viewer)

    def update(idx, box):
        _update((layout, fresh), idx, box)
        for layer in fresh.layers:
            layer._sync_shapes_from_bboxes()

    def failing(*args, **kwargs):
        raise RuntimeError("simulated napari failure before the mesh vertices are updated")

    with monkeypatch.context() as m, caplog.at_level(logging.ERROR):
        m.setattr(layout.layer(view)._data_view, "_update_mesh_vertices", failing)
        update(DRAGGED, _shifted(BOXES[DRAGGED], dx=1))  # in-slice in every view
    assert any("Error notifying listener" in r.getMessage() for r in caplog.records), view
    assert sum(layout.rebuilds.values()) == 0
    update(0, _shifted(BOXES[0], dx=1))
    _check(layout, fresh, "next update")
    assert layout.rebuilds == {**dict.fromkeys(VIEWS, 0), view: 1}, layout.rebuilds
    layout.reset_counts()
    update(DRAGGED, _shifted(BOXES[DRAGGED], dx=-1))
    update(0, _shifted(BOXES[0], dx=2))
    _check(layout, fresh, "edits after the rebuild")
    assert sum(layout.rebuilds.values()) == 0, layout.rebuilds


def _zero_extent(box, axis):
    box = np.asarray(box, float).copy()
    box[1, axis] = box[0, axis]
    return box


def _zero_extent_nan(backend):
    """Views whose napari mesh may hold NaN once a box has a zero extent (``nan_ok``).

    In 3D napari meshes the wireframe as a tube with vispy's ``_frenet_frames``: the
    path of a flat box runs to a corner and straight back, and the zero tangent there
    gives NaN offsets. In 2D every triangulation backend except the pure-Python one
    gives NaN offsets to the edge of a flat rectangle. Boxes with extent >= 1 give
    finite meshes, and all other tests require that.
    """
    views = ("3d",) if backend == "pure_python" else VIEWS
    return dict.fromkeys(views, ("mesh.vertices", "mesh.vertices_offsets"))


@pytest.mark.parametrize("backend", ["default", "pure_python"])
def test_zero_extent_edits_rebuild_the_view(make_napari_viewer, pure_python_triangulation, backend):
    """Store writes are not validated, so a box can get a zero extent, and its shape other
    vertex and triangle counts. ShapeList.edit keeps (pads) or moves the shape's rows for
    that, and napari < 0.7 then mis-colours a shrunk shape and shifts the triangles behind
    a grown one; so such an edit rebuilds the view. After every step every view equals a
    fresh rebuild, and boxes keep entering and leaving in place, also next to and as
    zero-extent boxes."""
    layout, fresh = Layout(make_napari_viewer), Layout(make_napari_viewer)
    if backend == "pure_python":
        pure_python_triangulation()
        for lay in (layout, fresh):
            for layer in lay.layers:
                layer._sync_shapes_from_bboxes()
    raw = backend == "pure_python"
    layout.reset_counts()
    _check(layout, fresh, "initial", raw_mesh=raw)  # no zero-extent box yet: all finite
    current = {i: np.asarray(b, float) for i, b in enumerate(BOXES)}
    current[3], current[5] = _zero_extent(BOXES[3], 2), _zero_extent(BOXES[5], 0)
    steps = [("box 3 becomes a line", 3, current[3]), ("box 5 becomes a line", 5, current[5])]
    for i in (0, DRAGGED, 5, 3, 0):
        steps += [(f"box {i} leaves", i, _off_slice(current[i], 1, Y)), (f"box {i} enters", i, current[i])]
    steps += [("a line enters", 4, _zero_extent(_shifted(BOXES[4], dy=-30), 0)),
              ("box 3 grows back", 3, BOXES[3]), ("box 4 leaves", 4, BOXES[4])]
    for where, idx, box in steps:
        _update((layout, fresh), idx, box)
        for layer in fresh.layers:
            layer._sync_shapes_from_bboxes()
        _check(layout, fresh, where, raw_mesh=raw, nan_ok=_zero_extent_nan(backend))
    # three edits changed a shape's counts in XZ (two shrink, one grows back)
    assert layout.rebuilds["xz"] == 3 and layout.added_or_removed["xz"] == 12, (
        layout.rebuilds, layout.added_or_removed)
