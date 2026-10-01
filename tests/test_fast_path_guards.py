"""Guards of the single-shape fast path (``TurboBoxLayer._edit_shape``).

The fast path updates one shape's vertices in napari's shape list and skips
napari's recomputation of the displayed arrays. These tests pin the cases in
which skipping it would be wrong: a non-uniform z-order, a box edited down to one
pixel (changed triangulation), and an edit inside an outer ``batched_updates``
that already scheduled a recomputation. Each compares the live arrays with a copy
of the shape list on which napari's own recomputation was forced. The last test
counts napari's recomputations when a box enters or leaves the slice
(``TurboBoxLayer._add_or_remove_shape``).
"""

import copy

import numpy as np
import pytest
from napari.layers.shapes._shape_list import ShapeList

from napari_turbobox import BBoxDataStore, TurboBoxLayer

IMAGE_SHAPE = (24, 96, 96)
BOXES = np.array([
    [[5, 10, 10], [15, 30, 30]],
    [[5, 40, 10], [15, 60, 30]],
    [[5, 10, 40], [15, 30, 60]],
    [[5, 40, 40], [15, 60, 60]],
], float)
DISPLAYED = (("dv", "_displayed"), ("dv", "displayed_vertices"), ("dv", "displayed_vertices_to_shape_num"),
             ("dv", "displayed_index"), ("dv", "_z_order"), ("mesh", "triangles_z_order"),
             ("mesh", "displayed_triangles"), ("mesh", "displayed_triangles_colors"),
             ("mesh", "displayed_triangles_to_shape_index"))


@pytest.fixture
def view(make_napari_viewer):
    def make(boxes=BOXES, edge_width=2.0):
        viewer = make_napari_viewer()
        viewer.add_image(np.zeros(IMAGE_SHAPE, np.uint8))
        store = BBoxDataStore(initial_data=np.array(boxes, float))
        layer = TurboBoxLayer(ndim=3, image_shape=IMAGE_SHAPE, bbox_data_store=store, edge_width=edge_width)
        viewer.add_layer(layer)
        viewer.dims.set_point(0, 10)
        return layer
    return make


def _forced(dv):
    c = copy.deepcopy(dv)
    with c.batched_updates():
        c._update_z_order()
    return c


def _assert_raw_equal_forced(dv, where=""):
    forced = _forced(dv)
    for owner, name in DISPLAYED:
        a = getattr(dv if owner == "dv" else dv._mesh, name)
        b = getattr(forced if owner == "dv" else forced._mesh, name)
        assert np.array_equal(a, b), f"{owner}.{name} differs from napari's recomputation {where}"


@pytest.fixture
def napari_calls(monkeypatch):
    """Count ShapeList.edit calls and executed (not merely scheduled) _update_displayed."""
    calls = {"edit": 0, "update_displayed": 0, "update_z_order": 0}
    edit, update_displayed, update_z_order = (ShapeList.edit, ShapeList._update_displayed,
                                              ShapeList._update_z_order)

    def counted_edit(self, *a, **k):
        calls["edit"] += 1
        return edit(self, *a, **k)

    def counted_update_displayed(self):
        calls["update_displayed"] += int(bool(self._ShapeList__batch_force_call))
        return update_displayed(self)

    def counted_update_z_order(self):
        calls["update_z_order"] += 1
        return update_z_order(self)

    monkeypatch.setattr(ShapeList, "edit", counted_edit)
    monkeypatch.setattr(ShapeList, "_update_displayed", counted_update_displayed)
    monkeypatch.setattr(ShapeList, "_update_z_order", counted_update_z_order)
    return calls


def test_non_uniform_z_order_then_edit(view, napari_calls):
    layer = view()
    layer.z_index = [3, -1, 0, 2]
    layer.selected_data = {1}
    layer.move_to_front()
    _assert_raw_equal_forced(layer._data_view, "before the edit")
    napari_calls.update(edit=0, update_displayed=0)
    store = layer.store
    store.begin_edit_session()
    for k in range(4):
        store.update_box(0, BOXES[0] + [[0, k + 1, k + 2], [0, k + 1, k + 2]])
        _assert_raw_equal_forced(layer._data_view, f"after edit {k}")
    store.end_edit_session()
    assert napari_calls["edit"] == 0, "the fast path fell back to ShapeList.edit"


@pytest.mark.parametrize("edge_width", [0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("target", [
    [[5, 29.5, 90], [15, 55.75, 91]],  # 1 px wide in x
    [[5, 29.5, 85], [15, 30.5, 95]],   # 1 px tall in y
    [[5, 29.5, 90], [15, 30.5, 91]],   # 1 x 1
    [[5, 29.5, 88], [15, 45.5, 92]],   # control
])
def test_one_pixel_edit_raw_arrays(view, edge_width, target):
    layer = view(boxes=[[[0, 24.75, 74.25], [22, 40.75, 76.25]]], edge_width=edge_width)
    store = layer.store
    store.begin_edit_session()
    store.update_box(0, np.array(target, float))
    store.end_edit_session()
    dv = layer._data_view
    _assert_raw_equal_forced(dv)
    fresh = view(boxes=np.array(store.data), edge_width=edge_width)
    for owner, name in (("dv", "_vertices"), ("dv", "_vertices_index"), ("mesh", "vertices_index"),
                        ("mesh", "triangles_index"), ("mesh", "triangles_colors")):
        a = getattr(dv if owner == "dv" else dv._mesh, name)
        b = getattr(fresh._data_view if owner == "dv" else fresh._data_view._mesh, name)
        assert np.array_equal(a, b), f"{owner}.{name} differs from a fresh layer"


def test_edit_inside_batched_updates_keeps_pending_update(view, napari_calls):
    layer = view()
    dv = layer._data_view
    key = np.array(dv.slice_key, float)
    path = np.asarray(layer.data[0], float) + np.array([0.0, 1.0, 1.0])
    napari_calls.update(edit=0, update_displayed=0)
    with dv.batched_updates():
        dv.slice_key = key + 5  # off every shape: schedules a recomputation
        layer._edit_shape(0, path)
        assert napari_calls["update_displayed"] == 0
    assert napari_calls["edit"] == 0, "the fast path fell back to ShapeList.edit"
    assert napari_calls["update_displayed"] == 1, "the scheduled recomputation was dropped"
    assert not dv._displayed.any()
    _assert_raw_equal_forced(dv, "after the batch")
    with dv.batched_updates():
        dv.slice_key = key
    assert dv._displayed.all()
    _assert_raw_equal_forced(dv, "back on the slice")


def test_fast_path_does_not_recompute_displayed(view, napari_calls):
    layer = view()
    napari_calls.update(edit=0, update_displayed=0, update_z_order=0)
    store = layer.store
    store.begin_edit_session()
    for k in range(5):
        store.update_box(2, BOXES[2] + [[0, k, -k], [0, k, -k]])
    store.end_edit_session()
    assert napari_calls == {"edit": 0, "update_displayed": 0, "update_z_order": 0}
    _assert_raw_equal_forced(layer._data_view)


def test_enter_leave_recomputes_displayed_once(view, napari_calls):
    """Adding or removing a shape recomputes napari's displayed arrays once, derives
    the z-order once (napari's own ``_update_z_order``) and rebuilds nothing."""
    layer = view()
    rebuilds = []
    rebuild = layer._sync_shapes_from_bboxes

    def counted_rebuild():
        rebuilds.append(1)
        rebuild()

    layer._sync_shapes_from_bboxes = counted_rebuild
    # z 15..23 leaves the slice z=10; then back, at position 1 again
    for where, box in (("leave", BOXES[1] + [[10, 0, 0], [8, 0, 0]]), ("enter", BOXES[1])):
        napari_calls.update(edit=0, update_displayed=0, update_z_order=0)
        layer.store.update_box(1, box)
        assert napari_calls == {"edit": 0, "update_displayed": 1, "update_z_order": 1}, where
        _assert_raw_equal_forced(layer._data_view, f"after {where}")
    assert not rebuilds
    assert list(layer._shape_box_ids) == [0, 1, 2, 3]
