"""Regression tests for spatial-index correctness on the production sync path.

Before the writable-copy fix, ``SpatialIndex._boxes`` aliased the store's
read-only view. ``BBoxDataStore.update_box`` mutates that buffer in place
before notifying, so the incremental ``spatial_index.update_box`` in
``_perform_sync`` (a) raised ``ValueError: assignment destination is
read-only`` (swallowed by the listener wrapper) and (b) left the index stale
-- ``query_point`` returned a moved box at its OLD location and not its new
one, breaking hit-testing on synced viewers.
"""

from unittest.mock import MagicMock, Mock

import numpy as np

from napari_turbobox import BBoxDataStore
from napari_turbobox.layer import TurboBoxLayer
from napari_turbobox.spatial_index import SpatialIndex


def _assert_index_matches_rebuild(layer):
    """The live index must agree with a fresh rebuild at every box centre."""
    boxes = np.asarray(layer.bounding_boxes)
    fresh = SpatialIndex(boxes)
    for i, box in enumerate(boxes):
        centre = (box[0] + box[1]) / 2.0
        assert set(layer._spatial_index.query_point(centre)) == set(
            fresh.query_point(centre)
        ), f"live index disagrees with rebuild at box {i}"


def test_synced_layer_index_owns_writable_copy():
    store = BBoxDataStore(initial_data=np.array([[[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]]))
    layer = TurboBoxLayer(bbox_data_store=store, interactive=False, image_shape=(50, 200, 200))
    # The index must NOT alias the store's read-only view.
    assert layer._spatial_index._boxes.flags.writeable
    assert layer._spatial_index._boxes.base is not store.data.base


def test_synced_layer_index_not_stale_after_store_update_box():
    store = BBoxDataStore(initial_data=np.array([[[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]]))
    layer = TurboBoxLayer(bbox_data_store=store, interactive=False, image_shape=(50, 200, 200))

    # Move the box far away through the production cross-viewer sync API.
    store.update_box(0, np.array([[30.0, 150.0, 150.0], [40.0, 160.0, 160.0]]))

    # Index must reflect the NEW location and not the old one.
    assert layer._spatial_index.query_point(np.array([35.0, 155.0, 155.0])) == [0]
    assert layer._spatial_index.query_point(np.array([15.0, 15.0, 15.0])) == []
    _assert_index_matches_rebuild(layer)


def _make_drag_ready_layer(store):
    from napari.layers.shapes._shape_list import ShapeList

    layer = TurboBoxLayer(
        ndim=3, image_shape=(10, 100, 100), bbox_data_store=store, sync_mode="live"
    )
    slice_input = MagicMock()
    slice_input.ndisplay = 2
    slice_input.displayed = [1, 2]
    slice_input.not_displayed = [0]
    slice_input.point = np.array([4.0, 49.0, 49.0])
    world_slice = MagicMock()
    world_slice.point = np.array([4.0, 49.0, 49.0])
    slice_input.world_slice = world_slice
    slice_input.data_slice = MagicMock(return_value=MagicMock(point=np.array([4.0, 49.0, 49.0])))
    layer._slice_input = slice_input
    layer._data_view = ShapeList(ndisplay=2)
    layer._ndisplay_stored = 2
    layer._set_view_slice = lambda: None
    layer._update_dims = lambda: None
    return layer


def test_drag_release_keeps_index_consistent():
    """After a real drag-release, the (incrementally updated) index must agree
    with a full rebuild -- locks in the layer.py:1154 incremental update."""
    store = BBoxDataStore()
    layer = _make_drag_ready_layer(store)
    layer.add_boxes(
        np.array(
            [[[2, 20, 20], [5, 50, 50]], [[3, 70, 70], [6, 90, 90]]],
            dtype=float,
        )
    )

    event = Mock()
    event.type = "mouse_press"
    event.position = np.array([3.5, 35.0, 35.0])  # inside box 0
    event.modifiers = []
    event.dims_displayed = [0, 1, 2]
    event.handled = False

    gen = layer._on_mouse_drag(layer, event)
    next(gen)
    assert layer._drag_state is not None

    for new_pos in ([3.5, 41.0, 41.0], [3.5, 45.0, 45.0]):
        event.type = "mouse_move"
        event.position = np.array(new_pos)
        next(gen)

    event.type = "mouse_release"
    try:
        next(gen)
    except StopIteration:
        pass

    # The dragged box moved; the index must not be stale.
    _assert_index_matches_rebuild(layer)
