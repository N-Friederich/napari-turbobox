"""Tests for per-instance ``sync_mode`` on ``TurboBoxLayer``.

Covers Commit 2: the ``sync_mode`` constructor kwarg, the runtime
getter/setter property, ndisplay-based assignment in
``create_synchronized_bbox_layers``, and the edit-session lifecycle wired
into the mouse-drag generator.
"""

from unittest.mock import MagicMock, Mock

import numpy as np

from napari_turbobox import BBoxDataStore
from napari_turbobox.layer import TurboBoxLayer


def test_layer_default_sync_mode_is_live():
    """A layer created with no explicit sync_mode subscribes as 'live'."""
    store = BBoxDataStore()
    layer = TurboBoxLayer(ndim=3, bbox_data_store=store)

    assert layer.sync_mode == "live"
    assert len(store._listeners) == 1
    assert store._listeners[0].sync_mode == "live"


def test_layer_explicit_sync_mode_on_commit():
    """Passing sync_mode='on_commit' registers the listener as 'on_commit'."""
    store = BBoxDataStore()
    layer = TurboBoxLayer(ndim=3, bbox_data_store=store, sync_mode="on_commit")

    assert layer.sync_mode == "on_commit"
    assert len(store._listeners) == 1
    assert store._listeners[0].sync_mode == "on_commit"


def test_layer_sync_mode_property_getter_setter():
    """The sync_mode property updates the store-side listener handle."""
    store = BBoxDataStore()
    layer = TurboBoxLayer(ndim=3, bbox_data_store=store, sync_mode="live")
    handle = store._listeners[0]
    assert handle.sync_mode == "live"

    layer.sync_mode = "on_commit"
    assert layer.sync_mode == "on_commit"
    assert handle.sync_mode == "on_commit"

    layer.sync_mode = "live"
    assert layer.sync_mode == "live"
    assert handle.sync_mode == "live"


def test_layer_follows_its_own_store_and_sync_mode_setter_applies():
    """A layer that owns its store also listens to it (edits via layer.store show up),
    and changing sync_mode updates that subscription."""
    layer = TurboBoxLayer(ndim=3)  # no shared store -> owns its own
    assert layer._owns_store is True
    assert len(layer._bbox_store._listeners) == 1

    layer.sync_mode = "on_commit"  # must not raise
    assert layer.sync_mode == "on_commit"
    assert layer._bbox_store._listeners[0].sync_mode == "on_commit"


def test_create_synchronized_bbox_layers_assigns_on_commit_to_3d_viewer(make_napari_viewer):
    """create_synchronized_bbox_layers picks sync_mode from each viewer's ndisplay."""
    from napari_turbobox.multi_view import create_synchronized_bbox_layers

    main_viewer = make_napari_viewer()
    sub_viewer = make_napari_viewer()

    # Establish a 3D coordinate space in each viewer so ndisplay=3 is allowed.
    main_viewer.add_image(np.zeros((10, 64, 64)))
    sub_viewer.add_image(np.zeros((10, 64, 64)))
    sub_viewer.dims.ndisplay = 3  # sub-viewer renders in 3D
    assert main_viewer.dims.ndisplay == 2
    assert sub_viewer.dims.ndisplay == 3

    layers = create_synchronized_bbox_layers(
        main_viewer=main_viewer,
        sub_viewers=[sub_viewer],
        image_shape=(10, 64, 64),
    )

    assert layers[0].sync_mode == "live", "2D main viewer layer should be live"
    assert layers[1].sync_mode == "on_commit", "3D sub viewer layer should be on_commit"


def _make_drag_ready_layer(store):
    """Build a TurboBoxLayer on a shared store, wired for headless drag tests.

    Mirrors the ``bbox_layer`` fixture in test_interactions.py: a mocked 2D
    ``_slice_input`` plus patched view-slice/dims hooks so napari's rendering
    chain does not need a real canvas.
    """
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
    data_slice_result = MagicMock()
    data_slice_result.point = np.array([4.0, 49.0, 49.0])
    slice_input.data_slice = MagicMock(return_value=data_slice_result)
    layer._slice_input = slice_input

    layer._data_view = ShapeList(ndisplay=2)
    layer._ndisplay_stored = 2

    def patched_set_view_slice():
        if hasattr(layer, "_sync_guard") and hasattr(layer, "_bbox_store"):
            if not layer._sync_guard and layer._bbox_store.data.size > 0:
                layer._sync_shapes_from_bboxes()

    layer._set_view_slice = patched_set_view_slice
    layer._update_dims = lambda: None
    return layer


def test_mouse_drag_session_lifecycle():
    """A mouse drag opens an edit session on press and flushes once on release.

    An on_commit listener must see zero calls during press + moves and
    exactly one call after release, carrying the dragged box index.
    """
    store = BBoxDataStore()
    layer = _make_drag_ready_layer(store)
    layer.add_boxes(np.array([[[2, 20, 20], [5, 50, 50]]], dtype=float))

    # Register the deferred (3D-sub-viewer-like) listener AFTER the box exists
    # so the add_boxes notification is not counted.
    calls = []

    def on_commit_listener(data, changed_idx=None):
        calls.append(changed_idx)

    store.register_listener(on_commit_listener, sync_mode="on_commit")

    # --- mouse press inside box 0 ---
    event = Mock()
    event.type = "mouse_press"
    event.position = np.array([3.5, 35.0, 35.0])
    event.modifiers = []
    event.dims_displayed = [0, 1, 2]
    event.handled = False

    gen = layer._on_mouse_drag(layer, event)
    next(gen)  # run press branch up to first yield

    assert layer._drag_state is not None, "drag should have started"
    assert layer._session_active is True, "edit session should be open on press"
    assert calls == [], "no notification expected on press"

    # --- two mouse moves ---
    for new_pos in ([3.5, 38.0, 38.0], [3.5, 40.0, 40.0]):
        event.type = "mouse_move"
        event.position = np.array(new_pos)
        next(gen)
        assert calls == [], "moves use update_box_silent -> no notification"

    # --- mouse release ---
    event.type = "mouse_release"
    try:
        next(gen)
    except StopIteration:
        pass

    assert layer._session_active is False, "edit session should be closed on release"
    assert len(calls) == 1, f"on_commit listener should fire once on release, got {calls}"
    assert calls[0] == 0, f"flush should carry the dragged box index, got {calls[0]}"
