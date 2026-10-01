"""Fix B: ``_allow_thumbnail_update`` suppresses thumbnail rasterization during
an interactive drag (and only then), reclaiming per-frame cost; the thumbnail
is rasterized once at release. This is purely an interactive-path optimization
and does not touch the benchmark (which bypasses ``_on_mouse_drag``).
"""

from unittest.mock import MagicMock, Mock

import numpy as np

from napari_turbobox import BBoxDataStore
from napari_turbobox.layer import TurboBoxLayer


def test_update_thumbnail_gated_by_flag(monkeypatch):
    import napari.layers.shapes.shapes as shapes_mod

    # Construct first, THEN install the spy, so napari's construction-time
    # refresh calls are not counted.
    layer = TurboBoxLayer(ndim=3)
    calls = []
    monkeypatch.setattr(
        shapes_mod.Shapes, "_update_thumbnail", lambda self, *a, **k: calls.append(1)
    )

    layer._allow_thumbnail_update = False
    layer._update_thumbnail()
    assert calls == [], "thumbnail must be suppressed while flag is False"

    layer._allow_thumbnail_update = True
    layer._update_thumbnail()
    assert calls == [1], "thumbnail must pass through to napari while flag is True"


def test_thumbnail_flag_default_true():
    layer = TurboBoxLayer(ndim=3)
    assert layer._allow_thumbnail_update is True


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
    slice_input.world_slice = MagicMock(point=np.array([4.0, 49.0, 49.0]))
    slice_input.data_slice = MagicMock(return_value=MagicMock(point=np.array([4.0, 49.0, 49.0])))
    layer._slice_input = slice_input
    layer._data_view = ShapeList(ndisplay=2)
    layer._ndisplay_stored = 2
    layer._set_view_slice = lambda: None
    layer._update_dims = lambda: None
    return layer


def test_thumbnail_suppressed_during_drag_restored_after():
    store = BBoxDataStore()
    layer = _make_drag_ready_layer(store)
    layer.add_boxes(np.array([[[2, 20, 20], [5, 50, 50]]], dtype=float))
    assert layer._allow_thumbnail_update is True

    event = Mock()
    event.type = "mouse_press"
    event.position = np.array([3.5, 35.0, 35.0])
    event.modifiers = []
    event.dims_displayed = [0, 1, 2]
    event.handled = False

    gen = layer._on_mouse_drag(layer, event)
    next(gen)  # press
    assert layer._allow_thumbnail_update is False, "suppressed during drag"

    event.type = "mouse_move"
    event.position = np.array([3.5, 41.0, 41.0])
    next(gen)
    assert layer._allow_thumbnail_update is False, "still suppressed mid-drag"

    event.type = "mouse_release"
    try:
        next(gen)
    except StopIteration:
        pass
    assert layer._allow_thumbnail_update is True, "restored at release"
