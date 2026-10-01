"""Regression tests for fixes landed during production hardening.

Each test corresponds to a specific previously-broken behaviour.

API note: deferred synchronization is per-listener. Register a listener with
``sync_mode="on_commit"`` and bracket the edit with ``begin_edit_session()`` /
``end_edit_session()``; the listener is then flushed exactly once.
"""

import numpy as np
import pytest


def test_clear_boxes_does_not_raise():
    """clear_boxes() must not raise NameError on undo description."""
    from napari_turbobox import TurboBoxLayer

    layer = TurboBoxLayer(ndim=3, image_shape=(10, 100, 100))
    layer.add_boxes(np.array([[[0, 0, 0], [5, 50, 50]]]))
    layer.clear_boxes()  # must not raise
    assert layer.nshapes == 0


def test_widget_cache_limit_callback_does_not_raise(qtbot):
    """Widget cache-limit spinner must delegate to layer, not mutate widget."""
    pytest.importorskip("qtpy")

    from napari_turbobox import TurboBoxLayer
    from napari_turbobox.widget import BoundingBoxControlWidget

    class _Evt:
        def connect(self, *a, **k):
            pass

    class _Layers(list):
        class events:  # noqa: N801 - mimics napari.utils.events.EventedList.events
            inserted = removed = changed = _Evt()

    class _MockViewer:
        layers = _Layers()

    widget = BoundingBoxControlWidget(_MockViewer())
    qtbot.addWidget(widget)  # pytest-qt manages widget lifecycle/cleanup
    layer = TurboBoxLayer(ndim=3)
    widget.layer_combo.addItem("test", layer)
    widget._on_cache_limit_changed(5000)  # must not raise
    assert layer.cache_limit == 5000


def test_batch_emits_all_event_types_once():
    """Batch context must deduplicate per event_type, not drop all but last."""
    from napari_turbobox.event_manager import BoundingBoxEventManager

    received = []

    class Sub:
        def handle_event(self, event_type, payload):
            received.append((event_type, payload))

    mgr = BoundingBoxEventManager()
    sub = Sub()
    mgr.subscribe(sub)
    with mgr.batch():
        mgr.emit("added", "a1")
        mgr.emit("updated", "u1")
        mgr.emit("updated", "u2")  # later payload should win for this type
    types = [t for t, _ in received]
    assert "added" in types
    assert "updated" in types
    assert len(received) == 2  # one emission per unique type
    assert dict(received)["updated"] == "u2"  # latest payload wins


def test_shape_edit_triggers_single_sync():
    """One shape edit must produce exactly one store-write notification."""
    from napari_turbobox import TurboBoxLayer

    layer = TurboBoxLayer(ndim=3, image_shape=(10, 100, 100))
    layer.add_boxes(np.array([[[0, 0, 0], [5, 50, 50]]]))
    counter = {"n": 0}
    layer.events.bboxes.connect(lambda e: counter.__setitem__("n", counter["n"] + 1))
    before = counter["n"]
    new_data = layer.bounding_boxes.copy()
    new_data[0, 1] = [8, 80, 80]  # valid in-bounds edit (image_shape 10,100,100)
    layer.bounding_boxes = new_data
    assert counter["n"] - before == 1, f"expected 1 emission, got {counter['n'] - before}"


def test_sync_mode_on_commit_only_fires_on_release():
    """In on_commit mode, sub-viewer must receive exactly one notification per commit."""
    from napari_turbobox import BBoxDataStore

    store = BBoxDataStore(np.zeros((3, 2, 3)))
    notifications = []

    # named function: the store keeps only a weak reference to listeners,
    # so the callback must stay reachable for the lifetime of the test
    def listener(data, **kw):
        notifications.append(len(data))

    store.register_listener(listener, sync_mode="on_commit")
    store.begin_edit_session()

    # update_box_silent is fully silent: it never notifies or marks pending.
    for i in range(10):
        new_box = np.array([[i, i, i], [i + 1, i + 1, i + 1]], dtype=float)
        store.update_box_silent(0, new_box)
    assert len(notifications) == 0, f"silent updates leaked: {len(notifications)}"

    # A real update inside the session must defer too.
    store.update_box(0, np.zeros((2, 3)))
    assert len(notifications) == 0, "update during session leaked a notification"

    store.end_edit_session()
    assert len(notifications) == 1, f"commit should fire once, got {len(notifications)}"
    store.end_edit_session()  # second end with nothing pending -> no-op
    assert len(notifications) == 1, "no-op commit fired anyway"


def test_new_box_in_2d_does_not_inherit_rectangle_type(make_napari_viewer):
    """Drawing a box in 2D must not crash via inherited shape_type.

    napari's internal shape_type memory can hold 'rectangle' (left by the
    add-rectangle mouse binding). Our 2D slice paths are 5-vertex closed
    rectangles; if napari builds them as Rectangle shapes it raises
    ``ValueError`` (a Rectangle expects exactly 4 vertices). The 2D sync
    must therefore pin shape_type='polygon' explicitly.
    """
    from napari.layers.shapes._shapes_constants import Mode

    from napari_turbobox import TurboBoxLayer

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((10, 100, 100), dtype=np.uint8))
    layer = TurboBoxLayer(ndim=3, image_shape=(10, 100, 100))
    viewer.add_layer(layer)
    viewer.dims.ndisplay = 2
    viewer.dims.set_point(0, 5)

    # Draw a box as a rectangle (what the ADD_RECTANGLE binding does) — this
    # leaves napari's shape_type memory holding 'rectangle'.
    layer.mode = Mode.ADD_RECTANGLE
    layer.add(
        np.array([[5, 20, 20], [5, 20, 60], [5, 60, 60], [5, 60, 20]], dtype=float),
        shape_type="rectangle",
    )

    # A re-sync emits 5-vertex closed slice rectangles — must not raise.
    layer.add_boxes(np.array([[[5, 70, 10], [5, 90, 40]]], dtype=float))

    assert all(st == "polygon" for st in layer.shape_type), (
        f"2D slice shapes must be pinned to 'polygon', got {list(layer.shape_type)}"
    )
