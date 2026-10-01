import gc
import importlib.util
import logging
import weakref
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "napari_turbobox" / "data_store.py"
spec = importlib.util.spec_from_file_location("napari_turbobox.data_store", MODULE_PATH)
data_store = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(data_store)

BBoxDataStore = data_store.BBoxDataStore


def test_listener_cleaned_when_target_deleted():
    store = BBoxDataStore()

    class Listener:
        def __init__(self):
            self.count = 0

        def callback(self, data):
            self.count += 1

    listener = Listener()
    store.register_listener(listener.callback)

    handle = store._listeners[0]
    assert handle.is_weak
    ref = weakref.ref(listener)

    store.add_boxes(np.zeros((1, 2, store.ndim), dtype=float))
    assert listener.count == 1

    del listener
    gc.collect()

    store.add_boxes(np.zeros((1, 2, store.ndim), dtype=float))
    assert not store._listeners
    assert ref() is None


def test_non_weak_listener_logs_warning(caplog):
    store = BBoxDataStore()

    class NoWeakSlots:
        __slots__ = ()

        def __call__(self, data):
            pass

    listener = NoWeakSlots()

    with caplog.at_level(logging.WARNING):
        store.register_listener(listener)

    assert "cannot be weak-referenced" in caplog.text
    assert len(store._listeners) == 1
    assert not store._listeners[0].is_weak

    store.unregister_listener(listener)
    assert not store._listeners


def test_notification_counts():
    """Verify that listeners and events are fired exactly once per update."""
    store = BBoxDataStore()

    # Mock listener and event handler
    listener_counts = 0
    event_counts = 0

    def on_listener_notify(data):
        nonlocal listener_counts
        listener_counts += 1

    def on_event_change(event):
        nonlocal event_counts
        event_counts += 1

    store.register_listener(on_listener_notify)
    store.events.changed.connect(on_event_change)

    # 1. Update data directly
    store.data = np.zeros((1, 2, 3))
    assert listener_counts == 1
    assert event_counts == 1

    # 2. Add boxes
    store.add_boxes(np.zeros((1, 2, 3)))
    assert listener_counts == 2
    assert event_counts == 2

    # 3. Remove boxes
    store.remove_boxes([0])
    assert listener_counts == 3
    assert event_counts == 3

    # 4. Clear
    store.clear()
    assert listener_counts == 4
    assert event_counts == 4


def test_store_data_is_readonly():
    store = BBoxDataStore()
    store.data = np.zeros((1, 2, 3))
    data = store.data
    import pytest

    with pytest.raises(ValueError):
        data[0, 0, 0] = 9.9


def test_batch_update_context():
    store = BBoxDataStore()
    notified = 0

    def callback(data):
        nonlocal notified
        notified += 1

    store.register_listener(callback)

    with store.batch_update():
        store.add_boxes(np.zeros((1, 2, 3)))
        store.add_boxes(np.zeros((1, 2, 3)))
        assert notified == 0

    assert notified == 1
    assert len(store) == 2


def test_edit_session_with_on_commit_listener():
    """An on_commit listener defers during a session and flushes exactly once.

    Renamed/rewritten from the former ``test_commit_modes`` (which exercised
    the removed store-wide ``commit_mode`` property).
    """
    store = BBoxDataStore(np.zeros((1, 2, 3)))
    calls = []

    def callback(data, changed_idx=None):
        calls.append(changed_idx)

    store.register_listener(callback, sync_mode="on_commit")
    store.begin_edit_session()

    for _ in range(5):
        store.update_box(0, np.ones((2, 3)))
    assert calls == [], "on_commit listener fired during session"

    store.end_edit_session()
    assert calls == [0], f"expected one flush with changed_idx=0, got {calls}"


def test_live_listener_unaffected_by_edit_session():
    """A live listener sees every update immediately, session or not."""
    store = BBoxDataStore(np.zeros((2, 2, 3)))
    calls = []

    def callback(data, changed_idx=None):
        calls.append(changed_idx)

    store.register_listener(callback)  # default sync_mode="live"

    store.begin_edit_session()
    store.update_box(0, np.ones((2, 3)))
    store.update_box(1, np.ones((2, 3)))
    assert calls == [0, 1], f"live listener deferred during session: {calls}"
    store.end_edit_session()
    assert calls == [0, 1], "end_edit_session re-fired a live listener"


def test_on_commit_listener_no_pending_no_flush():
    """An empty edit session flushes nothing."""
    store = BBoxDataStore()
    calls = []

    def callback(data, changed_idx=None):
        calls.append(changed_idx)

    store.register_listener(callback, sync_mode="on_commit")
    store.begin_edit_session()
    store.end_edit_session()
    assert calls == [], "empty session flushed a listener"


def test_on_commit_multi_index_pending_collapses_to_rebuild():
    """Touching multiple indices collapses to one None (rebuild) flush."""
    store = BBoxDataStore(np.zeros((2, 2, 3)))
    calls = []

    def callback(data, changed_idx=None):
        calls.append(changed_idx)

    store.register_listener(callback, sync_mode="on_commit")
    store.begin_edit_session()
    store.update_box(0, np.ones((2, 3)))
    store.update_box(1, np.ones((2, 3)))
    store.end_edit_session()
    assert calls == [None], f"expected one rebuild flush, got {calls}"


def test_set_listener_sync_mode_mid_session():
    """sync_mode can be flipped mid-session and takes effect immediately."""
    store = BBoxDataStore(np.zeros((1, 2, 3)))
    calls = []

    def callback(data, changed_idx=None):
        calls.append(changed_idx)

    store.register_listener(callback)  # starts live
    store.begin_edit_session()

    # Switch to on_commit -> subsequent updates defer.
    store.set_listener_sync_mode(callback, "on_commit")
    store.update_box(0, np.ones((2, 3)))
    store.update_box(0, np.full((2, 3), 2.0))
    assert calls == [], f"updates leaked while on_commit: {calls}"

    # Switch back to live -> updates fire immediately again.
    store.set_listener_sync_mode(callback, "live")
    store.update_box(0, np.full((2, 3), 3.0))
    assert calls == [0], f"live update did not fire immediately: {calls}"

    # The earlier deferred update is still pending and flushes on session end.
    store.end_edit_session()
    assert calls == [0, 0], f"expected pending flush after session end, got {calls}"

    import pytest

    with pytest.raises(ValueError):
        store.set_listener_sync_mode(lambda d: None, "live")


def test_update_boxes_atomic_notification():
    """update_boxes fires exactly one None notification for multi-index edits."""
    store = BBoxDataStore(np.zeros((3, 2, 3)))
    calls = []

    def callback(data, changed_idx=None):
        calls.append(changed_idx)

    store.register_listener(callback)
    store.update_boxes(
        {
            0: np.ones((2, 3)),
            1: np.full((2, 3), 2.0),
            2: np.full((2, 3), 3.0),
        }
    )
    assert calls == [None], f"expected one atomic notification, got {calls}"


def test_end_edit_session_idempotent():
    """Calling end_edit_session twice with nothing pending is harmless."""
    store = BBoxDataStore()
    calls = []

    def callback(data, changed_idx=None):
        calls.append(changed_idx)

    store.register_listener(callback, sync_mode="on_commit")
    store.end_edit_session()  # no active session
    store.end_edit_session()  # still no active session
    assert calls == [], "idempotent end_edit_session fired a listener"


def test_add_and_remove_empty():
    store = BBoxDataStore()
    notified = 0

    def callback(data):
        nonlocal notified
        notified += 1

    store.register_listener(callback)

    store.add_boxes(np.empty((0, 2, 3)))
    assert notified == 0

    store.remove_boxes([])
    assert notified == 0


def test_len_and_repr():
    store = BBoxDataStore()
    assert len(store) == 0
    assert repr(store) == "BBoxDataStore(num_boxes=0, ndim=3)"

    store.data = np.zeros((2, 2, 4))
    assert len(store) == 2
    assert repr(store) == "BBoxDataStore(num_boxes=2, ndim=4)"


def test_thread_warning_triggered(unittest_mock=None):
    import warnings
    from unittest.mock import MagicMock, patch

    store = BBoxDataStore()

    mock_app = MagicMock()
    mock_thread_app = MagicMock()
    mock_thread_current = MagicMock()

    mock_app.thread.return_value = mock_thread_app

    with (
        patch("qtpy.QtWidgets.QApplication.instance", return_value=mock_app),
        patch("qtpy.QtCore.QThread.currentThread", return_value=mock_thread_current),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        store.add_boxes(np.zeros((1, 2, 3)))
        assert len(w) == 1
        assert "mutation called off the main thread" in str(w[0].message)


def test_listener_with_changed_idx():
    store = BBoxDataStore()
    notified = []

    def callback(data, changed_idx=None):
        notified.append(changed_idx)

    store.register_listener(callback)
    store.data = np.zeros((1, 2, 3))
    assert notified == [None]

    store.update_box(0, np.ones((2, 3)))
    assert notified == [None, 0]

    # test listener notification error handling does not crash
    def bad_callback(data):
        raise RuntimeError("Oops")

    store.register_listener(bad_callback)
    store.add_boxes(np.zeros((1, 2, 3)))
