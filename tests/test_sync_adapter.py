"""Tests for the benchmark synchronization adapters."""

import numpy as np
import pytest


def test_turbobox_adapter_routes_through_store():
    """TurboBoxAdapter delegates to the real BBoxDataStore edit-session API.

    An on_commit listener must see exactly one flush at end_edit_session;
    a live listener must see one notification per update_box.
    """
    from napari_turbobox.layer import TurboBoxLayer
    from paper_benchmarks.sync_adapter import TurboBoxAdapter

    layer = TurboBoxLayer(ndim=3, image_shape=(100, 100, 100))
    layer.add_boxes(np.array([[[10, 10, 10], [20, 20, 20]]], dtype=float))
    store = layer._bbox_store
    adapter = TurboBoxAdapter(layer)

    # on_commit listener: deferred, flushed once at end_edit_session.
    commit_calls = []

    def on_commit_listener(data, changed_idx=None):
        commit_calls.append(changed_idx)

    store.register_listener(on_commit_listener, sync_mode="on_commit")

    adapter.begin_edit_session()
    for i in range(5):
        adapter.update_box(0, np.array([[10 + i, 10, 10], [20 + i, 20, 20]], dtype=float))
    assert commit_calls == [], "on_commit listener fired mid-session"
    adapter.end_edit_session()
    assert commit_calls == [0], f"on_commit listener should flush once for box 0, got {commit_calls}"

    # live listener: immediate, one call per update_box.
    live_calls = []

    def live_listener(data, changed_idx=None):
        live_calls.append(changed_idx)

    store.register_listener(live_listener, sync_mode="live")

    adapter.begin_edit_session()
    for i in range(5):
        adapter.update_box(0, np.array([[10 + i, 10, 10], [20 + i, 20, 20]], dtype=float))
    adapter.end_edit_session()
    assert live_calls == [0, 0, 0, 0, 0], f"live listener should fire per update, got {live_calls}"


def test_napari_bbox_adapter_coalesces_3d():
    """NapariBboxAdapter updates 2D layers live and coalesces 3D layers.

    A 2D viewer receives one ``layer.data`` assignment per update_box; a 3D
    viewer receives exactly one coalesced assignment at end_edit_session.
    """
    from paper_benchmarks.sync_adapter import NapariBboxAdapter

    class FakeLayer:
        """Minimal stand-in for a napari layer that tracks .data reassignments."""

        def __init__(self, data):
            self._data = data
            self.assign_count = 0

        @property
        def data(self):
            return self._data

        @data.setter
        def data(self, value):
            self._data = value
            self.assign_count += 1

    initial = np.zeros((3, 2, 3), dtype=float)
    layer_2d = FakeLayer(initial.copy())
    layer_3d = FakeLayer(initial.copy())

    adapter = NapariBboxAdapter([layer_2d, layer_3d], ndisplays=[2, 3])

    adapter.begin_edit_session()
    for i in range(4):
        adapter.update_box(0, np.full((2, 3), float(i)))

    assert layer_2d.assign_count == 4, "2D layer should get one assignment per update"
    assert layer_3d.assign_count == 0, "3D layer must not update mid-session"

    adapter.end_edit_session()

    assert layer_3d.assign_count == 1, "3D layer should get exactly one coalesced assignment"
    assert layer_2d.assign_count == 4, "end_edit_session must not re-assign 2D layers"

    # The coalesced 3D state reflects the final update.
    np.testing.assert_array_equal(layer_3d.data[0], np.full((2, 3), 3.0))


def _mouse_drags(monkeypatch, tool, n_drags=4, legacy_sync="per_box_nothumb"):
    """Run the harness's drags with --edit-view mouse; return (adapter, current boxes)."""
    pytest.importorskip("qtpy")
    import paper_benchmarks.drag_session_benchmark as dsb

    for name, value in (("DRAG_PROTOCOL", "xy_visible"), ("EDIT_VIEW", "mouse"), ("PACING", "spin"),
                        ("LEGACY_SYNC", legacy_sync), ("THUMBNAILS", "off"), ("ORTHO_FOLLOW", False),
                        ("WARMUP", 0)):
        monkeypatch.setattr(dsb, name, value)
    shape = (30, 64, 64)
    setup = dsb.setup_modern if tool == "modern" else dsb.setup_legacy
    adapter, viewers, _layers, current = setup(20, 2, shape, True, True)
    try:
        for d in range(n_drags):  # even d: translation, odd d: resize
            dsb.measure_drag_session(adapter, d, 20, current, shape, n_sub=6, target_hz=1000)
    except Exception:
        for v in viewers:
            v.close()
        raise
    return adapter, viewers, current


def _minmax(corners):
    c = np.asarray(corners, float)
    return np.stack([c.min(0), c.max(0)])


def test_turbobox_mouse_adapter_drags_through_the_handler(monkeypatch):
    adapter, viewers, current = _mouse_drags(monkeypatch, "modern")
    try:
        layer = adapter._edit_layer
        assert layer._drag_state is None and not layer._mouse_drag_gen
        np.testing.assert_allclose(layer._bbox_store.data, current, atol=1e-9)
        assert len(layer._undo_stack._history) >= 4  # one entry per drag release
    finally:
        for v in viewers:
            v.close()


@pytest.mark.parametrize("legacy_sync", ["per_box_nothumb", "per_box_quiet"])
def test_napari_bbox_mouse_adapter_drags_through_the_handler(monkeypatch, legacy_sync):
    pytest.importorskip("napari_bbox")
    adapter, viewers, current = _mouse_drags(monkeypatch, "legacy", legacy_sync=legacy_sync)
    try:
        edit = adapter._edit_layer
        assert adapter._quiet == (legacy_sync == "per_box_quiet")
        assert not edit._mouse_drag_gen and not edit._is_moving
        # the edited view changes boxes in place; the others follow per box (permutation)
        for pos in range(20):
            np.testing.assert_allclose(_minmax(edit.data[pos]), current[pos], atol=1e-9)
        for layer in adapter._synced_layers:
            for pos in range(20):
                np.testing.assert_allclose(_minmax(layer.data[pos]), current[adapter._perm[pos]], atol=1e-9)
        for layer in adapter._commit_layers:
            for pos in range(20):
                np.testing.assert_allclose(_minmax(layer.data[pos]), current[adapter._perm_commit[pos]], atol=1e-9)
    finally:
        for v in viewers:
            v.close()


def log_action(log):
    return lambda e: log.append(str(e.action))


def test_quiet_per_box_defers_data_events_to_the_release(monkeypatch):
    """per_box_quiet: no napari data event on the synchronized layers during a drag, one at the release."""
    pytest.importorskip("napari_bbox")
    import paper_benchmarks.drag_session_benchmark as dsb

    for name, value in (("DRAG_PROTOCOL", "xy_visible"), ("EDIT_VIEW", "mouse"), ("PACING", "spin"),
                        ("LEGACY_SYNC", "per_box_quiet"), ("ORTHO_FOLLOW", False), ("WARMUP", 0)):
        monkeypatch.setattr(dsb, name, value)
    shape = (30, 64, 64)
    adapter, viewers, _shims, current = dsb.setup_legacy(20, 3, shape, True, True)
    try:
        watched = adapter._follow_layers + adapter._commit_layers
        events = [[] for _ in watched]
        for layer, log in zip(watched, events):
            layer.events.data.connect(log_action(log))
        orig_update = adapter.update_box

        def update(idx, box):
            orig_update(idx, box)
            assert all(not ev for ev in events), events

        adapter.update_box = update
        dsb.measure_drag_session(adapter, 0, 20, current, shape, n_sub=6, target_hz=1000)
        for ev in events:
            assert len(ev) == 1 and ev[0].lower().endswith("changed"), ev
    finally:
        for v in viewers:
            v.close()
