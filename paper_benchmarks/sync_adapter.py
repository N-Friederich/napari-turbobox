"""Synchronization adapters for the drag-session benchmark.

A :class:`SyncAdapter` exposes a uniform three-method interface
(``begin_edit_session`` / ``update_box`` / ``end_edit_session``) so the
benchmark driver can treat napari-turbobox and napari-bbox identically.

- :class:`TurboBoxAdapter` is a thin pass-through to the real
  :class:`~napari_turbobox.BBoxDataStore` production API.
- :class:`NapariBboxAdapter` simulates the per-viewer sync behaviour for
  napari-bbox, which has no shared store: 2D viewers get an immediate
  ``layer.data = current`` per update, 3D viewers get a single coalesced
  ``layer.data = current`` at ``end_edit_session`` (mirroring the
  3D-coalescing that TurboBox provides natively via ``sync_mode="on_commit"``).
- :class:`NapariBboxPerBoxAdapter` does the same with napari-bbox's public
  per-box API (``selected_data`` / ``remove_selected`` / ``add``) instead of a
  full ``layer.data`` assignment, so each viewer rebuilds only the edited box;
  optionally with thumbnails blocked during the edit session (as TurboBox does).
- :class:`TurboBoxMouseAdapter` and :class:`NapariBboxMouseAdapter` drag the box
  with the mouse in the first (XY) view instead: press, moves and release go
  through napari's mouse dispatch into each tool's own drag handler. TurboBox's
  handler updates the other views through its store; for napari-bbox the other
  views follow per box as in :class:`NapariBboxPerBoxAdapter`.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod

import numpy as np


class SyncAdapter(ABC):
    """Uniform edit-session interface over a multi-viewer sync mechanism."""

    @abstractmethod
    def begin_edit_session(self) -> None:
        """Open an edit session (defer on_commit-style updates)."""

    @abstractmethod
    def update_box(self, idx: int, box: np.ndarray) -> None:
        """Apply a single box update at index ``idx``."""

    @abstractmethod
    def end_edit_session(self) -> None:
        """Close the edit session and flush any deferred updates."""


class TurboBoxAdapter(SyncAdapter):
    """Thin wrapper around BBoxDataStore — uses the real production API."""

    def __init__(self, main_layer):
        self._store = main_layer._bbox_store

    def begin_edit_session(self) -> None:
        self._store.begin_edit_session()

    def update_box(self, idx: int, box: np.ndarray) -> None:
        self._store.update_box(idx, box)

    def end_edit_session(self) -> None:
        self._store.end_edit_session()


class NapariBboxAdapter(SyncAdapter):
    """Simulates per-viewer sync_mode for napari-bbox.

    2D viewers receive immediate ``layer.data = current`` updates.
    3D viewers receive a single ``layer.data = current`` at
    ``end_edit_session`` (mirroring the 3D-coalescing TurboBox provides
    natively).
    """

    def __init__(self, layers: list, ndisplays: list[int]):
        assert len(layers) == len(ndisplays)
        self._live_layers = [l for l, nd in zip(layers, ndisplays) if nd == 2]
        self._commit_layers = [l for l, nd in zip(layers, ndisplays) if nd == 3]
        self._all_layers = list(layers)
        self._in_session = False
        self._pending = None

    def begin_edit_session(self) -> None:
        self._in_session = True
        self._pending = None

    def update_box(self, idx: int, box: np.ndarray) -> None:
        # Snapshot current state, mutate, broadcast.
        current = self._all_layers[0].data.copy()
        current[idx] = box
        for layer in self._live_layers:
            layer.data = current
        if self._in_session:
            self._pending = current
        else:
            for layer in self._commit_layers:
                layer.data = current

    def end_edit_session(self) -> None:
        if self._pending is not None:
            for layer in self._commit_layers:
                layer.data = self._pending
        self._in_session = False
        self._pending = None


class NapariBboxPerBoxAdapter(SyncAdapter):
    """Per-box napari-bbox sync through its public layer API.

    Each update removes the edited box from every live layer and adds the
    new box at the end, so the application keeps the index permutation
    ``perm`` (layer position -> logical box id), identical in all layers.
    3D layers receive the final box once at ``end_edit_session``.

    ``block_thumbnails``: symmetric to TurboBox, which skips thumbnails inside an
    edit session and rasterizes them once when the session ends: napari-bbox's
    public ``block_thumbnail_update()`` during the updates, one thumbnail per
    layer at the end.

    ``quiet``: ``remove_selected()`` and ``add()`` each emit two napari data
    events (removing/removed, adding/added), and the viewer answers every one
    by recomputing the layer extents, which is O(N). napari-bbox's own mouse
    drag emits one ``changing`` event at its first move and one ``changed``
    event at the release, none in between. ``quiet`` does the same: napari's
    public ``events.data.blocker()`` during the drag, then one ``changed`` data
    event per layer for the rows that hold a different box than before. It
    also selects the box by adding it to the (empty) selection set instead of
    assigning the selection, which skips the selection setter's side effects
    (copying the box's style into the layer's current style, and an edge-width
    event that pushes the whole mesh to vispy); the box's style is passed to
    ``add()`` instead.
    """

    def __init__(self, layers: list, ndisplays: list[int], block_thumbnails: bool = False,
                 quiet: bool = False):
        assert len(layers) == len(ndisplays)
        self._live_layers = [l for l, nd in zip(layers, ndisplays) if nd == 2]
        self._commit_layers = [l for l, nd in zip(layers, ndisplays) if nd == 3]
        self._follow_layers = self._live_layers  # live layers updated through remove/add
        self._block_thumbnails = block_thumbnails
        self._quiet = quiet
        n = len(layers[0].data)
        self._perm = np.arange(n)
        self._perm_commit = np.arange(n)
        self._in_session = False
        self._pending = {}
        self._lo_live = None  # first live-layer row replaced since the last data event

    def _replace(self, layer, pos: int, box: np.ndarray) -> None:
        # remove_selected() and add() each refresh the layer and rasterize its
        # thumbnail. The thumbnail after the removal shows a transient state that
        # add() replaces at once, so it is skipped with napari-bbox's public
        # block_thumbnail_update(): one thumbnail per replaced box, as for TurboBox.
        with contextlib.ExitStack() as stack:
            if self._quiet:
                stack.enter_context(layer.events.data.blocker())
            style = {}
            with layer.block_thumbnail_update():
                if self._quiet:
                    assert not layer.selected_data
                    style = {"edge_color": layer.edge_color[pos], "face_color": layer.face_color[pos],
                             "edge_width": layer.edge_width[pos]}
                    layer.selected_data.add(pos)
                else:
                    layer.selected_data = {pos}
                layer.remove_selected()
            layer.add(np.asarray(box, float)[None], **style)

    def _announce(self, layers: list, lo) -> None:
        """With ``quiet``: one ``changed`` data event per layer for rows ``lo`` to the end.

        remove/add moved each edited box to the end, so from the first removed
        row on every row holds a different box than before the drag.
        """
        if not self._quiet or lo is None:
            return
        from napari.layers.base._base_constants import ActionType

        for layer in layers:
            rows = tuple(range(lo, layer.nbounding_boxes))
            layer.events.data(
                value=layer.data,
                action=ActionType.CHANGED,
                data_indices=rows,
                vertex_indices=((),) * len(rows),
            )

    def begin_edit_session(self) -> None:
        self._in_session = True
        self._pending = {}

    def _follow(self, idx: int, box: np.ndarray) -> None:
        pos = int(np.flatnonzero(self._perm == idx)[0])
        for layer in self._follow_layers:
            if self._block_thumbnails and self._in_session:
                with layer.block_thumbnail_update():
                    self._replace(layer, pos, box)
            else:
                self._replace(layer, pos, box)
        self._perm = np.append(np.delete(self._perm, pos), idx)
        self._lo_live = pos if self._lo_live is None else min(self._lo_live, pos)
        self._pending[idx] = np.asarray(box, float)

    def _announce_live(self) -> None:
        self._announce(self._follow_layers, self._lo_live)
        self._lo_live = None

    def update_box(self, idx: int, box: np.ndarray) -> None:
        self._follow(idx, box)
        if not self._in_session:
            self._announce_live()
            self._flush()

    def _flush(self) -> None:
        lo = None
        for idx, box in self._pending.items():
            pos = int(np.flatnonzero(self._perm_commit == idx)[0])
            for layer in self._commit_layers:
                self._replace(layer, pos, box)
            self._perm_commit = np.append(np.delete(self._perm_commit, pos), idx)
            lo = pos if lo is None else min(lo, pos)
        self._announce(self._commit_layers, lo)
        self._pending = {}

    def end_edit_session(self) -> None:
        was_in_session = self._in_session
        self._in_session = False
        self._announce_live()
        self._flush()
        if self._block_thumbnails and was_in_session:
            for layer in self._follow_layers:
                layer._update_thumbnail()


class _MouseEvent:
    """The attributes of napari's mouse event that the drag handlers read."""

    def __init__(self, type_: str, position, dims_displayed):
        self.type = type_
        self.position = tuple(float(c) for c in position)
        self.modifiers = []
        self.dims_displayed = [int(d) for d in dims_displayed]
        self.handled = False
        self.is_dragging = type_ != "mouse_press"
        self.view_direction = None
        self.button = 1


class _MouseDrag:
    """One drag in ``layer`` through napari's layer-level mouse dispatch.

    The press, every move and the release go through napari's
    ``mouse_press_callbacks`` / ``mouse_move_callbacks`` / ``mouse_release_callbacks``,
    so all drag callbacks of the layer run as for a mouse in the canvas. The
    canvas-level work of a real mouse (cursor position, status bar, viewer
    callbacks) is not run, for either tool. Only the hit test at the press is
    given: the layer method ``hit_name`` reports ``hit_value`` (the box the
    benchmark drags, whatever box lies on top there).
    """

    def __init__(self, layer, position, hit_name: str, hit_value):
        from napari.utils.interactions import mouse_press_callbacks

        self._layer = layer
        setattr(layer, hit_name, lambda *a, **k: hit_value)
        try:
            mouse_press_callbacks(layer, self._event("mouse_press", position))
        finally:
            delattr(layer, hit_name)
        if not layer._mouse_drag_gen:
            raise RuntimeError("the press did not start a drag in the layer")

    def _event(self, type_: str, position):
        from napari.utils._proxies import ReadOnlyWrapper

        event = _MouseEvent(type_, position, self._layer._slice_input.displayed)
        return ReadOnlyWrapper(event, exceptions=("handled",))

    def move(self, position) -> None:
        from napari.utils.interactions import mouse_move_callbacks

        mouse_move_callbacks(self._layer, self._event("mouse_move", position))

    def release(self, position) -> None:
        from napari.utils.interactions import mouse_release_callbacks

        mouse_release_callbacks(self._layer, self._event("mouse_release", position))
        if self._layer._mouse_drag_gen:
            raise RuntimeError("a drag callback of the layer did not finish on release")


def _drag_kind(start: np.ndarray, end: np.ndarray, displayed) -> tuple[bool, np.ndarray]:
    """Resize or translation, and the grabbed point, of a drag from ``start`` to ``end``.

    The benchmark's drags either keep the size (translation, grabbed at the centre)
    or keep the min corner (resize, grabbed at the max corner in the view).
    """
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    resize = not np.allclose(end[1] - end[0], start[1] - start[0])
    if resize and not np.allclose(end[0], start[0]):
        raise ValueError("a resize drag keeps the min corner of the box fixed")
    grab = start.mean(0)
    if resize:
        grab[list(displayed)] = start[1][list(displayed)]
    return resize, grab


class _MouseAdapter(SyncAdapter):
    """Drags the box with the mouse in the edited view; subclasses sync the other views.

    ``prepare_drag`` is the press (untimed, before the drag), ``update_box`` one mouse
    move and ``end_edit_session`` the release. The cursor follows the grabbed point
    of the box, so the handler puts the box where the benchmark's trajectory says.
    """

    def __init__(self, edit_layer, hit_name: str):
        self._edit_layer = edit_layer
        self._hit_name = hit_name
        edit_layer.mode = "select"
        self._drag = None  # (idx, _MouseDrag, resize, grab - grabbed corner or centre)
        self._cursor = None

    def _hit_value(self, idx: int, resize: bool):
        raise NotImplementedError

    def _current_box(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def prepare_drag(self, idx: int, start: np.ndarray, end: np.ndarray) -> None:
        """Mouse press on box ``idx`` in the edited view."""
        assert self._drag is None
        resize, grab = _drag_kind(start, end, self._edit_layer._slice_input.displayed)
        start = np.asarray(start, float)
        offset = grab - (start[1] if resize else start.mean(0))
        drag = _MouseDrag(self._edit_layer, grab, self._hit_name, self._hit_value(idx, resize))
        self._drag = (idx, drag, resize, offset)
        self._cursor = grab

    def begin_edit_session(self) -> None:
        pass

    def _move(self, idx: int, box: np.ndarray) -> None:
        box = np.asarray(box, float)
        if self._drag is None:  # an update outside a prepared drag (e.g. the warm-up restore)
            self.prepare_drag(idx, self._current_box(idx), box)
        d_idx, drag, resize, offset = self._drag
        assert d_idx == idx, (d_idx, idx)
        self._cursor = (box[1] if resize else box.mean(0)) + offset
        drag.move(self._cursor)

    def _release(self) -> None:
        if self._drag is not None:
            drag = self._drag[1]
            self._drag = None
            drag.release(self._cursor)


class TurboBoxMouseAdapter(_MouseAdapter):
    """TurboBox dragged with the mouse in its first (XY) view.

    Its drag handler writes each move to the shared store, which updates the
    other views; the release closes the store's edit session (3D views, undo,
    thumbnails). Hit test given: ``_hit_test_bbox`` (box id, near an edge = resize).
    """

    def __init__(self, main_layer):
        super().__init__(main_layer, "_hit_test_bbox")
        self._store = main_layer._bbox_store

    def _hit_value(self, idx, resize):
        return (idx, resize)

    def _current_box(self, idx):
        return np.array(self._store.data[idx], float)

    def prepare_drag(self, idx, start, end):
        super().prepare_drag(idx, start, end)
        state = self._edit_layer._drag_state
        assert state and state["index"] == idx and state["mode"] == ("resize" if self._drag[2] else "translate")

    def update_box(self, idx: int, box: np.ndarray) -> None:
        self._move(idx, box)

    def end_edit_session(self) -> None:
        self._release()


class NapariBboxMouseAdapter(_MouseAdapter, NapariBboxPerBoxAdapter):
    """napari-bbox dragged with the mouse in its first (XY) view.

    Its select-mode handler moves or scales the box in place and refreshes the
    layer on every move; the release emits the data event and rasterizes the
    thumbnail. Hit test given: ``get_value`` (box index, and the max corner's
    vertex for a resize). The other live views follow per box through the
    public remove/add path with thumbnails deferred to the end of the drag
    (and, with ``quiet``, data events deferred to the release as well); 3D views
    get the final box at the release (:class:`NapariBboxPerBoxAdapter`).
    """

    def __init__(self, layers: list, ndisplays: list[int], quiet: bool = False):
        NapariBboxPerBoxAdapter.__init__(self, layers, ndisplays, block_thumbnails=True, quiet=quiet)
        _MouseAdapter.__init__(self, self._live_layers[0], "get_value")
        self._follow_layers = self._live_layers[1:]
        self._n = len(self._edit_layer.data)

    @property
    def _synced_layers(self):
        return self._follow_layers

    def _current_box(self, idx):
        # The edited view changes boxes in place: its positions are the box ids.
        d = np.asarray(self._edit_layer._data_view.bounding_boxes[idx].data, float)
        return np.stack([d.min(0), d.max(0)])

    def _hit_value(self, idx, resize):
        from napari_bbox.boundingbox.napari_0_4_15._bounding_box_constants import Box

        layer = self._edit_layer
        if not resize:
            return (idx, None)
        # A box's corners are grabbable once it is selected: select it, then the
        # press lands on its max corner (vertex opposite the fixed min corner).
        layer.selected_data = {idx}
        disp = list(layer._slice_input.displayed)
        start = self._current_box(idx)
        corners = layer._selected_box[: Box.LEN]
        vertex = int(np.argmin(np.abs(corners - start[1][disp]).sum(1)))
        assert np.allclose(corners[vertex], start[1][disp])
        assert np.allclose(corners[(vertex + 4) % Box.LEN], start[0][disp])
        return (idx, vertex)

    def prepare_drag(self, idx, start, end):
        assert len(self._edit_layer._data_view.bounding_boxes) == self._n
        super().prepare_drag(idx, start, end)

    def begin_edit_session(self) -> None:
        NapariBboxPerBoxAdapter.begin_edit_session(self)

    def update_box(self, idx: int, box: np.ndarray) -> None:
        self._move(idx, box)
        self._follow(idx, box)
        if not self._in_session:
            self._release()
            self._announce_live()
            self._flush()

    def end_edit_session(self) -> None:
        self._release()
        NapariBboxPerBoxAdapter.end_edit_session(self)
