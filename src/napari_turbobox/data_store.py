"""
Shared bounding box store for synchronized layers.

BBoxDataStore holds one (N, 2, D) float32 array of boxes. Every TurboBoxLayer
that shows these boxes, in one viewer or in several, registers with the same
store and redraws when the store notifies it.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable
from typing import Literal

import numpy as np
from napari.utils.events import EmitterGroup, Event

logger = logging.getLogger(__name__)

# Coordinate storage dtype. float32 holds every integer up to 2**24
# (16,777,216) exactly, far beyond any real volume axis, at half the memory of
# float64. Do not use float16: it is exact only up to 2048, and larger integer
# coordinates round to even values. The store may hold non-integer values
# during a drag, so it cannot be an integer dtype either. SpatialIndex uses the
# same dtype so that its queries see exactly the stored values.
_COORD_DTYPE = np.float32


class _BatchUpdateContext:
    """Context manager returned by :meth:`BBoxDataStore.batch_update`.

    Inside the block no listener is called; the store only records that a
    notification is due. When the outermost block exits, one notification
    with ``changed_idx=None`` is sent if any was held back. This also happens
    when the block raises; the exception is not suppressed.
    """

    def __init__(self, store: BBoxDataStore):
        self.store = store
        self.was_batching = False

    def __enter__(self):
        self.was_batching = self.store._batch_mode
        self.store._batch_mode = True
        logger.debug("Entered batch update mode")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.store._batch_mode = self.was_batching
        if not self.was_batching and self.store._pending_notification:
            logger.debug("Exiting batch mode - sending deferred notification")
            self.store._pending_notification = False
            self.store._notify_listeners()
        return False


class _ListenerHandle:
    """One registered listener, with its sync mode and held-back update.

    Bound methods are held through ``weakref.WeakMethod``, so a layer that is
    garbage-collected drops out by itself. Other callables are held strongly.

    ``sync_mode`` decides when the listener is called:

    - ``"live"``: on every notification. Inside
      :meth:`BBoxDataStore.batch_update` it waits for the end of the batch,
      like every listener.
    - ``"on_commit"``: while an edit session is active, notifications are
      held back and merged into one call at ``end_edit_session``. Outside an
      edit session it behaves like ``"live"``.

    ``accepts_changed_idx`` is True when the callback has a parameter named
    ``changed_idx``; only such callbacks are passed the index.

    ``_has_pending`` is needed in addition to ``_pending_changed_idx``: a
    held-back full update has ``changed_idx=None``, the same value as
    "nothing held back".
    """

    __slots__ = (
        "_callback",
        "_has_pending",
        "_is_weak",
        "_pending_changed_idx",
        "_ref",
        "accepts_changed_idx",
        "sync_mode",
    )

    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        sync_mode: Literal["live", "on_commit"] = "live",
    ):
        import inspect

        self.sync_mode: Literal["live", "on_commit"] = sync_mode
        self._pending_changed_idx: int | None = None
        self._has_pending: bool = False
        try:
            self._ref = weakref.WeakMethod(callback)
            self._callback = None
            self._is_weak = True
        except TypeError:
            # Not a bound method: a plain function, lambda or functools.partial.
            # These are often created inline and referenced nowhere else, so a
            # weak reference would die at once and the listener would never
            # run. Keep a strong reference; the caller has to unregister it.
            self._ref = None
            self._callback = callback
            self._is_weak = False
        try:
            sig = inspect.signature(callback)
            self.accepts_changed_idx = "changed_idx" in sig.parameters
        except Exception:
            self.accepts_changed_idx = False

    def __call__(self) -> Callable[[np.ndarray], None] | None:
        """Return the callback, or None if its object was garbage-collected."""
        if self._is_weak:
            return self._ref()
        return self._callback

    def matches(self, callback: Callable[[np.ndarray], None]) -> bool:
        """Return True if this handle holds ``callback``."""
        if self._is_weak:
            # WeakMethod builds a new bound-method object on every call, so an
            # `is` test never matches a bound method fetched again by the
            # caller. Bound methods compare equal when their __self__ and
            # __func__ match, so compare with ==.
            target = self._ref()
            return target is not None and target == callback
        return self._callback is callback

    @property
    def is_weak(self) -> bool:
        return self._is_weak

    def alive(self) -> bool:
        if self._is_weak:
            return self._ref() is not None
        return True


class BBoxDataStore:
    """
    Shared storage for bounding boxes, with change notifications.

    Several layers, in one viewer or in several, can display the same store.
    A change made through the store's methods is passed to every registered
    listener, so all of these layers show the same boxes.

    Parameters
    ----------
    initial_data : np.ndarray, optional
        Initial boxes, shape (N, 2, D): N boxes, each ``[mins, maxs]`` in D
        dimensions. Default: an empty (0, 2, 3) array.

    Attributes
    ----------
    events : EmitterGroup
        ``events.changed`` is emitted with ``value=`` (a read-only view of
        the boxes) after each round of listener calls, also when an edit
        session holds back the ``on_commit`` listeners.
        ``events.session_ended`` is emitted by :meth:`end_edit_session`
        after the held-back listeners have run.

    Examples
    --------
    Two layers that display the same boxes::

        store = BBoxDataStore(initial_data=boxes)
        layer1 = TurboBoxLayer(bbox_data_store=store)
        layer2 = TurboBoxLayer(bbox_data_store=store)
        # An edit in either layer, or through the store, shows in both.

    Notes
    -----
    The store converts boxes to float32 and does not check them: it does not
    check shapes, order mins and maxs, or compare boxes with the image
    bounds. TurboBoxLayer's ``add_boxes`` and ``bounding_boxes`` setter check
    boxes before they write to the store; direct store writes skip those
    checks.

    An input array that is already float32 is stored without a copy. Later
    in-place updates (:meth:`update_box`) then write into the caller's array,
    and fail if that array is read-only, such as another store's
    :attr:`data`. Pass a copy in these cases.

    Bound-method listeners are held through weak references, so a layer that
    is garbage-collected drops out by itself. Other callables (functions,
    lambdas, functools.partial) are held by strong references and must be
    removed with :meth:`unregister_listener`.

    .. warning::
        This class is not thread-safe. Change it only from the main Qt
        thread. To change it from a worker thread, run the change on the main
        thread with a Qt signal or ``QTimer.singleShot(0, callback)``.
    """

    def __init__(self, initial_data: np.ndarray | None = None):
        """Create the store, optionally with initial boxes.

        ``initial_data`` is converted to float32 without further checks.
        Change the store only from the main thread.
        """
        if initial_data is None:
            self._bboxes = np.empty((0, 2, 3), dtype=_COORD_DTYPE)
        else:
            self._bboxes = np.asarray(initial_data, dtype=_COORD_DTYPE)
        self._listeners: list[_ListenerHandle] = []
        self.events = EmitterGroup(source=self, changed=Event, session_ended=Event)
        self._batch_mode = False
        self._pending_notification = False
        self._in_edit_session = False

    def _assert_main_thread(self) -> None:
        """Warn once if the store is changed off the Qt main thread.

        Checked only when Python runs without ``-O`` and a QApplication
        exists.
        """
        if __debug__:
            try:
                from qtpy.QtCore import QThread
                from qtpy.QtWidgets import QApplication

                app = QApplication.instance()
                if app is not None and QThread.currentThread() != app.thread():
                    if not getattr(self, "_thread_warning_shown", False):
                        self._thread_warning_shown = True
                        import warnings

                        warnings.warn(
                            "BBoxDataStore mutation called off the main thread! All mutations must occur on the main thread. Use QTimer.singleShot(0, callback) or Qt signals to schedule on the main thread.",
                            UserWarning,
                            stacklevel=3,
                        )
            except ImportError:
                pass

    @property
    def data(self) -> np.ndarray:
        """
        Read-only view of the stored boxes.

        Returns
        -------
        np.ndarray
            Array of shape (N, 2, D), dtype float32. The view shares memory
            with the store: in-place updates (``update_box``,
            ``update_boxes``, ``update_box_silent``) show up in it, while
            methods that replace the array (``add_boxes``, ``remove_boxes``,
            ``clear``, setting ``data``) leave an older view as it was. Copy
            it to keep a snapshot.
        """
        view = self._bboxes.view()
        view.flags.writeable = False
        return view

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """
        Replace all boxes and notify every listener with ``changed_idx=None``.

        Parameters
        ----------
        value : np.ndarray
            New boxes, shape (N, 2, D). Converted to float32, not checked.
        """
        self._assert_main_thread()
        self._bboxes = np.asarray(value, dtype=_COORD_DTYPE)
        self._notify_listeners()

    def update(self, new_bboxes: np.ndarray) -> None:
        """
        Replace all boxes; the same as setting :attr:`data`.

        Parameters
        ----------
        new_bboxes : np.ndarray
            New boxes, shape (N, 2, D).
        """
        self.data = new_bboxes

    def update_box_silent(self, index: int, new_box: np.ndarray) -> None:
        """Overwrite one box in place without notifying anyone.

        No listener is called or marked as pending, so no layer redraws. The
        caller has to send a notification later, for example with
        :meth:`update_box` or :meth:`update`.
        """
        self._assert_main_thread()
        self._bboxes[index] = np.asarray(new_box, dtype=float)

    def update_box(self, index: int, new_box: np.ndarray) -> None:
        """Overwrite one box in place and notify the listeners.

        Listeners that accept ``changed_idx`` receive ``index``. TurboBoxLayer
        calls this on every drag step; ``on_commit`` listeners hold the
        notification back while an edit session is active. Only the row is
        written; the array is not copied. The new box is not checked.

        Parameters
        ----------
        index : int
            Row to overwrite.
        new_box : np.ndarray
            New ``[mins, maxs]``, shape (2, D).
        """
        self._assert_main_thread()
        self._bboxes[index] = np.asarray(new_box, dtype=float)
        self._notify_listeners(changed_idx=index)

    def update_boxes(self, updates: dict[int, np.ndarray]) -> None:
        """Overwrite several boxes in place, then notify once.

        Parameters
        ----------
        updates : dict of int -> np.ndarray
            Maps a box index to its new ``(2, D)`` value. An empty dict does
            nothing.

        Notes
        -----
        Listeners get one notification with ``changed_idx=None``, not one per
        index. TurboBoxLayer answers it with a full rebuild.
        """
        self._assert_main_thread()
        if not updates:
            return
        for index, new_box in updates.items():
            self._bboxes[index] = np.asarray(new_box, dtype=float)
        self._notify_listeners(changed_idx=None)

    def add_boxes(self, boxes: np.ndarray) -> None:
        """
        Append boxes and notify every listener with ``changed_idx=None``.

        The new boxes go after the existing ones, so existing indices stay
        the same. Empty input does nothing.

        Parameters
        ----------
        boxes : np.ndarray
            Boxes to add, shape (M, 2, D). Converted to float32, not checked.
        """
        self._assert_main_thread()
        boxes = np.asarray(boxes, dtype=_COORD_DTYPE)
        if boxes.size == 0:
            return
        if self._bboxes.size == 0:
            self._bboxes = boxes
        else:
            self._bboxes = np.concatenate([self._bboxes, boxes], axis=0)
        self._notify_listeners()

    def remove_boxes(self, indices: list[int]) -> None:
        """
        Delete boxes and notify every listener with ``changed_idx=None``.

        Later boxes move down to close the gaps, as with ``np.delete``. An
        empty list does nothing.

        Parameters
        ----------
        indices : list of int
            Indices of the boxes to remove.
        """
        self._assert_main_thread()
        if len(indices) == 0:
            return
        self._bboxes = np.delete(self._bboxes, indices, axis=0)
        self._notify_listeners()

    def clear(self) -> None:
        """Remove all boxes (keeping D) and notify every listener."""
        self._assert_main_thread()
        self._bboxes = np.empty((0, 2, self._bboxes.shape[-1]), dtype=_COORD_DTYPE)
        self._notify_listeners()

    def register_listener(
        self,
        callback: Callable[[np.ndarray], None],
        *,
        sync_mode: Literal["live", "on_commit"] = "live",
    ) -> _ListenerHandle:
        """
        Register a callback to be called when the boxes change.

        Parameters
        ----------
        callback : callable
            Called as ``callback(data)``, or as ``callback(data,
            changed_idx=...)`` if it has a parameter named ``changed_idx``.
            ``data`` is a read-only view of the stored array. ``changed_idx``
            is the edited row after a single-box update (or after held-back
            updates that all edited that row) and None after any other
            change. Exceptions from the callback are logged, not re-raised.
        sync_mode : {"live", "on_commit"}, optional
            ``"live"`` (default) calls the listener on every change.
            ``"on_commit"`` holds the calls back while an edit session is
            active and merges them into one call when it ends (see
            :meth:`begin_edit_session`).

        Returns
        -------
        _ListenerHandle
            The handle for this listener. Its ``sync_mode`` can be changed
            later, directly or with :meth:`set_listener_sync_mode`. If
            ``callback`` is already registered, the existing handle is
            returned and ``sync_mode`` is ignored.

        Notes
        -----
        Bound methods are held weakly; other callables are held strongly
        (a warning is logged) and must be unregistered. If ``callback`` is
        also connected to ``events.changed``, it is disconnected there so it
        does not run twice per change.
        """
        for handle in self._listeners:
            if handle.matches(callback):
                return handle
        handle = _ListenerHandle(callback, sync_mode=sync_mode)
        if not handle.alive():
            return handle
        if not handle.is_weak:
            logger.warning(
                "Listener %r cannot be weak-referenced; storing strong reference", callback
            )
        self._listeners.append(handle)
        if hasattr(self.events, "changed"):
            try:
                if callback in self.events.changed.callbacks:
                    self.events.changed.disconnect(callback)
                    logger.debug("Removed duplicate callback from events.changed")
            except (ValueError, TypeError, AttributeError):
                pass
        return handle

    def set_listener_sync_mode(
        self,
        callback: Callable[[np.ndarray], None],
        sync_mode: Literal["live", "on_commit"],
    ) -> None:
        """Change the ``sync_mode`` of a registered listener.

        The new mode applies from the next notification. An update that is
        already held back for this listener is still delivered when the edit
        session ends.

        Parameters
        ----------
        callback : callable
            The registered callback.
        sync_mode : {"live", "on_commit"}
            The new mode.

        Raises
        ------
        ValueError
            If ``callback`` is not a registered listener.
        """
        for handle in self._listeners:
            if handle.matches(callback):
                handle.sync_mode = sync_mode
                return
        raise ValueError(f"Callback {callback!r} is not a registered listener")

    def begin_edit_session(self) -> None:
        """Start an edit session.

        While it is active, ``on_commit`` listeners are not called; their
        updates are held back until :meth:`end_edit_session`. Calling this
        again during a session does nothing, and sessions do not nest: the
        first ``end_edit_session`` ends the session.
        """
        self._in_edit_session = True

    def end_edit_session(self) -> None:
        """End the edit session and deliver the held-back updates.

        Each listener with a held-back update is called once, with the merged
        ``changed_idx``, and its pending state is cleared. Then
        ``events.session_ended`` is emitted. Without an active session this
        does nothing.
        """
        if not self._in_edit_session:
            return
        self._in_edit_session = False
        self._listeners = [h for h in self._listeners if h.alive()]
        for handle in self._listeners:
            had_pending = handle._has_pending
            changed_idx = handle._pending_changed_idx
            handle._has_pending = False
            handle._pending_changed_idx = None
            if had_pending:
                self._call_listener(handle, changed_idx)
        self.events.session_ended()

    @property
    def in_edit_session(self) -> bool:
        """True between :meth:`begin_edit_session` and :meth:`end_edit_session`."""
        return self._in_edit_session

    def unregister_listener(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Remove a registered callback. Does nothing if it is not registered.

        Parameters
        ----------
        callback : callable
            The callback passed to :meth:`register_listener`.
        """
        self._listeners = [h for h in self._listeners if not h.matches(callback)]

    def batch_update(self):
        """
        Return a context manager that merges notifications into one.

        Inside the ``with`` block no listener is called. When the outermost
        block exits, the listeners get one notification with
        ``changed_idx=None``, which TurboBoxLayer answers with a full
        rebuild. This also happens if the block raises. Blocks can be nested.

        Examples
        --------
        Replace the first box with new ones and notify once::

            with store.batch_update():
                store.remove_boxes([0])
                store.add_boxes(new_boxes)
            # The listeners are called once, here.
        """
        return _BatchUpdateContext(self)

    def _call_listener(self, handle: _ListenerHandle, changed_idx: int | None) -> None:
        """Call one listener, passing ``changed_idx`` only if it accepts it.

        An exception from the listener is logged and not re-raised, so the
        other listeners still run. A listener that an exception can leave
        half-updated has to recover by itself: TurboBoxLayer, for example,
        drops its per-shape bookkeeping and rebuilds on the next update.
        """
        callback = handle()
        if callback is None:
            return
        view = self._bboxes.view()
        view.flags.writeable = False
        try:
            if handle.accepts_changed_idx:
                callback(view, changed_idx=changed_idx)
            else:
                callback(view)
        except Exception:
            logger.exception("Error notifying listener")

    def _notify_listeners(self, changed_idx: int | None = None) -> None:
        """Call the listeners after a change, or hold the call back.

        In batch mode nothing is called; the batch sends one notification
        when it ends. During an edit session, ``on_commit`` listeners are only
        marked as pending. Their pending ``changed_idx`` stays set while every
        held-back update edits the same row, and becomes None (full rebuild)
        once two differ, for example a bulk change after a single-box update.
        ``events.changed`` is emitted on every call outside batch mode.
        """
        if self._batch_mode:
            self._pending_notification = True
            logger.debug("Deferring listener notification (batch mode active)")
            return
        self._listeners = [h for h in self._listeners if h.alive()]
        logger.debug("Notifying%slisteners", len(self._listeners))
        view = self._bboxes.view()
        view.flags.writeable = False
        for handle in self._listeners:
            if handle.sync_mode == "on_commit" and self._in_edit_session:
                if not handle._has_pending:
                    handle._pending_changed_idx = changed_idx
                    handle._has_pending = True
                elif handle._pending_changed_idx != changed_idx:
                    handle._pending_changed_idx = None
                # else: the same row again, the pending index stays
            else:
                self._call_listener(handle, changed_idx)
        self.events.changed(value=view)

    @property
    def ndim(self) -> int:
        """Dimensionality D of the stored boxes; 3 for an empty store."""
        return self._bboxes.shape[-1] if self._bboxes.size > 0 else 3

    @property
    def num_boxes(self) -> int:
        """Number of stored boxes."""
        return len(self._bboxes)

    def __len__(self) -> int:
        """Return the number of stored boxes."""
        return self.num_boxes

    def __repr__(self) -> str:
        """Return a short summary: number of boxes and ndim."""
        return f"BBoxDataStore(num_boxes={self.num_boxes}, ndim={self.ndim})"
