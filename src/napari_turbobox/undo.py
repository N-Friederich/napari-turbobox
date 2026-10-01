"""Undo/redo history of bounding box arrays."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class UndoStack:
    """
    Undo/redo history that stores a full copy of the box array per entry.

    Each entry holds the boxes after an action and that action's
    description. The first entry is the starting state: undo stops there.
    TurboBoxLayer pushes one entry per completed action (one per drag, not
    one per mouse move).

    Parameters
    ----------
    max_size : int
        Maximum number of entries, the starting state included (default:
        50), so at most ``max_size - 1`` undo steps. When a push goes over
        the limit, the oldest entry is dropped. Lowering ``max_size`` later
        does not trim the history at once; each following push drops only
        one entry. For the float32 (N, 2, D) arrays TurboBoxLayer pushes,
        an entry takes 8 * D bytes per box (24 for 3D), so 50 entries of
        100 3D boxes take about 120 kB.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._history = []
        self._current_index = -1
        self._enabled = True
        logger.info("UndoStack initialized (max_size=%s)", max_size)

    def push(self, state: np.ndarray, description: str = "Edit"):
        """
        Record a new state. Does nothing while recording is disabled.

        Any redo entries after the current position are discarded.

        Parameters
        ----------
        state : np.ndarray
            The boxes after the action (copied).
        description : str
            Short description of the action, shown in the undo/redo labels.
        """
        if not self._enabled:
            return
        state_copy = state.copy()
        if self._current_index < len(self._history) - 1:
            self._history = self._history[: self._current_index + 1]
            logger.debug("Truncated redo history (%ssteps remain)", len(self._history))
        self._history.append((state_copy, description))
        self._current_index += 1
        if len(self._history) > self.max_size:
            self._history.pop(0)
            self._current_index -= 1
            logger.debug("Evicted oldest undo state (limit=%s)", self.max_size)
        logger.debug(
            "Pushed state: '%s' (index=%s/%s)",
            description,
            self._current_index,
            len(self._history) - 1,
        )

    def undo(self) -> tuple[np.ndarray, str] | None:
        """
        Step back one entry.

        Returns
        -------
        (state, description) or None
            A copy of the restored (previous) state and that state's own
            description, which names the action that produced it, not the
            action being undone; call :meth:`get_undo_description` first for
            that. None if there is nothing to undo.
        """
        if not self.can_undo():
            logger.warning("Cannot undo: no previous state")
            return None
        self._current_index -= 1
        state, description = self._history[self._current_index]
        logger.info(
            "Undo: '%s' (index=%s/%s)", description, self._current_index, len(self._history) - 1
        )
        return (state.copy(), description)

    def redo(self) -> tuple[np.ndarray, str] | None:
        """
        Step forward one entry.

        Returns
        -------
        (state, description) or None
            A copy of the next state and its description, which names the
            action being redone. None if there is nothing to redo.
        """
        if not self.can_redo():
            logger.warning("Cannot redo: no future state")
            return None
        self._current_index += 1
        state, description = self._history[self._current_index]
        logger.info(
            "Redo: '%s' (index=%s/%s)", description, self._current_index, len(self._history) - 1
        )
        return (state.copy(), description)

    def can_undo(self) -> bool:
        """True if there is an entry before the current one."""
        return self._current_index > 0

    def can_redo(self) -> bool:
        """True if there is an entry after the current one."""
        return self._current_index < len(self._history) - 1

    def clear(self):
        """Delete all entries. The next push becomes the new starting state."""
        self._history.clear()
        self._current_index = -1
        logger.info("Cleared undo history")

    def disable(self):
        """Stop recording: :meth:`push` does nothing until :meth:`enable`.

        TurboBoxLayer disables recording while it applies an undo or redo.
        """
        self._enabled = False

    def enable(self):
        """Resume recording after :meth:`disable`."""
        self._enabled = True

    def get_undo_description(self) -> str | None:
        """Return the description of the action that undo would revert, or None.

        Each entry holds the state after an action and that action's
        description, so this is the current entry's description.
        """
        if self.can_undo():
            return self._history[self._current_index][1]
        return None

    def get_redo_description(self) -> str | None:
        """Return the description of the action that redo would apply, or None."""
        if self.can_redo():
            return self._history[self._current_index + 1][1]
        return None

    @property
    def size(self) -> int:
        """Number of entries in the history."""
        return len(self._history)

    @property
    def current_position(self) -> int:
        """Index of the current entry (0-based; -1 when the history is empty)."""
        return self._current_index

    def get_memory_usage(self) -> int:
        """Estimate the memory held by the history, in bytes.

        The array sizes plus a flat 100 bytes per entry for the tuple and
        description.
        """
        if not self._history:
            return 0
        bytes_per_state = sum((state.nbytes for state, _ in self._history))
        overhead = len(self._history) * 100
        return bytes_per_state + overhead

    def __repr__(self) -> str:
        mem_kb = self.get_memory_usage() / 1024
        return f"UndoStack(size={self.size}/{self.max_size}, position={self._current_index}, memory={mem_kb:.1f}KB, can_undo={self.can_undo()}, can_redo={self.can_redo()})"
