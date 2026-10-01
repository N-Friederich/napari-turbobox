"""Spatial index for axis-aligned bounding boxes, built on sorted arrays."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Same float32 dtype as BBoxDataStore, so the index copy holds exactly the
# stored values. float32 is exact for integer coordinates up to 2**24; never
# use float16.
_COORD_DTYPE = np.float32


class SpatialIndex:
    """Spatial index for axis-aligned bounding boxes, built on sorted arrays.

    For every axis the index keeps the box minima and maxima in sorted order,
    the box index at each sorted position, and the sorted position of each
    box. A query along one axis is a binary search: a box can overlap
    ``[q_lo, q_hi]`` only if its minimum lies in ``[q_lo - max_extent,
    q_hi]``, and these boxes are found in O(log N + w), where w is the number
    of boxes in that window. ``max_extent`` is the largest box extent along
    the axis since the index was last built or rebuilt. It only grows, so
    after a large box shrinks or is removed the window can hold extra
    candidates; they are filtered out, and results stay exact.

    :meth:`intersection` with one constrained axis, such as the slice query
    of a 2D view of a 3D volume, costs one such search. :meth:`query_point`
    and :meth:`intersection` with several constrained axes search each axis
    and intersect the candidate sets with ``np.intersect1d``, which sorts
    them. :meth:`update_box` moves one entry per sorted array and shifts the
    entries in between, which is O(N) in the worst case.

    The index keeps its own copy of the boxes.

    Parameters
    ----------
    boxes : np.ndarray
        Array of bounding boxes with shape (N, 2, ndim).

    Examples
    --------
    >>> import numpy as np
    >>> bboxes = np.array([
    ...     [[0, 0], [10, 10]],
    ...     [[5, 5], [15, 15]],
    ... ])
    >>> index = SpatialIndex(bboxes)
    >>> index.query_point([7, 7])
    [0, 1]
    >>> index.query_point([12, 12])
    [1]
    """

    def __init__(self, boxes: np.ndarray):
        """Build the index.

        Parameters
        ----------
        boxes : np.ndarray
            Array of bounding boxes with shape (N, 2, ndim).
        """
        # Keep a private, writable copy. Callers pass the store's read-only
        # view (BBoxDataStore.data), and the store edits that buffer in place
        # before it notifies listeners. With an alias, update_box could not
        # write to the read-only array, and it would read the new coordinates
        # as the old ones, so it would not notice that the box moved.
        boxes = np.asarray(boxes, dtype=_COORD_DTYPE)
        self._boxes = boxes.copy()
        self._ndim = boxes.shape[-1] if boxes.size > 0 else 0
        self._count = boxes.shape[0] if boxes.size > 0 else 0
        self._sorted_mins = []
        self._sorted_maxs = []
        self._sorted_indices_mins = []
        self._sorted_indices_maxs = []
        self._position_in_sorted_mins = []
        self._position_in_sorted_maxs = []
        if self._count > 0:
            for d in range(self._ndim):
                mins = boxes[:, 0, d]
                sort_idx_min = np.argsort(mins)
                self._sorted_mins.append(mins[sort_idx_min])
                self._sorted_indices_mins.append(sort_idx_min)
                inv_mins = np.empty_like(sort_idx_min)
                inv_mins[sort_idx_min] = np.arange(self._count)
                self._position_in_sorted_mins.append(inv_mins)
                maxs = boxes[:, 1, d]
                sort_idx_max = np.argsort(maxs)
                self._sorted_maxs.append(maxs[sort_idx_max])
                self._sorted_indices_maxs.append(sort_idx_max)
                inv_maxs = np.empty_like(sort_idx_max)
                inv_maxs[sort_idx_max] = np.arange(self._count)
                self._position_in_sorted_maxs.append(inv_maxs)
        self._max_extent = (
            (boxes[:, 1].astype(float) - boxes[:, 0].astype(float)).max(axis=0)
            if self._count > 0
            else np.zeros(self._ndim)
        )

    def _window(self, d: int, q_lo: float, q_hi: float) -> np.ndarray:
        """Return the boxes that overlap ``[q_lo, q_hi]`` along axis ``d``.

        A box overlaps when ``max >= q_lo`` and ``min <= q_hi``. Its min then
        lies in ``[q_lo - max_extent[d], q_hi]``, one contiguous run of the
        sorted mins, so the search costs O(log N + w) with w the length of
        that run, instead of a scan over all mins <= q_hi. The result is in
        the order of the sorted mins.
        """
        lo = 0
        if np.isfinite(q_lo):
            ext = self._max_extent[d]
            # In float64, q_lo - ext can round a few ulps above the exact
            # bound; widen it slightly. Extra candidates are filtered below.
            bound = q_lo - ext - 4 * np.finfo(np.float64).eps * (abs(q_lo) + ext + 1e-300)
            lo = np.searchsorted(self._sorted_mins[d], bound, side="left")
        hi = np.searchsorted(self._sorted_mins[d], q_hi, side="right")
        cand = self._sorted_indices_mins[d][lo:hi]
        return cand[self._boxes[cand, 1, d].astype(np.float64) >= q_lo]

    def query_point(self, point: np.ndarray) -> list[int]:
        """Return the indices of the boxes that contain ``point``.

        Parameters
        ----------
        point : np.ndarray
            Point coordinates (ndim,). If it has fewer than ndim values, only
            the leading axes are checked.

        Returns
        -------
        List[int]
            Indices of the boxes with ``mins[i] <= point[i] <= maxs[i]`` on
            every checked axis (boundaries included). With two or more
            checked axes the list is sorted.

        Notes
        -----
        Each axis is one window search, O(log N + w) (see the class
        docstring). The candidate sets of the axes are then intersected with
        ``np.intersect1d``, which sorts them, so the cost depends on how many
        boxes overlap the point along each axis, not on log N alone.
        """
        if self._count == 0:
            return []
        candidates = None
        for d in range(min(len(point), self._ndim)):
            val = float(point[d])
            valid_in_dim = self._window(d, val, val)
            if candidates is None:
                candidates = valid_in_dim
            else:
                candidates = np.intersect1d(candidates, valid_in_dim, assume_unique=True)
            if len(candidates) == 0:
                return []
        return candidates.tolist() if candidates is not None else []

    def intersection(self, box: np.ndarray) -> list[int]:
        """Return the indices of the boxes that intersect ``box``.

        Parameters
        ----------
        box : np.ndarray
            Query box with shape (2, ndim): ``[[mins], [maxs]]``. An axis
            with ``(-inf, inf)`` is not constrained. Touching boundaries
            count as intersecting.

        Returns
        -------
        List[int]
            Indices of the intersecting boxes. With one constrained axis they
            come in the order of the box minima along that axis, not in index
            order; sort them if the order matters. With several constrained
            axes they are sorted. With none, all indices are returned.
        """
        if self._count == 0:
            return []
        candidates = None
        query_mins = box[0]
        query_maxs = box[1]
        for d in range(min(len(query_mins), self._ndim)):
            q_lo, q_hi = float(query_mins[d]), float(query_maxs[d])
            if q_lo == -np.inf and q_hi == np.inf:
                continue  # unbounded (displayed) axis: no constraint
            valid_in_dim = self._window(d, q_lo, q_hi)
            if candidates is None:
                candidates = valid_in_dim
            else:
                candidates = np.intersect1d(candidates, valid_in_dim, assume_unique=True)
            if len(candidates) == 0:
                return []
        if candidates is None:
            return list(range(self._count))
        return candidates.tolist()

    def query_point_approximate(self, point: np.ndarray, tolerance: float = 0.0) -> list[int]:
        """Same as :meth:`query_point`; ``tolerance`` is not used.

        Parameters
        ----------
        point : np.ndarray
            Point coordinates (ndim,).
        tolerance : float, optional
            Ignored. Default is 0.0.

        Returns
        -------
        List[int]
            Indices of the boxes containing the point.
        """
        return self.query_point(point)

    def rebuild(self, boxes: np.ndarray) -> None:
        """Rebuild the index from scratch for new boxes.

        Parameters
        ----------
        boxes : np.ndarray
            New array of bounding boxes with shape (N, 2, ndim).
        """
        # Private, writable copy, for the reasons given in __init__.
        boxes = np.asarray(boxes, dtype=_COORD_DTYPE)
        self._boxes = boxes.copy()
        self._ndim = boxes.shape[-1] if boxes.size > 0 else 0
        self._count = boxes.shape[0] if boxes.size > 0 else 0
        self._sorted_mins = []
        self._sorted_maxs = []
        self._sorted_indices_mins = []
        self._sorted_indices_maxs = []
        self._position_in_sorted_mins = []
        self._position_in_sorted_maxs = []
        if self._count > 0:
            for d in range(self._ndim):
                mins = boxes[:, 0, d]
                sort_idx_min = np.argsort(mins)
                self._sorted_mins.append(mins[sort_idx_min])
                self._sorted_indices_mins.append(sort_idx_min)
                inv_mins = np.empty_like(sort_idx_min)
                inv_mins[sort_idx_min] = np.arange(self._count)
                self._position_in_sorted_mins.append(inv_mins)
                maxs = boxes[:, 1, d]
                sort_idx_max = np.argsort(maxs)
                self._sorted_maxs.append(maxs[sort_idx_max])
                self._sorted_indices_maxs.append(sort_idx_max)
                inv_maxs = np.empty_like(sort_idx_max)
                inv_maxs[sort_idx_max] = np.arange(self._count)
                self._position_in_sorted_maxs.append(inv_maxs)
        self._max_extent = (
            (boxes[:, 1].astype(float) - boxes[:, 0].astype(float)).max(axis=0)
            if self._count > 0
            else np.zeros(self._ndim)
        )

    def update_box(self, index: int, new_bbox: np.ndarray) -> None:
        """Give one box new coordinates without rebuilding the index.

        For each axis whose min or max changed, the box's entry moves to its
        new sorted position and the entries in between shift by one place.
        The cost is proportional to the number of shifted entries, O(N) in
        the worst case. The stored maximum extent grows if the box got larger
        and never shrinks. An index outside ``[0, N)`` logs a warning and
        changes nothing.

        Parameters
        ----------
        index : int
            Index of the box to update.
        new_bbox : np.ndarray
            New bounding box with shape (2, ndim).
        """
        if index < 0 or index >= self._count:
            logger.warning("Invalid box index%s(valid range: 0-%s)", index, self._count - 1)
            return
        old_bbox = self._boxes[index].copy()
        self._boxes[index] = new_bbox
        self._max_extent = np.maximum(
            self._max_extent, self._boxes[index, 1].astype(float) - self._boxes[index, 0].astype(float)
        )
        for d in range(self._ndim):
            old_val_min = old_bbox[0, d]
            new_val_min = new_bbox[0, d]
            if old_val_min != new_val_min:
                pos = self._position_in_sorted_mins[d][index]
                if new_val_min < old_val_min:
                    # Moves down: entries in [insert_pos, pos) shift up by one.
                    insert_pos = np.searchsorted(self._sorted_mins[d][:pos], new_val_min)
                    slice_indices = self._sorted_indices_mins[d][insert_pos:pos].copy()
                    self._sorted_mins[d][insert_pos + 1 : pos + 1] = self._sorted_mins[d][
                        insert_pos:pos
                    ]
                    self._sorted_indices_mins[d][insert_pos + 1 : pos + 1] = slice_indices
                    self._sorted_mins[d][insert_pos] = new_val_min
                    self._sorted_indices_mins[d][insert_pos] = index
                    self._position_in_sorted_mins[d][slice_indices] += 1
                    self._position_in_sorted_mins[d][index] = insert_pos
                else:
                    # Moves up: entries in (pos, target_pos] shift down by one.
                    insert_offset = np.searchsorted(
                        self._sorted_mins[d][pos + 1 :], new_val_min, side="right"
                    )
                    target_pos = pos + insert_offset
                    slice_indices = self._sorted_indices_mins[d][pos + 1 : target_pos + 1].copy()
                    self._sorted_mins[d][pos:target_pos] = self._sorted_mins[d][
                        pos + 1 : target_pos + 1
                    ]
                    self._sorted_indices_mins[d][pos:target_pos] = slice_indices
                    self._sorted_mins[d][target_pos] = new_val_min
                    self._sorted_indices_mins[d][target_pos] = index
                    self._position_in_sorted_mins[d][slice_indices] -= 1
                    self._position_in_sorted_mins[d][index] = target_pos
            # The same for the maxima.
            old_val_max = old_bbox[1, d]
            new_val_max = new_bbox[1, d]
            if old_val_max != new_val_max:
                pos = self._position_in_sorted_maxs[d][index]
                if new_val_max < old_val_max:
                    insert_pos = np.searchsorted(self._sorted_maxs[d][:pos], new_val_max)
                    slice_indices = self._sorted_indices_maxs[d][insert_pos:pos].copy()
                    self._sorted_maxs[d][insert_pos + 1 : pos + 1] = self._sorted_maxs[d][
                        insert_pos:pos
                    ]
                    self._sorted_indices_maxs[d][insert_pos + 1 : pos + 1] = slice_indices
                    self._sorted_maxs[d][insert_pos] = new_val_max
                    self._sorted_indices_maxs[d][insert_pos] = index
                    self._position_in_sorted_maxs[d][slice_indices] += 1
                    self._position_in_sorted_maxs[d][index] = insert_pos
                else:
                    insert_offset = np.searchsorted(
                        self._sorted_maxs[d][pos + 1 :], new_val_max, side="right"
                    )
                    target_pos = pos + insert_offset
                    slice_indices = self._sorted_indices_maxs[d][pos + 1 : target_pos + 1].copy()
                    self._sorted_maxs[d][pos:target_pos] = self._sorted_maxs[d][
                        pos + 1 : target_pos + 1
                    ]
                    self._sorted_indices_maxs[d][pos:target_pos] = slice_indices
                    self._sorted_maxs[d][target_pos] = new_val_max
                    self._sorted_indices_maxs[d][target_pos] = index
                    self._position_in_sorted_maxs[d][slice_indices] -= 1
                    self._position_in_sorted_maxs[d][index] = target_pos
        logger.debug("Updated box%sin spatial index (incremental)", index)

    def add_boxes(self, new_boxes: np.ndarray) -> None:
        """Append boxes to the index without a full rebuild.

        The new boxes get the indices N to N + M - 1. If M <= N // 10, their
        values are inserted at their ``searchsorted`` positions, O(N + M log N)
        per axis; otherwise the old and new values are concatenated and
        sorted again, O((N + M) log(N + M)) per axis. Either way the position
        arrays are recomputed in O(N + M). On an empty index this is a full
        rebuild.

        Parameters
        ----------
        new_boxes : np.ndarray
            Array of new bounding boxes with shape (M, 2, ndim).
        """
        new_boxes = np.asarray(new_boxes, dtype=_COORD_DTYPE)
        if new_boxes.size == 0:
            return
        n_new = new_boxes.shape[0]
        old_count = self._count
        self._boxes = np.vstack([self._boxes, new_boxes]) if self._boxes.size > 0 else new_boxes
        self._count = self._boxes.shape[0]
        if old_count == 0:
            self.rebuild(self._boxes)
            logger.debug("Added%sboxes to spatial index (initial build)", n_new)
            return
        self._max_extent = np.maximum(
            self._max_extent, (new_boxes[:, 1].astype(float) - new_boxes[:, 0].astype(float)).max(axis=0)
        )
        use_incremental_merge = n_new <= old_count // 10
        for d in range(self._ndim):
            new_mins = new_boxes[:, 0, d]
            new_maxs = new_boxes[:, 1, d]
            if use_incremental_merge:
                sort_new_mins = np.argsort(new_mins)
                sorted_new_mins = new_mins[sort_new_mins]
                new_indices_mins = old_count + sort_new_mins
                insert_positions = np.searchsorted(self._sorted_mins[d], sorted_new_mins)
                self._sorted_mins[d] = np.insert(
                    self._sorted_mins[d], insert_positions, sorted_new_mins
                )
                self._sorted_indices_mins[d] = np.insert(
                    self._sorted_indices_mins[d], insert_positions, new_indices_mins
                )
                new_inv_mins = np.empty(self._count, dtype=int)
                new_inv_mins[self._sorted_indices_mins[d]] = np.arange(self._count)
                self._position_in_sorted_mins[d] = new_inv_mins
                sort_new_maxs = np.argsort(new_maxs)
                sorted_new_maxs = new_maxs[sort_new_maxs]
                new_indices_maxs = old_count + sort_new_maxs
                insert_positions_max = np.searchsorted(self._sorted_maxs[d], sorted_new_maxs)
                self._sorted_maxs[d] = np.insert(
                    self._sorted_maxs[d], insert_positions_max, sorted_new_maxs
                )
                self._sorted_indices_maxs[d] = np.insert(
                    self._sorted_indices_maxs[d], insert_positions_max, new_indices_maxs
                )
                new_inv_maxs = np.empty(self._count, dtype=int)
                new_inv_maxs[self._sorted_indices_maxs[d]] = np.arange(self._count)
                self._position_in_sorted_maxs[d] = new_inv_maxs
            else:
                extended_mins = np.concatenate([self._sorted_mins[d], new_mins])
                new_indices = np.arange(old_count, self._count)
                extended_indices_mins = np.concatenate([self._sorted_indices_mins[d], new_indices])
                sort_order = np.argsort(extended_mins)
                self._sorted_mins[d] = extended_mins[sort_order]
                self._sorted_indices_mins[d] = extended_indices_mins[sort_order]
                inv_mins = np.empty(self._count, dtype=int)
                inv_mins[self._sorted_indices_mins[d]] = np.arange(self._count)
                self._position_in_sorted_mins[d] = inv_mins
                extended_maxs = np.concatenate([self._sorted_maxs[d], new_maxs])
                extended_indices_maxs = np.concatenate([self._sorted_indices_maxs[d], new_indices])
                sort_order = np.argsort(extended_maxs)
                self._sorted_maxs[d] = extended_maxs[sort_order]
                self._sorted_indices_maxs[d] = extended_indices_maxs[sort_order]
                inv_maxs = np.empty(self._count, dtype=int)
                inv_maxs[self._sorted_indices_maxs[d]] = np.arange(self._count)
                self._position_in_sorted_maxs[d] = inv_maxs
        logger.debug(
            "Added%sboxes to spatial index (incremental=%s, total=%s)",
            n_new,
            use_incremental_merge,
            self._count,
        )

    def remove_boxes(self, indices: Sequence[int]) -> None:
        """Remove boxes from the index without sorting again.

        The remaining boxes are renumbered the way ``np.delete`` renumbers
        rows, and the sorted arrays are filtered in O(N) per axis. Indices
        outside ``[0, N)`` are dropped with a warning. The stored maximum
        extent is not reduced.

        Parameters
        ----------
        indices : Sequence[int]
            Unique indices of the boxes to remove, as a list or tuple.
        """
        if not indices:
            return
        indices_to_remove = np.asarray(indices, dtype=int)
        if len(indices_to_remove) == 0:
            return
        if np.any(indices_to_remove < 0) or np.any(indices_to_remove >= self._count):
            logger.warning("Some removal indices out of range (valid: 0-%s)", self._count - 1)
            indices_to_remove = indices_to_remove[
                (indices_to_remove >= 0) & (indices_to_remove < self._count)
            ]
            if len(indices_to_remove) == 0:
                return
        n_removed = len(indices_to_remove)
        keep_mask = np.ones(self._count, dtype=bool)
        keep_mask[indices_to_remove] = False
        self._boxes = self._boxes[keep_mask]
        self._count = self._boxes.shape[0]
        if self._count == 0:
            self._ndim = 0
            self._sorted_mins = []
            self._sorted_maxs = []
            self._sorted_indices_mins = []
            self._sorted_indices_maxs = []
            self._position_in_sorted_mins = []
            self._position_in_sorted_maxs = []
            logger.debug("Removed%sboxes from spatial index (now empty)", n_removed)
            return
        new_indices = np.full(self._count + n_removed, -1, dtype=int)
        new_indices[keep_mask] = np.arange(self._count)
        for d in range(self._ndim):
            old_sorted_indices = self._sorted_indices_mins[d]
            valid_mask = keep_mask[old_sorted_indices]
            self._sorted_mins[d] = self._sorted_mins[d][valid_mask]
            self._sorted_indices_mins[d] = new_indices[old_sorted_indices[valid_mask]]
            new_inv_mins = np.empty(self._count, dtype=int)
            new_inv_mins[self._sorted_indices_mins[d]] = np.arange(self._count)
            self._position_in_sorted_mins[d] = new_inv_mins
            old_sorted_indices = self._sorted_indices_maxs[d]
            valid_mask = keep_mask[old_sorted_indices]
            self._sorted_maxs[d] = self._sorted_maxs[d][valid_mask]
            self._sorted_indices_maxs[d] = new_indices[old_sorted_indices[valid_mask]]
            new_inv_maxs = np.empty(self._count, dtype=int)
            new_inv_maxs[self._sorted_indices_maxs[d]] = np.arange(self._count)
            self._position_in_sorted_maxs[d] = new_inv_maxs
        logger.debug(
            "Removed%sboxes from spatial index (incremental, remaining=%s)", n_removed, self._count
        )
