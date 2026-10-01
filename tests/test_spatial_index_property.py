"""Property-based equivalence test for SpatialIndex.

The canonical guard for every future index change: after ANY sequence of
add / remove / update_box operations, the incrementally-maintained index must
be **equivalent to a freshly rebuilt one** -- equivalence defined as identical
results for a battery of point and slice/intersection queries (query-result
equivalence, not internal-array equality, since sort stability may differ).

Run: pytest tests/test_spatial_index_property.py
On failure, hypothesis prints the minimal reproducing op sequence and a
@reproduce_failure blob.
"""

import numpy as np
import pytest

pytest.importorskip("hypothesis")  # testing extra; not installed in every environment

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from napari_turbobox.spatial_index import SpatialIndex

# Half-integer grid in a small range => frequent ties / touching / boundary
# boxes, which is exactly where incremental re-sort logic tends to break.
COORD = st.integers(min_value=0, max_value=40).map(lambda x: x / 2.0)


class SpatialIndexMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.ndim = None
        self.model = None  # ground-truth (N, 2, D) array
        self.index = None  # the incrementally maintained index under test

    @initialize(ndim=st.sampled_from([2, 3]), data=st.data())
    def setup(self, ndim, data):
        self.ndim = ndim
        n = data.draw(st.integers(0, 6))
        self.model = self._draw_boxes(data, n)
        self.index = SpatialIndex(self.model.copy())

    def _draw_box(self, data):
        c1 = np.array([data.draw(COORD) for _ in range(self.ndim)])
        c2 = np.array([data.draw(COORD) for _ in range(self.ndim)])
        return np.stack([np.minimum(c1, c2), np.maximum(c1, c2)])  # (2, D), maxs>=mins

    def _draw_boxes(self, data, n):
        if n == 0:
            return np.empty((0, 2, self.ndim), dtype=float)
        return np.stack([self._draw_box(data) for _ in range(n)])

    @rule(data=st.data())
    def op_add(self, data):
        new = self._draw_boxes(data, data.draw(st.integers(1, 4)))
        self.index.add_boxes(new.copy())
        self.model = np.concatenate([self.model, new], axis=0) if self.model.size else new

    @precondition(lambda self: self.model is not None and len(self.model) > 0)
    @rule(data=st.data())
    def op_update(self, data):
        i = data.draw(st.integers(0, len(self.model) - 1))
        new = self._draw_box(data)
        self.index.update_box(i, new.copy())
        self.model[i] = new

    @precondition(lambda self: self.model is not None and len(self.model) > 0)
    @rule(data=st.data())
    def op_remove(self, data):
        idxs = sorted(
            data.draw(
                st.lists(
                    st.integers(0, len(self.model) - 1),
                    unique=True,
                    min_size=1,
                    max_size=len(self.model),
                )
            )
        )
        self.index.remove_boxes(idxs)
        keep = [j for j in range(len(self.model)) if j not in set(idxs)]
        self.model = self.model[keep] if keep else np.empty((0, 2, self.ndim), dtype=float)

    @invariant()
    def equivalent_to_fresh_rebuild(self):
        if self.model is None:
            return
        fresh = SpatialIndex(self.model.copy())
        for p in self._query_points():
            assert set(self.index.query_point(p)) == set(
                fresh.query_point(p)
            ), f"query_point disagreement at {p.tolist()}"
        for qb in self._query_boxes():
            assert set(self.index.intersection(qb)) == set(
                fresh.intersection(qb)
            ), f"intersection disagreement at {qb.tolist()}"

    def _query_points(self):
        pts = [np.full(self.ndim, float(v)) for v in (0, 5, 10, 20)]
        for box in self.model:  # box-derived: centres, corners, edges
            pts.append((box[0] + box[1]) / 2.0)
            pts.append(box[0].copy())
            pts.append(box[1].copy())
        return pts

    def _query_boxes(self):
        qbs = []
        for v in (0, 5, 10, 20):  # thin slice-like probes
            qbs.append(np.stack([np.full(self.ndim, v - 0.5), np.full(self.ndim, v + 0.5)]))
        qbs.append(np.stack([np.zeros(self.ndim), np.full(self.ndim, 20.0)]))  # whole space
        return qbs


TestSpatialIndexEquivalence = SpatialIndexMachine.TestCase
TestSpatialIndexEquivalence.settings = settings(
    max_examples=200,
    stateful_step_count=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
