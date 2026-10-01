import os
import sys

import numpy as np

# Add src/napari_turbobox to path to import spatial_index directly without triggering package __init__
# (which might import napari)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/napari_turbobox")))
from spatial_index import SpatialIndex


def test_spatial_index_query():
    # [[0,0], [10,10]]
    # [[5,5], [15,15]]
    boxes = np.array(
        [
            [[0, 0], [10, 10]],
            [[5, 5], [15, 15]],
        ],
        dtype=float,
    )

    index = SpatialIndex(boxes)

    # Point inside both
    assert sorted(index.query_point(np.array([7, 7]))) == [0, 1]

    # Point inside box 0 only
    assert index.query_point(np.array([2, 2])) == [0]

    # Point inside box 1 only
    assert index.query_point(np.array([12, 12])) == [1]

    # Point outside both
    assert index.query_point(np.array([20, 20])) == []


def test_spatial_index_update_box():
    boxes = np.array(
        [
            [[0, 0], [10, 10]],  # Box 0
            [[20, 20], [30, 30]],  # Box 1
        ],
        dtype=float,
    )

    index = SpatialIndex(boxes)

    # Verify initial state
    assert index.query_point(np.array([5, 5])) == [0]
    assert index.query_point(np.array([25, 25])) == [1]

    # Move Box 0 to overlap with Box 1 location
    # New Box 0: [[22, 22], [28, 28]] (inside Box 1)
    new_box = np.array([[22, 22], [28, 28]], dtype=float)
    index.update_box(0, new_box)

    # Verify Box 0 is moved
    assert index.query_point(np.array([5, 5])) == []  # Old location empty

    # New location should contain both 0 and 1 (since 0 is inside 1)
    # Wait, Box 1 is 20-30. Box 0 is 22-28. Both contain 25.
    assert sorted(index.query_point(np.array([25, 25]))) == [0, 1]

    # Verify partial updates (add another box)
    new_box_2 = np.array([[100, 100], [110, 110]], dtype=float)
    index.add_boxes(new_box_2[None, ...])  # Add Box 2
    assert index.query_point(np.array([105, 105])) == [2]

    # Update added box
    index.update_box(2, np.array([[0, 0], [5, 5]]))
    assert index.query_point(np.array([2, 2])) == [2]
    assert index.query_point(np.array([105, 105])) == []


def test_spatial_index_incremental_ordering():
    """Verify strictly that ordering is maintained after updates."""
    # Create boxes [0,1], [2,3], [4,5] ...
    N = 10
    boxes = np.zeros((N, 2, 2))
    for i in range(N):
        boxes[i] = [[i, i], [i + 1, i + 1]]

    index = SpatialIndex(boxes)

    # Move box 0 to the end [N, N+1]
    index.update_box(0, np.array([[N, N], [N + 1, N + 1]]))

    # Check sorted arrays directly
    # Min of box 0 is now N. Max is N+1.
    # Prior sorted mins: 0, 1, 2, ... N-1
    # Expected sorted mins: 1, 2, ... N-1, N

    expected_mins = np.arange(1, N + 1)
    assert np.array_equal(index._sorted_mins[0], expected_mins)

    # The index for value N (which is box 0) should be at the end
    assert index._sorted_indices_mins[0][-1] == 0


def test_spatial_index_position_mappings_sync():
    """Verify that position mapping arrays are strictly inverse of sorted indices."""
    boxes = np.array(
        [
            [[0, 0], [10, 10]],
            [[5, 5], [15, 15]],
            [[2, 2], [8, 8]],
        ],
        dtype=float,
    )
    index = SpatialIndex(boxes)

    def assert_mappings_valid(idx):
        for d in range(idx._ndim):
            # Check mins
            for pos, box_id in enumerate(idx._sorted_indices_mins[d]):
                assert idx._position_in_sorted_mins[d][box_id] == pos
            # Check maxs
            for pos, box_id in enumerate(idx._sorted_indices_maxs[d]):
                assert idx._position_in_sorted_maxs[d][box_id] == pos

    assert_mappings_valid(index)

    # 1. Test update_box
    index.update_box(1, np.array([[1, 1], [6, 6]], dtype=float))
    assert_mappings_valid(index)

    # 2. Test add_boxes bulk add
    index.add_boxes(np.array([[[12, 12], [20, 20]]], dtype=float))
    assert_mappings_valid(index)

    # 3. Test remove_boxes
    index.remove_boxes([0])
    assert_mappings_valid(index)


def test_spatial_index_incremental_merge_trigger():
    """Verify that both bulk and true incremental merge produce correct indices & mappings."""
    # Create 30 boxes to allow n_new <= 30 // 10 = 3
    N = 30
    boxes = np.zeros((N, 2, 2))
    for i in range(N):
        boxes[i] = [[i, i], [i + 0.5, i + 0.5]]

    index = SpatialIndex(boxes)

    # Add 2 boxes -> triggers incremental merge (2 <= 3)
    new_boxes_inc = np.array(
        [
            [[5.1, 5.1], [5.6, 5.6]],
            [[15.1, 15.1], [15.6, 15.6]],
        ],
        dtype=float,
    )
    index.add_boxes(new_boxes_inc)

    # Verify correctness of sorted lists and query
    assert index._count == 32
    for d in range(2):
        assert np.all(np.diff(index._sorted_mins[d]) >= 0)
        assert np.all(np.diff(index._sorted_maxs[d]) >= 0)
        # Check mapping inverse property
        for pos, box_id in enumerate(index._sorted_indices_mins[d]):
            assert index._position_in_sorted_mins[d][box_id] == pos

    # Add 10 boxes -> triggers bulk fallback (10 > 32 // 10 = 3)
    new_boxes_bulk = np.array(
        [[[i + 0.2, i + 0.2], [i + 0.7, i + 0.7]] for i in range(10)], dtype=float
    )
    index.add_boxes(new_boxes_bulk)

    assert index._count == 42
    for d in range(2):
        assert np.all(np.diff(index._sorted_mins[d]) >= 0)
        assert np.all(np.diff(index._sorted_maxs[d]) >= 0)
        for pos, box_id in enumerate(index._sorted_indices_mins[d]):
            assert index._position_in_sorted_mins[d][box_id] == pos


def test_spatial_index_empty_rebuild_edge_cases():
    """Verify various edge cases for empty and re-added boxes."""
    index = SpatialIndex(np.empty((0, 2, 2), dtype=float))
    assert index._count == 0
    assert index.query_point(np.array([1, 1])) == []

    # Add box to empty index
    index.add_boxes(np.array([[[1, 1], [3, 3]]], dtype=float))
    assert index._count == 1
    assert index.query_point(np.array([2, 2])) == [0]

    # Remove last box
    index.remove_boxes([0])
    assert index._count == 0

    # Rebuild with new data
    index.rebuild(np.array([[[1, 1], [2, 2]]], dtype=float))
    assert index._count == 1
    assert index.query_point(np.array([1.5, 1.5])) == [0]


def test_spatial_index_empty_intersection():
    """Verify intersection returns empty list when index is empty."""
    index = SpatialIndex(np.empty((0, 2, 2), dtype=float))
    assert index.intersection(np.array([[0, 0], [10, 10]])) == []


def test_spatial_index_query_point_approximate():
    """Verify query_point_approximate returns containing boxes."""
    boxes = np.array(
        [
            [[0, 0], [10, 10]],
        ],
        dtype=float,
    )
    index = SpatialIndex(boxes)
    assert index.query_point_approximate(np.array([5, 5])) == [0]
    assert index.query_point_approximate(np.array([15, 15])) == []


if __name__ == "__main__":
    test_spatial_index_query()
    test_spatial_index_update_box()
    test_spatial_index_incremental_ordering()
    test_spatial_index_position_mappings_sync()
    test_spatial_index_incremental_merge_trigger()
    test_spatial_index_empty_rebuild_edge_cases()
    test_spatial_index_empty_intersection()
    test_spatial_index_query_point_approximate()
    print("✅ All spatial index tests passed!")
