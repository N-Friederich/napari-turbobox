"""Coordinate dtype: store + SpatialIndex are float32 (half the float64 memory,
exact for integer voxel indices up to 2**24). The round-trip exactness test is
the float16-regression guard: 4001 / 40001 are NOT float16-exact (float16 step
is 2 above 2048 and 32 above 32768), so it passes under float32/float64 and
fails under float16.
"""

import numpy as np

from napari_turbobox import BBoxDataStore
from napari_turbobox.export import boxes_to_coco
from napari_turbobox.spatial_index import SpatialIndex


def test_store_storage_is_float32():
    store = BBoxDataStore(initial_data=np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))
    assert store.data.dtype == np.float32
    store.add_boxes(np.array([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))
    assert store.data.dtype == np.float32  # add_boxes must not upcast


def test_index_storage_is_float32():
    idx = SpatialIndex(np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))
    assert idx._boxes.dtype == np.float32
    assert idx._sorted_mins[0].dtype == np.float32
    idx.add_boxes(np.array([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))
    assert idx._boxes.dtype == np.float32  # incremental add must not upcast


def test_large_integer_coord_roundtrips_exactly_store_index_export():
    for coord in (4001.0, 40001.0):
        box = np.array([[[10.0, 10.0, 10.0], [20.0, 20.0, coord]]])

        store = BBoxDataStore(initial_data=box)
        assert store.data[0, 1, 2] == coord, f"store corrupted coord {coord}"

        idx = SpatialIndex(store.data)
        assert idx._boxes[0, 1, 2] == coord, f"index corrupted coord {coord}"
        assert idx.query_point(np.array([15.0, 15.0, coord - 0.5])) == [0]

        coco = boxes_to_coco(store.data, image_shape=(50, 50, int(coord) + 10))
        assert coco["annotations"][0]["bbox"][2] == coord - 10.0, f"export corrupted {coord}"


def test_fractional_coords_preserved():
    # Powers-of-two fractions are exact in float32 -> mid-drag sub-pixel values
    # survive storage.
    data = np.array([[[2.5, 4.25, 8.125], [10.5, 20.25, 30.125]]])
    store = BBoxDataStore(initial_data=data)
    np.testing.assert_array_equal(store.data, data.astype(np.float32))
