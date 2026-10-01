"""Edits through ``layer.store`` and the minimum-extent rule."""

import numpy as np
import pytest

from napari_turbobox import TurboBoxLayer, create_synchronized_bbox_layers


def _drawn(layer):
    return sorted(np.round(np.asarray(p, float)[:, -2:].min(0), 3).tolist() for p in layer.data)


def test_store_edit_updates_a_standalone_layer(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((64, 64), np.uint8))
    layer = TurboBoxLayer(data=[[[5, 5], [15, 15]], [[40, 40], [50, 50]]], ndim=2, image_shape=(64, 64))
    viewer.add_layer(layer)
    layer.store.update_box(0, np.array([[25, 25], [35, 35]], float))
    assert _drawn(layer) == [[25.0, 25.0], [40.0, 40.0]]
    assert layer._spatial_index.query_point(np.array([30.0, 30.0])) == [0]
    layer.store.add_boxes(np.array([[[1, 1], [3, 3]]], float))
    assert layer.nshapes == 3
    layer.store.remove_boxes([1])
    assert _drawn(layer) == [[1.0, 1.0], [25.0, 25.0]]


def test_standalone_layer_api_still_consistent(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((64, 64), np.uint8))
    layer = TurboBoxLayer(ndim=2, image_shape=(64, 64))
    viewer.add_layer(layer)
    layer.add_boxes([[[5, 5], [15, 15]]])
    layer.bounding_boxes = [[[10, 10], [20, 20]], [[30, 30], [40, 40]]]
    assert _drawn(layer) == [[10.0, 10.0], [30.0, 30.0]]
    layer.undo()
    assert _drawn(layer) == [[5.0, 5.0]]
    layer.redo()
    layer.clear_boxes()
    assert layer.nshapes == 0 and len(layer.bounding_boxes) == 0


def test_store_edit_reaches_all_synchronized_layers(make_napari_viewer):
    viewers = [make_napari_viewer() for _ in range(2)]
    for v in viewers:
        v.add_image(np.zeros((20, 64, 64), np.uint8))
        v.dims.set_point(0, 10)
    layers = create_synchronized_bbox_layers(viewers[0], viewers[1:], image_shape=(20, 64, 64))
    layers[0].add_boxes([[[5, 5, 5], [15, 15, 15]]])
    layers[0].store.update_box(0, np.array([[5, 20, 20], [15, 30, 30]], float))
    for layer in layers:
        assert _drawn(layer) == [[20.0, 20.0]]


@pytest.mark.parametrize("lo", [7.9, 31.9, 1000.3, 7.0])
def test_unit_extent_survives_float32_storage(lo):
    layer = TurboBoxLayer(ndim=2, image_shape=(2048, 2048))
    layer.add_boxes([[[lo, lo], [lo + 1.0, lo + 1.0]]])
    layer.bounding_boxes = layer.bounding_boxes  # stored float32 values are accepted again
    assert len(layer.bounding_boxes) == 1


def test_extent_clearly_below_one_is_rejected():
    layer = TurboBoxLayer(ndim=2, image_shape=(64, 64))
    with pytest.raises(ValueError):
        layer.add_boxes([[[10, 10], [10.9, 20]]])
