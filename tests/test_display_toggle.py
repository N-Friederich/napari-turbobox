"""A box layer must survive switching the viewer between 2D and 3D display.

Regression test for an earlier bug: a 2D TurboBox layer (D = 2) raised on a
3D display toggle. A 2D layer keeps showing its rectangles in a 3D viewer, like
napari's Shapes; a 3D layer switches to wireframes and back.
"""

import napari
import numpy as np
import pytest
from packaging.version import Version

from napari_turbobox import TurboBoxLayer

# napari < 0.7 cannot set Shapes data while a 2D layer is shown in 3D; the layer
# then keeps its last drawing and catches up when the viewer returns to 2D.
REDRAWS_IN_3D = Version(napari.__version__) >= Version("0.7")


def _extent(layer):
    return [np.asarray(p, float).min(0).round(3).tolist() + np.asarray(p, float).max(0).round(3).tolist()
            for p in layer.data]


@pytest.mark.xfail(
    not REDRAWS_IN_3D, raises=ValueError, strict=True,
    reason="napari < 0.7 Shapes cannot return a 2D layer from 3D display (plain Shapes fails too)",
)
def test_2d_layer_toggle_3d_and_back(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((64, 64), np.uint8))
    layer = TurboBoxLayer(ndim=2, image_shape=(64, 64))
    viewer.add_layer(layer)
    layer.add_boxes([[[5, 5], [20, 30]], [[30, 30], [50, 60]]])
    shown_2d = _extent(layer)
    viewer.dims.ndisplay = 3
    assert layer.nshapes == 2
    assert _extent(layer) == shown_2d  # still the two rectangles (no exception)
    layer.add_boxes([[[40, 2], [60, 10]]])  # edits while in 3D display
    boxes = layer.bounding_boxes
    boxes[0] += 1
    layer.bounding_boxes = boxes
    layer.store.update_box(1, np.array([[31, 31], [51, 61]], float))
    if REDRAWS_IN_3D:
        assert layer.nshapes == 3
    viewer.dims.ndisplay = 2
    assert layer.nshapes == 3
    expected = [[b[0, 0], b[0, 1], b[1, 0], b[1, 1]] for b in np.asarray(layer.bounding_boxes, float)]
    assert sorted(_extent(layer)) == sorted(expected)


def test_3d_layer_toggle_shows_wireframes(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((20, 64, 64), np.uint8))
    layer = TurboBoxLayer(ndim=3, image_shape=(20, 64, 64))
    viewer.add_layer(layer)
    layer.add_boxes([[[2, 5, 5], [8, 20, 30]], [[12, 30, 30], [18, 50, 60]]])
    viewer.dims.set_point(0, 5)
    assert layer.nshapes == 1  # only the box on slice 5
    viewer.dims.ndisplay = 3
    assert layer.nshapes == 2  # one wireframe per box
    viewer.dims.ndisplay = 2
    assert layer.nshapes == 1
