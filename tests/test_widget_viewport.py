"""Control panel 'Fit Current Viewport' on identity and scaled/translated images.

Regression tests for an earlier bug: the viewer's world point and ranges
were used as data coordinates, so a scaled/translated image raised a
ValueError (or silently put the box on another slice).
"""

import numpy as np
import pytest

from napari_turbobox.widget import BoundingBoxControlWidget

SHAPE = (40, 96, 96)
Z = 20
# Middle half of the displayed axes, one slice thick at Z (data coordinates).
EXPECTED = [[[Z, 24.0, 24.0], [Z + 1, 72.0, 72.0]]]


def _panel(viewer, scale, translate, shape=SHAPE):
    image = viewer.add_image(np.zeros(shape, np.uint8), scale=scale, translate=translate)
    widget = BoundingBoxControlWidget(viewer)
    widget._create_layer()  # copies the image's scale/translate
    return image, widget, widget._current_layer()


@pytest.mark.parametrize(
    ("scale", "translate"),
    [((1, 1, 1), (0, 0, 0)), ((2, 0.5, 0.5), (10, 5, 5)), ((4, 2, 3), (7, 11, 13))],
)
def test_fit_viewport_uses_data_coordinates(make_napari_viewer, scale, translate):
    viewer = make_napari_viewer()
    image, widget, layer = _panel(viewer, scale, translate)
    viewer.dims.set_point(0, image.data_to_world([Z, 0, 0])[0])
    widget._create_from_viewport()
    np.testing.assert_allclose(layer.bounding_boxes, EXPECTED)
    assert layer.nshapes == 1  # on the current slice
    widget.close()


def test_fit_viewport_reoriented_view(make_napari_viewer):
    viewer = make_napari_viewer()
    image, widget, layer = _panel(viewer, (2, 0.5, 0.5), (10, 5, 5))
    viewer.dims.order = (1, 0, 2)  # displays (z, x), y hidden
    viewer.dims.set_point(1, image.data_to_world([0, 30, 0])[1])
    widget._create_from_viewport()
    np.testing.assert_allclose(layer.bounding_boxes, [[[10.0, 30.0, 24.0], [30.0, 31.0, 72.0]]])
    assert layer.nshapes == 1
    widget.close()


def test_fit_viewport_invalid_box_reports_status(make_napari_viewer):
    viewer = make_napari_viewer()
    # A single-slice image cannot hold a box of the minimum extent 1 along z.
    _, widget, layer = _panel(viewer, (2, 0.5, 0.5), (10, 5, 5), shape=(1, 96, 96))
    widget._create_from_viewport()  # must not raise in the Qt slot
    assert len(layer.bounding_boxes) == 0
    assert "Could not fit the viewport" in widget.status_label.text()
    widget.close()
