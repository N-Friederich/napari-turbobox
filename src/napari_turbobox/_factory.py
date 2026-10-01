"""Create a TurboBox layer or the control panel for a viewer.

``create_control_panel`` is the widget command in ``napari.yaml``.
``create_nd_bbox_layer`` is a Python helper without a menu entry.
"""

from __future__ import annotations

import logging

from napari import Viewer, current_viewer
from napari.layers import Image

from .layer import TurboBoxLayer
from .widget import BoundingBoxControlWidget

logger = logging.getLogger(__name__)


def create_nd_bbox_layer(viewer: Viewer | None = None) -> TurboBoxLayer:
    """Create a TurboBoxLayer that fits an image layer and add it to the viewer.

    The image is the selected Image layer, or else the topmost one. The new
    layer takes the image's ndim, its shape (without an RGB(A) axis) as
    ``image_shape``, and its scale and translate, so the boxes are in that
    image's voxel coordinates. Without an image the layer has the viewer's
    ndim (at least 2) and no ``image_shape``, so its boxes are not bounded.

    Parameters
    ----------
    viewer : napari.Viewer, optional
        Viewer to add the layer to. Defaults to ``napari.current_viewer()``.
        If there is no viewer at all, a 3D layer is returned without being
        added anywhere.

    Returns
    -------
    TurboBoxLayer
        The new layer.
    """
    logger.debug("create_nd_bbox_layer() called")
    logger.debug("Input viewer: %s", viewer)
    v = viewer or current_viewer()
    logger.debug("Resolved viewer: %s", v)
    image_shape = None
    transform = {}
    ndim = 3
    if v is not None:
        logger.debug("Viewer has %d layers:", len(v.layers))
        for i, lyr in enumerate(v.layers):
            logger.debug("[%d] %s (type: %s)", i, lyr.name, type(lyr).__name__)
        logger.debug("Checking selected layers...")
        active_image = next(
            (layer for layer in v.layers.selection if isinstance(layer, Image)), None
        )
        logger.debug("From selection: %s", active_image)
        if active_image is None:
            logger.debug("No selected image, checking all layers in reverse...")
            active_image = next(
                (layer for layer in reversed(v.layers) if isinstance(layer, Image)), None
            )
            logger.debug("From reversed layers: %s", active_image)
        if active_image is not None:
            ndim = active_image.ndim
            logger.debug("Found image: %s", active_image.name)
            logger.debug("ndim: %d", ndim)
            logger.debug("data.shape: %s", active_image.data.shape)
            try:
                image_shape = tuple(active_image.data.shape[: active_image.ndim])  # drops an RGB(A) axis
                transform = {"scale": active_image.scale, "translate": active_image.translate}
                logger.debug("image_shape: %s", image_shape)
            except Exception as e:
                logger.warning("Failed to get image_shape: %s", e)
                image_shape = None
        else:
            ndim = max(2, getattr(v.dims, "ndim", 2))
            logger.debug("No image found, using viewer.dims.ndim: %d", ndim)
    logger.debug("Creating TurboBoxLayer...")
    logger.debug("ndim: %d", ndim)
    logger.debug("image_shape: %s", image_shape)
    try:
        layer = TurboBoxLayer(ndim=ndim, image_shape=image_shape, **transform)
        logger.debug("Layer created: %s", layer)
        logger.debug("Layer name: %s", layer.name)
        logger.debug("Layer ndim: %d", layer.ndim)
    except Exception as e:
        logger.exception("Error creating layer: %s", e)
        raise
    if v is not None:
        logger.debug("Adding layer to viewer...")
        logger.debug("Before: viewer has %d layers", len(v.layers))
        try:
            v.add_layer(layer)
            logger.debug("Layer added successfully!")
            logger.debug("After: viewer has %d layers", len(v.layers))
            logger.debug("Layer in viewer? %s", layer in v.layers)
        except Exception as e:
            logger.exception("Error adding layer to viewer: %s", e)
            raise
    else:
        logger.warning("No viewer available, returning layer without adding")
    logger.debug("Returning layer: %s", layer)
    return layer


def create_control_panel(viewer: Viewer | None = None):
    """Return the control panel for ``viewer``, or for the current viewer if it is None.

    This is the ``napari-turbobox.create_widget`` command of the plugin
    manifest. The panel works on the TurboBoxLayer picked in its combo box.
    """
    logger.debug("create_control_panel() called")
    logger.debug("Input viewer: %s", viewer)
    v = viewer or current_viewer()
    logger.debug("Resolved viewer: %s", v)
    return BoundingBoxControlWidget(v)
