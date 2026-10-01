"""
napari-TurboBox
===============

Bounding-box annotation in napari, synchronized across several viewers. An
npe2 plugin for napari 0.6.6 up to (not including) 0.10 on Python 3.10 or
later. Boxes are axis-aligned and have 2 or 3 spatial axes.

Features
--------
- TurboBoxLayer: a napari Shapes layer that draws the boxes of a
  ``BBoxDataStore``, as slice rectangles in 2D views and one wireframe per
  box in 3D.
- Multi-viewer sync: ``create_synchronized_bbox_layers`` puts an editable
  layer and read-only layers in other viewers on one shared store.
- Editing: drag a box to move it. Shift+drag, or a drag that starts near an
  edge, resizes it.
- Slicing: a 2D view shows the boxes that cross its current slice.
- Control panel: add, list, delete, import and export boxes (native
  .npy/.json, per-slice COCO JSON, per-slice YOLO).

Usage
-----
Synchronized layers in several viewers::

    from napari_turbobox import create_synchronized_bbox_layers

    layers = create_synchronized_bbox_layers(
        main_viewer, [viewer_xz, viewer_yz, viewer_3d], image_shape=(10, 512, 512)
    )
    layers[0].add_boxes([[[z_min, y_min, x_min], [z_max, y_max, x_max]]])

One layer in the current viewer (shape, scale and translation are taken from
an image layer)::

    from napari_turbobox import create_nd_bbox_layer

    layer = create_nd_bbox_layer(viewer)

Or open the control panel from napari's Plugins menu:
"BBox Control Panel (napari-TurboBox)".

Components
----------
TurboBoxLayer
    napari Shapes layer that shows and edits the boxes of one store

BBoxDataStore
    Holds the (N, 2, D) box array and notifies every layer that uses it

create_synchronized_bbox_layers
    One editable and several read-only layers on one shared store

create_nd_bbox_layer
    One layer in a viewer, fitted to an image layer

create_control_panel
    The control panel for a viewer (the plugin's dock widget)

BoundingBoxControlWidget
    The control panel class

OptimizedBoundingBoxLayer, sync_existing_layer_to_viewers
    Deprecated. Use TurboBoxLayer and create_synchronized_bbox_layers
"""

import warnings

from ._factory import create_control_panel, create_nd_bbox_layer
from .data_store import BBoxDataStore
from .layer import TurboBoxLayer
from .multi_view import create_synchronized_bbox_layers, sync_existing_layer_to_viewers
from .widget import BoundingBoxControlWidget


# Old name of TurboBoxLayer, kept so that existing code still runs.
class OptimizedBoundingBoxLayer(TurboBoxLayer):
    """Deprecated alias of TurboBoxLayer. Creating one emits a DeprecationWarning."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OptimizedBoundingBoxLayer is deprecated and will be removed in a future version. "
            "Please use TurboBoxLayer instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "BBoxDataStore",
    "BoundingBoxControlWidget",
    "OptimizedBoundingBoxLayer",
    "TurboBoxLayer",
    "create_control_panel",
    "create_nd_bbox_layer",
    "create_synchronized_bbox_layers",
    "sync_existing_layer_to_viewers",
]
