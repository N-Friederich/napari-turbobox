napari-TurboBox
===============

napari-TurboBox (distribution ``napari-turbobox``, Python package
``napari_turbobox``) is a napari plugin for annotating axis-aligned bounding
boxes in 2D images and 3D volumes, in one viewer or in several synchronized
napari viewers. All box layers of a multi-viewer layout read one shared store
(``BBoxDataStore``). One viewer is used for editing; the other viewers show the
same boxes read-only, for example orthogonal XZ and YZ slices and a 3D view.
Boxes can be exported as NumPy/JSON arrays and as per-slice COCO and YOLO
annotations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   api
   contributing
   changelog

Features
--------

* **One store, several viewers.** ``create_synchronized_bbox_layers`` creates
  one editable box layer in the main viewer and one read-only box layer in each
  additional viewer, all attached to the same ``BBoxDataStore``.
* **Live orthogonal views.** While a box is dragged, 2D views follow on every
  mouse move. 3D views are updated once, when the drag ends; edits made outside
  a drag (adding, deleting, undo, API calls) reach 3D views immediately.
* **Single-box updates.** Editing one box updates, adds or removes only that
  box's shape in each view, when the checks of this incremental path pass.
  Changing the slice, bulk edits and the other cases rebuild a view's shapes
  (in a 2D view for the boxes on its current slice, in a 3D view for all
  boxes); see :doc:`architecture`.
* **Constrained mouse editing.** Drag inside a box to move it. Shift-drag, or
  start the drag near an edge, to resize it. Boxes stay axis-aligned with an
  extent of at least one voxel on every axis. When the layer knows the image
  shape, boxes stay inside the image and a moved box stops at the image border.
* **napari's own tools.** napari's *Add Rectangle* adds one box that spans the
  full range of the hidden axis. napari's *Delete* removes exactly the selected
  box(es). Read-only views cannot edit.
* **Undo/redo.** Ctrl+Z or Cmd+Z undoes, Ctrl+Y or Cmd+Shift+Z redoes (key
  bindings of the box layer, active while that layer is selected). On macOS,
  Cmd+Y is napari's own shortcut for switching between 2D and 3D display. The
  editable layer keeps up to 50 snapshots by default.
* **Control panel.** The *BBox Control Panel* widget adds a box layer, starts
  drawing, adds a box covering the whole image or the central half of the
  displayed axes, lists, selects and deletes boxes, and imports and exports
  them.
* **Export.** Native ``.npy`` / ``.json`` arrays, COCO JSON and YOLO label
  files. COCO and YOLO are 2D formats: a 3D box is written as one 2D box per
  image slice it covers.

Requirements
------------

* Python >= 3.10 (napari 0.8 and later require Python >= 3.11).
* napari >= 0.6.6 and < 0.10, NumPy, qtpy.
* A Qt binding: PyQt5, PyQt6 or PySide6 (accessed through qtpy).
* Tested with napari 0.6.6, 0.7.0 and 0.9.1.

napari does not install a Qt binding by itself. Its ``all`` extra installs one,
together with napari's optional compiled triangulation backend (``bermuda``),
which napari uses for Shapes layers when it is installed. napari-TurboBox does
not require that backend.

Installation
------------

Install napari-TurboBox together with napari and a Qt binding from PyPI:

.. code-block:: bash

   pip install "napari-turbobox[all]"

If napari with a Qt binding is already installed, ``pip install napari-turbobox``
is enough.

For development, install the package in editable mode with the development or
test dependencies (see :doc:`contributing`):

.. code-block:: bash

   pip install -e ".[dev]"       # tests, coverage, ruff, npe2, build (no Qt binding)
   pip install -e ".[testing]"   # pytest, pytest-qt, hypothesis only

Quick start in the napari GUI
-----------------------------

1. Start napari and open a 2D image or a 3D volume.
2. Open the control panel from napari's *Plugins* menu. The entry is
   *BBox Control Panel*, followed by the plugin's display name in parentheses.
3. Click *Add Bounding Box Layer*. The new layer takes its shape from the
   selected image layer (or, if no image layer is selected, from the top-most
   image layer), and also its scale and translation when the image has as many
   dimensions as the viewer.
4. Click *Draw New BBox (Interactive)* and drag a rectangle in the canvas. In a
   3D volume shown as 2D slices, the new box spans the full range of the hidden
   axis.
5. Switch to napari's *Select* tool to move a box (drag inside it) or to resize
   it (Shift-drag, or start the drag near an edge).
6. Export with *Export BBoxes...* (native, COCO or YOLO).

The control panel works on one box layer in the current viewer. A synchronized
layout over several viewers is created from Python.

Quick start in Python: four synchronized viewers
------------------------------------------------

.. code-block:: python

   import napari
   import numpy as np

   from napari_turbobox import create_synchronized_bbox_layers

   volume = np.random.random((60, 256, 256)).astype("float32")  # (Z, Y, X)
   titles = ("XY (edit)", "XZ", "YZ", "3D")
   viewers = [napari.Viewer(title=t) for t in titles]
   for viewer in viewers:
       viewer.add_image(volume)
   viewers[1].dims.order = (1, 0, 2)  # sliced along Y, shows Z and X
   viewers[2].dims.order = (2, 0, 1)  # sliced along X, shows Z and Y
   viewers[3].dims.ndisplay = 3       # set before creating the layers

   layers = create_synchronized_bbox_layers(
       main_viewer=viewers[0],
       sub_viewers=viewers[1:],
       image_shape=volume.shape,
   )
   # One box as [[z_min, y_min, x_min], [z_max, y_max, x_max]] (voxel indices).
   layers[0].add_boxes([[[10, 40, 40], [30, 90, 90]]])
   napari.run()

``layers[0]`` is the editable layer; ``layers[1:]`` are read-only. Each layer's
update mode is chosen once, when it is created, from its viewer's
``dims.ndisplay`` (2D: live, 3D: once per drag). If a viewer is switched between
2D and 3D afterwards, set ``layer.sync_mode`` by hand (see :doc:`architecture`).

Supported data and limitations
------------------------------

* Boxes have 2 or 3 spatial axes. Time and channel axes are not supported.
* Box coordinates are voxel indices of the image (data coordinates). Layers
  created from the control panel take the image layer's scale and translation
  (when the image has as many dimensions as the viewer), so anisotropic volumes
  are displayed in physical space.
* The depth of a box (its extent along the axis hidden in the editing view)
  cannot be dragged in the read-only orthogonal views, and a newly drawn box
  spans the full depth. To set it with the mouse, reorient the editable viewer
  so that it displays the depth axis (for example ``viewer.dims.order =
  (1, 0, 2)`` or napari's roll button) and drag or Shift-drag the box there;
  or set it through the Python API (``add_boxes``, ``bounding_boxes``) or by
  importing boxes.
* For 3D boxes, COCO files reference one image per slice
  (``slice_NNNN.png``) that the user provides; the images themselves are not
  written.
* The control panel and napari's tools edit the layer they are used on; a
  synchronized layout accepts edits only in the main viewer's layer.
