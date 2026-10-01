Architecture
============

This page describes how napari-TurboBox stores boxes, how an edit reaches the
views, and which conventions the export follows. Module names refer to
``src/napari_turbobox/``.

Overview
--------

.. code-block:: text

   mouse drag / napari Add Rectangle / Delete / control panel / Python API
                                   |
                                   v
            editable TurboBoxLayer (main viewer; validates, clamps)
                                   |
                                   v
               BBoxDataStore: one (N, 2, D) float32 array
                                   |
               notifies every registered layer (listener)
                 |                                   |
                 v                                   v
      2D views ("live"):                 3D views ("on_commit"):
      updated on every change            deferred while a drag is in
                                         progress, updated once at its end
                 |                                   |
                 v                                   v
      per view: own SpatialIndex -> shapes of the boxes on the current slice
                (2D rectangles) or of all boxes (3D wireframes)

Box representation
------------------

* Boxes are stored as an ``(N, 2, D)`` array: ``box[0]`` holds the minimum and
  ``box[1]`` the maximum coordinate on each axis, in napari axis order
  (``(z, y, x)`` for 3D, ``(y, x)`` for 2D). ``D`` is 2 or 3.
* Coordinates are data (voxel index) coordinates of the image. The store keeps
  them as ``float32``, which represents every integer index up to 2**24 exactly.
* Box invariant: on every axis ``max - min >= 1``. When the layer knows the
  image shape, additionally ``0 <= min`` and ``max <= shape - 1``.
* The Python API (``add_boxes``, the ``bounding_boxes`` setter) rejects boxes
  that violate the invariant with a ``ValueError``; it does not clamp. Mouse
  edits and rectangles drawn with napari's *Add Rectangle* are clamped into the
  invariant instead.
* A box whose extent on a hidden (not displayed) axis is zero, for example a
  rectangle drawn on one slice, is extended over the full range of that axis
  (``0`` to ``shape - 1``) when the image shape is known.

Shared store (``data_store.py``)
--------------------------------

``BBoxDataStore`` owns the box array and notifies registered listeners after
each change.

* ``data`` returns a read-only view of the array; listeners receive the same
  kind of read-only view.
* ``update_box(i, box)`` writes one row in place (no copy of the array) and
  notifies the listeners with ``changed_idx=i``. ``update_boxes`` writes several
  rows in place and sends one notification without an index.
  ``update_box_silent`` writes one row without notifying anyone.
* ``add_boxes``, ``remove_boxes``, ``clear`` and assigning ``data`` replace the
  array and notify without an index.
* ``batch_update()`` is a context manager that defers notifications and sends a
  single notification (without an index) when it exits.
* Listeners are registered with ``register_listener(callback, sync_mode=...)``.
  A callback that has a ``changed_idx`` parameter receives the index of the
  changed box, or ``None`` when more than one box may have changed. Bound
  methods are held through weak references; other callables (functions,
  lambdas, ``functools.partial``) are held strongly and must be removed with
  ``unregister_listener``. An exception raised by one listener is logged and
  does not stop the notification of the others.
* Edit sessions: between ``begin_edit_session()`` and ``end_edit_session()``,
  listeners registered with ``sync_mode="on_commit"`` are not called. At the
  end of the session each of them that missed a change is called once, with the
  index of the changed box if all changes in the session concerned that box, and
  with ``None`` otherwise. Outside a session, ``on_commit`` listeners are called
  like ``live`` listeners. ``events.session_ended`` is emitted when a session
  ends. ``events.changed`` is emitted after every change outside a
  ``batch_update()`` block, and once when such a block exits.
* The store is not thread-safe. Mutate it from the Qt main thread; with Python
  assertions enabled, the first mutation from another thread emits a warning.

Layers and views (``layer.py``, ``multi_view.py``)
--------------------------------------------------

``TurboBoxLayer`` subclasses napari's ``Shapes`` layer. The store is the
authoritative state; the layer's shapes are a rendering of it.

* In a 2D view the layer draws one closed rectangle for each box that
  intersects the current slice. A box is on the slice when
  ``min - 1e-6 <= position <= max + 1e-6`` on every hidden axis, so boxes whose
  face lies exactly on a slice plane are drawn on that slice.
* In a 3D view of 3D data the layer draws one wireframe path (the 12 box edges)
  per box, for all boxes.
* The 2D shapes are ordered by box index: shape ``k`` shows the ``k``-th box on
  the slice. Finding the shape of a box is a binary search in that sorted list;
  finding the box of a shape is a lookup.

``create_synchronized_bbox_layers(main_viewer, sub_viewers, bbox_data=None,
image_shape=None, sync_mode="live", **layer_kwargs)`` creates one shared store
and one layer per viewer:

* The main viewer's layer is editable. The other layers are read-only: they are
  not editable in napari, start in pan/zoom mode, ignore mouse drags, and any
  change napari makes to their shapes is replaced by a new rendering from the
  store. The Python editing methods (``add_boxes``, ``clear_boxes``, the
  ``bounding_boxes`` setter) return without changing anything on a read-only
  layer.
* Each layer's listener mode is taken from its viewer's ``dims.ndisplay`` at
  creation: ``live`` for 2D viewers, ``on_commit`` for 3D viewers. It is not
  updated when a viewer is later switched between 2D and 3D; assign
  ``layer.sync_mode`` in that case. The function-level values ``"live"`` and
  ``"on_commit"`` give the same result; ``"snapshot"`` creates independent,
  unsynchronized layers, each with its own copy of ``bbox_data``.
* ``layer_kwargs`` are passed to every layer (for example ``edge_color``,
  ``edge_width``, ``scale``, ``translate``).

A layer created without a shared store (for example with the control panel's
*Add Bounding Box Layer*) owns a private store and updates itself directly.

Update path
-----------

When a layer is notified (``TurboBoxLayer._perform_sync``):

1. **Spatial index.** If one box changed and the number of boxes is unchanged,
   only that box is updated in the view's index. If boxes were appended (at
   most 10, or fewer than 20 % of the existing number), they are inserted into
   the index. In all other cases (removals, bulk changes) the index is rebuilt.
2. **Shapes.** If one box changed and the view (slice, displayed axes, 2D or
   3D) is the one of the last full rendering, only that box's shape is touched
   (``_incremental_shape_update``):

   * A box that stays on the slice of a 2D view, or any box in a 3D view, is
     edited in place. If its vertex and triangle counts and its triangulation
     are unchanged, only the shape's own rows of napari's vertex, mesh and
     displayed-vertex arrays are rewritten; finding the displayed rows is one
     NumPy pass over the displayed vertices. With the same counts but another
     triangulation the edit goes through napari's ``ShapeList.edit``.
   * A box that enters or leaves the slice of a 2D view gets its shape added
     at, or removed from, its place in the sorted order
     (``_add_or_remove_shape``). This requires shapes without a style of their
     own (one edge and face colour, z-index 0, the current edge width, no
     features, constant text), a layer with an image shape, and a view that is
     not empty before or after.
   * A box that is off the slice before and after needs no drawing.

   In every other case, including a change of the shape's vertex count (only a
   box with zero extent has one), the layer renders all its shapes again: the
   boxes on the current slice in a 2D view, all boxes in a 3D view. If napari
   raises an error during an in-place step, the error is passed on to the
   store, which logs it, and the next update renders all shapes again.
3. The layer emits ``events.bboxes``.

A slice change in napari also renders the view's shapes again.

Layer thumbnails: napari rasterizes a layer's thumbnail on every refresh.
napari-TurboBox skips this while a store edit session is active (and, for the
editing layer, during its own drag) and rasterizes each thumbnail once when the
session ends. Setting a layer's ``suppress_thumbnail_in_session`` attribute to
``False`` turns the session rule off for that layer (the editing layer still
skips thumbnails during its own drag).

Mouse editing
-------------

Only the editable layer reacts to mouse drags, and not in napari's pan/zoom or
*Add Rectangle* mode.

* **Press.** The press position is converted to data coordinates and tested
  against the boxes on the current slice (spatial-index point query). If boxes
  are nested, the one with the smallest area in the displayed plane is chosen.
  The drag resizes the box if Shift is held or if the press lies within 20 % of
  the box's displayed size from one of its edges; otherwise it moves the box.
  The press starts a store edit session.
* **Move.** Each mouse move computes the new box from the box at the start of
  the drag and writes it with ``update_box``. Moving changes only the displayed
  axes and, when the image shape is known, stops the box at the image border.
  Resizing moves, on each displayed axis, the side of the box nearer to the
  press position; the box keeps an extent of at least 1 and is clamped to the
  image. The 2D views, including the
  editing view, update the moved box immediately; 3D views wait.
* **Release.** The edit session ends, so every 3D view receives one
  notification. The editing layer stores an undo snapshot and rasterizes its
  thumbnail; it rebuilds its shapes only if its bookkeeping no longer matches
  the store (the moves already updated the dragged box in place).
* napari's own select-drag does not run while napari-TurboBox handles a drag,
  so the stored box stays axis-aligned. napari's transform mode and its
  ellipse, line, path, polygon and lasso tools are switched back to *Select*.

napari's editing tools are mapped onto the store:

* **Add Rectangle.** Each finished rectangle adds one box. A rectangle smaller
  than one voxel on a displayed axis (a click without drag) is dropped. The
  hidden axes are extended over their full range (when the image shape is
  known) and the box is clamped into the image. The shape napari drew is
  replaced by the rendering of the stored box.
* **Delete.** napari's *Delete* removes exactly the boxes behind the selected
  shapes from the store.
* Any other change napari makes to the shapes (for example vertex edits) is
  discarded by rendering the store again.

Spatial index (``spatial_index.py``)
------------------------------------

Every layer keeps its own ``SpatialIndex``. It holds a private ``float32`` copy
of the box coordinates and, for each axis, the box minima and maxima in sorted
order together with the sort permutation and its inverse.

* **Window query.** For an interval ``[lo, hi]`` on axis ``d``, a binary search
  selects the boxes whose minimum lies in ``[lo - e_d, hi]``, where ``e_d`` is
  the largest box extent along ``d`` (as tracked by the index; it is recomputed
  on a rebuild and never shrinks otherwise). Of these, the boxes with maximum
  ``>= lo`` are returned. The cost is ``O(log N + w)``, where ``w`` is the number
  of boxes in the searched window, a superset of the intersecting boxes. A
  single very deep box therefore widens the window of every query on that axis.
* **Slice query.** Only the hidden axes are constrained (one axis for a 3D
  volume shown as 2D slices), so a slice query is one window query; the result
  is sorted by box index.
* **Point query** (hit test of a mouse press). One window query per axis; the
  per-axis candidate sets are intersected.
* **Incremental maintenance.** Updating one box moves its entries between the
  old and new sorted positions; the cost is proportional to the number of
  entries shifted. Adding up to N/10 boxes merges them into the sorted arrays,
  more are sorted in; removing boxes filters the arrays.
* A property-based test (``tests/test_spatial_index_property.py``, hypothesis)
  checks that after random sequences of additions, removals and updates the
  incrementally maintained index answers point and slice queries like a newly
  built index.

Undo and redo (``undo.py``)
---------------------------

Each layer keeps its own ``UndoStack`` of snapshots of the full box array
(default 50, adjustable from 1 to 200 in the control panel). A snapshot is
stored after adding boxes, assigning ``bounding_boxes``, clearing, deleting with
napari's *Delete*, and at the end of every drag. Undo and redo write the
snapshot back through the ``bounding_boxes`` setter, so every view renders
again. Changes made directly on a ``BBoxDataStore`` are not recorded.

Notifications for applications (``event_manager.py``)
------------------------------------------------------

Besides ``layer.events.bboxes`` and the store's ``events.changed`` and
``events.session_ended``, a layer offers ``layer.subscribe(obj)``: ``obj``
receives ``obj.handle_event(event_type, payload)`` with ``event_type`` one of
``"added"``, ``"updated"`` and ``"cleared"``. ``BoundingBoxEventManager``
delivers these synchronously, in the calling thread. Its ``batch()`` context
manager queues the events and, when it exits, delivers the last event of each
type. The manager is not involved in synchronizing views; that runs through the
store's listeners.

Control panel (``widget.py``)
-----------------------------

The *BBox Control Panel* (``BoundingBoxControlWidget``) works on one box layer
of the viewer it was opened in:

* *Add Bounding Box Layer* creates a layer with the shape of the selected (or
  top-most) image layer, and with its scale and translation when the image has
  as many dimensions as the viewer.
* *Draw New BBox (Interactive)* selects the box layer and switches it to
  napari's *Add Rectangle* mode.
* *Fit Current Image Layer* adds a box from voxel 0 to ``shape - 1`` on every
  axis of the top-most image layer.
* *Fit Current Viewport* adds a box that covers the central half of the
  viewer's range on each displayed axis and is one slice thick at the current
  position on each hidden axis. It does not use the camera's field of view, and
  it takes the viewer's (world) coordinates as voxel indices, so for layers with
  a scale or translation the box does not match the view and can be rejected.
* *Remove All Bounding Boxes* clears the layer.
* The box list shows every stored box. *Select in Layer* selects the chosen box
  if it is on the current slice; *Delete Selected* deletes it. The list is
  refreshed after a drag ends, not on every mouse move.
* The status line shows the number of stored boxes and the number of shapes in
  the current view.
* *Export BBoxes...* writes native, COCO or YOLO files (next section).
  *Import BBoxes...* reads an ``(N, 2, D)`` array from ``.npy`` or ``.json``
  and appends it to the layer; the boxes are validated like any API write.
* *Advanced Settings* sets the undo history size and the path-cache size.

Export (``export.py``)
----------------------

* **Native.** ``.npy`` (``numpy.save`` of the ``(N, 2, D)`` array) or ``.json``
  (the same array as nested lists).
* **COCO and YOLO** (``boxes_to_coco``, ``boxes_to_yolo``) are 2D, per-image
  formats. With the default ``mode="per_slice"``, a 3D box yields one 2D box on
  every integer slice from ``ceil(z_min)`` to ``floor(z_max)``; a box thinner
  than one slice yields one box on its rounded middle slice. With
  ``mode="max_projection"`` (Python API only) each box yields one box on a
  single image. 2D boxes yield one box on a single image.
* Axes: COCO/YOLO ``x`` is the last axis and ``y`` the second-to-last. Width and
  height are ``max - min`` of the stored box (its drawn extent).
* COCO: one ``images`` entry per slice that has at least one box, with
  ``file_name`` ``slice_NNNN.png`` (``image.png`` for a single image), and
  ``bbox = [x, y, width, height]``. All boxes get category 1 (``"object"``)
  unless the Python API is given per-box class ids. The image files themselves
  are not written.
* YOLO: one ``slice_NNNN.txt`` per slice that has at least one box (``image.txt``
  for a single image), one line ``class x_center y_center width height`` per
  box, normalized by the image width and height; class 0 unless class ids are
  given.
* The image size comes from the layer's image shape, else from the first image
  layer of the viewer, else from the box extents (with a warning). Coordinates
  outside the image are clamped, with a warning.

Limitations
-----------

* 2 or 3 spatial axes; time and channel axes are not supported.
* The depth of a box (its extent along the hidden axis of the editing view)
  cannot be dragged in the read-only orthogonal views, and new boxes span the
  full depth. To set it with the mouse, reorient the editable viewer so that it
  displays the depth axis and drag the box there; or use the Python API or an
  import.
* The listener mode of a layer is fixed at creation (see above).
* Each view keeps its own spatial index with a copy of the box coordinates.
