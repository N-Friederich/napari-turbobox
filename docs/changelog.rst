Changelog
=========

v0.1.0 (2026-10-01)
-------------------

First public release. The entries below compare it with the internal
development state of 2026-05-20, which was never published (last section).
The benchmark records in ``paper_results/`` name the version string 0.2.0 that
the measured code state carried in the development repository.

Added
~~~~~

* **Single-box view updates.** When one box changes, each view updates only
  that box: its entry in the view's spatial index and its shape.

  * A box that stays on a 2D view's slice, or any box in a 3D view, is edited
    in place. If its vertex and triangle counts and its triangulation stay the
    same, only the shape's own rows of napari's vertex and mesh arrays are
    rewritten; napari does not re-derive the other shapes.
  * A box that enters or leaves a 2D view's slice has its shape added or
    removed at its place in the view.
  * Slice changes, bulk edits (add, delete, import, undo) and the cases these
    paths do not cover rebuild the view: shapes with a style of their own
    (for example after *Move to front* or recolouring), feature columns or
    varying text, a layer without an image shape, an empty view, and a box
    with zero extent.
  * If napari raises an error during one of these in-place steps, the view is
    rebuilt at the next update instead of keeping stale shapes.
* **Live orthogonal views during mouse drags.** Every drag step is written to
  the shared store with ``update_box``, so all 2D views follow the dragged box
  while the mouse moves. 3D views are updated once, when the drag ends.
* **Read-only sub-views.** The layers that ``create_synchronized_bbox_layers``
  creates in the additional viewers are not editable in napari and start in
  pan/zoom mode; napari's tools cannot change the store through them. They
  stay read-only when napari's layer controls set ``editable`` (on creation
  and on every change of the display).
* **Per-listener update modes and edit sessions** in ``BBoxDataStore``:
  ``live`` and ``on_commit`` listeners, ``begin_edit_session`` /
  ``end_edit_session``, the ``in_edit_session`` property and the
  ``session_ended`` event. ``create_synchronized_bbox_layers`` gives layers in
  3D viewers the ``on_commit`` mode (decided once, at creation).
* **COCO JSON and YOLO export** (``napari_turbobox.export``: ``boxes_to_coco``,
  ``boxes_to_yolo``) with per-slice (default) and max-projection modes, and a
  format choice (native, COCO, YOLO) in the control panel's export dialog.
* **Spatial-index window query.** Slice and point queries search only the boxes
  whose minimum lies in the query interval or at most one maximum box extent
  below it (``O(log N + w)`` per axis, ``w`` = boxes in that window) instead of
  intersecting candidate lists over all boxes.
* A property-based test (hypothesis) that compares the incrementally maintained
  spatial index with a rebuilt one.
* A ``testing`` extra (pytest, pytest-qt, hypothesis) and ``CITATION.cff``.

Changed
~~~~~~~

* Licensed under the Apache License 2.0 (was: MIT); see ``LICENSE`` and
  ``NOTICE``.
* Requires napari >= 0.6.6 and < 0.10 and Python >= 3.10 (was: any napari,
  Python >= 3.9).
* **Sorted box-to-shape mapping.** The shapes of a 2D view are ordered by box
  index, so the shape of a box is found by binary search instead of a scan over
  all boxes.
* **Thumbnails are skipped inside edit sessions.** Every layer that shares the
  store skips napari's thumbnail rasterization while a drag is in progress and
  rasterizes it once when the drag ends.
* The store and the spatial index keep coordinates as ``float32`` (was
  ``float64``).
* 3D views draw one wireframe path per box instead of six face loops.
* Box invariant: every box has an extent of at least 1 on every axis and, when
  the image shape is known, lies inside the image. API writes that violate it
  raise ``ValueError``; mouse edits and drawn rectangles are clamped.
* When boxes are nested, a click selects the innermost box (smallest area in the
  displayed plane).
* The documentation was rewritten to describe the current code. It now also
  states what boxes do not carry: there is no class label (the control panel
  exports all boxes as one class; the Python converters take per-box classes
  as ``category_ids``), and only the native ``.npy``/``.json`` format can be
  imported, not COCO or YOLO.
* The release workflow publishes to PyPI only after the lint job and the full
  test matrix have passed for the tag.

Fixed
~~~~~

* **Data loss with napari's Add Rectangle and Delete.** Any shape change in
  napari used to rebuild the store from the shapes of the current slice, which
  deleted every box not visible on that slice. *Add Rectangle* now adds exactly
  one box, *Delete* removes exactly the selected box(es), and other napari shape
  edits are discarded.
* Every rectangle drawn with *Add Rectangle* is stored; rectangles after the
  first one could previously be dropped.
* A box moved against the image border stops there instead of being shrunk by
  the clamp.
* The layer's scale and translation are respected: mouse positions and the
  slice position are converted to data coordinates, and layers created with
  ``create_nd_bbox_layer`` (or from the control panel, when the image has as
  many dimensions as the viewer) take the image layer's scale and translation.
* napari's own select handler no longer moves the displayed shape while
  napari-TurboBox handles a drag.
* A box can be dragged more than once; the drag state is reset when a drag ends
  (previously only the first drag worked).
* The spatial index keeps its own copy of the box coordinates. It used to share
  the store's array, so the in-place single-box update could leave it stale.
* The spatial index is updated incrementally only when boxes were appended;
  re-inserted boxes (for example undoing a delete) now trigger a rebuild instead
  of leaving a stale index.
* Assigning ``bounding_boxes`` (also used by the control panel's *Delete
  Selected*) stores an undo snapshot.
* The undo label in the control panel and the settings widget
  (``get_undo_description``, ``undo_description``) names the action that
  *Undo* reverts; it showed the action before that one.
* Store listeners that are not bound methods (functions, lambdas,
  ``functools.partial``) are held by strong references. They were held by weak
  references before, so a listener created inline was collected at once and
  never called.
* Control panel:

  * *Draw New BBox* switches the layer to napari's *Add Rectangle* mode; the
    former viewer-level mouse callback never added a box with napari >= 0.6.
  * *Fit Current Image Layer* ends the box at the last voxel index
    (``shape - 1``), so the box is accepted, and ignores the channel axis of RGB
    images (also when adding a layer).
  * *Fit Current Viewport* reads the current position of each hidden axis by
    axis index and creates a valid box one slice thick.
  * The box list is refreshed when a drag ends instead of on every mouse move.
  * The status line counts the stored boxes (it showed the number of shapes on
    the current view) and maps a selected shape to the right box.

Internal development state (2026-05-20, not published)
------------------------------------------------------

* Renamed the package to ``napari-turbobox`` (module ``napari_turbobox``) and
  the layer class to ``TurboBoxLayer``; ``OptimizedBoundingBoxLayer`` remains
  as a deprecated alias.
* The store warns (once) when it is mutated from a thread other than the Qt
  main thread.
* Spatial index on sorted per-axis arrays with inverse position maps, so a
  single-box update moves only the entries between the old and the new sorted
  position instead of re-sorting.
* ``BoundingBoxEventManager``: synchronous notifications for layer subscribers
  with a ``batch()`` context manager that delivers the last event of each type.
* Undo/redo based on snapshots of the full box array.
* Tests ``test_undo.py``, ``test_geometry.py``, ``test_event_manager.py``,
  ``test_multi_view_additional.py`` and ``test_layer_additional.py``. Coverage
  measured at that state: 66 % overall; spatial_index 96 %, data_store 93 %,
  geometry 99 %, event_manager 100 %, multi_view 98 %, undo 100 %, layer 61 %,
  widget 41 %.
* A GitHub Actions test workflow, a ``pre-commit`` configuration and a Sphinx
  documentation configuration in ``docs/`` (no automated documentation build).
