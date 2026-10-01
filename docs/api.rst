API reference
=============

The public objects are importable from ``napari_turbobox``; the export
functions live in ``napari_turbobox.export``. Box arrays have the shape
``(N, 2, D)`` with ``D`` = 2 or 3 (see :doc:`architecture`).

.. currentmodule:: napari_turbobox

Multi-viewer layout
-------------------

.. autofunction:: create_synchronized_bbox_layers

Layer
-----

.. autoclass:: TurboBoxLayer
   :members: add_boxes, bounding_boxes, clear_boxes, sync_mode, undo, redo,
             can_undo, can_redo, clear_undo_history, undo_limit, cache_limit,
             get_undo_info, subscribe, unsubscribe
   :show-inheritance:

Shared store
------------

.. autoclass:: BBoxDataStore
   :members: data, update, update_box, update_boxes, add_boxes, remove_boxes,
             clear, register_listener, unregister_listener,
             set_listener_sync_mode, begin_edit_session, end_edit_session,
             in_edit_session, ndim, num_boxes

Control panel and factories
---------------------------

.. autoclass:: BoundingBoxControlWidget

.. autofunction:: create_control_panel

.. autofunction:: create_nd_bbox_layer

Export
------

.. autofunction:: napari_turbobox.export.boxes_to_coco

.. autofunction:: napari_turbobox.export.boxes_to_yolo

Deprecated
----------

* ``OptimizedBoundingBoxLayer``: alias of ``TurboBoxLayer`` that emits a
  ``DeprecationWarning``.
* ``sync_existing_layer_to_viewers``: emits a ``DeprecationWarning``; use
  ``create_synchronized_bbox_layers`` instead.
