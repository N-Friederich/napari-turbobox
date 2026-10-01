"""TurboBoxLayer: a napari Shapes layer that draws axis-aligned 2D and 3D bounding boxes."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np
from napari.layers import Shapes
from napari.utils.events import Event

from .data_store import BBoxDataStore
from .event_manager import BoundingBoxEventManager
from .geometry import (
    bbox_from_path,
    ensure_bbox_array,
    extrude_2d_to_3d_box,
    path_from_bbox,
    paths_from_bboxes,
    slice_paths_from_bbox,
    slice_paths_from_bboxes,
    wireframe_path_from_bbox,
)
from .spatial_index import SpatialIndex
from .undo import UndoStack

logger = logging.getLogger(__name__)


def _select_unless_bbox_drag(layer, event):
    """Run napari's select drag, unless TurboBox is already dragging a box.

    ``TurboBoxLayer._on_mouse_drag`` sits first in the drag callbacks and sets
    ``_drag_state`` when the press hits a box. napari does not check
    ``event.handled``, so without this its ``select`` would move the drawn
    shape as well.
    """
    if getattr(layer, "_drag_state", None) is not None:
        return
    from napari.layers.shapes._shapes_mouse_bindings import select

    yield from select(layer, event)


def _move_last_shape(dv, pos: int) -> None:
    """Move the last shape of napari's ShapeList ``dv`` to index ``pos``.

    Each shape owns one contiguous block of rows in the vertex, mesh-vertex and
    triangle arrays: ``*_index[i]`` is where the block of shape ``i`` starts and
    ``*_index[i + 1]`` where it ends. Moving the last shape to ``pos`` rotates
    the rows from block ``pos`` on, shifts the block starts behind it, and
    renumbers the mesh-vertex indices stored in the triangles. The shape's
    z-index and colour rows move with it. The caller re-derives the z-order and
    the displayed arrays with napari's ``_update_z_order``.
    """
    last = len(dv.shapes) - 1
    if pos == last:
        return
    mesh = dv._mesh

    def move_rows(index, *arrays):
        """Rotate the rows of ``arrays``.

        Returns the new block starts, the rotated arrays and the old
        ``(start, moved, end)`` row positions.
        """
        start, moved, end = int(index[pos]), int(index[last]), int(index[last + 1])
        rotated = [
            np.concatenate([a[:start], a[moved:end], a[start:moved], a[end:]]) for a in arrays
        ]
        new_index = index.copy()
        new_index[pos + 1 : last + 1] = index[pos:last] + (end - moved)
        return new_index, rotated, (start, moved, end)

    dv.shapes.insert(pos, dv.shapes.pop())
    dv._z_index, dv._edge_color, dv._face_color = (
        np.concatenate([a[:pos], a[last:], a[pos:last]])
        for a in (dv._z_index, dv._edge_color, dv._face_color)
    )
    dv._vertices_index, (dv._vertices,), _ = move_rows(dv._vertices_index, dv._vertices)
    mesh.vertices_index, rotated, (start, moved, end) = move_rows(
        mesh.vertices_index, mesh.vertices, mesh.vertices_centers, mesh.vertices_offsets
    )
    mesh.vertices, mesh.vertices_centers, mesh.vertices_offsets = rotated
    # New position of each old mesh-vertex row, used to renumber the triangles.
    dtype = mesh.triangles.dtype
    new_row = np.concatenate([
        np.arange(start, dtype=dtype),
        np.arange(start + end - moved, end, dtype=dtype),
        np.arange(start, start + end - moved, dtype=dtype),
        np.arange(end, len(mesh.vertices), dtype=dtype),
    ])
    mesh.triangles_index, (triangles, mesh.triangles_colors), _ = move_rows(
        mesh.triangles_index, mesh.triangles, mesh.triangles_colors
    )
    mesh.triangles = new_row[triangles]


class TurboBoxLayer(Shapes):
    """Axis-aligned bounding boxes in 2D images and 3D volumes, drawn as napari shapes.

    The boxes are kept in a :class:`BBoxDataStore` as an ``(N, 2, ndim)``
    array of ``[mins, maxs]`` pairs (``ndim`` is 2 or 3); the layer's shapes
    are only a drawing of that array. In a 2D view the layer draws one
    rectangle for each box that intersects the current slice, in a 3D view of
    3D data one wireframe path per box. Several layers, for example in XY, XZ,
    YZ and 3D viewers, can share one store; an edit in one of them then shows
    up in all of them.

    Attributes:
        layer_type (str): Layer type identifier, ``"turbobox"``.
        bounding_boxes (np.ndarray): Copy of the boxes, shape (N, 2, ndim).
    """

    layer_type = "turbobox"
    _type_string = "turbobox"

    def __init__(
        self,
        data: Iterable[Sequence[Sequence[float]]] | np.ndarray | None = None,
        *,
        ndim: int | None = None,
        image_shape: Sequence[int] | None = None,
        name: str | None = "Bounding Boxes",
        edge_color: str | Sequence[str] = "cyan",
        edge_width: float = 2.0,
        face_color: str = "transparent",
        auto_extrude: bool = False,
        bbox_data_store: BBoxDataStore | None = None,
        interactive: bool = True,
        sync_mode: Literal["live", "on_commit"] | None = None,
        **kwargs,
    ) -> None:
        """Create the layer and subscribe it to its store.

        Args:
            data: Initial boxes, or None for an empty layer. Either an array of
                shape (N, 2, ndim) or an iterable of ``[[mins], [maxs]]`` pairs.
                The two corners may be given in any order; each axis is sorted
                into min and max. These boxes are not validated (no
                ``image_shape`` or extent check).
            ndim: Number of axes, 2 or 3. If None, it is taken from ``data``,
                or 3 for an empty layer.
            image_shape: Shape of the image. If given, ``add_boxes`` and the
                ``bounding_boxes`` setter reject boxes outside
                ``[0, image_shape - 1]`` with ValueError, mouse edits are clamped
                to that range, and the layer extent covers the whole image.
            name: Layer name shown in napari's layer list.
            edge_color: Colour of the box outlines: a colour name ("cyan"), a
                hex code or an RGB(A) tuple.
            edge_width: Line width of the box outlines, passed on to napari's
                Shapes layer.
            face_color: Fill colour of the boxes. The default "transparent"
                keeps the image visible.
            auto_extrude: If True, a rectangle drawn with Add Rectangle is
                extruded along the hidden axis before it is stored
                (``extrude_2d_to_3d_box``). A flat hidden axis is extended in
                any case (``_expand_hidden_axes``): with ``image_shape`` both
                settings give ``[0, image_shape - 1]``. Without ``image_shape``,
                a rectangle drawn at slice ``z`` gets ``[0, max(z, 1)]`` with
                this option and ``[0, z + 1]`` without.
            bbox_data_store: Store to display. If None, the layer creates a
                private store. Pass the same BBoxDataStore to several layers to
                keep them in sync; boxes given in ``data`` are appended to it.
            interactive: If False, the layer is read-only: napari cannot edit
                it, mouse drags are ignored, and ``add_boxes``, ``clear_boxes``
                and the ``bounding_boxes`` setter only log a warning. Changes
                made through the store still show up. In a synchronized layout
                only one layer is interactive.
            sync_mode: How the layer follows its store. "live" updates the
                layer on every store change; "on_commit" holds back changes
                made during a store edit session (a mouse drag) until the
                session ends. None means "live". The layer also listens to a
                private store, so this applies there too.
                ``create_synchronized_bbox_layers`` passes "on_commit" for
                layers in 3D viewers.
            **kwargs: Passed on to napari's Shapes layer (for example ``scale``
                or ``translate``).

        Examples:
            Create layer with initial boxes::

                layer = TurboBoxLayer(
                    data=[[[10, 20, 20], [50, 80, 80]]],
                    ndim=3,
                    image_shape=(100, 512, 512),
                    edge_color="cyan",
                    edge_width=3.0
                )

            Create synchronized layers for multi-viewer::

                from napari_turbobox import BBoxDataStore

                shared_store = BBoxDataStore()
                layer1 = TurboBoxLayer(
                    bbox_data_store=shared_store,
                    interactive=True  # Main viewer
                )
                layer2 = TurboBoxLayer(
                    bbox_data_store=shared_store,
                    interactive=False  # Read-only display
                )

        Notes:
            - Boxes are always axis-aligned. napari's transform mode is
              switched back to Select.
            - With ``ndisplay == 2`` the layer draws one rectangle per box on
              the current slice; with ``ndisplay == 3`` one wireframe path per
              box (a 2D layer keeps its rectangles, as napari's Shapes does).
            - The drawing follows the viewer's slice and ``ndisplay`` by
              itself. ``sync_mode`` does not; see the ``sync_mode`` property.
        """
        boxes = ensure_bbox_array(data, ndim)
        ndim = boxes.shape[-1] if boxes.size else ndim or 3
        if bbox_data_store is None:
            self._bbox_store = BBoxDataStore(initial_data=boxes)
            self._owns_store = True
        else:
            self._bbox_store = bbox_data_store
            self._owns_store = False
            if boxes.size > 0:
                self._bbox_store.add_boxes(boxes)
        logger.debug(
            "__init__: Creating layer with%sboxes (shared_store=%s, interactive=%s)",
            len(self._bbox_store.data),
            not self._owns_store,
            interactive,
        )
        logger.debug("__init__: edge_width=%s, edge_color=%s", edge_width, edge_color)
        super().__init__(
            data=[],
            ndim=ndim,
            shape_type="path",
            edge_color=edge_color,
            edge_width=edge_width,
            face_color=face_color,
            name=name,
            **kwargs,
        )
        self._force_sync = True
        self.edge_width = edge_width
        self.current_edge_width = edge_width
        logger.debug("__init__: After super().__init__, self.edge_width =%s", self.edge_width)
        self._image_shape = self._normalize_image_shape(image_shape, ndim)
        self._interactive = interactive
        self._pending_remove = None
        self._napari_adding = False  # between napari's 'adding' and 'added' events
        self._reset_editable()  # read-only: editable=False, and napari then switches to PAN_ZOOM
        self._sync_guard = False
        self._event_manager = BoundingBoxEventManager()
        self._drag_state = None
        self._session_active = False
        self._auto_extrude = auto_extrude
        self._last_bbox_count = 0
        self._bbox_is_moving = False
        self._just_finished_drag = False
        self._drag_end_time = 0.0
        self._sync_mode: Literal["live", "on_commit"] = sync_mode or "live"
        # Every layer listens to its store, a private one included, so writes
        # through ``layer.store`` are drawn the same way as the layer's own edits.
        self._bbox_store.register_listener(self._on_shared_data_changed, sync_mode=self._sync_mode)
        self.events.add(bboxes=Event)
        self.events.data.connect(self._on_shapes_data_changed)
        self.mouse_drag_callbacks.insert(0, self._on_mouse_drag)
        self._allow_thumbnail_update = True
        # The thumbnail is not rasterized during a store edit session (see
        # _update_thumbnail); draw it once when the session ends.
        self._bbox_store.events.session_ended.connect(self._on_store_session_ended)
        self._moving_coordinates = None
        self.events.mode.connect(self._on_mode_change)
        self._dims_connected = False
        if not hasattr(self, "_value") or self._value is None:
            self._value = (None, None)
        if not hasattr(self, "_moving_value"):
            self._moving_value = (None, None)
        if not hasattr(self, "_drag_box"):
            self._drag_box = None
        if not hasattr(self, "_drag_box_stored"):
            self._drag_box_stored = None
        if not hasattr(self, "_value_stored"):
            self._value_stored = (None, None)
        self.current_edge_width = edge_width
        self.edge_width = edge_width
        self._spatial_index = SpatialIndex(self._bbox_store.data)
        self._path_cache = {}
        self._cache_max_size = 10000
        self._undo_stack = UndoStack(max_size=50)
        self._undo_stack.push(self._bbox_store.data.copy(), "Initial state")
        if self._bbox_store.data.size > 0:
            self._sync_shapes_from_bboxes()

    def _post_init(self) -> None:
        """Finish setup; napari calls this right after ``__init__`` returns.

        Draws the boxes, sets Select mode (pan/zoom for a read-only layer) and
        binds the undo keys: Ctrl+Z / Cmd+Z undo, Ctrl+Y / Cmd+Shift+Z redo.
        """
        logger.debug("_post_init() called")
        super()._post_init()
        self._dims_connected = True
        self._sync_shapes_from_bboxes()
        from napari.layers.shapes._shapes_constants import Mode

        self.mode = Mode.SELECT if self._interactive else Mode.PAN_ZOOM
        self.bind_key("Control-Z", self.undo, overwrite=True)
        self.bind_key("Control-Y", self.redo, overwrite=True)
        self.bind_key("Command-Z", self.undo, overwrite=True)
        self.bind_key("Command-Shift-Z", self.redo, overwrite=True)
        logger.info("Undo/Redo keyboard shortcuts registered (Ctrl+Z, Ctrl+Y)")
        logger.info("_post_init() completed, mode set to SELECT")
        logger.debug("Click anywhere inside a bbox to move it")
        logger.debug("Click near an edge/corner to resize it (or hold Shift anywhere)")

    @property
    def _slice_input(self):
        """napari's slice input, or ``_mock_slice_input`` while that is set.

        Assigning sets ``_mock_slice_input`` (tests use this);
        ``_safe_set_shape_data`` sets it briefly to cap ``ndisplay``.
        """
        if hasattr(self, "_mock_slice_input"):
            return self._mock_slice_input
        return super()._slice_input

    @_slice_input.setter
    def _slice_input(self, value):
        self._mock_slice_input = value

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of the layer in data coordinates.

        With ``image_shape`` this is ``[[0, ..., 0], image_shape]``, so the
        viewer's sliders cover the whole image. Without it napari's Shapes
        extent is used, which covers only the shapes currently drawn (in a 2D
        view, the boxes on the current slice).
        """
        if self._image_shape is not None:
            mins = np.zeros(self.ndim)
            maxs = np.asarray(self._image_shape, dtype=float)
            return np.stack([mins, maxs], axis=0)
        return super()._extent_data

    @property
    def sync_mode(self) -> Literal["live", "on_commit"]:
        """How this layer follows its store: ``"live"`` or ``"on_commit"``.

        ``"live"`` updates the layer on every store change. ``"on_commit"``
        holds back changes made during a store edit session (see
        :meth:`BBoxDataStore.begin_edit_session`) and updates the layer once
        when the session ends.

        Notes
        -----
        The mode is set when the layer is created;
        ``create_synchronized_bbox_layers`` takes it from the viewer's
        ``ndisplay`` (``"on_commit"`` for 3D viewers). It is **not** changed
        when the viewer later switches between 2D and 3D display. Assign this
        property in that case.
        """
        return self._sync_mode

    @sync_mode.setter
    def sync_mode(self, value: Literal["live", "on_commit"]) -> None:
        """Change how the layer follows its store (see the getter)."""
        self._sync_mode = value
        self._bbox_store.set_listener_sync_mode(self._on_shared_data_changed, value)
        logger.debug("Layer sync_mode changed to %r", value)

    @property
    def store(self) -> BBoxDataStore:
        """The :class:`BBoxDataStore` this layer displays (shared by synchronized layers).

        Writes to the store (for example ``store.update_box``) update every layer
        that displays it. They are not validated and not recorded for undo; use
        the ``bounding_boxes`` setter or ``add_boxes`` for that.
        """
        return self._bbox_store

    @property
    def bounding_boxes(self) -> np.ndarray:
        """Copy of the boxes, shape (N, 2, ndim).

        Assigning replaces all boxes. A flat hidden axis is first extended
        (``_expand_hidden_axes``), then every box is checked
        (``_validate_box``); an invalid box raises ValueError and nothing
        changes. One undo step, "Set boxes", is recorded. On a read-only layer
        the assignment only logs a warning.
        """
        return self._bbox_store.data.copy()

    @bounding_boxes.setter
    def bounding_boxes(self, value: Iterable[Sequence[Sequence[float]]] | np.ndarray) -> None:
        if not self._interactive:
            logger.warning(
                "Layer is in read-only mode (interactive=False), ignoring bbox modification"
            )
            return
        boxes = ensure_bbox_array(value, self.ndim)
        if boxes.size:
            boxes = np.asarray([self._expand_hidden_axes(b) for b in boxes], dtype=float)
            self._validate_box(boxes)
        self._bbox_store.data = boxes  # the store notifies this layer and all others
        if hasattr(self, "_undo_stack"):
            # Does nothing during undo/redo, which pause the stack.
            self._undo_stack.push(self.bounding_boxes.copy(), "Set boxes")

    def add_boxes(self, boxes: Iterable[Sequence[Sequence[float]]] | np.ndarray) -> None:
        """Append boxes to the store and record one undo step.

        ``boxes`` is one ``[mins, maxs]`` pair or an array of shape
        (N, 2, ndim). A flat hidden axis is first extended
        (``_expand_hidden_axes``), then every box is checked
        (``_validate_box``); an invalid box raises ValueError and nothing is
        added. Sends ``events.bboxes`` and an ``"added"`` event to
        subscribers. On a read-only layer it only logs a warning.
        """
        if not self._interactive:
            logger.warning("Layer is in read-only mode (interactive=False), ignoring add_boxes")
            return
        logger.debug("add_boxes() called with boxes:%s", boxes)
        logger.debug("self.ndim =%s", self.ndim)
        new_boxes = ensure_bbox_array(boxes, self.ndim)
        logger.debug("After ensure_bbox_array: new_boxes.shape =%s", new_boxes.shape)
        logger.debug("new_boxes =%s", new_boxes)
        if new_boxes.size == 0:
            logger.warning("new_boxes is empty, returning")
            return
        if new_boxes.size:
            new_boxes = np.asarray([self._expand_hidden_axes(b) for b in new_boxes], dtype=float)
        self._validate_box(new_boxes)
        self._bbox_store.add_boxes(new_boxes)
        logger.debug(
            "Before _sync_shapes_from_bboxes(), _bbox_data.shape =%s", self._bbox_store.data.shape
        )
        logger.debug("current_edge_width =%s", self.current_edge_width)
        self._spatial_index = SpatialIndex(self.bounding_boxes)
        self._sync_shapes_from_bboxes()
        if self.nshapes > 0 and hasattr(self, "current_edge_width"):
            logger.debug("Forcing edge_width to%s", self.current_edge_width)
            self.edge_width = self.current_edge_width
            logger.debug("After forcing: edge_width =%s", self.edge_width)
        logger.info("_sync_shapes_from_bboxes() completed")
        if hasattr(self, "_undo_stack"):
            n_added = len(new_boxes)
            self._undo_stack.push(
                self.bounding_boxes.copy(), f"Add {n_added} box{('es' if n_added > 1 else '')}"
            )
        self.events.bboxes(value=self.bounding_boxes)
        self._event_manager.emit("added", self.bounding_boxes)

    def _safe_set_shape_data(
        self, paths: list[np.ndarray], shape_type: str | None = None
    ) -> bool:
        """Write ``paths`` to the Shapes data; return False if napari refused them.

        An empty ``paths`` is written as an empty ``(0, 2, ndim)`` array. With
        ``shape_type``, every path is tagged with it so napari builds shapes of
        that type. The 3D wireframe must be ``'path'``: it is a non-planar
        polyline that touches itself and would be triangulated wrongly as a
        polygon. Without ``shape_type`` napari reuses the layer's current
        shape types.

        napari's data setter sizes the new shapes and the data view by the
        viewer's ``ndisplay``, while ``Shapes._set_view_slice`` keeps the data
        view at ``min(ndim, ndisplay)``, so any data assignment to a 2D layer
        in a 3D viewer raises (plain Shapes does the same). The setter
        therefore sees ``ndisplay`` capped at the layer's ndim. napari < 0.7
        ignores that cap; its ValueError is then caught, a warning is logged
        and False is returned.
        """
        slice_input = self._slice_input
        capped = not hasattr(self, "_mock_slice_input") and slice_input.ndisplay > self.ndim
        if capped:
            self._mock_slice_input = dataclasses.replace(slice_input, ndisplay=self.ndim)
        try:
            if not paths:
                self.data = np.empty((0, 2, self.ndim), dtype=float)
            elif shape_type is not None:
                self.data = [(np.asarray(p, dtype=float), shape_type) for p in paths]
            else:
                self.data = paths
        except ValueError:
            if not capped:
                raise
            # napari < 0.7 ignores _mock_slice_input and cannot set Shapes data
            # while a lower-dimensional layer is shown in 3D; the view is
            # redrawn when the viewer returns to 2D display.
            logger.warning("2D box layer not redrawn while the viewer shows 3D (napari < 0.7)")
            return False
        finally:
            if capped:
                del self._mock_slice_input
                # Leave the layer as napari does after a display toggle: store the
                # viewer's ndisplay, and make the layer editable only in 2D display.
                self._ndisplay_stored = slice_input.ndisplay
                self._reset_editable()
        return True

    def clear_boxes(self) -> None:
        """Remove all boxes and record one undo step.

        Does nothing if there are no boxes. On a read-only layer it only logs a
        warning.
        """
        if not self._interactive:
            logger.warning("Layer is in read-only mode (interactive=False), ignoring clear_boxes")
            return
        if self._bbox_store.data.size == 0:
            return
        n_boxes = len(self._bbox_store.data)
        self._bbox_store.clear()
        if hasattr(self, "_undo_stack"):
            self._undo_stack.push(
                self.bounding_boxes.copy(), f"Clear {n_boxes} box{('es' if n_boxes > 1 else '')}"
            )
        self._event_manager.emit("cleared", self.bounding_boxes)

    def subscribe(self, subscriber) -> None:
        """Register an object to be told about box changes.

        ``subscriber.handle_event(event_type, boxes)`` is called with
        ``"added"``, ``"updated"`` or ``"cleared"`` and the current boxes. Only
        a weak reference to the subscriber is kept.
        """
        self._event_manager.subscribe(subscriber)

    def unsubscribe(self, subscriber) -> None:
        """Remove a subscriber added with ``subscribe``; unknown ones are ignored."""
        self._event_manager.unsubscribe(subscriber)

    def _clamp_to_valid_range(self, boxes: np.ndarray) -> np.ndarray:
        """Return a copy of ``boxes`` changed just enough to be valid (see ``_validate_box``).

        Used for mouse edits (moving, resizing, Add Rectangle), which can go
        past the image or collapse a box and are corrected rather than
        rejected. With ``image_shape`` set, mins and maxs are clipped to
        ``[0, image_shape - 1]``. Then every axis thinner than 1 is widened,
        keeping the max where possible:

        - if ``max >= 1``: ``min = max - 1``
        - otherwise (box at the lower edge): ``min = 0``, ``max = 1``

        Parameters
        ----------
        boxes : np.ndarray
            Boxes of shape (N, 2, ndim).

        Returns
        -------
        np.ndarray
            A new array of valid boxes (same shape).
        """
        if boxes.size == 0:
            return boxes
        boxes = np.array(boxes, dtype=float)
        upper = self._axis_maxes()  # image_shape - 1, or None
        if upper is not None:
            boxes[:, 0] = np.clip(boxes[:, 0], 0.0, upper)
            boxes[:, 1] = np.clip(boxes[:, 1], 0.0, upper)
        mins = boxes[:, 0]
        maxs = boxes[:, 1]
        too_thin = (maxs - mins) < 1.0
        pull_min = too_thin & (maxs >= 1.0)
        mins[pull_min] = maxs[pull_min] - 1.0
        push_max = too_thin & (maxs < 1.0)
        maxs[push_max] = 1.0
        mins[push_max] = 0.0
        return boxes

    def _validate_box(self, boxes: np.ndarray) -> None:
        """Raise ``ValueError`` if any box is invalid.

        A valid box has, on every axis ``i``, ``0 <= mins[i]``,
        ``maxs[i] <= image_shape[i] - 1`` and ``maxs[i] - mins[i] >= 1``. The
        extent check allows for float32 rounding. Without ``image_shape`` only
        the extent is checked.

        ``add_boxes`` and the ``bounding_boxes`` setter call this and reject
        invalid boxes instead of correcting them. Mouse edits, including
        rectangles drawn with Add Rectangle, are clamped first
        (``_clamp_to_valid_range``).

        Parameters
        ----------
        boxes : np.ndarray
            Boxes of shape (N, 2, ndim).
        """
        boxes = np.asarray(boxes, dtype=float)
        if boxes.size == 0:
            return
        upper = self._axis_maxes()  # None if image_shape is None
        for b, box in enumerate(boxes):
            mins, maxs = box[0], box[1]
            for i in range(self.ndim):
                if upper is not None:
                    if mins[i] < 0:
                        raise ValueError(
                            f"Box {b} axis {i}: min ({mins[i]}) is below 0"
                        )
                    if maxs[i] > upper[i]:
                        raise ValueError(
                            f"Box {b} axis {i}: max ({maxs[i]}) exceeds "
                            f"image_shape - 1 ({upper[i]:g})"
                        )
                # float32 storage can round an extent of exactly 1 to slightly below 1
                tol = 4 * np.finfo(np.float32).eps * max(1.0, abs(mins[i]), abs(maxs[i]))
                if maxs[i] - mins[i] < 1.0 - tol:
                    raise ValueError(
                        f"Box {b} axis {i}: min-max distance "
                        f"({maxs[i] - mins[i]:g}) below required minimum 1"
                    )

    def _is_3d_view(self) -> bool:
        """True when the layer is shown in 3D: ``ndisplay == 3`` and ``ndim >= 3``.

        In this case ``_current_paths`` returns one wireframe path per box,
        which must be set with ``shape_type='path'``. A 2D layer in a 3D
        viewer has only two displayed axes and keeps its 2D rectangles.
        """
        slice_input = getattr(self, "_slice_input", None)
        return (
            slice_input is not None
            and getattr(slice_input, "ndisplay", None) == 3
            and self.ndim >= 3
        )

    def _sync_shapes_from_bboxes(self) -> None:
        """Rebuild all shapes from the store for the current view.

        This is the full rebuild. It also records what the incremental updates
        rely on: the draw mode (``"2d"``, ``"3d"`` or None), the box index
        behind each 2D shape and the view the shapes were built for. If napari
        refuses the data (``_safe_set_shape_data``), the mode is set to None so
        that the next update rebuilds again.
        """
        self._sync_guard = True
        try:
            self._pending_shape_state = (None, None, None)
            paths = self._current_paths()
            # Always pass shape_type. With None, napari reuses the shape types it
            # has, which can still hold 'rectangle' from its Add Rectangle tool;
            # napari then tries to build a 4-vertex Rectangle from our closed
            # 5-vertex slice path and raises ValueError.
            # 3D wireframe: 'path' (non-planar polyline); 2D slice: 'polygon'.
            shape_type = "path" if self._is_3d_view() else "polygon"
            # napari's data setter gives new shapes width 1 and refreshes; the
            # width is then set for all shapes, which refreshes again. Block both
            # and refresh once.
            with self._block_refresh():
                if not self._safe_set_shape_data(paths, shape_type):
                    self._shapes_mode = None  # nothing drawn: no incremental edits
                    return
                if hasattr(self, "current_edge_width"):
                    self.edge_width = self.current_edge_width
            self.refresh()
            # Remember which box each shape shows, so single-box edits can be
            # applied to one shape instead of rebuilding the whole layer.
            mode, box_ids, view_key = self._pending_shape_state
            self._shapes_mode = mode
            self._shape_box_ids = box_ids
            self._shapes_view_key = view_key
        finally:
            self._sync_guard = False

    def _on_store_session_ended(self, event=None) -> None:
        """Rasterize the thumbnail once if it was skipped during the session."""
        if getattr(self, "_thumbnail_stale", False):
            self._update_thumbnail()

    def _slice_query(self):
        """Return ``(point, displayed, view_key)`` for the current view.

        ``point`` is the slice point in data coordinates and ``displayed`` the
        displayed axes. Shapes built for one ``view_key`` are valid only while
        the key stays the same.
        """
        slice_input = self._slice_input
        displayed = tuple(slice_input.displayed)
        point = self._slice_point_data()
        # Keyed on the data-space point the paths are built at, so a changed
        # scale or translate (same world point, different data point) is a new view.
        view_key = (slice_input.ndisplay, displayed, tuple(np.round(point, decimals=6)))
        return point, displayed, view_key

    def _slice_point_data(self) -> np.ndarray:
        """Current slice point in this layer's data coordinates (boxes live in data space)."""
        pt = np.asarray(self._slice_input.world_slice.point, dtype=float)
        if len(pt) < self.ndim:
            pt = np.pad(pt, (self.ndim - len(pt), 0), constant_values=0)
        return np.asarray(self.world_to_data(pt), dtype=float)

    def _incremental_shape_update(self, changed_idx: int, new_data: np.ndarray) -> bool:
        """Update only the shape of box ``changed_idx`` instead of all shapes.

        In a 3D view, and in a 2D view for a box that stays on the slice, the
        shape is edited in place (``_edit_shape``). A box that enters or
        leaves the slice gets its shape added or removed
        (``_add_or_remove_shape``). A box that is off the slice before and
        after needs nothing.

        Returns ``False`` when the caller has to do a full rebuild: the view
        (slice, displayed axes, 2D or 3D) changed since the last full rebuild,
        the shape bookkeeping is missing or does not match, another sync is
        running, the edited shape's row counts changed, or an enter or leave
        that ``_add_or_remove_shape`` does not handle.

        If napari raises during the in-place edit, the shape can be left half
        edited. ``_shapes_mode`` is then cleared, so the next update rebuilds
        the view, and the exception is re-raised (``BBoxDataStore`` logs
        listener errors).
        """
        mode = getattr(self, "_shapes_mode", None)
        if mode is None or self._sync_guard:
            return False
        bbox = np.asarray(new_data[changed_idx], dtype=float)
        if mode == "3d":
            if not self._is_3d_view() or self.nshapes != len(new_data):
                return False
            pos = int(changed_idx)
            path = wireframe_path_from_bbox(bbox)
        else:
            if self._is_3d_view():
                return False
            box_ids = self._shape_box_ids
            current_step, displayed, view_key = self._slice_query()
            if view_key != self._shapes_view_key or self.nshapes != len(box_ids):
                return False
            pos = int(np.searchsorted(box_ids, changed_idx))
            was_visible = pos < len(box_ids) and box_ids[pos] == changed_idx
            new_paths = slice_paths_from_bbox(bbox, displayed, current_step)
            if not was_visible and not new_paths:
                return True  # off-slice before and after: nothing to draw
            if not (was_visible and new_paths):
                # enters or leaves the slice
                path = np.asarray(new_paths[0], dtype=float) if new_paths else None
                return self._add_or_remove_shape(pos, int(changed_idx), path)
            path = new_paths[0]
        self._sync_guard = True
        try:
            self._edit_shape(pos, np.asarray(path, dtype=float))
            if self._shapes_mode is None:
                return False  # the shape's row counts changed: full rebuild
            self.refresh()
            if pos in self.selected_data:
                # napari recomputes the selection's interaction box and outline
                # only when the selection changes; the selected box itself moved.
                self._selected_box = self.interaction_box(self.selected_data)
                self._set_highlight(force=True)
        except Exception:
            # The shape may already hold its new vertices while its mesh rows
            # are still the old ones. The store only logs a listener's error, so
            # without this a later edit of another box would take the fast path
            # and leave the stale rows on screen.
            self._shapes_mode = None
            raise
        finally:
            self._sync_guard = False
        return True

    def _add_or_remove_shape(self, pos: int, box_idx: int, path: np.ndarray | None) -> bool:
        """Show a box that entered the slice (``path``) or hide one that left it (``None``).

        A full rebuild re-creates every visible shape. Here only the one shape
        is added at, or removed from, its place ``pos`` in the sorted visible
        order. The selection and hover state that a rebuild resets are reset
        here too, so the shapes, colours, z-order and displayed arrays end up
        as after a full rebuild. Like an in-slice edit, this sends no napari
        ``data`` event.

        Returns ``False`` (the caller then rebuilds) when that result cannot be
        guaranteed: shapes with a style of their own (see
        ``_uniform_shape_style``), a view that is empty before or after, a
        view that is not 2D, a layer without ``image_shape`` (its extent
        follows the shapes drawn, and the viewer re-reads it only on the
        ``data`` events of a rebuild), index arrays that do not match the
        shape list, or a napari without the ShapeList internals used here.

        If napari raises part-way, the shape list is repaired as far as a
        rebuild needs it, ``_shapes_mode`` is cleared so the next update
        rebuilds the view, and the exception is re-raised. The failure points
        the tests cover are listed in the ``except`` branch.
        """
        dv = self._data_view
        n = len(dv.shapes)
        mesh = getattr(dv, "_mesh", None)
        indices = (getattr(dv, "_vertices_index", ()), getattr(mesh, "vertices_index", ()),
                   getattr(mesh, "triangles_index", ()))
        if (
            self._image_shape is None
            or n < (1 if path is not None else 2)
            or self._slice_input.ndisplay != 2
            or any(len(index) != n + 1 for index in indices)
            or not all(hasattr(dv, name) for name in ("_update_z_order", "_clear_cache"))
            or not hasattr(self, "_feature_table")
        ):
            return False
        style = self._uniform_shape_style(entering=path is not None)
        if style is None:
            return False
        edge_color, face_color = style
        self._sync_guard = True
        try:
            # What napari's data setter does before it replaces the shapes.
            with self._block_refresh():
                self._finish_drawing()
            if self.selected_data:
                self.selected_data = set()
            with dv.batched_updates():
                if path is not None:
                    from napari.layers.shapes._shapes_models import Polygon

                    slice_input = self._slice_input
                    shape = Polygon(
                        path,
                        edge_width=self.current_edge_width,
                        z_index=0,
                        dims_order=slice_input.order,
                        ndisplay=slice_input.ndisplay,
                    )
                    # The list form takes napari's batch path, the same one a
                    # full rebuild uses (it keeps napari's float32/int32 arrays).
                    dv.add([shape], face_color=[face_color], edge_color=[edge_color], z_refresh=False)
                    _move_last_shape(dv, pos)
                    dv._update_z_order()
                    dv._clear_cache()
                    self._shape_box_ids = np.insert(self._shape_box_ids, pos, box_idx)
                else:
                    dv.remove(pos)  # also re-derives the z-order
                    # The colour rows are dropped by napari's Shapes.remove, not by
                    # the ShapeList, so drop them here.
                    dv._edge_color = np.delete(dv._edge_color, pos, axis=0)
                    dv._face_color = np.delete(dv._face_color, pos, axis=0)
                    self._shape_box_ids = np.delete(self._shape_box_ids, pos)
            self._feature_table.resize(len(dv.shapes))
            self.refresh()
        except Exception:
            # napari failed part-way: its shape list may hold one shape more or one
            # fewer than its colour and index arrays. The next update rebuilds the
            # view, and a rebuild takes each shape's colour from these arrays, so
            # give every shape the colours they all share (see _uniform_shape_style).
            # The tests raise at five points: after napari appended the new shape's
            # arrays but not the shape, in the list add itself, after
            # _move_last_shape moved the shape object but not its rows, and in
            # ShapeList.remove before and after it removed the shape. A shape list
            # with fewer colour rows than shapes is repaired the same way, but no
            # test produces that state.
            self._shapes_mode = None
            n = len(dv.shapes)
            dv._edge_color = np.repeat([edge_color], n, axis=0).astype(dv._edge_color.dtype)
            dv._face_color = np.repeat([face_color], n, axis=0).astype(dv._face_color.dtype)
            dv._z_index = np.asarray(dv.z_indices, dtype=dv._z_index.dtype)
            raise
        finally:
            self._sync_guard = False
        return True

    def _uniform_shape_style(self, entering: bool) -> tuple[np.ndarray, np.ndarray] | None:
        """Edge and face colour for a shape added or removed in place, or None.

        napari's data setter hands z-index, colours and edge width to the new
        shapes by position (the first shapes keep their old values, the rest
        get defaults), and ``_sync_shapes_from_bboxes`` then sets every edge
        width to ``current_edge_width``. Adding or removing one shape in place
        gives the same layer only while no shape has a style of its own:

        - all z-indices are 0 (an entering shape gets 0) or, for a removal,
          all equal;
        - all shapes share one edge and one face colour, and for an entering
          shape these must be the colours napari gives a new shape;
        - every edge width is ``current_edge_width``;
        - the layer has no features and its text is constant.

        A new TurboBox layer meets these conditions. napari's per-shape tools
        (move to front, colouring the selected shapes) break them, and the
        layer then keeps using the full rebuild (None is returned).
        """
        dv = self._data_view
        z_index = np.asarray(dv._z_index)
        if np.any(z_index != (0 if entering else z_index[0])):
            return None
        if self._feature_table.values.shape[1] or not all(
            hasattr(encoding, "constant") for encoding in (self.text.string, self.text.color)
        ):
            return None
        width = self.current_edge_width
        if any(shape.edge_width != width for shape in dv.shapes):
            return None
        if entering:
            from napari.layers.utils.color_transformations import (
                normalize_and_broadcast_colors,
                transform_color_with_defaults,
            )

            colors = []
            for attribute in ("edge", "face"):
                color = transform_color_with_defaults(
                    num_entries=1,
                    colors=self._get_new_shape_color(1, attribute=attribute),
                    elem_name=f"{attribute}_color",
                    default="white",
                )
                colors.append(np.asarray(normalize_and_broadcast_colors(1, color), np.float32)[0])
            edge_color, face_color = colors
        else:
            edge_color, face_color = dv._edge_color[0], dv._face_color[0]
        if not (np.all(dv._edge_color == edge_color) and np.all(dv._face_color == face_color)):
            return None
        return edge_color, face_color

    def _edit_shape(self, pos: int, path: np.ndarray) -> None:
        """Give shape ``pos`` the vertices ``path`` in napari's shape list.

        ``ShapeList.edit`` re-derives the displayed shapes and the z-order of
        *all* shapes after every edit, which is O(N_visible) Python work.
        Moving or resizing a box changes neither: the shape stays on the slice
        and keeps its z-index. When its vertex and triangle counts and its
        triangulation are unchanged as well, only the shape's own rows of the
        vertex, mesh and displayed-vertex arrays are rewritten. Finding those
        displayed rows is still one NumPy pass over all displayed vertices.

        Otherwise:

        - Same counts but a different triangulation (for example a box that
          becomes one voxel thin), or a napari without these internals: the
          edit goes through ``ShapeList.edit``, which also rebuilds the
          displayed triangles.
        - Different counts (only a box with a zero extent has them): the
          shape's arrays are left as they are, ``_shapes_mode`` is cleared and
          the caller does a full rebuild. ``ShapeList.edit`` would keep or move
          the shape's row slot, which napari < 0.7 gets wrong.
        """
        dv = self._data_view
        shape = dv.shapes[pos]
        before = (shape.vertices_count, getattr(shape, "face_vertices_count", None),
                  shape.triangles_count, shape.data_displayed.shape[0])
        face0 = np.array(getattr(shape, "_face_triangles", None))
        edge0 = np.array(getattr(shape, "_edge_triangles", None))
        shape.data = path
        after = (shape.vertices_count, getattr(shape, "face_vertices_count", None),
                 shape.triangles_count, shape.data_displayed.shape[0])
        same_triangulation = (
            face0.ndim == 2
            and edge0.ndim == 2
            and np.array_equal(face0, getattr(shape, "_face_triangles", None))
            and np.array_equal(edge0, getattr(shape, "_edge_triangles", None))
        )
        rows = None
        if before == after and same_triangulation and hasattr(dv, "_ShapeList__update_displayed_called"):
            vslice = dv._vertices_slice_available(pos)
            rows = np.flatnonzero(dv.displayed_vertices_to_shape_num == pos)
            if len(rows) not in (0, vslice.stop - vslice.start):
                rows = None
        if rows is None:
            if before != after:
                self._shapes_mode = None
                return
            dv.edit(pos, path)
            return
        pending = dv._ShapeList__update_displayed_called
        with dv.batched_updates():
            dv._update_vertices(pos)
            dv._update_mesh_triangles(pos)
            dv._update_mesh_vertices(pos, edge=True, face=True)
            # The displayed shapes and their order are unchanged: skip the O(N)
            # recomputation that the updates above schedule.
            dv._ShapeList__update_displayed_called = pending
        if len(rows):
            dv.displayed_vertices[rows] = dv._vertices[vslice]

    def _update_thumbnail(self, *args, **kwargs):
        """Rasterize the thumbnail, except while boxes are being dragged.

        napari rasterizes the thumbnail on every refresh, which during a drag
        means once per mouse move for nothing visible. It is skipped

        - during this layer's own drag (``_allow_thumbnail_update`` is False
          from mouse press to release; the release draws it once), and
        - while the store has an edit session open, unless the layer's
          ``suppress_thumbnail_in_session`` attribute is False. The thumbnail
          is then marked stale and drawn once when the session ends
          (``_on_store_session_ended``).

        Otherwise napari's method runs unchanged. napari-bbox skips thumbnails
        during a drag in the same way (its ``_is_moving`` flag).
        """
        if not getattr(self, "_allow_thumbnail_update", True):
            return
        store = getattr(self, "_bbox_store", None)
        if (
            store is not None
            and store.in_edit_session
            and getattr(self, "suppress_thumbnail_in_session", True)
        ):
            # Rasterized once at session end (_on_store_session_ended).
            self._thumbnail_stale = True
            return
        self._thumbnail_stale = False
        return super()._update_thumbnail(*args, **kwargs)

    def _current_paths(self) -> list[np.ndarray]:
        """Return the paths to draw for the current view.

        - 2D view: one closed rectangle for each box on the current slice, in
          box index order. The spatial index finds the boxes. Above 1000
          visible boxes the rectangles come from one vectorized call. Below
          that each box goes through ``_path_cache``, keyed by the box's bytes;
          an entry is valid for one ``view_key``, and the older half of the
          entries is dropped when the cache is full.
        - 3D view of 3D data: one wireframe path per box, for all boxes, also
          cached. This branch does not trim the cache.
        - Otherwise (no slice input, or a 3D view of more than 3 axes): all box
          edges (``paths_from_bboxes``).

        For the 2D and 3D views it also sets ``_pending_shape_state``, which
        ``_sync_shapes_from_bboxes`` keeps once napari has taken the paths.
        """
        if not self._bbox_store.data.size:
            return []
        if not hasattr(self, "_slice_input") or self._slice_input is None:
            logger.warning(
                "_current_paths: _slice_input not available, using default edge representation"
            )
            return paths_from_bboxes(self._bbox_store.data)
        slice_input = self._slice_input
        view_key = (
            slice_input.ndisplay,
            tuple(slice_input.displayed) if hasattr(slice_input, "displayed") else None,
            None,
        )
        if not self._is_3d_view():
            current_step, _, view_key_2d = self._slice_query()
            view_key = view_key_2d
            query_mins = np.full(self.ndim, -np.inf)
            query_maxs = np.full(self.ndim, np.inf)
            tolerance = 1e-06
            hidden_axes = [ax for ax in range(self.ndim) if ax not in slice_input.displayed]
            for ax in hidden_axes:
                query_mins[ax] = current_step[ax] - tolerance
                query_maxs[ax] = current_step[ax] + tolerance
            query_box = np.stack([query_mins, query_maxs], axis=0)
            # Sorted, so shape k shows the k-th visible box in index order
            # (the window query returns candidates in coordinate order).
            visible_indices = np.sort(
                np.asarray(self._spatial_index.intersection(query_box), dtype=np.intp)
            )
            self._pending_shape_state = ("2d", visible_indices, view_key_2d)
            if not len(visible_indices):
                return []
            paths = []
            visible_boxes = self._bbox_store.data[visible_indices]
            if len(visible_indices) > 1000:
                return slice_paths_from_bboxes(visible_boxes, slice_input.displayed, current_step)
            for bbox in visible_boxes:
                box_hash = hash(bbox.tobytes())
                cached = self._path_cache.get(box_hash)
                if cached and cached[1] == view_key:
                    paths.extend(cached[0])
                else:
                    new_paths = slice_paths_from_bbox(bbox, slice_input.displayed, current_step)
                    self._path_cache[box_hash] = (new_paths, view_key)
                    paths.extend(new_paths)
                    if len(self._path_cache) > self._cache_max_size:
                        items = list(self._path_cache.items())
                        self._path_cache = dict(items[-self._cache_max_size // 2 :])
                        logger.debug(
                            "Cache eviction: kept%s/%sentries",
                            len(self._path_cache),
                            self._cache_max_size,
                        )
            return paths
        elif self.ndim == 3:
            logger.debug("_current_paths: 3D mode - generating wireframe paths")
            self._pending_shape_state = ("3d", None, None)
            paths = []
            for bbox in self._bbox_store.data:
                box_hash = hash(bbox.tobytes())
                cached = self._path_cache.get(box_hash)
                if cached and cached[1] == view_key:
                    paths.extend(cached[0])
                else:
                    # One wireframe path per box, cached as a one-element list
                    # like the 2D entries.
                    new_paths = [wireframe_path_from_bbox(bbox)]
                    self._path_cache[box_hash] = (new_paths, view_key)
                    paths.extend(new_paths)
            return paths
        else:
            logger.debug("_current_paths: Fallback - using edge representation")
            return paths_from_bboxes(self._bbox_store.data)

    def _on_shared_data_changed(self, new_data: np.ndarray, changed_idx=None) -> None:
        """Store listener: called with the store's read-only data after a change.

        ``changed_idx`` is the index of the one box that changed, or None for
        any other change. During a store edit session an ``on_commit`` layer
        is called once, when the session ends, with the index only if all
        changes in the session were to that one box.
        """
        self._perform_sync(new_data, changed_idx=changed_idx)

    def _perform_sync(self, new_data: np.ndarray, changed_idx=None) -> None:
        """Bring the spatial index and the shapes up to date with ``new_data``.

        Spatial index: for a single-box change (``changed_idx`` set, same
        number of boxes) only that box is updated; boxes appended at the end,
        with the old rows unchanged, are inserted if there are at most 10 of
        them or fewer than 20 % of the existing boxes; anything else rebuilds
        the index. Shapes: a single-box change goes through
        ``_incremental_shape_update``; everything else, and any case in which
        that returns False, gets a full rebuild. Then
        ``events.bboxes`` and an ``"updated"`` event are sent with the store's
        read-only data (no copy).
        """
        logger.debug("_perform_sync:%sboxes (interactive=%s)", len(new_data), self._interactive)
        old_count = self._spatial_index._count if hasattr(self, "_spatial_index") else 0
        new_count = len(new_data)
        single_box_edit = False
        if new_count == 0:
            self._spatial_index = SpatialIndex(self.bounding_boxes)
            logger.debug("Rebuilt spatial index (empty)")
        elif old_count == 0:
            self._spatial_index = SpatialIndex(self.bounding_boxes)
            logger.debug("Built spatial index (%sboxes)", new_count)
        elif new_count == old_count:
            if changed_idx is not None:
                self._spatial_index.update_box(changed_idx, new_data[changed_idx])
                single_box_edit = True
                logger.debug("Incremental spatial index update for box%s", changed_idx)
            else:
                self._spatial_index = SpatialIndex(self.bounding_boxes)
                logger.debug("Rebuilt spatial index (%sboxes, same count)", new_count)
        elif new_count > old_count:
            n_added = new_count - old_count
            # Incremental add is only valid for an append: a re-insert (undo of a
            # delete) or a batch with in-place edits shifts or changes old rows.
            appended = np.array_equal(
                self._spatial_index._boxes, np.asarray(new_data[:old_count], dtype=np.float32)
            )
            if appended and (n_added <= 10 or n_added / old_count < 0.2):
                new_boxes = self.bounding_boxes[old_count:]
                self._spatial_index.add_boxes(new_boxes)
                logger.debug("Incremental add:%sboxes (total=%s)", n_added, new_count)
            else:
                self._spatial_index = SpatialIndex(self.bounding_boxes)
                logger.debug("Rebuilt spatial index (added%sboxes, total=%s)", n_added, new_count)
        else:
            self._spatial_index = SpatialIndex(self.bounding_boxes)
            logger.debug(
                "Rebuilt spatial index (removed%sboxes, total=%s)", old_count - new_count, new_count
            )
            current_hashes = {hash(bbox.tobytes()) for bbox in new_data}
            self._path_cache = {h: v for h, v in self._path_cache.items() if h in current_hashes}
        if not (single_box_edit and self._incremental_shape_update(changed_idx, new_data)):
            self._sync_shapes_from_bboxes()
        # Read-only view of the shared store: no per-layer (N, 2, D) copy per notification.
        view = self._bbox_store.data
        self.events.bboxes(value=view)
        self._event_manager.emit("updated", view)

    def _set_view_slice(self) -> None:
        """Let napari slice the layer, then rebuild the shapes for the new view.

        napari calls this on every slice change and every ``refresh()``. The
        rebuild is skipped while the layer itself is writing shapes
        (``_sync_guard``), while napari is drawing a new shape, and when the
        store is empty.
        """
        if not hasattr(self, "_sync_guard") or not hasattr(self, "_bbox_store"):
            super()._set_view_slice()
            return
        super()._set_view_slice()
        if (
            not self._sync_guard
            and self._bbox_store.data.size > 0
            and not getattr(self, "_is_creating", False)
            and not getattr(self, "_napari_adding", False)
        ):
            logger.debug(
                "_set_view_slice called - regenerating paths (ndisplay=%s)",
                self._slice_input.ndisplay if self._slice_input else "?",
            )
            self._sync_shapes_from_bboxes()

    def _axis_maxes(self) -> np.ndarray | None:
        """Largest valid coordinate per axis (``image_shape - 1``, at least 0), or None."""
        if self._image_shape is not None:
            maxes = self._image_shape.astype(float) - 1.0
            maxes[maxes < 0] = 0.0
            return maxes
        return None

    def _expand_hidden_axes(self, bbox: np.ndarray) -> np.ndarray:
        """Return a copy of ``bbox`` with each flat hidden axis extended.

        On every axis the view does not show where min and max are equal
        (``np.isclose`` with ``atol=1e-5``), the box gets
        ``[0, image_shape - 1]``, or ``[0, max + 1]`` without ``image_shape``.
        Displayed axes are not changed. Without a slice input the box is
        returned unchanged.
        """
        bbox = np.asarray(bbox, dtype=float)
        mins, maxs = bbox
        expanded = np.stack([mins.copy(), maxs.copy()], axis=0)
        if hasattr(self, "_slice_input") and self._slice_input is not None:
            displayed = set(self._slice_input.displayed)
            maxes_hint = self._axis_maxes()
        else:
            if self._interactive:
                logger.warning(
                    "_expand_hidden_axes called but _slice_input is None. Expansion skipped."
                )
            return expanded
        expanded_count = 0
        for axis in range(self.ndim):
            if axis in displayed:
                continue
            if np.isclose(expanded[0, axis], expanded[1, axis], atol=1e-05):
                expanded[0, axis] = 0.0
                if (
                    maxes_hint is not None
                    and axis < maxes_hint.size
                    and (not np.isnan(maxes_hint[axis]))
                ):
                    expanded[1, axis] = maxes_hint[axis]
                else:
                    expanded[1, axis] = expanded[1, axis] + 1.0
                expanded_count += 1
        if expanded_count > 0:
            logger.debug("Expanded%shidden axes for new bbox", expanded_count)
        return expanded

    def _normalize_image_shape(
        self, image_shape: Sequence[int] | None, ndim: int
    ) -> np.ndarray | None:
        """Return ``image_shape`` as a float array with one entry per layer axis.

        Extra leading entries are dropped; missing leading entries are filled
        with the first given value.
        """
        if image_shape is None:
            return None
        arr = np.asarray(image_shape, dtype=float)
        if arr.size > ndim:
            arr = arr[-ndim:]
        elif arr.size < ndim:
            arr = np.pad(arr, (ndim - arr.size, 0), mode="edge")
        return arr

    def _connect_dims_events(self) -> None:
        """Does nothing except set ``_dims_connected`` and log once.

        No dims events need to be connected: ``_set_view_slice`` rebuilds the
        shapes on every slice change. napari does not call this method.
        """
        if not self._dims_connected:
            self._dims_connected = True
            logger.info(
                "_connect_dims_events: Using _set_view_slice for updates (no manual connection needed)"
            )

    def _on_mode_change(self, event) -> None:
        """Switch napari modes that cannot make axis-aligned boxes back to Select.

        These are Transform (rotation) and the ellipse, line, path, polygon and
        lasso tools. For the other modes only a hint is logged.
        """
        from napari.layers.shapes._shapes_constants import Mode

        mode = event.value if hasattr(event, "value") else self.mode
        if isinstance(mode, str):
            try:
                mode = Mode(mode)
            except ValueError:
                logger.warning("Unknown mode:%s", mode)
                return
        if mode == Mode.TRANSFORM:
            logger.error("TRANSFORM mode is NOT supported for axis-aligned bounding boxes!")
            logger.debug("Bounding boxes must stay parallel to X/Y/Z axes (no rotation allowed)")
            logger.debug("Switching back to SELECT mode...")
            self.mode = Mode.SELECT
            return
        blocked_add_modes = {
            Mode.ADD_ELLIPSE,
            Mode.ADD_LINE,
            Mode.ADD_PATH,
            Mode.ADD_POLYGON,
            Mode.ADD_POLYGON_LASSO,
        }
        if mode in blocked_add_modes:
            logger.error("'Add%s' mode is NOT supported for bounding boxes!", mode.name)
            logger.debug("Use 'Add Rectangle' mode instead to draw axis-aligned bboxes")
            self.mode = Mode.SELECT
            return
        elif mode == Mode.ADD_RECTANGLE:
            logger.info("ADD_RECTANGLE mode active")
            logger.debug("Click and drag to draw a bounding box")
            logger.debug("The box will automatically extend through hidden dimensions")
        elif mode == Mode.SELECT:
            logger.info("SELECT mode active")
            logger.debug("Click inside bbox center  Move it")
            logger.debug("Click near edge/corner  Resize it")
        elif mode == Mode.PAN_ZOOM:
            logger.debug("PAN_ZOOM mode: Navigate the view (hold Space anytime)")

    def _on_parent_change(self, event) -> None:
        """Rebuild the shapes from the store, unless a sync is running.

        napari does not call this method (checked for 0.6.6, 0.7.0 and 0.9.1);
        it only runs when called directly.
        """
        if not self._sync_guard and self._bbox_store.data.size > 0:
            self._sync_shapes_from_bboxes()

    def _on_shapes_data_changed(self, event) -> None:
        """Apply a shape edit made with napari's own tools to the store.

        The store holds all boxes while the view shows only those on the
        current slice, so the store is never rebuilt from ``self.data``.
        Instead each ``event.action`` becomes one store change:

        - ``adding``, ``removing``, ``changing`` (sent before napari changes
          the shapes): nothing is written. For ``removing`` the boxes behind
          the shapes about to go are remembered.
        - ``removed``: exactly those boxes are deleted, as one undo step.
        - ``added`` (Add Rectangle finished): the rectangle becomes one box.
          Its hidden axes are extended (see ``auto_extrude`` and
          ``_expand_hidden_axes``), it is clamped to the image and passed to
          ``add_boxes``. A click without drag (under 1 on a displayed axis)
          adds nothing. The shape napari drew is replaced by the drawing of
          the store.
        - anything else (for example vertex edits): discarded by redrawing
          from the store.

        Nothing happens while the layer itself writes shapes (``_sync_guard``)
        or drags a box. The first post-change event after a TurboBox drag
        clears ``_just_finished_drag`` and is dropped unless it is ``added`` or
        ``removed``. Read-only layers never write; they only redraw.
        """
        if self._sync_guard:
            return
        if hasattr(self, "_bbox_is_moving") and self._bbox_is_moving:
            return
        action = str(getattr(event, "action", "") or "")
        if action in ("adding", "removing", "changing"):
            # Pre-change events: remember what napari is about to do; never write.
            if action == "removing":
                box_ids = self._displayed_box_ids()
                self._pending_remove = [
                    int(box_ids[p])
                    for p in getattr(event, "data_indices", ())
                    if box_ids is not None and 0 <= p < len(box_ids)
                ]
            elif action == "adding":
                self._napari_adding = True  # keep napari's new shape until 'added'
            return
        self._napari_adding = False
        pending, self._pending_remove = self._pending_remove, None
        if getattr(self, "_just_finished_drag", False):
            # napari's own post-release event of a TurboBox drag: nothing to write.
            self._just_finished_drag = False
            if action not in ("added", "removed"):
                return
        if not self._interactive:
            self._sync_shapes_from_bboxes()
            return
        if action == "removed":
            if pending:
                self._bbox_store.remove_boxes(sorted(set(pending)))
                self._undo_stack.push(self.bounding_boxes.copy(), f"Delete {len(pending)} box(es)")
            else:
                self._sync_shapes_from_bboxes()
            return
        if action == "added" and len(self.data) > 0:
            bbox = bbox_from_path(np.asarray(self.data[-1], dtype=float))
            self._sync_shapes_from_bboxes()  # drop napari's shape; the store re-renders it
            if bbox is None or self._slice_input is None:
                return
            disp = list(self._slice_input.displayed)
            if np.any(bbox[1, disp] - bbox[0, disp] < 1.0):
                return  # click without drag: no box
            if self._auto_extrude and self._is_2d_box(bbox):
                bbox = self._auto_extrude_bbox(bbox)
            bbox = self._expand_hidden_axes(bbox)
            self.add_boxes(self._clamp_to_valid_range(bbox[None]))
            return
        self._sync_shapes_from_bboxes()  # 'changed' (vertex/move edits outside _on_mouse_drag)

    def _displayed_box_ids(self) -> np.ndarray | None:
        """Box index behind each currently displayed shape (None if unknown)."""
        mode = getattr(self, "_shapes_mode", None)
        if mode == "2d":
            return self._shape_box_ids
        if mode == "3d":
            return np.arange(len(self._bbox_store.data))
        return None

    def _is_2d_box(self, bbox: np.ndarray) -> bool:
        """True if the box is flat (min equals max, ``np.isclose``) on any axis."""
        mins, maxs = bbox
        for axis in range(self.ndim):
            if np.isclose(mins[axis], maxs[axis]):
                return True
        return False

    def _auto_extrude_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """Extrude a rectangle drawn in a 2D view along the hidden axis.

        With ``image_shape`` the hidden axis spans ``[0, image_shape - 1]``;
        without it, ``[0, max(z, 1)]`` for slice position ``z``
        (``extrude_2d_to_3d_box``). Without a slice input the box is returned
        unchanged.
        """
        if not hasattr(self, "_slice_input") or self._slice_input is None:
            logger.warning("Auto-extrude: _slice_input not available, returning bbox as-is")
            return bbox
        slice_input = self._slice_input
        displayed = slice_input.displayed
        current_step = self._slice_point_data()
        extrusion_range = None
        if self._image_shape is not None:
            hidden_axes = [ax for ax in range(self.ndim) if ax not in displayed]
            if hidden_axes:
                axis = hidden_axes[0]
                extrusion_range = (0.0, float(self._image_shape[axis] - 1))
        logger.debug(
            "Auto-extruding bbox with displayed=%s, current_step=%s, extrusion_range=%s",
            displayed,
            current_step,
            extrusion_range,
        )
        return extrude_2d_to_3d_box(bbox, displayed, current_step, extrusion_range)

    def _current_shape_box_ids(self) -> np.ndarray | None:
        """Box index shown by each 2D shape, or None if the bookkeeping does not match the view."""
        box_ids = getattr(self, "_shape_box_ids", None)
        if (
            box_ids is None
            or getattr(self, "_slice_input", None) is None
            or getattr(self, "_shapes_mode", None) != "2d"
            or self._is_3d_view()
            or self.nshapes != len(box_ids)
        ):
            return None
        if self._slice_query()[2] != self._shapes_view_key:
            return None
        return box_ids

    def _bbox_index_to_path_index(self, bbox_idx: int) -> int | None:
        """Index of the (first) shape that shows box ``bbox_idx``, or None.

        Uses the 2D bookkeeping (a binary search; None if the box is not on
        the slice) or, in a 3D view, the box index itself. Otherwise it counts
        the paths of all boxes before ``bbox_idx``, which is slow and does not
        check that the box itself is drawn.
        """
        if not self._bbox_store.data.size or bbox_idx >= self._bbox_store.data.shape[0]:
            return None
        box_ids = self._current_shape_box_ids()
        if box_ids is not None:
            # O(log N): shape k shows the k-th visible box (sorted indices).
            pos = int(np.searchsorted(box_ids, bbox_idx))
            return pos if pos < len(box_ids) and box_ids[pos] == bbox_idx else None
        if getattr(self, "_shapes_mode", None) == "3d" and self._is_3d_view():
            return int(bbox_idx)
        cumulative = 0
        for i in range(bbox_idx):
            paths = self._paths_for_single_bbox(self._bbox_store.data[i])
            cumulative += len(paths)
        return cumulative

    def _path_index_to_bbox_index(self, path_idx: int) -> int | None:
        """Index of the box that shape ``path_idx`` shows, or None.

        Uses the 2D bookkeeping when it matches the view. Otherwise it assumes
        that every box has as many paths as the first one, and walks the boxes
        only when that guess points past the last box.
        """
        if not self._bbox_store.data.size:
            return None
        box_ids = self._current_shape_box_ids()
        if box_ids is not None:
            return int(box_ids[path_idx]) if 0 <= path_idx < len(box_ids) else None
        paths_per_bbox = self._count_paths_per_bbox()
        if paths_per_bbox == 0:
            return None
        bbox_idx = path_idx // paths_per_bbox
        if bbox_idx >= self._bbox_store.data.shape[0]:
            cumulative = 0
            for i, bbox in enumerate(self._bbox_store.data):
                paths = self._paths_for_single_bbox(bbox)
                cumulative += len(paths)
                if path_idx < cumulative:
                    return i
            return None
        return bbox_idx

    def _count_paths_per_bbox(self) -> int:
        """Number of paths the first box has in the current view (0 without boxes)."""
        if not self._bbox_store.data.size:
            return 0
        first_bbox = self._bbox_store.data[0]
        paths = self._paths_for_single_bbox(first_bbox)
        return len(paths)

    def _paths_for_single_bbox(self, bbox: np.ndarray) -> list[np.ndarray]:
        """Paths of one box in the current view, as ``_current_paths`` draws them."""
        if not hasattr(self, "_slice_input") or self._slice_input is None:
            return path_from_bbox(bbox)
        slice_input = self._slice_input
        if not self._is_3d_view():
            current_step = self._slice_point_data()
            return slice_paths_from_bbox(bbox, slice_input.displayed, current_step)
        if self.ndim == 3:
            # One wireframe path per box, as in the 3D branch of _current_paths;
            # the shape <-> box index mapping relies on the two matching.
            return [wireframe_path_from_bbox(bbox)]
        return path_from_bbox(bbox)

    def _hit_test_bbox(self, position: np.ndarray) -> tuple[int | None, bool]:
        """Find the box under ``position`` (data coordinates) in the current view.

        Returns ``(index, near_edge)``, or ``(None, False)`` if no box is hit.
        The hidden axes are taken from the current slice point. If several
        boxes contain the point, the one with the smallest area in the
        displayed axes wins (the lower index on ties). ``near_edge`` is True
        when the point lies within 20 % of the box size (per displayed axis)
        of one of its sides; a drag from there resizes instead of moving.
        """
        if self._bbox_store.data.size == 0:
            return (None, False)
        current_step = np.zeros(self.ndim, dtype=float)
        displayed_axes = list(range(self.ndim))
        hidden_axes = []
        if hasattr(self, "_slice_input") and self._slice_input is not None:
            displayed_axes = tuple(self._slice_input.displayed)
            hidden_axes = [ax for ax in range(self.ndim) if ax not in displayed_axes]
            current_step = self._slice_point_data()
        elif hasattr(self, "dims"):
            displayed_axes = tuple(self.dims.displayed)
            hidden_axes = [ax for ax in range(self.ndim) if ax not in displayed_axes]
            current_step = np.asarray(self.dims.point, dtype=float)
        pos = np.asarray(position, dtype=float)
        if len(pos) < self.ndim:
            pos = np.pad(pos, (0, self.ndim - len(pos)), constant_values=0)
        elif len(pos) > self.ndim:
            pos = pos[: self.ndim]
        query_point = np.zeros(self.ndim, dtype=float)
        query_point[list(displayed_axes)] = pos[list(displayed_axes)]
        for ax in hidden_axes:
            query_point[ax] = current_step[ax]
        if hasattr(self, "_spatial_index"):
            candidates = self._spatial_index.query_point(query_point)
        else:
            candidates = range(self._bbox_store.data.shape[0])
        if len(candidates) == 0:
            return (None, False)
        candidates = sorted(candidates)
        # Nested or overlapping boxes: the innermost box (smallest displayed area)
        # wins; the stable sort keeps the lower index on ties.
        disp_list = list(displayed_axes)
        data = self._bbox_store.data
        candidates.sort(key=lambda i: float(np.prod(data[i, 1, disp_list] - data[i, 0, disp_list])))
        for i in candidates:
            bbox = self._bbox_store.data[i]
            mins, maxs = (bbox[0], bbox[1])
            pos_displayed = pos[list(displayed_axes)]
            mins_displayed = mins[list(displayed_axes)]
            maxs_displayed = maxs[list(displayed_axes)]
            bbox_size = maxs_displayed - mins_displayed
            edge_threshold = bbox_size * 0.2
            near_edge = np.any(pos_displayed - mins_displayed < edge_threshold) or np.any(
                maxs_displayed - pos_displayed < edge_threshold
            )
            if near_edge:
                logger.debug("Click is near edge/corner (resize mode)")
            else:
                logger.debug("Click is in center (move mode)")
            return (int(i), near_edge)
        logger.debug("Hit test: position%sis not inside any visible bbox on this slice", pos)
        return (None, False)

    def _reset_editable(self) -> None:
        """Keep a read-only layer read-only when napari resets ``editable``.

        napari calls this after every data assignment; for Shapes it makes the
        layer editable exactly in 2D display.
        """
        super()._reset_editable()
        if not getattr(self, "_interactive", True):
            self.editable = False

    @Shapes.editable.setter
    def editable(self, editable: bool) -> None:
        # napari's layer controls also set ``editable`` directly, on creation and
        # on every display change; a read-only layer stays read-only.
        Shapes.editable.fset(self, editable and getattr(self, "_interactive", True))

    def _compute_vertices_and_box(self):
        """Reset ``_value`` to ``(None, None)`` if it is None, then call napari.

        napari reads ``_value[0]`` here, which raises TypeError on None.
        """
        if self._value is None:
            self._value = (None, None)
        return super()._compute_vertices_and_box()

    def _get_value(self, position, *, world=False):
        """Return ``(shape_index, None)`` for the box under ``position``, else ``(None, None)``.

        Uses ``_hit_test_bbox`` instead of napari's hit test, so napari never
        selects a single vertex or edge. napari expects the vertex index to be
        an int or None, never a string; here it is always None. In Add
        Rectangle mode and while a mouse drag is in progress, napari's own hit
        test is used. ``world`` is ignored.
        """
        from napari.layers.shapes._shapes_constants import Mode

        in_add_mode = hasattr(self, "mode") and self.mode == Mode.ADD_RECTANGLE
        is_drawing = hasattr(self, "_mouse_drag_gen") and len(self._mouse_drag_gen) > 0
        if in_add_mode or is_drawing:
            logger.debug(
                "_get_value: Delegating to parent (in_add_mode=%s, is_drawing=%s)",
                in_add_mode,
                is_drawing,
            )
            return super()._get_value(position)
        if self._bbox_store.data.size == 0:
            return (None, None)
        bbox_idx, _ = self._hit_test_bbox(position)
        if bbox_idx is not None:
            path_idx = self._bbox_index_to_path_index(bbox_idx)
            if path_idx is not None:
                return (path_idx, None)
        return (None, None)

    def _on_mouse_drag(self, layer, event: Any):
        """Move or resize the box under the mouse.

        napari drives this generator: the code before the first ``yield`` runs
        on mouse press, the loop on each mouse move, and the ``finally`` block
        on release. A press inside a box selects it and opens a store edit
        session. The drag resizes the box when Shift is held or the press is
        near an edge (see ``_hit_test_bbox``) and moves it otherwise. Each
        move writes the new box with ``store.update_box``: live (2D) views
        update it at once, ``on_commit`` (3D) views at the release. On release
        the session ends, the layer rebuilds its shapes only if its
        bookkeeping no longer matches, draws its thumbnail once and records
        one undo step, "Move/Resize box".

        Nothing happens in Add Rectangle or pan/zoom mode, while napari is
        drawing a shape, or on a read-only layer. napari passes the layer as
        the first argument.
        """
        from napari.layers.shapes._shapes_constants import Mode

        in_add_mode = hasattr(self, "mode") and self.mode == Mode.ADD_RECTANGLE
        is_drawing = hasattr(self, "_mouse_drag_gen") and len(self._mouse_drag_gen) > 1
        in_pan_zoom = hasattr(self, "mode") and self.mode == Mode.PAN_ZOOM
        if in_add_mode or is_drawing or in_pan_zoom or not self._interactive:
            logger.debug(
                "_on_mouse_drag: Skipping (in_add_mode=%s, is_drawing=%s, in_pan_zoom=%s)",
                in_add_mode,
                is_drawing,
                in_pan_zoom,
            )
            return
        logger.debug("_on_mouse_drag: event.type=%s", event.type)
        if event.type == "mouse_press":
            logger.debug(
                "mouse_press at position=%s, modifiers=%s", event.position, event.modifiers
            )
            press_pos = np.asarray(self.world_to_data(event.position), dtype=float)
            bbox_idx, is_near_edge = self._hit_test_bbox(press_pos)
            if bbox_idx is not None:
                logger.debug("Click inside bbox %s, forcing selection", bbox_idx)
                path_idx = self._bbox_index_to_path_index(bbox_idx)
                if path_idx is not None:
                    self.selected_data = {path_idx}
                    self._set_highlight(force=True)  # outline the newly grabbed box right away
                mode = "resize" if is_near_edge or "Shift" in event.modifiers else "translate"
                self._drag_state = {
                    "index": bbox_idx,
                    "start_bbox": self.bounding_boxes[bbox_idx].copy(),
                    "start_pos": press_pos,
                    "dims_displayed": tuple(event.dims_displayed),
                    "mode": mode,
                }
                self._bbox_is_moving = True
                # No thumbnail per mouse move; it is drawn once at release
                # (see the finally block).
                self._allow_thumbnail_update = False
                self._bbox_store.begin_edit_session()
                self._session_active = True
                logger.debug("Drag state initialized: mode=%s, bbox_idx=%s", mode, bbox_idx)
                event.handled = True
            else:
                self._drag_state = None
                logger.debug("No bbox hit, clearing drag state")
                return
        try:
            yield
            while event.type == "mouse_move":
                if not self._drag_state:
                    break
                logger.debug("mouse_move: dragging bbox %s", self._drag_state["index"])
                event.handled = True
                bbox_idx = self._drag_state["index"]
                if bbox_idx >= self._bbox_store.data.shape[0]:
                    logger.warning("bbox_idx%sout of range", bbox_idx)
                    break
                current_pos = np.asarray(self.world_to_data(event.position), dtype=float)
                start_pos = self._drag_state["start_pos"]
                delta = current_pos - start_pos
                dims_displayed = self._drag_state["dims_displayed"]
                start_bbox = self._drag_state["start_bbox"]
                if self._drag_state["mode"] == "translate":
                    nd_delta = np.zeros(self.ndim, dtype=float)
                    for axis in dims_displayed:
                        if axis < self.ndim:
                            nd_delta[axis] = delta[axis]
                    upper = self._axis_maxes()
                    if upper is not None:
                        # Stop the box at the image border instead of letting
                        # the per-coordinate clamp below shrink it.
                        nd_delta = np.clip(nd_delta, -start_bbox[0], upper - start_bbox[1])
                    new_bbox = start_bbox + np.stack([nd_delta, nd_delta], axis=0)
                    logger.debug("Translating by%s", nd_delta)
                else:
                    new_bbox = start_bbox.copy()
                    center = start_bbox.mean(axis=0)
                    for axis in dims_displayed:
                        if axis >= self.ndim:
                            continue
                        if start_pos[axis] >= center[axis]:
                            new_bbox[1, axis] = start_bbox[1, axis] + delta[axis]
                        else:
                            new_bbox[0, axis] = start_bbox[0, axis] + delta[axis]
                    mins = np.minimum(new_bbox[0], new_bbox[1])
                    maxs = np.maximum(new_bbox[0], new_bbox[1])
                    # Keep an extent of at least 1 per axis before clamping, so
                    # resizing against an edge cannot collapse the box. Same rule
                    # as _clamp_to_valid_range (keep the max, move the min),
                    # which corrects the box again below if needed.
                    thin = (maxs - mins) < 1.0
                    pull = thin & (maxs >= 1.0)
                    mins[pull] = maxs[pull] - 1.0
                    push = thin & (maxs < 1.0)
                    maxs[push] = 1.0
                    mins[push] = 0.0
                    new_bbox = np.stack([mins, maxs], axis=0)
                    logger.debug("Resizing")
                new_bbox = self._clamp_to_valid_range(new_bbox[None, ...])[0]
                # Live (2D) views, this one included, update the box now;
                # on_commit (3D) views wait for the end of the session.
                self._bbox_store.update_box(bbox_idx, new_bbox)
                yield
        finally:
            if self._session_active:
                self._bbox_store.end_edit_session()
                self._session_active = False
                logger.debug("finally: Edit session closed")
            drag_state_idx = self._drag_state["index"] if self._drag_state else None
            if self._drag_state:
                logger.debug(
                    "finally: Cleaning up drag state for bbox%s", self._drag_state["index"]
                )
            self._drag_state = None
            self._bbox_is_moving = False
            # Allow thumbnails again. The code below draws this layer's thumbnail
            # once for the whole drag, through a rebuild or _update_thumbnail.
            self._allow_thumbnail_update = True
            logger.debug("finally: Cleanup complete. _bbox_is_moving = False")
            self._just_finished_drag = True
            self._drag_end_time = time.time()
            logger.debug("finally: Set _just_finished_drag = True and timestamp")
            # Every drag step went through update_box and this layer's sync, so
            # its spatial index and shapes already show the final box. Rebuild
            # only if that bookkeeping is off (index count differs from the
            # store, or the shapes no longer match the view); otherwise draw the
            # thumbnail skipped during the drag and refresh the selection outline.
            idx_count = getattr(getattr(self, "_spatial_index", None), "_count", -1)
            if idx_count != len(self._bbox_store.data):
                self._spatial_index = SpatialIndex(self.bounding_boxes)
                self._sync_shapes_from_bboxes()
            elif drag_state_idx is None or (
                self._current_shape_box_ids() is None
                and not (getattr(self, "_shapes_mode", None) == "3d" and self._is_3d_view())
            ):
                self._sync_shapes_from_bboxes()
            else:
                self._update_thumbnail()
                if self.selected_data:
                    self._selected_box = self.interaction_box(self.selected_data)
                    self._set_highlight(force=True)
            if hasattr(self, "_undo_stack"):
                self._undo_stack.push(self.bounding_boxes.copy(), "Move/Resize box")
                logger.debug("finally: Pushed undo state")
            self.events.bboxes(value=self.bounding_boxes)
            self._event_manager.emit("updated", self.bounding_boxes)

    def undo(self, *args) -> bool:
        """
        Undo the last recorded action (Ctrl+Z / Cmd+Z).

        The earlier boxes are restored through the ``bounding_boxes`` setter
        while undo recording is paused.

        Returns
        -------
        bool
            True if an action was undone, False if there was nothing to undo.
        """
        undone = self._undo_stack.get_undo_description()
        result = self._undo_stack.undo()
        if result:
            state, _ = result  # restored state; its label names the action before the undone one
            self._undo_stack.disable()
            try:
                self.bounding_boxes = state
                logger.info("Undone: %s", undone)
            finally:
                self._undo_stack.enable()
            return True
        return False

    def redo(self, *args) -> bool:
        """
        Redo the last undone action (Ctrl+Y / Cmd+Shift+Z).

        Returns
        -------
        bool
            True if an action was redone, False if there was nothing to redo.
        """
        result = self._undo_stack.redo()
        if result:
            state, description = result
            self._undo_stack.disable()
            try:
                self.bounding_boxes = state
                logger.info("Redone: %s", description)
            finally:
                self._undo_stack.enable()
            return True
        return False

    def can_undo(self) -> bool:
        """True if there is an action to undo."""
        return self._undo_stack.can_undo()

    def can_redo(self) -> bool:
        """True if there is an undone action to redo."""
        return self._undo_stack.can_redo()

    def clear_undo_history(self):
        """Clear the undo and redo history.

        The history is then empty, so the next recorded action becomes the
        starting point and cannot itself be undone.
        """
        self._undo_stack.clear()
        logger.info("Cleared undo history")

    @property
    def undo_limit(self) -> int:
        """Maximum number of states kept in the undo history.

        At most ``undo_limit - 1`` steps can be undone. Must be at least 1.
        Lowering it does not drop states already recorded.
        """
        return self._undo_stack.max_size

    @undo_limit.setter
    def undo_limit(self, value: int):
        """Set the undo history size; raises ValueError below 1."""
        if value < 1:
            raise ValueError("undo_limit must be at least 1")
        self._undo_stack.max_size = value
        logger.info("Undo limit set to%s", value)

    @property
    def cache_limit(self) -> int:
        """Maximum number of entries in the per-box path cache (at least 100)."""
        return self._cache_max_size

    @cache_limit.setter
    def cache_limit(self, value: int):
        """Set the path cache size; raises ValueError below 100."""
        if value < 100:
            raise ValueError("cache_limit must be at least 100")
        self._cache_max_size = value
        logger.info("Cache limit set to%s", value)

    def get_undo_info(self) -> dict:
        """Return the undo state as a dict.

        Keys: ``can_undo``, ``can_redo``, ``undo_description`` (label of the
        action undo would revert), ``redo_description``, ``history_size``,
        ``history_position`` and ``memory_usage_kb`` (an estimate).
        """
        return {
            "can_undo": self.can_undo(),
            "can_redo": self.can_redo(),
            "undo_description": self._undo_stack.get_undo_description(),
            "redo_description": self._undo_stack.get_redo_description(),
            "history_size": self._undo_stack.size,
            "history_position": self._undo_stack.current_position,
            "memory_usage_kb": self._undo_stack.get_memory_usage() / 1024,
        }


def _install_drag_modes() -> None:
    """Use ``_select_unless_bbox_drag`` for napari's Select and Direct modes on TurboBoxLayer."""
    from napari.layers.shapes._shapes_constants import Mode

    modes = dict(Shapes._drag_modes)
    modes[Mode.SELECT] = _select_unless_bbox_drag
    modes[Mode.DIRECT] = _select_unless_bbox_drag
    TurboBoxLayer._drag_modes = modes


_install_drag_modes()
