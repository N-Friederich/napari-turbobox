"""Control panel (Qt dock widget) for TurboBox layers.

The panel works on one TurboBoxLayer at a time, the one picked in its
"Target Bounding Box Layer" combo box. It adds layers and boxes, lists and
deletes boxes, imports and exports them, and shows the layer's undo settings.
"""

from __future__ import annotations

import logging

import numpy as np
from napari import Viewer
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .layer import TurboBoxLayer

logger = logging.getLogger(__name__)


class BoundingBoxControlWidget(QWidget):
    """Dock widget that manages the boxes of one TurboBoxLayer.

    The target is the layer selected in the combo box, not napari's active
    layer. With ``viewer=None`` the combo box stays empty.
    """

    def __init__(self, viewer: Viewer | None) -> None:
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("napari-TurboBox")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._watched_layer = None
        if self.viewer is not None:
            layers_events = self.viewer.layers.events
            layers_events.inserted.connect(self.refresh_layers)
            layers_events.removed.connect(self.refresh_layers)
            layers_events.changed.connect(self.refresh_layers)
        layout = QVBoxLayout()
        self.setLayout(layout)
        create_layer_btn = QPushButton("Add Bounding Box Layer")
        create_layer_btn.clicked.connect(self._create_layer)
        layout.addWidget(create_layer_btn)
        self.layer_combo = QComboBox()
        self.layer_combo.currentIndexChanged.connect(self._on_layer_change)
        layout.addWidget(QLabel("Target Bounding Box Layer"))
        layout.addWidget(self.layer_combo)
        self.status_label = QLabel("No bounding box layer selected.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addWidget(self._build_creation_group())
        layout.addWidget(self._build_bbox_list_group())
        layout.addWidget(self._build_io_group())
        layout.addWidget(self._build_settings_group())
        layout.addStretch(1)
        self.refresh_layers()

    def closeEvent(self, event):
        """Disconnect from the watched layer and the viewer's layer list.

        Qt deletes the widget on close (``WA_DeleteOnClose``), so its handlers
        must not stay connected to events that outlive it.
        """
        if self._watched_layer is not None:
            try:
                self._watched_layer.events.bboxes.disconnect(self._on_bboxes_changed)
                self._watched_layer._bbox_store.events.session_ended.disconnect(
                    self._on_store_session_ended
                )
            except Exception:
                pass
        if self.viewer is not None:
            try:
                layers_events = self.viewer.layers.events
                layers_events.inserted.disconnect(self.refresh_layers)
                layers_events.removed.disconnect(self.refresh_layers)
                layers_events.changed.disconnect(self.refresh_layers)
            except Exception:
                pass
        super().closeEvent(event)

    def _build_creation_group(self) -> QWidget:
        group = QGroupBox("Quick Create")
        vbox = QVBoxLayout()
        group.setLayout(vbox)
        btn_draw = QPushButton("Draw New BBox (Interactive)")
        btn_draw.clicked.connect(self._start_draw_mode)
        vbox.addWidget(btn_draw)
        btn_from_image = QPushButton("Fit Current Image Layer")
        btn_from_image.clicked.connect(self._create_from_image)
        vbox.addWidget(btn_from_image)
        btn_from_viewport = QPushButton("Fit Current Viewport")
        btn_from_viewport.clicked.connect(self._create_from_viewport)
        vbox.addWidget(btn_from_viewport)
        btn_clear = QPushButton("Remove All Bounding Boxes")
        btn_clear.clicked.connect(self._clear_boxes)
        vbox.addWidget(btn_clear)
        return group

    def _build_bbox_list_group(self) -> QWidget:
        """Build the group that lists every box of the target layer."""
        group = QGroupBox("Bounding Boxes")
        vbox = QVBoxLayout()
        group.setLayout(vbox)
        self.bbox_list = QListWidget()
        self.bbox_list.setMaximumHeight(150)
        vbox.addWidget(self.bbox_list)
        self.bbox_list.itemSelectionChanged.connect(self._on_list_selection_changed)
        button_row = QHBoxLayout()
        btn_select = QPushButton("Select in Layer")
        btn_select.clicked.connect(self._select_bbox_from_list)
        button_row.addWidget(btn_select)
        btn_delete = QPushButton("Delete Selected")
        btn_delete.clicked.connect(self._delete_selected_bbox)
        button_row.addWidget(btn_delete)
        vbox.addLayout(button_row)
        return group

    def _build_io_group(self) -> QWidget:
        """Build the row with the Export and Import buttons."""
        group = QGroupBox("Import/Export")
        hbox = QHBoxLayout()
        group.setLayout(hbox)
        btn_export = QPushButton("Export BBoxes...")
        btn_export.clicked.connect(self._export_boxes)
        hbox.addWidget(btn_export)
        btn_import = QPushButton("Import BBoxes...")
        btn_import.clicked.connect(self._import_boxes)
        hbox.addWidget(btn_import)
        return group

    def _build_settings_group(self) -> QWidget:
        """Build the "Advanced Settings" group.

        It holds the undo limit, the path cache limit, the memory used by the
        undo history, a button that clears the history and the next undo/redo
        action. The group is checkable and starts unchecked. Unchecked, Qt
        disables its controls (the group does not collapse).
        """
        group = QGroupBox("⚙️ Advanced Settings")
        group.setCheckable(True)
        group.setChecked(False)
        group.setStyleSheet(
            "\n            QGroupBox {\n                font-weight: bold;\n                border: 1px solid #666;\n                border-radius: 5px;\n                margin-top: 10px;\n                padding-top: 10px;\n            }\n            QGroupBox::title {\n                subcontrol-origin: margin;\n                subcontrol-position: top left;\n                padding: 5px 10px;\n            }\n        "
        )
        vbox = QVBoxLayout()
        vbox.setSpacing(8)
        undo_layout = QHBoxLayout()
        undo_label = QLabel("Undo History:")
        undo_label.setToolTip("Maximum number of undo steps to remember (1-200)")
        self.undo_spinbox = QSpinBox()
        self.undo_spinbox.setRange(1, 200)
        self.undo_spinbox.setValue(50)
        self.undo_spinbox.setSuffix(" steps")
        self.undo_spinbox.setToolTip("Each step stores a full copy of all bounding boxes")
        self.undo_spinbox.valueChanged.connect(self._on_undo_limit_changed)
        undo_layout.addWidget(undo_label)
        undo_layout.addWidget(self.undo_spinbox)
        vbox.addLayout(undo_layout)
        cache_layout = QHBoxLayout()
        cache_label = QLabel("Path Cache:")
        cache_label.setToolTip("Maximum number of cached geometry paths (100-50000)")
        self.cache_spinbox = QSpinBox()
        self.cache_spinbox.setRange(100, 50000)
        self.cache_spinbox.setSingleStep(1000)
        self.cache_spinbox.setValue(10000)
        self.cache_spinbox.setSuffix(" entries")
        self.cache_spinbox.setToolTip("Higher values use more memory but improve performance")
        self.cache_spinbox.valueChanged.connect(self._on_cache_limit_changed)
        cache_layout.addWidget(cache_label)
        cache_layout.addWidget(self.cache_spinbox)
        vbox.addLayout(cache_layout)
        self.memory_label = QLabel()
        self.memory_label.setStyleSheet("color: #888; font-size: 10pt;")
        self.memory_label.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.memory_label)
        clear_btn = QPushButton("🗑️ Clear Undo History")
        clear_btn.setToolTip("Clear all undo/redo history to free memory")
        clear_btn.clicked.connect(self._on_clear_history)
        vbox.addWidget(clear_btn)
        self.undo_status_label = QLabel()
        self.undo_status_label.setStyleSheet("color: #666; font-size: 9pt;")
        self.undo_status_label.setAlignment(Qt.AlignCenter)
        self.undo_status_label.setWordWrap(True)
        vbox.addWidget(self.undo_status_label)
        group.setLayout(vbox)
        return group

    def refresh_layers(self, event=None) -> None:
        """Refill the combo box with the viewer's TurboBoxLayers.

        Runs on every change of the viewer's layer list. Afterwards the first
        TurboBoxLayer is the target, even if another one was selected before.
        """
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        if self.viewer is not None:
            for layer in self.viewer.layers:
                if isinstance(layer, TurboBoxLayer):
                    self.layer_combo.addItem(layer.name, layer)
        self.layer_combo.blockSignals(False)
        self._on_layer_change(self.layer_combo.currentIndex())

    def _current_layer(self) -> TurboBoxLayer | None:
        layer = self.layer_combo.currentData()
        if isinstance(layer, TurboBoxLayer):
            return layer
        return None

    def _create_layer(self) -> None:
        """Add a TurboBoxLayer to the viewer and make it the target.

        The layer has the viewer's number of dimensions (at least 2). Its
        ``image_shape`` comes from the selected Image layer, or else from the
        topmost one. The image's scale and translate are copied only if the
        image has as many dimensions as the layer.
        """
        logger.debug("_create_layer() button clicked")
        if self.viewer is None:
            logger.error("No viewer available!")
            self.status_label.setText("❌ ERROR: No viewer available")
            return
        try:
            logger.debug("Getting ndim from viewer...")
            ndim = max(2, getattr(self.viewer.dims, "ndim", 2))
            logger.debug("ndim =%s", ndim)
        except Exception as e:
            logger.warning("Failed to get ndim:%s", e)
            ndim = 3
        image_shape = None
        transform = {}
        try:
            logger.debug("Looking for image layer to get shape...")
            active_image = next(
                (lyr for lyr in self.viewer.layers.selection if isinstance(lyr, Image)), None
            )
            if active_image is None:
                active_image = next(
                    (lyr for lyr in reversed(self.viewer.layers) if isinstance(lyr, Image)), None
                )
            if active_image is not None:
                image_shape = tuple(active_image.data.shape[: active_image.ndim])  # drops an RGB(A) axis
                if active_image.ndim == ndim:  # then boxes are in this image's voxel coordinates
                    transform = {"scale": active_image.scale, "translate": active_image.translate}
                logger.debug("Found image:%s, shape=%s", active_image.name, image_shape)
            else:
                logger.debug("No image layer found, bboxes won't be clamped to bounds")
        except Exception as e:
            logger.warning("Error getting image_shape:%s", e)
        try:
            logger.debug("Creating TurboBoxLayer(ndim=%d, image_shape=%s)...", ndim, image_shape)
            layer = TurboBoxLayer(
                ndim=ndim, image_shape=image_shape, edge_width=5.0, edge_color="cyan", **transform
            )
            logger.debug("Layer created: %s", layer)
            layer.name = self._unique_layer_name()
            logger.debug("Layer name set to:%s", layer.name)
            logger.debug("Adding layer to viewer...")
            self.viewer.add_layer(layer)
            logger.debug("Layer added!")
            logger.debug("Setting layer as active...")
            self.viewer.layers.selection.active = layer
            logger.debug("Refreshing layer combo...")
            self.refresh_layers()
            logger.debug("Finding layer in combo box...")
            for index in range(self.layer_combo.count()):
                if self.layer_combo.itemData(index) is layer:
                    self.layer_combo.setCurrentIndex(index)
                    break
            success_msg = f"Added '{layer.name}' with {layer.ndim}D bounding boxes. Use Quick Create to add boxes."
            self.status_label.setText(success_msg)
            logger.info(success_msg)
        except Exception as e:
            error_msg = f"❌ ERROR creating layer: {e}"
            logger.exception(error_msg)
            self.status_label.setText(error_msg)

    def _unique_layer_name(self) -> str:
        base = "Bounding Boxes"
        existing = {layer.name for layer in self.viewer.layers}
        if base not in existing:
            return base
        suffix = 1
        while f"{base} ({suffix})" in existing:
            suffix += 1
        return f"{base} ({suffix})"

    def _on_layer_change(self, index: int) -> None:
        """Make the layer selected in the combo box the target and refresh the panel.

        Moves the ``bboxes`` and store ``session_ended`` connections from the
        previous target to the new one.
        """
        if self._watched_layer is not None:
            try:
                self._watched_layer.events.bboxes.disconnect(self._on_bboxes_changed)
                self._watched_layer._bbox_store.events.session_ended.disconnect(
                    self._on_store_session_ended
                )
                logger.debug("Disconnected bbox events from '%s'", self._watched_layer.name)
            except Exception as e:
                logger.warning("Could not disconnect from previous layer:%s", e)
        layer = self._current_layer()
        if layer is None:
            self.status_label.setText("No bounding box layer selected.")
            self.bbox_list.clear()
            self._watched_layer = None
            return
        try:
            layer.events.bboxes.connect(self._on_bboxes_changed)
            layer._bbox_store.events.session_ended.connect(self._on_store_session_ended)
            self._watched_layer = layer
            logger.debug("Connected bbox events to '%s'", layer.name)
        except Exception as e:
            logger.warning("Could not connect to layer events:%s", e)
        self._update_bbox_list()
        self._update_settings_display()
        self.status_label.setText(
            f"{self._count_text(layer)} Click to select or use napari tools to edit."
        )

    @staticmethod
    def _count_text(layer: TurboBoxLayer) -> str:
        total = len(layer.bounding_boxes)
        return f"{total} bounding box(es) ({layer.nshapes} on this view)."

    def _on_bboxes_changed(self, event=None) -> None:
        """Refresh the list, status and settings after the target's boxes change.

        A layer with ``sync_mode="live"`` emits ``bboxes`` on every mouse move
        of a drag. Rebuilding the list of all N boxes each time would slow the
        drag down, so while the store is in an edit session the refresh waits
        for the store's ``session_ended`` event.
        """
        layer = self._current_layer()
        if layer is not None and layer._bbox_store.in_edit_session:
            self._refresh_pending = True
            return
        self._refresh_pending = False
        # A drag release reports the same boxes twice: at session end and then
        # through the layer's bboxes event. Rebuild the list only if they differ.
        data = layer._bbox_store.data if layer is not None else None
        signature = None if data is None else (id(layer), data.shape, hash(data.tobytes()))
        if signature is None or signature != getattr(self, "_list_signature", None):
            self._update_bbox_list()
            self._list_signature = signature
        if layer:
            self.status_label.setText(
                f"{self._count_text(layer)} Click to select or use napari tools to edit."
            )
        self._update_settings_display()

    def _on_store_session_ended(self, event=None) -> None:
        """Run the refresh that ``_on_bboxes_changed`` held back during the session."""
        if getattr(self, "_refresh_pending", False):
            self._on_bboxes_changed()

    def _selected_bbox_index(self, layer: TurboBoxLayer) -> int | None:
        if not layer.selected_data:
            return None
        path_idx = next(iter(layer.selected_data))  # a shape index of the current view
        return layer._path_index_to_bbox_index(path_idx)

    def _start_draw_mode(self) -> None:
        """Put the target layer in napari's Add Rectangle mode to draw a box."""
        layer = self._current_layer()
        if layer is None or self.viewer is None:
            self.status_label.setText(
                "❌ No bounding box layer selected. Click 'Add Bounding Box Layer' first!"
            )
            return
        logger.debug("DRAW MODE ACTIVATED")
        self.viewer.layers.selection.active = layer
        logger.debug("Layer '%s' set as active", layer.name)
        self.status_label.setText(
            "🖱️ DRAW MODE ACTIVE: Click and drag anywhere in the viewer to draw a bounding box!"
        )
        # Drawing uses the layer's Add Rectangle mode, not a viewer mouse_drag_callback:
        # napari >= 0.6 calls a non-generator drag callback only on press, and the layer's
        # own drag handler still runs. The layer turns the finished rectangle into a box.
        # A flat hidden axis is expanded (to the full image depth when the image shape is
        # known) and the box is clamped to the image. A click without a drag adds nothing.
        from napari.layers.shapes._shapes_constants import Mode

        layer.mode = Mode.ADD_RECTANGLE
        logger.info("Mouse drag callback registered for drawing")

    def _create_from_image(self) -> None:
        """Add one box covering the topmost Image layer, 0 to shape - 1 on every axis.

        An image with more dimensions than the layer contributes its trailing
        axes. With fewer dimensions the box is not added and the error is only
        logged.
        """
        layer = self._current_layer()
        if layer is None or self.viewer is None:
            return
        image = next((lyr for lyr in reversed(self.viewer.layers) if isinstance(lyr, Image)), None)
        if image is None:
            self.status_label.setText("No image layer available to derive bounds.")
            return
        mins = np.zeros(image.ndim)
        maxs = np.asarray(image.data.shape[: image.ndim], dtype=float) - 1.0  # last voxel index
        if image.ndim > layer.ndim:
            mins = mins[-layer.ndim :]
            maxs = maxs[-layer.ndim :]
        elif image.ndim < layer.ndim:
            logger.warning("image.ndim (%s) < layer.ndim (%s)", image.ndim, layer.ndim)
        try:
            layer.add_boxes([[mins, maxs]])
            logger.debug("add_boxes() completed")
        except Exception as e:
            logger.exception("ERROR in add_boxes():%s", e)
        self._update_status()

    def _create_from_viewport(self) -> None:
        """Add a box in the middle of the viewer's data range at the current slice.

        On each displayed axis the box covers the middle half (25% to 75%) of
        ``viewer.dims.range``, the extent of all layers. It does not follow the
        camera's zoom or pan. On hidden axes the box is one slice thick,
        starting at the current slice. The box is then clamped to the image
        (when its shape is known). If the layer rejects the box, the error goes
        to the status label.
        """
        layer = self._current_layer()
        if layer is None or self.viewer is None:
            return
        dims = self.viewer.dims
        displayed = list(dims.displayed)
        not_displayed = list(dims.not_displayed)
        logger.debug("Viewport: displayed=%s, not_displayed=%s", displayed, not_displayed)
        # dims.point and dims.range are world coordinates. Boxes are stored in the
        # layer's data coordinates (scale and translate copied from the image).
        point = np.asarray(layer.world_to_data(np.asarray(dims.point, dtype=float)), dtype=float)
        lo = np.asarray(layer.world_to_data([r[0] for r in dims.range]), dtype=float)
        hi = np.asarray(layer.world_to_data([r[1] for r in dims.range]), dtype=float)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)  # a negative scale flips the order
        offset = dims.ndim - layer.ndim  # layer axes are the trailing world axes
        mins = point.copy()
        maxs = point + 1.0  # one-slice-thick box along the hidden axes (minimum extent 1)
        for world_axis in displayed:
            axis = world_axis - offset
            if 0 <= axis < layer.ndim:
                span = hi[axis] - lo[axis]
                mins[axis] = lo[axis] + span * 0.25
                maxs[axis] = lo[axis] + span * 0.75
        box = layer._clamp_to_valid_range(np.asarray([[mins, maxs]]))  # [0, shape - 1], extent >= 1
        logger.debug("Creating viewport bbox: mins=%s, maxs=%s", box[0, 0], box[0, 1])
        try:
            layer.add_boxes(box)
        except Exception as e:
            self.status_label.setText(f"❌ Could not fit the viewport: {e}")
            logger.exception("Viewport bbox error:%s", e)
            return
        self._update_status()

    def _clear_boxes(self) -> None:
        layer = self._current_layer()
        if layer is None:
            return
        layer.clear_boxes()
        self._update_status()

    def _update_status(self) -> None:
        """Rebuild the box list and show the box count and selected box in the status label."""
        layer = self._current_layer()
        if layer is None:
            self.status_label.setText("No bounding box layer selected.")
            self.bbox_list.clear()
            return
        self._update_bbox_list()
        text = self._count_text(layer)
        idx = self._selected_bbox_index(layer)
        if idx is not None and idx < layer.bounding_boxes.shape[0]:
            mins, maxs = layer.bounding_boxes[idx]
            text += f" Selected #{idx}: mins={mins.astype(int)}, maxs={maxs.astype(int)}."
        self.status_label.setText(text)

    def _update_bbox_list(self) -> None:
        """Rebuild the list with one row per box of the target layer (row i is box i)."""
        self.bbox_list.clear()
        layer = self._current_layer()
        if layer is None:
            return
        boxes = layer.bounding_boxes
        for i in range(boxes.shape[0]):
            mins, maxs = boxes[i]
            label = f"BBox {i}: [{mins.astype(int).tolist()}] → [{maxs.astype(int).tolist()}]"
            self.bbox_list.addItem(label)

    def _on_list_selection_changed(self) -> None:
        """Do nothing: "Select in Layer" and "Delete Selected" act on the selected row."""
        pass

    def _select_bbox_from_list(self) -> None:
        """Select the shape of the box chosen in the list.

        This works only if the box is drawn in the current view. In a 2D view
        that means the box crosses the current slice.
        """
        layer = self._current_layer()
        if layer is None:
            return
        selected_items = self.bbox_list.selectedItems()
        if not selected_items:
            self.status_label.setText("⚠️  Select a bbox from the list first!")
            return
        bbox_idx = self.bbox_list.row(selected_items[0])
        path_idx = layer._bbox_index_to_path_index(bbox_idx)
        if path_idx is not None:
            layer.selected_data = {path_idx}
            self.status_label.setText(f"✅ Selected BBox {bbox_idx} in layer")
        else:
            self.status_label.setText(f"❌ Could not find BBox {bbox_idx} in layer")

    def _delete_selected_bbox(self) -> None:
        """Delete the box chosen in the list from the layer (one undo step)."""
        layer = self._current_layer()
        if layer is None:
            return
        selected_items = self.bbox_list.selectedItems()
        if not selected_items:
            self.status_label.setText("⚠️  Select a bbox from the list first!")
            return
        bbox_idx = self.bbox_list.row(selected_items[0])
        boxes = layer.bounding_boxes
        if bbox_idx < boxes.shape[0]:
            new_boxes = np.delete(boxes, bbox_idx, axis=0)
            layer.bounding_boxes = new_boxes
            self.status_label.setText(f"✅ Deleted BBox {bbox_idx}")
            self._update_status()
        else:
            self.status_label.setText(f"❌ Invalid bbox index: {bbox_idx}")

    def _export_boxes(self) -> None:
        """Export the target's boxes as native .npy/.json, COCO JSON or YOLO labels.

        COCO and YOLO hold 2D boxes per image, so a 3D box becomes one 2D box on
        each slice it covers (see ``export.py``). The panel has no class
        control: COCO output has the single category ``object`` (id 1) and
        YOLO output uses class 0 for every box.
        """
        layer = self._current_layer()
        if layer is None:
            self.status_label.setText("❌ No bbox layer selected!")
            return
        # Count the stored boxes, not nshapes: nshapes is 0 when no box crosses
        # the current slice, and the export should still run then.
        if len(layer.bounding_boxes) == 0:
            self.status_label.setText("❌ No bounding boxes to export!")
            return
        choices = ["Native (.npy / .json)", "COCO JSON (per-slice)", "YOLO labels (per-slice)"]
        fmt, ok = QInputDialog.getItem(self, "Export format", "Format:", choices, 0, False)
        if not ok:
            return
        try:
            boxes = layer.bounding_boxes
            if fmt.startswith("Native"):
                self._export_native(boxes)
            elif fmt.startswith("COCO"):
                self._export_coco(layer, boxes)
            else:
                self._export_yolo(layer, boxes)
        except Exception as e:
            self.status_label.setText(f"❌ Export failed: {e}")
            logger.exception("Export error:%s", e)

    def _export_image_shape(self, layer) -> tuple[int, ...] | None:
        """Return the image shape that bounds the COCO/YOLO output.

        This is the layer's ``image_shape``, else the first ``layer.ndim`` axes
        of the first Image layer in the viewer, else None. With None the
        converters derive H and W from the boxes and warn.
        """
        shape = getattr(layer, "_image_shape", None)
        if shape is not None:
            return tuple(int(s) for s in shape)
        if self.viewer is not None:
            for lyr in self.viewer.layers:
                if isinstance(lyr, Image):
                    return tuple(int(s) for s in lyr.data.shape[: layer.ndim])
        return None

    def _export_native(self, boxes: np.ndarray) -> None:
        """Save the (N, 2, D) array as .npy, or as nested lists if the name ends in .json."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Bounding Boxes", "", "NumPy Array (*.npy);;JSON (*.json);;All Files (*)"
        )
        if not filename:
            return
        if filename.endswith(".json"):
            import json

            with open(filename, "w") as f:
                json.dump(boxes.tolist(), f, indent=2)
        else:
            if not filename.endswith(".npy"):
                filename += ".npy"
            np.save(filename, boxes)
        self.status_label.setText(f"✅ Exported {boxes.shape[0]} bboxes to {filename}")
        logger.info("Exported bboxes to:%s", filename)

    def _export_coco(self, layer, boxes: np.ndarray) -> None:
        import json

        from .export import boxes_to_coco

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export COCO JSON", "", "COCO JSON (*.json);;All Files (*)"
        )
        if not filename:
            return
        if not filename.endswith(".json"):
            filename += ".json"
        coco = boxes_to_coco(boxes, self._export_image_shape(layer))
        with open(filename, "w") as f:
            json.dump(coco, f, indent=2)
        self.status_label.setText(
            f"✅ Exported {len(coco['annotations'])} COCO annotations to {filename}"
        )
        logger.info("Exported COCO annotations to:%s", filename)

    def _export_yolo(self, layer, boxes: np.ndarray) -> None:
        """Write one YOLO .txt per image into a chosen directory, overwriting same-named files."""
        import os

        from .export import boxes_to_yolo

        directory = QFileDialog.getExistingDirectory(self, "Export YOLO labels to directory")
        if not directory:
            return
        files = boxes_to_yolo(boxes, self._export_image_shape(layer))
        for name, content in files.items():
            with open(os.path.join(directory, name), "w") as f:
                f.write(content)
        self.status_label.setText(f"✅ Exported {len(files)} YOLO label file(s) to {directory}")
        logger.info("Exported%sYOLO label files to:%s", len(files), directory)

    def _import_boxes(self) -> None:
        """Add the boxes from a native .npy or .json file to the target layer.

        The file must hold an (N, 2, D) array with D equal to the layer's ndim.
        The boxes go through ``add_boxes``: a flat hidden axis is expanded, and
        boxes outside the image or thinner than 1 are rejected. Errors are
        shown in the status label. COCO and YOLO files cannot be imported.
        """
        layer = self._current_layer()
        if layer is None:
            self.status_label.setText("❌ No bbox layer selected! Create one first.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Bounding Boxes", "", "NumPy Array (*.npy);;JSON (*.json);;All Files (*)"
        )
        if not filename:
            return
        try:
            if filename.endswith(".json"):
                import json

                with open(filename) as f:
                    boxes = np.array(json.load(f))
            else:
                boxes = np.load(filename)
            logger.info("Loaded boxes: shape=%s, dtype=%s", boxes.shape, boxes.dtype)
            if boxes.ndim != 3 or boxes.shape[1] != 2:
                raise ValueError(f"Invalid box shape: {boxes.shape}. Expected (N, 2, ndim)")
            if boxes.shape[2] != layer.ndim:
                raise ValueError(
                    f"Dimension mismatch: boxes have {boxes.shape[2]}D but layer is {layer.ndim}D"
                )
            layer.add_boxes(boxes)
            self.status_label.setText(f"✅ Imported {boxes.shape[0]} bboxes from {filename}")
            logger.info("Imported%sbboxes", boxes.shape[0])
            self._update_status()
        except Exception as e:
            self.status_label.setText(f"❌ Import failed: {e}")
            logger.exception("Import error:%s", e)

    def _on_undo_limit_changed(self, value: int) -> None:
        """Update layer's undo limit when spinbox changes."""
        layer = self._current_layer()
        if layer is not None:
            layer.undo_limit = value
            self._update_settings_display()
            logger.debug("Undo limit changed to%s", value)

    def _on_cache_limit_changed(self, value: int) -> None:
        """Update layer's cache limit when spinbox changes."""
        layer = self._current_layer()
        if layer is not None:
            layer.cache_limit = value
            self._update_settings_display()
            logger.debug("Cache limit changed to %d", value)

    def _on_clear_history(self) -> None:
        """Clear undo/redo history."""
        layer = self._current_layer()
        if layer is not None:
            layer.clear_undo_history()
            self._update_settings_display()
            self.status_label.setText("✅ Undo history cleared")
            logger.debug("Undo history cleared")

    def _update_settings_display(self) -> None:
        """Show the target's limits, the undo memory use and the undo/redo state.

        The spin boxes are set with their signals blocked, so this does not
        write the values back to the layer. The undo/redo text names the
        action that undo or redo would apply next.
        """
        layer = self._current_layer()
        if layer is None:
            self.memory_label.setText("📊 Memory: N/A")
            self.undo_status_label.setText("No layer selected")
            return
        self.undo_spinbox.blockSignals(True)
        self.cache_spinbox.blockSignals(True)
        self.undo_spinbox.setValue(layer.undo_limit)
        self.cache_spinbox.setValue(layer.cache_limit)
        self.undo_spinbox.blockSignals(False)
        self.cache_spinbox.blockSignals(False)
        info = layer.get_undo_info()
        mem_kb = info["memory_usage_kb"]
        if mem_kb < 1:
            mem_str = f"{mem_kb * 1024:.0f} bytes"
        elif mem_kb < 1024:
            mem_str = f"{mem_kb:.1f} KB"
        else:
            mem_str = f"{mem_kb / 1024:.1f} MB"
        self.memory_label.setText(f"📊 Memory: {mem_str}")
        parts = []
        if info["can_undo"]:
            parts.append(f"↶ {info['undo_description']}")
        if info["can_redo"]:
            parts.append(f"↷ {info['redo_description']}")
        if parts:
            status = " | ".join(parts)
        else:
            status = "No undo/redo available"
        history_info = f"({info['history_position'] + 1}/{info['history_size']} states)"
        self.undo_status_label.setText(f"{status}\n{history_info}")
