"""Settings panel for one TurboBoxLayer: undo limit, path cache limit, undo memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .layer import TurboBoxLayer


class BBoxSettingsWidget(QWidget):
    """
    Settings panel for one TurboBoxLayer.

    Sets the undo history limit and the path cache limit, shows the memory
    used by the undo history and the next undo/redo action, and has a button
    that clears the undo history.

    Nothing in the package creates this widget, and the plugin manifest does
    not list it. The control panel (``BoundingBoxControlWidget``) has its own
    copy of these settings.
    """

    def __init__(self, layer: TurboBoxLayer, parent=None):
        super().__init__(parent)
        self.layer = layer
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        self.settings_group = QGroupBox("⚙️ Advanced Settings")
        self.settings_group.setCheckable(True)
        self.settings_group.setChecked(False)
        self.settings_group.setStyleSheet(
            "\n            QGroupBox {\n                font-weight: bold;\n                border: 1px solid #666;\n                border-radius: 5px;\n                margin-top: 10px;\n                padding-top: 10px;\n            }\n            QGroupBox::title {\n                subcontrol-origin: margin;\n                subcontrol-position: top left;\n                padding: 5px 10px;\n            }\n        "
        )
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(8)
        undo_layout = QHBoxLayout()
        undo_label = QLabel("Undo History Limit:")
        undo_label.setToolTip("Maximum number of undo steps to remember (1-200)")
        self.undo_spinbox = QSpinBox()
        self.undo_spinbox.setRange(1, 200)
        self.undo_spinbox.setValue(self.layer.undo_limit)
        self.undo_spinbox.setSuffix(" steps")
        self.undo_spinbox.setToolTip("Each step stores a full copy of all bounding boxes")
        self.undo_spinbox.valueChanged.connect(self._on_undo_limit_changed)
        undo_layout.addWidget(undo_label)
        undo_layout.addWidget(self.undo_spinbox)
        settings_layout.addLayout(undo_layout)
        cache_layout = QHBoxLayout()
        cache_label = QLabel("Path Cache Limit:")
        cache_label.setToolTip("Maximum number of cached geometry paths (100-50000)")
        self.cache_spinbox = QSpinBox()
        self.cache_spinbox.setRange(100, 50000)
        self.cache_spinbox.setSingleStep(1000)
        self.cache_spinbox.setValue(self.layer.cache_limit)
        self.cache_spinbox.setSuffix(" entries")
        self.cache_spinbox.setToolTip("Higher values use more memory but improve performance")
        self.cache_spinbox.valueChanged.connect(self._on_cache_limit_changed)
        cache_layout.addWidget(cache_label)
        cache_layout.addWidget(self.cache_spinbox)
        settings_layout.addLayout(cache_layout)
        self.memory_label = QLabel()
        self.memory_label.setStyleSheet("color: #888; font-size: 10pt;")
        self.memory_label.setAlignment(Qt.AlignCenter)
        self._update_memory_display()
        settings_layout.addWidget(self.memory_label)
        self.clear_button = QPushButton("🗑️ Clear Undo History")
        self.clear_button.setToolTip("Clear all undo/redo history to free memory")
        self.clear_button.clicked.connect(self._on_clear_history)
        settings_layout.addWidget(self.clear_button)
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self._update_status_display()
        settings_layout.addWidget(self.status_label)
        self.settings_group.setLayout(settings_layout)
        layout.addWidget(self.settings_group)
        layout.addStretch()
        self.setLayout(layout)
        if hasattr(self.layer, "events") and hasattr(self.layer.events, "bboxes"):
            self.layer.events.bboxes.connect(self._on_layer_changed)

    def _on_undo_limit_changed(self, value: int):
        """Apply the spin box value as the layer's undo limit."""
        self.layer.undo_limit = value
        self._update_memory_display()

    def _on_cache_limit_changed(self, value: int):
        """Apply the spin box value as the layer's path cache limit."""
        self.layer.cache_limit = value
        self._update_memory_display()

    def _on_clear_history(self):
        """Clear undo/redo history."""
        self.layer.clear_undo_history()
        self._update_memory_display()
        self._update_status_display()

    def _on_layer_changed(self, event=None):
        """Refresh the undo/redo text after the layer's boxes change.

        The memory label is refreshed only by the settings controls.
        """
        self._update_status_display()

    def _update_memory_display(self):
        """Show the estimated memory use of the undo history."""
        info = self.layer.get_undo_info()
        mem_kb = info["memory_usage_kb"]
        if mem_kb < 1:
            mem_str = f"{mem_kb * 1024:.0f} bytes"
        elif mem_kb < 1024:
            mem_str = f"{mem_kb:.1f} KB"
        else:
            mem_str = f"{mem_kb / 1024:.1f} MB"
        self.memory_label.setText(f"📊 Memory: {mem_str}")

    def _update_status_display(self):
        """Show the actions undo and redo would apply next and the history position."""
        info = self.layer.get_undo_info()
        parts = []
        if info["can_undo"]:
            parts.append(f"↶ {info['undo_description']}")
        if info["can_redo"]:
            parts.append(f"↷ {info['redo_description']}")
        if parts:
            status = " | ".join(parts)
        else:
            status = "No undo/redo available"
        history_info = f"History: {info['history_position'] + 1}/{info['history_size']}"
        self.status_label.setText(f"{status}\n{history_info}")
