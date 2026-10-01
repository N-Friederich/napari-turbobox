"""Record the README demo GIF and screenshots from a real napari session.

    python docs/make_demo.py

The script needs a display and ffmpeg on PATH. It is not part of the test
suite. On the first run it downloads the C. elegans nuclei archive of the
first tutorial (84 MB, Long et al. 2022, CC-BY-4.0) into the tutorial data
cache, checks its MD5 and derives one box per nucleus (555 boxes). It picks
one nucleus (``plan_scenes``) and moves its box off the nucleus and shrinks it,
so the recording can correct it. It then opens the synchronized four-viewer
layout (XY editable; XZ, YZ and 3D read-only), records the edits and writes
three files:

- ``docs/demo.gif``: the four views while the box is dragged onto its nucleus
  and fitted to it, a slice is stepped through, the fit is undone and redone,
  and the 3D view is turned;
- ``docs/screenshot-views.png``: the four views at full resolution;
- ``docs/screenshot-panel.png``: the XY viewer window with the BBox Control
  Panel.

Every edit is a real Qt mouse or key event sent to the XY canvas (the 3D turn
to the 3D canvas), so it goes through napari's own event handling into the
plugin, as a user's edit does. Three things are drawn into the frames on top
of what napari renders: the pointer, a dashed line in XZ and YZ at the depth
of the XY slice, and a cyan outline around the edited box in every view.
The outline is computed from the vertices those views draw, so it appears only
where napari shows the box. Each step checks its result in the store and in the
other views and stops the recording if a check fails; the checked values are
printed at the end (``--facts`` also writes them to a JSON file). Run it again
after a change to the plugin; the steps are the same each time.

The script uses some of napari's private attributes (the Qt window, the canvas
transform, a layer's slice input) and has been run with napari 0.7.0 and PyQt6
on macOS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# Record this checkout's plugin, also when another copy is installed.
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tutorials"))

# The part of the volume the 2D views show (z, y and x ranges in voxels). The
# worm lies along x; the y and x ranges are centred on the edited nuclei once
# they are chosen (see produce).
REGION = {0: (38.0, 103.0), 1: (40.0, 105.0), 2: (150.0, 350.0)}
X_SPAN = 200.0

WIDTH = 900  # width of the GIF in pixels
GAP = 4  # space between the views
CAPTION_H = 38
FPS = 10
TITLES = ("XY · editable", "XZ · read-only", "YZ · read-only", "3D · read-only")
PLANES = (((0, 1, 2), 0), ((1, 0, 2), 1), ((2, 0, 1), 2))  # dims.order and sliced axis of XY, XZ, YZ
BACKGROUND = (38, 41, 48)  # napari's dark theme
CAPTION_TEXT = (230, 230, 230)


def load_sample():
    """Image, boxes, label image and label ids of the first tutorial's volume (same download and checks)."""
    import tifffile
    import tutorial_utils as tu

    data = tu.data_dir("celegans_nuclei")
    archive = tu.fetch(
        "https://zenodo.org/api/records/5942575/files/c_elegans_nuclei.zip/content",
        data / "c_elegans_nuclei.zip",
        size=84_403_531,
        md5="7d616d91c4a6cce75a7d4dd37d7e666a",
    )
    raw = tu.extract_member(archive, "c_elegans_nuclei/train/images/C18G1_2L1_1.tif", data / "C18G1_2L1_1_raw.tif")
    gt = tu.extract_member(archive, "c_elegans_nuclei/train/masks/C18G1_2L1_1.tif", data / "C18G1_2L1_1_masks.tif")
    image, labels = tifffile.imread(raw), tifffile.imread(gt)
    boxes, ids = tu.labels_to_boxes(labels)
    return image, boxes, labels, ids


def _on_slice(boxes: np.ndarray, z: float) -> np.ndarray:
    """Boolean mask of the boxes that the XY view draws at slice ``z``."""
    return (boxes[:, 0, 0] <= z) & (z <= boxes[:, 1, 0])


def _touching(rect, boxes: np.ndarray, margin: float) -> np.ndarray:
    """Mask of ``boxes`` that overlap or touch the (y0, x0, y1, x1) rectangle grown by ``margin`` in y and x.

    Boxes are inclusive voxel ranges, so two boxes that share an edge value share
    a row of voxels; the margin makes such boxes count as touching.
    """
    y0, x0, y1, x1 = rect
    return ((boxes[:, 0, 1] <= y1 + margin) & (y0 - margin <= boxes[:, 1, 1])
            & (boxes[:, 0, 2] <= x1 + margin) & (x0 - margin <= boxes[:, 1, 2]))


def _rect(box) -> tuple[float, float, float, float]:
    return float(box[0, 1]), float(box[0, 2]), float(box[1, 1]), float(box[1, 2])


def _crowding(boxes: np.ndarray, k: int) -> int:
    """How many other boxes the XZ and YZ views through the centre of box ``k`` draw over or next to it."""
    box, centre = boxes[k], np.round(boxes[k].mean(axis=0))  # the slices the views are set to
    z_touch = (boxes[:, 0, 0] <= box[1, 0] + 1) & (box[0, 0] - 1 <= boxes[:, 1, 0])
    on_xz = (boxes[:, 0, 1] <= centre[1]) & (centre[1] <= boxes[:, 1, 1])
    on_yz = (boxes[:, 0, 2] <= centre[2]) & (centre[2] <= boxes[:, 1, 2])
    x_touch = (boxes[:, 0, 2] <= box[1, 2] + 1) & (box[0, 2] - 1 <= boxes[:, 1, 2])
    y_touch = (boxes[:, 0, 1] <= box[1, 1] + 1) & (box[0, 1] - 1 <= boxes[:, 1, 1])
    near = z_touch & ((on_xz & x_touch) | (on_yz & y_touch))
    near[k] = False
    return int(near.sum())


def plan_scenes(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> dict:
    """Choose the nucleus whose box is corrected in the recording.

    Box ``k`` belongs to a nucleus in the middle of the shown stretch whose box
    keeps at least one voxel of space to every other box drawn on its XY slice,
    and which at most one other box crowds in the XZ and YZ views through its
    centre; among these, the largest nucleus is taken, so the box is easy to
    see. Before the recording, box ``k`` is moved ``dx`` voxels along x onto a
    spot of the XY slice where no nucleus is (no labelled voxel within one
    voxel of it) and no other box is within 1.5 voxels, and its y and x extent
    are cut to 70 % (but still reach one voxel past the XZ and YZ slices
    through the nucleus); the recording drags it back and pulls its corner out
    until it matches the nucleus again.
    """
    boxes = np.asarray(boxes, float)
    centre = boxes.mean(axis=1)
    extent = boxes[:, 1] - boxes[:, 0]
    middle = (170 <= centre[:, 2]) & (centre[:, 2] <= 400) & (45 <= centre[:, 1]) & (centre[:, 1] <= 95)
    for k in np.argsort(-np.prod(extent, axis=1)):
        if not middle[k] or _crowding(boxes, k) > 1:
            continue
        z0 = round(centre[k, 0])
        others = _on_slice(boxes, z0)
        others[k] = False
        drawn = boxes[others]
        if _touching(_rect(boxes[k]), drawn, 1.0).any():
            continue
        options = []
        for dx in (*np.arange(-60.0, -19.0, 5.0), *np.arange(20.0, 61.0, 5.0)):
            prepared = boxes[k].copy()
            prepared[1, 1:] = prepared[0, 1:] + np.round(0.7 * extent[k, 1:])
            # once dragged back, the cut box must still reach one voxel past the
            # XZ and YZ slices (at the rounded centre), or those views miss it
            slices = np.round(centre[k, 1:])
            prepared[1, 1:] = np.maximum(prepared[1, 1:], slices + 1)
            prepared[:, 2] += dx
            y0, x0, y1, x1 = (int(v) for v in _rect(prepared))
            if x0 < 155 or x1 > 415 or _touching(_rect(prepared), drawn, 1.5).any():
                continue
            if labels[z0, max(y0 - 1, 0):y1 + 2, max(x0 - 1, 0):x1 + 2].any():
                continue  # a nucleus under or next to the misplaced box
            darkness = float(image[z0, y0:y1 + 1, x0:x1 + 1].mean())
            options.append((darkness, abs(dx), float(dx), prepared))
        if not options:
            continue
        darkness, _, dx, prepared = min(options, key=lambda o: o[:2])
        return {"k": int(k), "dx": dx, "gt": boxes[k].copy(), "prepared": prepared, "z0": int(z0),
                "background_mean": round(darkness, 1), "label_k": int(ids[k]), "crowding_k": _crowding(boxes, k)}
    raise LookupError("no suitable nucleus in the shown stretch")


# --------------------------------------------------------------------- viewers
def build_layout(image: np.ndarray, boxes: np.ndarray):
    """The four viewers and their layers, shown, in napari's dark theme."""
    import napari
    from napari.settings import get_settings
    from qtpy.QtWidgets import QApplication

    from napari_turbobox import create_synchronized_bbox_layers

    get_settings().appearance.theme = "dark"
    get_settings().application.gui_notification_level = "warning"
    viewers = []
    for title, (order, _axis) in zip(TITLES[:3], PLANES):
        viewer = napari.Viewer(title=title, show=True)
        viewer.add_image(image, name="image")
        viewer.dims.order = order
        viewers.append(viewer)
    viewer3d = napari.Viewer(title=TITLES[3], show=True)
    viewer3d.add_image(image, name="image", rendering="mip")
    viewer3d.dims.ndisplay = 3  # before the layers are made: the 3D layer then updates when a drag ends
    viewers.append(viewer3d)
    for viewer, size in zip(viewers, ((1100, 520), (1100, 520), (520, 520), (900, 520))):
        viewer.window._qt_window.resize(*size)
    layers = create_synchronized_bbox_layers(
        viewers[0], viewers[1:], image_shape=image.shape, edge_color="yellow", edge_width=1.0
    )
    layers[3].current_edge_width = 0.2  # thin, half-transparent wireframes keep the 3D view readable
    layers[3].opacity = 0.4
    layers[0].add_boxes(boxes)
    for _ in range(20):
        QApplication.processEvents()
    return viewers, layers


def frame_region(viewer) -> None:
    """Centre a 2D viewer's camera on REGION and zoom so that it fills the canvas."""
    canvas = viewer.window._qt_viewer.canvas.native
    displayed = list(viewer.dims.displayed)
    viewer.camera.center = tuple(sum(REGION[d]) / 2 for d in displayed)
    heights = [REGION[d][1] - REGION[d][0] for d in displayed]
    viewer.camera.zoom = min(canvas.height() / heights[0], canvas.width() / heights[1])


def world_to_canvas(viewer, world) -> tuple[float, float]:
    """A world point in canvas coordinates (logical pixels) of a 2D viewer.

    The inverse of napari's own ``_map_canvas2world``; the round trip is
    checked, so a change in napari's canvas transform stops the recording.
    """
    vispy_canvas = viewer.window._qt_viewer.canvas
    transform = vispy_canvas.view.transform * vispy_canvas.view.scene.transform
    displayed = list(viewer.dims.displayed)
    x, y = transform.map([world[displayed[1]], world[displayed[0]]])[:2]
    back = vispy_canvas._map_canvas2world((x, y), vispy_canvas.view)
    assert np.allclose([back[d] for d in displayed], [world[d] for d in displayed], atol=1e-3), (world, back)
    return float(x), float(y)


def project_3d(viewer, world) -> tuple[float, float]:
    """A world point in canvas coordinates of a 3D viewer (perspective divide included)."""
    vispy_canvas = viewer.window._qt_viewer.canvas
    transform = vispy_canvas.view.transform * vispy_canvas.view.scene.transform
    mapped = transform.map([world[2], world[1], world[0]])
    return float(mapped[0] / mapped[3]), float(mapped[1] / mapped[3])


def drawn_path(layer, index: int):
    """The vertices a layer draws for box ``index`` (a copy), or None if it draws none."""
    position = layer._bbox_index_to_path_index(index)
    return None if position is None else np.array(layer.data[position], float)


def shown_box(layer, index: int):
    """The (min, max) of box ``index`` as a read-only layer draws it, on its displayed axes; None if not drawn."""
    position = layer._bbox_index_to_path_index(index)
    if position is None:
        return None
    path = np.asarray(layer.data[position], float)
    displayed = list(layer._slice_input.displayed)
    return path[:, displayed].min(0), path[:, displayed].max(0)


# ------------------------------------------------------------------- recording
class Recorder:
    """Drives the XY canvas with real Qt events and composes the four views into frames."""

    def __init__(self, viewers, layers) -> None:
        self.viewers, self.layers = viewers, layers
        self.canvas = viewers[0].window._qt_viewer.canvas.native
        self.frames: list[np.ndarray] = []
        self.caption = ""
        self.cursor: tuple[float, float] | None = None  # in XY canvas coordinates
        self.cursor3d: tuple[float, float] | None = None  # in 3D canvas coordinates
        self.ring = 0.0
        self.slice_marks = True
        self.highlight: int | None = None  # store index of the box outlined in cyan in every view
        self.beats: dict[str, int] = {}
        # One scale for all 2D views: WIDTH pixels for the x range.
        nz, ny, nx = (REGION[d][1] - REGION[d][0] for d in range(3))
        h_xy, h_z = round(WIDTH * ny / nx), round(WIDTH * nz / nx)
        w_yz = h_xy  # YZ shows y across, at the same scale
        bottom = h_xy + h_z + 2 * GAP
        self.rects = (
            (0, 0, WIDTH, h_xy),
            (0, h_xy + GAP, WIDTH, h_z),
            (0, bottom, w_yz, h_z),
            (w_yz + GAP, bottom, WIDTH - w_yz - GAP, h_z),
        )
        self.height = bottom + h_z + CAPTION_H

    def mark(self, name: str) -> None:
        self.beats[name] = len(self.frames)

    # -- composing a frame --------------------------------------------------
    def _tile(self, index: int):
        """The canvas of one viewer, cropped to REGION (2D) or to the tile's aspect (3D), as a QImage."""
        from qtpy.QtGui import QImage

        viewer = self.viewers[index]
        shot = np.ascontiguousarray(np.asarray(viewer.screenshot(canvas_only=True, flash=False))[..., :3])
        canvas = viewer.window._qt_viewer.canvas.native
        ratio = shot.shape[1] / canvas.width()  # device pixels per logical pixel
        _x, _y, w, h = self.rects[index]
        if index < 3:
            corner0 = world_to_canvas(viewer, [REGION[d][0] for d in range(3)])
            corner1 = world_to_canvas(viewer, [REGION[d][1] for d in range(3)])
            x0, x1 = sorted((corner0[0], corner1[0]))
            y0, y1 = sorted((corner0[1], corner1[1]))
        else:
            cw, ch = canvas.width(), canvas.height()
            scale = min(cw / w, ch / h)
            x0, y0 = (cw - w * scale) / 2, (ch - h * scale) / 2
            x1, y1 = x0 + w * scale, y0 + h * scale
        r0, r1 = max(0, round(y0 * ratio)), min(shot.shape[0], round(y1 * ratio))
        c0, c1 = max(0, round(x0 * ratio)), min(shot.shape[1], round(x1 * ratio))
        crop = np.ascontiguousarray(shot[r0:r1, c0:c1])
        image = QImage(crop.data, crop.shape[1], crop.shape[0], 3 * crop.shape[1], QImage.Format.Format_RGB888).copy()
        return image, (x0, y0, x1, y1)

    def compose(self, scale: float = 1.0, caption_strip: bool = True) -> np.ndarray:
        """One frame: the four views, their titles, the slice marks, the pointer and the caption."""
        from qtpy.QtCore import QPointF, QRectF, Qt
        from qtpy.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()
        width = round(WIDTH * scale)
        height = round((self.height if caption_strip else self.height - CAPTION_H) * scale)
        frame = QImage(width, height, QImage.Format.Format_RGB888)
        frame.fill(QColor(*BACKGROUND))
        painter = QPainter(frame)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        crops = []
        for index, rect in enumerate(self.rects):
            tile, crop = self._tile(index)
            crops.append(crop)
            target = QRectF(*(v * scale for v in rect))
            painter.drawImage(target, tile)
        if self.slice_marks:
            self._draw_slice_marks(painter, scale)
        font = QFont()
        font.setPixelSize(round(13 * scale))
        font.setBold(True)
        painter.setFont(font)
        for title, rect in zip(TITLES, self.rects):
            x, y = (rect[0] + 6) * scale, (rect[1] + 5) * scale
            box = painter.boundingRect(QRectF(x, y, 400 * scale, 30 * scale), 0, title)
            painter.fillRect(box.adjusted(-4 * scale, -2 * scale, 4 * scale, 2 * scale), QColor(0, 0, 0, 150))
            painter.setPen(QColor(235, 235, 235))
            painter.drawText(box, 0, title)
        if self.caption:
            font.setPixelSize(round(15 * scale))
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QColor(*CAPTION_TEXT))
            caption_rect = QRectF(0, (self.height - CAPTION_H) * scale, width, CAPTION_H * scale)
            painter.drawText(caption_rect, int(Qt.AlignmentFlag.AlignCenter), self.caption)
        if self.highlight is not None:
            self._draw_highlight(painter, crops, scale)
        for cursor, index in ((self.cursor, 0), (self.cursor3d, 3)):
            if cursor is None:
                continue
            rx, ry, rw, rh = self.rects[index]
            painter.setClipRect(QRectF(rx * scale, ry * scale, rw * scale, rh * scale))
            px, py = self._to_frame(index, cursor, crops, scale)
            if self.ring > 0:
                radius = (6 + 14 * self.ring) * scale
                pen = QPen(QColor(255, 255, 255, round(255 * (1 - self.ring))))
                pen.setWidthF(2 * scale)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(px, py), radius, radius)
            arrow = [(0, 0), (0, 17), (4.5, 13), (7.5, 20), (10, 19), (7, 12.5), (12.5, 12.5)]
            pen = QPen(QColor(15, 15, 15))
            pen.setWidthF(1.4 * scale)
            painter.setPen(pen)
            painter.setBrush(QColor(250, 250, 250))
            painter.drawPolygon(QPolygonF([QPointF(px + ax * scale, py + ay * scale) for ax, ay in arrow]))
            painter.setClipping(False)
        painter.end()
        return _qimage_to_array(frame)

    def _to_frame(self, index: int, point, crops, scale: float) -> tuple[float, float]:
        """A point in the canvas coordinates of view ``index``, in frame coordinates."""
        x0, y0, x1, y1 = crops[index]
        rx, ry, rw, rh = self.rects[index]
        return (rx + (point[0] - x0) / (x1 - x0) * rw) * scale, (ry + (point[1] - y0) / (y1 - y0) * rh) * scale

    def _draw_highlight(self, painter, crops, scale: float) -> None:
        """Outline the highlighted box in cyan where the views draw it.

        The outline is computed from the vertices each view's layer draws, not
        from the store, so it appears only where napari shows the box.
        """
        from qtpy.QtCore import QPointF, QRectF, Qt
        from qtpy.QtGui import QColor, QPen, QPolygonF

        pen = QPen(QColor(0, 220, 255))
        pen.setWidthF(2.2 * scale)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for index in (0, 1, 2, 3):
            layer = self.layers[index]
            if self.highlight >= len(layer.bounding_boxes):
                continue
            path = drawn_path(layer, self.highlight)
            if path is None:
                continue
            viewer = self.viewers[index]
            rx, ry, rw, rh = self.rects[index]
            painter.setClipRect(QRectF(rx * scale, ry * scale, rw * scale, rh * scale))
            if index < 3:
                corners = [self._to_frame(index, world_to_canvas(viewer, v), crops, scale) for v in path]
                xs, ys = [c[0] for c in corners], [c[1] for c in corners]
                pad = 3 * scale
                painter.drawRect(QRectF(min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad,
                                        max(ys) - min(ys) + 2 * pad))
            else:
                points = [self._to_frame(index, project_3d(viewer, v), crops, scale) for v in path]
                painter.drawPolyline(QPolygonF([QPointF(*p) for p in points]))
        painter.setClipping(False)

    def _draw_slice_marks(self, painter, scale: float) -> None:
        """A thin dashed line in XZ and YZ at the depth of the XY view's slice."""
        from qtpy.QtCore import QLineF, Qt
        from qtpy.QtGui import QColor, QPen

        z = self.viewers[0].dims.point[0]
        fraction = (z - REGION[0][0]) / (REGION[0][1] - REGION[0][0])
        pen = QPen(QColor(90, 200, 255, 200))
        pen.setWidthF(1.2 * scale)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for index in (1, 2):
            x, y, w, h = self.rects[index]
            yy = (y + fraction * h) * scale
            painter.drawLine(QLineF(x * scale, yy, (x + w) * scale, yy))

    def grab(self, frames: int = 1) -> None:
        frame = self.compose()
        for _ in range(frames):
            self.frames.append(frame)

    def click_ring(self) -> None:
        for step in range(1, 4):
            self.ring = step / 3
            self.grab()
        self.ring = 0.0

    # -- real input events --------------------------------------------------
    def _mouse(self, kind, point, *, buttons=True, modifiers=None) -> None:
        from qtpy.QtCore import QEvent, QPointF, Qt
        from qtpy.QtGui import QMouseEvent
        from qtpy.QtWidgets import QApplication

        types = {
            "press": QEvent.Type.MouseButtonPress,
            "move": QEvent.Type.MouseMove,
            "release": QEvent.Type.MouseButtonRelease,
        }
        local = QPointF(*point)
        button = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
        held = Qt.MouseButton.LeftButton if (buttons and kind != "release") else Qt.MouseButton.NoButton
        mods = Qt.KeyboardModifier.NoModifier if modifiers is None else modifiers
        event = QMouseEvent(
            types[kind], local, QPointF(self.canvas.mapToGlobal(local.toPoint())), button, held, mods
        )
        QApplication.sendEvent(self.canvas, event)
        QApplication.processEvents()

    def move_to(self, point, frames: int = 8) -> None:
        """Move the drawn pointer (no button held) to ``point``, eased."""
        start = self.cursor or (point[0] - 120, point[1] + 60)
        for step in range(1, frames + 1):
            t = _ease(step / frames)
            self.cursor = (start[0] + (point[0] - start[0]) * t, start[1] + (point[1] - start[1]) * t)
            self._mouse("move", self.cursor, buttons=False)
            self.grab()

    def drag(self, start, stop, frames: int, *, modifiers=None, during=None) -> None:
        """Press at ``start``, move to ``stop`` in ``frames`` steps, release; one frame per step."""
        self.move_to(start)
        self._mouse("press", start, modifiers=modifiers)
        self.cursor = start
        self.ring = 0.5
        self.grab()
        self.ring = 0.0
        for step in range(1, frames + 1):
            t = _ease(step / frames)
            self.cursor = (start[0] + (stop[0] - start[0]) * t, start[1] + (stop[1] - start[1]) * t)
            self._mouse("move", self.cursor, modifiers=modifiers)
            if during is not None:
                during(step)
            self.grab()
        self._mouse("release", stop, modifiers=modifiers)

    def key(self, key, modifiers=None, widget=None) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtTest import QTest
        from qtpy.QtWidgets import QApplication

        mods = Qt.KeyboardModifier.NoModifier if modifiers is None else modifiers
        QTest.keyClick(widget or self.canvas, key, mods)
        QApplication.processEvents()

    def turn_3d(self, frames: int = 24) -> None:
        """Rotate the 3D view by dragging on its canvas, as a user turns it."""
        from qtpy.QtCore import QEvent, QPointF, Qt
        from qtpy.QtGui import QMouseEvent
        from qtpy.QtWidgets import QApplication

        canvas = self.viewers[3].window._qt_viewer.canvas.native
        start = QPointF(canvas.width() * 0.45, canvas.height() * 0.45)
        span = canvas.width() * 0.12

        def send(kind, point, held):
            event = QMouseEvent(
                kind, point, QPointF(canvas.mapToGlobal(point.toPoint())), Qt.MouseButton.LeftButton
                if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton, held, Qt.KeyboardModifier.NoModifier
            )
            QApplication.sendEvent(canvas, event)
            QApplication.processEvents()

        self.cursor, self.cursor3d = None, (start.x(), start.y())
        self.grab(3)
        send(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
        for step in range(1, frames + 1):
            point = QPointF(start.x() + span * _ease(step / frames), start.y() + 0.8 * span * _ease(step / frames))
            send(QEvent.Type.MouseMove, point, Qt.MouseButton.LeftButton)
            self.cursor3d = (point.x(), point.y())
            self.grab()
        send(QEvent.Type.MouseButtonRelease, point, Qt.MouseButton.NoButton)

    def write_gif(self, path: Path) -> float:
        """Encode with a palette computed from the frames (ffmpeg's default palette bands)."""
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg is not on PATH; install it and run again")
        height, width = self.frames[0].shape[:2]
        with tempfile.TemporaryDirectory() as tmp:
            raw, palette = Path(tmp) / "frames.raw", Path(tmp) / "palette.png"
            with raw.open("wb") as handle:
                for frame in self.frames:
                    handle.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            source = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                      "-video_size", f"{width}x{height}", "-framerate", str(FPS), "-i", str(raw)]
            subprocess.run([*source, "-vf", "palettegen=max_colors=128:stats_mode=diff", str(palette)], check=True)
            subprocess.run([*source, "-i", str(palette), "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                            "-loop", "0", str(path)], check=True)
        size_mb = path.stat().st_size / 1024**2
        print(f"wrote {path} ({len(self.frames)} frames, {len(self.frames) / FPS:.1f} s, {size_mb:.2f} MB)")
        return size_mb


def _ease(t: float) -> float:
    """Smoothstep: slow at both ends, like a hand."""
    return t * t * (3.0 - 2.0 * t)


def _qimage_to_array(image) -> np.ndarray:
    from qtpy.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height, stride = image.width(), image.height(), image.bytesPerLine()
    bits = image.constBits()
    if hasattr(bits, "setsize"):
        bits.setsize(image.sizeInBytes())
    buffer = np.frombuffer(bits, dtype=np.uint8, count=height * stride)
    return buffer.reshape(height, stride)[:, : 3 * width].reshape(height, width, 3).copy()


def _write_png(frame: np.ndarray, path: Path) -> None:
    from qtpy.QtGui import QImage

    frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
    QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1], QImage.Format.Format_RGB888).copy().save(
        str(path)
    )


# ----------------------------------------------------------------------- beats
def produce(out_dir: Path, gif: Path) -> dict:
    import napari
    import qtpy
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    import napari_turbobox
    from napari_turbobox.widget import BoundingBoxControlWidget

    source = Path(napari_turbobox.__file__).resolve()
    assert source.is_relative_to(REPO / "src"), f"napari_turbobox was imported from {source}, not from this checkout"

    image, nuclei, labels, ids = load_sample()
    plan = plan_scenes(image, nuclei, labels, ids)
    k, dx = plan["k"], plan["dx"]
    gt, prepared = plan["gt"], plan["prepared"]
    # Centre the shown x range on the box and its nucleus; move the y range
    # (which covers the width of the worm) only as far as needed to show them
    # with a margin of 5 voxels.
    scene = np.stack([gt, prepared])
    low = (scene[:, :, 2].min() + scene[:, :, 2].max() - X_SPAN) / 2
    low = min(max(low, 0.0), image.shape[2] - 1 - X_SPAN)
    REGION[2] = (low, low + X_SPAN)
    span = REGION[1][1] - REGION[1][0]
    low = min(REGION[1][0], scene[:, 0, 1].min() - 5)
    low = max(low, scene[:, 1, 1].max() + 5 - span, 0.0)
    REGION[1] = (low, low + span)
    for axis in (0, 1, 2):
        assert REGION[axis][0] + 5 <= scene[:, 0, axis].min() and scene[:, 1, axis].max() <= REGION[axis][1] - 5, (
            f"the edited nuclei are not inside the shown part of axis {axis}: {scene[:, :, axis]}, {REGION[axis]}"
        )
    boxes = np.asarray(nuclei, float).copy()
    boxes[k] = prepared  # off its nucleus and too small
    viewers, layers = build_layout(image, boxes)
    xy, xz, yz, v3d = viewers
    main, lay_xz, lay_yz, lay_3d = layers
    layer_py = Path(napari_turbobox.__file__).with_name("layer.py")
    facts: dict = {
        "napari": napari.__version__,
        "qt": f"{qtpy.API_NAME} {qtpy.QT_VERSION}",
        "napari_turbobox": napari_turbobox.__file__,
        "layer_py_sha256": hashlib.sha256(layer_py.read_bytes()).hexdigest(),
        "image_shape": list(image.shape),
        "nuclei": len(nuclei),
        "boxes_loaded": len(boxes),
        "corrected_box": {"index": k, "nucleus": gt.tolist(), "loaded_as": prepared.tolist(), "dx": dx},
    }

    # Every 2D view slices through the nucleus of box k: XY at its centre z, XZ
    # at its centre y, YZ at its centre x. The box is loaded off to the side, so
    # the YZ view does not show it until it is dragged onto the nucleus.
    centre = gt.mean(axis=0)
    xy.dims.set_point(0, plan["z0"])
    xz.dims.set_point(1, round(centre[1]))
    yz.dims.set_point(2, round(centre[2]))
    for viewer in viewers[:3]:
        frame_region(viewer)
    v3d.camera.center = (sum(REGION[0]) / 2, sum(REGION[1]) / 2, sum(REGION[2]) / 2)
    v3d.camera.angles = (-25.0, 20.0, 90.0)
    v3d.camera.perspective = 30.0
    v3d.camera.zoom = 1.3 * v3d.window._qt_viewer.canvas.native.width() / (REGION[2][1] - REGION[2][0])
    main.mode = "select"
    main.selected_data = set()
    for _ in range(10):
        QApplication.processEvents()
    box = np.asarray(main.bounding_boxes[k], float).copy()
    assert np.allclose(box, prepared), "the loaded box is not the prepared one"
    assert shown_box(lay_yz, k) is None, "the YZ view should not show the box before the drag"
    rec = Recorder(viewers, layers)
    rec.highlight = k
    # Mouse positions arrive in whole logical pixels: a drag can end up to about
    # one pixel short, so positions are checked to 1.5 pixels (in voxels).
    tol = 1.5 / xy.camera.zoom
    facts["position_tolerance_voxels"] = round(tol, 4)

    # -- 1. The layout. -----------------------------------------------------
    rec.caption = "One box store, four synchronized napari viewers. The cyan box has slipped off its nucleus"
    rec.mark("layout")
    rec.grab(24)

    # -- 2. Drag: XZ follows every move, YZ shows the box on arrival, 3D at the release.
    rec.caption = "Drag it onto its nucleus: XZ follows every move, YZ shows it on arrival, 3D when you let go"
    rec.mark("drag")
    grip = box.mean(axis=0)
    grip[0] = plan["z0"]
    start = world_to_canvas(xy, grip)
    stop = world_to_canvas(xy, grip - np.array([0.0, 0.0, dx]))
    three_d_before = drawn_path(lay_3d, k)
    assert three_d_before is not None, "the 3D view does not draw the box"
    yz_x = yz.dims.point[2]  # the x of the YZ view's slice
    seen = {"x_min_per_step": [], "xz_follows": [], "yz_shows_iff_on_slice": [], "yz_entered_at": None,
            "3d_unchanged_during_drag": True}

    def during(step: int) -> None:
        now = np.asarray(main.bounding_boxes[k], float)
        seen["x_min_per_step"].append(float(now[0, 2]))
        on_xz = shown_box(lay_xz, k)
        seen["xz_follows"].append(bool(on_xz is not None and np.isclose(on_xz[0][1], now[0, 2], atol=1e-3)))
        on_yz = shown_box(lay_yz, k)
        on_slice = bool(now[0, 2] <= yz_x <= now[1, 2])
        seen["yz_shows_iff_on_slice"].append((on_yz is not None) == on_slice)
        if seen["yz_entered_at"] is None and on_yz is not None:
            seen["yz_entered_at"] = step
        if not np.array_equal(drawn_path(lay_3d, k), three_d_before):
            seen["3d_unchanged_during_drag"] = False

    rec.drag(start, stop, 22, during=during)
    moved = np.asarray(main.bounding_boxes[k], float)
    assert np.allclose(moved[0, 1:], gt[0, 1:], atol=tol), f"the box did not land on its nucleus: {moved} vs {gt}"
    assert np.allclose(moved[:, :2], box[:, :2]), "the drag moved the box in z or y"
    # Mouse positions arrive in whole pixels, so the eased last moves (less than
    # a pixel) may leave the box where it is; every other move must move it.
    steps = np.diff([box[0, 2], *seen["x_min_per_step"]]) * -np.sign(dx)
    assert np.all(steps >= 0) and (steps > 0).sum() >= len(steps) - 3, (
        f"the store did not follow the moves: {seen['x_min_per_step']}"
    )
    assert all(seen["xz_follows"]), "the XZ view did not follow every move"
    assert all(seen["yz_shows_iff_on_slice"]), "the YZ view did not draw the box exactly while it was on its slice"
    assert seen["yz_entered_at"] is not None, "the box never entered the YZ view"
    assert seen["3d_unchanged_during_drag"], "the 3D view changed before the release"
    assert not np.array_equal(drawn_path(lay_3d, k), three_d_before), "the 3D view did not update"
    three_d_moved = drawn_path(lay_3d, k)
    facts["drag"] = {"after": moved.tolist(), **seen}
    rec.grab(8)

    # -- 3. Resize: pull the max corner out in y and x. ----------------------
    rec.caption = "Pull its corner until the box fits the nucleus: XZ and YZ show the new extent at once"
    rec.mark("resize")
    edge = moved.mean(axis=0)  # grabbed near the max corner, 10 % inside on y and x
    edge[0] = plan["z0"]
    edge[1:] = moved[1, 1:] - 0.1 * (moved[1, 1:] - moved[0, 1:])
    extent_shown, max_per_step = [], []

    def during_resize(step: int) -> None:
        now = np.asarray(main.bounding_boxes[k], float)
        max_per_step.append(now[1, 1:].copy())
        on_xz, on_yz = shown_box(lay_xz, k), shown_box(lay_yz, k)
        extent_shown.append(
            on_xz is not None and on_yz is not None
            and bool(np.isclose(on_xz[1][1], now[1, 2], atol=1e-3))  # XZ: max x
            and bool(np.isclose(on_yz[1][1], now[1, 1], atol=1e-3))  # YZ: max y
        )

    pull = np.array([0.0, *(gt[1, 1:] - moved[1, 1:])])
    rec.drag(world_to_canvas(xy, edge), world_to_canvas(xy, edge + pull), 16, during=during_resize)
    resized = np.asarray(main.bounding_boxes[k], float)
    assert np.allclose(resized, gt, atol=tol), f"the box does not fit its nucleus: {resized} vs {gt}"
    # every voxel of the nucleus, with its half-voxel extent, is inside the box
    assert np.all(resized[0] <= gt[0] + 0.5) and np.all(resized[1] >= gt[1] - 0.5), f"the box cuts the nucleus: {resized}"
    assert np.allclose(resized[0], moved[0]) and resized[1, 0] == moved[1, 0], "only the max corner's y and x may change"
    assert all(extent_shown), "XZ or YZ did not show the new extent on every move"
    growth = np.diff([moved[1, 1:], *max_per_step], axis=0)
    assert np.all(growth >= 0) and (np.any(growth > 0, axis=1)).sum() >= len(growth) - 3, (
        f"the store did not follow the resize moves: {np.asarray(max_per_step).tolist()}"
    )
    three_d_resized = drawn_path(lay_3d, k)
    assert not np.array_equal(three_d_resized, three_d_moved), "the 3D view did not update after the resize"
    facts["resize"] = {"after": resized.tolist(), "xz_and_yz_follow_every_move": all(extent_shown)}
    rec.grab(8)

    # -- 4. Step through z with the arrow keys. ------------------------------
    rec.caption = "Step through z with the arrow keys: a 2D view draws only the boxes on its slice"
    rec.mark("slices")
    z0 = xy.dims.current_step[0]
    counts, expected = [], []
    rec.cursor = None
    store = np.asarray(main.bounding_boxes, float)
    for key, repeat in ((Qt.Key.Key_Right, 8), (Qt.Key.Key_Left, 16), (Qt.Key.Key_Right, 8)):
        for _ in range(repeat):
            rec.key(key)
            z = xy.dims.point[0]
            counts.append(int(main.nshapes))
            expected.append(int(((store[:, 0, 0] <= z) & (z <= store[:, 1, 0])).sum()))
            shown = {i for i in range(len(store)) if main._bbox_index_to_path_index(i) is not None}
            wanted = set(np.flatnonzero((store[:, 0, 0] <= z) & (z <= store[:, 1, 0])).tolist())
            assert shown == wanted, f"at z = {z}, XY draws {sorted(shown ^ wanted)} wrongly"
            rec.grab()
    assert xy.dims.current_step[0] == z0, "the arrow keys did not bring the slice back"
    assert len(set(counts)) > 3, f"the slice did not change: {counts}"
    assert counts == expected, f"XY does not draw exactly the boxes on its slice: {counts} vs {expected}"
    facts["slices"] = {"z": int(z0), "boxes_drawn_per_step": counts}
    rec.grab(4)

    # -- 5. Undo and redo the fit. ---------------------------------------------
    def views_show(expected: np.ndarray, three_d) -> bool:
        """Every view draws box k as ``expected`` (XY, XZ, YZ on their axes; 3D as the recorded path)."""
        for layer, axes in ((main, [1, 2]), (lay_xz, [0, 2]), (lay_yz, [0, 1])):
            drawn = shown_box(layer, k)
            if drawn is None or not np.allclose(np.stack(drawn), expected[:, axes], atol=1e-3):
                return False
        return bool(np.array_equal(drawn_path(lay_3d, k), three_d))

    rec.caption = "Ctrl+Z / Cmd+Z undoes the fit in every view"
    rec.mark("undo")
    rec.grab(5)
    store_fitted = np.asarray(main.bounding_boxes).copy()
    rec.key(Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert np.allclose(main.bounding_boxes[k], moved), "undo did not bring back the box before the fit"
    others = np.ones(len(store_fitted), bool)
    others[k] = False
    assert np.array_equal(np.asarray(main.bounding_boxes)[others], store_fitted[others]), "undo changed other boxes"
    assert views_show(moved, three_d_moved), "not every view shows the box before the fit after undo"
    rec.grab(12)
    # Qt's Control modifier is the Command key on macOS: this is Cmd+Shift+Z there
    # (Cmd+Y is napari's own shortcut for switching to 3D).
    rec.caption = "Cmd+Shift+Z redoes it"
    rec.mark("redo")
    rec.grab(4)
    rec.key(Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert xy.dims.ndisplay == 2, "the redo key switched the XY viewer to 3D"
    assert np.array_equal(np.asarray(main.bounding_boxes), store_fitted), "redo did not bring back the fit"
    assert views_show(resized, three_d_resized), "not every view shows the fitted box after redo"
    facts["undo_redo"] = {"undo_key": "Qt Control+Z", "redo_key": "Qt Control+Shift+Z",
                          "views_checked": ["XY", "XZ", "YZ", "3D"]}
    rec.grab(12)

    # -- 6. Turn the 3D view. -------------------------------------------------
    rec.caption = "The 3D view draws every box as a wireframe"
    rec.mark("turn 3D")
    assert lay_3d.nshapes == len(main.bounding_boxes), "the 3D view does not draw one wireframe per box"
    angles = tuple(v3d.camera.angles)
    rec.turn_3d(frames=24)
    assert tuple(v3d.camera.angles) != angles, "the 3D view did not turn"
    rec.cursor3d = None
    rec.grab(14)

    # -- stills and the GIF ----------------------------------------------------
    assert len(main.bounding_boxes) == len(nuclei), "the stills should show one box per nucleus"
    views_png = out_dir / "screenshot-views.png"
    rec.caption = ""
    _write_png(rec.compose(scale=2.0, caption_strip=False), views_png)
    panel = BoundingBoxControlWidget(xy)
    xy.window.add_dock_widget(panel, name="BBox Control Panel", area="right")
    xy.window._qt_window.resize(1300, 760)
    main.selected_data = {main._bbox_index_to_path_index(k)}  # the dragged box, selected in the canvas
    for _ in range(20):
        QApplication.processEvents()
    xy.camera.center = tuple(resized.mean(axis=0)[1:])  # closer in, around the dragged box
    xy.camera.zoom = xy.window._qt_viewer.canvas.native.width() / 130.0
    for _ in range(10):
        QApplication.processEvents()
    panel_png = out_dir / "screenshot-panel.png"
    _write_png(np.asarray(xy.window.screenshot(canvas_only=False, flash=False)), panel_png)
    facts["gif_mb"] = round(rec.write_gif(gif), 2)
    facts["frames"] = len(rec.frames)
    facts["beats"] = rec.beats
    for viewer in viewers:
        viewer.close()
    return facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, default=REPO / "docs" / "demo.gif")
    parser.add_argument("--facts", type=Path, help="also write the checked values as JSON here")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    facts = produce(args.output.parent, args.output)
    text = json.dumps(facts, indent=1, default=str)
    print(text)
    if args.facts:
        args.facts.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
