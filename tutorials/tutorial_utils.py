"""Shared helpers for the napari-TurboBox tutorials.

The application notebooks in this folder all follow the same steps:

1. download one public volume (``fetch``), verified against the size and
   checksum published by the data repository, into a local cache;
2. turn its instance labels (or, where a dataset has none, a simple threshold
   segmentation) into one axis-aligned box per object (``labels_to_boxes``);
3. load the boxes into the synchronized four-viewer layout of napari-TurboBox
   (``four_viewer_layout``), take a screenshot of all views (``screenshot``)
   and replay a mouse drag on one box (``replay_drag``);
4. export the boxes as native, COCO and YOLO files (``export``) and record the
   counts (``save_result``).

Box convention: ``(N, 2, D)`` arrays ``[[min...], [max...]]`` in napari axis
order ``(z, y, x)``, inclusive voxel indices, ``max - min >= 1`` on every axis.

Data cache: ``$TURBOBOX_DATA`` if set, else ``~/.cache/napari-turbobox``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent / "examples"))  # labels_to_boxes.py

from labels_to_boxes import export_boxes, labels_to_boxes  # noqa: E402

__all__ = [
    "check_boxes",
    "data_dir",
    "export",
    "extract_member",
    "fetch",
    "fetch_zip_member",
    "four_viewer_layout",
    "labels_to_boxes",
    "pick_box",
    "replay_drag",
    "save_result",
    "screenshot",
    "show_path",
    "threshold_labels",
]


# --------------------------------------------------------------------------- data
def data_dir(subdir: str = "") -> Path:
    """Cache directory for downloaded data (``$TURBOBOX_DATA`` or ``~/.cache/napari-turbobox``)."""
    root = Path(os.environ.get("TURBOBOX_DATA") or Path.home() / ".cache" / "napari-turbobox")
    path = root / subdir if subdir else root
    path.mkdir(parents=True, exist_ok=True)
    return path


def show_path(path) -> str:
    """``path`` for printing: relative to the repository, or with the home directory as ``~``."""
    path = Path(path).resolve()
    for root, prefix in ((HERE.parent, ""), (Path.home(), "~/")):
        try:
            return prefix + path.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(path)


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    url: str,
    dest: str | Path,
    *,
    size: int | None = None,
    md5: str | None = None,
    sha256: str | None = None,
) -> Path:
    """Download ``url`` to ``dest`` once and verify it; return ``dest``.

    An existing file is reused when it has the expected size and checksum. The
    checksum of a verified file is remembered in ``<dest>.verified`` so that
    large files are hashed only once.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = dest.with_name(dest.name + ".verified")
    expected = sha256 or md5 or ""

    def ok() -> bool:
        if not dest.is_file() or (size is not None and dest.stat().st_size != size):
            return False
        if not expected:
            return True
        if marker.is_file() and marker.read_text().strip() == expected:
            return True
        actual = _digest(dest, "sha256" if sha256 else "md5")
        if actual != expected:
            return False
        marker.write_text(expected)
        return True

    if ok():
        return dest
    part = dest.with_name(dest.name + ".part")
    print(f"Downloading {url}\n  -> {show_path(dest)}" + (f" ({size / 1e6:.1f} MB)" if size else ""))
    request = urllib.request.Request(url, headers={"User-Agent": "napari-turbobox-tutorials"})
    t0 = time.time()
    with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as f:
        shutil.copyfileobj(response, f, 1 << 22)
    part.replace(dest)
    print(f"  done in {time.time() - t0:.0f} s")
    if not ok():
        raise RuntimeError(f"{dest} does not match the published size/checksum; delete it and rerun")
    return dest


def extract_member(archive: str | Path, member: str, dest: str | Path) -> Path:
    """Extract one member of a zip archive to ``dest`` (skipped if ``dest`` exists)."""
    dest = Path(dest)
    if dest.is_file():
        return dest
    part = dest.with_name(dest.name + ".part")
    with zipfile.ZipFile(archive) as z, z.open(member) as src, open(part, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 22)
    part.replace(dest)
    return dest


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for ``zipfile``).

    Lets ``zipfile`` read the central directory and single members of a large
    remote archive without downloading all of it.
    """

    def __init__(self, url: str, chunk: int = 1 << 20):
        headers = {"User-Agent": "napari-turbobox-tutorials", "Range": "bytes=0-0"}
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
            self.url = r.geturl()  # resolve redirects once (e.g. a CDN behind a stable URL)
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        headers = {"User-Agent": "napari-turbobox-tutorials", "Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_zip_member(url: str, member: str, dest: str | Path, *, sha256: str | None = None) -> Path:
    """Extract one member of a remote zip archive to ``dest`` via HTTP range requests.

    Only the archive's directory and the member's bytes are transferred.
    ``zipfile`` checks the member's CRC-32; ``sha256`` (if given) is checked too.
    An existing, verified ``dest`` is reused.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (sha256 is None or _digest(dest, "sha256") == sha256):
        return dest
    print(f"Reading {member!r}\n  from {url}\n  -> {show_path(dest)}")
    part = dest.with_name(dest.name + ".part")
    remote = io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20)
    with zipfile.ZipFile(remote) as z, z.open(member) as src, open(part, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 22)  # raises BadZipFile on a CRC-32 mismatch
    if sha256 is not None and _digest(part, "sha256") != sha256:
        raise RuntimeError(f"{member}: sha256 does not match the published value")
    part.replace(dest)
    return dest


# ------------------------------------------------------------------ segmentation
def threshold_labels(
    image: np.ndarray, *, sigma: float = 1.0, min_voxels: int = 20, threshold: float | None = None
) -> tuple[np.ndarray, float]:
    """Instance labels from a global threshold, for data without annotations.

    Gaussian smoothing (``sigma`` voxels), Otsu threshold (unless ``threshold``
    is given), 3D connected components (face connectivity), and removal of
    components smaller than ``min_voxels``. Returns ``(labels, threshold)``.
    This is a deliberately simple proposal step: the boxes it yields are meant
    to be reviewed and corrected in napari.
    """
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    smooth = ndimage.gaussian_filter(np.asarray(image, dtype=np.float32), sigma)
    thr = float(threshold_otsu(smooth)) if threshold is None else float(threshold)
    labels, n = ndimage.label(smooth > thr)
    if n:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_voxels
        keep[0] = False
        remap = np.zeros(len(sizes), dtype=np.int32)
        remap[keep] = np.arange(1, int(keep.sum()) + 1)
        labels = remap[labels]
    return labels, thr


def check_boxes(boxes: np.ndarray, shape: tuple[int, ...]) -> dict[str, int]:
    """Count boxes that satisfy the napari-TurboBox invariant for an image of ``shape``."""
    boxes = np.asarray(boxes, dtype=float)
    upper = np.asarray(shape, dtype=float) - 1
    inside = np.all((boxes[:, 0] >= 0) & (boxes[:, 1] <= upper), axis=1)
    extent = np.all(boxes[:, 1] - boxes[:, 0] >= 1, axis=1)
    return {"boxes": len(boxes), "valid": int(np.sum(inside & extent))}


# ------------------------------------------------------------------------ napari
PLANES = [  # title, dims.order, sliced axis (napari displays the last two axes of the order)
    ("XY (editable)", (0, 1, 2), 0),
    ("XZ (read-only)", (1, 0, 2), 1),
    ("YZ (read-only)", (2, 0, 1), 2),
]


def four_viewer_layout(
    image: np.ndarray,
    boxes: np.ndarray,
    *,
    show: bool = False,
    scale=None,
    contrast_limits=None,
    focus_box: int | None = None,
    edge_width: float = 2.0,
):
    """XY (editable), XZ and YZ (read-only) and 3D (read-only) viewers sharing one store.

    Returns ``(viewers, layers)``. ``show=False`` keeps the windows hidden (for
    scripted runs); set ``show=True`` to work with the layout interactively.
    ``focus_box`` moves every 2D view to the centre of that box.
    """
    import napari

    from napari_turbobox import create_synchronized_bbox_layers

    shape = tuple(int(s) for s in image.shape)
    centre = [s // 2 for s in shape]
    if focus_box is not None:
        centre = [round(c) for c in np.asarray(boxes[focus_box], float).mean(axis=0)]
    kw = {} if scale is None else {"scale": scale}
    img_kw = dict(kw, contrast_limits=contrast_limits) if contrast_limits is not None else dict(kw)
    viewers = []
    for title, order, axis in PLANES:
        v = napari.Viewer(title=title, show=show)
        v.add_image(image, name="image", **img_kw)
        v.dims.order = order
        v.dims.set_point(axis, centre[axis] * (1 if scale is None else scale[axis]))
        viewers.append(v)
    v3 = napari.Viewer(title="3D (read-only)", show=show)
    v3.add_image(image, name="image", rendering="mip", **img_kw)
    v3.dims.ndisplay = 3  # before the layers are created: the 3D layer updates once per edit session
    viewers.append(v3)
    layers = create_synchronized_bbox_layers(
        viewers[0], viewers[1:], image_shape=shape, edge_color="yellow", edge_width=edge_width, **kw
    )
    layers[0].add_boxes(boxes)
    for v in viewers:
        v.reset_view()
    return viewers, layers


def screenshot(viewers, path: str | Path | None = None, *, size=(400, 400)):
    """Canvas screenshots of all viewers side by side; saved as PNG if ``path`` is given."""
    from qtpy.QtWidgets import QApplication

    # A window that has never been shown renders nothing (black screenshots on
    # macOS); show hidden windows once, briefly, and hide them again.
    hidden = [v for v in viewers if not v.window._qt_window.isVisible()]
    for v in hidden:
        v.window.show()
    for _ in range(10):
        QApplication.processEvents()
    tiles = [np.asarray(v.screenshot(size=size, canvas_only=True, flash=False))[..., :3] for v in viewers]
    for v in hidden:
        v.window._qt_window.hide()
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 6), (0, 0)), constant_values=255) for t in tiles]
    strip = np.concatenate(tiles, axis=1)[:, :-6]
    if path is not None:
        from imageio.v3 import imwrite

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imwrite(path, strip)
    return strip


def pick_box(boxes: np.ndarray) -> int:
    """Index of a median-sized box whose centre lies in no other box (so a click grabs it)."""
    boxes = np.asarray(boxes, float)
    centres = boxes.mean(axis=1)
    volume = np.prod(boxes[:, 1] - boxes[:, 0], axis=1)
    for i in np.argsort(np.abs(volume - np.median(volume))):
        inside = np.all((boxes[:, 0] <= centres[i]) & (centres[i] <= boxes[:, 1]), axis=1)
        if inside.sum() == 1:
            return int(i)
    return int(np.argsort(np.abs(volume - np.median(volume)))[0])


def replay_drag(viewers, layers, box_index: int, steps: int = 8) -> dict:
    """Replay a mouse drag that moves one box in the editable XY view.

    The drag goes through napari's own mouse-event dispatch, the path a real
    mouse takes (the events are built with napari's test helper). The box is
    moved by a quarter of its size in y and x. Beforehand, every 2D view is set
    to a slice through the box. Returns the box before and after, whether it is
    still a valid box, and whether each read-only view (XZ, YZ, 3D) shows it at
    its new position.
    """
    from napari.utils._test_utils import read_only_mouse_event
    from napari.utils.interactions import (
        mouse_move_callbacks,
        mouse_press_callbacks,
        mouse_release_callbacks,
    )
    from qtpy.QtWidgets import QApplication

    main = layers[0]
    scale = np.asarray(main.scale, float)
    before = np.asarray(main.bounding_boxes[box_index], float).copy()
    centre = before.mean(axis=0)
    for v, (_title, _order, axis) in zip(viewers[:3], PLANES):
        v.dims.set_point(axis, centre[axis] * scale[axis])
    for _ in range(3):
        QApplication.processEvents()
    extent = before[1] - before[0]
    delta = np.array([0.0, max(1.0, np.floor(extent[1] / 4)), max(1.0, np.floor(extent[2] / 4))])
    start = centre * scale
    stop = (centre + delta) * scale
    kw = {"modifiers": [], "dims_displayed": list(main._slice_input.displayed), "view_direction": None,
              "up_direction": None}
    pts = [start + t * (stop - start) for t in np.linspace(0.0, 1.0, steps)]
    main.mode = "select"
    mouse_press_callbacks(main, read_only_mouse_event(type="mouse_press", position=tuple(pts[0]), **kw))
    for p in pts[1:]:
        mouse_move_callbacks(main, read_only_mouse_event(type="mouse_move", is_dragging=True, position=tuple(p), **kw))
        QApplication.processEvents()
    mouse_release_callbacks(main, read_only_mouse_event(type="mouse_release", position=tuple(pts[-1]), **kw))
    for _ in range(3):
        QApplication.processEvents()
    after = np.asarray(main.bounding_boxes[box_index], float)
    shape = main._image_shape
    valid = True if shape is None else check_boxes(after[None], tuple(int(s) for s in shape))["valid"] == 1
    views = {}
    for viewer, layer in zip(viewers[1:], layers[1:]):
        pos = layer._bbox_index_to_path_index(box_index)
        if pos is None:
            views[viewer.title] = None  # the box is not on this view's slice
            continue
        shown = np.asarray(layer.data[pos], float)
        disp = list(layer._slice_input.displayed)
        views[viewer.title] = bool(np.allclose(shown[:, disp].min(0), after[0, disp], atol=1e-3)
                                   and np.allclose(shown[:, disp].max(0), after[1, disp], atol=1e-3))
    return {"box_index": int(box_index), "before": before.tolist(), "after": after.tolist(),
            "intended_shift_zyx": delta.tolist(), "shift_zyx": (after[0] - before[0]).tolist(),
            "valid": bool(valid), "views_show_moved_box": views}


# ------------------------------------------------------------------------ export
def export(boxes: np.ndarray, shape: tuple[int, ...], out_dir: str | Path, label_ids=None) -> dict[str, int]:
    """Write ``boxes.npy``, ``coco_per_slice.json`` and ``yolo/`` (see ``examples/labels_to_boxes.py``)."""
    return export_boxes(boxes, shape, out_dir, label_ids=label_ids)


def save_result(case: str, result: dict) -> Path:
    """Store a tutorial's summary as ``tutorials/results/<case>.json``."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{case}.json"
    path.write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return path
