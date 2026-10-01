"""Box classes: what the store, the layer, the control panel, the exporters and
the importer carry.

Every assertion states what the code does today:

- ``BBoxDataStore`` holds coordinates only, one ``(N, 2, D)`` float32 array.
  No per-box class is stored.
- napari ``features`` on a ``TurboBoxLayer`` belong to the rendered shapes of
  the current view (one per box that crosses the slice), not to the boxes.
  A value set while one box is shown reappears on another box after a slice
  change.
- ``export.boxes_to_coco`` / ``boxes_to_yolo`` write classes only when the
  caller passes ``category_ids``: COCO ``category_id = class + 1`` plus a
  ``categories`` list, YOLO class index = class.
- The control panel passes no ``category_ids``. Its COCO export always has
  one category (``{"id": 1, "name": "object"}``) and its YOLO export always
  writes class 0.
- The control panel imports only the native ``(N, 2, D)`` ``.npy`` / ``.json``
  files. It rejects a COCO JSON or a YOLO ``.txt`` file. No reader in the
  package turns COCO or YOLO back into boxes, so classes do not round-trip.
  The COCO/YOLO parsers below are local to this test and show only that an
  API-exported file contains the classes.
"""

import json

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("qtpy")

from napari_turbobox import BBoxDataStore, TurboBoxLayer  # noqa: E402
from napari_turbobox.export import boxes_to_coco, boxes_to_yolo  # noqa: E402
from napari_turbobox.widget import BoundingBoxControlWidget  # noqa: E402

# (Z, Y, X); Y != X so an x/y swap in an exporter or parser shows up.
SHAPE = (10, 50, 60)
# Box 0 covers slices 1..4, box 1 slices 3..6: slices 3 and 4 hold both boxes.
BOXES = np.array(
    [
        [[1.0, 5.0, 5.0], [4.0, 20.0, 25.0]],
        [[3.0, 25.0, 30.0], [6.0, 40.0, 55.0]],
    ],
    dtype=np.float32,
)
CLASSES = [0, 1]
NAMES = {0: "nucleus", 1: "cell"}


# --------------------------------------------------------------- helpers ---


def _coco_to_boxes(coco):
    """Test-side COCO reader for per-slice exports: one 3D box per
    (x/y box, category) run of slices. Returns ``(boxes, classes)`` sorted by
    the box's min corner."""
    runs = {}
    for ann in coco["annotations"]:
        key = (tuple(float(v) for v in ann["bbox"]), int(ann["category_id"]))
        runs.setdefault(key, []).append(int(ann["image_id"]))
    out = []
    for ((x, y, w, h), cat), slices in runs.items():
        box = [[min(slices), y, x], [max(slices), y + h, x + w]]
        out.append((box, cat - 1))
    out.sort(key=lambda item: tuple(item[0][0]))
    return np.array([b for b, _ in out], dtype=float), [c for _, c in out]


def _yolo_to_boxes(files, shape):
    """Test-side YOLO reader for per-slice exports (``slice_NNNN.txt``)."""
    h_img, w_img = shape[-2], shape[-1]
    runs = {}
    for name, text in files.items():
        z = int(name[len("slice_"):-len(".txt")])
        for line in text.strip().splitlines():
            cls, xc, yc, w, h = line.split()
            x0 = (float(xc) - float(w) / 2) * w_img
            y0 = (float(yc) - float(h) / 2) * h_img
            key = (int(cls), round(y0, 3), round(x0, 3),
                   round(float(h) * h_img, 3), round(float(w) * w_img, 3))
            runs.setdefault(key, []).append(z)
    out = []
    for (cls, y0, x0, h, w), slices in runs.items():
        out.append(([[min(slices), y0, x0], [max(slices), y0 + h, x0 + w]], cls))
    out.sort(key=lambda item: tuple(item[0][0]))
    return np.array([b for b, _ in out], dtype=float), [c for _, c in out]


def _yolo_classes_per_file(files):
    return {name: sorted(line.split()[0] for line in text.strip().splitlines())
            for name, text in files.items()}


def _fake_dialogs(save_path=None, directory=None, open_path=None, item=None):
    class _FileDialog:
        @staticmethod
        def getSaveFileName(*a, **k):
            return (str(save_path), "")

        @staticmethod
        def getExistingDirectory(*a, **k):
            return str(directory)

        @staticmethod
        def getOpenFileName(*a, **k):
            return (str(open_path), "")

    class _InputDialog:
        @staticmethod
        def getItem(*a, **k):
            return (item, True)

    return _FileDialog, _InputDialog


@pytest.fixture
def viewer(make_napari_viewer):
    """A hidden viewer from napari's ``make_napari_viewer`` (which also closes it).

    A ``napari.Viewer`` made directly registers the installed plugins with napari's
    own app model, and the first one made after a ``make_napari_viewer`` test
    registers them again: in one pytest process that fails with "Command
    '<plugin>.get_reader' already registered".
    """
    v = make_napari_viewer()
    v.add_image(np.zeros(SHAPE, np.uint8))
    return v


@pytest.fixture
def panel(viewer):
    """A control panel on a hidden viewer whose selected box layer holds BOXES."""
    layer = TurboBoxLayer(ndim=3, image_shape=SHAPE)
    viewer.add_layer(layer)
    layer.add_boxes(BOXES)
    widget = BoundingBoxControlWidget(viewer)
    try:
        widget.refresh_layers()
        assert widget._current_layer() is layer
        yield widget, layer
    finally:
        widget.close()
        widget.deleteLater()


# ----------------------------------------------- storage: no class field ---


def test_store_and_layer_hold_coordinates_only():
    store = BBoxDataStore(BOXES)
    assert store.data.shape == (2, 2, 3)
    assert store.data.dtype == np.float32
    assert store.data.dtype.names is None  # plain array, no structured class field

    def class_like(names):
        # ``class_keymap`` is napari's per-class key-binding table (KeymapProvider).
        return [n for n in names
                if ("class" in n.lower() or "categ" in n.lower()) and n != "class_keymap"]

    assert class_like(vars(store)) == []
    assert class_like(vars(BBoxDataStore)) == []
    assert class_like(vars(TurboBoxLayer)) == []
    # The add/set entry points take coordinates only.
    import inspect

    assert list(inspect.signature(BBoxDataStore.add_boxes).parameters) == ["self", "boxes"]
    assert list(inspect.signature(TurboBoxLayer.add_boxes).parameters) == ["self", "boxes"]


def test_layer_features_follow_rendered_shapes_not_boxes(viewer):
    # One value per box cannot be given at construction: the Shapes layer is
    # created empty and napari checks the feature length against 0 shapes.
    with pytest.raises(ValueError):
        TurboBoxLayer(BOXES, ndim=3, image_shape=SHAPE, features={"cls": CLASSES})

    layer = TurboBoxLayer(ndim=3, image_shape=SHAPE)
    viewer.add_layer(layer)
    layer.add_boxes(BOXES)

    viewer.dims.set_point(0, 6)  # only box 1 crosses slice 6
    assert layer.nshapes == 1
    layer.features = {"cls": [CLASSES[1]]}
    assert layer.features["cls"].tolist() == [1]

    viewer.dims.set_point(0, 1)  # only box 0 crosses slice 1
    assert layer.nshapes == 1
    shown = np.asarray(layer.data[0])[:, -2:]
    np.testing.assert_allclose(shown.min(axis=0), BOXES[0, 0, 1:])  # it is box 0 ...
    np.testing.assert_allclose(shown.max(axis=0), BOXES[0, 1, 1:])
    assert layer.features["cls"].tolist() == [1]  # ... carrying box 1's class
    assert len(layer.features) == layer.nshapes != len(layer.bounding_boxes)

    # A per-box table cannot be assigned while fewer boxes are shown.
    with pytest.raises(ValueError):
        layer.features = {"cls": CLASSES}


def test_feature_column_disables_in_place_enter_leave(viewer):
    """With any feature column the layer falls back from the in-place shape
    add/remove (box entering or leaving a slice) to a full rebuild."""
    if not hasattr(TurboBoxLayer, "_uniform_shape_style"):
        pytest.skip("no in-place enter/leave path in this version")
    layer = TurboBoxLayer(ndim=3, image_shape=SHAPE)
    viewer.add_layer(layer)
    layer.add_boxes(BOXES)
    viewer.dims.set_point(0, 3)  # both boxes shown
    assert layer.nshapes == 2
    assert layer._uniform_shape_style(entering=False) is not None
    layer.features = {"cls": CLASSES}
    assert layer._uniform_shape_style(entering=False) is None


# ------------------------------------------- export API: classes written ---


def test_api_coco_and_yolo_write_classes_given_at_call_time():
    coco = boxes_to_coco(BOXES, SHAPE, category_ids=CLASSES, category_names=NAMES)
    assert coco["categories"] == [{"id": 1, "name": "nucleus"}, {"id": 2, "name": "cell"}]
    by_cat = {}
    for ann in coco["annotations"]:
        by_cat.setdefault(ann["category_id"], []).append(ann["image_id"])
    assert by_cat == {1: [1, 2, 3, 4], 2: [3, 4, 5, 6]}  # category_id = class + 1

    yolo = boxes_to_yolo(BOXES, SHAPE, category_ids=CLASSES)
    assert _yolo_classes_per_file(yolo) == {
        "slice_0001.txt": ["0"],
        "slice_0002.txt": ["0"],
        "slice_0003.txt": ["0", "1"],
        "slice_0004.txt": ["0", "1"],
        "slice_0005.txt": ["1"],
        "slice_0006.txt": ["1"],
    }

    # The files carry the classes: a reader outside the package recovers them.
    boxes, classes = _coco_to_boxes(json.loads(json.dumps(coco)))
    np.testing.assert_allclose(boxes, BOXES)
    assert classes == CLASSES
    boxes, classes = _yolo_to_boxes(yolo, SHAPE)
    np.testing.assert_allclose(boxes, BOXES, atol=1e-3)
    assert classes == CLASSES


def test_api_2d_boxes_write_classes_on_one_image():
    boxes_2d = BOXES[:, :, 1:]
    coco = boxes_to_coco(boxes_2d, SHAPE[1:], category_ids=CLASSES)
    assert [a["category_id"] for a in coco["annotations"]] == [1, 2]
    assert coco["categories"] == [{"id": 1, "name": "class_0"}, {"id": 2, "name": "class_1"}]
    yolo = boxes_to_yolo(boxes_2d, SHAPE[1:], category_ids=CLASSES)
    assert _yolo_classes_per_file(yolo) == {"image.txt": ["0", "1"]}


def test_api_without_category_ids_writes_one_class():
    coco = boxes_to_coco(BOXES, SHAPE)
    assert coco["categories"] == [{"id": 1, "name": "object"}]
    assert {a["category_id"] for a in coco["annotations"]} == {1}
    yolo = boxes_to_yolo(BOXES, SHAPE)
    assert {c for v in _yolo_classes_per_file(yolo).values() for c in v} == {"0"}


# ------------------------------- control panel: export has a single class ---


def test_panel_has_no_class_control(panel):
    from qtpy.QtWidgets import QAbstractButton, QGroupBox, QLabel

    widget, _ = panel
    texts = [w.text() for w in widget.findChildren(QLabel)]
    texts += [w.text() for w in widget.findChildren(QAbstractButton)]
    texts += [w.title() for w in widget.findChildren(QGroupBox)]
    texts += [w.toolTip() for w in widget.findChildren(QLabel) + widget.findChildren(QAbstractButton)]
    # the scan must see the panel's I/O controls, otherwise the negative check below is vacuous
    assert "Export BBoxes..." in texts and "Import BBoxes..." in texts, texts
    assert not [t for t in texts if "class" in t.lower() or "categ" in t.lower()]


def test_panel_coco_export_writes_one_category(panel, tmp_path, monkeypatch):
    widget, _layer = panel
    out = tmp_path / "boxes_coco.json"
    file_dialog, input_dialog = _fake_dialogs(save_path=out, item="COCO JSON (per-slice)")
    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", file_dialog)
    monkeypatch.setattr("napari_turbobox.widget.QInputDialog", input_dialog)

    widget._export_boxes()

    coco = json.loads(out.read_text())
    assert coco["categories"] == [{"id": 1, "name": "object"}]
    assert {a["category_id"] for a in coco["annotations"]} == {1}
    boxes, classes = _coco_to_boxes(coco)
    np.testing.assert_allclose(boxes, BOXES)  # coordinates survive ...
    assert classes == [0, 0]  # ... two classes cannot be exported from the panel
    assert classes != CLASSES


def test_panel_yolo_export_writes_class_zero(panel, tmp_path, monkeypatch):
    widget, _layer = panel
    file_dialog, input_dialog = _fake_dialogs(directory=tmp_path, item="YOLO labels (per-slice)")
    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", file_dialog)
    monkeypatch.setattr("napari_turbobox.widget.QInputDialog", input_dialog)

    widget._export_boxes()

    files = {p.name: p.read_text() for p in sorted(tmp_path.glob("*.txt"))}
    assert sorted(files) == [f"slice_{z:04d}.txt" for z in range(1, 7)]
    assert {c for v in _yolo_classes_per_file(files).values() for c in v} == {"0"}
    boxes, classes = _yolo_to_boxes(files, SHAPE)
    np.testing.assert_allclose(boxes, BOXES, atol=1e-3)
    assert classes == [0, 0]


# ------------------------------- control panel: import is native-only ---


@pytest.mark.parametrize("fmt", ["coco", "yolo"])
def test_panel_import_rejects_coco_and_yolo(panel, tmp_path, monkeypatch, fmt):
    widget, layer = panel
    if fmt == "coco":
        path = tmp_path / "with_classes.json"
        coco = boxes_to_coco(BOXES, SHAPE, category_ids=CLASSES, category_names=NAMES)
        path.write_text(json.dumps(coco))
    else:
        path = tmp_path / "slice_0003.txt"
        path.write_text(boxes_to_yolo(BOXES, SHAPE, category_ids=CLASSES)["slice_0003.txt"])
    file_dialog, _ = _fake_dialogs(open_path=path)
    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", file_dialog)
    before = layer.bounding_boxes

    widget._import_boxes()

    assert "Import failed" in widget.status_label.text()
    np.testing.assert_array_equal(layer.bounding_boxes, before)  # nothing added


@pytest.mark.parametrize("suffix", [".json", ".npy"])
def test_panel_native_round_trip_keeps_coordinates_only(panel, tmp_path, monkeypatch, suffix):
    widget, layer = panel
    path = tmp_path / f"boxes{suffix}"
    file_dialog, input_dialog = _fake_dialogs(save_path=path, open_path=path,
                                              item="Native (.npy / .json)")
    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", file_dialog)
    monkeypatch.setattr("napari_turbobox.widget.QInputDialog", input_dialog)

    widget._export_boxes()
    saved = np.array(json.loads(path.read_text())) if suffix == ".json" else np.load(path)
    assert saved.shape == (2, 2, 3)  # (N, 2, D): no column for a class
    assert saved.dtype.names is None

    layer.clear_boxes()
    assert len(layer.bounding_boxes) == 0
    widget._import_boxes()
    # The success message is replaced at once by the box count (_update_status).
    assert "Import failed" not in widget.status_label.text()
    np.testing.assert_allclose(layer.bounding_boxes, BOXES)
