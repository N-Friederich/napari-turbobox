"""Widget export-path tests (native / COCO / YOLO).

The widget is a plain QWidget (no GL canvas), so it constructs headless with
``viewer=None``. File dialogs are replaced by fakes returning temp paths, so
the export dispatch + writers are exercised without GUI interaction.
"""

import json

import numpy as np
import pytest

from napari_turbobox.layer import TurboBoxLayer
from napari_turbobox.widget import BoundingBoxControlWidget


@pytest.fixture
def widget(qapp):
    return BoundingBoxControlWidget(None)


@pytest.fixture
def layer():
    lyr = TurboBoxLayer(ndim=3, image_shape=(50, 200, 200))
    lyr.add_boxes(np.array([[[2.0, 10.0, 20.0], [5.0, 30.0, 80.0]]]))
    return lyr


def _fake_save(path):
    class _D:
        @staticmethod
        def getSaveFileName(*a, **k):
            return (str(path), "")

        @staticmethod
        def getExistingDirectory(*a, **k):
            return str(path)

    return _D


def test_export_native_npy(widget, layer, tmp_path, monkeypatch):
    out = tmp_path / "boxes.npy"
    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _fake_save(out))
    widget._export_native(layer.bounding_boxes)
    assert out.exists()
    assert np.load(out).shape == (1, 2, 3)


def test_export_native_json(widget, layer, tmp_path, monkeypatch):
    out = tmp_path / "boxes.json"
    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _fake_save(out))
    widget._export_native(layer.bounding_boxes)
    assert out.exists()
    assert np.array(json.loads(out.read_text())).shape == (1, 2, 3)


def test_export_coco(widget, layer, tmp_path, monkeypatch):
    out = tmp_path / "ann.json"
    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _fake_save(out))
    widget._export_coco(layer, layer.bounding_boxes)
    coco = json.loads(out.read_text())
    assert set(coco) == {"images", "annotations", "categories"}
    # box z in [2,5] -> per_slice slices 2,3,4,5 ; footprint [x=20,y=10,w=60,h=20]
    assert len(coco["annotations"]) == 4
    assert coco["annotations"][0]["bbox"] == [20.0, 10.0, 60.0, 20.0]


def test_export_yolo_writes_one_file_per_slice(widget, layer, tmp_path, monkeypatch):
    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _fake_save(tmp_path))
    widget._export_yolo(layer, layer.bounding_boxes)
    txts = sorted(tmp_path.glob("*.txt"))
    assert len(txts) == 4  # slices 2..5
    parts = txts[0].read_text().strip().split()
    assert len(parts) == 5  # class xc yc w h
    assert parts[0] == "0"


def test_export_image_shape_from_layer(widget, layer):
    assert widget._export_image_shape(layer) == (50, 200, 200)


def test_export_dispatch_coco_end_to_end(widget, layer, tmp_path, monkeypatch):
    out = tmp_path / "dispatch.json"
    monkeypatch.setattr(widget, "_current_layer", lambda: layer)
    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _fake_save(out))

    class _Input:
        @staticmethod
        def getItem(*a, **k):
            return ("COCO JSON (per-slice)", True)

    monkeypatch.setattr("napari_turbobox.widget.QInputDialog", _Input)
    widget._export_boxes()
    assert out.exists()
    assert "annotations" in json.loads(out.read_text())


def test_export_dispatch_cancelled_is_noop(widget, layer, monkeypatch):
    monkeypatch.setattr(widget, "_current_layer", lambda: layer)

    class _Input:
        @staticmethod
        def getItem(*a, **k):
            return ("", False)  # user cancelled

    monkeypatch.setattr("napari_turbobox.widget.QInputDialog", _Input)
    widget._export_boxes()  # must not raise


@pytest.mark.parametrize("image_shape", [(50, 200, 200), None])
@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_import_rejects_non_finite_boxes(widget, tmp_path, monkeypatch, image_shape, bad_val):
    lyr = TurboBoxLayer(ndim=3, image_shape=image_shape)
    monkeypatch.setattr(widget, "_current_layer", lambda: lyr)
    bad_file = tmp_path / "bad.npy"
    np.save(bad_file, np.array([[[bad_val, 10.0, 10.0], [5.0, 30.0, 30.0]]], dtype=np.float32))

    class _Open:
        @staticmethod
        def getOpenFileName(*a, **k):
            return (str(bad_file), "NumPy Array (*.npy)")

    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _Open)
    widget._import_boxes()
    assert len(lyr.bounding_boxes) == 0
    assert "Import failed" in widget.status_label.text()


def _panel_shape(make_napari_viewer, images, layer_ndim, select=None):
    viewer = make_napari_viewer()
    added = [viewer.add_image(np.zeros(shape, dtype=np.uint8), rgb=rgb) for shape, rgb in images]
    lyr = TurboBoxLayer(ndim=layer_ndim, image_shape=None)
    viewer.add_layer(lyr)
    if select is not None:
        viewer.layers.selection = {added[select]}
    w = BoundingBoxControlWidget(viewer)
    try:
        return w._export_image_shape(lyr)
    finally:
        w.close()


@pytest.mark.parametrize(
    ("images", "layer_ndim", "expected"),
    [
        ([((20, 100, 120), False)], 3, (20, 100, 120)),  # as in v0.1.0
        ([((100, 120), False)], 2, (100, 120)),  # as in v0.1.0
        ([((100, 120), False)], 3, (100, 100, 120)),  # v0.1.0 gave (100, 120): same H, W and slice clamp
        ([((100, 120, 3), True)], 2, (100, 120)),  # as in v0.1.0
        ([((20, 100, 120, 3), True)], 3, (20, 100, 120)),  # as in v0.1.0
        ([((100, 120, 3), True)], 3, (100, 100, 120)),  # v0.1.0 took the channel axis (3) as W
        ([((20, 100, 120), False)], 2, (100, 120)),  # v0.1.0 took the leading axes (20, 100)
        ([((2, 20, 100, 120), False)], 3, (20, 100, 120)),  # v0.1.0 took the leading axes (2, 20, 100)
    ],
)
def test_export_image_shape_fallback(make_napari_viewer, images, layer_ndim, expected):
    assert _panel_shape(make_napari_viewer, images, layer_ndim) == expected


@pytest.mark.parametrize("select", [None, 0, 1])
def test_export_image_shape_keeps_first_image_as_in_v010(make_napari_viewer, select):
    # Exported numbers must not depend on the selection or change against v0.1.0.
    images = [((10, 50, 60), False), ((20, 100, 120), False)]
    assert _panel_shape(make_napari_viewer, images, 3, select) == (10, 50, 60)


def test_export_native_json_filter_without_extension(widget, layer, tmp_path, monkeypatch):
    out_base = tmp_path / "boxes_no_ext"

    class _JsonFilterDialog:
        @staticmethod
        def getSaveFileName(*a, **k):
            return (str(out_base), "JSON (*.json)")

    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _JsonFilterDialog)
    widget._export_native(layer.bounding_boxes)
    out_json = tmp_path / "boxes_no_ext.json"
    assert out_json.exists()
    assert not (tmp_path / "boxes_no_ext.npy").exists()
    assert np.array(json.loads(out_json.read_text())).shape == (1, 2, 3)


def test_native_export_names_in_status_exist_and_reimport(widget, layer, tmp_path, monkeypatch):
    for name, filt in [("A.JSON", "JSON (*.json)"), ("B.NPY", "NumPy Array (*.npy)"), ("c", "JSON (*.json)")]:

        class _D:
            @staticmethod
            def getSaveFileName(*a, _n=name, _f=filt, **k):
                return (str(tmp_path / _n), _f)

        monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _D)
        widget._export_native(layer.bounding_boxes)
        written = widget.status_label.text().split(" to ", 1)[1]
        assert (tmp_path / written).exists(), written

        target = TurboBoxLayer(ndim=3, image_shape=(50, 200, 200))
        monkeypatch.setattr(widget, "_current_layer", lambda t=target: t)

        class _O:
            @staticmethod
            def getOpenFileName(*a, _w=written, **k):
                return (_w, "")

        monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _O)
        widget._import_boxes()
        assert len(target.bounding_boxes) == 1, widget.status_label.text()


def test_import_rejects_values_beyond_float32(widget, tmp_path, monkeypatch):
    lyr = TurboBoxLayer(ndim=3, image_shape=None)
    monkeypatch.setattr(widget, "_current_layer", lambda: lyr)
    bad_file = tmp_path / "big.json"
    bad_file.write_text("[[[1, 1, 1], [3, 3, 1e39]]]")

    class _Open:
        @staticmethod
        def getOpenFileName(*a, **k):
            return (str(bad_file), "")

    monkeypatch.setattr("napari_turbobox.widget.QFileDialog", _Open)
    widget._import_boxes()
    assert len(lyr.bounding_boxes) == 0
    assert "Import failed" in widget.status_label.text()
