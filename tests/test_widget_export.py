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
