"""Tests for COCO / YOLO export. Expected values are hand-computed, not
derived from the implementation, so the assertions are non-tautological.

Coordinate convention under test: boxes are napari ``(z, y, x)`` (3D) or
``(y, x)`` (2D); COCO/YOLO ``x`` = last axis (column), ``y`` = 2nd-to-last
(row). COCO ``bbox`` = ``[x, y, w, h]`` (top-left + size); YOLO is
center-normalized.
"""

import numpy as np
import pytest

from napari_turbobox.export import boxes_to_coco, boxes_to_yolo


def test_coco_yolo_2d_box_exact_values():
    # 2D box, axes (y, x): mins (y=10, x=20), maxs (y=30, x=80)
    boxes = np.array([[[10.0, 20.0], [30.0, 80.0]]])  # (1, 2, 2)
    image_shape = (100, 200)  # (Y, X)

    coco = boxes_to_coco(boxes, image_shape)
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    # x=20, y=10, w=80-20=60, h=30-10=20  -> [x, y, w, h]
    assert ann["bbox"] == [20.0, 10.0, 60.0, 20.0]
    assert ann["area"] == 60.0 * 20.0
    assert ann["iscrowd"] == 0
    assert ann["category_id"] == 1
    assert coco["categories"] == [{"id": 1, "name": "object"}]
    img = coco["images"][0]
    assert (img["width"], img["height"]) == (200, 100)
    assert img["file_name"] == "image.png"  # 2D -> single image

    yolo = boxes_to_yolo(boxes, image_shape)
    assert list(yolo) == ["image.txt"]
    # xc=(20+30)/200=0.25, yc=(10+10)/100=0.20, w=60/200=0.30, h=20/100=0.20
    cls, xc, yc, w, h = yolo["image.txt"].strip().split()
    assert cls == "0"
    assert (float(xc), float(yc), float(w), float(h)) == (0.25, 0.20, 0.30, 0.20)


def test_coco_bbox_is_xywh_not_xyxy():
    boxes = np.array([[[10.0, 20.0], [30.0, 80.0]]])
    ann = boxes_to_coco(boxes, (100, 200))["annotations"][0]
    # If this were [x_min, y_min, x_max, y_max] the 3rd value would be 80, not w=60.
    assert ann["bbox"][2] == 60.0
    assert ann["bbox"][3] == 20.0


def test_3d_per_slice_three_slices_identical_box():
    # z spans 2.3..5.7 -> ceil(2.3)=3, floor(5.7)=5 -> slices 3, 4, 5
    boxes = np.array([[[2.3, 10.0, 20.0], [5.7, 30.0, 80.0]]])  # (1, 2, 3) (z,y,x)
    image_shape = (10, 100, 200)  # (Z, Y, X)

    coco = boxes_to_coco(boxes, image_shape, mode="per_slice")
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 3
    assert sorted(im["file_name"] for im in coco["images"]) == [
        "slice_0003.png",
        "slice_0004.png",
        "slice_0005.png",
    ]
    # every slice shares the same footprint [x=20, y=10, w=60, h=20]
    for ann in coco["annotations"]:
        assert ann["bbox"] == [20.0, 10.0, 60.0, 20.0]
    assert {ann["image_id"] for ann in coco["annotations"]} == {3, 4, 5}

    yolo = boxes_to_yolo(boxes, image_shape, mode="per_slice")
    assert sorted(yolo) == ["slice_0003.txt", "slice_0004.txt", "slice_0005.txt"]


def test_3d_subslice_thin_box_emits_single_midslice():
    # z spans 4.2..4.6 -> ceil(4.2)=5 > floor(4.6)=4 -> single slice round(4.4)=4
    boxes = np.array([[[4.2, 10.0, 20.0], [4.6, 30.0, 80.0]]])
    coco = boxes_to_coco(boxes, (10, 100, 200), mode="per_slice")
    assert len(coco["annotations"]) == 1
    assert coco["images"][0]["file_name"] == "slice_0004.png"
    assert coco["annotations"][0]["image_id"] == 4


def test_max_projection_one_annotation_per_box():
    boxes = np.array(
        [
            [[2.3, 10.0, 20.0], [5.7, 30.0, 80.0]],
            [[1.0, 5.0, 5.0], [9.0, 15.0, 25.0]],
        ]
    )  # 2 boxes, each spanning several slices
    coco = boxes_to_coco(boxes, (10, 100, 200), mode="max_projection")
    assert len(coco["annotations"]) == 2  # NOT expanded per slice
    assert len(coco["images"]) == 1
    assert coco["images"][0]["file_name"] == "image.png"

    yolo = boxes_to_yolo(boxes, (10, 100, 200), mode="max_projection")
    assert list(yolo) == ["image.txt"]
    assert len(yolo["image.txt"].strip().splitlines()) == 2


def test_axis_order_non_square_no_xy_swap():
    # x-extent (axis 2) = 60, y-extent (axis 1) = 20 -> must NOT be swapped
    boxes = np.array([[[0.0, 10.0, 20.0], [2.0, 30.0, 80.0]]])
    image_shape = (10, 100, 200)  # (Z, Y, X): H=100, W=200

    ann = boxes_to_coco(boxes, image_shape, mode="max_projection")["annotations"][0]
    assert ann["bbox"][2] == 60.0  # w = x-extent
    assert ann["bbox"][3] == 20.0  # h = y-extent
    assert ann["bbox"][2] != ann["bbox"][3]  # genuinely non-square

    yolo = boxes_to_yolo(boxes, image_shape, mode="max_projection")
    _cls, _xc, _yc, w, h = yolo["image.txt"].strip().split()
    # w normalized against W=200 (x), h against H=100 (y)
    assert float(w) == 60.0 / 200.0
    assert float(h) == 20.0 / 100.0


def test_empty_store():
    boxes = np.empty((0, 2, 3), dtype=float)
    coco = boxes_to_coco(boxes, (10, 100, 200))
    assert coco["images"] == []
    assert coco["annotations"] == []
    assert coco["categories"] == [{"id": 1, "name": "object"}]
    assert boxes_to_yolo(boxes, (10, 100, 200)) == {}


def test_out_of_bounds_box_clamped_and_warns():
    # x_max = 120 exceeds W = 100 -> clamp to 100; w = 100 - 20 = 80
    boxes = np.array([[[0.0, 10.0, 20.0], [2.0, 30.0, 120.0]]])
    image_shape = (10, 100, 100)  # W = H = 100
    with pytest.warns(UserWarning, match="outside image bounds"):
        coco = boxes_to_coco(boxes, image_shape, mode="max_projection")
    assert coco["annotations"][0]["bbox"] == [20.0, 10.0, 80.0, 20.0]


def test_category_ids_mapping():
    boxes = np.array(
        [
            [[0.0, 10.0, 20.0], [2.0, 30.0, 80.0]],
            [[0.0, 5.0, 5.0], [2.0, 15.0, 25.0]],
        ]
    )
    coco = boxes_to_coco(
        boxes,
        (10, 100, 200),
        mode="max_projection",
        category_ids=[0, 2],
        category_names={0: "nucleus", 2: "cell"},
    )
    # COCO category_id = class_id + 1
    assert {c["id"]: c["name"] for c in coco["categories"]} == {1: "nucleus", 3: "cell"}
    assert sorted(a["category_id"] for a in coco["annotations"]) == [1, 3]

    yolo = boxes_to_yolo(boxes, (10, 100, 200), mode="max_projection", category_ids=[0, 2])
    classes = sorted(int(line.split()[0]) for line in yolo["image.txt"].strip().splitlines())
    assert classes == [0, 2]  # YOLO uses class id verbatim (0-indexed)
