"""Real-data application workflow: C. elegans nuclei (Zenodo 10.5281/zenodo.5942575).

Runs ``examples/celegans_workflow.py`` in export mode (no viewer window) on the
first training volume and compares the boxes and the COCO/YOLO exports with
``tests/fixtures/celegans_volume001_expected.json``: counts and hashes taken
from a recorded run (see the fixture's ``provenance`` block).

Data lookup: ``$TURBOBOX_DATA``, else ``~/.cache/napari-turbobox``, holding
either the extracted ``volume001_raw.tif`` and ``volume001_gt.tif`` or
``c_elegans_nuclei.zip``. If neither is there the tests are skipped, unless
``TURBOBOX_ALLOW_DOWNLOAD=1`` is set; then the 84 MB zip is downloaded from
Zenodo and its md5 is checked.
"""

from __future__ import annotations

import collections
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "celegans_volume001_expected.json"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def expected():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def workflow_run(tmp_path_factory):
    """Run the example CLI once; return ``(module, mask_path, out_dir, counts)``."""
    pytest.importorskip("tifffile")
    pytest.importorskip("scipy")
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(EXAMPLES))  # celegans_workflow imports labels_to_boxes
        workflow = importlib.import_module("celegans_workflow")
    data_dir = workflow.default_data_dir()
    allow_download = os.environ.get("TURBOBOX_ALLOW_DOWNLOAD", "") not in ("", "0")
    try:
        _, mask_path = workflow.fetch_volume(data_dir, allow_download=allow_download)
    except FileNotFoundError:
        pytest.skip(
            f"C. elegans volume not found in {data_dir}; "
            "set TURBOBOX_DATA to a folder that has it, or TURBOBOX_ALLOW_DOWNLOAD=1"
        )
    out_dir = tmp_path_factory.mktemp("celegans_export")
    counts = workflow.main(["--data", str(data_dir), "--out", str(out_dir), "--no-download"])
    return workflow, mask_path, out_dir, counts


def test_one_valid_box_per_label(workflow_run, expected):
    workflow, mask_path, out_dir, counts = workflow_run
    boxes = np.load(out_dir / "boxes.npy")
    label_ids = np.load(out_dir / "label_ids.npy")
    shape = np.asarray(expected["image_shape"], dtype=float)

    assert counts["boxes"] == expected["n_boxes"]
    assert boxes.shape == (expected["n_boxes"], 2, 3)
    assert boxes.dtype == np.float32  # the store's dtype, as in the panel's native export
    # The invariant TurboBoxLayer enforces on add_boxes and on import.
    assert np.all(boxes[:, 0] >= 0)
    assert np.all(boxes[:, 1] <= shape - 1)
    assert np.all(boxes[:, 1] - boxes[:, 0] >= 1)

    labels = workflow.load_volume(mask_path)
    assert labels.shape == tuple(expected["image_shape"])
    present = np.unique(labels)
    present = present[present > 0]
    assert len(present) == expected["n_labels"]
    np.testing.assert_array_equal(label_ids, present)


def test_plugin_layer_accepts_boxes_without_correction(workflow_run, expected):
    from napari_turbobox import TurboBoxLayer

    _, _, out_dir, _ = workflow_run
    boxes = np.load(out_dir / "boxes.npy")
    layer = TurboBoxLayer(ndim=3, image_shape=tuple(expected["image_shape"]))
    layer.add_boxes(boxes)  # raises ValueError if any box breaks the invariant
    np.testing.assert_array_equal(layer.bounding_boxes, boxes)


def test_coco_and_yolo_counts_match_fixture(workflow_run, expected):
    _, _, out_dir, counts = workflow_run
    coco = json.loads((out_dir / "coco_per_slice.json").read_text())
    exp = expected["coco"]
    height, width = expected["image_shape"][1:]

    assert counts["coco_images"] == len(coco["images"]) == exp["images"]
    assert counts["coco_annotations"] == len(coco["annotations"]) == exp["annotations"]
    assert coco["categories"] == exp["categories"]
    slices = sorted(image["id"] for image in coco["images"])
    assert slices == list(range(exp["first_slice"], exp["last_slice"] + 1))
    assert all(im["height"] == height and im["width"] == width for im in coco["images"])
    per_slice = collections.Counter(ann["image_id"] for ann in coco["annotations"])
    assert [per_slice[s] for s in slices] == exp["annotations_per_slice"]

    yolo_files = sorted((out_dir / "yolo").glob("*.txt"))
    assert counts["yolo_files"] == len(yolo_files) == expected["yolo"]["files"]
    assert [p.name for p in yolo_files] == [f"slice_{s:04d}.txt" for s in slices]
    n_lines = sum(len(p.read_text().splitlines()) for p in yolo_files)
    assert n_lines == expected["yolo"]["lines"]


def test_exports_match_recorded_run(workflow_run, expected):
    _, _, out_dir, _ = workflow_run
    coco = json.loads((out_dir / "coco_per_slice.json").read_text())
    canonical = json.dumps(coco, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == expected["coco"]["sha256"]

    digest = hashlib.sha256()
    for path in sorted((out_dir / "yolo").glob("*.txt"), key=lambda p: p.name):
        digest.update(path.name.encode() + b"\0" + path.read_text().encode() + b"\0")
    assert digest.hexdigest() == expected["yolo"]["sha256"]
