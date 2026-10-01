"""Application workflow on a public confocal volume: C. elegans nuclei.

Dataset
    Long F, Peng H, Liu X, Kim SK, Myers E, Kainmueller D, Weigert M (2022).
    3D nuclei instance segmentation dataset of fluorescence microscopy volumes
    of C. elegans. Zenodo. https://doi.org/10.5281/zenodo.5942575
    Licence CC-BY-4.0. File c_elegans_nuclei.zip, 84,403,531 bytes,
    md5 7d616d91c4a6cce75a7d4dd37d7e666a.
    If you use the data, the record asks you to cite Long et al. (2009),
    Nat. Methods 6:667-672 (raw data). The curated masks are from Hirsch and
    Kainmueller (2020), MIDL; the train/val/test split follows Weigert et al.
    (2020), WACV.

What this script does
    1. Downloads the zip once into the data directory (--data, else
       $TURBOBOX_DATA, else ~/.cache/napari-turbobox), checks its size and md5,
       and extracts the first training volume and its mask,
       c_elegans_nuclei/train/{images,masks}/C18G1_2L1_1.tif, as
       volume001_raw.tif and volume001_gt.tif. If these two files are already
       in the data directory, nothing is downloaded.
    2. Derives one box per nucleus from the mask (labels_to_boxes.py in this
       folder). tifffile loads both files with shape (140, 140, 1244), used as
       (z, y, x).
    3. --headless (default): writes boxes.npy, label_ids.npy,
       coco_per_slice.json and yolo/ to --out (default <data>/celegans_export)
       and prints the counts. No viewer window is opened, but napari and a Qt
       binding must be installed.
       --gui: opens four napari windows, XY (editable), XZ and YZ (read-only)
       and 3D (read-only), with the boxes and the BBox Control Panel.

Expected export result (tests/fixtures/celegans_volume001_expected.json):
555 boxes; COCO 64 images (z slices 39 to 102) with 7,262 annotations;
64 YOLO label files.

Usage (from the repository root, with napari-turbobox installed)::

    python examples/celegans_workflow.py [--data DIR] [--out DIR] [--no-download]
    python examples/celegans_workflow.py --gui
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from labels_to_boxes import (
    export_boxes,
    labels_to_boxes,
    load_volume,
    open_multiview,
    print_counts,
)

ZIP_URL = "https://zenodo.org/api/records/5942575/files/c_elegans_nuclei.zip/content"
ZIP_NAME = "c_elegans_nuclei.zip"
ZIP_BYTES = 84_403_531
ZIP_MD5 = "7d616d91c4a6cce75a7d4dd37d7e666a"
# archive member -> (file name in the data directory, size in bytes)
IMAGE = ("c_elegans_nuclei/train/images/C18G1_2L1_1.tif", "volume001_raw.tif", 24_400_239)
MASK = ("c_elegans_nuclei/train/masks/C18G1_2L1_1.tif", "volume001_gt.tif", 439_784)


def default_data_dir() -> Path:
    """``$TURBOBOX_DATA`` if set, else ``~/.cache/napari-turbobox``."""
    return Path(os.environ.get("TURBOBOX_DATA") or Path.home() / ".cache" / "napari-turbobox")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _present(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


def fetch_volume(data_dir: str | Path, allow_download: bool = True) -> tuple[Path, Path]:
    """Return ``(image_path, mask_path)``, downloading and extracting if needed.

    Raises ``FileNotFoundError`` if the files are missing and ``allow_download``
    is False.
    """
    data = Path(data_dir).expanduser()
    image_path, mask_path = data / IMAGE[1], data / MASK[1]
    if _present(image_path, IMAGE[2]) and _present(mask_path, MASK[2]):
        return image_path, mask_path
    zip_path = data / ZIP_NAME
    if not zip_path.is_file():
        if not allow_download:
            raise FileNotFoundError(f"{zip_path} is missing and downloading is disabled")
        data.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {ZIP_URL} ({ZIP_BYTES / 1e6:.1f} MB) to {zip_path}")
        part = zip_path.with_name(ZIP_NAME + ".part")
        request = urllib.request.Request(ZIP_URL, headers={"User-Agent": "napari-turbobox"})
        with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as f:
            shutil.copyfileobj(response, f, 1 << 20)
        part.replace(zip_path)
    if zip_path.stat().st_size != ZIP_BYTES or _md5(zip_path) != ZIP_MD5:
        raise RuntimeError(f"{zip_path} does not match the Zenodo record; delete it and rerun")
    with zipfile.ZipFile(zip_path) as archive:
        for member, name, _size in (IMAGE, MASK):
            part = data / (name + ".part")
            with archive.open(member) as src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            part.replace(data / name)
    return image_path, mask_path


def main(argv: list[str] | None = None) -> dict[str, int] | None:
    parser = argparse.ArgumentParser(description="C. elegans nuclei -> boxes -> exports.")
    parser.add_argument("--data", default=None, help="data directory (see module docstring)")
    parser.add_argument("--out", default=None, help="export directory")
    parser.add_argument("--no-download", action="store_true", help="fail if data is missing")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="export files (default)")
    mode.add_argument("--gui", action="store_true", help="open napari instead of exporting")
    args = parser.parse_args(argv)

    data = Path(args.data).expanduser() if args.data else default_data_dir()
    image_path, mask_path = fetch_volume(data, allow_download=not args.no_download)
    image, labels = load_volume(image_path), load_volume(mask_path)
    if image.shape != labels.shape:
        raise ValueError(f"shape mismatch: image {image.shape}, mask {labels.shape}")
    boxes, ids = labels_to_boxes(labels)
    print(f"{len(boxes)} nuclei boxes, volume shape (z, y, x) = {labels.shape}")
    if args.gui:
        import napari

        _session = open_multiview(image, boxes)  # keep references while napari runs
        napari.run()
        return None
    out = Path(args.out).expanduser() if args.out else data / "celegans_export"
    counts = export_boxes(boxes, image.shape, out, label_ids=ids)
    print_counts(counts, out)
    return counts


if __name__ == "__main__":
    main()
