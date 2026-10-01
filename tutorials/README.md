# napari-TurboBox tutorials

| Notebook | What it shows |
|---|---|
| [`00_napari_turbobox_api.ipynb`](00_napari_turbobox_api.ipynb) | The Python API on a synthetic volume: box layer, validation, undo, slice-aware drawing, the synchronized four-viewer layout and its shared store, listeners and bulk edits, COCO/YOLO/native export and import, boxes as prompts, 2D images, the control panel |
| [`01_confocal_celegans_nuclei.ipynb`](01_confocal_celegans_nuclei.ipynb) | Confocal microscopy: *C. elegans* nuclei (Zenodo 5942575) |
| [`02_electron_microscopy_nucmm_z.ipynb`](02_electron_microscopy_nucmm_z.ipynb) | Serial-section electron microscopy: zebrafish neuronal nuclei (NucMM-Z, authors' HuggingFace release) |
| [`03_micro_ct_nucmm_m.ipynb`](03_micro_ct_nucmm_m.ipynb) | X-ray micro-CT: mouse cortex nuclei (NucMM-M) |
| [`04_light_sheet_cellseg3d_mouse_brain.ipynb`](04_light_sheet_cellseg3d_mouse_brain.ipynb) | Light-sheet microscopy: nuclei in a cleared mouse brain (CellSeg3D, Zenodo 11095111) |
| [`05_instant_sim_polii_clusters.ipynb`](05_instant_sim_polii_clusters.ipynb) | Instant-SIM: nuclei and RNA Pol II clusters in a zebrafish embryo (Hajiabadi et al. 2022, Zenodo 5568871), boxes from a threshold segmentation |

Each application notebook downloads one public volume into a local cache,
checks it against the size and checksum published by the data repository,
derives one 3D box per object, loads the boxes into the synchronized
four-viewer layout (XY editable; XZ, YZ and 3D read-only), replays a mouse
drag on one box through napari's mouse-event dispatch, and exports the boxes
as native, COCO and YOLO files. The counts of every notebook are stored in
[`results/`](results/) together with a screenshot of the four views.

## Data sources and licences

| Notebook | Data | Licence | Download |
|---|---|---|---|
| 01 | Long et al. (2022), Zenodo [10.5281/zenodo.5942575](https://doi.org/10.5281/zenodo.5942575) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 84 MB zip (MD5 checked) |
| 02, 03 | Lin et al. (2021), NucMM, HuggingFace [`pytc/NucMM`](https://huggingface.co/datasets/pytc/NucMM) @ `e571f8b` | MIT | one crop each via HTTP range requests (0.3 MB / 5.2 MB, sha256 checked) |
| 04 | Mathis Laboratory of Adaptive Intelligence; Achard et al. (CellSeg3D), Zenodo [10.5281/zenodo.11095111](https://doi.org/10.5281/zenodo.11095111) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 31 MB zip (MD5 checked) |
| 05 | Hajiabadi et al. (2022), Zenodo [10.5281/zenodo.5568871](https://doi.org/10.5281/zenodo.5568871) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | one 40 MB stack via HTTP range requests (12 MB transferred, CRC-32 and sha256 checked) |

Please cite the data sources above when you use the data. The screenshots in
`results/` and in the notebook outputs show crops of these volumes.

## Running the notebooks

```bash
pip install -e "..[all,tutorials]"   # from this folder: napari, a Qt binding, JupyterLab, h5py, ...
jupyter lab
```

- Data are cached in `$TURBOBOX_DATA` if set, else in `~/.cache/napari-turbobox`.
- `SHOW = False` (first code cell) keeps the napari windows hidden, so a
  notebook also runs unattended (`jupyter nbconvert --to notebook --execute`).
  Set `SHOW = True` to work with the four viewers interactively.
- The helpers shared by the notebooks are in [`tutorial_utils.py`](tutorial_utils.py);
  the box derivation and export functions come from
  [`../examples/labels_to_boxes.py`](../examples/labels_to_boxes.py), which also
  works as a command-line tool for your own image and label files.
- The mouse-drag replay uses napari's test helper
  `napari.utils._test_utils.read_only_mouse_event` to build the events; it is a
  check, not something you need for your own annotation work.
- The notebooks print paths through `tu.show_path`, relative to the repository
  or with the home directory as `~`. In the saved notebooks, the output lines
  with paths (data directory, result file and, where printed, export directory)
  were edited to this form after the run, without executing the notebooks again.
  All other outputs are as recorded.
