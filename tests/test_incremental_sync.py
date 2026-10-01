"""Single-box incremental view updates must match a full rebuild.

A store edit of one box updates one shape per view (``_incremental_shape_update``)
instead of re-assigning the whole Shapes data. After every edit, each view must
show exactly what a full rebuild from the store would show, including boxes that
enter or leave a view's slice, and the 3D view after the edit session ends.
"""

import numpy as np

from napari_turbobox.geometry import wireframe_path_from_bbox
from napari_turbobox.multi_view import create_synchronized_bbox_layers

IMAGE_SHAPE = (40, 96, 96)
# (dims order, sliced axis) for XY, YZ, XZ
ORIENT = [((0, 1, 2), 0), ((2, 0, 1), 2), ((1, 0, 2), 1)]


def _layout(make_napari_viewer, n=40, seed=3):
    viewers = []
    for order, sax in ORIENT:
        v = make_napari_viewer()
        v.add_image(np.zeros(IMAGE_SHAPE, np.uint8))
        v.dims.ndisplay = 2
        v.dims.order = order
        v.dims.set_point(sax, IMAGE_SHAPE[sax] // 2)
        viewers.append(v)
    v3 = make_napari_viewer()
    v3.add_image(np.zeros(IMAGE_SHAPE, np.uint8))
    v3.dims.ndisplay = 3
    viewers.append(v3)
    layers = create_synchronized_bbox_layers(
        main_viewer=viewers[0], sub_viewers=viewers[1:], image_shape=IMAGE_SHAPE
    )
    rng = np.random.default_rng(seed)
    size = np.array([8, 12, 12])
    mins = rng.uniform(0, np.array(IMAGE_SHAPE) - 1 - size, size=(n, 3)).round()
    layers[0].add_boxes(np.stack([mins, mins + size], axis=1))
    return viewers, layers


def _rows(paths):
    """Order-independent comparison key for a list of vertex arrays."""
    return sorted(np.round(np.asarray(p, float), 4).ravel().tolist() for p in paths)


def _assert_view_matches_rebuild(layer):
    shown = [np.asarray(d, float) for d in layer.data]
    expected = layer._current_paths()
    assert len(shown) == len(expected)
    assert _rows(shown) == _rows(expected)


def test_single_box_edits_match_full_rebuild(make_napari_viewer):
    _, layers = _layout(make_napari_viewer)
    store = layers[0]._bbox_store
    views_2d, layer_3d = layers[:3], layers[3]
    calls = {"incremental": 0}
    original = type(layers[0])._incremental_shape_update

    def counting(self, changed_idx, new_data):
        handled = original(self, changed_idx, new_data)
        calls["incremental"] += int(handled)
        return handled

    type(layers[0])._incremental_shape_update = counting
    try:
        rng = np.random.default_rng(7)
        for _drag in range(6):
            idx = int(rng.integers(0, len(store)))
            start = np.array(store.data[idx], float)
            # Large moves so boxes cross the XY/YZ/XZ slices at the centre.
            delta = rng.uniform(-30, 30, size=3)
            store.begin_edit_session()
            for t in np.linspace(0.0, 1.0, 15):
                box = start + t * np.stack([delta, delta])
                box = np.clip(box, 0, np.array(IMAGE_SHAPE) - 1)
                box[1] = np.maximum(box[1], box[0] + 1)
                store.update_box(idx, box)
                for layer in views_2d:
                    _assert_view_matches_rebuild(layer)
            store.end_edit_session()
            expected_3d = [wireframe_path_from_bbox(b) for b in store.data]
            assert _rows(layer_3d.data) == _rows(expected_3d)
            for layer in views_2d:
                _assert_view_matches_rebuild(layer)
    finally:
        type(layers[0])._incremental_shape_update = original
    assert calls["incremental"] > 0, "incremental path was never taken"


def test_path_index_mapping_matches_view(make_napari_viewer):
    _, layers = _layout(make_napari_viewer)
    store = layers[0]._bbox_store
    for layer in layers[:3]:
        box_ids = layer._current_shape_box_ids()
        assert box_ids is not None
        for pos, idx in enumerate(box_ids):
            assert layer._bbox_index_to_path_index(int(idx)) == pos
            assert layer._path_index_to_bbox_index(pos) == int(idx)
            shown = np.asarray(layer.data[pos], float)
            expected = layer._paths_for_single_bbox(np.asarray(store.data[idx]))[0]
            assert np.allclose(shown, expected)


def test_thumbnail_skipped_in_session_and_updated_after(make_napari_viewer):
    _, layers = _layout(make_napari_viewer)
    store = layers[0]._bbox_store
    xy = layers[0]
    counts = {"n": 0}
    original = type(xy).__mro__[1]._update_thumbnail

    def counting(self, *args, **kwargs):
        counts["n"] += 1
        return original(self, *args, **kwargs)

    type(xy).__mro__[1]._update_thumbnail = counting
    try:
        store.begin_edit_session()
        box = np.array(store.data[0], float)
        for _ in range(5):
            box = box + 0.5
            store.update_box(0, box)
        during = counts["n"]
        store.end_edit_session()
        after = counts["n"]
    finally:
        type(xy).__mro__[1]._update_thumbnail = original
    assert during == 0
    assert after >= 1
