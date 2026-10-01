"""Drive shipped drags through napari's real mouse dispatch (not ``_on_mouse_drag``
directly), so napari's own select-mode handler runs as it does for a user.

Gap closed: every drag test in the suite calls ``layer._on_mouse_drag`` with a
``Mock`` event, which bypasses ``napari.utils.interactions`` and napari's built-in
Shapes select/move handler that runs after ours.
"""

import numpy as np
import pytest
from napari.layers.shapes._shapes_constants import Mode
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from napari_turbobox import create_synchronized_bbox_layers

SHAPE = (20, 100, 100)


def _drag(layer, start, stop, n_moves=20, modifiers=(), check_mid=None):
    """Press, n_moves moves, release through napari's dispatch. ``check_mid`` is called
    after the last move and BEFORE release (what the user sees while dragging)."""
    pts = [np.asarray(start, float) + t * (np.asarray(stop, float) - np.asarray(start, float))
           for t in np.linspace(0.0, 1.0, n_moves)]
    kw = {"modifiers": list(modifiers), "dims_displayed": [1, 2], "view_direction": None, "up_direction": None}
    mouse_press_callbacks(layer, read_only_mouse_event(type="mouse_press", position=tuple(pts[0]), **kw))
    try:
        for p in pts:
            mouse_move_callbacks(layer, read_only_mouse_event(type="mouse_move", is_dragging=True,
                                                              position=tuple(p), **kw))
        if check_mid is not None:
            check_mid()
    finally:
        mouse_release_callbacks(layer, read_only_mouse_event(type="mouse_release",
                                                             position=tuple(pts[-1]), **kw))


def _view_extent(layer):
    """(y, x) min/max of the rendered 2D rectangle of box 0 in the editing view."""
    verts = np.asarray(layer.data[0])[:, -2:]
    return np.stack([verts.min(axis=0), verts.max(axis=0)])


@pytest.fixture
def realistic(make_napari_viewer):
    viewers = [make_napari_viewer() for _ in range(3)]
    for v in viewers:
        v.add_image(np.zeros(SHAPE, np.uint8))
    viewers[1].dims.order = (2, 0, 1)
    viewers[2].dims.order = (1, 0, 2)
    for v in viewers:
        v.dims.set_point(0, 10)
    layers = create_synchronized_bbox_layers(viewers[0], viewers[1:], image_shape=SHAPE)
    layers[0].add_boxes(np.array([[[5.0, 40.0, 40.0], [15.0, 58.0, 58.0]]]))
    layers[0].mode = Mode.SELECT
    return layers


def _assert_view_is_store(layer, when):
    box = np.asarray(layer.bounding_boxes[0])
    assert np.all(box[1] - box[0] >= 1), f"store box collapsed ({when})"
    np.testing.assert_allclose(_view_extent(layer), box[:, 1:], atol=1e-3,
                               err_msg=f"editing view diverged from the store ({when})")


@pytest.mark.parametrize("stop", [(10, 50, 70), (10, 50, 400)])  # inside, and past the +x edge
def test_translate_view_matches_store(realistic, stop):
    main = realistic[0]
    _drag(main, (10, 49, 49), stop, check_mid=lambda: _assert_view_is_store(main, "mid-drag"))
    _assert_view_is_store(main, "after release")


def test_shift_resize_keeps_selection_and_resizes_view(realistic):
    main = realistic[0]

    def mid():
        assert main.selected_data, "Shift-drag deselected the box (no live feedback)"
        _assert_view_is_store(main, "mid-drag")

    _drag(main, (10, 49, 49), (10, 60, 70), modifiers=("Shift",), check_mid=mid)
    _assert_view_is_store(main, "after release")
    assert np.asarray(main.bounding_boxes[0])[1, 2] > 58.0, "Shift-drag did not resize the stored box"


def test_other_2d_views_follow_each_sub_edit_as_documented(realistic):
    """The paper states that orthogonal 2D views update on every mouse sub-event.
    Count sub-viewer syncs during the moves (before release)."""
    main, yz, xz = realistic[:3]
    calls = {"yz": 0, "xz": 0}
    for name, lyr in (("yz", yz), ("xz", xz)):
        orig = lyr._perform_sync

        def spy(*a, _o=orig, _n=name, **k):
            calls[_n] += 1
            return _o(*a, **k)

        lyr._perform_sync = spy
    counted = {}

    def mid():
        counted.update(calls)

    _drag(main, (10, 49, 49), (10, 49, 59), n_moves=10, check_mid=mid)
    assert counted["yz"] >= 10 and counted["xz"] >= 10, f"sub-views not updated per sub-edit: {counted}"
