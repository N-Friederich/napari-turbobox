"""napari's own edit tools (Add Rectangle, Delete) must edit single boxes.

Regression tests for two earlier bugs: the store was rebuilt from the visible
slice, and rectangles after the first one were dropped. Adding or deleting one
shape must add or remove exactly one box and leave all other boxes untouched,
including the ones that are not visible on the current slice.

Also checks that a click, drag or Shift-resize leaves the box selected at
release, so an immediate Delete removes exactly that box.
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
Z = 10


def _draw(layer, start, stop, n_moves=5, modifiers=(), mid=None):
    """Press, move, release through napari's dispatch (``n_moves=1``: a plain click).
    ``mid`` is called after the moves and before the release."""
    kw = {"modifiers": list(modifiers), "dims_displayed": [1, 2], "view_direction": None, "up_direction": None}
    pts = [np.asarray(start, float) + t * (np.asarray(stop, float) - np.asarray(start, float))
           for t in np.linspace(0.0, 1.0, n_moves)]
    mouse_press_callbacks(layer, read_only_mouse_event(type="mouse_press", position=tuple(pts[0]), **kw))
    for p in pts[1:]:
        mouse_move_callbacks(
            layer, read_only_mouse_event(type="mouse_move", is_dragging=True, position=tuple(p), **kw)
        )
    if mid is not None:
        mid()
    mouse_release_callbacks(layer, read_only_mouse_event(type="mouse_release", position=tuple(pts[-1]), **kw))


@pytest.fixture
def two_views(make_napari_viewer):
    viewers = [make_napari_viewer() for _ in range(2)]
    for v in viewers:
        v.add_image(np.zeros(SHAPE, np.uint8))
    viewers[1].dims.order = (1, 0, 2)
    for v in viewers:
        v.dims.set_point(0, Z)
        v.dims.set_point(1, 50)
    return create_synchronized_bbox_layers(viewers[0], viewers[1:], image_shape=SHAPE)


def _boxes(n=12):
    rng = np.random.default_rng(1)
    mins = rng.uniform(0, 80, size=(n, 3)).round()
    mins[:, 0] = rng.uniform(0, 15, size=n).round()
    return np.stack([mins, mins + np.array([4, 15, 15])], axis=1)


def _rows(a):
    return sorted(np.round(np.asarray(a, float), 3).reshape(len(a), -1).tolist())


def test_add_rectangle_adds_one_box_and_keeps_the_others(two_views):
    main = two_views[0]
    main.add_boxes(_boxes())
    before = np.array(main.bounding_boxes)
    main.mode = Mode.ADD_RECTANGLE
    _draw(main, (Z, 5, 5), (Z, 30, 40))
    after = np.array(main.bounding_boxes)
    assert len(after) == len(before) + 1
    assert _rows(after[:-1]) == _rows(before)  # nothing else changed
    new = after[-1]
    assert np.allclose(new[:, 1:], [[5, 5], [30, 40]])
    assert new[0, 0] <= Z <= new[1, 0]  # extruded through the drawn slice
    assert _rows(two_views[1].bounding_boxes) == _rows(after)  # shared store


def test_three_rectangles_on_empty_layer_are_all_stored(two_views):
    main = two_views[0]
    main.mode = Mode.ADD_RECTANGLE
    for k in range(3):
        _draw(main, (Z, 5 + 20 * k, 5), (Z, 15 + 20 * k, 25))
    assert len(main.bounding_boxes) == 3


def test_delete_removes_exactly_the_selected_box(two_views):
    main = two_views[0]
    main.add_boxes(_boxes())
    before = np.array(main.bounding_boxes)
    box_ids = main._current_shape_box_ids()
    assert box_ids is not None and len(box_ids) > 0
    victim = int(box_ids[0])
    main.mode = Mode.SELECT
    main.selected_data = {0}
    main.remove_selected()
    after = np.array(main.bounding_boxes)
    assert len(after) == len(before) - 1
    assert _rows(after) == _rows(np.delete(before, victim, axis=0))


# Known click targets at z=Z: shapes show boxes [0, 2, 3] (shape 1 = box 2); the
# y=50 sub-view shows boxes [1, 2]; at z=17 the main view shows boxes [3, 4].
SEL_BOXES = np.array([
    [[5, 10, 10], [15, 30, 30]],
    [[0, 40, 40], [3, 60, 60]],  # off the main slice
    [[5, 40, 40], [15, 60, 60]],
    [[5, 70, 70], [19, 90, 90]],
    [[16, 5, 5], [19, 8, 8]],  # only on slices z >= 16
], float)


@pytest.fixture
def sel_views(make_napari_viewer):
    viewers = [make_napari_viewer() for _ in range(2)]
    for v in viewers:
        v.add_image(np.zeros(SHAPE, np.uint8))
    viewers[1].dims.order = (1, 0, 2)
    for v in viewers:
        v.dims.set_point(0, Z)
        v.dims.set_point(1, 50)
    layers = create_synchronized_bbox_layers(viewers[0], viewers[1:], image_shape=SHAPE)
    layers[0].add_boxes(SEL_BOXES)
    layers[0].mode = Mode.SELECT
    return viewers, layers


def _selected_boxes(layer):
    ids = layer._displayed_box_ids()
    return sorted(int(ids[p]) for p in layer.selected_data)


@pytest.mark.parametrize(
    "stop, n_moves, modifiers",
    [((Z, 50, 50), 1, ()), ((Z, 55, 52), 6, ()), ((Z, 58, 57), 6, ("Shift",))],
    ids=["click", "drag", "shift_resize"],
)
def test_click_or_drag_then_delete_removes_that_box(sel_views, stop, n_moves, modifiers):
    """The release must not clear the selection napari's Delete acts on."""
    _, (main, sub) = sel_views
    before = np.array(main.bounding_boxes)
    _draw(main, (Z, 50, 50), stop, n_moves=n_moves, modifiers=modifiers)
    assert _selected_boxes(main) == [2]
    moved = np.array(main.bounding_boxes)
    if n_moves == 1:
        assert np.array_equal(moved, before)  # a click does not move the box
    sub_shapes = sub.nshapes
    main.remove_selected()
    after = np.array(main.bounding_boxes)
    assert _rows(after) == _rows(np.delete(moved, 2, axis=0))
    assert _rows(sub.bounding_boxes) == _rows(after)  # shared store
    assert sub.nshapes == sub_shapes - 1  # the y=50 sub-view dropped the deleted box


def test_selection_survives_box_leaving_a_sub_view(sel_views):
    """The dragged box leaves the y=50 sub-view (full rebuild there); the editing
    view updates incrementally and keeps its selection."""
    _, (main, sub) = sel_views
    _draw(main, (Z, 50, 50), (Z, 80, 52), n_moves=6)
    assert 2 not in sub._displayed_box_ids()
    assert _selected_boxes(main) == [2]
    moved = np.array(main.bounding_boxes)
    main.remove_selected()
    assert _rows(main.bounding_boxes) == _rows(np.delete(moved, 2, axis=0))


@pytest.mark.parametrize("start, victim", [((Z, 50, 50), 2), ((Z, 80, 80), 3)], ids=["leaves", "stays"])
def test_slice_change_mid_drag_leaves_no_stale_selection(sel_views, start, victim):
    """A translate or resize moves displayed axes only, so the dragged box can leave the
    editing view's slice only when the slice changes mid-drag (full rebuild). Box 2
    leaves (shape 1 would now be box 4); box 3 stays but moves to another shape index.
    napari clears the selection on a slice change; no shape may point at another box."""
    viewers, (main, _) = sel_views
    before = np.array(main.bounding_boxes)
    stop = np.asarray(start, float) + np.array([0, 5, 2])
    _draw(main, start, stop, n_moves=6, mid=lambda: viewers[0].dims.set_point(0, 17))
    moved = np.array(main.bounding_boxes)
    changed = [i for i in range(len(before)) if not np.array_equal(before[i], moved[i])]
    assert changed == [victim]
    selected = _selected_boxes(main)
    assert selected in ([], [victim])
    main.remove_selected()
    after = np.array(main.bounding_boxes)
    expected = np.delete(moved, victim, axis=0) if selected else moved
    assert _rows(after) == _rows(expected)


def test_read_only_view_cannot_delete(two_views):
    main, sub = two_views
    main.add_boxes(_boxes())
    before = np.array(main.bounding_boxes)
    assert sub.editable is False
    if sub.nshapes:
        sub.selected_data = {0}
        sub.remove_selected()
    assert _rows(main.bounding_boxes) == _rows(before)
