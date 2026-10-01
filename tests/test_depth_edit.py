"""Setting a box's depth (its extent along the axis hidden in the editing view) with the mouse.

A rectangle drawn with Add Rectangle is extruded through the full
depth, and the orthogonal views made by ``create_synchronized_bbox_layers`` are
read-only. README "Limitations" says the depth can be set by reorienting the
editable viewer so that it displays the depth axis (``viewer.dims.order`` or
napari's roll button) and dragging or Shift-dragging the box there.

Layout: XY editable viewer, YZ and XZ read-only viewers, one 3D viewer, all
hidden. Every mouse event goes through napari's dispatch
(``mouse_press_callbacks`` / ``mouse_move_callbacks`` / ``mouse_release_callbacks``),
with ``dims_displayed`` taken from the viewer at the time of the event, as in
tests/test_mouse_dispatch_realpath.py. Undo goes through the viewer's key map
handler (the Ctrl+Z binding the layer registers in ``_post_init``).
"""

import math

import numpy as np
import pytest

napari = pytest.importorskip("napari")

from napari.layers.shapes._shapes_constants import Mode  # noqa: E402
from napari.utils._test_utils import read_only_mouse_event  # noqa: E402
from napari.utils.interactions import (  # noqa: E402
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from napari_turbobox import create_synchronized_bbox_layers  # noqa: E402
from napari_turbobox.export import boxes_to_coco, boxes_to_yolo  # noqa: E402

SHAPE = (20, 100, 100)  # (z, y, x)
Z0 = 10  # slice the rectangle is drawn on
Y_C = X_C = 50.0  # centre of the drawn rectangle in y and x (also every view's slice point)
FULL = (0.0, SHAPE[0] - 1.0)  # full depth after extrusion
Z1, Z2 = 4.3, 12.7  # target depth; non-integer so ceil/floor matter for export
ATOL = 1e-4  # the store keeps float32 coordinates


def _dispatch(viewer, layer, start, stop, n_moves=8, modifiers=()):
    """Press, ``n_moves`` moves, release on ``layer`` through napari's mouse dispatch.

    ``dims_displayed`` is read from the viewer now, so it follows a reorientation.
    """
    kw = {
        "modifiers": list(modifiers),
        "dims_displayed": list(viewer.dims.displayed),
        "view_direction": None,
        "up_direction": None,
    }
    start, stop = np.asarray(start, float), np.asarray(stop, float)
    pts = [start + t * (stop - start) for t in np.linspace(0.0, 1.0, n_moves)]
    mouse_press_callbacks(layer, read_only_mouse_event(type="mouse_press", position=tuple(pts[0]), **kw))
    try:
        for p in pts[1:]:
            mouse_move_callbacks(
                layer, read_only_mouse_event(type="mouse_move", is_dragging=True, position=tuple(p), **kw)
            )
    finally:
        mouse_release_callbacks(layer, read_only_mouse_event(type="mouse_release", position=tuple(pts[-1]), **kw))


def _at(z):
    """World position at depth ``z`` over the box centre (the hidden axis sits at 50 too)."""
    return (float(z), Y_C, X_C)


def _press_key(viewer, layer, key):
    """Press ``key`` through the viewer's key map handler with ``layer`` active."""
    viewer.layers.selection.active = layer
    handler = viewer.window._qt_viewer._key_map_handler
    assert layer in handler.keymap_providers, "the box layer's key bindings are not active"
    return handler.press_key(key)


def _extent(shape_data):
    v = np.asarray(shape_data, float)
    return np.stack([v.min(axis=0), v.max(axis=0)])


def _assert_view_shows_store(viewer, layer, when):
    """The view shows box 0 exactly where the store says, or nothing if its slice misses it."""
    box = np.asarray(layer.bounding_boxes[0], float)
    if viewer.dims.ndisplay == 3:
        assert layer.nshapes == 1, f"3D view: {layer.nshapes} shapes ({when})"
        np.testing.assert_allclose(_extent(layer.data[0]), box, atol=ATOL,
                                   err_msg=f"3D wireframe differs from the store ({when})")
        return
    displayed = [int(a) for a in viewer.dims.displayed]
    (hidden,) = [a for a in range(3) if a not in displayed]
    p = float(viewer.dims.point[hidden])
    inside = box[0, hidden] - 1e-6 <= p <= box[1, hidden] + 1e-6
    if not inside:
        assert layer.nshapes == 0, (
            f"view {displayed} at axis {hidden}={p} shows a box that ends at "
            f"{box[:, hidden]} ({when})"
        )
        return
    assert layer.nshapes == 1, f"view {displayed} at axis {hidden}={p}: {layer.nshapes} shapes ({when})"
    ext = _extent(layer.data[0])
    np.testing.assert_allclose(ext[:, displayed], box[:, displayed], atol=ATOL,
                               err_msg=f"view {displayed} differs from the store ({when})")


def _exported_slices(boxes):
    coco = boxes_to_coco(boxes, SHAPE, mode="per_slice")
    yolo = boxes_to_yolo(boxes, SHAPE, mode="per_slice")
    coco_ids = sorted({a["image_id"] for a in coco["annotations"]})
    yolo_ids = sorted(int(name[len("slice_"):-len(".txt")]) for name in yolo)
    return coco_ids, yolo_ids


@pytest.fixture
def layout(make_napari_viewer):
    """XY (editable), YZ, XZ (read-only 2D) and 3D (read-only) hidden viewers.

    The viewers come from napari's ``make_napari_viewer``, which also closes them.
    A ``napari.Viewer`` made directly registers the installed plugins with napari's
    own app model, and the first one made after a ``make_napari_viewer`` test
    registers them again: in one pytest process that fails with "Command
    '<plugin>.get_reader' already registered".
    """
    viewers = [make_napari_viewer() for _ in range(4)]
    for v in viewers:
        v.add_image(np.zeros(SHAPE, np.uint8))
    xy, yz, xz, v3d = viewers
    yz.dims.order = (2, 0, 1)  # shows (z, y), sliced along x
    xz.dims.order = (1, 0, 2)  # shows (z, x), sliced along y
    v3d.dims.ndisplay = 3  # before the layers are created (README)
    for v in viewers:
        v.dims.set_point(0, Z0)
        v.dims.set_point(1, Y_C)
        v.dims.set_point(2, X_C)
    layers = create_synchronized_bbox_layers(xy, [yz, xz, v3d], image_shape=SHAPE)
    return viewers, layers


def _draw_full_depth_box(viewers, layers):
    """Draw one rectangle with Add Rectangle in XY at z = Z0; it spans the full depth."""
    xy, main = viewers[0], layers[0]
    main.mode = Mode.ADD_RECTANGLE
    _dispatch(xy, main, (Z0, 30.0, 30.0), (Z0, 70.0, 70.0))
    main.mode = Mode.SELECT
    boxes = main.bounding_boxes
    assert boxes.shape == (1, 2, 3), f"Add Rectangle did not store exactly one box: {boxes}"
    np.testing.assert_allclose(boxes[0][:, 0], FULL, atol=ATOL,
                               err_msg="the drawn rectangle is not extruded through the full depth")
    np.testing.assert_allclose(boxes[0][:, 1:], [[30, 30], [70, 70]], atol=1e-3)
    return boxes[0].copy()


def _reorient(viewer, how):
    if how == "order":  # README: viewer.dims.order = (1, 0, 2)
        viewer.dims.order = (1, 0, 2)
    else:  # napari's roll button
        viewer.dims.roll()
    assert 0 in viewer.dims.displayed, f"depth axis not displayed after {how}: {viewer.dims.displayed}"


def _resize_depth(viewer, layer, method):
    """Two drags in the reoriented editable view: z_min 0 -> Z1, then z_max 19 -> Z2."""
    if method == "edge":
        # Press within 20 % of the box size of the z edge (layer.py _hit_test_bbox).
        _dispatch(viewer, layer, _at(1.0), _at(1.0 + Z1))
        _dispatch(viewer, layer, _at(18.0), _at(18.0 - (FULL[1] - Z2)))
    else:  # Shift-drag from the box interior; the half of the press picks the edge
        _dispatch(viewer, layer, _at(7.0), _at(7.0 + Z1), modifiers=("Shift",))
        _dispatch(viewer, layer, _at(14.0), _at(14.0 - (FULL[1] - Z2)), modifiers=("Shift",))


@pytest.mark.parametrize("method", ["edge", "shift"])
@pytest.mark.parametrize("how", ["order", "roll"])
def test_depth_set_by_reorienting_the_editable_viewer(layout, how, method):
    viewers, layers = layout
    xy, main = viewers[0], layers[0]
    drawn = _draw_full_depth_box(viewers, layers)
    for v, lyr in zip(viewers, layers):
        _assert_view_shows_store(v, lyr, "after drawing")
    assert _exported_slices(main.bounding_boxes) == (list(range(20)), list(range(20)))

    # (a) show the depth axis in the editable viewer and resize the box along it.
    _reorient(xy, how)
    assert tuple(int(a) for a in main._slice_input.displayed) == tuple(int(a) for a in xy.dims.displayed)
    _resize_depth(xy, main, method)

    box = main.bounding_boxes[0]
    np.testing.assert_allclose(box[:, 0], [Z1, Z2], atol=ATOL, err_msg="store z-extent")
    np.testing.assert_allclose(box[:, 1:], drawn[:, 1:], atol=ATOL, err_msg="y/x changed by the depth edit")
    for v, lyr in zip(viewers, layers):  # reoriented editable view, YZ, XZ, 3D
        _assert_view_shows_store(v, lyr, f"after the depth edit ({how}, {method})")

    # Back to XY: the box shows on the slices inside [Z1, Z2] only.
    xy.dims.order = (0, 1, 2)
    for z in range(SHAPE[0]):
        xy.dims.set_point(0, z)
        _assert_view_shows_store(xy, main, f"XY at z={z}")
        assert (main.nshapes == 1) == (Z1 <= z <= Z2), f"XY at z={z}: {main.nshapes} shapes"
    xy.dims.set_point(0, Z0)

    expected = list(range(math.ceil(Z1), math.floor(Z2) + 1))  # 5..12
    assert _exported_slices(main.bounding_boxes) == (expected, expected)

    # Undo both drags with the Ctrl+Z binding: the full depth comes back everywhere.
    assert main.get_undo_info()["undo_description"] == "Move/Resize box"
    assert _press_key(xy, main, "Control-Z"), "Ctrl+Z is not bound on the editable layer"
    np.testing.assert_allclose(main.bounding_boxes[0][:, 0], [Z1, FULL[1]], atol=ATOL,
                               err_msg="first undo did not restore z_max")
    assert _press_key(xy, main, "Control-Z")
    np.testing.assert_allclose(main.bounding_boxes[0], drawn, atol=ATOL,
                               err_msg="second undo did not restore the drawn full-depth box")
    for v, lyr in zip(viewers, layers):
        _assert_view_shows_store(v, lyr, "after undo")
    assert _exported_slices(main.bounding_boxes) == (list(range(20)), list(range(20)))


def test_depth_is_kept_unless_the_depth_axis_is_displayed(layout):
    """Without reorientation the mouse cannot change the depth (A-MUST-25's valid part).

    Edge and Shift resizes in the XY view change only y/x; with the editable
    viewer in 3D display napari makes the layer non-editable (PAN_ZOOM).
    """
    viewers, layers = layout
    xy, main = viewers[0], layers[0]
    drawn = _draw_full_depth_box(viewers, layers)
    _dispatch(xy, main, (Z0, 31.0, X_C), (Z0, 36.0, X_C))  # near the y-min edge
    _dispatch(xy, main, (Z0, 45.0, 45.0), (Z0, 40.0, 40.0), modifiers=("Shift",))
    box = main.bounding_boxes[0]
    np.testing.assert_allclose(box[:, 0], drawn[:, 0], atol=ATOL, err_msg="an XY drag changed the depth")
    assert not np.allclose(box[:, 1:], drawn[:, 1:]), "the XY resizes did not act at all"
    before = main.bounding_boxes.copy()
    xy.dims.ndisplay = 3
    assert not main.editable
    main.mode = Mode.SELECT  # napari keeps a non-editable layer in PAN_ZOOM
    _dispatch(xy, main, _at(1.0), _at(1.0 + Z1), modifiers=("Shift",))
    np.testing.assert_allclose(main.bounding_boxes, before, atol=ATOL, err_msg="a drag in 3D display edited the box")


def test_readonly_orthogonal_view_cannot_take_the_editing_role(layout):
    """(b) No public switch makes an orthogonal view editable (README Limitations).

    ``editable``/``mode`` are napari's public knobs; ``sync_mode`` only changes
    when a layer is notified. None of them lets a drag in the XZ view write the
    store, and the XZ view keeps showing the store.
    """
    viewers, layers = layout
    xz, sub = viewers[2], layers[2]
    drawn = _draw_full_depth_box(viewers, layers)
    for mode in ("live", "on_commit"):
        sub.sync_mode = mode
        sub.editable = True
        sub.mode = Mode.SELECT
        for modifiers in ((), ("Shift",)):
            _dispatch(xz, sub, _at(1.0), _at(1.0 + Z1), modifiers=modifiers)
            np.testing.assert_allclose(layers[0].bounding_boxes[0], drawn, atol=ATOL,
                                       err_msg=f"a drag in the read-only XZ view changed the store "
                                               f"(sync_mode={mode}, modifiers={modifiers})")
            _assert_view_shows_store(xz, sub, f"XZ after a rejected drag ({mode}, {modifiers})")
    sub.bounding_boxes = np.array([[[Z1, 30.0, 30.0], [Z2, 70.0, 70.0]]])  # API on a read-only layer
    np.testing.assert_allclose(layers[0].bounding_boxes[0], drawn, atol=ATOL)
