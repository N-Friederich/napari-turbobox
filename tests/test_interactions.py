"""
Tests for OptimizedBoundingBoxLayer interactions.

These tests simulate user interactions like clicking, dragging, drawing, etc.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from napari.layers.shapes._shape_list import ShapeList
from napari.layers.shapes._shapes_constants import Mode


@pytest.fixture
def bbox_layer():
    """Create a basic bbox layer for testing."""
    from napari_turbobox.layer import TurboBoxLayer

    layer = TurboBoxLayer(ndim=3, image_shape=(10, 100, 100), edge_width=2.0, edge_color="cyan")

    # Mock _slice_input so _current_paths() works
    # Use MagicMock to avoid Mock comparison issues
    slice_input_mock = MagicMock()
    slice_input_mock.ndisplay = 2
    slice_input_mock.displayed = [1, 2]  # Y, X axes visible
    slice_input_mock.not_displayed = [0]  # Z axis hidden
    # Use numpy array so indexing works properly
    slice_input_mock.point = np.array([4.0, 49.0, 49.0])  # Direct point access

    # Mock world_slice with the correct nested structure
    world_slice_mock = MagicMock()
    world_slice_mock.point = np.array([4.0, 49.0, 49.0])  # Current slice position
    slice_input_mock.world_slice = world_slice_mock

    # Mock data_slice method which is called to compute _data_slice property
    data_slice_result = MagicMock()
    data_slice_result.point = np.array([4.0, 49.0, 49.0])
    slice_input_mock.data_slice = MagicMock(return_value=data_slice_result)

    layer._slice_input = slice_input_mock

    # Use actual ShapeList for _data_view to support all napari operations
    layer._data_view = ShapeList(ndisplay=2)

    # Set _ndisplay_stored to avoid Mock comparison issues
    layer._ndisplay_stored = 2

    # Patch problematic napari methods to avoid refresh issues in tests
    # These methods try to access viewer/canvas which don't exist in unit tests

    def patched_set_view_slice():
        # Only call our custom logic, skip parent's problematic refresh
        if hasattr(layer, "_sync_guard") and hasattr(layer, "_bbox_store"):
            if not layer._sync_guard and layer._bbox_store.data.size > 0:
                layer._sync_shapes_from_bboxes()
        # Don't call super() to avoid Mock/refresh issues in napari's code

    def patched_update_dims():
        # Skip _update_dims which triggers problematic refresh chain
        pass

    layer._set_view_slice = patched_set_view_slice
    layer._update_dims = patched_update_dims

    return layer


@pytest.fixture
def bbox_layer_with_boxes(bbox_layer):
    """Create a bbox layer with some test boxes."""
    boxes = np.array(
        [
            [[2, 20, 20], [5, 50, 50]],  # Box 0
            [[3, 60, 60], [6, 80, 80]],  # Box 1
        ]
    )
    bbox_layer.add_boxes(boxes)
    return bbox_layer


def _drive_drag(layer, start_pos, move_positions, modifiers=None):
    """Drive a real press -> move(s) -> release drag through ``_on_mouse_drag``.

    ``_on_mouse_drag`` is a generator that captures ``event`` ONCE; its ``yield``
    statements ignore the value passed to ``gen.send(...)``. So a drag must be
    driven by mutating a SINGLE shared event object in place and calling
    ``next(gen)`` -- creating a fresh event per step (or ``send``-ing one) leaves
    the move loop reading the stale press event and never executes its body
    (the classic false-pass these tests previously had).
    """
    event = Mock()
    event.type = "mouse_press"
    event.position = np.asarray(start_pos, dtype=float)
    event.modifiers = list(modifiers or [])
    event.dims_displayed = [0, 1, 2]
    event.handled = False
    gen = layer._on_mouse_drag(layer, event)
    next(gen)  # run the press branch
    for pos in move_positions:
        event.type = "mouse_move"
        event.position = np.asarray(pos, dtype=float)
        next(gen)  # run ONE move-loop iteration (clamp + update_box_silent)
    event.type = "mouse_release"
    try:
        next(gen)
    except StopIteration:
        pass
    return event


class TestLayerCreation:
    """Test basic layer creation and initialization."""

    def test_create_layer_basic(self):
        """Test creating a layer with minimal parameters."""
        from napari_turbobox.layer import TurboBoxLayer

        layer = TurboBoxLayer(ndim=3)

        assert layer.ndim == 3
        assert layer.nshapes == 0
        assert layer.mode == Mode.SELECT

    def test_create_layer_with_image_shape(self):
        """Test creating a layer with image bounds."""
        from napari_turbobox.layer import TurboBoxLayer

        image_shape = (10, 100, 100)
        layer = TurboBoxLayer(ndim=3, image_shape=image_shape)

        assert np.array_equal(layer._image_shape, image_shape)

    def test_create_2d_layer(self):
        """Test creating a 2D layer."""
        from napari_turbobox.layer import TurboBoxLayer

        layer = TurboBoxLayer(ndim=2)

        assert layer.ndim == 2
        assert layer.nshapes == 0


class TestAddingBBoxes:
    """Test adding bounding boxes to the layer."""

    def test_add_single_bbox(self, bbox_layer):
        """Test adding a single bounding box."""
        boxes = np.array([[[2, 20, 20], [5, 50, 50]]])

        bbox_layer.add_boxes(boxes)

        assert bbox_layer.nshapes == 1
        assert np.array_equal(bbox_layer.bounding_boxes, boxes)

    def test_add_multiple_bboxes(self, bbox_layer):
        """Test adding multiple bounding boxes at once."""
        boxes = np.array(
            [
                [[2, 20, 20], [5, 50, 50]],
                [[3, 60, 60], [6, 80, 80]],
            ]
        )

        bbox_layer.add_boxes(boxes)

        assert bbox_layer.nshapes == 2
        assert np.array_equal(bbox_layer.bounding_boxes, boxes)

    def test_add_bbox_out_of_bounds_raises(self, bbox_layer):
        """Out-of-bounds boxes are rejected by the hard box invariant.

        The API layer no longer silently clamps — invalid input raises.
        """
        # Box extends beyond image bounds (10, 100, 100).
        boxes = np.array([[[2, 20, 20], [15, 150, 150]]])

        with pytest.raises(ValueError):
            bbox_layer.add_boxes(boxes)


class TestHitTesting:
    """Test hit testing (clicking on bboxes)."""

    def test_hit_inside_bbox(self, bbox_layer_with_boxes):
        """Test clicking inside a bbox."""
        # Click in the center of box 0: [[2, 20, 20], [5, 50, 50]]
        position = np.array([3.5, 35, 35])

        bbox_idx, is_near_edge = bbox_layer_with_boxes._hit_test_bbox(position)

        assert bbox_idx == 0
        assert not is_near_edge

    def test_hit_near_edge(self, bbox_layer_with_boxes):
        """Test clicking near the edge of a bbox."""
        # Click near the edge of box 0
        position = np.array([2.1, 20.5, 35])  # Near Z-min edge

        bbox_idx, is_near_edge = bbox_layer_with_boxes._hit_test_bbox(position)

        assert bbox_idx == 0
        assert is_near_edge

    def test_hit_outside_bboxes(self, bbox_layer_with_boxes):
        """Test clicking outside all bboxes."""
        position = np.array([1, 10, 10])  # Outside all boxes

        bbox_idx, is_near_edge = bbox_layer_with_boxes._hit_test_bbox(position)

        assert bbox_idx is None
        assert not is_near_edge

    def test_hit_second_bbox(self, bbox_layer_with_boxes):
        """Test clicking on the second bbox."""
        # Click in box 1: [[3, 60, 60], [6, 80, 80]]
        position = np.array([4.5, 70, 70])

        bbox_idx, _is_near_edge = bbox_layer_with_boxes._hit_test_bbox(position)

        assert bbox_idx == 1


class TestMouseDragSimulation:
    """Test simulating mouse drag events."""

    def test_mouse_press_on_bbox(self, bbox_layer_with_boxes):
        """Test mouse press event on a bbox."""
        # Create a mock event
        event = Mock()
        event.type = "mouse_press"
        event.position = np.array([3.5, 35, 35])  # Inside box 0
        event.modifiers = []
        event.dims_displayed = [0, 1, 2]
        event.handled = False

        # Call the drag handler (returns a generator)
        gen = bbox_layer_with_boxes._on_mouse_drag(bbox_layer_with_boxes, event)

        # Consume the generator to execute the code until the first yield
        try:
            next(gen)
        except StopIteration:
            pass  # Generator finished

        # The generator should have initialized drag state
        assert hasattr(bbox_layer_with_boxes, "_drag_state")
        assert bbox_layer_with_boxes._drag_state is not None
        assert bbox_layer_with_boxes._drag_state["index"] == 0
        assert bbox_layer_with_boxes._drag_state["mode"] == "translate"

    def test_mouse_press_outside_bbox(self, bbox_layer_with_boxes):
        """Test mouse press event outside any bbox."""
        event = Mock()
        event.type = "mouse_press"
        event.position = np.array([1, 10, 10])  # Outside all boxes
        event.modifiers = []
        event.handled = False

        # Call the drag handler (returns a generator)
        gen = bbox_layer_with_boxes._on_mouse_drag(bbox_layer_with_boxes, event)

        # Consume the generator to execute the code until the first yield
        try:
            next(gen)
        except StopIteration:
            pass  # Generator finished

        # Should not create drag state
        assert bbox_layer_with_boxes._drag_state is None

    def test_resize_mode_with_shift(self, bbox_layer_with_boxes):
        """Test that Shift key triggers resize mode."""
        event = Mock()
        event.type = "mouse_press"
        event.position = np.array([3.5, 35, 35])  # Inside box 0
        event.modifiers = ["Shift"]
        event.dims_displayed = [0, 1, 2]
        event.handled = False

        # Call the drag handler (returns a generator)
        gen = bbox_layer_with_boxes._on_mouse_drag(bbox_layer_with_boxes, event)

        # Consume the generator to execute the code until the first yield
        try:
            next(gen)
        except StopIteration:
            pass  # Generator finished

        assert bbox_layer_with_boxes._drag_state["mode"] == "resize"


class TestRectangleDrawing:
    """Test rectangle drawing mode."""

    def test_add_rectangle_mode_allowed(self, bbox_layer):
        """Test that ADD_RECTANGLE mode is allowed."""
        bbox_layer.mode = Mode.ADD_RECTANGLE

        # Create a mock event
        event = Mock()
        event.value = Mode.ADD_RECTANGLE

        # Should not force back to SELECT
        bbox_layer._on_mode_change(event)

        assert bbox_layer.mode == Mode.ADD_RECTANGLE

    def test_add_ellipse_mode_blocked(self, bbox_layer):
        """Test that ADD_ELLIPSE mode is blocked."""
        bbox_layer.mode = Mode.ADD_ELLIPSE

        event = Mock()
        event.value = Mode.ADD_ELLIPSE

        bbox_layer._on_mode_change(event)

        # Should be forced back to SELECT
        assert bbox_layer.mode == Mode.SELECT

    def test_transform_mode_blocked(self, bbox_layer):
        """Test that TRANSFORM mode is blocked."""
        bbox_layer.mode = Mode.TRANSFORM

        event = Mock()
        event.value = Mode.TRANSFORM

        bbox_layer._on_mode_change(event)

        # Should be forced back to SELECT
        assert bbox_layer.mode == Mode.SELECT


class TestBBoxManipulation:
    """Test moving and resizing bboxes."""

    def test_move_bbox_property(self, bbox_layer_with_boxes):
        """Test moving a bbox by modifying the bounding_boxes property."""
        original = bbox_layer_with_boxes.bounding_boxes.copy()

        # Move box 0 by (1, 5, 5)
        new_boxes = bbox_layer_with_boxes.bounding_boxes.copy()
        new_boxes[0] += np.array([[1, 5, 5], [1, 5, 5]])
        bbox_layer_with_boxes.bounding_boxes = new_boxes

        result = bbox_layer_with_boxes.bounding_boxes
        expected_mins = original[0, 0] + np.array([1, 5, 5])
        expected_maxs = original[0, 1] + np.array([1, 5, 5])

        assert np.array_equal(result[0, 0], expected_mins)
        assert np.array_equal(result[0, 1], expected_maxs)

    def test_resize_bbox(self, bbox_layer_with_boxes):
        """Test resizing a bbox."""
        original = bbox_layer_with_boxes.bounding_boxes.copy()

        # Make box 0 larger
        new_boxes = bbox_layer_with_boxes.bounding_boxes.copy()
        new_boxes[0, 1] += np.array([2, 10, 10])  # Increase max corner
        bbox_layer_with_boxes.bounding_boxes = new_boxes

        result = bbox_layer_with_boxes.bounding_boxes
        expected_maxs = original[0, 1] + np.array([2, 10, 10])

        assert np.array_equal(result[0, 1], expected_maxs)

    def test_delete_bbox(self, bbox_layer_with_boxes):
        """Test deleting a bbox."""
        assert bbox_layer_with_boxes.nshapes == 2

        # Delete box 0
        boxes = bbox_layer_with_boxes.bounding_boxes
        new_boxes = np.delete(boxes, 0, axis=0)
        bbox_layer_with_boxes.bounding_boxes = new_boxes

        assert bbox_layer_with_boxes.nshapes == 1

    def test_clear_all_bboxes(self, bbox_layer_with_boxes):
        """Test clearing all bboxes."""
        bbox_layer_with_boxes.clear_boxes()

        assert bbox_layer_with_boxes.nshapes == 0


class TestPanZoomMode:
    """Test that pan/zoom mode doesn't interfere with bbox interaction."""

    def test_pan_zoom_mode_skips_drag_handler(self, bbox_layer_with_boxes):
        """Test that PAN_ZOOM mode skips our drag handler."""
        bbox_layer_with_boxes.mode = Mode.PAN_ZOOM

        event = Mock()
        event.type = "mouse_press"
        event.position = np.array([3.5, 35, 35])  # Inside box 0
        event.modifiers = []

        # Call the drag handler
        result = bbox_layer_with_boxes._on_mouse_drag(bbox_layer_with_boxes, event)

        # In PAN_ZOOM mode, function returns before yield
        # But Python still makes it a generator. Test that it immediately ends.
        if result is not None:
            # If it's a generator, it should immediately raise StopIteration
            with pytest.raises(StopIteration):
                next(result)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_add_empty_box_array(self, bbox_layer):
        """Test adding an empty array of boxes."""
        boxes = np.empty((0, 2, 3))

        bbox_layer.add_boxes(boxes)

        assert bbox_layer.nshapes == 0

    def test_wrong_ndim_bbox(self, bbox_layer):
        """Test adding a bbox with wrong dimensionality."""
        # 2D box for 3D layer
        boxes = np.array([[[20, 20], [50, 50]]])

        with pytest.raises((ValueError, AssertionError)):
            bbox_layer.add_boxes(boxes)

    def test_inverted_bbox_coordinates(self, bbox_layer):
        """Test that inverted coordinates (max < min) are corrected."""
        # Box with min > max (should be auto-corrected)
        boxes = np.array([[[5, 50, 50], [2, 20, 20]]])

        bbox_layer.add_boxes(boxes)

        result = bbox_layer.bounding_boxes[0]
        # Should be corrected so min < max
        assert np.all(result[0] <= result[1])


class TestSliceNavigation:
    """Test slice navigation in 3D - BBox visibility changes (Bug A validation)."""

    def test_bbox_visible_at_slice(self, bbox_layer_with_boxes):
        """Test that bbox is visible when slice is within its Z range."""
        # Box 0: Z=2 to Z=5, should be visible at Z=4
        bbox_layer_with_boxes._slice_input.world_slice.point = np.array([4.0, 49.0, 49.0])

        # Regenerate paths for new slice
        bbox_layer_with_boxes._sync_shapes_from_bboxes()

        # Should have 2 paths (both boxes visible at Z=4)
        assert bbox_layer_with_boxes.nshapes == 2

    def test_bbox_invisible_outside_slice(self, bbox_layer_with_boxes):
        """Test that bbox disappears when slice is outside its Z range."""
        # Box 0: Z=2 to Z=5, should be invisible at Z=7
        bbox_layer_with_boxes._slice_input.world_slice.point = np.array([7.0, 49.0, 49.0])

        # Regenerate paths for new slice
        bbox_layer_with_boxes._sync_shapes_from_bboxes()

        # Box 0 should be invisible, only box 1 visible (Z=3 to Z=6)
        assert bbox_layer_with_boxes.nshapes == 0  # Both boxes out of range at Z=7

    def test_bbox_reappears_on_slice_change(self, bbox_layer_with_boxes):
        """Test that bbox reappears when scrolling back into its range."""
        # Start at Z=7 (box invisible)
        bbox_layer_with_boxes._slice_input.world_slice.point = np.array([7.0, 49.0, 49.0])
        bbox_layer_with_boxes._sync_shapes_from_bboxes()
        initial_shapes = bbox_layer_with_boxes.nshapes

        # Scroll back to Z=4 (box visible again)
        bbox_layer_with_boxes._slice_input.world_slice.point = np.array([4.0, 49.0, 49.0])
        bbox_layer_with_boxes._sync_shapes_from_bboxes()

        # Should have more shapes now
        assert bbox_layer_with_boxes.nshapes > initial_shapes

    def test_slice_at_bbox_boundary(self, bbox_layer_with_boxes):
        """Test bbox visibility exactly at min/max boundaries."""
        # Box 0: Z=2 to Z=5
        # Test at exact min boundary
        bbox_layer_with_boxes._slice_input.world_slice.point = np.array([2.0, 49.0, 49.0])
        bbox_layer_with_boxes._sync_shapes_from_bboxes()
        shapes_at_min = bbox_layer_with_boxes.nshapes

        # Test at exact max boundary
        bbox_layer_with_boxes._slice_input.world_slice.point = np.array([5.0, 49.0, 49.0])
        bbox_layer_with_boxes._sync_shapes_from_bboxes()
        shapes_at_max = bbox_layer_with_boxes.nshapes

        # Should be visible at both boundaries
        assert shapes_at_min > 0
        assert shapes_at_max > 0


class TestCompleteDragSequence:
    """Test complete drag sequences: press → move → release (Bug B validation)."""

    def test_translate_drag_sequence(self, bbox_layer_with_boxes):
        """Test that translate drag initializes correctly."""
        # Press event starts translate drag
        press_event = Mock()
        press_event.type = "mouse_press"
        press_event.position = np.array([3.5, 35, 35])
        press_event.modifiers = []
        press_event.dims_displayed = [0, 1, 2]
        press_event.handled = False

        gen = bbox_layer_with_boxes._on_mouse_drag(bbox_layer_with_boxes, press_event)
        try:
            next(gen)
        except StopIteration:
            pass

        # Should have initialized translate drag state
        assert bbox_layer_with_boxes._drag_state is not None
        assert bbox_layer_with_boxes._drag_state["mode"] == "translate"
        assert bbox_layer_with_boxes._drag_state["index"] == 0

        # Verify bbox_is_moving flag is set (prevents redraw during drag)
        assert bbox_layer_with_boxes._bbox_is_moving

    def test_resize_drag_sequence(self, bbox_layer_with_boxes):
        """Test that resize drag initializes correctly with Shift key."""
        # Press with Shift starts resize drag
        press_event = Mock()
        press_event.type = "mouse_press"
        press_event.position = np.array([3.5, 35, 35])
        press_event.modifiers = ["Shift"]
        press_event.dims_displayed = [0, 1, 2]
        press_event.handled = False

        gen = bbox_layer_with_boxes._on_mouse_drag(bbox_layer_with_boxes, press_event)
        try:
            next(gen)
        except StopIteration:
            pass

        # Should have initialized resize drag state
        assert bbox_layer_with_boxes._drag_state is not None
        assert bbox_layer_with_boxes._drag_state["mode"] == "resize"
        assert bbox_layer_with_boxes._drag_state["index"] == 0

        # Verify bbox_is_moving flag is set
        assert bbox_layer_with_boxes._bbox_is_moving

    def test_translate_drag_actually_moves_box(self, bbox_layer_with_boxes):
        """Driving the move loop must translate the box in the store."""
        layer = bbox_layer_with_boxes
        before = layer.bounding_boxes[0].copy()  # [[2,20,20],[5,50,50]]
        _drive_drag(layer, [3.5, 35, 35], [[3.5, 45, 40]])  # +10 in Y, +5 in X
        after = layer.bounding_boxes[0]
        assert not np.array_equal(before, after), "box did not move (move loop never ran)"
        # translate preserves size; min Y/X shifted by the cursor delta
        np.testing.assert_allclose(after[1] - after[0], before[1] - before[0], atol=1e-4)
        assert after[0, 1] > before[0, 1]  # moved +Y
        assert after[0, 2] > before[0, 2]  # moved +X

    def test_drag_state_cleared_on_release(self, bbox_layer_with_boxes):
        """Drag state and the moving flag are cleared after a real release."""
        layer = bbox_layer_with_boxes
        _drive_drag(layer, [3.5, 35, 35], [[3.5, 40, 40]])
        assert layer._drag_state is None
        assert layer._bbox_is_moving is False


class TestBoundaryClamping:
    """Test that bboxes are clamped to image boundaries."""

    def test_add_bbox_outside_bounds_raises(self, bbox_layer):
        """A box exceeding image_shape is rejected with ValueError."""
        # Image shape is (10, 100, 100); this box exceeds Z, Y and X.
        boxes = np.array([[[0, 0, 0], [20, 150, 150]]])

        with pytest.raises(ValueError):
            bbox_layer.add_boxes(boxes)

    def test_drag_bbox_against_wall(self, bbox_layer_with_boxes):
        """Dragging a box far past the Y boundary clamps it to image_shape - 1.

        image_shape is (10, 100, 100) -> valid max index is 99. Box 0 starts at
        [[2,20,20],[5,50,50]]; we translate it ~+265 in Y.
        """
        layer = bbox_layer_with_boxes
        _drive_drag(layer, [3.5, 35, 35], [[3.5, 300, 35]])
        result = layer.bounding_boxes[0]
        assert result[1, 1] <= 99, "max Y not clamped to image_shape - 1"
        assert result[0, 1] >= 0, "min Y below 0"
        assert result[1, 1] - result[0, 1] >= 1, "box collapsed at the wall"

    def test_resize_bbox_against_boundary(self, bbox_layer_with_boxes):
        """Resizing (Shift+drag) a box past the boundary clamps both axes."""
        layer = bbox_layer_with_boxes
        _drive_drag(layer, [3.5, 35, 35], [[3.5, 300, 300]], modifiers=["Shift"])
        result = layer.bounding_boxes[0]
        assert result[1, 1] <= 99
        assert result[1, 2] <= 99
        assert result[1, 1] - result[0, 1] >= 1
        assert result[1, 2] - result[0, 2] >= 1

    def test_bbox_negative_coordinates_raise(self, bbox_layer):
        """A box with negative min coordinates is rejected with ValueError."""
        boxes = np.array([[[-5, -10, -10], [5, 50, 50]]])

        with pytest.raises(ValueError):
            bbox_layer.add_boxes(boxes)


@pytest.mark.skip(reason="Qt widgets cause segfault in headless environment")
class TestWidgetLiveUpdates:
    """Test that widget updates live when bboxes change."""

    def test_widget_shows_correct_count_after_add(self, bbox_layer):
        """Test that widget displays correct bbox count after adding."""
        from napari_turbobox.widget import BoundingBoxControlWidget

        # Create widget with mock viewer
        mock_viewer = Mock()
        mock_viewer.layers = Mock()
        mock_viewer.layers.events = Mock()
        mock_viewer.layers.events.inserted = Mock()
        mock_viewer.layers.events.inserted.connect = Mock()
        mock_viewer.layers.events.removed = Mock()
        mock_viewer.layers.events.removed.connect = Mock()
        mock_viewer.layers.events.changed = Mock()
        mock_viewer.layers.events.changed.connect = Mock()

        widget = BoundingBoxControlWidget(mock_viewer)
        widget._watched_layer = bbox_layer

        # Add boxes and trigger update
        boxes = np.array([[[2, 20, 20], [5, 50, 50]]])
        bbox_layer.add_boxes(boxes)
        widget._update_bbox_list()

        # Widget should show 1 bbox
        assert widget.bbox_list.count() == 1

    def test_widget_updates_after_delete(self, bbox_layer_with_boxes):
        """Test that widget updates when bbox is deleted."""
        from napari_turbobox.widget import BoundingBoxControlWidget

        mock_viewer = Mock()
        mock_viewer.layers = Mock()
        mock_viewer.layers.events = Mock()
        mock_viewer.layers.events.inserted = Mock()
        mock_viewer.layers.events.inserted.connect = Mock()
        mock_viewer.layers.events.removed = Mock()
        mock_viewer.layers.events.removed.connect = Mock()
        mock_viewer.layers.events.changed = Mock()
        mock_viewer.layers.events.changed.connect = Mock()

        widget = BoundingBoxControlWidget(mock_viewer)
        widget._watched_layer = bbox_layer_with_boxes

        initial_count = 2
        widget._update_bbox_list()
        assert widget.bbox_list.count() == initial_count

        # Delete one bbox
        new_boxes = np.delete(bbox_layer_with_boxes.bounding_boxes, 0, axis=0)
        bbox_layer_with_boxes.bounding_boxes = new_boxes
        widget._update_bbox_list()

        # Should have one less
        assert widget.bbox_list.count() == initial_count - 1

    def test_bbox_event_fires_on_change(self, bbox_layer):
        """Test that layer fires bboxes event when modified."""
        event_fired = []

        def on_bbox_change(event):
            event_fired.append(True)

        bbox_layer.events.bboxes.connect(on_bbox_change)

        # Add a bbox
        boxes = np.array([[[2, 20, 20], [5, 50, 50]]])
        bbox_layer.add_boxes(boxes)

        # Event should have fired
        assert len(event_fired) > 0


class TestOverlappingBBoxes:
    """Test scenarios with overlapping bboxes."""

    def test_hit_test_with_overlapping_boxes(self, bbox_layer):
        """Test hit testing when multiple bboxes overlap."""
        # Add two overlapping boxes
        boxes = np.array(
            [
                [[2, 20, 20], [5, 50, 50]],  # Box 0
                [[3, 30, 30], [6, 60, 60]],  # Box 1 - overlaps with Box 0
            ]
        )
        bbox_layer.add_boxes(boxes)

        # Click in overlap region: Z=3.5, Y=35, X=35
        position = np.array([3.5, 35, 35])
        bbox_idx, _is_near_edge = bbox_layer._hit_test_bbox(position)

        # Should hit one of them (first one found)
        assert bbox_idx is not None
        assert bbox_idx in [0, 1]

    def test_multiple_boxes_at_same_slice(self, bbox_layer):
        """Test that multiple boxes can be visible at the same slice."""
        # Add boxes at different XY positions but same Z range
        boxes = np.array(
            [
                [[2, 20, 20], [5, 40, 40]],  # Box 0
                [[2, 60, 60], [5, 80, 80]],  # Box 1 - same Z, different XY
            ]
        )
        bbox_layer.add_boxes(boxes)

        # Both should be visible at Z=3.5
        bbox_layer._slice_input.world_slice.point = np.array([3.5, 49.0, 49.0])
        bbox_layer._sync_shapes_from_bboxes()

        # Should have 2 shapes
        assert bbox_layer.nshapes == 2

    def test_delete_from_overlapping_boxes(self, bbox_layer):
        """Test deleting specific bbox from overlapping set."""
        boxes = np.array(
            [[[2, 20, 20], [5, 50, 50]], [[3, 30, 30], [6, 60, 60]], [[4, 40, 40], [7, 70, 70]]]
        )
        bbox_layer.add_boxes(boxes)

        # Delete middle box
        new_boxes = np.delete(bbox_layer.bounding_boxes, 1, axis=0)
        bbox_layer.bounding_boxes = new_boxes

        assert bbox_layer.nshapes == 2
        # Remaining boxes should be 0 and 2 (original indices)


class TestExtremeCases:
    """Test extreme coordinate and size cases."""

    def test_bbox_at_origin(self, bbox_layer):
        """Test bbox at (0,0,0)."""
        boxes = np.array([[[0, 0, 0], [3, 10, 10]]])
        bbox_layer.add_boxes(boxes)

        result = bbox_layer.bounding_boxes[0]
        assert np.array_equal(result[0], [0, 0, 0])

    def test_bbox_at_or_above_max_raises(self, bbox_layer):
        """A box whose max reaches image_shape is rejected.

        Valid maxes are <= image_shape - 1; max == image_shape is one past
        the last valid index.
        """
        # Image shape is (10, 100, 100); max (10, 100, 100) is out of bounds.
        boxes = np.array([[[6, 90, 90], [10, 100, 100]]])

        with pytest.raises(ValueError):
            bbox_layer.add_boxes(boxes)

    def test_very_small_bbox(self, bbox_layer):
        """Test 1x1x1 bbox."""
        boxes = np.array([[[5, 50, 50], [6, 51, 51]]])
        bbox_layer.add_boxes(boxes)

        result = bbox_layer.bounding_boxes[0]
        # Should maintain size
        diff = result[1] - result[0]
        assert np.all(diff >= 1)

    def test_bbox_spanning_image_extent_raises(self, bbox_layer):
        """A box whose max equals image_shape (one past bounds) is rejected.

        A box that genuinely spans the image must stop at image_shape - 1.
        """
        boxes = np.array([[[0, 0, 0], [10, 100, 100]]])

        with pytest.raises(ValueError):
            bbox_layer.add_boxes(boxes)

    def test_many_bboxes_performance(self, bbox_layer):
        """Test adding many bboxes at once."""
        # Create 50 bboxes
        boxes = []
        for i in range(50):
            z = i % 8  # Cycle through Z
            y = (i * 10) % 80
            x = (i * 10) % 80
            boxes.append([[z, y, x], [z + 2, y + 10, x + 10]])

        boxes = np.array(boxes)
        bbox_layer.add_boxes(boxes)

        # Should have all 50
        assert len(bbox_layer.bounding_boxes) == 50

    def test_bbox_with_zero_width_on_displayed_axis_raises(self, bbox_layer):
        """A box flat on a displayed axis violates the min-size-1 invariant.

        Y (axis 1) is displayed in the bbox_layer fixture, so it is not
        auto-expanded by _expand_hidden_axes; a zero-width Y must therefore
        be rejected with ValueError (flat-on-hidden-axis boxes, by contrast,
        are legally expanded — covered in test_geometry.py).
        """
        # Y min == Y max -> zero width on a displayed axis.
        boxes = np.array([[[5, 20, 20], [5, 20, 50]]])

        with pytest.raises(ValueError):
            bbox_layer.add_boxes(boxes)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
