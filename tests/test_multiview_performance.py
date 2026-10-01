"""
Automated tests for MultiViewWidget with synchronized bbox layers.

These tests verify:
1. Performance of multi-viewer synchronization
2. BBox visibility in all 2D slice views (XY, XZ, YZ)
3. BBox visibility in 3D rendering
4. Synchronization accuracy across all views
"""

import time

import napari
import numpy as np
import pytest

from napari_turbobox import (
    TurboBoxLayer,
    create_synchronized_bbox_layers,
)


def spin_qt_events():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app:
        app.processEvents()


class TestMultiViewPerformance:
    """Test performance of synchronized multi-viewer setup."""

    def test_sync_layer_creation_performance(self, make_napari_viewer):
        """Test that creating synchronized layers is fast."""
        main_viewer = make_napari_viewer()
        sub_viewers = [make_napari_viewer() for _ in range(3)]

        start_time = time.time()
        layers = create_synchronized_bbox_layers(
            main_viewer=main_viewer,
            sub_viewers=sub_viewers,
            image_shape=(100, 512, 512),
        )
        creation_time = time.time() - start_time

        assert len(layers) == 4, "Should create 4 layers (1 main + 3 sub)"
        assert creation_time < 5.0, f"Creation took {creation_time:.3f}s, should be < 5s"

        # Cleanup
        for viewer in sub_viewers:
            viewer.close()

    def test_synchronization_performance(self, make_napari_viewer):
        """Test that synchronization is fast when adding boxes."""
        main_viewer = make_napari_viewer()
        sub_viewers = [make_napari_viewer() for _ in range(3)]

        layers = create_synchronized_bbox_layers(
            main_viewer=main_viewer,
            sub_viewers=sub_viewers,
            image_shape=(100, 512, 512),
        )

        # Add single box - should be very fast
        start_time = time.time()
        layers[0].add_boxes([[[10, 100, 100], [20, 200, 200]]])
        sync_time = time.time() - start_time

        assert sync_time < 0.5, f"Single box sync took {sync_time:.3f}s, should be < 0.5s"

        # Verify synchronization
        time.sleep(0.05)  # Small delay for event propagation
        for layer in layers:
            assert len(layer.bounding_boxes) == 1, "All layers should have 1 box"

        # Cleanup
        for viewer in sub_viewers:
            viewer.close()

    def test_bulk_add_performance(self, make_napari_viewer):
        """Test performance when adding many boxes at once."""
        main_viewer = make_napari_viewer()
        sub_viewers = [make_napari_viewer() for _ in range(3)]

        layers = create_synchronized_bbox_layers(
            main_viewer=main_viewer,
            sub_viewers=sub_viewers,
            image_shape=(100, 512, 512),
        )

        # Add 50 boxes at once. z_start wraps within [0, 88] so every box
        # stays in-bounds for the (100, 512, 512) image (z_end <= 93 <= 99).
        boxes = []
        for i in range(50):
            z_start = (i * 2) % 90
            z_end = z_start + 5
            boxes.append([[z_start, 50, 50], [z_end, 100, 100]])

        start_time = time.time()
        layers[0].add_boxes(boxes)
        bulk_add_time = time.time() - start_time

        assert bulk_add_time < 1.0, f"Adding 50 boxes took {bulk_add_time:.3f}s, should be < 1s"

        # Verify synchronization
        time.sleep(0.1)
        for layer in layers:
            assert len(layer.bounding_boxes) == 50, "All layers should have 50 boxes"

        # Cleanup
        for viewer in sub_viewers:
            viewer.close()


class TestMultiView2DSliceVisibility:
    """Test bbox visibility in different 2D slice orientations."""

    def setup_method(self):
        """Create a 3D bbox for testing."""
        # BBox spanning z=10-20, y=100-200, x=150-250
        self.test_bbox = np.array(
            [
                [10.0, 100.0, 150.0],  # mins
                [20.0, 200.0, 250.0],  # maxs
            ]
        )

    def test_xy_slice_visibility(self, make_napari_viewer):
        """Test bbox visibility in XY view (Z-slice navigation)."""
        viewer = make_napari_viewer()

        layer = TurboBoxLayer(ndim=3, image_shape=(100, 512, 512), name="XY Test")
        layer.add_boxes([self.test_bbox])
        viewer.add_layer(layer)

        # Set to 2D view (XY plane)
        viewer.dims.ndisplay = 2

        # Test different Z positions
        test_cases = [
            (5, False, "z=5 (before bbox)"),
            (10, True, "z=10 (start boundary)"),
            (15, True, "z=15 (middle)"),
            (20, True, "z=20 (end boundary)"),
            (25, False, "z=25 (after bbox)"),
        ]

        for z_pos, should_be_visible, description in test_cases:
            viewer.dims.set_point(0, z_pos)  # Set Z position
            spin_qt_events()
            time.sleep(0.05)  # Allow render update
            spin_qt_events()

            # Check if shapes are visible
            num_visible_shapes = len(layer.data)
            is_visible = num_visible_shapes > 0

            assert is_visible == should_be_visible, (
                f"XY view {description}: expected visible={should_be_visible}, got {is_visible}"
            )

    def test_xz_slice_visibility(self, make_napari_viewer):
        """Test bbox visibility in XZ view (Y-slice navigation)."""
        viewer = make_napari_viewer()

        layer = TurboBoxLayer(ndim=3, image_shape=(100, 512, 512), name="XZ Test")
        layer.add_boxes([self.test_bbox])
        viewer.add_layer(layer)

        # Set to 2D view with Z,X displayed (Y slicing)
        viewer.dims.ndisplay = 2
        viewer.dims.order = (1, 0, 2)  # Y, Z, X

        # Test different Y positions
        test_cases = [
            (50, False, "y=50 (before bbox)"),
            (100, True, "y=100 (start boundary)"),
            (150, True, "y=150 (middle)"),
            (200, True, "y=200 (end boundary)"),
            (250, False, "y=250 (after bbox)"),
        ]

        for y_pos, should_be_visible, description in test_cases:
            viewer.dims.set_point(1, y_pos)  # Set Y position
            spin_qt_events()
            time.sleep(0.05)
            spin_qt_events()

            num_visible_shapes = len(layer.data)
            is_visible = num_visible_shapes > 0

            assert is_visible == should_be_visible, (
                f"XZ view {description}: expected visible={should_be_visible}, got {is_visible}"
            )

    def test_yz_slice_visibility(self, make_napari_viewer):
        """Test bbox visibility in YZ view (X-slice navigation)."""
        viewer = make_napari_viewer()

        layer = TurboBoxLayer(ndim=3, image_shape=(100, 512, 512), name="YZ Test")
        layer.add_boxes([self.test_bbox])
        viewer.add_layer(layer)

        # Set to 2D view with Z,Y displayed (X slicing)
        viewer.dims.ndisplay = 2
        viewer.dims.order = (2, 0, 1)  # X, Z, Y

        # Test different X positions
        test_cases = [
            (100, False, "x=100 (before bbox)"),
            (150, True, "x=150 (start boundary)"),
            (200, True, "x=200 (middle)"),
            (250, True, "x=250 (end boundary)"),
            (300, False, "x=300 (after bbox)"),
        ]

        for x_pos, should_be_visible, description in test_cases:
            viewer.dims.set_point(2, x_pos)  # Set X position
            spin_qt_events()
            time.sleep(0.05)
            spin_qt_events()

            num_visible_shapes = len(layer.data)
            is_visible = num_visible_shapes > 0

            assert is_visible == should_be_visible, (
                f"YZ view {description}: expected visible={should_be_visible}, got {is_visible}"
            )


class TestMultiView3DVisibility:
    """Test bbox visibility in 3D rendering."""

    def test_bbox_visible_in_3d(self, make_napari_viewer):
        """Test that bbox is visible in 3D view."""
        viewer = make_napari_viewer()

        # Create bbox
        test_bbox = np.array(
            [
                [10.0, 100.0, 150.0],
                [20.0, 200.0, 250.0],
            ]
        )

        layer = TurboBoxLayer(ndim=3, image_shape=(100, 512, 512), name="3D Test")
        layer.add_boxes([test_bbox])
        viewer.add_layer(layer)

        # Switch to 3D view
        viewer.dims.ndisplay = 3
        time.sleep(0.1)  # Allow 3D render to update

        # In 3D mode a bbox renders as a single wireframe path covering all
        # 12 edges (one napari shape per box).
        num_shapes = len(layer.data)

        assert num_shapes > 0, "BBox should be visible in 3D view"
        assert num_shapes == 1, f"3D bbox should be 1 wireframe path, got {num_shapes}"

    def test_multiple_bboxes_in_3d(self, make_napari_viewer):
        """Test that multiple bboxes are all visible in 3D."""
        viewer = make_napari_viewer()

        # Create 3 bboxes at different locations
        bboxes = [
            [[10, 100, 100], [20, 150, 150]],
            [[30, 200, 200], [40, 250, 250]],
            [[50, 300, 300], [60, 350, 350]],
        ]

        layer = TurboBoxLayer(ndim=3, image_shape=(100, 512, 512), name="Multiple 3D Test")
        layer.add_boxes(bboxes)
        viewer.add_layer(layer)

        # Switch to 3D
        viewer.dims.ndisplay = 3
        time.sleep(0.1)

        # Each bbox renders as one wireframe path -> 3 boxes = 3 shapes.
        num_shapes = len(layer.data)

        assert num_shapes == 3, f"3 bboxes should be 3 wireframe paths, got {num_shapes}"


class TestMultiViewSynchronizationAccuracy:
    """Test synchronization accuracy across multiple views."""

    def test_all_views_show_same_boxes(self, make_napari_viewer):
        """Test that all synchronized views show the same boxes."""
        main_viewer = make_napari_viewer()
        sub_viewers = [make_napari_viewer() for _ in range(3)]

        # Create synchronized layers
        layers = create_synchronized_bbox_layers(
            main_viewer=main_viewer,
            sub_viewers=sub_viewers,
            image_shape=(100, 512, 512),
        )

        # Add boxes
        test_boxes = [
            [[10, 100, 100], [20, 200, 200]],
            [[30, 150, 150], [40, 250, 250]],
        ]
        layers[0].add_boxes(test_boxes)
        time.sleep(0.1)

        # All layers should have identical bbox data
        main_boxes = layers[0].bounding_boxes
        for i, layer in enumerate(layers[1:], 1):
            layer_boxes = layer.bounding_boxes
            assert np.array_equal(main_boxes, layer_boxes), (
                f"Layer {i} boxes don't match main layer"
            )

        # Cleanup
        for viewer in sub_viewers:
            viewer.close()

    def test_boxes_visible_at_correct_slices(self, make_napari_viewer):
        """Test that boxes appear at correct slice positions in all views."""
        main_viewer = make_napari_viewer()
        xy_viewer = make_napari_viewer()
        xz_viewer = make_napari_viewer()
        yz_viewer = make_napari_viewer()

        # Create synchronized layers
        layers = create_synchronized_bbox_layers(
            main_viewer=main_viewer,
            sub_viewers=[xy_viewer, xz_viewer, yz_viewer],
            image_shape=(100, 512, 512),
        )

        # Add a bbox at known position
        bbox = np.array([[[15, 125, 175], [25, 225, 275]]])
        layers[0].add_boxes(bbox)
        time.sleep(0.1)

        # Configure each viewer for different slice orientations
        # XY view (default)
        # Test XY view visible position
        xy_viewer.dims.set_point(0, 15)  # Z=15 (should be visible)
        spin_qt_events()
        time.sleep(0.05)
        spin_qt_events()
        assert len(layers[1].data) > 0, "BBox should be visible in XY view at z=15"

        # XZ view
        xz_viewer.dims.ndisplay = 2
        xz_viewer.dims.order = (1, 0, 2)  # Y slicing
        xz_viewer.dims.set_point(1, 150)  # Y=150 (should be visible)
        spin_qt_events()
        time.sleep(0.05)
        spin_qt_events()
        assert len(layers[2].data) > 0, "BBox should be visible in XZ view at y=150"

        # YZ view
        yz_viewer.dims.ndisplay = 2
        yz_viewer.dims.order = (2, 0, 1)  # X slicing
        yz_viewer.dims.set_point(2, 200)  # X=200 (should be visible)
        spin_qt_events()
        time.sleep(0.05)
        spin_qt_events()
        assert len(layers[3].data) > 0, "BBox should be visible in YZ view at x=200"

        # Test invisible positions
        xy_viewer.dims.set_point(0, 5)  # Z=5 (before bbox)
        spin_qt_events()
        time.sleep(0.05)
        spin_qt_events()
        assert len(layers[1].data) == 0, "BBox should NOT be visible in XY view at z=5"

        # Cleanup
        xy_viewer.close()
        xz_viewer.close()
        yz_viewer.close()

    def test_interactive_vs_readonly_synchronization(self, make_napari_viewer):
        """Test that only interactive layer can modify data."""
        main_viewer = make_napari_viewer()
        sub_viewers = [make_napari_viewer() for _ in range(2)]

        layers = create_synchronized_bbox_layers(
            main_viewer=main_viewer,
            sub_viewers=sub_viewers,
            image_shape=(100, 512, 512),
        )

        # Main layer should be interactive
        assert layers[0]._interactive is True, "Main layer should be interactive"

        # Sub layers should be read-only
        assert layers[1]._interactive is False, "Sub layer 1 should be read-only"
        assert layers[2]._interactive is False, "Sub layer 2 should be read-only"

        # Add box via main layer
        layers[0].add_boxes([[[10, 100, 100], [20, 200, 200]]])
        time.sleep(0.1)

        # All should have 1 box
        for layer in layers:
            assert len(layer.bounding_boxes) == 1

        # Try to add via read-only layer (should be ignored)
        layers[1].add_boxes([[[30, 100, 100], [40, 200, 200]]])
        time.sleep(0.1)

        # Still should have only 1 box
        for layer in layers:
            assert len(layer.bounding_boxes) == 1, "Read-only layer modification should be ignored"

        # Cleanup
        for viewer in sub_viewers:
            viewer.close()


class TestMultiViewStressTest:
    """Stress tests for multi-viewer synchronization."""

    def test_rapid_updates(self, make_napari_viewer):
        """Test rapid successive updates to bbox data."""
        main_viewer = make_napari_viewer()
        sub_viewers = [make_napari_viewer() for _ in range(2)]

        layers = create_synchronized_bbox_layers(
            main_viewer=main_viewer,
            sub_viewers=sub_viewers,
            image_shape=(100, 512, 512),
        )

        # Rapidly add boxes
        start_time = time.time()
        for i in range(20):
            layers[0].add_boxes([[[i * 2, 100, 100], [i * 2 + 5, 150, 150]]])

        update_time = time.time() - start_time

        # Give time for all updates to propagate
        time.sleep(0.2)

        # All layers should have 20 boxes
        for layer in layers:
            assert len(layer.bounding_boxes) == 20, (
                f"Expected 20 boxes, got {len(layer.bounding_boxes)}"
            )

        assert update_time < 2.0, f"20 rapid updates took {update_time:.3f}s, should be < 2s"

        # Cleanup
        for viewer in sub_viewers:
            viewer.close()

    def test_many_viewers(self, make_napari_viewer):
        """Test synchronization with many viewers."""
        main_viewer = make_napari_viewer()
        # Create 10 sub-viewers
        sub_viewers = [make_napari_viewer() for _ in range(10)]

        start_time = time.time()
        layers = create_synchronized_bbox_layers(
            main_viewer=main_viewer,
            sub_viewers=sub_viewers,
            image_shape=(100, 512, 512),
        )
        creation_time = time.time() - start_time

        assert len(layers) == 11, "Should create 11 layers (1 main + 10 sub)"
        assert creation_time < 10.0, f"Creating 11 layers took {creation_time:.3f}s"

        # Add box and verify all are synced
        layers[0].add_boxes([[[10, 100, 100], [20, 200, 200]]])
        time.sleep(0.2)

        for i, layer in enumerate(layers):
            assert len(layer.bounding_boxes) == 1, f"Layer {i} not synchronized"

        # Cleanup
        for viewer in sub_viewers:
            viewer.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
