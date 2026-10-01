from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from napari_turbobox.data_store import BBoxDataStore
from napari_turbobox.layer import TurboBoxLayer
from napari_turbobox.widget import BoundingBoxControlWidget


@pytest.fixture
def mock_viewer():
    viewer = MagicMock()
    viewer.add_layer = MagicMock()
    return viewer


def test_layer_init_with_shared_store_and_data():
    """Verify that if layer is initialized with both a shared store and initial data,

    the initial data is successfully added to the shared store (triggers line 119).
    """
    store = BBoxDataStore()
    assert store.data.size == 0

    initial_boxes = np.array([[[0, 0], [10, 10]]], dtype=float)

    # Initialize TurboBoxLayer with the store and initial boxes
    layer = TurboBoxLayer(bbox_data_store=store, data=initial_boxes, ndim=2)

    assert layer._bbox_store is store
    assert layer._owns_store is False
    # Check that initial boxes were added to store
    assert np.array_equal(store.data, initial_boxes)


def test_on_parent_change():
    """Verify _on_parent_change call triggers synchronization."""
    layer = TurboBoxLayer(ndim=3)
    layer._sync_guard = False

    # Add box to store
    layer.add_boxes([[[0, 0, 0], [10, 10, 10]]])

    # Mock _sync_shapes_from_bboxes
    layer._sync_shapes_from_bboxes = MagicMock()

    # Call _on_parent_change
    layer._on_parent_change(None)

    # Assert synchronization was triggered
    layer._sync_shapes_from_bboxes.assert_called_once()


def test_on_shapes_data_changed_with_sync_guard():
    """Verify that if _sync_guard is active, editing shapes does not cause synchronization feedback loops."""
    layer = TurboBoxLayer(ndim=2)
    layer._sync_guard = True
    layer._bbox_store.data = np.empty((0, 2, 2))

    # Call _on_shapes_data_changed
    layer._on_shapes_data_changed(None)

    # The store's data should remain empty
    assert layer._bbox_store.data.size == 0


@pytest.mark.skip(reason="Qt widgets cause segfault in headless environment")
def test_widget_layer_settings_routing(mock_viewer):
    """Verify that BoundingBoxControlWidget routes settings correctly to the active layer."""
    layer = TurboBoxLayer(ndim=3, name="BBoxes")

    class MockSelection(set):
        pass

    class MockLayersList(list):
        pass

    selection = MockSelection({layer})
    selection.active = layer

    layers = MockLayersList([layer])
    layers.selection = selection
    mock_viewer.layers = layers

    # Instantiate control widget
    widget = BoundingBoxControlWidget(mock_viewer)

    # Test setting cache limit
    assert layer.cache_limit == 10000  # Default
    widget._on_cache_limit_changed(50)
    assert layer.cache_limit == 50

    # Test clearing history
    layer.clear_undo_history = MagicMock()
    widget._on_clear_history_clicked()
    layer.clear_undo_history.assert_called_once()
