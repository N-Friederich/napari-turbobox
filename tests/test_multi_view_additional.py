from unittest.mock import MagicMock

import numpy as np
import pytest

from napari_turbobox.layer import TurboBoxLayer
from napari_turbobox.multi_view import (
    create_synchronized_bbox_layers,
    sync_existing_layer_to_viewers,
)


@pytest.fixture
def mock_viewer():
    viewer = MagicMock()
    # Mock add_layer so it accepts TurboBoxLayer
    viewer.add_layer = MagicMock()
    return viewer


def test_create_synchronized_bbox_layers_snapshot(mock_viewer):
    main_viewer = mock_viewer
    sub_viewer = MagicMock()
    sub_viewer.add_layer = MagicMock()

    # Snapshot mode with some initial data
    initial_data = np.array([[[0, 0, 0], [10, 10, 10]]])

    layers = create_synchronized_bbox_layers(
        main_viewer=main_viewer,
        sub_viewers=[sub_viewer],
        bbox_data=initial_data,
        sync_mode="snapshot",
        name="SnapshotBoxes",
    )

    assert len(layers) == 2
    main_layer, sub_layer = layers

    # In snapshot mode, the layers should NOT share the same data store
    assert main_layer._bbox_store is not sub_layer._bbox_store

    # But they should have the same initial boxes
    assert np.array_equal(main_layer._bbox_store.data, initial_data)
    assert np.array_equal(sub_layer._bbox_store.data, initial_data)


def test_sync_existing_layer_to_viewers(mock_viewer):
    source_layer = TurboBoxLayer(ndim=3, edge_width=2.5, edge_color="red", name="ExistingSource")

    target_viewer = MagicMock()
    target_viewer.add_layer = MagicMock()

    # Call deprecated function, should emit DeprecationWarning
    with pytest.warns(DeprecationWarning, match="sync_existing_layer_to_viewers is deprecated"):
        synced_layers = sync_existing_layer_to_viewers(
            source_layer=source_layer, target_viewers=[target_viewer]
        )

    assert len(synced_layers) == 1
    synced_layer = synced_layers[0]

    # Shared store verify
    assert synced_layer._bbox_store is source_layer._bbox_store
    assert synced_layer.name == "ExistingSource (Sync 1)"
    # Clean check of color and width
    assert np.array_equal(synced_layer.edge_width, source_layer.edge_width)
