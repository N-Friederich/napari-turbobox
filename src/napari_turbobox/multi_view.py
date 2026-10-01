"""
Show one set of boxes in several napari viewers.

``create_synchronized_bbox_layers`` puts one TurboBoxLayer into each viewer,
all on the same ``BBoxDataStore``, for layouts such as XY/XZ/YZ slice views
plus a 3D view. The layer in the main viewer is editable, the others are
read-only.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import napari
import numpy as np

from .data_store import BBoxDataStore
from .layer import TurboBoxLayer

logger = logging.getLogger(__name__)


def _sync_mode_for_viewer(viewer: napari.Viewer) -> Literal["live", "on_commit"]:
    """Pick a layer's ``sync_mode`` from its viewer's current display mode.

    A viewer showing 3D (``ndisplay == 3``) gets ``"on_commit"``: its layer is
    updated once, when the edit session ends, instead of on every store update
    during a drag. 2D viewers get ``"live"`` so they follow the drag.

    The choice is made once, when the layer is created. No listener is
    attached to ``viewer.dims.events.ndisplay``, so a viewer that later
    switches between 2D and 3D keeps its mode. Set ``layer.sync_mode`` by hand
    in that case. A viewer without ``dims.ndisplay`` gets ``"live"`` and a
    warning is logged.
    """
    try:
        ndisplay = viewer.dims.ndisplay
    except AttributeError:
        logger.warning(
            "Viewer %r exposes no dims.ndisplay; defaulting sync_mode to 'live'", viewer
        )
        return "live"
    return "on_commit" if ndisplay == 3 else "live"


def create_synchronized_bbox_layers(
    main_viewer: napari.Viewer,
    sub_viewers: list[napari.Viewer],
    bbox_data: np.ndarray | None = None,
    image_shape: tuple | None = None,
    sync_mode: Literal["live", "on_commit", "snapshot"] = "live",
    **layer_kwargs,
) -> list[TurboBoxLayer]:
    """
    Create one TurboBoxLayer per viewer, all showing the same boxes.

    The layer in ``main_viewer`` is editable (``interactive=True``). The
    layers in ``sub_viewers`` are read-only (``interactive=False``): they
    cannot be edited with the mouse, and their ``add_boxes``,
    ``bounding_boxes`` setter and ``clear_boxes`` only log a warning. In the
    default layout all layers use one ``BBoxDataStore``, so an edit made
    through the main layer shows up in every viewer. 2D viewers follow a drag
    move by move, 3D viewers are updated when the drag ends.

    Parameters
    ----------
    main_viewer : napari.Viewer
        Viewer that gets the editable layer.
    sub_viewers : list of napari.Viewer
        Viewers that get read-only layers.
    bbox_data : np.ndarray, optional
        Initial boxes, shape (N, 2, D).
    image_shape : tuple, optional
        Shape of the image volume. Layer API writes outside it are rejected,
        and mouse edits are clamped to it.
    sync_mode : {"live", "on_commit", "snapshot"}, optional
        Layout of the created layers:

        - "live" (default) or "on_commit": all layers share one
          BBoxDataStore. Each layer's own sync mode comes from its viewer's
          ``ndisplay`` (3D viewers "on_commit", 2D viewers "live"), so the two
          values give the same result.
        - "snapshot": every layer gets its own store and nothing is
          synchronized. The sub-viewer stores start from copies of
          ``bbox_data``.
    **layer_kwargs : dict
        Passed on to TurboBoxLayer (e.g. edge_color, edge_width, name).
        Sub-viewer layers are named ``"<name> (View i)"``.

    Returns
    -------
    list of TurboBoxLayer
        The main viewer's layer first, then one layer per sub-viewer in the
        given order. Each layer has already been added to its viewer.

    Examples
    --------
    Layers for a multi-viewer widget::

        bbox_layers = create_synchronized_bbox_layers(
            main_viewer=viewer,
            sub_viewers=[viewer_xy, viewer_xz, viewer_yz, viewer_3d],
            image_shape=(10, 512, 512),
            edge_color="cyan",
            edge_width=2.0
        )

        # A box added through the main layer appears in every viewer.
        bbox_layers[0].add_boxes([[[0, 10, 10], [5, 50, 50]]])

        main_layer = bbox_layers[0]       # editable
        display_layers = bbox_layers[1:]  # read-only

    Notes
    -----
    Each layer's sync mode is taken from its viewer's ``ndisplay`` once, at
    creation. A viewer that is switched between 2D and 3D afterwards keeps its
    mode. Reassign ``layer.sync_mode`` if needed.
    """
    logger.info(
        "Creating synchronized bbox layers: main=%s sub_viewers=%d sync_mode=%s",
        main_viewer,
        len(sub_viewers),
        sync_mode,
    )
    layers = []
    if sync_mode == "snapshot":
        main_store = BBoxDataStore(initial_data=bbox_data)
        main_layer = TurboBoxLayer(
            bbox_data_store=main_store, interactive=True, image_shape=image_shape, **layer_kwargs
        )
        main_viewer.add_layer(main_layer)
        layers.append(main_layer)
        logger.info("Created main snapshot layer '%s'", main_layer.name)
        for i, sub_viewer in enumerate(sub_viewers):
            sub_store = BBoxDataStore(
                initial_data=bbox_data.copy() if bbox_data is not None else None
            )
            sub_layer = TurboBoxLayer(
                bbox_data_store=sub_store,
                interactive=False,
                image_shape=image_shape,
                name=f"{layer_kwargs.get('name', 'Bounding Boxes')} (View {i + 1})",
                **{k: v for k, v in layer_kwargs.items() if k != "name"},
            )
            sub_viewer.add_layer(sub_layer)
            layers.append(sub_layer)
            logger.info("Created snapshot sub layer %d '%s'", i + 1, sub_layer.name)
    else:
        shared_store = BBoxDataStore(initial_data=bbox_data)
        main_layer_sync = _sync_mode_for_viewer(main_viewer)
        main_layer = TurboBoxLayer(
            bbox_data_store=shared_store,
            interactive=True,
            image_shape=image_shape,
            sync_mode=main_layer_sync,
            **layer_kwargs,
        )
        main_viewer.add_layer(main_layer)
        layers.append(main_layer)
        logger.info(
            "Created main synchronized layer '%s' (sync_mode=%s)",
            main_layer.name,
            main_layer_sync,
        )
        for i, sub_viewer in enumerate(sub_viewers):
            sub_layer_sync = _sync_mode_for_viewer(sub_viewer)
            sub_layer = TurboBoxLayer(
                bbox_data_store=shared_store,
                interactive=False,
                image_shape=image_shape,
                sync_mode=sub_layer_sync,
                name=f"{layer_kwargs.get('name', 'Bounding Boxes')} (View {i + 1})",
                **{k: v for k, v in layer_kwargs.items() if k != "name"},
            )
            sub_viewer.add_layer(sub_layer)
            layers.append(sub_layer)
            logger.info(
                "Created synchronized sub layer %d '%s' (sync_mode=%s)",
                i + 1,
                sub_layer.name,
                sub_layer_sync,
            )
    logger.info("Synchronized %d layers", len(layers))
    return layers


def sync_existing_layer_to_viewers(
    source_layer: TurboBoxLayer, target_viewers: list[napari.Viewer], **layer_kwargs
) -> list[TurboBoxLayer]:
    """
    Show an existing layer's boxes in more viewers (deprecated).

    Deprecated: emits a DeprecationWarning and will be removed in a future
    release. Use :func:`create_synchronized_bbox_layers` instead.

    The new layers use the source layer's own store, so edits made through the
    source layer show up in them too. They are read-only (``interactive=False``).
    Unlike :func:`create_synchronized_bbox_layers`, this function ignores the
    target viewer's ``ndisplay``: the layers use ``sync_mode="live"`` unless
    ``layer_kwargs`` sets it. They take the source's ``image_shape`` and the
    edge color and width of its first shape (cyan and 2.0 if it shows none).

    Parameters
    ----------
    source_layer : TurboBoxLayer
        Layer whose store the new layers share. It is not changed.
    target_viewers : list of napari.Viewer
        Viewers that get one new layer each, named
        ``"<source name> (Sync i)"``.
    **layer_kwargs : dict
        Further TurboBoxLayer arguments. They must not repeat the ones set
        here (bbox_data_store, interactive, image_shape, edge_color,
        edge_width, name).

    Returns
    -------
    list of TurboBoxLayer
        The new layers, one per target viewer, each already added to its viewer.

    Examples
    --------
    ::

        bbox_layer = viewer.layers['Bounding Boxes']
        sync_layers = sync_existing_layer_to_viewers(
            source_layer=bbox_layer,
            target_viewers=[viewer_xy, viewer_xz, viewer_yz]
        )
    """
    warnings.warn(
        "sync_existing_layer_to_viewers is deprecated and will be removed in a future release. Use create_synchronized_bbox_layers instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    shared_store = source_layer._bbox_store
    current_data = shared_store.data
    logger.info(
        "Syncing existing layer '%s' to %d viewers with %d boxes",
        source_layer.name,
        len(target_viewers),
        len(current_data),
    )
    layers = []
    for i, viewer in enumerate(target_viewers):
        edge_color = source_layer.edge_color
        edge_width = source_layer.edge_width
        if hasattr(edge_color, "__len__") and (not isinstance(edge_color, str)):
            edge_color = edge_color[0] if len(edge_color) > 0 else "cyan"
        if hasattr(edge_width, "__len__"):
            edge_width = edge_width[0] if len(edge_width) > 0 else 2.0
        sync_layer = TurboBoxLayer(
            bbox_data_store=shared_store,
            interactive=False,
            image_shape=source_layer._image_shape,
            edge_color=edge_color,
            edge_width=edge_width,
            name=f"{source_layer.name} (Sync {i + 1})",
            **layer_kwargs,
        )
        viewer.add_layer(sync_layer)
        layers.append(sync_layer)
        logger.info("Created synchronized copy %d '%s'", i + 1, sync_layer.name)
    logger.info("Finished creating %d synchronized layers", len(layers))
    return layers
