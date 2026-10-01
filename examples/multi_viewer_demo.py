"""
Worked Example: Synchronized Multi-Viewer Bounding Box Annotation
================================================================

This script shows how to set up synchronized multi-viewer 3D bounding box
annotation with `napari-turbobox` (`create_synchronized_bbox_layers`).

It opens three napari windows on a synthetic 3D volume, all showing the same
boxes from one shared `BBoxDataStore`:
1. XY viewer: slice view where boxes are edited (editable layer)
2. XZ viewer: orthogonal slice view (read-only layer)
3. 3D viewer: volume rendering with box wireframes (read-only layer)

Each layer's sync mode is chosen from its viewer's display mode when the layers
are created: layers in 2D viewers are "live" (they follow every change,
including each step of a drag), the layer in the 3D viewer is "on_commit" (it
updates once per drag, when the mouse is released). The viewers are therefore
configured before the layers are created.
"""

import napari
import numpy as np

from napari_turbobox import create_synchronized_bbox_layers


def main():
    # 1. Create a synthetic 3D image volume (e.g., ZYX = 64, 256, 256)
    print("Generating synthetic 3D image volume...")
    shape = (64, 256, 256)
    image = np.zeros(shape, dtype=np.float32)

    # Draw some synthetic structural spheres
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    centers = [(16, 64, 64), (32, 128, 128), (48, 192, 192)]
    radii = [12, 20, 16]
    for (z, y, x), r in zip(centers, radii):
        mask = ((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) <= r**2
        image[mask] = 1.0

    # Add random noise
    rng = np.random.default_rng(42)
    image += rng.normal(0, 0.1, shape).astype(np.float32)
    image = np.clip(image, 0, 1)

    # 2. Define initial bounding boxes around the synthetic structures
    # BBox coordinates format: [[z_min, y_min, x_min], [z_max, y_max, x_max]]
    initial_bboxes = np.array(
        [
            [[4, 52, 52], [28, 76, 76]],  # Around first structure
            [[12, 108, 108], [52, 148, 148]],  # Around second structure
        ]
    )

    print("Initializing napari viewers...")
    # 3. Create three napari viewers. Their display mode (ndisplay) must be set
    # before the layers are created, because it decides each layer's sync mode.
    # Main XY slice plane annotator
    viewer_main = napari.Viewer(title="XY View (editable)")
    viewer_main.add_image(image, name="Volume", colormap="gray")
    viewer_main.dims.ndisplay = 2
    viewer_main.dims.set_point(0, 20)  # z = 20 crosses both boxes

    # Orthogonal XZ plane view
    viewer_ortho = napari.Viewer(title="XZ Orthogonal View (read-only)")
    viewer_ortho.add_image(image, name="Volume", colormap="gray")
    viewer_ortho.dims.ndisplay = 2
    viewer_ortho.dims.order = (1, 0, 2)  # slices along Y, displays Z and X
    viewer_ortho.dims.set_point(1, 128)  # y = 128 crosses the second box

    # 3D visualization view
    viewer_3d = napari.Viewer(title="3D Render View (read-only)")
    viewer_3d.add_image(image, name="Volume", colormap="gray", rendering="mip")
    viewer_3d.dims.ndisplay = 3

    # 4. Create the synchronized bounding box layers, one per viewer.
    # The layer in the main viewer is editable; the layers in the sub-viewers are
    # created read-only (interactive=False at construction). The sync mode of each
    # layer is derived from its viewer: "live" for the 2D viewers, "on_commit" for
    # the 3D viewer.
    print("Setting up synchronized bounding box layers...")
    layers = create_synchronized_bbox_layers(
        main_viewer=viewer_main,
        sub_viewers=[viewer_ortho, viewer_3d],
        bbox_data=initial_bboxes,
        image_shape=shape,
        edge_width=2.0,
    )

    # Label the layers for clarity
    layers[0].name = "BBoxes (Active XY)"
    layers[1].name = "BBoxes (Sync XZ)"
    layers[2].name = "BBoxes (Sync 3D)"

    for layer in layers:
        print(f"{layer.name}: sync_mode={layer.sync_mode}, editable={layer.editable}")

    # Print out user instructions
    print("\n" + "=" * 50)
    print("DEMO INSTRUCTIONS:")
    print("1. Arrange the three napari windows so that all are visible.")
    print("2. In the 'XY View' window the box layer starts in Select mode.")
    print("3. Drag inside a box to move it; Shift-drag or grab near an edge to resize it.")
    print("4. The XZ View (slice y = 128) shows the second box and follows it during the drag.")
    print("5. The 3D View updates once, when you release the mouse button.")
    print("6. A box drawn with napari's 'Add Rectangle' tool spans the full Z range.")
    print("=" * 50 + "\n")

    # Start the napari event loop
    napari.run()


if __name__ == "__main__":
    main()
