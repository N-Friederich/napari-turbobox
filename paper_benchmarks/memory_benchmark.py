"""Memory benchmark for the two box plugins (Figure 3c).

Two measurements, both written to the output directory:

- ``benchmark_memory.csv``: memory added by loading N boxes. A hidden napari
  viewer with an image and an empty box layer is created first; tracemalloc
  then records the net Python allocations of creating the (N, 2, 3) input
  array and loading it (``add_boxes`` for TurboBox, ``layer.data = boxes`` for
  napari-bbox) and of the ten event-loop passes after it. Memory held before
  (viewer, image, empty layer) and allocations outside Python's allocator (Qt,
  GPU) are not counted.
- ``benchmark_memory_datastructure.csv``: without a viewer. For TurboBox, the
  exact bytes of its box store and spatial index and the traced allocations of
  creating the layer and adding the boxes; for napari-bbox, the traced
  allocations of creating the layer and assigning the boxes (it has no separate
  store or index).

Run on its own, the script picks one plugin from the napari version (napari-bbox
for 0.4/0.5, TurboBox otherwise). ``paper_benchmarks/final_run/memory_final.py``
runs both plugins in one environment, as in the reported run.
"""

from __future__ import annotations

import gc
import os
import sys
import tracemalloc

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), "src"))

def get_environment_info():
    """Detect which napari environment we're in."""
    try:
        import napari
        napari_version = napari.__version__
    except ImportError:
        napari_version = "unknown"

    plugin_name = None
    # Pick the plugin that MATCHES the running napari: napari-bbox targets
    # 0.4/0.5, napari-turbobox targets 0.6.6+. src/ may be on sys.path in either
    # environment, so a version-blind "import napari_turbobox first" wrongly picks
    # turbobox under napari 0.4/0.5 and then fails instantiating TurboBoxLayer.
    is_legacy = napari_version.startswith(("0.4", "0.5"))
    if is_legacy:
        try:
            import napari_bbox
            plugin_name = "napari-bbox"
        except ImportError:
            pass
    if plugin_name is None:
        try:
            import napari_turbobox
            plugin_name = "napari-turbobox"
        except ImportError:
            try:
                import napari_bbox
                plugin_name = "napari-bbox"
            except ImportError:
                pass

    return {
        "napari_version": napari_version,
        "plugin_name": plugin_name,
    }

def generate_random_boxes(n_boxes: int, image_shape: tuple[int, int, int]) -> np.ndarray:
    """Generate random bounding boxes."""
    boxes = []
    Z, Y, X = image_shape
    size = np.array([20, 20, 20])

    np.random.seed(42)
    for _ in range(n_boxes):
        z_start = np.random.randint(0, max(1, Z - size[0]))
        y_start = np.random.randint(0, max(1, Y - size[1]))
        x_start = np.random.randint(0, max(1, X - size[2]))
        mins = np.array([z_start, y_start, x_start], dtype=float)
        maxs = mins + size
        boxes.append([mins, maxs])

    return np.array(boxes, dtype=float)

def measure_memory_usage(n_boxes: int, plugin_name: str) -> float | None:
    """Measure memory usage increase for n_boxes using tracemalloc."""
    gc.collect()

    try:
        import napari
        from qtpy.QtWidgets import QApplication

        # Hidden viewer (show=False, native Qt platform)
        viewer = napari.Viewer(show=False)
        viewer.add_image(np.zeros((100, 512, 512), dtype=np.uint8), name='background')

        # Add boxes layer. (Fixed: the layer is TurboBoxLayer, not the
        # long-removed OptimizedBoundingBoxLayer; runs under a real Viewer on a
        # GUI session -- do NOT force QT_QPA_PLATFORM=offscreen, which segfaults
        # the GL context on this stack.)
        if plugin_name == "napari-turbobox":
            from napari_turbobox.layer import TurboBoxLayer
            layer = TurboBoxLayer(ndim=3, edge_color='cyan', image_shape=(100, 512, 512))
            viewer.add_layer(layer)
        elif plugin_name == "napari-bbox":
            from napari_bbox import BoundingBoxLayer
            layer = BoundingBoxLayer(ndim=3, edge_color='cyan')
            viewer.add_layer(layer)
        else:
            return None

        # Force baseline state
        QApplication.processEvents()
        gc.collect()

        # Start tracking
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        # Generate and add data. TurboBoxLayer stores boxes via add_boxes (its
        # `.data` is the rendered Shapes paths, not the bbox array); napari-bbox
        # takes the (N, 2, 3) array directly through `.data`.
        boxes = generate_random_boxes(n_boxes, (100, 512, 512))
        if plugin_name == "napari-turbobox":
            layer.add_boxes(boxes)
        else:
            layer.data = boxes

        # Force update and render logic
        for _ in range(10):
            QApplication.processEvents()

        # Take snapshot
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate diff
        # Filter for relevant allocations to avoid noise from background threads?
        # Actually, we want total impact.
        stats = snapshot_after.compare_to(snapshot_before, 'lineno')

        total_diff_bytes = sum(stat.size_diff for stat in stats)
        total_diff_mb = total_diff_bytes / 1024 / 1024

        # Cleanup
        viewer.close()
        gc.collect()
        QApplication.processEvents()

        # Sanity check: ensure positive
        return max(0.0, total_diff_mb)

    except Exception as e:
        print(f"❌ Error measuring memory: {e}")
        import traceback
        traceback.print_exc()
        return None

def _index_bytes(idx) -> int:
    """Exact byte size of the SpatialIndex's owned arrays (dtype-aware)."""
    total = idx._boxes.nbytes
    for name in ("_sorted_mins", "_sorted_maxs", "_sorted_indices_mins",
                 "_sorted_indices_maxs", "_position_in_sorted_mins", "_position_in_sorted_maxs"):
        total += sum(a.nbytes for a in getattr(idx, name))
    return total


def measure_datastructure(n_boxes: int, plugin_name: str) -> dict | None:
    """Measure the plugin's own data-structure footprint, viewer-less.

    Modern: exact store + spatial-index bytes (dtype-aware) AND total tracemalloc
    allocation around layer construction + add_boxes. Legacy: total allocation
    around ``BoundingBoxLayer`` + ``layer.data = boxes`` (it has no separable
    store/index). Returns a row for ``benchmark_memory_datastructure.csv``.
    """
    from qtpy.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    boxes = generate_random_boxes(n_boxes, (100, 512, 512))
    gc.collect()
    try:
        if plugin_name == "napari-turbobox":
            from napari_turbobox.layer import TurboBoxLayer
            tracemalloc.start()
            base = tracemalloc.take_snapshot()
            layer = TurboBoxLayer(ndim=3, image_shape=(100, 512, 512))
            layer.add_boxes(boxes)
            after = tracemalloc.take_snapshot()
            total = sum(s.size_diff for s in after.compare_to(base, "lineno"))
            tracemalloc.stop()
            store_b = layer._bbox_store._bboxes.nbytes
            idx_b = _index_bytes(layer._spatial_index)
            return {"tool": "modern", "n_boxes": n_boxes, "store_bytes": store_b, "index_bytes": idx_b,
                        "store_plus_index": store_b + idx_b, "total_alloc_bytes": max(0, total),
                        "store_dtype": str(layer._bbox_store._bboxes.dtype)}
        elif plugin_name == "napari-bbox":
            from napari_bbox import BoundingBoxLayer
            tracemalloc.start()
            base = tracemalloc.take_snapshot()
            layer = BoundingBoxLayer(ndim=3)
            layer.data = boxes
            after = tracemalloc.take_snapshot()
            total = sum(s.size_diff for s in after.compare_to(base, "lineno"))
            tracemalloc.stop()
            return {"tool": "legacy", "n_boxes": n_boxes, "store_bytes": "", "index_bytes": "",
                        "store_plus_index": "", "total_alloc_bytes": max(0, total), "store_dtype": ""}
        return None
    except Exception as e:
        print(f"❌ Error measuring data structure: {e}")
        return None


def run_datastructure_benchmark(plugin: str, output_dir: str) -> None:
    """Write/merge the store+index data-structure footprint CSV."""
    rows = []
    for n in (100, 1000, 10000):
        r = measure_datastructure(n, plugin)
        if r is not None:
            rows.append(r)
            print(f"  data-structure N={n}: store+index={r['store_plus_index'] or '-'} "
                  f"total_alloc={r['total_alloc_bytes']} bytes ({r.get('store_dtype', '')})")
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["tool", "n_boxes", "store_bytes", "index_bytes",
                                     "store_plus_index", "total_alloc_bytes", "store_dtype"])
    this_tool = "modern" if plugin == "napari-turbobox" else "legacy"
    filename = os.path.join(output_dir, "benchmark_memory_datastructure.csv")
    if os.path.exists(filename):
        try:
            existing = pd.read_csv(filename)
            existing = existing[existing["tool"] != this_tool]
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(filename, index=False)
    print(f"✅ Saved data-structure memory to {filename}")


def run_benchmark(output_dir="."):
    env = get_environment_info()
    plugin = env['plugin_name']

    if not plugin:
        print("❌ No supported plugin detected!")
        return

    print(f"📊 Running MEMORY benchmark for {plugin}...")

    results = []

    # Box counts to test
    box_counts = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    for n in box_counts:
        print(f"  Testing {n} boxes...", end="", flush=True)
        mem_usage = measure_memory_usage(n, plugin)

        if mem_usage is not None:
            print(f" {mem_usage:.2f} MB")
            results.append({
                "plugin": plugin,
                "n_boxes": n,
                "memory_increase_mb": mem_usage
            })
        else:
            print(" Failed")

    if results:
        df = pd.DataFrame(results)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "benchmark_memory.csv")

        # If file exists, append (so we can merge legacy and modern runs)
        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename)
                # Remove old entries for this plugin to avoid dupes
                existing_df = existing_df[existing_df['plugin'] != plugin]
                df = pd.concat([existing_df, df], ignore_index=True)
            except:
                pass

        df.to_csv(filename, index=False)
        print(f"\n✅ Saved total-memory results to {filename}")

    print(f"\n📊 Running DATA-STRUCTURE memory benchmark for {plugin}...")
    run_datastructure_benchmark(plugin, output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run memory benchmark")
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args()

    run_benchmark(output_dir=args.output)
