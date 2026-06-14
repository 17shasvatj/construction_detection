"""
Step 2: Pull Dynamic World quarterly mode maps from Google Earth Engine.

Downloads as GeoTIFF (not sampleRectangle — pixel count exceeds limit).
Aligns to S2 grid via PIL nearest-neighbor resize.

Standalone: python -m pipeline.pull_dw --aoi babcock
"""

import numpy as np
import json
import ee
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


DW_COLORS = [
    '#419BDF', '#397D49', '#88B053', '#7A87C6',
    '#E49635', '#DFC35A', '#C4281B', '#A59B8F',
    '#B39FE1', '#FFFFFF',
]
DW_LABELS = ['water', 'trees', 'grass', 'flooded', 'crops',
             'shrub', 'built', 'bare', 'snow', 'clouds']
DW_CLASSES = {
    "0": "water", "1": "trees", "2": "grass",
    "3": "flooded_veg", "4": "crops", "5": "shrub",
    "6": "built", "7": "bare", "8": "snow", "9": "clouds",
    "255": "nodata",
}


def quarter_dates(key):
    """Convert '2021-Q1' to (start, end) date strings."""
    year, q = int(key[:4]), int(key[-1])
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends   = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    if year == 2026 and q >= 2:
        return f"{year}-04-01", f"{year}-06-01"
    return f"{year}-{starts[q]}", f"{year}-{ends[q]}"


def run(data_dir):
    import urllib.request
    import tempfile
    import rasterio

    data_dir = Path(data_dir)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    bbox    = meta["bbox"]
    quarters = meta["quarters"]
    H, W    = meta["shape"][1], meta["shape"][2]

    print(f"AOI bbox: {bbox}")
    print(f"Grid: {H}x{W} pixels")
    print(f"Quarters: {len(quarters)}")

    print("\nInitializing Earth Engine...")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import EE_PROJECT
    ee.Initialize(project=EE_PROJECT)

    aoi = ee.Geometry.Rectangle(bbox)

    def pull_dw_quarter(key):
        start, end = quarter_dates(key)
        dw = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterDate(start, end)
            .filterBounds(aoi)
            .select("label")
        )
        n_scenes = dw.size().getInfo()
        if n_scenes == 0:
            print(f"  {key}: NO SCENES")
            return None, 0

        mode_img = dw.mode().clip(aoi)
        url = mode_img.getDownloadURL({
            "region": aoi.getInfo(),
            "scale": 10,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        })

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            with rasterio.open(tmp.name) as src:
                arr = src.read(1)

        print(f"  {key}: {n_scenes} scenes → shape {arr.shape}, "
              f"unique classes: {np.unique(arr)}")
        return arr, n_scenes

    print(f"\nPulling {len(quarters)} DW quarterly mode maps...\n")

    dw_arrays   = {}
    scene_counts = {}

    for key in quarters:
        arr, n = pull_dw_quarter(key)
        if arr is not None:
            dw_arrays[key]    = arr
            scene_counts[key] = n

    print(f"\nSuccessfully pulled {len(dw_arrays)}/{len(quarters)} quarters")

    print(f"\nAligning to S2 grid ({H}x{W})...")

    dw_cube = np.full((len(quarters), H, W), 255, dtype=np.uint8)

    for i, key in enumerate(quarters):
        if key not in dw_arrays:
            print(f"  {key}: missing, filled with 255")
            continue
        arr = dw_arrays[key]
        if arr.shape == (H, W):
            dw_cube[i] = arr
        else:
            img = Image.fromarray(arr.astype(np.uint8))
            img_resized = img.resize((W, H), Image.NEAREST)
            dw_cube[i] = np.array(img_resized)
            print(f"  {key}: resized {arr.shape} → ({H},{W})")

    np.save(data_dir / "dw_cube.npy", dw_cube)

    dw_meta = {
        "quarters":     quarters,
        "shape":        [len(quarters), H, W],
        "scene_counts": scene_counts,
        "classes":      DW_CLASSES,
    }
    with open(data_dir / "dw_metadata.json", "w") as f:
        json.dump(dw_meta, f, indent=2)

    print(f"\nSaved dw_cube.npy {dw_cube.shape}")
    print("Saved dw_metadata.json")

    print("\nClass distribution:")
    for label_idx, key in [(0, quarters[0]), (-1, quarters[-1])]:
        arr   = dw_cube[label_idx]
        valid = arr[arr != 255]
        if len(valid) == 0:
            continue
        counts = np.bincount(valid, minlength=10)
        total  = counts.sum()
        print(f"\n  {key}:")
        for cls, name in DW_CLASSES.items():
            if cls == "255":
                continue
            c = int(cls)
            if c < len(counts) and counts[c] > 0:
                print(f"    {name:<12}: {counts[c]:>6} px ({counts[c] / total * 100:>5.1f}%)")

    # Visualization
    dw_cmap = ListedColormap(DW_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(dw_cube[0], cmap=dw_cmap, vmin=0, vmax=9)
    axes[0].set_title(f"DW — {quarters[0]}")
    axes[0].axis('off')

    mid = len(quarters) // 2
    axes[1].imshow(dw_cube[mid], cmap=dw_cmap, vmin=0, vmax=9)
    axes[1].set_title(f"DW — {quarters[mid]}")
    axes[1].axis('off')

    axes[2].imshow(dw_cube[-1], cmap=dw_cmap, vmin=0, vmax=9)
    axes[2].set_title(f"DW — {quarters[-1]}")
    axes[2].axis('off')

    patches = [mpatches.Patch(color=DW_COLORS[i], label=DW_LABELS[i]) for i in range(10)]
    fig.legend(handles=patches, loc='lower center', ncol=10, fontsize=8)

    plt.suptitle("Dynamic World quarterly mode maps", fontsize=13)
    plt.tight_layout()
    plt.savefig(data_dir / "dw_overview.png", dpi=150, bbox_inches='tight')
    print("\nSaved dw_overview.png")


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import AOIS, DATA_ROOT

    parser = argparse.ArgumentParser(description="Pull DW quarterly mode maps")
    parser.add_argument("--aoi", required=True, choices=list(AOIS))
    args = parser.parse_args()

    run(Path(DATA_ROOT) / args.aoi)
