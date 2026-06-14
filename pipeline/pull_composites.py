"""
Step 1: Pull quarterly S2 L2A median composites via Planetary Computer.

Standalone: python -m pipeline.pull_composites --aoi babcock
"""

import numpy as np
import planetary_computer as pc
import pystac_client
import stackstac
import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLOUD_THRESH = 20
N_SCENES = 15
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

QUARTERS = []
for _year in range(2021, 2027):
    _max_q = 2 if _year == 2026 else 4
    for _q in range(1, _max_q + 1):
        QUARTERS.append((_year, _q))


def quarter_dates(year, quarter):
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends   = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    if year == 2026 and quarter == 2:
        return f"{year}-04-01", f"{year}-06-01"
    return f"{year}-{starts[quarter]}", f"{year}-{ends[quarter]}"


def run(aoi_name, bbox, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"AOI: {aoi_name} — {bbox}")
    print(f"Quarters: {len(QUARTERS)} ({QUARTERS[0]} → {QUARTERS[-1]})")
    print(f"Output: {output_dir}/\n")

    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)

    def pull_quarter(year, quarter):
        start, end = quarter_dates(year, quarter)
        key = f"{year}-Q{quarter}"

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": CLOUD_THRESH}},
        )
        all_items = search.item_collection()
        tile_counts = Counter(i.properties.get("s2:mgrs_tile") for i in all_items)
        if not tile_counts:
            print(f"  {key}: no items found — SKIPPING")
            return None

        best_tile = tile_counts.most_common(1)[0][0]
        tiles = set(i.properties.get("s2:mgrs_tile") for i in all_items)
        print(f"MGRS tiles: {tiles}")
        if quarter == 1 and year == 2021:
            print(f"  Using MGRS tile: {best_tile}")

        tile_items = [i for i in all_items if i.properties.get("s2:mgrs_tile") == best_tile]
        tile_items = sorted(
            tile_items, key=lambda x: x.properties.get("eo:cloud_cover", 100)
        )[:N_SCENES]

        if len(tile_items) < 2:
            print(f"  {key}: only {len(tile_items)} scenes — SKIPPING")
            return None

        stack = stackstac.stack(
            tile_items,
            assets=BANDS + ["SCL"],
            bounds_latlon=bbox,
            resolution=10,
            dtype=np.float64,
            fill_value=np.nan,
            rescale=False,
        )

        scl = stack.sel(band="SCL")
        cloud_mask = (scl != 3) & (scl != 8) & (scl != 9) & (scl != 10) & (scl != 11)

        spectral = stack.sel(band=BANDS)
        masked = spectral.where(cloud_mask)
        composite = masked.median(dim="time").compute()

        b04 = composite.sel(band="B04").values
        b08 = composite.sel(band="B08").values
        b02 = composite.sel(band="B02").values
        b11 = composite.sel(band="B11").values

        ndvi = (b08 - b04) / (b08 + b04 + 1e-8)
        bsi  = ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02) + 1e-8)

        all_bands = np.concatenate([
            composite.values,
            ndvi[np.newaxis, :, :],
            bsi[np.newaxis, :, :],
        ], axis=0)

        nan_pct = np.isnan(all_bands[0]).mean() * 100
        print(f"  {key}: {len(tile_items)} scenes → {all_bands.shape} NaN={nan_pct:.0f}%")
        return all_bands

    print("=" * 60)
    print("PULLING QUARTERLY COMPOSITES")
    print("=" * 60)

    composites = {}
    band_names = BANDS + ["NDVI", "BSI"]
    shape_ref = None

    for year, quarter in QUARTERS:
        key = f"{year}-Q{quarter}"
        result = pull_quarter(year, quarter)
        if result is not None:
            composites[key] = result
            if shape_ref is None:
                shape_ref = result.shape
            elif result.shape != shape_ref:
                print(f"  WARNING: {key} shape {result.shape} != {shape_ref}")

    print(f"\nSuccessfully pulled {len(composites)}/{len(QUARTERS)} quarters")
    print(f"Image shape: {shape_ref}")
    print(f"Bands: {band_names}")

    print("\n" + "=" * 60)
    print("BUILDING TEMPORAL CUBES")
    print("=" * 60)

    sorted_keys = sorted(composites.keys())
    ndvi_idx = band_names.index("NDVI")
    bsi_idx  = band_names.index("BSI")

    ndvi_cube     = np.stack([composites[k][ndvi_idx] for k in sorted_keys], axis=0)
    bsi_cube      = np.stack([composites[k][bsi_idx]  for k in sorted_keys], axis=0)
    spectral_cube = np.stack([composites[k][:6]        for k in sorted_keys], axis=0)

    print(f"NDVI cube:     {ndvi_cube.shape}")
    print(f"BSI cube:      {bsi_cube.shape}")
    print(f"Spectral cube: {spectral_cube.shape}")
    print(f"Time steps:    {sorted_keys}")

    ndvi_mean = np.nanmean(ndvi_cube, axis=0)
    ndvi_std  = np.nanstd(ndvi_cube, axis=0)
    print(f"\nNDVI temporal mean (image avg): {np.nanmean(ndvi_mean):.3f}")
    print(f"NDVI temporal std  (image avg): {np.nanmean(ndvi_std):.3f}")

    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)

    for key, data in composites.items():
        np.save(output_dir / f"composite_{key}.npy", data)

    np.save(output_dir / "ndvi_cube.npy",     ndvi_cube)
    np.save(output_dir / "bsi_cube.npy",      bsi_cube)
    np.save(output_dir / "spectral_cube.npy", spectral_cube)

    metadata = {
        "aoi":        aoi_name,
        "bbox":       bbox,
        "mgrs_tile":  "auto",
        "quarters":   sorted_keys,
        "band_names": band_names,
        "shape":      list(shape_ref) if shape_ref else None,
        "n_quarters": len(composites),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(composites)} composites to {output_dir}/")
    print("Saved ndvi_cube, bsi_cube, spectral_cube, metadata.json")

    # Visualization
    def norm_rgb(data):
        arr = data[[2, 1, 0]].copy()
        out = np.zeros_like(arr)
        for i in range(3):
            b = arr[i]
            v = b[~np.isnan(b)]
            if len(v):
                lo, hi = np.percentile(v, 2), np.percentile(v, 98)
                out[i] = np.clip((b - lo) / (hi - lo + 1e-8), 0, 1)
        return np.nan_to_num(np.transpose(out, (1, 2, 0)))

    H, W = ndvi_cube.shape[1], ndvi_cube.shape[2]
    cx, cy = H // 2, W // 2

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    points = {
        "Center (construction)": (cx, cy),
        "Top-left (likely stable)": (50, 50),
        "Bottom-right": (H - 50, W - 50),
        "Top-right": (50, W - 50),
    }
    for label, (r, c) in points.items():
        ts = ndvi_cube[:, r, c]
        axes[0, 0].plot(range(len(sorted_keys)), ts, marker='.', label=label)
    axes[0, 0].set_xticks(range(len(sorted_keys)))
    axes[0, 0].set_xticklabels(sorted_keys, rotation=90, fontsize=7)
    axes[0, 0].set_ylabel("NDVI")
    axes[0, 0].set_title("NDVI Time Series at Sample Points")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    im = axes[0, 1].imshow(ndvi_std, cmap='hot_r', vmin=0,
                           vmax=np.nanpercentile(ndvi_std, 95))
    axes[0, 1].set_title("NDVI Temporal Std Dev\n(bright = high variability over time)")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    first_key = sorted_keys[0]
    last_key  = sorted_keys[-1]
    axes[1, 0].imshow(norm_rgb(composites[first_key]))
    axes[1, 0].set_title(f"RGB — {first_key}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(norm_rgb(composites[last_key]))
    axes[1, 1].set_title(f"RGB — {last_key}")
    axes[1, 1].axis('off')

    plt.suptitle(
        f"{aoi_name} Temporal Overview — {W}×{H}px — "
        f"{len(sorted_keys)} quarters ({sorted_keys[0]} → {sorted_keys[-1]})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_overview.png", dpi=150, bbox_inches='tight')
    print("Saved temporal_overview.png")

    print("\n" + "=" * 60)
    print("DONE — Ready for DW pull")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import AOIS, DATA_ROOT

    parser = argparse.ArgumentParser(description="Pull S2 quarterly composites")
    parser.add_argument("--aoi", required=True, choices=list(AOIS))
    args = parser.parse_args()

    cfg = AOIS[args.aoi]
    if cfg["bbox"] is None:
        sys.exit(f"AOI '{args.aoi}' has no bbox configured yet.")
    run(args.aoi, cfg["bbox"], Path(DATA_ROOT) / args.aoi)
