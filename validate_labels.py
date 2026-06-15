"""
Step 4: Validate label quality.

Runs NDVI separation, density check, mean NDVI trajectory, and DW class
distribution of confirmed pixels — statistical sanity-check on the labeling
method's outputs.

Usage:
    python validate_labels.py --aoi babcock
"""

import numpy as np
import json
import sys
from pathlib import Path


DW_NAMES = {
    0: "water", 1: "trees", 2: "grass", 3: "flooded", 4: "crops",
    5: "shrub", 6: "built", 7: "bare", 8: "snow", 9: "clouds", 255: "nodata",
}


# ── Diagnostics (original behaviour, unchanged) ───────────────────────────────

def run_diagnostics(data_dir: Path):
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    quarters = meta["quarters"]
    T = len(quarters)
    aoi_name = meta.get("aoi", str(data_dir.name))

    label_cube = np.load(data_dir / "label_cube.npy")
    ndvi_cube  = np.load(data_dir / "ndvi_cube.npy")
    dw_cube    = np.load(data_dir / "dw_cube.npy")

    ever_grading     = (label_cube == 1).any(axis=0)
    ever_constructed = (label_cube == 2).any(axis=0)
    confirmed = ever_grading & ever_constructed

    print("=" * 60)
    print(f"LABEL VALIDATION — {aoi_name}")
    print("=" * 60)
    print(f"Confirmed trajectory pixels: {confirmed.sum()}")

    # [1] NDVI separation at peak-grading quarter
    grading_counts = [(label_cube[t] == 1).sum() for t in range(T)]
    peak_t = int(np.argmax(grading_counts))
    print(f"\n[1] NDVI per class at peak-grading quarter ({quarters[peak_t]}):")
    ndvi_peak  = ndvi_cube[peak_t]
    label_peak = label_cube[peak_t]
    for cls, name in [(0, "baseline"), (1, "grading"), (2, "constructed")]:
        mask = label_peak == cls
        if mask.sum() > 0:
            v = ndvi_peak[mask]
            print(f"    {name:12}: n={mask.sum():>6}  "
                  f"NDVI={np.nanmean(v):.3f}±{np.nanstd(v):.3f}")

    # [2] Density check
    print("\n[2] Development-zone density check (final quarter):")
    constructed_final = label_cube[-1] == 2
    dw_built_final    = dw_cube[-1] == 6
    if constructed_final.sum() > 0:
        overlap = (constructed_final & dw_built_final).sum()
        density = overlap / constructed_final.sum() * 100
        print(f"    Confirmed-constructed pixels:        {constructed_final.sum()}")
        print(f"    With DW=built in final quarter:      {overlap}  ({density:.1f}%)")
    else:
        print("    No constructed pixels in final quarter.")

    # [3] Mean NDVI trajectory across confirmed pixels
    print(f"\n[3] Mean NDVI trajectory for confirmed pixels ({confirmed.sum()} px):")
    if confirmed.sum() > 0:
        print(f"    {'Quarter':<12}  {'Mean NDVI':>10}  {'Std':>8}")
        print(f"    {'-'*32}")
        for t in range(T):
            vals = ndvi_cube[t][confirmed]
            print(f"    {quarters[t]:<12}  "
                  f"{np.nanmean(vals):>10.3f}  {np.nanstd(vals):>8.3f}")
    else:
        print("    No confirmed pixels found.")

    # [4] DW class distribution of confirmed pixels per quarter
    key_classes = [1, 2, 5, 6, 7]  # trees, grass, shrub, built, bare
    print(f"\n[4] DW class distribution for confirmed pixels (% per quarter):")
    if confirmed.sum() > 0:
        header = (f"    {'Quarter':<12}"
                  + "".join(f"  {DW_NAMES[c]:<8}" for c in key_classes))
        print(header)
        print(f"    {'-' * (len(header) - 4)}")
        for t in range(T):
            dw_t = dw_cube[t][confirmed]
            n    = len(dw_t)
            row  = f"    {quarters[t]:<12}"
            for c in key_classes:
                pct = (dw_t == c).sum() / n * 100
                row += f"  {pct:>6.1f}%  "
            print(row)
    else:
        print("    No confirmed pixels found.")

    print(f"\n{'=' * 60}")
    print("DIAGNOSTICS COMPLETE")
    print(f"{'=' * 60}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from config import AOIS, DATA_ROOT

    parser = argparse.ArgumentParser(description="Validate label quality")
    parser.add_argument("--aoi", required=True, choices=list(AOIS))
    args = parser.parse_args()

    data_dir = Path(DATA_ROOT) / args.aoi
    run_diagnostics(data_dir)