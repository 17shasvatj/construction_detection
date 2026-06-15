"""
Step 4: Validate label quality.

Two modes (composable — pass both flags to run both):

  --diagnostics  (default when no flags given)
      NDVI separation, density check, mean NDVI trajectory, DW class
      distribution of confirmed pixels.

  --spot-check N
      Sample N confirmed-trajectory pixels spatially stratified, print
      one row per site with coordinates and label-transition quarters.

Usage:
    python validate_labels.py --aoi babcock
    python validate_labels.py --aoi babcock --spot-check 30
    python validate_labels.py --aoi babcock --diagnostics --spot-check 30
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


# ── Spot-check ────────────────────────────────────────────────────────────────

def run_spot_check(aoi: str, n: int, data_root: Path, seed: int = 42):
    """Sample n confirmed-trajectory pixels and print one row per site."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from spot_check import (
        load_aoi, pixel_to_lonlat,
        label_transitions, stratified_sample,
        format_row, format_label_transitions,
        DW_NAMES as SC_DW_NAMES,
    )

    d = load_aoi(aoi, str(data_root))
    label_cube = np.array(d['label_cube'])
    dw_cube    = d['dw_cube']
    quarters   = d['quarters']
    bbox, H, W = d['bbox'], d['H'], d['W']

    ever_g = (label_cube == 1).any(axis=0)
    ever_b = (label_cube == 2).any(axis=0)
    confirmed = ever_g & ever_b

    print("=" * 72)
    print(f"SPOT-CHECK SITES — {aoi}  (confirmed-trajectory pixels: {int(confirmed.sum())})")
    print("=" * 72)

    if confirmed.sum() == 0:
        print("No confirmed-trajectory pixels in this AOI. Nothing to sample.")
        return

    sites = stratified_sample(confirmed, n, seed=seed)
    if not sites:
        print("Stratified sampling returned 0 sites — mask appears empty.")
        return

    print(f"Sampled {len(sites)} sites (target: {n}, 4×4 spatial stratification)\n")

    for i, (y, x) in enumerate(sites, start=1):
        lon, lat = pixel_to_lonlat(y, x, bbox, H, W)

        if dw_cube is not None:
            from spot_check import dw_trajectory
            dw_traj = dw_trajectory(np.array(dw_cube), y, x, names=SC_DW_NAMES)
        else:
            dw_traj = '(no dw_cube.npy)'

        g_onset, b_onset = label_transitions(label_cube, y, x, quarters)
        transitions_str  = format_label_transitions(g_onset, b_onset)

        print(format_row(
            idx=i,
            class_name='confirmed',
            lat=lat,
            lon=lon,
            dw_traj=dw_traj,
            label_transitions_str=transitions_str,
            # url omitted intentionally
        ))

    print()
    print("Tally these in Google Earth Pro's historical slider.")
    print("Use the grading/built onset quarters to know which dates to compare.")
    print("Record counts as: real / ambiguous / wrong.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from config import AOIS, DATA_ROOT

    parser = argparse.ArgumentParser(description="Validate label quality")
    parser.add_argument("--aoi", required=True, choices=list(AOIS))
    parser.add_argument("--diagnostics", action="store_true",
                        help="Run NDVI/density/trajectory/DW diagnostics.")
    parser.add_argument("--spot-check", type=int, default=0, metavar="N",
                        help="Print N confirmed-trajectory spot-check sites.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.diagnostics and args.spot_check == 0:
        args.diagnostics = True

    data_dir = Path(DATA_ROOT) / args.aoi

    if args.diagnostics:
        run_diagnostics(data_dir)

    if args.spot_check > 0:
        if args.diagnostics:
            print()
        run_spot_check(args.aoi, args.spot_check, Path(DATA_ROOT), seed=args.seed)