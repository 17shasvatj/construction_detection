"""
Sample confirmed-grading and confirmed-constructed pixels from DW-derived labels
for manual Google Earth spot-check.

Usage:
    python validate_labels.py {aoi} [--n 15] [--seed 42] [--data-root data] [--dry-run]

Prints one row per site to stdout. No files written in any mode.
"""

import argparse
import sys

from spot_check import (
    load_aoi, pixel_to_lonlat, earth_url, dw_trajectory,
    stratified_sample, format_row, DW_NAMES, LABEL_NAMES,
)


def main():
    parser = argparse.ArgumentParser(
        description='Print Google Earth spot-check sites from DW-derived labels.')
    parser.add_argument('aoi', help='AOI name, e.g. sunterra, wendell, babcock')
    parser.add_argument('--n', type=int, default=15, metavar='N',
                        help='Sites per class (default 15)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-root', default='data', metavar='DIR')
    parser.add_argument('--dry-run', action='store_true',
                        help='Add [DRY RUN] prefix; behaviour is otherwise identical '
                             '(this script never writes files)')
    args = parser.parse_args()

    try:
        d = load_aoi(args.aoi, args.data_root)
    except FileNotFoundError as e:
        print(f'[validate_labels] ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    label    = d['label_cube'][-1]   # (H, W) final quarter
    quarter  = d['quarters'][-1]
    bbox, H, W = d['bbox'], d['H'], d['W']

    # Use dw_cube for richer trajectory strings if available
    cube  = d['dw_cube'] if d['dw_cube'] is not None else d['label_cube']
    names = DW_NAMES     if d['dw_cube'] is not None else LABEL_NAMES

    sites_grading     = stratified_sample(label == 1, args.n, args.seed)
    sites_constructed = stratified_sample(label == 2, args.n, args.seed)

    prefix = '[DRY RUN] ' if args.dry_run else ''
    print(
        f"{prefix}=== {args.aoi} spot-check — {quarter}  "
        f"({len(sites_grading)} grading, {len(sites_constructed)} constructed sampled) ==="
    )

    idx = 1

    if sites_grading:
        print('\n--- GRADING (label=1) ---')
        for y, x in sites_grading:
            lon, lat = pixel_to_lonlat(y, x, bbox, H, W)
            traj = dw_trajectory(cube, y, x, names)
            print(format_row(idx, 'grading', lat, lon, traj, earth_url(lat, lon)))
            idx += 1
    else:
        print('\n[validate_labels] WARNING: no grading pixels (label=1) in final quarter.')

    if sites_constructed:
        print('\n--- CONSTRUCTED (label=2) ---')
        for y, x in sites_constructed:
            lon, lat = pixel_to_lonlat(y, x, bbox, H, W)
            traj = dw_trajectory(cube, y, x, names)
            print(format_row(idx, 'constructed', lat, lon, traj, earth_url(lat, lon)))
            idx += 1
    else:
        print('\n[validate_labels] WARNING: no constructed pixels (label=2) in final quarter.')


if __name__ == '__main__':
    main()
