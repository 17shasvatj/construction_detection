"""
Build a one-page reviewer demo for a single AOI at a given timepoint.

Writes to outputs/demo/{aoi}/:
  map.png      3-panel figure: model prediction | DW labels | disagreement
  metrics.json per-pixel P/R vs DW labels + placeholder Google Earth tally
  demo.md      markdown embedding map + metrics + spot-check site list

Prints the ~n predicted-construction spot-check sites to stdout.
Site IDs in stdout match pin numbers on map.png.

Usage:
    python make_demo.py {aoi} [--pred data/{aoi}/predictions.npy]
                               [--t -1] [--n 20] [--seed 42]
                               [--data-root data] [--out-dir outputs/demo]
                               [--dry-run]

Recommended AOI: wendell (held-out from training). Running on a train AOI
(babcock, sunterra) makes vs_dw_labels measure agreement with the training
signal — not an independent metric. See metrics.json aoi_role + _note.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle


from spot_check import (
    load_aoi, pixel_to_lonlat, earth_url, dw_trajectory,
    stratified_sample, format_row, DW_NAMES, LABEL_NAMES,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _percentile_rgb(spectral_t):
    """
    spectral_t: (6, H, W) float64, bands [B02,B03,B04,B08,B11,B12].
    Returns (H, W, 3) uint8 RGB (R=B04 idx2, G=B03 idx1, B=B02 idx0),
    2nd–98th percentile stretch per band.
    """
    rgb_bands = spectral_t[[2, 1, 0]]   # R, G, B
    H, W = rgb_bands.shape[1], rgb_bands.shape[2]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for i, band in enumerate(rgb_bands):
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        if hi > lo:
            scaled = np.clip((band - lo) / (hi - lo) * 255, 0, 255)
            out[..., i] = scaled.astype(np.uint8)
    return out


def _class_overlay(cls_mask, color_rgb, alpha=0.4):
    """
    cls_mask: (H, W) bool.
    Returns (H, W, 4) float RGBA overlay.
    """
    H, W = cls_mask.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[cls_mask, :3] = color_rgb
    overlay[cls_mask, 3]  = alpha
    return overlay


def _compute_metrics(pred, label):
    """
    Per-pixel precision/recall/F1 for grading (1) and constructed (2).
    Excludes pixels where label == 255.
    """
    valid = label != 255
    out = {}
    for cls, name in [(1, 'grading'), (2, 'constructed')]:
        tp = int(((pred == cls) & (label == cls) & valid).sum())
        fp = int(((pred == cls) & (label != cls) & valid).sum())
        fn = int(((pred != cls) & (label == cls) & valid).sum())
        support = tp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[name] = {'precision': round(p, 4), 'recall': round(r, 4),
                     'f1': round(f, 4), 'support': support}
    return out


def _aoi_role(aoi):
    try:
        from config import AOIS
        return AOIS.get(aoi, {}).get('role', 'unknown')
    except Exception:
        return 'unknown'


# ── Map figure ─────────────────────────────────────────────────────────────────

def _build_map(rgb, pred_t, label_t, all_sites, out_path):
    H, W = rgb.shape[:2]
    pin_r = max(3, max(H, W) // 100)
    pin_fs = max(5, pin_r * 1.8)

    # Color scheme: orange=grading (class 1), cyan=constructed (class 2)
    orange = [1.0, 0.55, 0.0]
    cyan   = [0.0, 0.85, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, cls_map, title in [
        (axes[0], pred_t,  'Model prediction'),
        (axes[1], label_t, 'DW labels'),
    ]:
        ax.imshow(rgb, origin='upper')
        ax.imshow(_class_overlay(cls_map == 1, orange, 0.45), origin='upper')
        ax.imshow(_class_overlay(cls_map == 2, cyan,   0.45), origin='upper')
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # Disagreement panel: magenta where pred ≠ label (both valid)
    axes[2].imshow(rgb, origin='upper')
    valid    = label_t != 255
    disagree = valid & (pred_t != label_t)
    axes[2].imshow(_class_overlay(disagree, [1.0, 0.0, 1.0], 0.65), origin='upper')
    axes[2].set_title('Disagreement (pred ≠ DW label)', fontsize=11)
    axes[2].axis('off')

    # Numbered pins on left panel only; IDs match stdout sample list
    for i, (y, x) in enumerate(all_sites):
        num = i + 1
        circ = Circle((x, y), pin_r, color='white', ec='black', linewidth=1.5, zorder=5)
        axes[0].add_patch(circ)
        axes[0].text(x, y, str(num), ha='center', va='center',
                     fontsize=pin_fs, fontweight='bold', color='black', zorder=6)

    legend_elems = [
        mpatches.Patch(facecolor='orange',  alpha=0.7, label='Grading (1)'),
        mpatches.Patch(facecolor='cyan',    alpha=0.7, label='Constructed (2)'),
        mpatches.Patch(facecolor='magenta', alpha=0.7, label='Disagree (right panel)'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.01), fontsize=9)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Demo markdown ──────────────────────────────────────────────────────────────

def _build_demo_md(aoi, quarter, aoi_role, metrics, sample_lines, out_path, map_rel):
    g = metrics['grading']
    c = metrics['constructed']

    note = {
        'train':       'aoi_role=train: metric measures agreement with training labels — '
                       'not an independent measurement. Use Google Earth tally as ground truth.',
        'val':         'aoi_role=val: model used this AOI for early stopping; DW-label '
                       'agreement is spatially independent but shares the labeling method.',
        'held_out':    'aoi_role=held_out: model never saw these pixels during training. '
                       'Labeling method (DW) is still shared with train set.',
        'validation':  'aoi_role=validation: same note as held_out.',
        'failure_demo':'aoi_role=failure_demo: out-of-distribution AOI; metrics may be low.',
    }.get(aoi_role, f'aoi_role={aoi_role}: see config.py for role definition.')

    lines = [
        f'# {aoi} Construction Detection Demo — {quarter}',
        '',
        f'**AOI role:** {aoi_role}',
        '',
        f'![Map]({map_rel})',
        '',
        '## Metrics vs DW Labels',
        '',
        f'> {note}',
        '',
        '| Class | Precision | Recall | F1 | Support |',
        '|-------|-----------|--------|----|---------|',
        f'| Grading     | {g["precision"]:.3f} | {g["recall"]:.3f} | {g["f1"]:.3f} | {g["support"]} |',
        f'| Constructed | {c["precision"]:.3f} | {c["recall"]:.3f} | {c["f1"]:.3f} | {c["support"]} |',
        '',
        '## Google Earth Spot-Check Sites',
        '',
        '*Visit each URL, verify, then fill in `metrics.json → google_earth_tally`.*',
        '',
        '```',
        *sample_lines,
        '```',
    ]
    out_path.write_text('\n'.join(lines) + '\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build reviewer demo (map.png, metrics.json, demo.md) for one AOI.')
    parser.add_argument('aoi', help='AOI name, e.g. wendell, sunterra, babcock')
    parser.add_argument('--pred', default=None, metavar='PATH',
                        help='Path to predictions.npy (T,H,W) uint8. '
                             'Default: data/{aoi}/predictions.npy')
    parser.add_argument('--t', type=int, default=-1,
                        help='Timepoint index (default -1 = final quarter)')
    parser.add_argument('--n', type=int, default=20, metavar='N',
                        help='Total spot-check sites (split evenly by class, default 20)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-root', default='data', metavar='DIR')
    parser.add_argument('--out-dir', default='outputs/demo', metavar='DIR')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be sampled/written; write nothing.')
    args = parser.parse_args()

    # ── Load AOI data ──────────────────────────────────────────────────────────
    try:
        d = load_aoi(args.aoi, args.data_root)
    except FileNotFoundError as e:
        print(f'[make_demo] ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    # ── Load predictions ───────────────────────────────────────────────────────
    pred_path = Path(args.pred) if args.pred else Path(args.data_root) / args.aoi / 'predictions.npy'
    if not pred_path.exists():
        print(
            f'[make_demo] ERROR: predictions not found at {pred_path}\n'
            f'Run evaluate.py to generate predictions, or pass --pred <path>.',
            file=sys.stderr,
        )
        sys.exit(1)

    predictions = np.load(pred_path)   # (T, H, W) uint8

    # ── Resolve timepoint ──────────────────────────────────────────────────────
    T = d['T']
    t_idx = args.t % T               # handles negative indices
    quarter = d['quarters'][t_idx]
    bbox, H, W = d['bbox'], d['H'], d['W']

    pred_t  = predictions[t_idx]      # (H, W) uint8
    label_t = d['label_cube'][t_idx]  # (H, W) uint8

    # Sampling and map visualization use the same t_idx — they cannot drift apart.
    cube  = d['dw_cube'] if d['dw_cube'] is not None else d['label_cube']
    names = DW_NAMES     if d['dw_cube'] is not None else LABEL_NAMES

    # ── Class-stratified sampling ──────────────────────────────────────────────
    n_grading     = args.n // 2
    n_constructed = args.n - n_grading
    sites_grading     = stratified_sample(pred_t == 1, n_grading,     args.seed)
    sites_constructed = stratified_sample(pred_t == 2, n_constructed, args.seed)
    all_sites = sites_grading + sites_constructed   # grading first, then constructed

    # ── Build sample lines (stdout + demo.md) ─────────────────────────────────
    sample_lines = []
    header_line = (
        f"=== {args.aoi} demo spot-check — {quarter}  "
        f"({len(sites_grading)} grading, {len(sites_constructed)} constructed predicted) ==="
    )
    sample_lines.append(header_line)

    idx = 1
    if sites_grading:
        sample_lines.append('')
        sample_lines.append('--- GRADING (predicted=1) ---')
        for y, x in sites_grading:
            lon, lat = pixel_to_lonlat(y, x, bbox, H, W)
            traj = dw_trajectory(cube, y, x, names)
            sample_lines.append(format_row(idx, 'grading', lat, lon, traj, earth_url(lat, lon)))
            idx += 1

    if sites_constructed:
        sample_lines.append('')
        sample_lines.append('--- CONSTRUCTED (predicted=2) ---')
        for y, x in sites_constructed:
            lon, lat = pixel_to_lonlat(y, x, bbox, H, W)
            traj = dw_trajectory(cube, y, x, names)
            sample_lines.append(format_row(idx, 'constructed', lat, lon, traj, earth_url(lat, lon)))
            idx += 1

    # ── Dry-run: print and exit ────────────────────────────────────────────────
    out_dir  = Path(args.out_dir) / args.aoi
    map_path = out_dir / 'map.png'
    mj_path  = out_dir / 'metrics.json'
    md_path  = out_dir / 'demo.md'

    print('\n'.join(sample_lines))

    if args.dry_run:
        print('\n[DRY RUN] Would write:')
        for p in [map_path, mj_path, md_path]:
            print(f'  {p}')
        return

    # ── Write outputs ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    # map.png
    rgb = _percentile_rgb(np.array(d['spectral_cube'][t_idx]))
    _build_map(rgb, pred_t, np.array(label_t), all_sites, map_path)
    print(f'Wrote: {map_path}')

    # metrics.json
    aoi_role = _aoi_role(args.aoi)
    vs_dw    = _compute_metrics(pred_t, np.array(label_t))
    role_note = {
        'train':       'model was trained on this AOI — metric measures agreement with '
                       'training labels, not independent ground truth',
        'val':         'model used this AOI for early stopping; metric is spatially '
                       'independent but labeling method is shared with train set',
        'held_out':    'model never saw these pixels during training; labeling method '
                       '(DW) is still shared with train set',
        'validation':  'same as held_out — DW labels are spatially independent of training',
        'failure_demo':'out-of-distribution AOI; expect lower metric values',
    }.get(aoi_role, f'see config.py for aoi_role={aoi_role!r} definition')

    metrics = {
        'aoi': args.aoi,
        'aoi_role': aoi_role,
        'quarter': quarter,
        'timepoint_index': t_idx,
        'vs_dw_labels': {
            '_note': (
                f'Per-pixel agreement with DW-derived labels ({role_note}). '
                'Independent ground truth is the Google Earth tally below.'
            ),
            **vs_dw,
        },
        'google_earth_tally': {
            '_note': 'Fill in after manual verification. This is the independent ground truth.',
            'grading':     {'verified': 0, 'false_positive': 0, 'pending': len(sites_grading)},
            'constructed': {'verified': 0, 'false_positive': 0, 'pending': len(sites_constructed)},
        },
    }
    mj_path.write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'Wrote: {mj_path}')

    # demo.md
    _build_demo_md(
        aoi=args.aoi,
        quarter=quarter,
        aoi_role=aoi_role,
        metrics=vs_dw,
        sample_lines=sample_lines,
        out_path=md_path,
        map_rel='map.png',
    )
    print(f'Wrote: {md_path}')


if __name__ == '__main__':
    main()
