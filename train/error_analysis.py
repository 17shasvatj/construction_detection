"""
Error analysis for the trained construction-detection model.

Loads the pred_cube + label_cube from a Wendell eval run and produces four
breakdowns to localize where the model fails:

  1. Spatial pattern — boundary vs interior errors (10m label resolution check)
  2. Temporal pattern — error rate as a function of how much history is available
  3. Per-class confidence — for missed construction pixels, what probability did
     the model assign to construction? (Threshold-rescuable vs feature-limited.)
  4. Spatial heatmap — where on the Wendell tile do errors concentrate?

Usage:
    python train/error_analysis.py
    python train/error_analysis.py --aoi wendell --device cuda
    python train/error_analysis.py --device cuda --ckpt checkpoints/best.pt

Requires the checkpoint and norm_stats from a completed training run.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset import (
    load_aoi,
    patch_positions,
    IGNORE_LABEL,
    PATCH_SIZE,
    EVAL_STRIDE,
    T_MIN,
    K,
)
from train.model import load_model
from train.train import set_seed

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('[error_analysis] WARNING: matplotlib not available — figures will be skipped.')


# ── constants ──────────────────────────────────────────────────────────────────
DATA_ROOT   = '../data'
OUTPUT_DIR  = 'outputs'
CKPT_DIR    = 'checkpoints'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

NUM_CLASSES = 3
CLASS_NAMES = ['baseline', 'grading', 'constructed']


# ── tiled inference returning PROBABILITIES (not argmax) ──────────────────────

def tile_inference_probs_at_t(
    model: torch.nn.Module,
    spectral_cube: np.ndarray,
    dates: np.ndarray,
    t: int,
    norm_stats: Dict,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    stride: int = EVAL_STRIDE,
) -> np.ndarray:
    """
    Tile inference at timepoint t. Returns (C, H, W) softmax probabilities,
    not argmax — needed for the confidence breakdown.
    Returns None-filled probability shape if t < K.
    """
    T, C, H, W = spectral_cube.shape
    if t < K:
        # Return uniform probs as placeholder
        return np.full((NUM_CLASSES, H, W), 1.0 / NUM_CLASSES, dtype=np.float32)

    mean       = np.array(norm_stats['mean'], dtype=np.float32)
    std        = np.array(norm_stats['std'],  dtype=np.float32)
    std        = np.where(std < 1e-8, 1.0, std)
    data_scale = float(norm_stats.get('data_scale', 1.0))

    prob_sum  = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.int32)

    positions = patch_positions(H, W, patch_size, stride)

    for (r, c) in positions:
        P = patch_size

        spectral = spectral_cube[t-K:t, :, r:r+P, c:c+P].astype(np.float32)

        if data_scale != 1.0:
            spectral = spectral * data_scale
        for b in range(6):
            band = spectral[:, b, :, :]
            spectral[:, b, :, :] = np.where(np.isnan(band), mean[b], band)
        spectral = (spectral - mean[None, :, None, None]) / std[None, :, None, None]

        date_patch = dates[t-K:t]

        s_t = torch.from_numpy(spectral).unsqueeze(0).to(device)
        d_t = torch.from_numpy(date_patch).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(s_t, temporal_coords=d_t)        # (1, C, P, P)
            probs  = torch.softmax(logits, dim=1)            # (1, C, P, P)
        probs_np = probs[0].cpu().float().numpy()            # (C, P, P)

        prob_sum[:, r:r+P, c:c+P] += probs_np
        count_map[r:r+P, c:c+P]   += 1

    valid = count_map > 0
    out = np.full((NUM_CLASSES, H, W), 1.0/NUM_CLASSES, dtype=np.float32)
    out[:, valid] = prob_sum[:, valid] / count_map[None, valid]

    return out


# ── analysis #1: boundary vs interior errors ──────────────────────────────────

def boundary_interior_breakdown(
    pred_cube: np.ndarray,    # (T, H, W)
    label_cube: np.ndarray,
    t_start: int = T_MIN,
) -> Dict:
    """
    Decompose constructed→baseline misses into boundary errors vs interior errors.
    "Boundary" = a pixel adjacent to a class transition in the ground truth.
    "Interior" = a pixel where all 8 neighbors share the same ground-truth class.
    If interior errors dominate, the model has a feature-discrimination problem.
    If boundary errors dominate, it's a label-resolution problem (10m DW blocky).
    """
    from scipy.ndimage import binary_dilation

    T, H, W = pred_cube.shape
    results = {'boundary_miss': 0, 'interior_miss': 0,
               'boundary_total': 0, 'interior_total': 0}

    for t in range(t_start, T):
        label = label_cube[t]
        pred  = pred_cube[t]

        # Find boundary pixels: where a 3x3 neighborhood contains >1 class
        constructed_mask = (label == 2)
        if not constructed_mask.any():
            continue
        # A constructed pixel is "boundary" if any of its 8 neighbors is non-constructed
        dilated_non_constructed = binary_dilation(~constructed_mask & (label != IGNORE_LABEL))
        boundary_mask = constructed_mask & dilated_non_constructed
        interior_mask = constructed_mask & ~boundary_mask

        # Misses on each
        missed = constructed_mask & (pred == 0)
        results['boundary_miss']  += int((missed & boundary_mask).sum())
        results['interior_miss']  += int((missed & interior_mask).sum())
        results['boundary_total'] += int(boundary_mask.sum())
        results['interior_total'] += int(interior_mask.sum())

    if results['boundary_total'] > 0:
        results['boundary_miss_rate'] = results['boundary_miss'] / results['boundary_total']
    else:
        results['boundary_miss_rate'] = float('nan')

    if results['interior_total'] > 0:
        results['interior_miss_rate'] = results['interior_miss'] / results['interior_total']
    else:
        results['interior_miss_rate'] = float('nan')

    return results


# ── analysis #2: temporal pattern (already partially in eval, deepen here) ────

def temporal_error_breakdown(
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    t_start: int = T_MIN,
) -> List[Dict]:
    """
    Per-timepoint error breakdown specifically for the constructed class:
    miss rate = (true=2 AND pred=0) / (true=2)
    Tracks how the model improves as more history accumulates.
    """
    T = pred_cube.shape[0]
    results = []
    for t in range(t_start, T):
        label = label_cube[t]
        pred  = pred_cube[t]
        constructed_mask = (label == 2)
        total = int(constructed_mask.sum())
        if total == 0:
            continue
        missed_as_baseline = int(((label == 2) & (pred == 0)).sum())
        confused_grading   = int(((label == 2) & (pred == 1)).sum())
        correct            = int(((label == 2) & (pred == 2)).sum())
        results.append({
            't': t,
            'total_constructed': total,
            'predicted_baseline': missed_as_baseline,
            'predicted_grading':  confused_grading,
            'predicted_correct':  correct,
            'recall':             correct / total,
            'miss_rate':          missed_as_baseline / total,
        })
    return results


# ── analysis #3: confidence on missed construction ────────────────────────────

def confidence_on_missed(
    prob_cube: np.ndarray,    # (T, C, H, W) softmax probs
    label_cube: np.ndarray,
    pred_cube: np.ndarray,
    t_start: int = T_MIN,
) -> Dict:
    """
    For pixels where (true=2, pred=0), what was the model's probability for
    each class? Tells us whether errors are "threshold-rescuable" (model was
    actually 0.4-0.5 confident in class 2 but argmax went to class 0) vs
    "feature-limited" (model was 0.05 on class 2 — no signal).
    """
    T = label_cube.shape[0]
    missed_constructed_probs = {0: [], 1: [], 2: []}
    missed_grading_probs     = {0: [], 1: [], 2: []}

    for t in range(t_start, T):
        if t >= prob_cube.shape[0]:
            continue
        label = label_cube[t]
        pred  = pred_cube[t]
        probs = prob_cube[t]        # (C, H, W)

        # Missed constructed: true=2 AND pred=0
        missed_c_mask = (label == 2) & (pred == 0)
        if missed_c_mask.any():
            for c in range(NUM_CLASSES):
                missed_constructed_probs[c].extend(probs[c][missed_c_mask].tolist())

        # Missed grading: true=1 AND pred=0
        missed_g_mask = (label == 1) & (pred == 0)
        if missed_g_mask.any():
            for c in range(NUM_CLASSES):
                missed_grading_probs[c].extend(probs[c][missed_g_mask].tolist())

    def summarize(d):
        out = {}
        for c, vals in d.items():
            if not vals:
                out[CLASS_NAMES[c]] = {'mean': None, 'median': None, 'p25': None, 'p75': None, 'n': 0}
                continue
            arr = np.array(vals)
            out[CLASS_NAMES[c]] = {
                'mean':   float(arr.mean()),
                'median': float(np.median(arr)),
                'p25':    float(np.percentile(arr, 25)),
                'p75':    float(np.percentile(arr, 75)),
                'n':      int(len(arr)),
            }
        return out

    return {
        'missed_constructed_probs': summarize(missed_constructed_probs),
        'missed_grading_probs':     summarize(missed_grading_probs),
    }


# ── analysis #4: spatial heatmap of errors ───────────────────────────────────

def spatial_error_map(
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    aoi_name: str,
    figures_dir: str,
    t_start: int = T_MIN,
):
    """
    Render per-pixel summed error frequency across all eval timepoints.
    For each pixel, count the number of timesteps where (true=2, pred=0) or
    (true=1, pred=0). Output a heatmap showing where errors concentrate.
    """
    if not HAS_MPL:
        return
    T, H, W = pred_cube.shape
    constructed_miss_count = np.zeros((H, W), dtype=np.int32)
    grading_miss_count     = np.zeros((H, W), dtype=np.int32)
    constructed_total      = np.zeros((H, W), dtype=np.int32)
    grading_total          = np.zeros((H, W), dtype=np.int32)

    for t in range(t_start, T):
        label = label_cube[t]
        pred  = pred_cube[t]
        constructed_miss_count += ((label == 2) & (pred == 0)).astype(np.int32)
        grading_miss_count     += ((label == 1) & (pred == 0)).astype(np.int32)
        constructed_total      += (label == 2).astype(np.int32)
        grading_total          += (label == 1).astype(np.int32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0,0].imshow(constructed_miss_count, cmap='hot', interpolation='nearest')
    axes[0,0].set_title(f'{aoi_name}: Constructed-missed-as-baseline\n(count of timesteps where true=2, pred=0)')
    axes[0,0].axis('off')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046)

    im1 = axes[0,1].imshow(constructed_total, cmap='Blues', interpolation='nearest')
    axes[0,1].set_title(f'{aoi_name}: Constructed total\n(count of timesteps where true=2)')
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1], fraction=0.046)

    im2 = axes[1,0].imshow(grading_miss_count, cmap='hot', interpolation='nearest')
    axes[1,0].set_title(f'{aoi_name}: Grading-missed-as-baseline\n(count of timesteps where true=1, pred=0)')
    axes[1,0].axis('off')
    plt.colorbar(im2, ax=axes[1,0], fraction=0.046)

    im3 = axes[1,1].imshow(grading_total, cmap='Oranges', interpolation='nearest')
    axes[1,1].set_title(f'{aoi_name}: Grading total\n(count of timesteps where true=1)')
    axes[1,1].axis('off')
    plt.colorbar(im3, ax=axes[1,1], fraction=0.046)

    fig.suptitle(f'{aoi_name} — Spatial Error Concentration', fontsize=12)
    plt.tight_layout()

    path = os.path.join(figures_dir, f'{aoi_name}_spatial_error_map.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[error_analysis] Saved spatial error map → {path}')


# ── confidence histogram plot ──────────────────────────────────────────────────

def plot_confidence_histogram(conf_results: Dict, aoi_name: str, figures_dir: str):
    """
    Histogram of model's class-2 probability on pixels where true=2 but pred=0.
    Bimodal at high values would indicate threshold-rescuable errors.
    Concentrated at low values would indicate feature-limited errors.
    """
    if not HAS_MPL:
        return

    # We need the raw probabilities to plot; reconstruct from summary won't work.
    # The summary gives quartiles only. So we just plot mean/median/p25/p75 as bar.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (key, title) in enumerate([
        ('missed_constructed_probs', 'Missed constructed pixels (true=2, pred=0)'),
        ('missed_grading_probs',     'Missed grading pixels (true=1, pred=0)'),
    ]):
        ax = axes[ax_idx]
        data = conf_results[key]
        classes = list(data.keys())
        means = [data[c]['mean'] if data[c]['mean'] is not None else 0 for c in classes]
        p25s  = [data[c]['p25']  if data[c]['p25']  is not None else 0 for c in classes]
        p75s  = [data[c]['p75']  if data[c]['p75']  is not None else 0 for c in classes]
        x = np.arange(len(classes))
        ax.bar(x, means, yerr=[np.array(means)-np.array(p25s), np.array(p75s)-np.array(means)],
               capsize=5, color=['green','orange','red'], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylabel('Mean model probability')
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.axhline(0.33, color='gray', linestyle='--', alpha=0.5, label='uniform prior')
        ax.legend()

    fig.suptitle(f'{aoi_name} — Model probability on missed pixels\n(error bars = P25-P75 range)',
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(figures_dir, f'{aoi_name}_missed_pixel_confidence.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[error_analysis] Saved confidence breakdown → {path}')


# ── temporal plot ─────────────────────────────────────────────────────────────

def plot_temporal_breakdown(temporal_results: List[Dict], quarters: List[str],
                            aoi_name: str, figures_dir: str):
    """Plot constructed recall and miss-rate over time."""
    if not HAS_MPL or not temporal_results:
        return
    ts        = [r['t'] for r in temporal_results]
    recalls   = [r['recall'] for r in temporal_results]
    misses    = [r['miss_rate'] for r in temporal_results]
    q_labels  = [quarters[t] if t < len(quarters) else str(t) for t in ts]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ts, recalls, marker='o', color='green', label='Recall (correct)', linewidth=2)
    ax.plot(ts, misses,  marker='s', color='red',   label='Miss rate (predicted baseline)', linewidth=2)
    ax.set_xticks(ts)
    ax.set_xticklabels(q_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Fraction of true constructed pixels')
    ax.set_xlabel('Timepoint t (causal history of K=6 frames)')
    ax.set_title(f'{aoi_name} — Constructed: recall vs miss-as-baseline by timepoint')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(figures_dir, f'{aoi_name}_constructed_temporal.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[error_analysis] Saved temporal breakdown → {path}')


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Error analysis for trained construction model')
    p.add_argument('--aoi',         default='wendell',
                   help='AOI to analyze (default: wendell, the held-out test region)')
    p.add_argument('--ckpt',        default=os.path.join(CKPT_DIR, 'best.pt'))
    p.add_argument('--device',      default=os.environ.get('DEVICE', 'cpu'))
    p.add_argument('--data-root',   default=DATA_ROOT)
    p.add_argument('--output-dir',  default=OUTPUT_DIR)
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── load checkpoint ────────────────────────────────────────────────────────
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)
    print(f'[error_analysis] Loaded checkpoint epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f}')

    # ── norm stats ─────────────────────────────────────────────────────────────
    norm_stats_path = ckpt.get('norm_stats_path', os.path.join(args.output_dir, 'norm_stats.json'))
    if not os.path.exists(norm_stats_path):
        norm_stats_path = os.path.join(args.output_dir, 'norm_stats.json')
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    # ── load AOI data ──────────────────────────────────────────────────────────
    print(f'[error_analysis] Loading {args.aoi}...')
    data = load_aoi(args.aoi, args.data_root)
    if data is None:
        print(f'[error_analysis] ERROR: could not load {args.aoi}')
        return
    T = data['T']
    quarters = data['quarters']
    label_cube = data['labels'][:]   # (T, H, W) uint8

    # ── rebuild model ──────────────────────────────────────────────────────────
    model = load_model(num_frames_max=K, num_classes=NUM_CLASSES,
                       device=args.device, smoke_test=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ── run inference returning both argmax preds AND probs ────────────────────
    spectral = data['spectral'][:]
    dates    = data['dates']

    print(f'[error_analysis] Running probabilistic inference at each t (T={T})...')
    prob_cube_list = []
    pred_cube_list = []
    for t in range(T):
        if t < T_MIN or t < K:
            prob_cube_list.append(np.full((NUM_CLASSES, data['H'], data['W']),
                                          1.0/NUM_CLASSES, dtype=np.float32))
            pred_cube_list.append(np.full((data['H'], data['W']),
                                          IGNORE_LABEL, dtype=np.uint8))
            continue
        probs = tile_inference_probs_at_t(model, spectral, dates, t,
                                          norm_stats, device)
        preds = probs.argmax(axis=0).astype(np.uint8)
        prob_cube_list.append(probs)
        pred_cube_list.append(preds)
        print(f'  t={t}/{T-1}', end='\r', flush=True)
    print()

    prob_cube = np.stack(prob_cube_list, axis=0)   # (T, C, H, W)
    pred_cube = np.stack(pred_cube_list, axis=0)   # (T, H, W)

    # ── analysis #1: boundary vs interior ──────────────────────────────────────
    print('\n[1] Boundary vs interior errors (constructed missed as baseline):')
    try:
        bi = boundary_interior_breakdown(pred_cube, label_cube)
        print(f'    Boundary pixels missed: {bi["boundary_miss"]:>8,} / {bi["boundary_total"]:>8,}  '
              f'({100*bi["boundary_miss_rate"]:.1f}%)')
        print(f'    Interior pixels missed: {bi["interior_miss"]:>8,} / {bi["interior_total"]:>8,}  '
              f'({100*bi["interior_miss_rate"]:.1f}%)')
        if bi['boundary_miss_rate'] > bi['interior_miss_rate'] * 1.5:
            print('    → Boundary errors dominate. Label-resolution (10m blocky) is the main issue.')
        elif bi['interior_miss_rate'] > bi['boundary_miss_rate'] * 1.5:
            print('    → Interior errors dominate. Feature-discrimination is the main issue.')
        else:
            print('    → Boundary and interior errors are roughly equal — both contribute.')
    except ImportError:
        print('    SciPy not installed — skipping boundary/interior breakdown.')
        bi = None

    # ── analysis #2: temporal ──────────────────────────────────────────────────
    print('\n[2] Temporal pattern (constructed class recall over t):')
    temporal = temporal_error_breakdown(pred_cube, label_cube)
    for r in temporal:
        print(f'    t={r["t"]:>2} ({quarters[r["t"]]:s}): '
              f'true_total={r["total_constructed"]:>6,}  '
              f'correct={r["predicted_correct"]:>5,}  '
              f'missed→baseline={r["predicted_baseline"]:>5,}  '
              f'recall={r["recall"]:.3f}')

    # ── analysis #3: confidence on missed ──────────────────────────────────────
    print('\n[3] Model probability on missed pixels:')
    conf = confidence_on_missed(prob_cube, label_cube, pred_cube)
    print('    Missed constructed pixels (true=2, pred=0):')
    for c, stats in conf['missed_constructed_probs'].items():
        if stats['mean'] is None:
            print(f'      P(class={c}): no data')
            continue
        print(f'      P(class={c}): mean={stats["mean"]:.3f}  '
              f'median={stats["median"]:.3f}  '
              f'P25-P75=[{stats["p25"]:.3f}, {stats["p75"]:.3f}]  n={stats["n"]:,}')
    print('    Missed grading pixels (true=1, pred=0):')
    for c, stats in conf['missed_grading_probs'].items():
        if stats['mean'] is None:
            print(f'      P(class={c}): no data')
            continue
        print(f'      P(class={c}): mean={stats["mean"]:.3f}  '
              f'median={stats["median"]:.3f}  '
              f'P25-P75=[{stats["p25"]:.3f}, {stats["p75"]:.3f}]  n={stats["n"]:,}')

    # Interpretation help
    c2_on_missed = conf['missed_constructed_probs'].get('constructed', {}).get('median')
    if c2_on_missed is not None:
        if c2_on_missed > 0.3:
            print(f'    → P(constructed)={c2_on_missed:.2f} on missed pixels: significant signal exists.')
            print('       Many errors are threshold-rescuable (model nearly fired class 2).')
        elif c2_on_missed < 0.15:
            print(f'    → P(constructed)={c2_on_missed:.2f} on missed pixels: weak signal.')
            print('       Errors are feature-limited (model is genuinely confident in baseline).')
        else:
            print(f'    → P(constructed)={c2_on_missed:.2f} on missed pixels: borderline.')
            print('       Mixed errors — threshold + retraining both could help.')

    # ── analysis #4: spatial heatmap ───────────────────────────────────────────
    print('\n[4] Spatial error concentration:')
    spatial_error_map(pred_cube, label_cube, args.aoi, FIGURES_DIR)

    # ── plots ──────────────────────────────────────────────────────────────────
    plot_temporal_breakdown(temporal, quarters, args.aoi, FIGURES_DIR)
    plot_confidence_histogram(conf, args.aoi, FIGURES_DIR)

    # ── save full results ──────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, 'error_analysis.json')

    def _serialise(obj):
        if isinstance(obj, dict):
            return {str(k): _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        'aoi':                 args.aoi,
        'checkpoint':          args.ckpt,
        'epoch':               ckpt['epoch'],
        'val_loss':            ckpt['val_loss'],
        'boundary_interior':   bi,
        'temporal':            temporal,
        'confidence_on_missed': conf,
    }
    with open(out_path, 'w') as f:
        json.dump(_serialise(results), f, indent=2)
    print(f'\n[error_analysis] Full results saved → {out_path}')


if __name__ == '__main__':
    main()