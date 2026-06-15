"""
Evaluation script for the trained construction-detection model.

Produces:
  (a) Per-timepoint metrics on wendell (held-out transfer)
  (b) Progression / trajectory maps (predicted class timeline per pixel)
  (c) Non-circularity / early-detection count (model predicts before DW label)
  (d) Two-tier grading eval (quantitative for confirmed trajectories; qualitative only
      for active-grading pixels — no precision number is possible, stated explicitly)
  (e) Failure demos on desert + austin

Usage:
    python train/evaluate.py                      # uses checkpoints/best.pt
    python train/evaluate.py --smoke-test         # fast run on sunterra
    python train/evaluate.py --ckpt path/to.pt --device cuda
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset import (
    load_aoi,
    patch_positions,
    quarter_to_doy,
    IGNORE_LABEL,
    PATCH_SIZE,
    EVAL_STRIDE,
    T_MIN,
    K,
)
from train.model import load_model
from train.train import compute_per_class_metrics, print_metrics, set_seed

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('[eval] WARNING: matplotlib not available — figures will be skipped.')

# ── constants ──────────────────────────────────────────────────────────────────
DATA_ROOT  = '../data'
OUTPUT_DIR = 'outputs'
CKPT_DIR   = 'checkpoints'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

VAL_AOIS          = ['wendell']
FAILURE_DEMO_AOIS = ['desert', 'austin']

NUM_CLASSES = 3
CLASS_NAMES = {0: 'baseline', 1: 'grading', 2: 'constructed'}

# RGB colours for prediction maps
CLASS_COLORS = np.array([
    [0,   180,  0],    # 0 baseline  → green
    [255, 165,  0],    # 1 grading   → orange
    [200,   0,  0],    # 2 built     → red
    [50,   50, 50],    # 255 ignore  → dark grey
], dtype=np.uint8)


def _class_to_rgb(arr: np.ndarray) -> np.ndarray:
    """(H, W) uint8 → (H, W, 3) RGB using CLASS_COLORS."""
    mapped = arr.copy().astype(np.int32)
    mapped[mapped == IGNORE_LABEL] = 3
    mapped = np.clip(mapped, 0, 3)
    return CLASS_COLORS[mapped]


# ── tiled inference ────────────────────────────────────────────────────────────

def tile_inference_at_t(
    model: torch.nn.Module,
    spectral_cube: np.ndarray,    # (T, 6, H, W) float
    nan_flag: np.ndarray,         # (T, H, W) bool  (unused here but kept for API compat)
    dates: np.ndarray,            # (T,) float32
    t: int,
    norm_stats: Dict,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    stride: int = EVAL_STRIDE,
) -> np.ndarray:
    """
    Tile inference for a single target timepoint t using the fixed K-frame window.
    Input: spectral[t-K:t] (K consecutive frames, all causally valid).
    NaN pixels are filled with band mean; the window length stays fixed at K.
    Skips patches where t < K. Returns (H, W) predicted class map.
    Overlapping logits are averaged before argmax.
    """
    T, C, H, W = spectral_cube.shape
    assert t < T, f'Target t={t} out of range for T={T}'

    if t < K:
        return np.full((H, W), IGNORE_LABEL, dtype=np.uint8)

    mean       = np.array(norm_stats['mean'], dtype=np.float32)
    std        = np.array(norm_stats['std'],  dtype=np.float32)
    std        = np.where(std < 1e-8, 1.0, std)
    data_scale = float(norm_stats.get('data_scale', 1.0))

    logit_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.int32)

    positions = patch_positions(H, W, patch_size, stride)

    for (r, c) in positions:
        P = patch_size

        # Fixed K-frame causal window: spectral[t-K:t]
        spectral = spectral_cube[t-K:t, :, r:r+P, c:c+P].astype(np.float32)  # (K,6,P,P)

        # Scale then fill NaN pixels with band mean, then standardize.
        if data_scale != 1.0:
            spectral = spectral * data_scale
        for b in range(6):
            band = spectral[:, b, :, :]
            spectral[:, b, :, :] = np.where(np.isnan(band), mean[b], band)
        spectral = (spectral - mean[None, :, None, None]) / std[None, :, None, None]

        date_patch = dates[t-K:t]   # (K,)

        s_t = torch.from_numpy(spectral).unsqueeze(0).to(device)    # (1, K, 6, P, P)
        d_t = torch.from_numpy(date_patch).unsqueeze(0).to(device)  # (1, K)

        with torch.no_grad():
            logits = model(s_t, temporal_coords=d_t)   # (1, C, P, P)
        logits_np = logits[0].cpu().float().numpy()    # (C, P, P)

        logit_sum[:, r:r+P, c:c+P] += logits_np
        count_map[r:r+P, c:c+P]    += 1

    valid = count_map > 0
    pred  = np.full((H, W), IGNORE_LABEL, dtype=np.uint8)
    avg   = logit_sum[:, valid] / count_map[valid][None, :]
    pred[valid] = avg.argmax(axis=0).astype(np.uint8)

    return pred


def run_tiled_inference(
    model: torch.nn.Module,
    aoi_data: Dict,
    norm_stats: Dict,
    device: torch.device,
    t_start: int = T_MIN,
    patch_size: int = PATCH_SIZE,
    stride: int = EVAL_STRIDE,
) -> np.ndarray:
    """
    Run causal inference at every timepoint t ∈ [t_start .. T-1].
    Returns pred_cube (T, H, W) uint8 (IGNORE_LABEL where inference skipped).
    """
    spectral = aoi_data['spectral'][:]    # load fully: (T, 6, H, W)
    nan_flag = aoi_data['nan_flag']
    dates    = aoi_data['dates']
    T, _, H, W = spectral.shape

    pred_cube = np.full((T, H, W), IGNORE_LABEL, dtype=np.uint8)

    for t in range(t_start, T):
        print(f'[eval]   inference at t={t}/{T-1}...', end='\r', flush=True)
        pred_cube[t] = tile_inference_at_t(
            model, spectral, nan_flag, dates, t, norm_stats, device,
            patch_size, stride,
        )
    print()
    return pred_cube


# ── per-timepoint metrics ──────────────────────────────────────────────────────

def per_timepoint_metrics(
    pred_cube: np.ndarray,    # (T, H, W) uint8
    label_cube: np.ndarray,   # (T, H, W) uint8
    t_start: int,
) -> List[Dict]:
    """Compute per-class P/R/F1/IoU + confusion matrix at each timepoint t."""
    T = pred_cube.shape[0]
    results = []

    for t in range(t_start, T):
        preds  = pred_cube[t].ravel().astype(np.int32)
        labels = label_cube[t].ravel().astype(np.int32)
        mask   = labels != IGNORE_LABEL

        preds_valid  = preds[mask]
        labels_valid = labels[mask]

        # Per-class metrics
        class_metrics = compute_per_class_metrics(preds_valid, labels_valid)

        # Confusion matrix (3×3, ignoring 255)
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for true_c in range(NUM_CLASSES):
            for pred_c in range(NUM_CLASSES):
                confusion[true_c, pred_c] = int(np.sum(
                    (labels_valid == true_c) & (preds_valid == pred_c)
                ))

        results.append({
            't':           t,
            'n_valid_px':  int(mask.sum()),
            'metrics':     class_metrics,
            'confusion':   confusion.tolist(),
        })

    return results


# ── (b) trajectory / progression figures ──────────────────────────────────────

def save_trajectory_maps(
    pred_cube: np.ndarray,    # (T, H, W)
    label_cube: np.ndarray,
    quarters: List[str],
    aoi_name: str,
    figures_dir: str,
    t_start: int,
):
    """Save side-by-side predicted vs DW label maps for each timepoint."""
    if not HAS_MPL:
        return
    T = pred_cube.shape[0]
    n_t = T - t_start
    if n_t == 0:
        return

    cols = min(n_t, 6)
    rows = ((n_t - 1) // cols + 1) * 2   # predicted + label rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]
    axes = np.array(axes).reshape(rows, cols)

    for i, t in enumerate(range(t_start, T)):
        col_i = i % cols
        row_pred  = (i // cols) * 2
        row_label = row_pred + 1

        axes[row_pred,  col_i].imshow(_class_to_rgb(pred_cube[t]))
        axes[row_label, col_i].imshow(_class_to_rgb(label_cube[t]))
        axes[row_pred,  col_i].set_title(f'Pred  {quarters[t]}', fontsize=7)
        axes[row_label, col_i].set_title(f'Label {quarters[t]}', fontsize=7)
        for ax in (axes[row_pred, col_i], axes[row_label, col_i]):
            ax.axis('off')

    # Hide unused axes
    for i in range(n_t, rows // 2 * cols):
        col_i = i % cols
        for row in range(rows):
            axes[row, col_i].axis('off')

    from matplotlib.patches import Patch
    legend = [Patch(color=c/255., label=l) for c, l in [
        ([0, 180, 0],   'baseline'),
        ([255, 165, 0], 'grading'),
        ([200, 0, 0],   'constructed'),
        ([50, 50, 50],  'ignore'),
    ]]
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=8)
    fig.suptitle(f'{aoi_name} — Predicted (top) vs DW Label (bottom)', fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    path = os.path.join(figures_dir, f'{aoi_name}_trajectory_maps.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] Saved trajectory maps → {path}')


def save_sample_pixel_trajectories(
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    quarters: List[str],
    aoi_name: str,
    figures_dir: str,
    t_start: int,
    n_pixels: int = 5,
):
    """Plot predicted vs DW label class over time for N sample pixels."""
    if not HAS_MPL:
        return
    T, H, W = label_cube.shape

    # Select pixels that have confirmed transitions (0→1→2)
    has_grading = np.any(label_cube == 1, axis=0)
    has_built   = np.any(label_cube == 2, axis=0)
    candidate   = has_grading & has_built
    idxs        = np.argwhere(candidate)

    if len(idxs) == 0:
        idxs = np.argwhere(has_built)
    if len(idxs) == 0:
        print(f'[eval] {aoi_name}: no confirmed-transition pixels for trajectory plot.')
        return

    rng = np.random.default_rng(42)
    chosen = idxs[rng.choice(len(idxs), min(n_pixels, len(idxs)), replace=False)]

    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2.5 * len(chosen)), squeeze=False)
    t_range = list(range(t_start, T))
    q_labels = [quarters[t] for t in t_range]

    for ax, (h, w) in zip(axes[:, 0], chosen):
        pred_traj  = [int(pred_cube[t, h, w])  for t in t_range]
        label_traj = [int(label_cube[t, h, w]) for t in t_range]
        ax.step(range(len(t_range)), pred_traj,  where='post', color='blue',  label='model', lw=2)
        ax.step(range(len(t_range)), label_traj, where='post', color='grey', label='DW label',
                lw=1.5, linestyle='--')
        ax.set_xticks(range(len(t_range)))
        ax.set_xticklabels(q_labels, rotation=45, ha='right', fontsize=6)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['baseline', 'grading', 'built'])
        ax.set_title(f'Pixel ({h},{w})', fontsize=8)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_ylim(-0.2, 2.5)

    fig.suptitle(f'{aoi_name} — Sample pixel trajectories', fontsize=10)
    plt.tight_layout()
    path = os.path.join(figures_dir, f'{aoi_name}_pixel_trajectories.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] Saved pixel trajectories → {path}')


def save_per_timepoint_curve(
    timepoint_results: List[Dict],
    aoi_name: str,
    figures_dir: str,
):
    """Plot F1 / IoU per class vs timepoint t (accuracy-vs-history-length curve)."""
    if not HAS_MPL:
        return
    ts    = [r['t'] for r in timepoint_results]
    colors = {'0': 'green', '1': 'orange', '2': 'red'}
    names  = {'0': 'baseline', '1': 'grading', '2': 'constructed'}

    fig, (ax_f1, ax_iou) = plt.subplots(1, 2, figsize=(12, 4))
    for c_str in ['0', '1', '2']:
        f1s  = [r['metrics'].get(int(c_str), {}).get('f1',  0.0) for r in timepoint_results]
        ious = [r['metrics'].get(int(c_str), {}).get('iou', 0.0) for r in timepoint_results]
        ax_f1.plot(ts,  f1s,  marker='o', color=colors[c_str], label=names[c_str], ms=4)
        ax_iou.plot(ts, ious, marker='o', color=colors[c_str], label=names[c_str], ms=4)

    for ax, ylabel in [(ax_f1, 'F1'), (ax_iou, 'IoU')]:
        ax.set_xlabel('Timepoint t (quarters of past-only context)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

    ax_f1.set_title(f'{aoi_name} — F1 vs history length')
    ax_iou.set_title(f'{aoi_name} — IoU vs history length')
    plt.tight_layout()

    path = os.path.join(figures_dir, f'{aoi_name}_per_timepoint_curve.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] Saved per-timepoint curve → {path}')


# ── (c) early detection ────────────────────────────────────────────────────────

def compute_early_detection(
    pred_cube: np.ndarray,    # (T, H, W)
    label_cube: np.ndarray,
    t_start: int,
) -> Dict:
    """
    Find pixels where the model predicts construction (1 or 2) before the DW
    label reaches construction, AND the pixel is confirmed construction by
    its full trajectory.

    Returns count + pixel fraction for reporting.
    """
    T, H, W = label_cube.shape

    # Confirmed construction pixels: label eventually reaches 2
    ever_built = np.any(label_cube == 2, axis=0)   # (H, W) bool

    early_count = 0
    confirmed_count = int(ever_built.sum())

    for t in range(t_start, T):
        pred  = pred_cube[t]
        label = label_cube[t]

        # Model predicts construction (grading=1 or built=2), DW still baseline (0)
        model_detects = (pred == 1) | (pred == 2)
        dw_baseline   = label == 0
        confirmed     = ever_built

        early_this_t = int(np.sum(model_detects & dw_baseline & confirmed))
        early_count += early_this_t

    return {
        'early_detection_pixel_timesteps': early_count,
        'confirmed_construction_pixels':   confirmed_count,
        'early_fraction': early_count / max(confirmed_count, 1),
        'description': (
            'Count of (pixel, t) instances where the model predicts grading/built '
            'but DW label is still baseline, for pixels confirmed as construction '
            'in their full trajectory. Non-circular detection beyond the label source.'
        ),
    }


def save_early_detection_examples(
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    quarters: List[str],
    aoi_name: str,
    figures_dir: str,
    t_start: int,
    n_examples: int = 5,
):
    """Show trajectory plots highlighting timesteps where model leads DW."""
    if not HAS_MPL:
        return
    T, H, W = label_cube.shape
    ever_built = np.any(label_cube == 2, axis=0)

    # Find pixels with at least one early-detection timestep
    candidates = []
    for t in range(t_start, T):
        early_mask = ((pred_cube[t] == 1) | (pred_cube[t] == 2)) & (label_cube[t] == 0) & ever_built
        hits = np.argwhere(early_mask)
        for h, w in hits:
            candidates.append((h, w))
        if len(candidates) >= n_examples * 5:
            break

    if not candidates:
        print(f'[eval] {aoi_name}: no early-detection pixels found.')
        return

    rng = np.random.default_rng(42)
    seen = set()
    unique = []
    for hw in candidates:
        if hw not in seen:
            seen.add(hw)
            unique.append(hw)
    chosen = unique[:n_examples]

    t_range   = list(range(t_start, T))
    q_labels  = [quarters[t] for t in t_range]

    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2.5 * len(chosen)), squeeze=False)
    for ax, (h, w) in zip(axes[:, 0], chosen):
        pred_traj  = [int(pred_cube[t, h, w])  for t in t_range]
        label_traj = [int(label_cube[t, h, w]) for t in t_range]
        # Highlight early-detection steps
        early_steps = [
            i for i, t in enumerate(t_range)
            if (pred_cube[t, h, w] in (1, 2)) and (label_cube[t, h, w] == 0)
        ]
        ax.step(range(len(t_range)), pred_traj,  where='post', color='blue',  label='model', lw=2)
        ax.step(range(len(t_range)), label_traj, where='post', color='grey', label='DW label',
                lw=1.5, linestyle='--')
        for s in early_steps:
            ax.axvspan(s, s + 1, alpha=0.25, color='yellow', label='_nolegend_')
        if early_steps:
            ax.axvspan(early_steps[0], early_steps[0] + 1, alpha=0.25, color='yellow',
                       label='early detection')
        ax.set_xticks(range(len(t_range)))
        ax.set_xticklabels(q_labels, rotation=45, ha='right', fontsize=6)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['baseline', 'grading', 'built'])
        ax.set_title(f'Pixel ({h},{w}) — yellow=model leads DW', fontsize=8)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_ylim(-0.2, 2.5)

    fig.suptitle(f'{aoi_name} — Early detection examples (model ahead of DW label)', fontsize=10)
    plt.tight_layout()
    path = os.path.join(figures_dir, f'{aoi_name}_early_detection.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] Saved early-detection examples → {path}')


# ── (d) two-tier grading eval ──────────────────────────────────────────────────

def two_tier_grading_eval(
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    quarters: List[str],
    aoi_name: str,
    figures_dir: str,
    t_start: int,
) -> Dict:
    """
    Tier 1 (quantitative): pixels with a confirmed veg→bare→built trajectory.
      - Metrics on these pixels only, per timepoint.

    Tier 2 (qualitative, NO precision): pixels with current label=1 (grading)
      that have NOT yet reached label=2 in their full trajectory.
      - Map only, with explicit disclaimer.
    """
    T, H, W = label_cube.shape
    ever_built = np.any(label_cube == 2, axis=0)
    ever_bare  = np.any(label_cube == 1, axis=0)

    # Confirmed-trajectory mask: ever reached grading AND eventually built
    confirmed_mask = ever_bare & ever_built   # (H, W)

    tier1_results = []
    for t in range(t_start, T):
        preds  = pred_cube[t]
        labels = label_cube[t]
        mask   = (labels != IGNORE_LABEL) & confirmed_mask

        if mask.sum() == 0:
            continue

        m = compute_per_class_metrics(preds[mask].ravel(), labels[mask].ravel())
        tier1_results.append({'t': t, 'metrics': m, 'n_pixels': int(mask.sum())})

    # Tier 2 map: currently grading (label=1), never confirmed built
    active_grading_maps = {}
    if HAS_MPL:
        # Save one map at the latest t with significant active grading
        for t in range(T - 1, t_start - 1, -1):
            active = (label_cube[t] == 1) & ~ever_built
            if active.sum() > 50:
                fig, ax = plt.subplots(figsize=(6, 5))
                disp = np.zeros((H, W, 3), dtype=np.uint8)
                disp[pred_cube[t] == 0] = [0, 180, 0]
                disp[pred_cube[t] == 1] = [255, 165, 0]
                disp[pred_cube[t] == 2] = [200, 0, 0]
                ax.imshow(disp)
                # Overlay active-grading pixels
                ay, ax_ = np.where(active)
                ax.scatter(ax_, ay, c='cyan', s=1, marker='.', label='active grading')
                ax.set_title(
                    f'{aoi_name} t={t} ({quarters[t]})\n'
                    'ACTIVE GRADING (cyan) — No Ground Truth Available\n'
                    'NO PRECISION NUMBER IS REPORTED FOR THESE PIXELS',
                    fontsize=8, color='darkred'
                )
                ax.legend(fontsize=7)
                ax.axis('off')
                path = os.path.join(figures_dir, f'{aoi_name}_active_grading_t{t}.png')
                fig.savefig(path, dpi=120, bbox_inches='tight')
                plt.close(fig)
                print(f'[eval] Saved active-grading map (Tier 2) → {path}')
                active_grading_maps[t] = path
                break

    return {
        'tier1_confirmed_trajectory': tier1_results,
        'tier2_active_grading_maps':  active_grading_maps,
        'tier2_note': (
            'Active-grading pixels (label=1 at time t, no confirmed built outcome) '
            'are shown on the map ONLY. No precision/recall is computed — '
            'these pixels have no verified ground truth.'
        ),
    }


# ── (e) failure demos ──────────────────────────────────────────────────────────

def resolve_failure_aoi_dir(aoi_name: str, data_root: str) -> Optional[str]:
    """Try data/{aoi}/ then {aoi}_data/ at project root."""
    standard = os.path.join(data_root, aoi_name)
    if os.path.exists(os.path.join(standard, 'spectral_cube.npy')):
        return standard
    # Fallback: {aoi}_data/ at project root (one level up from data_root)
    project_root = os.path.dirname(os.path.abspath(data_root))
    alt = os.path.join(project_root, f'{aoi_name}_data')
    if os.path.exists(os.path.join(alt, 'spectral_cube.npy')):
        return alt
    return None


def run_failure_demo(
    aoi_name: str,
    data_root: str,
    model: torch.nn.Module,
    norm_stats: Dict,
    device: torch.device,
    figures_dir: str,
    wendell_metrics: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Run inference on a failure-demo AOI (OOD climate/geography).
    Reports at the latest available t. Compares to wendell baseline if provided.
    """
    aoi_dir = resolve_failure_aoi_dir(aoi_name, data_root)
    if aoi_dir is None:
        print(f'[eval] {aoi_name}: no data found (checked data/{aoi_name}/ and {aoi_name}_data/). Skipping.')
        return None

    # Load from resolved dir (may differ from standard layout)
    data = load_aoi(aoi_name, aoi_dir.replace(f'/{aoi_name}', ''))
    if data is None:
        # Fallback: load directly if dir isn't under data_root
        import importlib
        ds_mod = importlib.import_module('train.dataset')
        data = ds_mod.load_aoi.__wrapped__(aoi_name, aoi_dir) if hasattr(
            ds_mod.load_aoi, '__wrapped__') else None
        if data is None:
            # Direct load
            import json as _json
            sp = os.path.join(aoi_dir, 'spectral_cube.npy')
            lb = os.path.join(aoi_dir, 'label_cube.npy')
            mt = os.path.join(aoi_dir, 'metadata.json')
            if not all(os.path.exists(p) for p in [sp, lb, mt]):
                print(f'[eval] {aoi_name}: incomplete data in {aoi_dir}. Skipping.')
                return None
            with open(mt) as f:
                meta = _json.load(f)
            quarters = meta['quarters']
            T = len(quarters)
            spectral = np.load(sp, mmap_mode='r')
            labels   = np.load(lb, mmap_mode='r')
            dates    = np.array([quarter_to_doy(q) for q in quarters], dtype=np.float32)
            nan_flag = np.any(np.isnan(np.array(spectral)), axis=1)
            H, W     = spectral.shape[2], spectral.shape[3]
            data = {'spectral': spectral, 'labels': labels, 'dates': dates,
                    'nan_flag': nan_flag, 'quarters': quarters, 'T': T, 'H': H, 'W': W}

    if data is None:
        return None

    T = data['T']
    # Use second-to-last t to ensure at least T_MIN input frames
    t_demo = max(T_MIN, T - 2)
    quarters = data['quarters']

    print(f'[eval] {aoi_name}: running failure demo at t={t_demo} ({quarters[t_demo]})...')

    spectral = data['spectral'][:]
    nan_flag = data['nan_flag']
    dates    = data['dates']

    pred = tile_inference_at_t(
        model, spectral, nan_flag, dates, t_demo, norm_stats, device,
    )
    label = data['labels'][t_demo]

    # Metrics (if labels exist)
    mask   = label != IGNORE_LABEL
    metrics = None
    if mask.sum() > 0:
        valid_classes = np.unique(label[mask])
        if len(valid_classes) > 0:
            metrics = compute_per_class_metrics(pred[mask].ravel(), label[mask].ravel())

    # Figure: predicted map
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(_class_to_rgb(pred))
        axes[0].set_title(f'{aoi_name} predicted t={t_demo}\n({quarters[t_demo]})', fontsize=9)
        axes[0].axis('off')
        axes[1].imshow(_class_to_rgb(label))
        axes[1].set_title(f'{aoi_name} DW label t={t_demo}', fontsize=9)
        axes[1].axis('off')

        if wendell_metrics and metrics:
            # Annotate degradation
            w_f1 = wendell_metrics.get(1, {}).get('f1', float('nan'))
            d_f1 = metrics.get(1, {}).get('f1', float('nan'))
            fig.text(0.5, 0.01,
                     f'Grading F1: wendell={w_f1:.3f}  {aoi_name}={d_f1:.3f}  '
                     f'Δ={d_f1 - w_f1:+.3f}',
                     ha='center', fontsize=9, color='darkred')

        path = os.path.join(figures_dir, f'{aoi_name}_failure_demo_t{t_demo}.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'[eval] Saved failure-demo map → {path}')

    return {
        'aoi':     aoi_name,
        't_demo':  t_demo,
        'quarter': quarters[t_demo],
        'metrics': metrics,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate causal construction-detection model')
    p.add_argument('--ckpt',       default=os.path.join(CKPT_DIR, 'best.pt'))
    p.add_argument('--device',     default=os.environ.get('DEVICE', 'cpu'))
    p.add_argument('--data-root',  default=DATA_ROOT)
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    p.add_argument('--smoke-test', action='store_true',
                   help='Fast run: use sunterra, fewer timepoints')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── load checkpoint ────────────────────────────────────────────────────────
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f'Checkpoint not found: {args.ckpt}\n'
            'Run train/train.py first to produce a checkpoint.'
        )

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f'[eval] Loaded checkpoint from {args.ckpt} (epoch={ckpt["epoch"]}, '
          f'val_loss={ckpt["val_loss"]:.4f})')

    # ── norm stats ─────────────────────────────────────────────────────────────
    norm_stats_path = ckpt.get('norm_stats_path', os.path.join(args.output_dir, 'norm_stats.json'))
    if not os.path.exists(norm_stats_path):
        norm_stats_path = os.path.join(args.output_dir, 'norm_stats.json')
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    print(f'[eval] Using norm stats from {norm_stats_path}')

    # ── rebuild model ──────────────────────────────────────────────────────────
    smoke_test = args.smoke_test
    model = load_model(
        num_frames_max=K,
        num_classes=NUM_CLASSES,
        device=args.device,
        smoke_test=smoke_test,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    eval_results: Dict = {
        'checkpoint': args.ckpt,
        'epoch':      ckpt['epoch'],
        'val_loss':   ckpt['val_loss'],
    }

    # ── held-out AOIs (or smoke-test on sunterra) ──────────────────────────────
    eval_aois = ['sunterra'] if smoke_test else VAL_AOIS

    wendell_grading_metrics = None   # for failure-demo comparison

    for aoi_name in eval_aois:
        print(f'\n[eval] ════ {aoi_name.upper()} ════')
        data = load_aoi(aoi_name, args.data_root)
        if data is None:
            print(f'[eval] {aoi_name}: skipping (no data).')
            continue

        T        = data['T']
        quarters = data['quarters']
        label_cube = data['labels'][:]   # fully load for metrics

        # ── (a) tiled inference at every t ─────────────────────────────────
        print(f'[eval] Running causal inference at each t (T={T})...')
        pred_cube = run_tiled_inference(
            model, data, norm_stats, device, t_start=T_MIN,
        )

        # ── per-timepoint metrics ───────────────────────────────────────────
        tp_results = per_timepoint_metrics(pred_cube, label_cube, t_start=T_MIN)
        print(f'[eval] Per-timepoint metrics ({aoi_name}):')
        for r in tp_results:
            print(f'  t={r["t"]} ({quarters[r["t"]]}):')
            print_metrics(r['metrics'], prefix='    ')

        eval_results[aoi_name] = {
            'per_timepoint': tp_results,
        }

        if aoi_name == 'wendell' and tp_results:
            # Use latest timepoint's grading metrics as baseline for failure demo comparison
            wendell_grading_metrics = tp_results[-1]['metrics']

        # ── (b) trajectory / progression figures ───────────────────────────
        save_trajectory_maps(pred_cube, label_cube, quarters, aoi_name, FIGURES_DIR, T_MIN)
        save_per_timepoint_curve(tp_results, aoi_name, FIGURES_DIR)
        save_sample_pixel_trajectories(
            pred_cube, label_cube, quarters, aoi_name, FIGURES_DIR, T_MIN
        )

        # ── (c) early detection ─────────────────────────────────────────────
        early = compute_early_detection(pred_cube, label_cube, t_start=T_MIN)
        print(f'\n[eval] Early detection ({aoi_name}):')
        print(f'  Early (pixel,t) instances: {early["early_detection_pixel_timesteps"]}')
        print(f'  Confirmed-construction px:  {early["confirmed_construction_pixels"]}')
        print(f'  Early fraction:             {early["early_fraction"]:.3f}')
        eval_results[aoi_name]['early_detection'] = early
        save_early_detection_examples(pred_cube, label_cube, quarters, aoi_name, FIGURES_DIR, T_MIN)

        # ── (d) two-tier grading eval ───────────────────────────────────────
        tier = two_tier_grading_eval(pred_cube, label_cube, quarters, aoi_name, FIGURES_DIR, T_MIN)
        print(f'\n[eval] Two-tier grading eval ({aoi_name}):')
        print(f'  Tier 1 confirmed-trajectory timepoints: {len(tier["tier1_confirmed_trajectory"])}')
        print(f'  {tier["tier2_note"]}')
        eval_results[aoi_name]['two_tier_grading'] = {
            'tier1_n_timepoints': len(tier['tier1_confirmed_trajectory']),
            'tier2_note':         tier['tier2_note'],
        }

    # ── (e) failure demos ──────────────────────────────────────────────────────
    if not smoke_test:
        eval_results['failure_demos'] = {}
        for aoi_name in FAILURE_DEMO_AOIS:
            print(f'\n[eval] ════ FAILURE DEMO: {aoi_name.upper()} ════')
            result = run_failure_demo(
                aoi_name, args.data_root, model, norm_stats, device,
                FIGURES_DIR, wendell_metrics=wendell_grading_metrics,
            )
            if result is not None:
                print(f'[eval] {aoi_name} failure-demo metrics:')
                if result['metrics']:
                    print_metrics(result['metrics'], prefix='  ')
                eval_results['failure_demos'][aoi_name] = result

    # ── save results ───────────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, 'eval_results.json')

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

    with open(results_path, 'w') as f:
        json.dump(_serialise(eval_results), f, indent=2)
    print(f'\n[eval] Results saved → {results_path}')

    if smoke_test:
        print('[eval] Smoke test evaluation completed successfully.')


if __name__ == '__main__':
    main()
