"""
Evaluation script for the trained construction-detection model.

Produces:
  (a) Per-timepoint metrics on a held-out AOI
  (b) Progression / trajectory maps (predicted vs DW labels per quarter)
  (c) Non-circularity / early-detection count (model predicts before DW label)
  (d) Two-tier grading eval (quantitative for confirmed trajectories; qualitative-only
      for active-grading pixels — no precision number, stated explicitly)
  (e) Failure demos on desert + austin (default run only)
  (f) Summary table of macro-averaged + peak metrics ready to drop into the report
  (g) Spot-check sites — spatially stratified predicted-construction pixels
      with lat/lon for live demo verification (--aoi mode only)

Usage:
    # default: evaluate on VAL_AOIS (wendell), run failure demos
    python train/evaluate.py

    # evaluate on a specific AOI (must have spectral_cube.npy + metadata.json;
    # label_cube.npy optional — metrics skipped if missing)
    python train/evaluate.py --aoi lakewood_ranch
    python train/evaluate.py --aoi wendell --device cuda

    # smoke-test (sunterra, no failure demos)
    python train/evaluate.py --smoke-test
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

CLASS_COLORS = np.array([
    [0,   180,  0],
    [255, 165,  0],
    [200,   0,  0],
    [50,   50, 50],
], dtype=np.uint8)


def _class_to_rgb(arr: np.ndarray) -> np.ndarray:
    mapped = arr.copy().astype(np.int32)
    mapped[mapped == IGNORE_LABEL] = 3
    mapped = np.clip(mapped, 0, 3)
    return CLASS_COLORS[mapped]


# ── tiled inference ────────────────────────────────────────────────────────────

def tile_inference_at_t(
    model: torch.nn.Module,
    spectral_cube: np.ndarray,
    nan_flag: np.ndarray,
    dates: np.ndarray,
    t: int,
    norm_stats: Dict,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    stride: int = EVAL_STRIDE,
) -> np.ndarray:
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
            logits = model(s_t, temporal_coords=d_t)
        logits_np = logits[0].cpu().float().numpy()

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
    spectral = aoi_data['spectral'][:]
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
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    t_start: int,
) -> List[Dict]:
    T = pred_cube.shape[0]
    results = []

    for t in range(t_start, T):
        preds  = pred_cube[t].ravel().astype(np.int32)
        labels = label_cube[t].ravel().astype(np.int32)
        mask   = labels != IGNORE_LABEL

        preds_valid  = preds[mask]
        labels_valid = labels[mask]

        class_metrics = compute_per_class_metrics(preds_valid, labels_valid)

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


# ── summary table for the report ───────────────────────────────────────────────

def print_summary_table(
    tp_results: List[Dict],
    aoi_name: str,
):
    valid = [
        r for r in tp_results
        if any(r['metrics'].get(c, {}).get('f1', 0.0) > 0 or
               r['metrics'].get(c, {}).get('precision', 0.0) > 0 or
               r['metrics'].get(c, {}).get('recall', 0.0) > 0
               for c in [0, 1, 2])
    ]
    if not valid:
        print(f'\n[eval] {aoi_name}: no valid timepoints for summary.')
        return

    print(f'\n{"="*70}')
    print(f'SUMMARY METRICS FOR REPORT — {aoi_name.upper()}')
    print(f'{"="*70}')
    print(f'Eval timepoints: t={valid[0]["t"]} → t={valid[-1]["t"]} '
          f'(n={len(valid)} timepoints)')
    print()

    print(f'MACRO-AVERAGES ACROSS VALID TIMEPOINTS')
    print(f'  {"class":<14s} {"P":>7s} {"R":>7s} {"F1":>7s} {"IoU":>7s} {"n_t":>5s}')
    print(f'  {"-"*14} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*5}')
    for c in [0, 1, 2]:
        rows = [
            r for r in valid
            if r['metrics'].get(c, {}).get('tp', 0) + r['metrics'].get(c, {}).get('fn', 0) > 0
        ]
        if not rows:
            print(f'  {CLASS_NAMES[c]:<14s} {"n/a":>7s} {"n/a":>7s} {"n/a":>7s} {"n/a":>7s} {0:>5d}')
            continue
        ps  = [r['metrics'][c]['precision'] for r in rows]
        rs  = [r['metrics'][c]['recall']    for r in rows]
        f1s = [r['metrics'][c]['f1']        for r in rows]
        ious= [r['metrics'][c]['iou']       for r in rows]
        print(f'  {CLASS_NAMES[c]:<14s} '
              f'{np.mean(ps):>7.3f} {np.mean(rs):>7.3f} '
              f'{np.mean(f1s):>7.3f} {np.mean(ious):>7.3f} '
              f'{len(rows):>5d}')

    print()
    print(f'PEAK F1 PER CLASS (best single timepoint)')
    print(f'  {"class":<14s} {"peak F1":>9s} {"@ t":>5s} {"P":>7s} {"R":>7s}')
    print(f'  {"-"*14} {"-"*9} {"-"*5} {"-"*7} {"-"*7}')
    for c in [0, 1, 2]:
        rows = [r for r in valid if r['metrics'].get(c, {}).get('f1', 0) > 0]
        if not rows:
            print(f'  {CLASS_NAMES[c]:<14s} {"n/a":>9s}')
            continue
        best = max(rows, key=lambda r: r['metrics'][c]['f1'])
        m = best['metrics'][c]
        print(f'  {CLASS_NAMES[c]:<14s} '
              f'{m["f1"]:>9.3f} {best["t"]:>5d} '
              f'{m["precision"]:>7.3f} {m["recall"]:>7.3f}')

    print()
    print(f'CONSTRUCTED RECALL vs HISTORY LENGTH (architecture validation)')
    cons_rows = [r for r in valid if r['metrics'].get(2, {}).get('recall', 0) > 0]
    if cons_rows:
        n = len(cons_rows)
        snapshot_idxs = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
        snapshot_idxs = sorted(set(snapshot_idxs))
        print(f'  {"t":>4s} {"R(constructed)":>16s}')
        for i in snapshot_idxs:
            r = cons_rows[i]
            print(f'  {r["t"]:>4d} {r["metrics"][2]["recall"]:>16.3f}')

    print()
    print(f'FINAL-QUARTER METRICS (persistent classes)')
    last = valid[-1]
    print(f'  t={last["t"]}')
    for c in [0, 2]:
        m = last['metrics'].get(c, {})
        if m.get('precision', 0) + m.get('recall', 0) > 0:
            print(f'  {CLASS_NAMES[c]:<14s} P={m["precision"]:.3f}  '
                  f'R={m["recall"]:.3f}  F1={m["f1"]:.3f}  IoU={m["iou"]:.3f}')

    grad_rows = [r for r in valid
                 if r['metrics'].get(1, {}).get('tp', 0) + r['metrics'].get(1, {}).get('fn', 0) > 100]
    if grad_rows:
        print()
        print(f'MID-TRAJECTORY GRADING (timepoints with >100 ground-truth grading px)')
        ps = [r['metrics'][1]['precision'] for r in grad_rows]
        rs = [r['metrics'][1]['recall']    for r in grad_rows]
        f1s= [r['metrics'][1]['f1']        for r in grad_rows]
        print(f'  range: P {min(ps):.2f}–{max(ps):.2f}, '
              f'R {min(rs):.2f}–{max(rs):.2f}, '
              f'F1 {min(f1s):.2f}–{max(f1s):.2f}')
        print(f'  mean : P={np.mean(ps):.3f}  '
              f'R={np.mean(rs):.3f}  F1={np.mean(f1s):.3f}')

    print(f'{"="*70}\n')


# ── (g) spot-check sites (demo aid for --aoi mode) ─────────────────────────────

def print_spot_check_sites(
    pred_cube: np.ndarray,             # (T, H, W) uint8
    label_cube: Optional[np.ndarray],  # (T, H, W) uint8 or None
    quarters: List[str],
    bbox: List[float],                  # [lon_min, lat_min, lon_max, lat_max]
    aoi_name: str,
    output_dir: str,
    t_idx: int = -1,
    n_per_class: int = 5,
    grid_size: int = 4,
    seed: int = 42,
):
    """
    Sample spatially-stratified predicted-construction and predicted-grading
    pixels at a given timepoint (default: final quarter) for live demo
    verification.

    Spatially stratifies by dividing the AOI into a grid_size × grid_size grid
    and selecting at most one pixel per grid cell — so the printed sites are
    spread across the AOI rather than clustering in one subdivision.

    Prints a table (pixel coords, lat/lon, predicted class, DW label if
    available) to stdout and saves the same content to
    outputs/spot_check_{aoi}.txt.
    """
    T, H, W = pred_cube.shape
    if t_idx < 0:
        t_idx = T + t_idx
    quarter = quarters[t_idx] if 0 <= t_idx < len(quarters) else f't={t_idx}'

    pred_t  = pred_cube[t_idx]
    label_t = label_cube[t_idx] if label_cube is not None else None

    lon_min, lat_min, lon_max, lat_max = bbox

    def pixel_to_lonlat(y: int, x: int) -> Tuple[float, float]:
        """Pixel (row, col) → (lon, lat). Row 0 is the NORTH edge."""
        lon = lon_min + (x + 0.5) / W * (lon_max - lon_min)
        lat = lat_max - (y + 0.5) / H * (lat_max - lat_min)
        return lon, lat

    rng = np.random.default_rng(seed)

    def stratified_sample(mask: np.ndarray, n: int) -> List[Tuple[int, int]]:
        """Pick up to n pixels from mask, at most one per grid cell."""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return []
        cell_h = max(1, H // grid_size)
        cell_w = max(1, W // grid_size)
        by_cell: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for y, x in zip(ys, xs):
            cell = (int(y) // cell_h, int(x) // cell_w)
            by_cell.setdefault(cell, []).append((int(y), int(x)))
        cells = list(by_cell.keys())
        rng.shuffle(cells)
        sampled = []
        for cell in cells:
            if len(sampled) >= n:
                break
            pixels_in_cell = by_cell[cell]
            idx = rng.integers(0, len(pixels_in_cell))
            sampled.append(pixels_in_cell[idx])
        return sampled

    grading_sites     = stratified_sample(pred_t == 1, n_per_class)
    constructed_sites = stratified_sample(pred_t == 2, n_per_class)

    # Build lines (printed AND saved)
    lines = []
    lines.append(f'\n{"="*70}')
    lines.append(f'SPOT-CHECK SITES — {aoi_name.upper()}')
    lines.append(f'{"="*70}')
    lines.append(f'Timepoint: {quarter} (t={t_idx})')
    lines.append(f'Spatial stratification: {grid_size}x{grid_size} grid '
                 f'(max one pixel per cell)')
    lines.append(f'Found {len(grading_sites)} grading + {len(constructed_sites)} '
                 f'constructed predicted sites')
    lines.append('')

    def fmt_section(header: str, sites: List[Tuple[int, int]]):
        lines.append(header)
        if not sites:
            lines.append('  (no predictions in this class at this timepoint)')
            return
        if label_t is not None:
            lines.append(f'  {"#":>3s}  {"pixel(y,x)":>14s}  {"lat":>10s}  {"lon":>11s}  '
                         f'{"pred":>12s}  {"DW label":>12s}')
        else:
            lines.append(f'  {"#":>3s}  {"pixel(y,x)":>14s}  {"lat":>10s}  {"lon":>11s}  '
                         f'{"pred":>12s}')
        for i, (y, x) in enumerate(sites, start=1):
            lon, lat = pixel_to_lonlat(y, x)
            pred_name = CLASS_NAMES.get(int(pred_t[y, x]), f'cls{int(pred_t[y, x])}')
            if label_t is not None:
                lbl_val = int(label_t[y, x])
                if lbl_val == IGNORE_LABEL:
                    lbl_name = 'ignore'
                else:
                    lbl_name = CLASS_NAMES.get(lbl_val, f'cls{lbl_val}')
                lines.append(f'  {i:>3d}  ({y:>4d},{x:>4d})  '
                             f'{lat:>10.5f}  {lon:>11.5f}  '
                             f'{pred_name:>12s}  {lbl_name:>12s}')
            else:
                lines.append(f'  {i:>3d}  ({y:>4d},{x:>4d})  '
                             f'{lat:>10.5f}  {lon:>11.5f}  '
                             f'{pred_name:>12s}')

    fmt_section('GRADING (predicted=1)', grading_sites)
    lines.append('')
    fmt_section('CONSTRUCTED (predicted=2)', constructed_sites)
    lines.append(f'{"="*70}')

    text = '\n'.join(lines) + '\n'
    print(text)

    txt_path = os.path.join(output_dir, f'spot_check_{aoi_name}.txt')
    with open(txt_path, 'w') as f:
        f.write(text)
    print(f'[eval] Spot-check sites saved -> {txt_path}')


# ── (b) trajectory / progression figures ──────────────────────────────────────

def save_trajectory_maps(
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    quarters: List[str],
    aoi_name: str,
    figures_dir: str,
    t_start: int,
):
    if not HAS_MPL:
        return
    T = pred_cube.shape[0]
    n_t = T - t_start
    if n_t == 0:
        return

    cols = min(n_t, 6)
    rows = ((n_t - 1) // cols + 1) * 2
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

    for i in range(n_t, rows // 2 * cols):
        col_i = i % cols
        for row in range(rows):
            axes[row, col_i].axis('off')

    from matplotlib.patches import Patch
    legend = [Patch(color=np.array(c) / 255., label=l) for c, l in [
        ([0, 180, 0],   'baseline'),
        ([255, 165, 0], 'grading'),
        ([200, 0, 0],   'constructed'),
        ([50, 50, 50],  'ignore'),
    ]]
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=8)
    fig.suptitle(f'{aoi_name} - Predicted (top) vs DW Label (bottom)', fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    path = os.path.join(figures_dir, f'{aoi_name}_trajectory_maps.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] Saved trajectory maps -> {path}')


def save_sample_pixel_trajectories(
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    quarters: List[str],
    aoi_name: str,
    figures_dir: str,
    t_start: int,
    n_pixels: int = 5,
):
    if not HAS_MPL:
        return
    T, H, W = label_cube.shape

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

    fig.suptitle(f'{aoi_name} - Sample pixel trajectories', fontsize=10)
    plt.tight_layout()
    path = os.path.join(figures_dir, f'{aoi_name}_pixel_trajectories.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] Saved pixel trajectories -> {path}')


def save_per_timepoint_curve(
    timepoint_results: List[Dict],
    aoi_name: str,
    figures_dir: str,
):
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

    ax_f1.set_title(f'{aoi_name} - F1 vs history length')
    ax_iou.set_title(f'{aoi_name} - IoU vs history length')
    plt.tight_layout()

    path = os.path.join(figures_dir, f'{aoi_name}_per_timepoint_curve.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] Saved per-timepoint curve -> {path}')


# ── (c) early detection ────────────────────────────────────────────────────────

def compute_early_detection(
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    t_start: int,
) -> Dict:
    T, H, W = label_cube.shape
    ever_built = np.any(label_cube == 2, axis=0)

    early_count = 0
    confirmed_count = int(ever_built.sum())

    for t in range(t_start, T):
        pred  = pred_cube[t]
        label = label_cube[t]

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
            'in their full trajectory.'
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
    if not HAS_MPL:
        return
    T, H, W = label_cube.shape
    ever_built = np.any(label_cube == 2, axis=0)

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
        ax.set_title(f'Pixel ({h},{w}) - yellow=model leads DW', fontsize=8)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_ylim(-0.2, 2.5)

    fig.suptitle(f'{aoi_name} - Early detection examples (model ahead of DW label)', fontsize=10)
    plt.tight_layout()
    path = os.path.join(figures_dir, f'{aoi_name}_early_detection.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] Saved early-detection examples -> {path}')


# ── (d) two-tier grading eval ──────────────────────────────────────────────────

def two_tier_grading_eval(
    pred_cube: np.ndarray,
    label_cube: np.ndarray,
    quarters: List[str],
    aoi_name: str,
    figures_dir: str,
    t_start: int,
) -> Dict:
    T, H, W = label_cube.shape
    ever_built = np.any(label_cube == 2, axis=0)
    ever_bare  = np.any(label_cube == 1, axis=0)

    confirmed_mask = ever_bare & ever_built

    tier1_results = []
    for t in range(t_start, T):
        preds  = pred_cube[t]
        labels = label_cube[t]
        mask   = (labels != IGNORE_LABEL) & confirmed_mask

        if mask.sum() == 0:
            continue

        m = compute_per_class_metrics(preds[mask].ravel(), labels[mask].ravel())
        tier1_results.append({'t': t, 'metrics': m, 'n_pixels': int(mask.sum())})

    active_grading_maps = {}
    if HAS_MPL:
        for t in range(T - 1, t_start - 1, -1):
            active = (label_cube[t] == 1) & ~ever_built
            if active.sum() > 50:
                fig, ax = plt.subplots(figsize=(6, 5))
                disp = np.zeros((H, W, 3), dtype=np.uint8)
                disp[pred_cube[t] == 0] = [0, 180, 0]
                disp[pred_cube[t] == 1] = [255, 165, 0]
                disp[pred_cube[t] == 2] = [200, 0, 0]
                ax.imshow(disp)
                ay, ax_ = np.where(active)
                ax.scatter(ax_, ay, c='cyan', s=1, marker='.', label='active grading')
                ax.set_title(
                    f'{aoi_name} t={t} ({quarters[t]})\n'
                    'ACTIVE GRADING (cyan) - No Ground Truth Available\n'
                    'NO PRECISION NUMBER IS REPORTED FOR THESE PIXELS',
                    fontsize=8, color='darkred'
                )
                ax.legend(fontsize=7)
                ax.axis('off')
                path = os.path.join(figures_dir, f'{aoi_name}_active_grading_t{t}.png')
                fig.savefig(path, dpi=120, bbox_inches='tight')
                plt.close(fig)
                print(f'[eval] Saved active-grading map (Tier 2) -> {path}')
                active_grading_maps[t] = path
                break

    return {
        'tier1_confirmed_trajectory': tier1_results,
        'tier2_active_grading_maps':  active_grading_maps,
        'tier2_note': (
            'Active-grading pixels (label=1 at time t, no confirmed built outcome) '
            'are shown on the map ONLY. No precision/recall is computed - '
            'these pixels have no verified ground truth.'
        ),
    }


# ── (e) failure demos ──────────────────────────────────────────────────────────

def resolve_failure_aoi_dir(aoi_name: str, data_root: str) -> Optional[str]:
    standard = os.path.join(data_root, aoi_name)
    if os.path.exists(os.path.join(standard, 'spectral_cube.npy')):
        return standard
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
    aoi_dir = resolve_failure_aoi_dir(aoi_name, data_root)
    if aoi_dir is None:
        print(f'[eval] {aoi_name}: no data found. Skipping.')
        return None

    data = load_aoi(aoi_name, aoi_dir.replace(f'/{aoi_name}', ''))
    if data is None:
        sp = os.path.join(aoi_dir, 'spectral_cube.npy')
        lb = os.path.join(aoi_dir, 'label_cube.npy')
        mt = os.path.join(aoi_dir, 'metadata.json')
        if not all(os.path.exists(p) for p in [sp, lb, mt]):
            print(f'[eval] {aoi_name}: incomplete data in {aoi_dir}. Skipping.')
            return None
        with open(mt) as f:
            meta = json.load(f)
        quarters = meta['quarters']
        T = len(quarters)
        spectral = np.load(sp, mmap_mode='r')
        labels   = np.load(lb, mmap_mode='r')
        dates    = np.array([quarter_to_doy(q) for q in quarters], dtype=np.float32)
        nan_flag = np.any(np.isnan(np.array(spectral)), axis=1)
        H, W     = spectral.shape[2], spectral.shape[3]
        data = {'spectral': spectral, 'labels': labels, 'dates': dates,
                'nan_flag': nan_flag, 'quarters': quarters, 'T': T, 'H': H, 'W': W}

    T = data['T']
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

    mask   = label != IGNORE_LABEL
    metrics = None
    if mask.sum() > 0:
        valid_classes = np.unique(label[mask])
        if len(valid_classes) > 0:
            metrics = compute_per_class_metrics(pred[mask].ravel(), label[mask].ravel())

    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(_class_to_rgb(pred))
        axes[0].set_title(f'{aoi_name} predicted t={t_demo}\n({quarters[t_demo]})', fontsize=9)
        axes[0].axis('off')
        axes[1].imshow(_class_to_rgb(label))
        axes[1].set_title(f'{aoi_name} DW label t={t_demo}', fontsize=9)
        axes[1].axis('off')

        if wendell_metrics and metrics:
            w_f1 = wendell_metrics.get(1, {}).get('f1', float('nan'))
            d_f1 = metrics.get(1, {}).get('f1', float('nan'))
            fig.text(0.5, 0.01,
                     f'Grading F1: wendell={w_f1:.3f}  {aoi_name}={d_f1:.3f}  '
                     f'd={d_f1 - w_f1:+.3f}',
                     ha='center', fontsize=9, color='darkred')

        path = os.path.join(figures_dir, f'{aoi_name}_failure_demo_t{t_demo}.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'[eval] Saved failure-demo map -> {path}')

    return {
        'aoi':     aoi_name,
        't_demo':  t_demo,
        'quarter': quarters[t_demo],
        'metrics': metrics,
    }


# ── bbox lookup ────────────────────────────────────────────────────────────────

def _lookup_aoi_bbox(aoi_name: str) -> Optional[List[float]]:
    """Try config.py for the bbox; return None if not found."""
    try:
        from config import AOIS
        return AOIS.get(aoi_name, {}).get('bbox')
    except Exception:
        return None


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate causal construction-detection model')
    p.add_argument('--ckpt',       default=os.path.join(CKPT_DIR, 'best.pt'))
    p.add_argument('--device',     default=os.environ.get('DEVICE', 'cpu'))
    p.add_argument('--data-root',  default=DATA_ROOT)
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    p.add_argument('--aoi',        default=None,
                   help='Single AOI to evaluate (overrides VAL_AOIS). '
                        'Data must exist at <data-root>/<aoi>/. '
                        'Triggers spot-check site listing. '
                        'Failure demos are skipped when --aoi is set.')
    p.add_argument('--spot-check-n', type=int, default=5,
                   help='Number of spot-check sites per class (default 5, '
                        'only applies in --aoi mode)')
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

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f'Checkpoint not found: {args.ckpt}\n'
            'Run train/train.py first to produce a checkpoint.'
        )

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f'[eval] Loaded checkpoint from {args.ckpt} (epoch={ckpt["epoch"]}, '
          f'val_loss={ckpt["val_loss"]:.4f})')

    norm_stats_path = ckpt.get('norm_stats_path', os.path.join(args.output_dir, 'norm_stats.json'))
    if not os.path.exists(norm_stats_path):
        norm_stats_path = os.path.join(args.output_dir, 'norm_stats.json')
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    print(f'[eval] Using norm stats from {norm_stats_path}')

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

    if smoke_test:
        eval_aois = ['sunterra']
        run_failure_demos = False
    elif args.aoi is not None:
        eval_aois = [args.aoi]
        run_failure_demos = False
    else:
        eval_aois = VAL_AOIS
        run_failure_demos = True

    wendell_grading_metrics = None

    for aoi_name in eval_aois:
        print(f'\n[eval] ==== {aoi_name.upper()} ====')
        data = load_aoi(aoi_name, args.data_root)
        if data is None:
            print(f'[eval] {aoi_name}: skipping (no data).')
            continue

        T        = data['T']
        quarters = data['quarters']
        label_cube = data['labels'][:]

        print(f'[eval] Running causal inference at each t (T={T})...')
        pred_cube = run_tiled_inference(
            model, data, norm_stats, device, t_start=T_MIN,
        )

        tp_results = per_timepoint_metrics(pred_cube, label_cube, t_start=T_MIN)
        print(f'[eval] Per-timepoint metrics ({aoi_name}):')
        for r in tp_results:
            print(f'  t={r["t"]} ({quarters[r["t"]]}):')
            print_metrics(r['metrics'], prefix='    ')

        print_summary_table(tp_results, aoi_name)

        eval_results[aoi_name] = {
            'per_timepoint': tp_results,
        }

        if aoi_name == 'wendell' and tp_results:
            wendell_grading_metrics = tp_results[-1]['metrics']

        # ── (g) spot-check sites — only in --aoi mode ─────────────────────────
        if args.aoi is not None:
            bbox = _lookup_aoi_bbox(aoi_name)
            if bbox is None:
                print(f'[eval] {aoi_name}: bbox not found in config.AOIS - '
                      f'skipping spot-check site listing.')
            else:
                print_spot_check_sites(
                    pred_cube=pred_cube,
                    label_cube=label_cube,
                    quarters=quarters,
                    bbox=bbox,
                    aoi_name=aoi_name,
                    output_dir=args.output_dir,
                    t_idx=-1,
                    n_per_class=args.spot_check_n,
                )

        try:
            save_trajectory_maps(pred_cube, label_cube, quarters, aoi_name, FIGURES_DIR, T_MIN)
            save_per_timepoint_curve(tp_results, aoi_name, FIGURES_DIR)
            save_sample_pixel_trajectories(
                pred_cube, label_cube, quarters, aoi_name, FIGURES_DIR, T_MIN
            )
        except Exception as _fig_err:
            print(f'[eval] figure generation failed (non-fatal): {_fig_err}')

        early = compute_early_detection(pred_cube, label_cube, t_start=T_MIN)
        print(f'\n[eval] Early detection ({aoi_name}):')
        print(f'  Early (pixel,t) instances: {early["early_detection_pixel_timesteps"]}')
        print(f'  Confirmed-construction px:  {early["confirmed_construction_pixels"]}')
        print(f'  Early fraction:             {early["early_fraction"]:.3f}')
        eval_results[aoi_name]['early_detection'] = early
        try:
            save_early_detection_examples(
                pred_cube, label_cube, quarters, aoi_name, FIGURES_DIR, T_MIN
            )
        except Exception as _fig_err:
            print(f'[eval] early-detection figure failed (non-fatal): {_fig_err}')

        try:
            tier = two_tier_grading_eval(
                pred_cube, label_cube, quarters, aoi_name, FIGURES_DIR, T_MIN
            )
        except Exception as _fig_err:
            print(f'[eval] two-tier grading figure failed (non-fatal): {_fig_err}')
            tier = {'tier1_confirmed_trajectory': [],
                    'tier2_note': 'figure generation failed'}
        print(f'\n[eval] Two-tier grading eval ({aoi_name}):')
        print(f'  Tier 1 confirmed-trajectory timepoints: {len(tier["tier1_confirmed_trajectory"])}')
        print(f'  {tier["tier2_note"]}')
        eval_results[aoi_name]['two_tier_grading'] = {
            'tier1_n_timepoints': len(tier['tier1_confirmed_trajectory']),
            'tier2_note':         tier['tier2_note'],
        }

    if run_failure_demos:
        eval_results['failure_demos'] = {}
        for aoi_name in FAILURE_DEMO_AOIS:
            print(f'\n[eval] ==== FAILURE DEMO: {aoi_name.upper()} ====')
            result = run_failure_demo(
                aoi_name, args.data_root, model, norm_stats, device,
                FIGURES_DIR, wendell_metrics=wendell_grading_metrics,
            )
            if result is not None:
                print(f'[eval] {aoi_name} failure-demo metrics:')
                if result['metrics']:
                    print_metrics(result['metrics'], prefix='  ')
                eval_results['failure_demos'][aoi_name] = result

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
    print(f'\n[eval] Results saved -> {results_path}')

    if smoke_test:
        print('[eval] Smoke test evaluation completed successfully.')


if __name__ == '__main__':
    main()