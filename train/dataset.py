"""
ConstructionDataset: fixed-K causal window emission for Prithvi-EO-2.0 fine-tuning.

Causal formulation: for target timepoint t, the model receives the K=6 most recent
spectral frames ending at t-1 (the causal window spectral[t-K:t]) and predicts the
per-pixel land-cover state AT t. No frame at index >= t ever enters the input.

Fixed window size K=6 gives Prithvi a uniform (B, 6, K, 128, 128) tensor every
batch — no padding, no variable-length reshape, no bucketing required.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── constants ──────────────────────────────────────────────────────────────────
K = 6                      # Fixed causal window length (consecutive frames)
T_MIN = 4                  # Used by evaluate.py for per-timepoint eval start
PATCH_SIZE = 128
TRAIN_STRIDE = 64
EVAL_STRIDE = 128
BASELINE_CAP = 0.15        # Fraction of pure-baseline patches to keep
MAX_BASELINE_PREFIXES = 2  # Max cutoff t values emitted per baseline patch
NAN_FRAME_THRESH = 0.5     # Frame flagged NaN if >50% pixels are NaN
IGNORE_LABEL = 255
IGNORE_TARGET_THRESH = 0.9 # Skip example if >90% target pixels are IGNORE

_QUARTER_DOY = {'Q1': 1, 'Q2': 91, 'Q3': 182, 'Q4': 274}

# Prithvi-EO-2.0-300M pretraining normalization statistics (HLS 0–10000 scale).
# Bands: Blue=B02, Green=B03, Red=B04, NIR=B08, SWIR1=B11, SWIR2=B12.
PRITHVI_MEAN = [775.2290211032589,  1080.992780391705,  1228.5855250417867,
                2497.2022620507532, 2204.2139147975554, 1610.8324823273745]
PRITHVI_STD  = [1281.526139861424,  1369.4656152478244, 1368.3978679245926,
                1461.8524578008785, 1356.8007467645994, 1294.7874235425885]


# ── helpers ────────────────────────────────────────────────────────────────────

def quarter_to_doy(q: str) -> float:
    """'2021-Q3' → normalized day-of-year in [0, 1]."""
    _, qpart = q.split('-')
    return _QUARTER_DOY[qpart] / 365.0


def load_aoi(aoi_name: str, data_root: str) -> Optional[Dict]:
    """
    Load one AOI's arrays and metadata via mmap.
    Returns None (with warning) if any required file is missing.
    """
    aoi_dir = os.path.join(data_root, aoi_name)
    spectral_path = os.path.join(aoi_dir, 'spectral_cube.npy')
    label_path    = os.path.join(aoi_dir, 'label_cube.npy')
    meta_path     = os.path.join(aoi_dir, 'metadata.json')

    for p in (spectral_path, label_path, meta_path):
        if not os.path.exists(p):
            print(f'[dataset] WARNING: {aoi_name} missing {os.path.basename(p)}, skipping.')
            return None

    with open(meta_path) as f:
        meta = json.load(f)

    quarters = meta['quarters']
    T = len(quarters)   # always use quarters list length, not metadata["shape"]

    spectral = np.load(spectral_path, mmap_mode='r')   # (T, 6, H, W) float64
    labels   = np.load(label_path,    mmap_mode='r')   # (T, H, W) uint8

    assert spectral.shape[0] == T, f'{aoi_name}: spectral T mismatch'
    assert spectral.shape[1] == 6, f'{aoi_name}: expected 6 bands, got {spectral.shape[1]}'
    assert labels.shape[0]   == T, f'{aoi_name}: label T mismatch'

    H, W  = spectral.shape[2], spectral.shape[3]
    dates = np.array([quarter_to_doy(q) for q in quarters], dtype=np.float32)

    print(f'[dataset] {aoi_name}: precomputing NaN mask ({T}×{H}×{W})...')
    nan_flag = np.any(np.isnan(np.array(spectral)), axis=1)   # (T, H, W) bool

    return {
        'spectral': spectral,    # mmap — accessed patch-by-patch in __getitem__
        'labels':   labels,      # mmap
        'dates':    dates,       # (T,) float32
        'nan_flag': nan_flag,    # (T, H, W) bool — fully in RAM
        'quarters': quarters,
        'T': T, 'H': H, 'W': W,
    }


def detect_data_scale(aoi_data: Dict) -> float:
    """
    Inspect a small sample and return the scale factor to reach Prithvi's 0–10000 range.
    Prints observed min/max for manual verification.
    """
    spectral = aoi_data['spectral']
    T, C, H, W = spectral.shape
    r0, c0 = H // 4, W // 4
    sample = np.array(spectral[:min(3, T), :, r0:r0+64, c0:c0+64], dtype=np.float64)
    valid  = sample[~np.isnan(sample)]

    if len(valid) == 0:
        print('[dataset] WARNING: sample patch is all-NaN; defaulting data_scale=1.')
        return 1.0

    vmin, vmax, vmean = float(valid.min()), float(valid.max()), float(valid.mean())
    print(f'[dataset] Spectral sample stats (first 3 frames, central 64×64):')
    print(f'  min={vmin:.5f}  max={vmax:.5f}  mean={vmean:.5f}')

    if vmax <= 2.0:
        scale = 10000.0
        print(f'  → 0-1 reflectance → applying data_scale={scale}')
    else:
        scale = 1.0
        print(f'  → 0-10000 DN → data_scale={scale} (no rescaling)')
    return scale


def compute_norm_stats(train_aoi_list: List[str], data_root: str,
                       data_scale: Optional[float] = None) -> Dict:
    """Return Prithvi HLS pretraining stats. Detects data scale from the first AOI."""
    if data_scale is None:
        for aoi_name in train_aoi_list:
            d = load_aoi(aoi_name, data_root)
            if d is not None:
                data_scale = detect_data_scale(d)
                break
        else:
            data_scale = 1.0

    print('[dataset] Using hardcoded Prithvi-EO-2.0 HLS pretraining norm stats.')
    print(f'          (data_scale={data_scale} applied before normalization)')
    return {
        'mean':       PRITHVI_MEAN,
        'std':        PRITHVI_STD,
        'data_scale': data_scale,
        'source':     'prithvi_eo2_hls_pretrain',
    }


def compute_norm_stats_from_data(train_aoi_list: List[str], data_root: str,
                                 data_scale: float = 1.0) -> Dict:
    """Fallback: compute per-band mean/std from train AOIs (NaN-safe)."""
    print('[dataset] Computing norm stats from train AOIs (nanmean/nanstd)...')
    band_values: List[List] = [[] for _ in range(6)]
    for aoi_name in train_aoi_list:
        data = load_aoi(aoi_name, data_root)
        if data is None:
            continue
        spectral = np.array(data['spectral'], dtype=np.float64) * data_scale
        for b in range(6):
            flat = spectral[:, b, :, :].ravel()
            band_values[b].append(flat[~np.isnan(flat)])
    means, stds = [], []
    for b in range(6):
        all_vals = np.concatenate(band_values[b]) if band_values[b] else np.array([0.0])
        means.append(float(np.nanmean(all_vals)))
        stds.append(float(np.nanstd(all_vals)))
    return {'mean': means, 'std': stds, 'data_scale': data_scale,
            'source': 'computed_from_train', 'train_aois': train_aoi_list}


def patch_positions(H: int, W: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    """Return (row, col) top-left corners of all valid patches, deduped."""
    seen: set = set()
    positions = []

    def add(r: int, c: int):
        if (r, c) not in seen:
            seen.add((r, c))
            positions.append((r, c))

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            add(r, c)
    if H >= patch_size:
        for c in range(0, W - patch_size + 1, stride):
            add(H - patch_size, c)
    if W >= patch_size:
        for r in range(0, H - patch_size + 1, stride):
            add(r, W - patch_size)
    if H >= patch_size and W >= patch_size:
        add(H - patch_size, W - patch_size)
    return positions


def _augment(
    spectral: torch.Tensor,   # (K, 6, P, P)
    target:   torch.Tensor,   # (P, P)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random hflip / vflip / rot90 + per-band spectral jitter. Train split only."""
    if random.random() < 0.5:
        spectral = torch.flip(spectral, dims=[-1])
        target   = torch.flip(target,   dims=[-1])
    if random.random() < 0.5:
        spectral = torch.flip(spectral, dims=[-2])
        target   = torch.flip(target,   dims=[-2])
    k = random.randint(0, 3)
    if k:
        spectral = torch.rot90(spectral, k=k, dims=[-2, -1])
        target   = torch.rot90(target,   k=k, dims=[-2, -1])
    scale    = 0.9 + 0.2 * torch.rand(1, 6, 1, 1)
    spectral = spectral * scale
    return spectral, target


# ── dataset ────────────────────────────────────────────────────────────────────

class ConstructionDataset(Dataset):
    """
    Fixed-K causal window dataset.

    Each example has a fixed temporal depth of K=6 frames:
      - input  : spectral_cube[t-K : t]  → (K, 6, 128, 128) — all frames < t
      - dates  : quarter DOYs for those K frames → (K,)
      - target : label_cube[t]           → (128, 128)

    Fixed K means all batches are uniform shape — no padding, no bucketing,
    no variable-length reshape. Prithvi gets (B, 6, K, H, W) every time.
    """

    def __init__(
        self,
        aoi_list:   List[str],
        data_root:  str,
        split:      str = 'train',
        patch_size: int = PATCH_SIZE,
        norm_stats: Optional[Dict] = None,
        seed:       int = 42,
        smoke_test: bool = False,
    ):
        assert split in ('train', 'val', 'eval')
        self.split      = split
        self.patch_size = patch_size
        self.norm_stats = norm_stats
        self.seed       = seed
        self.smoke_test = smoke_test
        self.data_scale = float(norm_stats.get('data_scale', 1.0)) if norm_stats else 1.0
        stride = TRAIN_STRIDE if split == 'train' else EVAL_STRIDE

        rng = random.Random(seed)
        self.aoi_data: Dict[str, Dict] = {}
        self.examples: List[Dict]      = []

        for aoi_name in aoi_list:
            data = load_aoi(aoi_name, data_root)
            if data is None:
                continue
            self.aoi_data[aoi_name] = data
            self._emit_prefixes(aoi_name, data, stride, rng)

        if not self.examples:
            raise RuntimeError(
                f'No examples built for split={split}, AOIs={aoi_list}. '
                'Ensure spectral_cube.npy / label_cube.npy / metadata.json exist.'
            )

        if smoke_test:
            self.examples = self.examples[:3]

        self._print_class_distribution()

    # ── prefix emission ──────────────────────────────────────────────────────

    def _emit_prefixes(
        self,
        aoi_name: str,
        data:     Dict,
        stride:   int,
        rng:      random.Random,
    ):
        T        = data['T']
        H, W     = data['H'], data['W']
        nan_flag = data['nan_flag']   # (T, H, W) bool
        labels   = data['labels']     # mmap (T, H, W)
        P        = self.patch_size

        positions = patch_positions(H, W, P, stride)
        n_emitted = 0

        for (r, c) in positions:
            label_patch = labels[:, r:r+P, c:c+P].astype(np.int32)

            non_ignore     = label_patch < IGNORE_LABEL
            has_transition = bool(
                np.any(((label_patch == 1) | (label_patch == 2)) & non_ignore)
            )

            if not has_transition:
                if rng.random() > BASELINE_CAP:
                    continue
                emit_range = range(max(K, T - MAX_BASELINE_PREFIXES), T)
            else:
                transition_ts = [
                    t for t in range(T)
                    if np.any(
                        ((label_patch[t] == 1) | (label_patch[t] == 2))
                        & (label_patch[t] < IGNORE_LABEL)
                    )
                ]
                first_t = transition_ts[0] if transition_ts else T - 1
                t_start = max(K, first_t - 2)
                emit_range = range(t_start, T)

            for t in emit_range:
                if t < K:
                    continue

                target_patch = label_patch[t]
                if np.mean(target_patch == IGNORE_LABEL) > IGNORE_TARGET_THRESH:
                    continue

                # Check NaN density in the K-frame window [t-K:t].
                # Individual NaN pixels are filled with band mean later; we only
                # skip an example if more than half the frames are mostly NaN.
                win_nan    = nan_flag[t-K:t, r:r+P, c:c+P]   # (K, P, P)
                nan_frac   = win_nan.mean(axis=(1, 2))          # (K,)
                n_nan_frames = int(np.sum(nan_frac > NAN_FRAME_THRESH))
                if n_nan_frames > K // 2:
                    continue

                self.examples.append({
                    'aoi':       aoi_name,
                    'patch_row': r,
                    'patch_col': c,
                    't':         t,
                })
                n_emitted += 1

        print(f'[dataset]   {aoi_name} ({self.split}): '
              f'{n_emitted} examples from {len(positions)} patches')

    # ── class distribution check ─────────────────────────────────────────────

    def _print_class_distribution(self):
        counts    = {0: 0, 1: 0, 2: 0, IGNORE_LABEL: 0}
        sample_sz = min(500, len(self.examples))
        sample    = random.Random(self.seed).sample(self.examples, sample_sz)

        for ex in sample:
            data = self.aoi_data[ex['aoi']]
            r, c, t, P = ex['patch_row'], ex['patch_col'], ex['t'], self.patch_size
            tgt = data['labels'][t, r:r+P, c:c+P].astype(np.int32)
            for cls in counts:
                counts[cls] += int(np.sum(tgt == cls))

        total = sum(counts.values())
        if total == 0:
            return

        print(f'\n[dataset] {self.split.upper()} class distribution '
              f'(sampled {sample_sz} examples):')
        for cls, name in [(0, 'baseline'), (1, 'grading'),
                          (2, 'constructed'), (IGNORE_LABEL, 'ignore')]:
            pct = 100.0 * counts[cls] / total
            print(f'  class {cls:3d} ({name:12s}): {counts[cls]:9,d} px  ({pct:.1f}%)')

        non_ignore = total - counts[IGNORE_LABEL]
        if non_ignore > 0 and self.split == 'train':
            grading_pct = 100.0 * counts[1] / non_ignore
            if grading_pct < 1.0:
                raise RuntimeError(
                    f'Grading class is only {grading_pct:.2f}% of non-ignore '
                    'train targets. Risk of class collapse. '
                    'Check label generation or adjust class weights.'
                )
        print()

    # ── dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex   = self.examples[idx]
        data = self.aoi_data[ex['aoi']]
        r, c = ex['patch_row'], ex['patch_col']
        t    = ex['t']
        P    = self.patch_size

        # Fixed K-frame causal window: spectral[t-K:t] — all frame indices < t
        spectral = data['spectral'][t-K:t, :, r:r+P, c:c+P].astype(np.float32)
        # shape: (K, 6, P, P)

        # Normalize: scale to Prithvi's 0-10000 range, fill per-pixel NaN with
        # band mean, then standardize. NaN frames are kept (K stays fixed);
        # individual NaN pixels are imputed so no NaN reaches the model.
        if self.norm_stats is not None:
            mean = np.array(self.norm_stats['mean'], dtype=np.float32)   # (6,)
            std  = np.array(self.norm_stats['std'],  dtype=np.float32)
            std  = np.where(std < 1e-8, 1.0, std)
            if self.data_scale != 1.0:
                spectral = spectral * self.data_scale
            for b in range(6):
                band = spectral[:, b, :, :]
                spectral[:, b, :, :] = np.where(np.isnan(band), mean[b], band)
            spectral = (spectral - mean[None, :, None, None]) / std[None, :, None, None]
        else:
            spectral = np.where(np.isnan(spectral), 0.0, spectral)

        assert not np.any(np.isnan(spectral)), (
            f'NaN survived fill in {ex["aoi"]} patch ({r},{c}) t={t}')

        dates  = data['dates'][t-K:t]                              # (K,) float32
        target = data['labels'][t, r:r+P, c:c+P].astype(np.int64) # (P, P)

        spectral_t = torch.from_numpy(spectral.copy())   # (K, 6, P, P)
        dates_t    = torch.from_numpy(dates.copy())       # (K,)
        target_t   = torch.from_numpy(target.copy())      # (P, P)

        if self.split == 'train':
            spectral_t, target_t = _augment(spectral_t, target_t)

        return spectral_t, dates_t, target_t


# ── collate ────────────────────────────────────────────────────────────────────

def construction_collate_fn(batch):
    """
    Collate for fixed-K examples. All tensors are uniform shape so plain stacking works.
    Returns (spectral, dates, target) — no n_frames since K=6 is always the same.
    """
    spectral_list, dates_list, target_list = zip(*batch)
    return (
        torch.stack(spectral_list),   # (B, K, 6, P, P)
        torch.stack(dates_list),       # (B, K)
        torch.stack(target_list),      # (B, P, P)
    )
