"""
ConstructionDataset: causal exhaustive-prefix emission for Prithvi-EO-2.0 fine-tuning.

Causal formulation: for target timepoint t, the model receives only spectral frames
0..t-1 (after cloud/NaN frame dropping) and predicts the per-pixel land-cover state
AT t. No frame at index >= t ever enters the input.

Key design choices:
  - Adaptive per-patch t_start avoids early-t all-baseline glut.
  - 15% cap on pure-baseline patches so "stable veg" is learned without flooding.
  - NaN frames dropped (not zero-filled) to avoid injecting fake bare pixels.
  - Bucketing by n_valid_frames gives same-length batches without padding.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# ── constants ──────────────────────────────────────────────────────────────────
T_MIN = 4             # minimum valid input frames required per prefix
PATCH_SIZE = 128
TRAIN_STRIDE = 64
EVAL_STRIDE = 128
BASELINE_CAP = 0.15   # fraction of pure-baseline patches to include
MAX_BASELINE_PREFIXES = 2  # emit at most this many prefixes for capped baseline patches
NAN_FRAME_THRESH = 0.5     # drop frame if >50% pixels NaN
IGNORE_LABEL = 255
IGNORE_TARGET_THRESH = 0.9  # skip prefix if >90% target pixels are IGNORE

# Quarter → first day of that quarter (for temporal coordinate encoding)
_QUARTER_DOY = {'Q1': 1, 'Q2': 91, 'Q3': 182, 'Q4': 274}


# ── helpers ────────────────────────────────────────────────────────────────────

def quarter_to_doy(q: str) -> float:
    """'2021-Q3' → normalized day-of-year in [0, 1]."""
    _, qpart = q.split('-')
    return _QUARTER_DOY[qpart] / 365.0


def load_aoi(aoi_name: str, data_root: str) -> Optional[Dict]:
    """
    Load one AOI's arrays and metadata via mmap. Returns None (with warning)
    if any required file is missing.
    """
    aoi_dir = os.path.join(data_root, aoi_name)
    spectral_path = os.path.join(aoi_dir, 'spectral_cube.npy')
    label_path = os.path.join(aoi_dir, 'label_cube.npy')
    meta_path = os.path.join(aoi_dir, 'metadata.json')

    for p in (spectral_path, label_path, meta_path):
        if not os.path.exists(p):
            print(f'[dataset] WARNING: {aoi_name} missing {os.path.basename(p)}, skipping.')
            return None

    with open(meta_path) as f:
        meta = json.load(f)

    quarters = meta['quarters']
    T = len(quarters)  # always use quarters list length, not metadata["shape"]

    spectral = np.load(spectral_path, mmap_mode='r')   # (T, 6, H, W) float64
    labels = np.load(label_path, mmap_mode='r')         # (T, H, W) uint8

    assert spectral.shape[0] == T, (
        f'{aoi_name}: spectral T={spectral.shape[0]} != metadata quarters T={T}')
    assert spectral.shape[1] == 6, (
        f'{aoi_name}: expected 6 bands, got {spectral.shape[1]}')
    assert labels.shape[0] == T, (
        f'{aoi_name}: label T={labels.shape[0]} != metadata quarters T={T}')

    H, W = spectral.shape[2], spectral.shape[3]
    dates = np.array([quarter_to_doy(q) for q in quarters], dtype=np.float32)  # (T,)

    # Precompute NaN flag: True if ANY band is NaN at that (t, h, w) pixel.
    # Load the whole spectral array once to build this (T, H, W) bool mask;
    # it's small (~4.5 MB for sunterra) and avoids repeated mmap seeks during build.
    print(f'[dataset] {aoi_name}: precomputing NaN mask ({T}×{H}×{W})...')
    nan_flag = np.any(np.isnan(np.array(spectral)), axis=1)  # (T, H, W) bool

    return {
        'spectral': spectral,    # mmap — accessed patch-by-patch in __getitem__
        'labels': labels,        # mmap
        'dates': dates,          # (T,) float32
        'nan_flag': nan_flag,    # (T, H, W) bool — fully in RAM
        'quarters': quarters,
        'T': T,
        'H': H,
        'W': W,
    }


def compute_norm_stats(train_aoi_list: List[str], data_root: str) -> Dict:
    """
    Return per-band mean/std for normalization.

    Priority:
      1. Prithvi-EO-2.0 pretraining stats from TerraTorch (best: matches pretraining).
      2. Compute from train AOIs (NaN pixels excluded).

    Caller should save the result to outputs/norm_stats.json.
    """
    try:
        # TerraTorch exposes pretraining stats for Prithvi-EO-2.0
        from terratorch.models.backbones.prithvi_eo_v2 import PRITHVI_MEAN, PRITHVI_STD
        print('[dataset] Using Prithvi-EO-2.0 pretraining norm stats from TerraTorch.')
        return {
            'mean': list(float(v) for v in PRITHVI_MEAN[:6]),
            'std':  list(float(v) for v in PRITHVI_STD[:6]),
            'source': 'terratorch_pretrain',
        }
    except Exception:
        pass

    print('[dataset] Computing norm stats from train AOIs (TerraTorch stats unavailable)...')
    sums   = np.zeros(6, dtype=np.float64)
    sq_sums = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.int64)

    for aoi_name in train_aoi_list:
        data = load_aoi(aoi_name, data_root)
        if data is None:
            continue
        spectral = np.array(data['spectral'])   # (T, 6, H, W) — full load for stats
        for b in range(6):
            band = spectral[:, b, :, :]
            valid = band[~np.isnan(band)]
            sums[b]    += valid.sum()
            sq_sums[b] += (valid ** 2).sum()
            counts[b]  += len(valid)

    means = sums / counts
    stds  = np.sqrt(np.maximum(sq_sums / counts - means ** 2, 1e-10))
    return {
        'mean':       means.tolist(),
        'std':        stds.tolist(),
        'source':     'computed_from_train',
        'train_aois': train_aoi_list,
    }


def patch_positions(H: int, W: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    """Return (row, col) top-left corners of all valid patches, deduped."""
    seen = set()
    positions = []

    def add(r, c):
        if (r, c) not in seen:
            seen.add((r, c))
            positions.append((r, c))

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            add(r, c)
    # Ensure right/bottom edges are covered
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
    spectral: torch.Tensor,   # (T, 6, P, P)
    target: torch.Tensor,     # (P, P)
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
    # Per-band multiplicative jitter; broadcast over (T, P, P)
    scale = 0.9 + 0.2 * torch.rand(1, 6, 1, 1)
    spectral = spectral * scale
    return spectral, target


# ── dataset ────────────────────────────────────────────────────────────────────

class ConstructionDataset(Dataset):
    """
    Emits causal prefix examples (input_frames, dates, target_label_at_t).
    Bucketed by n_valid_frames so same-bucket batches need no padding.
    """

    def __init__(
        self,
        aoi_list: List[str],
        data_root: str,
        split: str = 'train',           # 'train' | 'val' | 'eval'
        patch_size: int = PATCH_SIZE,
        norm_stats: Optional[Dict] = None,
        seed: int = 42,
        smoke_test: bool = False,
    ):
        assert split in ('train', 'val', 'eval')
        self.split      = split
        self.patch_size = patch_size
        self.norm_stats = norm_stats
        self.seed       = seed
        self.smoke_test = smoke_test
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
                'Ensure spectral_cube.npy / label_cube.npy / metadata.json exist under data/<aoi>/.'
            )

        if smoke_test:
            self.examples = self.examples[:3]

        self._print_class_distribution()

    # ── prefix emission ──────────────────────────────────────────────────────

    def _emit_prefixes(
        self,
        aoi_name: str,
        data: Dict,
        stride: int,
        rng: random.Random,
    ):
        T        = data['T']
        H, W     = data['H'], data['W']
        nan_flag = data['nan_flag']   # (T, H, W) bool — RAM
        labels   = data['labels']     # mmap (T, H, W) uint8
        P        = self.patch_size

        positions = patch_positions(H, W, P, stride)
        n_emitted = 0

        for (r, c) in positions:
            # Load label patch for all T at once (one mmap read per patch)
            label_patch = labels[:, r:r+P, c:c+P].astype(np.int32)  # (T, P, P)

            # Does this patch ever transition to grading or construction?
            non_ignore = label_patch < IGNORE_LABEL
            has_transition = bool(
                np.any(((label_patch == 1) | (label_patch == 2)) & non_ignore)
            )

            if not has_transition:
                # Pure-baseline patch: apply 15% cap and emit only last few prefixes
                if rng.random() > BASELINE_CAP:
                    continue
                emit_range = range(max(T_MIN, T - MAX_BASELINE_PREFIXES), T)
            else:
                # Find first quarter where any non-ignore pixel is grading or built
                transition_ts = [
                    t for t in range(T)
                    if np.any(
                        ((label_patch[t] == 1) | (label_patch[t] == 2))
                        & (label_patch[t] < IGNORE_LABEL)
                    )
                ]
                first_t = transition_ts[0] if transition_ts else T - 1
                t_start = max(T_MIN, first_t - 2)

                if t_start >= T - 1:
                    print(
                        f'[dataset] WARNING: {aoi_name} patch ({r},{c}) very early '
                        f'transition (t_start={t_start}, T={T}). Emitting only t={T-1}.'
                    )
                    emit_range = range(T - 1, T)
                else:
                    emit_range = range(t_start, T)

            for t in emit_range:
                target_patch = label_patch[t]   # (P, P)

                # Skip mostly-ignore targets
                if np.mean(target_patch == IGNORE_LABEL) > IGNORE_TARGET_THRESH:
                    continue

                # Valid input frames: indices 0..t-1 with ≤50% NaN pixels
                nan_patch_slice = nan_flag[:t, r:r+P, c:c+P]   # (t, P, P) bool
                nan_frac = nan_patch_slice.mean(axis=(1, 2))     # (t,)
                valid_idxs = [i for i, f in enumerate(nan_frac)
                              if f <= NAN_FRAME_THRESH]

                # Leakage guard: all input indices must be < t (guaranteed by range)
                assert all(idx < t for idx in valid_idxs), (
                    f'Temporal leakage detected: frame index >= t={t} in {aoi_name}')

                if len(valid_idxs) < T_MIN:
                    continue

                self.examples.append({
                    'aoi':            aoi_name,
                    'patch_row':      r,
                    'patch_col':      c,
                    't':              t,
                    'valid_frame_idxs': valid_idxs,
                    'n_frames':       len(valid_idxs),
                })
                n_emitted += 1

        print(f'[dataset]   {aoi_name} ({self.split}): {n_emitted} prefix examples from {len(positions)} patches')

    # ── class distribution check ─────────────────────────────────────────────

    def _print_class_distribution(self):
        """Sample targets, print distribution, abort if grading < 1% (train only)."""
        counts = {0: 0, 1: 0, 2: 0, IGNORE_LABEL: 0}
        sample_size = min(500, len(self.examples))
        sample = random.Random(self.seed).sample(self.examples, sample_size)

        for ex in sample:
            data = self.aoi_data[ex['aoi']]
            r, c, t, P = ex['patch_row'], ex['patch_col'], ex['t'], self.patch_size
            tgt = data['labels'][t, r:r+P, c:c+P].astype(np.int32)
            for cls in counts:
                counts[cls] += int(np.sum(tgt == cls))

        total = sum(counts.values())
        if total == 0:
            return

        print(f'\n[dataset] {self.split.upper()} class distribution (sampled {sample_size} examples):')
        for cls, name in [(0, 'baseline'), (1, 'grading'), (2, 'constructed'), (IGNORE_LABEL, 'ignore')]:
            pct = 100.0 * counts[cls] / total
            print(f'  class {cls:3d} ({name:12s}): {counts[cls]:9,d} px  ({pct:.1f}%)')

        non_ignore = total - counts[IGNORE_LABEL]
        if non_ignore > 0 and self.split == 'train':
            grading_pct = 100.0 * counts[1] / non_ignore
            if grading_pct < 1.0:
                raise RuntimeError(
                    f'Grading class is only {grading_pct:.2f}% of non-ignore targets '
                    f'in the TRAIN split. Risk of class collapse. '
                    'Check label generation or bump grading class weight.'
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
        idxs = np.array(ex['valid_frame_idxs'], dtype=np.int64)

        # ── load spectral patch for valid input frames only ─────────────────
        spectral = data['spectral'][idxs, :, r:r+P, c:c+P].astype(np.float32)
        # (n_valid, 6, P, P)

        # Sanity: no all-NaN frame should have survived the NaN-drop filter
        for fi, frame in enumerate(spectral):
            assert not np.all(np.isnan(frame)), (
                f'All-NaN frame (original idx={idxs[fi]}) reached model in '
                f'{ex["aoi"]} patch=({r},{c}). NaN-drop filter bug.')

        # ── normalize ────────────────────────────────────────────────────────
        if self.norm_stats is not None:
            mean = np.array(self.norm_stats['mean'], dtype=np.float32)  # (6,)
            std  = np.array(self.norm_stats['std'],  dtype=np.float32)  # (6,)
            std  = np.where(std < 1e-8, 1.0, std)
            # Replace remaining within-patch NaN pixels with band mean before norm
            for b in range(6):
                band = spectral[:, b, :, :]
                spectral[:, b, :, :] = np.where(np.isnan(band), mean[b], band)
            spectral = (spectral - mean[None, :, None, None]) / std[None, :, None, None]
        else:
            spectral = np.where(np.isnan(spectral), 0.0, spectral)

        # ── dates for valid frames ───────────────────────────────────────────
        dates = data['dates'][idxs]   # (n_valid,) float32

        # ── target ──────────────────────────────────────────────────────────
        target = data['labels'][t, r:r+P, c:c+P].astype(np.int64)   # (P, P)

        spectral_t = torch.from_numpy(spectral.copy())   # (n_valid, 6, P, P)
        dates_t    = torch.from_numpy(dates.copy())       # (n_valid,)
        target_t   = torch.from_numpy(target.copy())      # (P, P)

        if self.split == 'train':
            spectral_t, target_t = _augment(spectral_t, target_t)

        return spectral_t, dates_t, target_t, ex['n_frames']

    @property
    def max_t(self) -> int:
        """Maximum n_valid_frames across all examples (used to set model num_frames_max)."""
        return max(ex['n_frames'] for ex in self.examples)


# ── bucketed batch sampler ─────────────────────────────────────────────────────

class BucketSampler(Sampler):
    """
    Groups examples by n_valid_frames. Yields batches where every example has
    the same sequence length, so collation is a simple torch.stack (no padding).
    """

    def __init__(
        self,
        dataset: ConstructionDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self.drop_last  = drop_last
        self._epoch     = 0

        # Build buckets: n_frames → [example_index, ...]
        self.buckets: Dict[int, List[int]] = {}
        for i, ex in enumerate(dataset.examples):
            self.buckets.setdefault(ex['n_frames'], []).append(i)

        n_buckets = len(self.buckets)
        n_examples = sum(len(v) for v in self.buckets.values())
        print(f'[dataset] BucketSampler: {n_examples} examples across {n_buckets} length-buckets '
              f'(min_len={min(self.buckets)}, max_len={max(self.buckets)})')

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _build_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        bucket_order = list(self.buckets.keys())
        if self.shuffle:
            rng.shuffle(bucket_order)

        batches: List[List[int]] = []
        for n in bucket_order:
            idxs = list(self.buckets[n])
            if self.shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                chunk = idxs[start:start + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                if chunk:
                    batches.append(chunk)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        for batch in self._build_batches():
            yield batch

    def __len__(self) -> int:
        n = sum(len(v) for v in self.buckets.values())
        if self.drop_last:
            return sum(
                len(v) // self.batch_size for v in self.buckets.values()
            )
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size
            for v in self.buckets.values()
        )


def construction_collate_fn(batch):
    """
    Collate for BucketSampler batches: all examples have the same n_valid_frames,
    so plain stacking works with no masking.

    Returns:
        spectral  (B, T, 6, P, P)   float32
        dates     (B, T)             float32
        target    (B, P, P)          int64
        n_frames  int                uniform within batch
    """
    spectral_list, dates_list, target_list, n_frames_list = zip(*batch)
    return (
        torch.stack(spectral_list),   # (B, T, 6, P, P)
        torch.stack(dates_list),       # (B, T)
        torch.stack(target_list),      # (B, P, P)
        n_frames_list[0],              # scalar int
    )
