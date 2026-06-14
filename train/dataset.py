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

# Prithvi-EO-2.0-300M pretraining normalization statistics.
# Source: HuggingFace ibm-nasa-geospatial/Prithvi-EO-2.0-300M model card + TerraTorch configs.
# Bands (in order): Blue=B02, Green=B03, Red=B04, NIR=B08, SWIR1=B11, SWIR2=B12.
# These are computed from HLS (Harmonized Landsat Sentinel-2) data where pixel values
# are surface reflectance × 10000, i.e., the expected input range is ~0–10000.
# If your spectral_cube is in 0–1 reflectance scale, set data_scale=10000 to rescale
# before applying these stats.
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


def detect_data_scale(aoi_data: Dict) -> float:
    """
    Inspect a small sample of non-NaN spectral values and return the scale factor
    needed to bring them into Prithvi's expected 0–10000 range.

    Planetary Computer S2 L2A composites are typically 0–1 reflectance → scale=10000.
    If values are already ~0–10000 (raw DN) → scale=1.

    Prints the observed min/max so the user can verify.
    """
    spectral = aoi_data['spectral']    # mmap (T, 6, H, W)
    # Sample a central spatial crop from the first available time frame
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
        print(f'  → data appears to be 0–1 reflectance scale → applying data_scale={scale}')
    else:
        scale = 1.0
        print(f'  → data appears to be 0–10000 DN scale → data_scale={scale} (no rescaling)')

    return scale


def compute_norm_stats(train_aoi_list: List[str], data_root: str,
                       data_scale: Optional[float] = None) -> Dict:
    """
    Return per-band mean/std and data_scale for normalization.

    Priority:
      1. Hardcoded Prithvi-EO-2.0 HLS pretraining stats (PRITHVI_MEAN / PRITHVI_STD).
         These are in 0–10000 scale; data_scale is detected from the data.
      2. Try TerraTorch module constants (same values, but confirms installed version).
      3. Compute from train AOIs using np.nanmean / np.nanstd (NaN-safe).
         Stats are computed AFTER applying data_scale so they're in the same space
         as Prithvi's pretraining stats if scale was wrong.

    The returned dict includes 'data_scale' so evaluate.py reuses it identically.
    Caller saves result to outputs/norm_stats.json.
    """
    # ── resolve data_scale from the first available train AOI ─────────────────
    if data_scale is None:
        for aoi_name in train_aoi_list:
            d = load_aoi(aoi_name, data_root)
            if d is not None:
                data_scale = detect_data_scale(d)
                break
        else:
            data_scale = 1.0

    # ── priority 1: hardcoded Prithvi-EO-2.0 HLS pretraining stats ────────────
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
    """
    Fallback: compute per-band mean/std from train AOIs using np.nanmean/np.nanstd.
    Values are computed in the SCALED space (after multiplying by data_scale).
    Only call this if Prithvi's pretraining stats are genuinely inappropriate.
    """
    print('[dataset] Computing norm stats from train AOIs (nanmean/nanstd, NaN-safe)...')
    band_values: List[List[float]] = [[] for _ in range(6)]

    for aoi_name in train_aoi_list:
        data = load_aoi(aoi_name, data_root)
        if data is None:
            continue
        spectral = np.array(data['spectral'], dtype=np.float64) * data_scale  # (T,6,H,W)
        for b in range(6):
            flat = spectral[:, b, :, :].ravel()
            band_values[b].append(flat[~np.isnan(flat)])

    means, stds = [], []
    for b in range(6):
        all_vals = np.concatenate(band_values[b]) if band_values[b] else np.array([0.0])
        means.append(float(np.nanmean(all_vals)))
        stds.append(float(np.nanstd(all_vals)))

    return {
        'mean':       means,
        'std':        stds,
        'data_scale': data_scale,
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
        num_frames_max: Optional[int] = None,
    ):
        assert split in ('train', 'val', 'eval')
        self.split      = split
        self.patch_size = patch_size
        self.norm_stats = norm_stats
        self.seed       = seed
        self.smoke_test = smoke_test
        # data_scale: multiply raw spectral values before applying norm stats.
        # Prithvi's HLS stats are in 0–10000; PC S2 composites are 0–1 → scale=10000.
        self.data_scale     = float(norm_stats.get('data_scale', 1.0)) if norm_stats else 1.0
        self.num_frames_max = num_frames_max   # if set, pad every example to this length
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
            # Scale raw reflectance to Prithvi's expected range (e.g., 0–1 → 0–10000)
            if self.data_scale != 1.0:
                spectral = spectral * self.data_scale
            # Fill within-patch NaN pixels with the band mean (in scaled units)
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

    def get_t(self, idx: int) -> int:
        """Return n_valid_frames for example idx (used by BucketSampler)."""
        return self.examples[idx]['n_frames']

    @property
    def max_t(self) -> int:
        """Maximum n_valid_frames across all examples."""
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
        super().__init__()
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
