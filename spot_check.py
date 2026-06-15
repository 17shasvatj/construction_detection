"""
Shared utilities for Google Earth spot-check of construction-detection labels/predictions.
Imported by validate_labels.py and make_demo.py — no CLI, no side effects.
"""

import json
import math
from pathlib import Path

import numpy as np


# ── Class name tables ──────────────────────────────────────────────────────────

DW_NAMES = {
    0: 'water', 1: 'trees', 2: 'grass', 3: 'flooded_veg', 4: 'crops',
    5: 'shrub', 6: 'built', 7: 'bare', 8: 'snow',
}
LABEL_NAMES = {0: 'baseline', 1: 'grading', 2: 'built', 255: 'excluded'}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_aoi(aoi: str, data_root: str = 'data') -> dict:
    """
    Load AOI data from data/{aoi}/.

    Returns dict with keys:
      meta, quarters, bbox ([lon_min, lat_min, lon_max, lat_max]), H, W, T,
      label_cube (T,H,W) uint8,
      spectral_cube (T,6,H,W) float64,
      dw_cube (T,H,W) uint8 or None if dw_cube.npy is absent.
    """
    aoi_dir = Path(data_root) / aoi
    meta_path = aoi_dir / 'metadata.json'
    if not meta_path.exists():
        available = [p.name for p in Path(data_root).iterdir() if p.is_dir()] if Path(data_root).exists() else []
        raise FileNotFoundError(
            f"No metadata.json at {meta_path}.\n"
            f"Available AOIs under {data_root}/: {available}"
        )

    meta = json.loads(meta_path.read_text())
    quarters = meta['quarters']
    bbox = meta['bbox']           # [lon_min, lat_min, lon_max, lat_max]
    H = meta['shape'][1]
    W = meta['shape'][2]
    T = len(quarters)             # use quarters length, not shape[0]

    label_cube    = np.load(aoi_dir / 'label_cube.npy',    mmap_mode='r')
    spectral_cube = np.load(aoi_dir / 'spectral_cube.npy', mmap_mode='r')

    dw_path  = aoi_dir / 'dw_cube.npy'
    dw_cube  = np.load(dw_path, mmap_mode='r') if dw_path.exists() else None

    return {
        'meta': meta,
        'quarters': quarters,
        'bbox': bbox,
        'H': H, 'W': W, 'T': T,
        'label_cube': label_cube,
        'spectral_cube': spectral_cube,
        'dw_cube': dw_cube,
    }


# ── Coordinate conversion ──────────────────────────────────────────────────────

def pixel_to_lonlat(y: int, x: int, bbox: list, H: int, W: int):
    """
    Convert pixel (row=y, col=x) to (lon, lat) at pixel centre.
    Formula verified against overpass_validation.py and original spot_check.py.
    bbox = [lon_min, lat_min, lon_max, lat_max]
    """
    lon = bbox[0] + (x + 0.5) / W * (bbox[2] - bbox[0])
    lat = bbox[3] - (y + 0.5) / H * (bbox[3] - bbox[1])
    return lon, lat


# ── Google Earth URL ───────────────────────────────────────────────────────────

def earth_url(lat: float, lon: float) -> str:
    return f"https://earth.google.com/web/@{lat:.6f},{lon:.6f},150a,500d,1y,0h,0t,0r"


# ── Trajectory string ──────────────────────────────────────────────────────────

def dw_trajectory(cube: np.ndarray, y: int, x: int, names: dict = None) -> str:
    """
    Collapsed trajectory at pixel (y, x) from a (T, H, W) cube.
    Consecutive duplicate labels are collapsed: [1,1,7,7,6] → 'trees→bare→built'.
    names defaults to DW_NAMES if cube values are in 0-8, else LABEL_NAMES.
    """
    seq = cube[:, y, x].tolist()
    if names is None:
        names = DW_NAMES if set(seq) <= set(DW_NAMES) else LABEL_NAMES

    collapsed = [seq[0]]
    for v in seq[1:]:
        if v != collapsed[-1]:
            collapsed.append(v)

    return '→'.join(names.get(v, str(v)) for v in collapsed)


# ── Stratified spatial sampling ────────────────────────────────────────────────

def stratified_sample(mask: np.ndarray, n: int, seed: int = 42) -> list:
    """
    Sample up to n (y, x) positions from a boolean mask, spread across a 4×4 spatial grid.
    Returns a list of (y, x) tuples in stable top-left→bottom-right scan order.
    If fewer than n pixels exist in the mask, returns all of them.
    """
    rng = np.random.default_rng(seed)
    H, W = mask.shape
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return []
    if len(rows) <= n:
        order = np.lexsort((cols, rows))
        return list(zip(rows[order].tolist(), cols[order].tolist()))

    cell_r = (rows * 4 // H).clip(0, 3)
    cell_c = (cols * 4 // W).clip(0, 3)
    cell_id = cell_r * 4 + cell_c   # 0..15

    nonempty = sorted(set(cell_id.tolist()))
    per_cell = math.ceil(n / len(nonempty))

    sites = []
    for cid in nonempty:
        idxs = np.where(cell_id == cid)[0]
        chosen = rng.choice(idxs, size=min(per_cell, len(idxs)), replace=False)
        for i in sorted(chosen.tolist()):
            sites.append((int(rows[i]), int(cols[i])))

    sites = sites[:n]
    sites.sort()
    return sites


# ── Row formatter ──────────────────────────────────────────────────────────────

def format_row(idx: int, class_name: str, lat: float, lon: float, traj: str, url: str) -> str:
    return f"[{idx:02d}] {class_name:<12} | {lat:.4f}, {lon:.4f} | {traj} | {url}"
