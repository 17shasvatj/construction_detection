"""
Non-degeneracy test: is the model a genuine construction detector, or just a
vegetation-loss (veg->bare) detector?

Logic:
  Construction pixels in training are veg->bare->built. If the model merely learned
  "vegetation turned to bare soil," it will ALSO fire on pixels that go
  veg->bare->veg (agricultural/seasonal/drought bare that REVEGETATES and never builds).

  We find veg->bare->veg pixels (bare at some point, but vegetation at the end,
  never built) and measure what fraction the model wrongly calls construction
  (grading=1 or built=2) during/after their bare phase.

  LOW false-positive rate  -> model distinguishes construction from generic bare soil
                              (uses geometry / irreversibility) -> NOT a degenerate
                              veg->bare detector. Strong non-circularity evidence.
  HIGH false-positive rate -> model is substantially a vegetation-loss detector.
                              Honest caveat required.

Run from ~/construction_detection/train/:
    python circularity_test.py --device cuda
Optionally test multiple AOIs:
    python circularity_test.py --device cuda --aois wendell sunterra babcock
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset import load_aoi, IGNORE_LABEL, K, T_MIN
from train.model import load_model
from train.evaluate import run_tiled_inference

# DW class codes
DW_WATER, DW_TREES, DW_GRASS, DW_FLOODVEG, DW_CROPS, DW_SHRUB, DW_BUILT, DW_BARE, DW_SNOW = range(9)
DW_VEG = [DW_TREES, DW_GRASS, DW_SHRUB, DW_CROPS]   # vegetation classes


def find_veg_bare_veg_pixels(dw):
    """
    Return a (H, W) bool mask of pixels that:
      - start as vegetation
      - go bare at some point (DW bare)
      - end as vegetation (NOT built)
      - never reach built
    These are agricultural / seasonal / drought bare -> revegetated. NOT construction.
    Also return, per such pixel, the first bare quarter (for scoring the bare-phase onward).
    """
    T, H, W = dw.shape
    was_veg_start = np.isin(dw[0], DW_VEG)
    ends_veg      = np.isin(dw[-1], DW_VEG)
    ever_built    = np.any(dw == DW_BUILT, axis=0)
    ever_bare     = np.any(dw == DW_BARE, axis=0)

    veg_bare_veg = was_veg_start & ends_veg & ever_bare & (~ever_built)

    first_bare = np.full((H, W), -1, dtype=np.int32)
    ys, xs = np.where(veg_bare_veg)
    for y, x in zip(ys, xs):
        for t in range(T):
            if dw[t, y, x] == DW_BARE:
                first_bare[y, x] = t
                break
    return veg_bare_veg, first_bare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=os.environ.get('DEVICE', 'cuda'))
    ap.add_argument('--ckpt', default='checkpoints/best.pt')
    ap.add_argument('--norm', default='outputs/norm_stats.json')
    ap.add_argument('--data-root', default='../data')
    ap.add_argument('--aois', nargs='+', default=['wendell'])
    args = ap.parse_args()

    device = torch.device(args.device)
    norm_stats = json.load(open(args.norm))

    ckpt = torch.load(args.ckpt, map_location=device)
    model = load_model(num_frames_max=K, num_classes=3, device=args.device, smoke_test=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'[circ] loaded checkpoint epoch={ckpt.get("epoch","?")}')

    for aoi in args.aois:
        print(f'\n[circ] ==== {aoi.upper()} ====')
        data = load_aoi(aoi, args.data_root)
        if data is None:
            print(f'[circ] {aoi}: no data, skipping.')
            continue

        dw_path = os.path.join(args.data_root, aoi, 'dw_cube.npy')
        if not os.path.exists(dw_path):
            print(f'[circ] {aoi}: no dw_cube.npy, skipping.')
            continue
        dw = np.load(dw_path)                      # (T, H, W)
        label_cube = data['labels'][:]
        T = data['T']

        # Predictions at every t (causal)
        pred_cube = run_tiled_inference(model, data, norm_stats, device, t_start=T_MIN)

        # --- 1) The key test: veg->bare->veg (NOT construction) ---
        vbv_mask, first_bare = find_veg_bare_veg_pixels(dw)
        n_vbv = int(vbv_mask.sum())
        print(f'[circ] veg->bare->veg (non-construction bare) pixels: {n_vbv:,}')

        if n_vbv > 0:
            # For each such pixel, look at predictions from its first-bare quarter onward.
            # Count (pixel, t) instances the model WRONGLY calls construction (1 or 2).
            fp_instances = 0
            total_instances = 0
            ys, xs = np.where(vbv_mask)
            for y, x in zip(ys, xs):
                fb = first_bare[y, x]
                if fb < 0:
                    continue
                for t in range(max(fb, T_MIN), T):
                    p = pred_cube[t, y, x]
                    if p == IGNORE_LABEL:
                        continue
                    total_instances += 1
                    if p in (1, 2):              # model calls construction on non-construction bare
                        fp_instances += 1
            fp_rate = fp_instances / max(total_instances, 1)
            print(f'[circ]   scored (pixel,t) instances:    {total_instances:,}')
            print(f'[circ]   wrongly called construction:   {fp_instances:,}')
            print(f'[circ]   FALSE-POSITIVE RATE on non-construction bare: {fp_rate:.3f}')
            print(f'[circ]   --> LOW (<~0.15) = model distinguishes construction from generic bare soil')
            print(f'[circ]   --> HIGH (>~0.4) = model is substantially a vegetation-loss detector')
        else:
            print(f'[circ]   (no veg->bare->veg pixels in this AOI to test)')

        # --- 2) Reference: true-positive rate on genuine construction (veg->bare->built) ---
        # On confirmed construction pixels, how often does the model call construction?
        constr_mask = np.any((label_cube == 1) | (label_cube == 2), axis=0)
        tp_inst = 0
        tot_inst = 0
        ys, xs = np.where(constr_mask)
        # subsample for speed if huge
        if len(ys) > 200000:
            idx = np.random.default_rng(0).choice(len(ys), 200000, replace=False)
            ys, xs = ys[idx], xs[idx]
        for y, x in zip(ys, xs):
            for t in range(T_MIN, T):
                lab = label_cube[t, y, x]
                if lab in (1, 2):
                    p = pred_cube[t, y, x]
                    if p == IGNORE_LABEL:
                        continue
                    tot_inst += 1
                    if p in (1, 2):
                        tp_inst += 1
        tp_rate = tp_inst / max(tot_inst, 1)
        print(f'[circ]   (reference) detection rate on TRUE construction: {tp_rate:.3f} '
              f'({tp_inst:,}/{tot_inst:,})')

        # --- 3) Discrimination summary ---
        if n_vbv > 0 and tot_inst > 0:
            print(f'\n[circ]   DISCRIMINATION:')
            print(f'[circ]     detects real construction at {tp_rate:.1%}')
            print(f'[circ]     false-fires on non-construction bare at {fp_rate:.1%}')
            if fp_rate < 0.5 * tp_rate:
                print(f'[circ]     => fires MORE on real construction than on generic bare '
                      f'soil: evidence it is NOT a degenerate veg->bare detector.')
            else:
                print(f'[circ]     => fires comparably on generic bare soil: model is '
                      f'substantially a vegetation-loss detector (honest caveat needed).')


if __name__ == '__main__':
    main()