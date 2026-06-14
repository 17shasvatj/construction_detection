"""
Step 3: Generate confirmed veg→bare→built trajectory labels.

Labels: 0=baseline, 1=grading, 2=constructed, 255=excluded (active grading).
Temporal ordering enforced: bare phase must precede built phase.

Standalone: python -m pipeline.generate_labels --aoi babcock
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def run(data_dir):
    data_dir = Path(data_dir)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    quarters = meta["quarters"]
    H, W = meta["shape"][1], meta["shape"][2]
    T = len(quarters)

    dw   = np.load(data_dir / "dw_cube.npy")
    ndvi = np.load(data_dir / "ndvi_cube.npy")

    # ── Identify confirmed construction pixels ────────────────────────────
    print("[1] Finding confirmed veg→bare→built trajectories")
    was_veg   = np.isin(dw[0], [1, 2, 5])
    now_built = dw[-1] == 6

    bare_start  = np.full((H, W), -1, dtype=np.int16)
    built_start = np.full((H, W), -1, dtype=np.int16)

    contaminated = 0

    ys, xs = np.where(was_veg & now_built)
    print(f"    Processing {len(ys)} veg→built pixels...")
    for y, x in zip(ys, xs):
        for t in range(T):
            if dw[t, y, x] == 7 and bare_start[y, x] == -1:
                bare_start[y, x] = t
                break

        if bare_start[y, x] >= 0:
            for t in range(bare_start[y, x], T):
                if dw[t, y, x] == 6:
                    built_start[y, x] = t
                    break

        first_bare       = bare_start[y, x]
        first_built_ever = next((t for t in range(T) if dw[t, y, x] == 6), -1)
        if first_built_ever >= 0 and first_bare >= 0 and first_built_ever < first_bare:
            contaminated += 1

        if bare_start[y, x] < 0:
            for t in range(T):
                if dw[t, y, x] == 6:
                    built_start[y, x] = t
                    break

    print(f"    Flicker (built before bare): {contaminated} pixels")

    confirmed = (bare_start >= 0) & (built_start >= 0) & (built_start > bare_start) & was_veg & now_built
    direct    = (bare_start < 0)  & (built_start >= 0) & was_veg & now_built
    print(f"    Confirmed grading (veg→bare→built, ordered): {confirmed.sum()}")
    print(f"    Direct construction (veg→built, no bare):    {direct.sum()}")

    # ── Active grading (veg→bare now, not built yet) ─────────────────────
    now_bare      = dw[-1] == 7
    active_grading = was_veg & now_bare
    active_bare_start = np.full((H, W), -1, dtype=np.int16)
    ys2, xs2 = np.where(active_grading)
    for y, x in zip(ys2, xs2):
        for t in range(T):
            if dw[t, y, x] == 7 and active_bare_start[y, x] == -1:
                active_bare_start[y, x] = t
                break
    print(f"    Active grading (veg→bare now):               {active_grading.sum()}")

    # ── Baseline: stayed vegetation throughout ────────────────────────────
    baseline = was_veg & np.isin(dw[-1], [1, 2, 5])
    print(f"    Baseline (stayed vegetation):                {baseline.sum()}")

    # ── Assign temporal labels ────────────────────────────────────────────
    print("\n[2] Assigning temporal labels")
    label_cube = np.full((T, H, W), 255, dtype=np.uint8)

    label_cube[:, baseline] = 0

    ys, xs = np.where(confirmed)
    for y, x in zip(ys, xs):
        bs = bare_start[y, x]
        bt = built_start[y, x]
        label_cube[:bs, y, x] = 0
        label_cube[bs:bt, y, x] = 1
        label_cube[bt:, y, x] = 2

    ys, xs = np.where(direct)
    for y, x in zip(ys, xs):
        bt = built_start[y, x]
        label_cube[:bt, y, x] = 0
        label_cube[bt:, y, x] = 2

    ys, xs = np.where(active_grading & (active_bare_start >= 0))
    for y, x in zip(ys, xs):
        bs = active_bare_start[y, x]
        label_cube[:bs, y, x] = 0
        label_cube[bs:, y, x] = 255

    np.save(data_dir / "label_cube.npy", label_cube)
    print(f"    Saved label_cube.npy {label_cube.shape}")

    # ── Print counts per quarter ──────────────────────────────────────────
    print("\n[3] Class counts per quarter:")
    for t in range(T):
        b = (label_cube[t] == 0).sum()
        g = (label_cube[t] == 1).sum()
        c = (label_cube[t] == 2).sum()
        x = (label_cube[t] == 255).sum()
        print(f"    {quarters[t]}: baseline={b:>6} grading={g:>5} constructed={c:>5} excluded={x:>6}")

    # ── NDVI validation ───────────────────────────────────────────────────
    print("\n[4] NDVI per class (latest quarter):")
    ndvi_latest = ndvi[-1]
    for cls, name in [(0, "baseline"), (1, "grading"), (2, "constructed")]:
        mask = label_cube[-1] == cls
        if mask.sum() > 0:
            v = ndvi_latest[mask]
            print(f"    {name:12}: n={mask.sum():>6} NDVI={np.nanmean(v):.3f}±{np.nanstd(v):.3f}")

    # ── Visualize ─────────────────────────────────────────────────────────
    print("\n[5] Visualizing")

    def norm_rgb(data):
        arr = data[[2, 1, 0]].copy()
        out = np.zeros_like(arr)
        for i in range(3):
            b = arr[i]
            v = b[~np.isnan(b)]
            if len(v):
                lo, hi = np.percentile(v, 2), np.percentile(v, 98)
                out[i] = np.clip((b - lo) / (hi - lo + 1e-8), 0, 1)
        return np.nan_to_num(np.transpose(out, (1, 2, 0)))

    show_idx = [0, T // 2, T - 1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for col, ti in enumerate(show_idx):
        comp = np.load(data_dir / f"composite_{quarters[ti]}.npy")
        rgb  = norm_rgb(comp)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"RGB — {quarters[ti]}")
        axes[0, col].axis('off')

        axes[1, col].imshow(rgb, alpha=0.4)
        lbl = label_cube[ti]
        for val, cmap in [(1, 'Oranges'), (2, 'Reds')]:
            mask = lbl == val
            if mask.sum() > 0:
                axes[1, col].imshow(np.where(mask, 1, np.nan),
                                    cmap=cmap, alpha=0.7, vmin=0, vmax=1)
        axes[1, col].set_title(f"Labels — {quarters[ti]}")
        axes[1, col].axis('off')

    axes[1, 0].legend(handles=[
        mpatches.Patch(color='green',  alpha=0.5, label='Baseline'),
        mpatches.Patch(color='orange', alpha=0.7, label='Grading'),
        mpatches.Patch(color='red',    alpha=0.7, label='Constructed'),
    ], loc='lower left', fontsize=10)

    aoi_name = meta.get("aoi", str(data_dir.name))
    plt.suptitle(
        f"{aoi_name} — Confirmed trajectory pseudo-labels\n"
        "(active grading excluded, temporal order enforced)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(data_dir / "pseudo_labels.png", dpi=150, bbox_inches='tight')
    print("    Saved pseudo_labels.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    for cls, color, name in [(0, 'green', 'Baseline'), (1, 'orange', 'Grading'), (2, 'red', 'Constructed')]:
        counts = [(label_cube[t] == cls).sum() for t in range(T)]
        ax.plot(range(T), counts, '.-', color=color, lw=2, label=name)
    excl = [(label_cube[t] == 255).sum() for t in range(T)]
    ax.plot(range(T), excl, '.--', color='gray', lw=1, alpha=0.5, label='Excluded')

    ax.set_xticks(range(T))
    ax.set_xticklabels(quarters, rotation=90, fontsize=7)
    ax.set_ylabel("Pixel count")
    ax.set_title(f"{aoi_name} — Class counts over time\n"
                 "Baseline ↓ as grading ↑ then converts to constructed ↑")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(data_dir / "class_timeline.png", dpi=150, bbox_inches='tight')
    print("    Saved class_timeline.png")

    print("\n" + "=" * 50)
    print("DONE — check pseudo_labels.png and class_timeline.png")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import AOIS, DATA_ROOT

    parser = argparse.ArgumentParser(description="Generate confirmed-trajectory labels")
    parser.add_argument("--aoi", required=True, choices=list(AOIS))
    args = parser.parse_args()

    run(Path(DATA_ROOT) / args.aoi)
