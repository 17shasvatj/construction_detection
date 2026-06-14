"""
Training script for Prithvi-EO-2.0 causal construction detection.

Usage:
    # Local CPU smoke-test (no TerraTorch needed):
    python train/train.py --smoke-test

    # Full GPU training (TerraTorch must be installed):
    python train/train.py --device cuda --epochs 50

    # Custom device via env var:
    DEVICE=cuda python train/train.py --epochs 50

Checkpoint selection: wendell val loss is used for early stopping and best-model
selection. This is an explicitly disclosed limitation — transfer metrics reported
by evaluate.py are a mild upper bound because the test region influenced training.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Allow running from project root: python train/train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset import (
    ConstructionDataset,
    BucketSampler,
    construction_collate_fn,
    compute_norm_stats,
    IGNORE_LABEL,
)
from train.model import load_model


# ── AOI roles ──────────────────────────────────────────────────────────────────
TRAIN_AOIS   = ['babcock', 'sunterra']
VAL_AOIS     = ['wendell']   # TEST-REGION SELECTION — see module docstring
DATA_ROOT    = '../data'
OUTPUT_DIR   = 'outputs'
CKPT_DIR     = 'checkpoints'

SEED         = 42
T_MIN        = 4
NUM_CLASSES  = 3


# ── reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── class weights ──────────────────────────────────────────────────────────────

def compute_class_weights(
    aoi_list: List[str],
    data_root: str,
    num_classes: int = NUM_CLASSES,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from train-AOI label pixels.
    Ignores IGNORE_LABEL (255). Prints counts and weights.
    """
    counts = np.zeros(num_classes, dtype=np.int64)

    for aoi_name in aoi_list:
        label_path = os.path.join(data_root, aoi_name, 'label_cube.npy')
        if not os.path.exists(label_path):
            print(f'[train] WARNING: {aoi_name} labels missing, skipping for class weights.')
            continue
        labels = np.load(label_path, mmap_mode='r')
        flat = labels.ravel()
        for c in range(num_classes):
            counts[c] += int(np.sum(flat == c))

    total_non_ignore = counts.sum()
    if total_non_ignore == 0:
        raise RuntimeError('No valid (non-ignore) labels found in train AOIs.')

    freq    = counts / total_non_ignore
    weights = np.where(freq > 0, 1.0 / (freq + 1e-8), 0.0)
    weights = weights / weights.sum() * num_classes   # normalize so mean weight ≈ 1

    print('\n[train] Class weights (inverse frequency):')
    for c, name in enumerate(['baseline', 'grading', 'constructed']):
        print(f'  class {c} ({name:12s}): count={counts[c]:10,}  freq={freq[c]:.4f}  weight={weights[c]:.3f}')
    print()

    return torch.tensor(weights, dtype=torch.float32, device=device)


# ── per-class metrics ──────────────────────────────────────────────────────────

def compute_per_class_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict:
    """Precision, recall, F1, IoU per class (ignore_label=255 excluded)."""
    metrics = {}
    mask = all_labels != IGNORE_LABEL
    preds  = all_preds[mask]
    labels = all_labels[mask]

    for c in range(num_classes):
        tp = int(np.sum((preds == c) & (labels == c)))
        fp = int(np.sum((preds == c) & (labels != c)))
        fn = int(np.sum((preds != c) & (labels == c)))
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        iou       = tp / (tp + fp + fn + 1e-8)
        metrics[c] = dict(precision=precision, recall=recall, f1=f1, iou=iou,
                          tp=tp, fp=fp, fn=fn)
    return metrics


def print_metrics(metrics: Dict, prefix: str = ''):
    names = {0: 'baseline', 1: 'grading', 2: 'constructed'}
    for c, m in metrics.items():
        print(f'{prefix}  class {c} ({names[c]:12s}): '
              f'P={m["precision"]:.3f}  R={m["recall"]:.3f}  '
              f'F1={m["f1"]:.3f}  IoU={m["iou"]:.3f}')


# ── train / eval epoch ─────────────────────────────────────────────────────────

def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for spectral, dates, target, _n_frames in loader:
        spectral = spectral.to(device)   # (B, T, 6, P, P)
        dates    = dates.to(device)       # (B, T)
        target   = target.to(device)      # (B, P, P)

        optimizer.zero_grad()
        logits = model(spectral, temporal_coords=dates)   # (B, C, P, P)
        loss   = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_labels  = []

    for spectral, dates, target, _n_frames in loader:
        spectral = spectral.to(device)
        dates    = dates.to(device)
        target   = target.to(device)

        logits = model(spectral, temporal_coords=dates)
        loss   = criterion(logits, target)

        total_loss += loss.item()
        n_batches  += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(target.cpu().numpy())

    all_preds  = np.concatenate([p.ravel() for p in all_preds])
    all_labels = np.concatenate([l.ravel() for l in all_labels])
    metrics    = compute_per_class_metrics(all_preds, all_labels)

    return total_loss / max(n_batches, 1), metrics


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train Prithvi-EO-2.0 for construction detection')
    p.add_argument('--device',      default=os.environ.get('DEVICE', 'cpu'))
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch-size',  type=int,   default=4)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--patience',    type=int,   default=10)
    p.add_argument('--seed',        type=int,   default=SEED)
    p.add_argument('--data-root',   default=DATA_ROOT)
    p.add_argument('--output-dir',  default=OUTPUT_DIR)
    p.add_argument('--ckpt-dir',    default=CKPT_DIR)
    p.add_argument('--smoke-test',  action='store_true',
                   help='2-epoch end-to-end check on CPU; uses _SmokeStub model')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir,   exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

    if args.smoke_test:
        print('[train] *** SMOKE TEST MODE: 3 examples, 2 epochs, stub model ***')
        args.epochs    = 2
        args.batch_size = 2
        args.patience  = 999  # don't early-stop in smoke test

    # ── normalization stats ────────────────────────────────────────────────────
    norm_stats_path = os.path.join(args.output_dir, 'norm_stats.json')
    if os.path.exists(norm_stats_path):
        print(f'[train] Loading norm stats from {norm_stats_path}')
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
    else:
        norm_stats = compute_norm_stats(TRAIN_AOIS, args.data_root)
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        print(f'[train] Saved norm stats → {norm_stats_path}')

    # ── datasets ───────────────────────────────────────────────────────────────
    print('\n[train] Building train dataset...')
    train_ds = ConstructionDataset(
        aoi_list=TRAIN_AOIS,
        data_root=args.data_root,
        split='train',
        norm_stats=norm_stats,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )

    print('[train] Building val dataset (wendell)...')
    # TEST-REGION SELECTION: wendell used for early stopping / best checkpoint —
    # transfer metrics reported by evaluate.py are a mild upper bound.
    # Explicit, disclosed limitation acceptable for this time budget.
    val_ds = ConstructionDataset(
        aoi_list=VAL_AOIS,
        data_root=args.data_root,
        split='val',
        norm_stats=norm_stats,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )

    # ── loaders ────────────────────────────────────────────────────────────────
    train_sampler = BucketSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_sampler = BucketSampler(
        val_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=construction_collate_fn,
        num_workers=0,    # mmap + fork can deadlock; set >0 only if you verify
        pin_memory=(args.device != 'cpu'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=construction_collate_fn,
        num_workers=0,
        pin_memory=(args.device != 'cpu'),
    )

    # ── class weights ──────────────────────────────────────────────────────────
    class_weights = compute_class_weights(TRAIN_AOIS, args.data_root, device=device)

    # ── model ──────────────────────────────────────────────────────────────────
    num_frames_max = max(train_ds.max_t, val_ds.max_t)
    print(f'\n[train] num_frames_max={num_frames_max}')

    # Rebuild datasets with num_frames_max so __getitem__ pads to fixed length.
    # Must happen AFTER num_frames_max is known but before loaders are used.
    train_ds.num_frames_max = num_frames_max
    val_ds.num_frames_max   = num_frames_max

    model = load_model(
        num_frames_max=num_frames_max,
        num_classes=NUM_CLASSES,
        device=args.device,
        smoke_test=args.smoke_test,
        patch_size=128,
    )

    # ── optimizer + loss ───────────────────────────────────────────────────────
    head_params = model.head_params() if hasattr(model, 'head_params') else list(model.parameters())
    optimizer   = AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    scheduler   = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_LABEL)

    # ── run config ─────────────────────────────────────────────────────────────
    run_config = {
        'timestamp':      datetime.now().isoformat(),
        'seed':           args.seed,
        'device':         args.device,
        'epochs':         args.epochs,
        'batch_size':     args.batch_size,
        'lr':             args.lr,
        'patience':       args.patience,
        'smoke_test':     args.smoke_test,
        'train_aois':     TRAIN_AOIS,
        'val_aois':       VAL_AOIS,
        'data_root':      args.data_root,
        'norm_stats_path': norm_stats_path,
        'class_weights':  class_weights.tolist(),
        'num_frames_max': num_frames_max,
        'n_train_examples': len(train_ds),
        'n_val_examples':   len(val_ds),
        'val_selection_note': (
            'wendell used for early stopping — transfer metrics are a mild upper bound '
            '(disclosed limitation)'
        ),
    }
    run_config_path = os.path.join(args.output_dir, 'run_config.json')
    with open(run_config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f'[train] Run config saved → {run_config_path}\n')

    # ── training loop ──────────────────────────────────────────────────────────
    best_val_loss  = float('inf')
    epochs_no_improve = 0
    history = []

    ckpt_path = os.path.join(args.ckpt_dir, 'best.pt')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_sampler.set_epoch(epoch)

        train_loss            = train_epoch(train_loader, model, optimizer, criterion, device)
        val_loss, val_metrics = eval_epoch(val_loader,   model, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f'[train] Epoch {epoch:3d}/{args.epochs}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.1f}s)')
        print_metrics(val_metrics, prefix='[train]')

        epoch_record = {
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'metrics':    val_metrics,
        }
        history.append(epoch_record)

        # ── checkpoint ────────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch':           epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':        val_loss,
                'class_weights':   class_weights.tolist(),
                'norm_stats_path': norm_stats_path,
                'config':          run_config,
            }, ckpt_path)
            print(f'[train]   ✓ New best checkpoint (val_loss={val_loss:.4f}) → {ckpt_path}')
        else:
            epochs_no_improve += 1
            print(f'[train]   No improvement ({epochs_no_improve}/{args.patience})')

        # ── early stopping ────────────────────────────────────────────────────
        if epochs_no_improve >= args.patience:
            print(f'[train] Early stopping at epoch {epoch} (patience={args.patience}).')
            break

    # ── save training history ─────────────────────────────────────────────────
    history_path = os.path.join(args.output_dir, 'training_history.json')

    # Convert numpy int/float in metrics to plain Python for JSON serialisation
    def _to_serialisable(obj):
        if isinstance(obj, dict):
            return {str(k): _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, list):
            return [_to_serialisable(x) for x in obj]
        return obj

    with open(history_path, 'w') as f:
        json.dump(_to_serialisable(history), f, indent=2)
    print(f'\n[train] Training history saved → {history_path}')
    print(f'[train] Best val_loss={best_val_loss:.4f}  checkpoint → {ckpt_path}')

    if args.smoke_test:
        print('\n[train] Smoke test completed successfully.')


if __name__ == '__main__':
    main()
