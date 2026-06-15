import numpy as np
import json, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root for `train.` imports

from train.dataset import load_aoi, IGNORE_LABEL, K, T_MIN
from train.model import load_model
from train.evaluate import run_tiled_inference

device = torch.device('cuda')

# paths relative to ~/construction_detection/train/
norm_stats = json.load(open('outputs/norm_stats.json'))      # train/outputs/norm_stats.json
CKPT = 'checkpoints/best.pt'                                  # train/checkpoints/best.pt
DATA_ROOT = '../data'                                         # ~/construction_detection/data

ckpt = torch.load(CKPT, map_location=device)
model = load_model(num_frames_max=K, num_classes=3, device='cuda', smoke_test=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

data = load_aoi('wendell', DATA_ROOT)
label_cube = data['labels'][:]
spec       = np.array(data['spectral'][:])
print(f"[diag] spectral shape {spec.shape}, label shape {label_cube.shape}")
pred_cube = run_tiled_inference(model, data, norm_stats, device, t_start=T_MIN)

mean = np.array(norm_stats['mean'], dtype=np.float32)
std  = np.array(norm_stats['std'],  dtype=np.float32)
std  = np.where(std < 1e-8, 1.0, std)
data_scale = float(norm_stats.get('data_scale', 1.0))

spec_real = spec * std[None, :, None, None] + mean[None, :, None, None]
if data_scale != 1.0:
    spec_real = spec_real / data_scale

red = spec_real[:, 2, :, :]
nir = spec_real[:, 3, :, :]
ndvi = (nir - red) / (nir + red + 1e-8)
print(f"[diag] NDVI range: min={np.nanmin(ndvi):.3f} max={np.nanmax(ndvi):.3f} mean={np.nanmean(ndvi):.3f}")

labeled_constr = (label_cube == 1) | (label_cube == 2)
valid          = label_cube != IGNORE_LABEL
missed   = labeled_constr & (pred_cube == 0) & valid
detected = labeled_constr & ((pred_cube == 1) | (pred_cube == 2)) & valid

m_ndvi = ndvi[missed];   m_ndvi = m_ndvi[~np.isnan(m_ndvi)]
d_ndvi = ndvi[detected]; d_ndvi = d_ndvi[~np.isnan(d_ndvi)]

print(f"\n[diag] Labeled construction: {int(labeled_constr.sum()):,}")
print(f"[diag]   Missed (pred baseline): {int(missed.sum()):,}")
print(f"[diag]   Detected:               {int(detected.sum()):,}")
if len(m_ndvi) and len(d_ndvi):
    print(f"\n[diag] MISSED   NDVI: mean={m_ndvi.mean():.3f} median={np.median(m_ndvi):.3f}")
    print(f"[diag] DETECTED NDVI: mean={d_ndvi.mean():.3f} median={np.median(d_ndvi):.3f}")
    print(f"\n  MISSED ≈ DETECTED (both low) → labels fine, model conservative → CALIBRATION")
    print(f"  MISSED >> DETECTED (missed vegetated) → LABEL problem")

missed_grading = (label_cube == 1) & (pred_cube == 0) & valid
mg = ndvi[missed_grading]; mg = mg[~np.isnan(mg)]
if len(mg):
    print(f"\n[diag] MISSED-GRADING NDVI: mean={mg.mean():.3f} median={np.median(mg):.3f} (n={len(mg):,})")