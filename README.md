# Construction Detection

Detect land grading and construction from Sentinel-2 time-series imagery using a
fine-tuned [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
backbone. The model sees the last 6 quarterly composites (K=6 causal window) and predicts
per-pixel state at the current quarter: **baseline / grading / constructed**.

---

## Repository layout

```
pipeline/           Data prep modules (pull composites, pull DW, generate labels)
train/
  dataset.py        ConstructionDataset, normalization, fixed K=6 window
  model.py          Prithvi-EO-2.0 wrapper via TerraTorch
  train.py          Training loop
  evaluate.py       Per-timepoint eval, trajectory figures, failure demos
  requirements.txt  GPU VM dependencies
data/{aoi}/         Per-AOI cubes produced by the data pipeline
  spectral_cube.npy   (T, 6, H, W) float64 — Sentinel-2 surface reflectance
  dw_cube.npy         (T, H, W) uint8      — Dynamic World land cover
  label_cube.npy      (T, H, W) uint8      — 0=baseline 1=grading 2=constructed 255=excluded
  metadata.json
checkpoints/        best.pt saved by train.py
outputs/            Metrics JSON, figures, norm stats
config.py           AOI registry (name, bbox, biome, role)
run_pipeline.py     Orchestrates the data pipeline steps
validate_labels.py  Google Earth spot-check from DW-derived labels (stdout only)
make_demo.py        Reviewer demo: map.png, metrics.json, demo.md
```

---

## Setup

### Data pipeline dependencies
```bash
pip install earthengine-api numpy
earthengine authenticate          # one-time OAuth login
```

### Training dependencies (GPU VM)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install terratorch timm einops matplotlib tqdm
# verify TerraTorch:
python -c "from terratorch.models import EncoderDecoderFactory; print('ok')"
```

### Local smoke-test (CPU, no TerraTorch needed)
```bash
pip install torch torchvision numpy matplotlib tqdm
```

---

## Step 1 — Data pipeline

Runs per-AOI. Each step is skipped automatically if its output already exists;
use `--force` to re-run everything.

```bash
python run_pipeline.py --aoi babcock
python run_pipeline.py --aoi sunterra
python run_pipeline.py --aoi wendell
```

**Steps executed in order:**

| Step | Output | What it does |
|------|--------|--------------|
| `pull_composites` | `spectral_cube.npy` | Downloads quarterly Sentinel-2 composites from Google Earth Engine |
| `pull_dw` | `dw_cube.npy` | Downloads Dynamic World land-cover time series |
| `generate_labels` | `label_cube.npy` | Derives grading/constructed labels from veg→bare→built trajectories |

To re-run a single AOI from scratch:
```bash
python run_pipeline.py --aoi babcock --force
```

**Label validation** (optional, after `generate_labels`):
```bash
python validate_labels.py --aoi babcock
```
Prints NDVI separation, density check, mean NDVI trajectory, and DW class distribution
for confirmed pixels — sanity-check on the labeling method's outputs.

---

## Step 2 — Train

All training commands run from the **project root**. The `--data-root data` flag
overrides the default path so the scripts find `data/{aoi}/` correctly.

### Smoke-test (CPU, <1 min, no TerraTorch needed)
Verifies the full code path — dataset, model stub, loss, checkpoint — without a GPU or real weights.

```bash
python train/train.py --smoke-test --data-root data
```

### Full training (GPU)
```bash
python train/train.py --device cuda --data-root data
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cpu` (or `$DEVICE`) | `cpu` or `cuda` |
| `--epochs` | `50` | Maximum epochs |
| `--batch-size` | `4` | Patches per batch |
| `--lr` | `1e-3` | AdamW learning rate |
| `--patience` | `15` | Early-stopping patience (epochs) |
| `--data-root` | `../data` | Path to `data/` directory |
| `--ckpt-dir` | `checkpoints` | Where `best.pt` is saved |
| `--output-dir` | `outputs` | Where `run_config.json` and `norm_stats.json` go |
| `--smoke-test` | off | Tiny stub model, 2 epochs, CPU only |

**Train AOIs** (hardcoded in `train/train.py`): `babcock`, `sunterra`, `santa_rita_ranch`  
**Val AOI**: `wendell` — used for early stopping and checkpoint selection.
> **Disclosed limitation**: wendell val loss drives model selection, so transfer metrics
> reported by `evaluate.py` are a mild upper bound for that region.

Training prints first-batch `spectral min/max/mean` — confirm values are in the `~0–1`
reflectance range, not `0–10000` HLS scale. NaN inputs raise immediately.

**Outputs after training:**
```
checkpoints/best.pt          Model weights + run config
outputs/run_config.json      Resolved hyperparameters, AOIs, class weights
outputs/norm_stats.json      Per-band mean/std used for normalization
```

---

## Step 3 — Evaluate

Loads checkpoints/best.pt and runs tiled inference on the held-out
evaluation AOIs (default: wendell). Use --aoi to evaluate any AOI
with prepared data.

python train/evaluate.py --device cuda --data-root data
python train/evaluate.py --aoi lakewood_ranch --device cuda --data-root data

Key options:

Flag           Default                Description
--ckpt         checkpoints/best.pt    Checkpoint to load
--device       cpu (or $DEVICE)       cpu or cuda
--data-root    ../data                Path to data/ directory
--output-dir   outputs                Where results and figures go
--aoi          (none)                 Evaluate a single AOI (overrides VAL_AOIS;
                                      data/<aoi>/ must contain spectral_cube.npy,
                                      label_cube.npy, metadata.json). Failure
                                      demos are skipped when --aoi is set.
--smoke-test   off                    Fast run on sunterra only

Console output:
  Per-timepoint P/R/F1/IoU per class, followed by a summary table
  with macro-averages, peak F1 per class, recall-vs-history
  snapshots, final-quarter persistent-class metrics, and
  mid-trajectory grading range (ready to drop into the report).

Outputs:
  outputs/eval_results.json          Per-timepoint P/R/F1/IoU for all eval AOIs
  outputs/figures/
    {aoi}_trajectory_maps.png        Predicted vs DW label maps per quarter
    {aoi}_per_timepoint_curve.png    F1/IoU vs history length
    {aoi}_pixel_trajectories.png     Sample pixel class timelines (model vs DW)
    {aoi}_early_detection.png        Pixels predicted before DW confirms built
    {aoi}_active_grading_t{t}.png    Active-grading qualitative map (no precision)
    {aoi}_failure_demo_t{t}.png      Failure-demo maps (default run only)
```

---

## Step 4 — Reviewer demo and spot-check (optional)

After evaluation, generate a one-page reviewer artifact and a list of sites to verify
manually in Google Earth.

**Reviewer demo** (requires `predictions.npy` at `data/{aoi}/predictions.npy`):
```bash
python make_demo.py wendell
```

Writes to `outputs/demo/wendell/`:
- `map.png` — 3-panel figure: model prediction | DW labels | disagreement
- `metrics.json` — per-pixel P/R vs DW labels + placeholder Google Earth tally
- `demo.md` — one-page markdown for Fuxun (embeds map + metrics + site list)

Use `--dry-run` to preview what would be sampled/written without touching disk:
```bash
python make_demo.py wendell --dry-run
```

---

## Quick reference

```bash
# 1. Fetch data for all train + val AOIs
for aoi in babcock sunterra wendell; do
    python run_pipeline.py --aoi $aoi
done

# 2. Smoke-test the training code locally (CPU, no GPU or TerraTorch needed)
python train/train.py --smoke-test --data-root data

# 3. Train on GPU VM (after scp / clone)
python train/train.py --device cuda --data-root data --epochs 50

# 4. Evaluate
python train/evaluate.py --device cuda --data-root data

# 5. Validate labels (optional, after generate_labels)
python validate_labels.py --aoi wendell
```
