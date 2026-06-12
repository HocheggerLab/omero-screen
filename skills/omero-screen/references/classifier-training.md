# Classifier Training — End-to-End Workflow

CNN-based cell classification: generate labelled training crops via napari → build a dataset → train PyTorch models → extract for inference → apply in omero-screen.

Package: `packages/cellclass/`
CLI entry points: `cellclass-dataset`, `cellclass-train`, `cellclass-test`, `cellclass-extract`, `cellclass-batch`, `cellclass-sbatch`, `cellclass-sample`

---

## Overview

```
OMERO plate
    ↓ [napari Training widget — label cell crops → saves .npy files]
    ↓ [cellclass-dataset — convert .npy files to .npz dataset]
    ↓ [write train.txt — one training run per line]
    ↓ [cellclass-batch — generate batch.sh from train.txt]
    ↓ [bash batch.sh — run all training jobs]
    ↓ [cellclass-extract — extract best checkpoint to TorchScript .pt]
    ↓ [omero-screen --inference model.pt — apply at scale]
```

---

## Data Format

Training data is stored as `.npy` files (pickled dicts):

```python
{
  'data': ([img1, img2, ...], [mask1, mask2, ...]),  # images (YXC), masks (YX)
  'target': ['normal', 'micronuclei', ...]            # string class labels
}
```

A `metadata.json` must be present in the data directory:

```json
{
  "user_data": {
    "channels": ["DAPI", "Tub"]
  }
}
```

These files are generated automatically by the napari **Training** widget when you label crops.

---

## Step 1 — Generate Training Data in Napari

1. Launch napari: `napari`
2. Open the **Setup Training** widget → create a new classifier, define class names, set crop size
3. Open the **Training** widget → select classifier → load a plate/well → label crops with number keys
4. Data is saved as `.npy` files in the training data directory
5. Use the **Session Manager** widget to track and resume labelling sessions

See `references/napari.md` for full napari widget documentation.

Recommended minimum: **≥200 labelled crops per class** before training.

---

## Step 2 — Create Dataset

Convert `.npy` files to a single normalised `.npz` for training:

```bash
cellclass-dataset <data_dir> --name rois
```

This produces `<data_dir>/rois.npz`.

**What it does:**
- Reads all `.npy` files in `<data_dir>/`
- Reads channel names from `metadata.json` (or pass `--channels DAPI Tub`)
- Applies (1, 99) percentile normalisation → converts to uint8
- Masks each crop to the target cell's ROI (centres on the labelled cell)
- Deduplicates exact copies (sha256 hash)
- Skips `unassigned` labels by default

```bash
# Full options
cellclass-dataset <data_dir> \
    --name rois \             # output filename stem (default: rois)
    --out /output/dir \       # output directory (default: data_dir)
    --channels DAPI Tub \     # override channel names from metadata.json
    --ignore unassigned difficult \  # labels to exclude (default: unassigned)
    --single-label \          # only use crops where a single cell label exists (default: on)
    --duplicates              # log duplicate images
```

**Output statistics printed:** class counts and proportions, duplicate rate, mask shift stats.

---

## Step 3 — Write train.txt

Create a plain-text file with one training run per line. Lines starting with `#` are skipped (use to comment out completed runs).

**Standard recipe for 50×50 px crops (sweeps 4 models × 2 learning rates):**

```
# micronuclei training — 2025-11-01
/path/to/data/rois.npz --model efficientnetb3s --freeze-weights --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 64 -d cuda --cudnn-benchmark --pin-memory --num-workers 4
/path/to/data/rois.npz --model efficientnetb3s --freeze-weights --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 64 -d cuda --cudnn-benchmark --pin-memory --num-workers 4
/path/to/data/rois.npz --model densenet121 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 64 -d cuda --cudnn-benchmark --pin-memory --num-workers 4
/path/to/data/rois.npz --model densenet121 --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 64 -d cuda --cudnn-benchmark --pin-memory --num-workers 4
/path/to/data/rois.npz --model shufflenet2x1_0 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 128 -d cuda --cudnn-benchmark --pin-memory --num-workers 4
/path/to/data/rois.npz --model squeezenet1_0 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 128 -d cuda --cudnn-benchmark --pin-memory --num-workers 4
```

**Model selection guide:**

| Model | Use case | `--freeze-weights` | Batch size |
|---|---|---|---|
| `efficientnetb3s` | Primary — ImageNet pretrained, 300px | Yes | 64 |
| `densenet121` | Primary — good general baseline | No | 64 |
| `shufflenet2x1_0` | Lightweight baseline | No | 128 |
| `squeezenet1_0` | Lightweight baseline | No | 128 |
| `densenet161/169/201` | Larger capacity, more data needed | No | 32 |
| `efficientnetb4s` | Large pretrained, 380px | Yes | 32 |

**Key training flags:**

| Flag | Default | Notes |
|---|---|---|
| `--model` | `densenet121` | See model table above |
| `--lr` | `1e-4` | Sweep 1e-3 and 1e-4 as minimum |
| `--lr-scheduler` | `step` | Use `plateau` for cell data — adapts to variable convergence |
| `--flip` | `1` | `0`=none; `1`=horizontal; `2`=vertical; `3`=both (cells have no orientation bias → use 3) |
| `--rotate` | `180` | Random rotation range ±degrees |
| `--translate` | `0.1` | Random translation ±10% of image size |
| `--batch-size` | `32` | 64 for heavy models, 128 for lightweight |
| `--epochs` | `2000` | Early stopping kicks in before this |
| `--patience` | `10` | Epochs without val-loss improvement before stopping |
| `--loss-function` | `focal_loss` | Use `cross_entropy` if classes are balanced |
| `--loss-weights` | off | Add inverse-frequency weights to loss — helps with class imbalance |
| `--freeze-weights` | off | Freeze pretrained layers; only train input conv + classifier head |
| `--dropout` | `0.4` | Regularisation in final layers |
| `--device` | `cuda` | `cuda` or `cpu` |

---

## Step 4 — Generate and Run batch.sh

```bash
# Generate batch.sh from train.txt
cellclass-batch train.txt --script batch.sh

# Run locally (sequential)
bash batch.sh

# OR on SLURM (Artemis HPC) — one job per run
cellclass-batch train.txt --script batch.sh --cmd "./src/bin/sbatch_training.py --args"
bash batch.sh
```

`cellclass-batch` auto-increments output filenames: if `rois.1.pt` exists, next run is `rois.2.pt`. Always adds `--wandb` to each command automatically.

**Weights & Biases setup (one-time):**
```bash
wandb login       # stores credentials in ~/.netrc
wandb status      # verify before batch jobs
```
Runs log to the `cellclass` project. Use `--entity hocheggerlab` to log to the shared team workspace.

**Comment out completed runs in `train.txt` before regenerating `batch.sh`** — this prevents overwriting finished checkpoints.

---

## Step 5 — Evaluate Models

```bash
# Print classification report for a dataset against a trained checkpoint
cellclass-test rois.npz --model densenet121 --name model.pt
```

Or check W&B run metrics for the best validation F1 across runs.

---

## Step 6 — Extract Best Model for Inference

```bash
# Extract from a training state file (saves .pt + .json sidecar)
cellclass-extract training.json --save
```

Output: `<model>_c<N_channels>_l<N_labels>.pt` and matching `.json` metadata sidecar.

The `.json` sidecar contains: labels, channels, epoch, val/test F1, precision, recall.

The extracted `.pt` is a TorchScript model loadable by `omero-screen --inference`.

**To continue a stopped training run:**
```bash
cellclass-train training.json    # reads state, loads last checkpoint, resumes
```

---

## Step 7 — Apply Classifier in omero-screen

```bash
# Single model
omero-screen 1821 --inference micronuclei.pt

# Multiple models simultaneously
omero-screen 1821 --inference micronuclei.pt mitotic_index.pt --gallery 15 --batch 32
```

Results added to `final_data_cc.csv` as a new column per classifier, plus gallery PNGs attached to the OMERO plate.

---

## File Naming Convention

Training outputs follow the pattern `<dataset_stem>.<run_number>.pt`:
- `rois.1.pt` — first run
- `rois.1.pt.best` — best checkpoint from that run
- `rois.1.json` — training state (use this to resume or extract)

---

## Complete Example: Micronuclei Classifier

```bash
# 1. Create dataset from napari-generated NPY files
cd /data/micronuclei-training
cellclass-dataset . --name rois

# 2. Write train.txt (see Step 3 recipe above)
# 3. Generate and run batch
cellclass-batch train.txt --script batch.sh
bash batch.sh

# 4. Check W&B for best F1; identify winning run (e.g. rois.3.json)

# 5. Extract
cellclass-extract rois.3.json --save
# → produces micronuclei_c2_l2.pt + micronuclei_c2_l2.json

# 6. Apply
omero-screen 1821 1822 1823 --inference micronuclei_c2_l2.pt --env production
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Missing metadata file` in `cellclass-dataset` | Add `metadata.json` with `{"user_data": {"channels": [...]}}` or pass `--channels` |
| `Incorrect number of channels` | Dataset channel count doesn't match `--channels`; check metadata.json |
| High duplicate rate (>20%) | Normal for small datasets; not an error — duplicates are skipped silently |
| Training loss not decreasing | Try lower LR (`1e-4` → `1e-5`); check class balance with `--loss-weights` |
| `wandb` offline | Run `wandb login`; check `~/.netrc`; confirm with `wandb status` |
| `cuda out of memory` | Reduce `--batch-size`; use a lighter model |
| Cannot resume run | Pass `training.json` (state file) as input to `cellclass-train` instead of `.npz` |
| Inference classification looks wrong | Verify channel order in `.json` sidecar matches `OMERO_SCREEN_INFERENCE_MODEL` channels |
