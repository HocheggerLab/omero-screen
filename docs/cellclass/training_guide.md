# Training Guide

## Data format

Training data is stored as `.npy` files — pickled Python dicts with the following structure:

```python
{
    'data': ([img1, img2, ...], [mask1, mask2, ...]),  # images (YXC), masks (YX)
    'target': ['interphase', 'mitosis', ...]           # string class labels
}
```

The [Training Widget](../omero-screen-napari/training_widget.md) produces files in exactly
this format. Each file contains all annotations from one session (one plate / well / image
combination). Multiple files can coexist in the same directory and are combined by
`cellclass-dataset`.

---

## Step 1 — Create dataset

```bash
cellclass-dataset <data_dir> --name rois
```

What this does:

1. Globs all `*.npy` files in `<data_dir>`.
2. For each crop: applies (1, 99) percentile stretch, zeros all pixels outside the mask,
   converts to `uint8`.
3. Deduplicates identical crops (by pixel hash) to avoid overfitting on repeated cells.
4. Writes `<data_dir>/rois.npz` — a single compressed numpy archive.

Channel names are read from `metadata.json` in the data directory. Override with `--channels`:

```bash
cellclass-dataset ~/data/ppase_screen --channels DAPI RFP --name rois
```

Useful options:

| Flag | Default | Description |
|---|---|---|
| `--name` | `rois` | Output filename stem |
| `--channels` | from metadata.json | Channel names (order must match image data) |
| `--ignore` | none | Class labels to exclude |
| `--size` | none | Resize all crops to `N×N` pixels |

---

## Step 2 — Sample and verify

Before training, visually verify your dataset:

```bash
cellclass-sample ~/data/mitosis-rpe/rois.npz --output ~/data/mitosis-rpe
```

This writes one TIFF per class containing a random grid of crops in ImageJ hyperstack
format. Open in ImageJ with `Image → Color → Channels Tool → Composite` to check
channel assignments and label quality.

---

## Step 3 — Write `train.txt`

A batch file is a plain-text list of arguments for `cellclass-train`, one run per line.
Comment lines (`#`) and blank lines are ignored.

### Full parameter reference

```text
<dataset.npz>
  --model <name>            # Architecture (see Model Guide)
  --lr <float>              # Initial learning rate (try 1e-3 and 1e-4)
  --lr-scheduler <type>     # plateau (recommended) or step
  --flip <0-3>              # Augmentation: 0=none,1=h,2=v,3=both
  --batch-size <int>        # 64 for heavy models, 128 for light
  --freeze-weights          # Freeze pretrained backbone (EfficientNet only)
  -d <device>               # cpu | mps | cuda | cuda:0 | cuda:1
  --wandb                   # Log to Weights & Biases
  -n <name>                 # Checkpoint filename prefix
  -s <name>                 # Metadata JSON filename prefix
  --no-loss-weights         # Disable class-imbalance weighting
  --epochs <int>            # Max epochs (default: 100)
  --patience <int>          # Early stopping patience (default: 10)
```

### Recommended sweep for 50 × 50 px images

```text
# --- ShuffleNet (fast, good baseline) ---
/path/to/rois.npz --model shufflenet2x1_0 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 128 -d cuda --wandb
/path/to/rois.npz --model shufflenet2x1_0 --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 128 -d cuda --wandb

# --- DenseNet121 (strong performer) ---
/path/to/rois.npz --model densenet121 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 64 -d cuda --wandb
/path/to/rois.npz --model densenet121 --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 64 -d cuda --wandb

# --- EfficientNetB3s (ImageNet pretrained, freeze backbone) ---
/path/to/rois.npz --model efficientnetb3s --freeze-weights --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 64 -d cuda --wandb
/path/to/rois.npz --model efficientnetb3s --freeze-weights --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 64 -d cuda --wandb

# --- SqueezeNet (lightest, useful sanity check) ---
/path/to/rois.npz --model squeezenet1_0 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 128 -d cuda --wandb
```

---

## Step 4 — Generate and run `batch.sh`

```bash
cellclass-batch train.txt --script batch.sh
bash batch.sh
```

`cellclass-batch` assigns unique checkpoint and metadata filenames automatically so runs
never overwrite each other. Comment out completed lines in `train.txt` before regenerating.

To run jobs in parallel (one per GPU on a multi-GPU machine):

```bash
cellclass-batch train.txt --script batch.sh --background
bash batch.sh
```

Each run launches as a background process; add `--device cuda:0`, `cuda:1`, etc. per line
to assign GPUs explicitly.

---

## Step 5 — Compare on Weights & Biases

Open your W&B dashboard at <https://wandb.ai> and navigate to the **cellclass** project in your workspace. The key metrics to compare:

- **val/acc** — validation accuracy (primary metric for model selection).
- **test/acc** — held-out test accuracy (reported at end of training).
- **val/loss** — validation loss curve (check for overfitting).

Download the best run's metadata JSON from the W&B artifacts panel, or find it locally
in your working directory.

---

## Step 6 — Extract the model

```bash
cellclass-extract best_run.json
```

Without `--save` this prints the performance metrics without writing any files — useful
for a quick sanity check. Add `--save` to export:

```bash
cellclass-extract best_run.json --save
```

Output files:

- `<model>_c<channels>_l<labels>.pt` — TorchScript model.
- `<model>_c<channels>_l<labels>.json` — sidecar metadata (channels, labels, image size).

---

## Step 7 — Test the exported model

```bash
cellclass-test -s shufflenet2x1_0_c2_l2.pt ~/data/mitosis-rpe/rois.npz
```

Prints a per-class precision/recall/F1 report and overall accuracy on the test split.

---

## Resuming a crashed run

Training checkpoints are saved after each epoch. To restart from the last checkpoint:

```bash
cellclass-train best_run.json
```

Passing the metadata JSON resumes from the saved checkpoint. Note: some training state
(e.g. convergence checker) is not persisted and is recreated from scratch.
