# Quick Start

This page walks through the complete workflow from annotated crop files to a deployed
TorchScript model, using a two-class mitosis classifier as a running example.

## Prerequisites

- Annotated `.npy` crop files produced by the
  [Training Widget](../omero-screen-napari/training_widget.md), stored in a single
  directory (e.g. `~/data/mitosis-rpe/`).
- A `metadata.json` file in that directory containing at minimum:

  ```json
  { "user_data": { "channels": ["DAPI", "Tub"] } }
  ```

  The Training Widget writes this file automatically.
- [Weights & Biases](installation.md#weights-biases) credentials configured.

---

## Step 1 — Create the dataset

Pack all `.npy` files into a single normalised `.npz` archive:

```bash
cellclass-dataset ~/data/mitosis-rpe --name rois
```

This applies (1, 99) percentile normalisation, zeros all pixels outside each cell mask,
converts to `uint8`, and writes `~/data/mitosis-rpe/rois.npz`.

Inspect a random sample of each class as a TIFF montage:

```bash
cellclass-sample ~/data/mitosis-rpe/rois.npz --output ~/data/mitosis-rpe
```

Open the resulting TIFFs in ImageJ (`Image → Color → Channels Tool → Composite`) to
verify your labels look correct before training.

---

## Step 2 — Write `train.txt`

Create a plain-text batch file, one training run per line. See the
[Model Guide](models.md) for recommended architectures for small cell images.

A typical starting point for 50 × 50 px crops:

```text
~/data/mitosis-rpe/rois.npz --model shufflenet2x1_0 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 128 -d mps --wandb
~/data/mitosis-rpe/rois.npz --model shufflenet2x1_0 --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 128 -d mps --wandb
~/data/mitosis-rpe/rois.npz --model densenet121      --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 64  -d mps --wandb
~/data/mitosis-rpe/rois.npz --model densenet121      --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 64  -d mps --wandb
~/data/mitosis-rpe/rois.npz --model efficientnetb3s  --freeze-weights --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 64 -d mps --wandb
~/data/mitosis-rpe/rois.npz --model efficientnetb3s  --freeze-weights --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 64 -d mps --wandb
```

Key flags:

- `--flip 3` — random horizontal and vertical flips (cells have no orientation bias).
- `--lr-scheduler plateau` — adapts learning rate to convergence speed.
- `--freeze-weights` — only for `efficientnetb3s` / `efficientnetb4s` (ImageNet pretrained).
- `-d mps` — Apple Silicon GPU. Use `-d cuda` on Linux, `-d cpu` as fallback.
- `--wandb` — log to Weights & Biases.

Lines starting with `#` are ignored; comment out completed runs before regenerating
`batch.sh` to avoid overwriting results.

---

## Step 3 — Generate and run `batch.sh`

```bash
cellclass-batch train.txt --script batch.sh
bash batch.sh
```

For SLURM on the Artemis cluster see the [SLURM guide](slurm.md).

---

## Step 4 — Review results on Weights & Biases

Open your W&B dashboard at <https://wandb.ai> and navigate to the **cellclass** project
in your workspace. Compare validation accuracy across runs and note the metadata `.json`
filename of the best run.

---

## Step 5 — Extract the model

```bash
cellclass-extract best_run.json --save
```

This saves two files alongside the checkpoint:

- `shufflenet2x1_0_c2_l2.pt` — TorchScript model (architecture + weights).
- `shufflenet2x1_0_c2_l2.json` — sidecar with channel names, class labels, and image size.

The filename encodes the architecture, number of input channels (`c2`), and number of
classes (`l2`).

---

## Step 6 — Verify the model

```bash
cellclass-test -s shufflenet2x1_0_c2_l2.pt ~/data/mitosis-rpe/rois.npz
```

This prints a full classification report (precision, recall, F1 per class) on the held-out
test split.

---

## Step 7 — Deploy

Copy the `.pt` + `.json` pair to the location expected by `omero-screen` and set
`OMERO_SCREEN_INFERENCE_MODEL` to the `.pt` path. The pipeline will load the model
automatically during plate analysis. See the
[pipeline configuration](../configuration.rst) for details.
