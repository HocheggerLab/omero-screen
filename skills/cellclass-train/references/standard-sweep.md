# Standard sweep — 10x low-res crops

The lab standard for classifiers trained on **10x low-resolution crops** (small ROIs, typically 1–2 channels, ~50 px crop). Six runs across four architectures and two learning rates: light nets to establish a fast baseline, heavier nets (with transfer learning) for accuracy. Compare on test F1.

## Common flags (every line)

| Flag | Value | Why |
|---|---|---|
| `--lr-scheduler` | `plateau` | Adaptive to variable convergence; better than fixed `step` for these data |
| `--flip` | `3` | Horizontal + vertical — cells have no canonical orientation |
| `--loss-weights` | on | Inverse-frequency class weighting; essential when class ratio ≥ ~3:1 (focal loss is already the default) |
| `--wandb` | on | Log to W&B |
| `--project` | `cellclass-<dataset>` | **Per-sweep project** — all runs of this sweep land here (auto-created); the metrics pull just reads the project, no tag filter |
| `--num-workers` | `4` | Parallel data loading |
| `-d` | auto | `cuda` on the GPU box; `mps`/`cpu` elsewhere (auto-detect in Step 0) |
| `--cudnn-benchmark` `--pin-memory` | cuda only | **Drop both** when device is `mps`/`cpu` |

Defaults left implicit: `--epochs 2000` (early-stopping ends it sooner), `--rotate 180`, `--translate 0.1`, `--loss-function focal_loss`, `--weights DEFAULT`, `--validation-size 0.2`, `--testing-size 0.2`.

## The six runs

| # | model | lr | batch | extra |
|---|---|---|---|---|
| 1 | efficientnetb3s | 1e-3 | 64 | `--freeze-weights` (frozen backbone, train head) |
| 2 | efficientnetb3s | 1e-4 | 64 | `--freeze-weights` |
| 3 | densenet121 | 1e-3 | 64 | full fine-tune |
| 4 | densenet121 | 1e-4 | 64 | full fine-tune |
| 5 | shufflenet2x1_0 | 1e-3 | 128 | light/fast |
| 6 | squeezenet1_0 | 1e-3 | 128 | light/fast |

## train.txt template

Substitute `<NPZ>` (path relative to where `batch.sh` runs), `<PROJECT>` (per-sweep W&B project, e.g. `cellclass-<dataset>`), and `<DEV>` (detected device). On `mps`/`cpu` remove `--cudnn-benchmark --pin-memory`.

```text
# cellclass standard sweep — project <PROJECT>
<NPZ> --model efficientnetb3s --freeze-weights --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 64 --loss-weights -d <DEV> --cudnn-benchmark --pin-memory --num-workers 4 --wandb --project <PROJECT>
<NPZ> --model efficientnetb3s --freeze-weights --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 64 --loss-weights -d <DEV> --cudnn-benchmark --pin-memory --num-workers 4 --wandb --project <PROJECT>
<NPZ> --model densenet121 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 64 --loss-weights -d <DEV> --cudnn-benchmark --pin-memory --num-workers 4 --wandb --project <PROJECT>
<NPZ> --model densenet121 --lr 1e-4 --lr-scheduler plateau --flip 3 --batch-size 64 --loss-weights -d <DEV> --cudnn-benchmark --pin-memory --num-workers 4 --wandb --project <PROJECT>
<NPZ> --model shufflenet2x1_0 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 128 --loss-weights -d <DEV> --cudnn-benchmark --pin-memory --num-workers 4 --wandb --project <PROJECT>
<NPZ> --model squeezenet1_0 --lr 1e-3 --lr-scheduler plateau --flip 3 --batch-size 128 --loss-weights -d <DEV> --cudnn-benchmark --pin-memory --num-workers 4 --wandb --project <PROJECT>
```

> To compare *across* sweeps later, keep a constant prefix (`cellclass-…`) or add `--tags <dataset>` as well — tags still work for cross-project filtering, they're just no longer needed to identify a single sweep.

## When to deviate

- **Balanced classes (< ~3:1):** `--loss-weights` can be dropped.
- **More channels / higher-res crops:** larger nets (densenet161/201, efficientnetb4) may pay off; revisit batch size for VRAM (32 GB on the RTX5090).
- **Quick local smoke test (mps/cpu):** add `--size 500 --epochs 20` to one line to confirm the dataset loads and a model fits before committing to the full sweep.
- **VRAM pressure:** lower `--batch-size`; efficientnet `*s` frozen variants are the lightest accurate option.
