# Installation

cellclass is part of the omero-screen monorepo workspace. It is installed automatically
when you run `uv sync` from the repository root.

```bash
git clone https://github.com/Helfrid/omero-screen.git
cd omero-screen
uv sync --dev
source .venv/bin/activate
```

The unified `cellclass` command provides all CellClass workflows; run
`cellclass --help` to see its subcommands. Existing `cellclass-*` entry points remain
available as compatibility aliases during the migration:

| Command | Purpose |
|---|---|
| `cellclass dataset` | Create `.npz` training archive from `.npy` crop files |
| `cellclass train` | Train a single model |
| `cellclass batch` | Convert `train.txt` batch file into `batch.sh` |
| `cellclass sbatch` | Submit a single job to a SLURM cluster |
| `cellclass test` | Evaluate a model on a dataset |
| `cellclass extract` | Export best checkpoint to TorchScript |
| `cellclass sample` | Sample example images from a dataset |

## Weights & Biases

Training runs are logged to [Weights & Biases](https://wandb.ai). Set up your credentials once:

```bash
wandb login
```

Visit <https://wandb.ai/authorize>, log in, and paste the API key when prompted.
The key is saved to `~/.netrc` and persists across sessions and on the Artemis cluster.
Verify the setup with:

```bash
wandb status
```

By default, runs log to the **cellclass** project in your own W&B workspace — no
shared credentials needed. To collaborate with a team, pass `--entity <team>` to
`cellclass train` (e.g. `--entity hocheggerlab`), then share the project via the
W&B interface.
