# Pulling sweep metrics

Two sources. **W&B** is primary (per-epoch curves + summary). The **local checkpoints** are the offline fallback (final metrics only).

## A. W&B (primary)

Requires the terminal to be authed (`wandb login`; the Step 0 preflight checks `wandb.Api().default_entity`). Each sweep has its **own project** (`--project cellclass-<dataset>`), so every run in that project belongs to the sweep — no tag filter needed.

```python
import wandb
import pandas as pd

api = wandb.Api()
entity = api.default_entity          # or set explicitly
project = "cellclass-<dataset>"      # the sweep's per-sweep project

runs = api.runs(f"{entity}/{project}")
# (optional, only if you reused one project across sweeps:)
#   runs = api.runs(f"{entity}/{project}", filters={"tags": "<dataset>"})

# --- final summary, one row per run ---
summary = pd.DataFrame([
    {
        "run": r.name,
        "model": r.config.get("model"),
        "lr": r.config.get("learning_rate") or r.config.get("lr"),
        "f1": r.summary.get("f1"),
        "test_f1": r.summary.get("test_f1"),
        "acc": r.summary.get("acc"),
        "val_loss": r.summary.get("val_loss") or r.summary.get("loss"),
        "epoch": r.summary.get("epoch") or r.summary.get("_step"),
        "state": r.state,
    }
    for r in runs
])

# --- per-epoch history for curves ---
hist = []
for r in runs:
    h = r.history(samples=5000)      # logged keys: train_loss, val_loss, train_acc, acc, precision, recall, f1, test_*
    h["run"] = r.name
    h["model"] = r.config.get("model")
    hist.append(h)
history = pd.concat(hist, ignore_index=True) if hist else pd.DataFrame()
```

Notes:
- With project-per-sweep, `api.runs(f"{entity}/{project}")` returns exactly the sweep. `filters` (MongoDB syntax) is only needed to narrow further, e.g. `{"config.model": "densenet121"}`, or `{"tags": "<dataset>"}` if you ever share one project across sweeps.
- Logged metric keys come from `run_training.py` (`train_loss`, `val_loss`, `train_acc`, `acc`, `precision`, `recall`, `f1`, and `test_*` every `--testing-interval`).
- Select the best run by `test_f1` (fall back to `f1`).

## B. Local checkpoints (fallback — offline / not logged in)

Each sweep run writes `<npz>.{N}.pt` and a best copy `<npz>.{N}.pt.best`. The checkpoint dict carries **final/best metrics only** (no per-epoch history).

```python
import glob, torch
import pandas as pd

rows = []
for f in sorted(glob.glob("<crops_dir>/*.npz.*.pt.best")):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    rows.append({
        "file": f,
        "model": ck.get("model"),
        "epoch": ck.get("epoch"),
        "f1": ck.get("f1"),
        "test_f1": ck.get("test_f1"),
        "acc": ck.get("acc"),
        "val_loss": ck.get("loss"),
        "labels": ck.get("labels"),
    })
summary = pd.DataFrame(rows)
```

Use `.pt.best` (best-F1 snapshot); fall back to `.pt` if no `.best` exists. The matching `<npz>.{N}.json` settings file is what `cellclass-extract` consumes for the winning run.
