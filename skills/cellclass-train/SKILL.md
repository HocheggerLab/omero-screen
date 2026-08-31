---
name: cellclass-train
description: Train a cellclass CNN classifier end-to-end from collected labelled crops — build the .npz dataset, run the standard architecture/hyperparameter sweep, pull W&B (or local) metrics, generate a deterministic Jupyter report notebook (fixed A–E panels, vector PDFs), and extract the best model to TorchScript. Use when the user wants to train, sweep, or evaluate a cell-feature classifier from labelled crops (cellclass / omero-screen). Runtime-agnostic — runs wherever Claude Code is running (laptop or GPU box); contains NO SSH/remote orchestration.
---

# cellclass-train

Executable runbook for the **cellclass** classifier training pipeline. Drives a deterministic CLI progression from collected labelled crops to a TorchScript model. The hyperparameters vary per dataset, but the *progression* is fixed — that is what this skill automates.

**This skill runs in-place.** It does not SSH anywhere. If training must happen on a GPU box, the user opens a Claude Code session *on that box* and runs this skill there. Device is auto-detected, never hardcoded.

**Background reference:** the conceptual pipeline (stages, internals) is documented in the Obsidian vault note `CellClass training pipeline` under `[[&CellClass]]`. The CLI lives in the `cellclass` package of the omero-screen workspace; the unified `cellclass` command is available in any environment that has the workspace installed. Legacy `cellclass-*` executables remain compatibility aliases.

---

## Pipeline at a glance

```
labelled .npy crops
  └─[1 dataset]→ <name>.npz
       └─[2 sweep]→ train.txt → batch.sh → <name>.npz.{N}.pt(.best) + .json   (one per run)
            └─[3 metrics]→ W&B (group tag)  ||  local .pt.best fallback
                 └─[4 report]→ Jupyter report notebook + vector PDFs (assets/build_report.py)
                      └─[5 select+extract]→ cellclass extract <best>.json --save → model.pt + .json
```

Read `references/standard-sweep.md` for the architecture/hyperparameter standard (10x low-res crops) and `references/wandb-pull.md` for the metrics query before running steps 2–4.

---

## Step 0 — Preflight (always run first)

Work out the working directory (where the crops / `.npz` live) and confirm the environment.

```bash
# device auto-detect — use the printed value as -d below
uv run python -c "import torch;print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')"
```

- **device == cuda** → keep `--cudnn-benchmark --pin-memory --num-workers 4` in the sweep.
- **device == mps / cpu** → **drop** `--cudnn-benchmark` and `--pin-memory` (CUDA-only; they warn and no-op). `mps` = Apple Metal GPU; `cpu` = slow, smoke-tests only.

**W&B auth check (non-interactive — never automate the key):**
```bash
uv run python -c "import wandb;print('wandb entity:', wandb.Api().default_entity)" 2>&1 | tail -1
```
- Prints an entity → authed, proceed.
- Errors / no entity → tell the user to run `wandb login` in *this* terminal (key from https://wandb.ai/authorize). Do **not** attempt to enter the key for them. Alternatively run the sweep with `WANDB_MODE=offline` (sync later) — training still works; live dashboards don't.
- **Project creation is automatic** — the first `wandb.init(project=...)` creates the project under the default entity. Nothing to pre-create.

---

## Step 1 — Build the dataset

Crops are a directory of `.npy` files (each a pickled dict of image/mask lists + `target` labels), optionally with a napari `metadata.json` carrying `channels` under `user_data`.

```bash
uv run cellclass dataset <crops_dir> --name <name> --channels <CH1> [<CH2> ...]
```
- Pass `--channels` explicitly (`cellclass dataset` may not parse napari's nested `user_data.channels`). One channel → `c1` models; two → `c2`, etc.
- `.npz` is written to `--out` (default = crops dir): `<crops_dir>/<name>.npz`.

**Parse the printed class balance.** It lists per-class counts and fractions. If the max:min class ratio is roughly ≥ 3:1, **recommend `--loss-weights`** in the sweep (focal loss is on by default; inverse-frequency weighting handles the imbalance). Note any large `ignored class` count (unassigned crops) for the user.

---

## Step 2 — Generate the sweep and run it

Generate `train.txt` from the standard in `references/standard-sweep.md`, substituting: dataset `.npz` path (relative to the dir where `batch.sh` will run), detected device, and a **per-sweep W&B project** (`--project <name>`) on every line so the sweep's runs land in their own project. Use a descriptive name, e.g. `cellclass-<dataset>`. W&B auto-creates the project on the first run.

```bash
uv run cellclass batch <crops_dir>/train.txt --script batch.sh   # writes batch.sh, one training run per line
bash batch.sh                                                     # run the sweep
```
- `cellclass batch` derives per-run output names `<npz>.{N}.pt` / `<npz>.{N}.json` (best checkpoint → `.pt.best`).
- **Re-runs collide:** `cellclass train --existing` defaults to `error`, so a second sweep over the same names fails. Remove prior `*.npz.*.pt*` / `*.npz.*.json` (or use a fresh group) before re-running.
- **Long runs:** if on a remote box, advise the user to launch `bash batch.sh` inside `tmux` so it survives SSH drops (the user manages the SSH/tmux session — this skill does not).
- Use `--dry-run` first to preview, or `--background` to fan out runs as background processes.

---

## Step 3 — Pull metrics

After the sweep, gather results. Follow `references/wandb-pull.md`:
- **W&B (primary):** query `wandb.Api().runs(f"{entity}/{project}")` for the sweep's project → per-epoch history (loss / accuracy / F1 curves) + final summary. (Every run in the per-sweep project belongs to this sweep, so no tag filter is needed.)
- **Local (fallback, offline-safe):** `torch.load("<npz>.{N}.pt.best", map_location="cpu", weights_only=False)` per run → final metrics only (`f1`, `test_f1`, `loss`, `acc`, `model`, `labels`). No per-epoch curves locally.

---

## Step 4 — Build the report notebook

Generate the standard **A–E panel report** with `assets/build_report.py`. Unlike the interactive marimo notebook, this emits the *same* static, re-runnable Jupyter notebook every time, with figures saved as Illustrator-friendly **vector** PDFs (the confusion-matrix heatmap is drawn as rectangles, not a rasterised image — which Illustrator garbles). It also folds in the Step 3 pull. **Run it from the project env** (needs `cellclass` on PATH, plus nbformat/wandb/pandas/numpy/pillow):

```bash
uv run python <skill_dir>/assets/build_report.py <npz> \
    --best <npz>.{N}.pt.best --model <arch> \      # winner checkpoint + its architecture
    --project cellclass-<dataset> \                # or: --runs-dir <dir> for offline (local checkpoints)
    --device <DEV> --execute
# optional Panel C overlay (e.g. loss-weighted vs not):
#   --compare <npz>.{M}.pt.best --compare-model <arch> --compare-label "loss weights"
```

This (1) pulls the sweep → `sweep_summary.csv` + `sweep_history.csv`; (2) runs `cellclass test` for the winner (and `--compare`) → `confusion-matrix.csv` (skipped if it already exists); (3) samples one representative crop per class from the `.npz`; (4) copies the lab style assets (`hhlab_style01.mplstyle`, `colors.py`); (5) writes `<name>_report.ipynb` and, with `--execute`, renders the panel PDFs.

Panels (fixed order): **A** training curves of the top runs (W&B only) · **B** all runs ranked by test F1 → winner (bold) · **C** winner per-class precision & recall (overlays `--compare` when given) · **D** confusion matrix + representative crops (vector) · **E** placeholder for a napari gallery on unseen data.

Notes:
- `--best` may be a training checkpoint (`.pt.best` + `--model <arch>`) or an already-extracted scripted model (`.pt` with sidecar `.json`, omit `--model`).
- The notebook reads only the local CSVs/assets, so it re-runs offline and edits cleanly in Jupyter. Delete `confusion-matrix.csv` to force a recompute.
- *Optional interactive alternative:* `uv run --with marimo marimo edit <skill_dir>/assets/training_runs.py` for ad-hoc run exploration (overlaid curves, best-F1 bar chart, summary table; `wandb`/`local` source toggle).

---

## Step 5 — Select the best & extract

Pick the run with the highest **test F1** (fall back to val F1 if no test split). Extract its settings JSON to TorchScript:
```bash
uv run cellclass extract <npz>.{N}.json --save
```
Produces `<model>_c<channels>_l<labels>.pt` (TorchScript) + a `.json` metadata sidecar (`labels`, `model`, `channels`, `input_shape`, metrics). This `.pt` is the artefact consumed at inference via `omero-screen <plate> --inference <model>.pt`.

**Report:** summary table (model × lr → val/test F1), the winning run, and the extracted model path. Offer inference wiring as an explicit optional follow-up — do not run it unprompted.

---

## Notes & gotchas

- **Publication repos** typically `.gitignore` `*.npy *.npz *.pt *.pth` — the trained model and dataset are *not* version-controlled. Remind the user to archive the final `.pt` + `.npz` (Zenodo/OSF) or force-add the model for reproducibility.
- **Standard is for 10x low-res crops.** For higher-res or different channels, the architecture/HP set may need revisiting — see `references/standard-sweep.md`.
- Augmentation defaults: `--rotate 180 --translate 0.1`; set `--flip 3` (H+V) for orientation-free cells.
- Always confirm the detected device and class balance with the user before launching a long sweep.
