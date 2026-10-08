# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "pandas", "matplotlib", "wandb"]
# ///
"""cellclass training-run comparison.

Compares the runs of a cellclass sweep from either Weights & Biases (per-epoch
curves + summary) or local checkpoints (final metrics only, offline fallback).

Launch from a project env that has torch/wandb/pandas (needed for local mode):
    uv run --with marimo marimo edit training_runs.py
Or fully isolated (W&B mode only; local mode needs torch):
    uvx marimo edit --sandbox training_runs.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    return mo, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
    # cellclass training runs

    Set the source and parameters, then explore the sweep below.

    - **wandb** — needs the terminal authed (`wandb login`); reads the sweep's per-sweep `--project` (tag filter optional).
    - **local** — reads `*.npz.*.pt.best` checkpoints from a directory (final metrics only; needs `torch`).
    """
    )
    return


@app.cell
def _(mo):
    source = mo.ui.dropdown(["wandb", "local"], value="wandb", label="Source")
    entity = mo.ui.text(label="W&B entity (blank = default)")
    project = mo.ui.text(value="cellclass-", label="W&B project (per-sweep)")
    group = mo.ui.text(label="Tag filter (optional)")
    local_dir = mo.ui.text(label="Local dir (local source)")
    refresh = mo.ui.run_button(label="Load / refresh")
    mo.vstack([source, entity, project, group, local_dir, refresh])
    return entity, group, local_dir, project, refresh, source


@app.cell
def _(entity, group, local_dir, pd, project, refresh, source):
    # re-runs when `refresh` is clicked (referenced for the dependency edge)
    _ = refresh.value
    summary = pd.DataFrame()
    history = pd.DataFrame()
    error = None

    if source.value == "wandb":
        if not project.value.strip() or project.value.strip() == "cellclass-":
            error = "Enter the sweep's W&B project (e.g. cellclass-<dataset>)."
        else:
            try:
                import wandb

                api = wandb.Api()
                ent = entity.value.strip() or api.default_entity
                _filters = (
                    {"tags": group.value.strip()}
                    if group.value.strip()
                    else None
                )
                runs = list(
                    api.runs(
                        f"{ent}/{project.value.strip()}", filters=_filters
                    )
                )
                summary = pd.DataFrame(
                    [
                        {
                            "run": r.name,
                            "model": r.config.get("model"),
                            "lr": r.config.get("learning_rate")
                            or r.config.get("lr"),
                            "f1": r.summary.get("f1"),
                            "test_f1": r.summary.get("test_f1"),
                            "acc": r.summary.get("acc"),
                            "val_loss": r.summary.get("val_loss")
                            or r.summary.get("loss"),
                            "epoch": r.summary.get("epoch")
                            or r.summary.get("_step"),
                            "state": r.state,
                        }
                        for r in runs
                    ]
                )
                frames = []
                for r in runs:
                    h = r.history(samples=5000)
                    if len(h):
                        h["run"] = r.name
                        h["model"] = r.config.get("model")
                        frames.append(h)
                history = (
                    pd.concat(frames, ignore_index=True)
                    if frames
                    else pd.DataFrame()
                )
                if not len(summary):
                    error = f"No runs found in {ent}/{project.value}" + (
                        f" with tag '{group.value}'."
                        if group.value.strip()
                        else "."
                    )
            except Exception as e:  # noqa: BLE001
                error = f"W&B load failed: {e}"
    else:
        import glob

        if not local_dir.value.strip():
            error = "Enter the local directory containing the *.pt.best checkpoints."
        else:
            try:
                import torch

                files = sorted(
                    glob.glob(f"{local_dir.value.strip()}/*.npz.*.pt.best")
                )
                if not files:
                    files = sorted(
                        glob.glob(f"{local_dir.value.strip()}/*.npz.*.pt")
                    )
                rows = []
                for f in files:
                    ck = torch.load(f, map_location="cpu", weights_only=False)
                    rows.append(
                        {
                            "run": f.split("/")[-1],
                            "model": ck.get("model"),
                            "lr": ck.get("lr"),
                            "f1": ck.get("f1"),
                            "test_f1": ck.get("test_f1"),
                            "acc": ck.get("acc"),
                            "val_loss": ck.get("loss"),
                            "epoch": ck.get("epoch"),
                        }
                    )
                summary = pd.DataFrame(rows)
                if not len(summary):
                    error = f"No *.pt(.best) checkpoints in {local_dir.value}."
            except Exception as e:  # noqa: BLE001
                error = f"Local load failed (torch needed): {e}"

    return error, history, summary


@app.cell
def _(error, mo, summary):
    mo.vstack(
        [
            mo.md(f"> ⚠️ {error}")
            if error
            else mo.md(f"### {len(summary)} run(s)"),
            mo.ui.table(summary, selection=None)
            if len(summary)
            else mo.md("_no data loaded_"),
        ]
    )
    return


@app.cell
def _(plt, summary):
    fig_bar = None
    if len(summary):
        _s = summary.copy()
        _ycol = (
            "test_f1"
            if _s.get("test_f1") is not None and _s["test_f1"].notna().any()
            else "f1"
        )
        _s = _s.sort_values(_ycol, na_position="first")
        fig_bar, _ax = plt.subplots(figsize=(7, 0.5 * len(_s) + 1.5))
        _ax.barh(_s["run"].astype(str), _s[_ycol].fillna(0))
        _ax.set_xlabel(_ycol)
        _ax.set_xlim(0, 1)
        _ax.set_title(f"{_ycol} per run (higher = better)")
        fig_bar.tight_layout()
    fig_bar
    return


@app.cell
def _(history, plt):
    fig_loss = None
    if len(history) and "_step" in history and "val_loss" in history:
        fig_loss, _ax = plt.subplots(figsize=(8, 5))
        for _run, _g in history.groupby("run"):
            _g = _g.sort_values("_step")
            if "train_loss" in _g:
                _ax.plot(_g["_step"], _g["train_loss"], "--", alpha=0.4)
            _ax.plot(_g["_step"], _g["val_loss"], label=str(_run))
        _ax.set_xlabel("epoch")
        _ax.set_ylabel("loss")
        _ax.set_title("Loss — solid = val, dashed = train")
        _ax.legend(fontsize=7)
        fig_loss.tight_layout()
    fig_loss
    return


@app.cell
def _(history, plt):
    fig_f1 = None
    if len(history) and "_step" in history and "f1" in history:
        fig_f1, _ax = plt.subplots(figsize=(8, 5))
        for _run, _g in history.groupby("run"):
            _g = _g.sort_values("_step")
            _ax.plot(_g["_step"], _g["f1"], label=str(_run))
            if "test_f1" in _g and _g["test_f1"].notna().any():
                _ax.plot(_g["_step"], _g["test_f1"], ":", alpha=0.6)
        _ax.set_xlabel("epoch")
        _ax.set_ylabel("F1")
        _ax.set_ylim(0, 1)
        _ax.set_title("F1 — solid = val, dotted = test")
        _ax.legend(fontsize=7)
        fig_f1.tight_layout()
    fig_f1
    return


if __name__ == "__main__":
    app.run()
