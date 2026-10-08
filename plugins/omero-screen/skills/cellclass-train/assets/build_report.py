#!/usr/bin/env python3
r"""Generate a deterministic cellclass training-report notebook (panels A-E).

Produces the *same* Jupyter notebook layout for every sweep, so a training run
always yields a comparable report. Unlike the interactive marimo notebook, the
output is a static, re-runnable ``.ipynb`` whose figures are saved as
Illustrator-friendly **vector** PDFs (no rasterised heatmaps).

Panels (fixed order):
  A  Training curves (val loss + val F1) for the top runs        [W&B only]
  B  All runs ranked by test F1 -> winner highlighted
  C  Winner per-class precision & recall (+ optional --compare overlay)
  D  Confusion matrix of the winner + representative crops (vector)
  E  Gallery on unseen data -- placeholder (e.g. napari), not built here

What it does:
  1. Pulls the sweep from W&B (--project) or local checkpoints (--runs-dir) and
     writes ``sweep_summary.csv`` (+ ``sweep_history.csv`` for W&B) into --outdir,
     so the notebook reads local files and is reproducible offline.
  2. Runs ``cellclass test`` for the winner (--best) and optional --compare model
     to produce the confusion-matrix CSV(s); skips a CSV that already exists.
  3. Samples a representative crop per class from the dataset .npz.
  4. Copies the lab style assets (hhlab_style01.mplstyle, colors.py) into --outdir.
  5. Writes ``<name>_report.ipynb`` and, with --execute, renders the panel PDFs.

Run it in the project env (needs nbformat, pandas, numpy, wandb, pillow, and
``cellclass`` on PATH):
    uv run python <skill>/assets/build_report.py DATA.npz --best DATA.npz.4.pt.best \\
        --model densenet161 --project cellclass-mydata --execute
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SKILL_ASSETS = Path(__file__).resolve().parent

# Summary columns the notebook expects (missing ones are filled with NaN).
SUMMARY_COLS = [
    "run",
    "model",
    "loss_weights",
    "learning_rate",
    "lr_scheduler",
    "batch_size",
    "freeze_weights",
    "f1",
    "acc",
    "test_f1",
    "test_acc",
    "test_precision",
    "test_recall",
    "epoch",
    "state",
]


# --------------------------------------------------------------------------- #
# 1. Gather sweep metrics
# --------------------------------------------------------------------------- #
def gather_wandb(
    entity: str | None, project: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull per-run summary and per-epoch history from a W&B project."""
    import wandb

    api = wandb.Api()
    entity = entity or api.default_entity
    runs = list(api.runs(f"{entity}/{project}"))
    if not runs:
        raise SystemExit(f"No runs found in {entity}/{project}")

    rows, hist = [], []
    for r in runs:
        c = r.config
        rows.append(
            {
                "run": r.name,
                "model": c.get("model"),
                "loss_weights": bool(c.get("loss_weights")),
                "learning_rate": c.get("learning_rate") or c.get("lr"),
                "lr_scheduler": c.get("lr_scheduler"),
                "batch_size": c.get("batch_size"),
                "freeze_weights": bool(c.get("freeze_weights")),
                "f1": r.summary.get("f1"),
                "acc": r.summary.get("acc"),
                "test_f1": r.summary.get("test_f1"),
                "test_acc": r.summary.get("test_acc"),
                "test_precision": r.summary.get("test_precision"),
                "test_recall": r.summary.get("test_recall"),
                "epoch": r.summary.get("epoch") or r.summary.get("_step"),
                "state": r.state,
            }
        )
        h = r.history(samples=10000)
        if len(h):
            h = h.reset_index(drop=True)
            h["epoch"] = np.arange(len(h))
            h["run"] = r.name
            h["model"] = c.get("model")
            keep = [
                k
                for k in [
                    "run",
                    "model",
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "loss",
                    "f1",
                    "acc",
                ]
                if k in h.columns
            ]
            hist.append(h[keep])

    summary = pd.DataFrame(rows)
    history = pd.concat(hist, ignore_index=True) if hist else pd.DataFrame()
    return summary, history


def gather_local(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read final metrics from ``*.pt.best`` checkpoints (no per-epoch history)."""
    import torch

    rows = []
    for f in sorted(runs_dir.glob("*.pt.best")) or sorted(
        runs_dir.glob("*.pt")
    ):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict):
            continue
        rows.append(
            {
                "run": f.name,
                "model": ck.get("model"),
                "loss_weights": np.nan,
                "learning_rate": np.nan,
                "lr_scheduler": np.nan,
                "batch_size": np.nan,
                "freeze_weights": np.nan,
                "f1": ck.get("f1"),
                "acc": ck.get("acc"),
                "test_f1": ck.get("test_f1"),
                "test_acc": ck.get("test_acc"),
                "test_precision": ck.get("test_precision"),
                "test_recall": ck.get("test_recall"),
                "epoch": ck.get("epoch"),
                "state": "local",
            }
        )
    if not rows:
        raise SystemExit(f"No checkpoints found in {runs_dir}")
    return pd.DataFrame(rows), pd.DataFrame()


# --------------------------------------------------------------------------- #
# 2. Confusion matrix via cellclass test
# --------------------------------------------------------------------------- #
def run_confusion(
    npz: Path,
    best: Path,
    out_csv: Path,
    *,
    model: str | None,
    device: str,
    testing_size: float,
    data_seed: int,
) -> None:
    """Shell out to ``cellclass test`` to write a confusion-matrix CSV."""
    if out_csv.exists():
        print(f"  reuse existing {out_csv.name}")
        return
    cmd = [
        "cellclass",
        "test",
        str(npz),
        "--data-seed",
        str(data_seed),
        "--testing-size",
        str(testing_size),
        "--device",
        device,
        "--num-workers",
        "0",
        "--matrix-csv",
        str(out_csv),
    ]
    if model:
        cmd += ["--model", model, "--name", str(best)]
    else:
        cmd += ["--script", str(best)]
    print(f"  cellclass test -> {out_csv.name} ...")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# 3. Representative crops
# --------------------------------------------------------------------------- #
def sample_crops(npz: Path, assets_dir: Path) -> list[str]:
    """Save one median-intensity crop per class; return the class label order."""
    from PIL import Image

    data = np.load(npz, allow_pickle=True)
    X, y = data["X"], data["y_names"]
    labels = sorted(np.unique(y).tolist())
    assets_dir.mkdir(parents=True, exist_ok=True)
    for c in labels:
        idx = np.flatnonzero(y == c)
        means = X[idx].reshape(len(idx), -1).mean(axis=1)
        chosen = idx[
            int(np.argsort(means)[len(means) // 2])
        ]  # median-intensity
        img = X[chosen]
        img = img[0] if img.ndim == 3 else img  # first channel
        Image.fromarray(img.astype(np.uint8), mode="L").save(
            assets_dir / f"{c}.jpg"
        )
    return labels


# --------------------------------------------------------------------------- #
# 4. Notebook construction
# --------------------------------------------------------------------------- #
def build_notebook(
    outdir: Path,
    *,
    name: str,
    project: str | None,
    has_history: bool,
    winner_run: str,
    compare_label: str | None,
) -> Path:
    """Write the panel A-E report notebook into ``outdir``."""
    import nbformat as nbf

    cells: list = []

    def md(t: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(t))

    def code(t: str) -> None:
        cells.append(nbf.v4.new_code_cell(t))

    compare_line = (
        f"- **C** overlays a comparison model (*{compare_label}*).\n"
        if compare_label
        else ""
    )
    md(
        f"# {name} — Classifier Training Report\n\n"
        "Deterministic panel report for a cellclass sweep "
        f"({'W&B `' + project + '`' if project else 'local checkpoints'}).\n\n"
        "- **A** — Training curves (loss + val F1) for the top runs.\n"
        "- **B** — All runs ranked by test F1 → winner.\n"
        "- **C** — Winner: per-class precision & recall.\n"
        f"{compare_line}"
        "- **D** — Confusion matrix of the winner + representative crops.\n"
        "- **E** — Gallery on unseen data — *generated separately (e.g. napari)*.\n\n"
        f"Winner run: `{winner_run}`. Figures save as vector PDFs "
        "(`figA`–`figD`) for Illustrator."
    )

    code(
        "import sys\n"
        "sys.path.insert(0, '.')\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.ticker as mticker\n"
        "from matplotlib import colormaps\n"
        "from matplotlib.colors import Normalize\n"
        "from matplotlib.patches import Rectangle, Patch\n\n"
        "from colors import COLOR\n\n"
        "plt.style.use('hhlab_style01.mplstyle')\n"
        "plt.rcParams.update({\n"
        "    'font.size': 6, 'axes.titlesize': 8, 'axes.labelsize': 6,\n"
        "    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,\n"
        "    'figure.dpi': 300,\n"
        "})\n"
        "CM = 1 / 2.54\n\n"
        "PALETTE = [COLOR.BLUE.value, COLOR.PINK.value, COLOR.YELLOW.value,\n"
        "           COLOR.TURQUOISE.value, COLOR.LAVENDER.value, COLOR.OLIVE.value]\n"
        "PRETTY = {'densenet121': 'DenseNet-121', 'densenet161': 'DenseNet-161',\n"
        "          'densenet201': 'DenseNet-201', 'efficientnetb3s': 'EfficientNet-B3s',\n"
        "          'shufflenet2x1_0': 'ShuffleNet-2x', 'squeezenet1_0': 'SqueezeNet-1.0'}\n\n"
        "def pretty(m):\n"
        "    return PRETTY.get(m, str(m))\n\n"
        "summary = pd.read_csv('sweep_summary.csv')\n"
        "models = list(dict.fromkeys(summary['model']))\n"
        "ARCH_COLOR = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}\n\n"
        "# Which hyperparameters vary across the sweep -> used for run labels.\n"
        "VARY = [k for k in ['learning_rate', 'loss_weights', 'freeze_weights',\n"
        "                    'lr_scheduler', 'batch_size']\n"
        "        if k in summary and summary[k].nunique(dropna=True) > 1]\n\n"
        "def run_label(row):\n"
        "    parts = [pretty(row['model'])]\n"
        "    if 'learning_rate' in VARY and pd.notna(row['learning_rate']):\n"
        "        parts.append(f\"lr{row['learning_rate']:.0e}\")\n"
        "    if 'loss_weights' in VARY and bool(row['loss_weights']):\n"
        "        parts.append('+LW')\n"
        "    if 'freeze_weights' in VARY and bool(row['freeze_weights']):\n"
        "        parts.append('frozen')\n"
        "    if 'batch_size' in VARY and pd.notna(row['batch_size']):\n"
        "        parts.append(f\"b{int(row['batch_size'])}\")\n"
        "    return ' '.join(parts)\n\n"
        "summary['label'] = summary.apply(run_label, axis=1)\n"
        "summary[['run', 'model', 'label', 'test_f1', 'test_acc']]"
    )

    # ---- Panel A ----
    md(
        "## Panel A — Training curves\n\nPer-epoch loss (train dashed / val solid) "
        "and validation F1 for the top runs by test F1."
    )
    if has_history:
        code(
            "history = pd.read_csv('sweep_history.csv')\n"
            "TOP = (summary.sort_values('test_f1', ascending=False)\n"
            "       .drop_duplicates('model').head(3)['run'].tolist())\n\n"
            "fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12 * CM, 5 * CM))\n"
            "for run in TOP:\n"
            "    h = history[history['run'] == run].sort_values('epoch')\n"
            "    if not len(h):\n"
            "        continue\n"
            "    model = h['model'].iloc[0]\n"
            "    c = ARCH_COLOR.get(model, COLOR.BLUE.value)\n"
            "    lab = summary.loc[summary['run'] == run, 'label'].iloc[0]\n"
            "    vloss = h['val_loss'] if 'val_loss' in h else h.get('loss')\n"
            "    if 'train_loss' in h:\n"
            "        ax_loss.plot(h['epoch'], h['train_loss'], color=c, lw=0.8, ls='--', alpha=0.7)\n"
            "    ax_loss.plot(h['epoch'], vloss, color=c, lw=1.2, label=lab)\n"
            "    if 'f1' in h:\n"
            "        ax_f1.plot(h['epoch'], h['f1'], color=c, lw=1.2, label=lab)\n\n"
            "ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss')\n"
            "ax_loss.set_title('Loss (dashed = train, solid = val)')\n"
            "ax_loss.legend(frameon=False)\n"
            "ax_f1.set_xlabel('Epoch'); ax_f1.set_ylabel('Validation F1')\n"
            "ax_f1.set_title('Validation F1')\n"
            "ax_f1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))\n"
            "plt.suptitle('Training curves (top runs)', fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.savefig('figA_training_curves.pdf', bbox_inches='tight')\n"
            "plt.show()"
        )
    else:
        md(
            "> Per-epoch curves require W&B history; this report was built from "
            "local checkpoints (final metrics only), so Panel A is omitted."
        )

    # ---- Panel B ----
    md(
        "## Panel B — Model comparison by test F1\n\nEvery run ranked by test F1; "
        "the winner (deployed) is in bold."
    )
    code(
        "d = summary.sort_values('test_f1', ascending=True).reset_index(drop=True)\n"
        "winner_pos = int(d['test_f1'].idxmax())\n\n"
        "fig, ax = plt.subplots(figsize=(9.5 * CM, max(4, 0.9 * len(d)) * CM))\n"
        "for i, row in d.iterrows():\n"
        "    hatch = '////' if ('loss_weights' in VARY and bool(row['loss_weights'])) else None\n"
        "    ax.barh(i, row['test_f1'], height=0.7, color=ARCH_COLOR.get(row['model'], COLOR.BLUE.value),\n"
        "            hatch=hatch, edgecolor='white', linewidth=0.5)\n"
        "    tag = '  ← deployed' if i == winner_pos else ''\n"
        "    acc = f\"  (acc {row['test_acc']:.3f})\" if pd.notna(row['test_acc']) else ''\n"
        "    ax.text(row['test_f1'] + 0.0015, i, f\"{row['test_f1']:.3f}{acc}{tag}\", va='center', fontsize=5)\n\n"
        "ax.set_yticks(np.arange(len(d)))\n"
        "ax.set_yticklabels(d['label'])\n"
        "ax.get_yticklabels()[winner_pos].set_fontweight('bold')\n"
        "ax.set_xlabel('Test F1')\n"
        "lo = float(np.nanmin(d['test_f1'])) - 0.02\n"
        "ax.set_xlim(max(0, lo), float(np.nanmax(d['test_f1'])) + 0.04)\n"
        "ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))\n"
        "ax.set_title('Model comparison — best test F1 wins')\n"
        "handles = [Patch(facecolor=ARCH_COLOR[m], label=pretty(m)) for m in models]\n"
        "if 'loss_weights' in VARY:\n"
        "    handles.append(Patch(facecolor=COLOR.GREY.value, hatch='////', edgecolor='white', label='loss weights'))\n"
        "ax.legend(handles=handles, frameon=False, fontsize=5, loc='lower right')\n"
        "plt.tight_layout()\n"
        "plt.savefig('figB_model_comparison.pdf', bbox_inches='tight')\n"
        "plt.show()"
    )

    # ---- Panel C ----
    has_compare = compare_label is not None
    md(
        "## Panel C — Winner: per-class precision & recall\n\n"
        + (
            "Comparison of the winner against an alternative model on one identical "
            "held-out split."
            if has_compare
            else "Per-class precision and recall for the winning model."
        )
    )
    code(
        "cmw = pd.read_csv('confusion-matrix.csv', index_col=0)\n"
        "classes = cmw.index.tolist()\n"
        "W = cmw.values.astype(float)\n\n"
        "def prec_rec(c):\n"
        "    tp = np.diag(c)\n"
        "    return tp / c.sum(axis=0), tp / c.sum(axis=1)\n\n"
        "pW, rW = prec_rec(W)\n"
        + (
            "HAS_CMP = True\n"
            "cmp_df = pd.read_csv('confusion-matrix-compare.csv', index_col=0)\n"
            "C2 = cmp_df.values.astype(float)\n"
            "pC, rC = prec_rec(C2)\n"
            if has_compare
            else "HAS_CMP = False\n"
        )
        + "fig, axes = plt.subplots(1, 2, figsize=(11 * CM, 5 * CM), sharey=True)\n"
        "x = np.arange(len(classes)); width = 0.38 if HAS_CMP else 0.6\n"
        "for ax, (mW, mC, title) in zip(axes, [(rW, rC if HAS_CMP else None, 'Recall'),\n"
        "                                       (pW, pC if HAS_CMP else None, 'Precision')]):\n"
        "    if HAS_CMP:\n"
        "        ax.bar(x - width / 2, mW, width, color=COLOR.BLUE.value, label='Winner')\n"
        f"        ax.bar(x + width / 2, mC, width, color=COLOR.PINK.value, label='{compare_label}')\n"
        "    else:\n"
        "        ax.bar(x, mW, width, color=COLOR.BLUE.value)\n"
        "    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in classes])\n"
        "    ax.set_title(title); ax.set_ylim(0.5, 1.0)\n"
        "    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))\n"
        "axes[0].set_ylabel('Score')\n"
        "if HAS_CMP:\n"
        "    axes[0].legend(frameon=False, loc='lower left', fontsize=5)\n"
        "plt.suptitle('Per-class precision & recall', fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.savefig('figC_per_class.pdf', bbox_inches='tight')\n"
        "plt.show()"
    )

    # ---- Panel D ----
    md(
        "## Panel D — Confusion matrix of the winner\n\nRow-normalised (per-class "
        "recall), drawn as vector rectangles for Illustrator. Representative crops "
        "down the left edge."
    )
    code(
        "from PIL import Image\n"
        "import os\n\n"
        "cm = pd.read_csv('confusion-matrix.csv', index_col=0)\n"
        "classes = cm.index.tolist(); M = cm.values.astype(int); n = len(classes)\n"
        "Mn = M / M.sum(axis=1, keepdims=True)\n"
        "total = int(M.sum()); acc = np.trace(M) / M.sum()\n"
        "cmap = colormaps['Blues']; norm = Normalize(vmin=0, vmax=1)\n\n"
        "ex = {c: np.asarray(Image.open(f'report_assets/{c}.jpg'))\n"
        "      for c in classes if os.path.exists(f'report_assets/{c}.jpg')}\n\n"
        "fig = plt.figure(figsize=(7.8 * CM, 5.5 * CM))\n"
        "gs = fig.add_gridspec(n, 3, width_ratios=[1, 5, 0.35], wspace=0.08, hspace=0.15)\n"
        "for i, c in enumerate(classes):\n"
        "    axi = fig.add_subplot(gs[i, 0])\n"
        "    if c in ex:\n"
        "        axi.imshow(ex[c], cmap='gray')\n"
        "    axi.set_xticks([]); axi.set_yticks([])\n"
        "    axi.set_ylabel(c.capitalize(), rotation=0, ha='right', va='center', fontsize=6)\n"
        "    for s in axi.spines.values():\n"
        "        s.set_visible(False)\n"
        "ax = fig.add_subplot(gs[:, 1])\n"
        "for i in range(n):\n"
        "    for j in range(n):\n"
        "        v = Mn[i, j]\n"
        "        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(norm(v)),\n"
        "                               edgecolor='white', linewidth=0.6))\n"
        "        ax.text(j, i, f'{int(M[i, j])}\\n({v:.1%})', ha='center', va='center',\n"
        "                fontsize=6, color='white' if v > 0.5 else COLOR.DARKGREY.value)\n"
        "ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)\n"
        "ax.set_xticks(range(n)); ax.set_yticks(range(n))\n"
        "ax.set_xticklabels([c.capitalize() for c in classes]); ax.set_yticklabels([])\n"
        "ax.set_xlabel('Predicted label')\n"
        "ax.set_title(f'Confusion matrix (n={total}, accuracy={acc:.1%})', pad=4)\n"
        "cax = fig.add_subplot(gs[:, 2]); steps = 50\n"
        "for k in range(steps):\n"
        "    vv = k / (steps - 1)\n"
        "    cax.add_patch(Rectangle((0, vv), 1, 1 / steps, facecolor=cmap(norm(vv)), edgecolor='none'))\n"
        "cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.set_xticks([])\n"
        "cax.yaxis.tick_right(); cax.yaxis.set_label_position('right')\n"
        "cax.set_yticks([0, 0.5, 1.0]); cax.set_yticklabels(['0%', '50%', '100%'], fontsize=5)\n"
        "cax.set_ylabel('Recall', fontsize=6)\n"
        "for s in cax.spines.values():\n"
        "    s.set_visible(False)\n"
        "plt.savefig('figD_confusion_matrix.pdf', bbox_inches='tight')\n"
        "plt.show()"
    )

    # ---- Panel E ----
    md(
        "## Panel E — Gallery on new, unseen data\n\n*Generated separately (e.g. "
        "napari)* — crops from an unseen dataset tiled under each predicted class. "
        "Not produced in this notebook."
    )

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    out_nb = outdir / f"{name}_report.ipynb"
    with open(out_nb, "w") as f:
        nbf.write(nb, f)
    return out_nb


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Generate the deterministic CellClass report notebook."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "npz", type=Path, help="Dataset .npz used for the confusion matrix"
    )
    p.add_argument(
        "--best",
        type=Path,
        required=True,
        help="Winner checkpoint (.pt.best) or scripted model (.pt)",
    )
    p.add_argument(
        "--model",
        help="Architecture name if --best is a checkpoint "
        "(omit if --best is a scripted model with sidecar .json)",
    )
    p.add_argument(
        "--compare", type=Path, help="Optional second model for Panel C"
    )
    p.add_argument(
        "--compare-model", help="Architecture for --compare checkpoint"
    )
    p.add_argument(
        "--compare-label",
        default="Alternative",
        help="Legend label for the --compare model (default: %(default)s)",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--project", help="W&B project (per-sweep)")
    src.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory of local *.pt.best checkpoints",
    )
    p.add_argument("--entity", help="W&B entity (default: api default)")
    p.add_argument(
        "--outdir", type=Path, help="Output dir (default: dataset dir)"
    )
    p.add_argument("--name", help="Report base name (default: dataset stem)")
    p.add_argument("--testing-size", type=float, default=0.2)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--execute", action="store_true", help="Render the notebook to PDFs"
    )
    args = p.parse_args()

    outdir = args.outdir or args.npz.resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.npz.stem

    print("[1/5] Gathering sweep metrics")
    if args.project:
        summary, history = gather_wandb(args.entity, args.project)
    else:
        summary, history = gather_local(args.runs_dir)
    for col in SUMMARY_COLS:
        if col not in summary:
            summary[col] = np.nan
    summary = summary[SUMMARY_COLS]
    summary.to_csv(outdir / "sweep_summary.csv", index=False)
    if len(history):
        history.to_csv(outdir / "sweep_history.csv", index=False)
    winner_run = str(summary.loc[summary["test_f1"].idxmax(), "run"])
    print(f"      {len(summary)} runs; winner = {winner_run}")

    print("[2/5] Confusion matrix (winner)")
    run_confusion(
        args.npz,
        args.best,
        outdir / "confusion-matrix.csv",
        model=args.model,
        device=args.device,
        testing_size=args.testing_size,
        data_seed=args.data_seed,
    )
    compare_label = None
    if args.compare:
        print("[2b ] Confusion matrix (compare)")
        run_confusion(
            args.npz,
            args.compare,
            outdir / "confusion-matrix-compare.csv",
            model=args.compare_model,
            device=args.device,
            testing_size=args.testing_size,
            data_seed=args.data_seed,
        )
        compare_label = args.compare_label

    print("[3/5] Sampling representative crops")
    sample_crops(args.npz, outdir / "report_assets")

    print("[4/5] Copying style assets")
    for asset in ("hhlab_style01.mplstyle", "colors.py"):
        shutil.copy(SKILL_ASSETS / asset, outdir / asset)

    print("[5/5] Building notebook")
    nb_path = build_notebook(
        outdir,
        name=name,
        project=args.project,
        has_history=bool(len(history)),
        winner_run=winner_run,
        compare_label=compare_label,
    )
    print(f"      wrote {nb_path}")

    if args.execute:
        print("      executing notebook ...")
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(nb_path),
            ],
            check=True,
        )
        print(
            f"      panels: {sorted(str(x.name) for x in outdir.glob('fig*.pdf'))}"
        )


if __name__ == "__main__":
    main()
