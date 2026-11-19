---
name: omero-screen
description: Run and troubleshoot OMERO-Screen, the Hochegger lab's high-content imaging pipeline (OMERO → Cellpose → measurements → CellView → figures). Use for checking or fixing an install (`omero-screen doctor`, `setup`, login or VPN problems), running the pipeline on OMERO plates (segmentation, stitching, tracking, `--inference` classifiers), submitting cluster jobs (`omero-screen-sbatch`), importing and exporting CellView data, cell-cycle and feature plots with omero-screen-plots, browsing plates in napari, and galleries or well images with `omero-screen-images`. Typical requests - "analyse plate 1234", "my login fails", "import plate 1234 into CellView", "plot p21 per condition", "show me well B3". For training a cell classifier use cellclass-train; for reviewing tracked cells use track-review.
---

# OMERO-Screen

A pipeline for high-content immunofluorescence screens: plates in OMERO are
segmented with Cellpose, measured per cell, assigned cell-cycle phases, stored
in CellView (a local DuckDB database) and plotted with omero-screen-plots.

Docs: https://hocheggerlab.github.io/omero-screen/ ·
GitHub: https://github.com/HocheggerLab/omero-screen

## Plots: use the omero-screen-plots API first

**For any plot of OMERO-Screen data, use an omero-screen-plots function when
one covers the plot type. Do not hand-roll matplotlib or seaborn for these.**
The functions apply the lab's normalisation, cell-cycle gating, colours,
sizes and per-plate statistics; a hand-made version looks different from the
lab's figures and has no statistics.

```python
from cellview.api import cellview_load_data
from omero_screen_plots import combplot_cellcycle, cellcycle_stacked, feature_plot
df, _ = cellview_load_data(1234)
fig, ax = feature_plot(df, "intensity_mean_p21_nucleus", ["control", "palbo"],
                       selector_val="RPE1", save=True, path="figures")
```

Available: `combplot_cellcycle`, `combplot_feature`, `cellcycle_stacked`,
`cellcycle_plot`, `feature_plot`, `feature_norm_plot`, `count_plot`,
`histogram_plot`, `scatter_plot`, `classification_plot`, `well_qc_plot`.
**Read `references/plotting.md` before plotting**: it maps requests to
functions and gives their arguments. Hand-roll only a plot type none of them
covers, in the lab style described there, and say that it is custom.

## Which install is this?

- **Installed release** (most users): `command -v omero-screen` points into
  `~/.local/bin`. Run commands directly (`omero-screen …`), never `uv run`.
  Settings come from `~/.config/omero-screen/config.toml`, written by
  `omero-screen setup`. There is no repository; don't suggest repo scripts,
  `.env` files or `--env`.
- **Developer checkout**: the user works in a clone and runs `uv run …`.
  Only then read `references/developer.md`.

If unsure, run `command -v omero-screen` and `omero-screen --version`.

## Working with someone who does not code

Many users are biologists who know chat assistants but not the terminal.

- **Check first.** Start with `omero-screen doctor`; when something fails,
  run it again and fix what it reports before trying anything else. Most
  problems are the VPN or a missing `omero-screen setup`.
- **Say what you'll do in plain words** before each command ("I'll look up
  plate 1234 in OMERO") and what the result means after it. Summarise logs;
  don't paste them.
- **napari is a window on the user's screen.** Start it (`napari &`), then
  say exactly which menu to open and what to click.
- **Results go into CellView**; analyse them with `cellview_load_data` and
  the plot functions; save figures as PDF and say where.

## Safety rules

- **Ask first** before anything that writes to OMERO, CellView or the user's
  install: running the pipeline (`omero-screen <plate>`, `--delete`,
  `omero-screen-sbatch`), `omero-screen models publish`,
  `cellview import`/`delete`/`clean`/`curate`, `omero-train delete`,
  `omero-screen self update`/`self uninstall`. Reading, looking and plotting
  need no permission.
- **Never** ask for, read or print the password, `~/.config/omero-screen/`,
  `.env*` files or `~/.netrc`. The user runs `omero-screen setup` themselves,
  in their terminal; it asks for the password there.
- **Never run the pipeline on an HPC login node**; submit a job.

## Dispatch table

Read the reference for the area before answering; for a question spanning
areas, read each relevant one.

| The user asks about… | Read |
|---|---|
| Plots and figures from CellView data | `references/plotting.md` |
| Running the pipeline, options, models, stitching, tracking, HPC jobs | `references/pipeline.md` |
| CellView: import, export, projects, the Python API, notebooks | `references/cellview.md` |
| Images for a finding without napari: galleries, well overviews, `omero-screen-images` | `references/images.md` |
| napari: browsing wells, masks, collecting training crops | `references/napari.md` |
| Classifiers: collecting crops, publishing, `--inference` | `references/classifier-training.md` (training itself: the cellclass-train skill) |
| Install, setup, doctor, config, update, HPC install | `references/environment.md` |
| A developer checkout: `uv run`, `.env`, tests, test server, releases | `references/developer.md` |

## Quick reference

```bash
omero-screen doctor                         # check everything; prints a fix per failure
omero-screen setup                          # the user runs this; asks for the password
omero-screen config show                    # settings in effect and their source
omero-screen --help                         # pipeline options (Click: full names only)

omero-screen 1234                           # run a plate (ask first)
omero-screen 1234 --segmentation            # masks only
omero-screen 1234 --delete                  # segment again from scratch
omero-screen 1234 --inference micronuclei   # with a published classifier
omero-screen-sbatch 1234 --stitch --track   # on an HPC cluster: submit a job

cellview projects                           # what's in the database
cellview import plate 1234                  # results from OMERO into CellView (ask first)
cellview explore 1234 --template cellcycle  # a ready-made notebook
omero-screen-images --help                  # galleries and well images without napari
```

## Common issues

| Problem | Fix |
|---|---|
| Anything fails | `omero-screen doctor`, then fix what it reports |
| Can't connect to OMERO | VPN; then `omero-screen setup` |
| `command not found` | New terminal window, or `uv tool update-shell` |
| Which model segmented a plate? | The `omero-screen/provenance` map annotation on the plate |
| Default segmentation model | Nothing configured: Cellpose 4 (cpsam) on an NVIDIA GPU, stock Cellpose 3 (nuclei, cyto3) on a Mac or CPU; `--cp4`/`--model` override |
| A re-run didn't change the masks | Stored masks are reused; add `--delete` |
| CellView import of a CSV fails | It needs `plate_id`, `cell_line`, `condition` plus measurement columns |
