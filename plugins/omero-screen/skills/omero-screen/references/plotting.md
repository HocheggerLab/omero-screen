# omero-screen-plots: the plotting API

**Use these functions for every plot they cover.** They apply the lab's
normalisation, cell-cycle gating, colours, figure sizes (in cm) and statistics
(per-plate medians, significance marks). A hand-rolled matplotlib or seaborn
version of the same plot looks different from the lab's published figures and
skips the statistics. Hand-roll only a plot type that no function below
covers, and then keep the lab style (see the end of this file).

Docs, with an example figure for each function:
https://hocheggerlab.github.io/omero-screen/ (omero-screen-plots section).

---

## Load the data

```python
from cellview.api import cellview_load_data

df, variable_names = cellview_load_data(12345)                 # one plate
df, variable_names = cellview_load_data(12345, 12346, 12347)   # several plates
df, variable_names = cellview_load_data(experiment="palb_washout")  # by experiment
```

Look at the values before plotting, and use them exactly:

```python
df["condition"].unique(), df["cell_line"].unique(), df["plate_id"].unique()
```

Every function takes `conditions` (a list, in plot order; the first one is the
control for statistics) and optional `selector_col` / `selector_val` to pick one
cell line (`selector_col="cell_line", selector_val="RPE1"`). `condition_col`
defaults to `"condition"`.

**Statistics** need repeats: with three or more plates (`plate_id`), each plot
shows per-plate medians and significance marks against the first condition.
With fewer, it shows the data without tests.

---

## Which function

| The user wants… | Function |
|---|---|
| Cell-cycle overview: DNA histogram, EdU vs DNA scatter, phase bars | `combplot_cellcycle` |
| One feature across conditions with the cell-cycle scatter | `combplot_feature` |
| Percent of cells per phase, stacked | `cellcycle_stacked` |
| One panel per phase (G1, S, G2/M, …) | `cellcycle_plot` |
| Phases for grouped conditions (e.g. pairs) | `cellcycle_grouped` (import from `omero_screen_plots.cellcycleplot_api`) |
| A feature's single-cell distribution (box/boxen or violin) | `feature_plot` |
| Fraction of cells above a threshold of a normalised feature | `feature_norm_plot` |
| Cell number per condition (relative or absolute) | `count_plot` |
| Histogram of a feature | `histogram_plot` |
| Any x/y scatter (default: EdU vs DNA) | `scatter_plot` |
| Classifier results: percent per class | `classification_plot` |
| Per-well QC overview | `well_qc_plot` |
| Drug synergy (Bliss, HSA) | `omero_screen_plots.synergy` |

All are importable from `omero_screen_plots` unless noted.

---

## The functions

Signatures are abbreviated to the arguments you need most; every function
also takes `title`, `fig_size` + `size_units="cm"`, `save`, `path`,
`file_format` (`"pdf"` default for single plots, `"png"` for combined ones)
and `dpi`. Check the full signature with `help(function)` before using an
argument not shown here.

```python
from omero_screen_plots import (
    combplot_cellcycle, combplot_feature, cellcycle_stacked, cellcycle_plot,
    feature_plot, feature_norm_plot, count_plot, PlotType, histogram_plot,
    scatter_plot, classification_plot, well_qc_plot, save_fig,
)
conditions = ["control", "palbo_100nM", "palbo_1uM"]

# Cell cycle
fig, axes = combplot_cellcycle(df, conditions, selector_val="RPE1",
                               cell_number=3000, save=True, path="figures")
fig, ax = cellcycle_stacked(df, conditions, selector_val="RPE1",
                            show_triplicates=True, save=True, path="figures")
fig, axes = cellcycle_plot(df, conditions, selector_val="RPE1",
                           show_subG1=False, save=True, path="figures")

# One feature
fig, axes = combplot_feature(df, conditions, feature="intensity_mean_p21_nucleus",
                             threshold=None, save=True, path="figures")
fig, ax = feature_plot(df, "intensity_mean_p21_nucleus", conditions,
                       violin=False, ymax=None, save=True, path="figures")
fig, ax = feature_norm_plot(df, "intensity_mean_p21_nucleus", conditions,
                            threshold=1.5, save=True, path="figures")

# Counts: norm_control comes before conditions
fig, ax = count_plot(df, "control", conditions,
                     plot_type=PlotType.NORMALISED,   # or PlotType.ABSOLUTE
                     save=True, path="figures")

# Distributions and scatter
fig, ax = histogram_plot(df, "integrated_int_DAPI_norm", conditions,
                         bins=100, log_scale=True, kde_overlay=False,
                         save=True, path="figures")
fig, ax = scatter_plot(df, conditions,
                       x_feature="integrated_int_DAPI_norm",
                       y_feature="intensity_mean_EdU_nucleus_norm",
                       hue="cell_cycle", cell_number=3000,
                       save=True, path="figures")

# Classifier results: classes as named in class_col
fig, ax = classification_plot(df, classes=["normal", "micronuclei"],
                              conditions=conditions, class_col="Class",
                              save=True, path="figures")

# Well QC
fig = well_qc_plot(df, save=True, path="figures")
```

`save=True` writes the figure into `path` (a folder; nothing is written without
a `path`) under a name the function chooses; `save_stats=True` also writes the
p-values and per-plate medians as CSV. To choose the file name, save afterwards:
`save_fig(fig, Path("figures"), "fig2_p21", fig_extension="pdf")`.

Several functions take `axes=` so you can place them in your own layout
(e.g. `fig, axs = plt.subplots(1, 2)`, then `feature_plot(..., axes=axs[0],
save=False)`).

---

## Columns

The pipeline writes, per cell:

```
area_{nucleus,cell,cytoplasm}
intensity_mean_<channel>_<compartment>      e.g. intensity_mean_p21_nucleus
intensity_max_… / intensity_min_…
integrated_int_DAPI                         DNA content
integrated_int_DAPI_norm                    DNA content, G1 peak = 2 (cell-cycle plots use this)
intensity_mean_EdU_nucleus_norm             EdU, normalised (cell-cycle scatter)
cell_cycle                                  Sub-G1 | G1 | S | G2/M | Polyploid (G2 and M separately with a mitosis marker)
cell_cycle_detailed                         also splits Early S / Late S and polyploid (non-)replicating
plate_id, well, condition, cell_line, …     from the plate metadata
```

Channel names come from the plate's channel annotation, so check
`variable_names` or `df.columns` rather than guessing.

To normalise another intensity the same way (mode = 1.0):

```python
from omero_screen_plots.normalise import normalize_by_mode
df = normalize_by_mode(df, "intensity_mean_p21_nucleus", group_column="plate_id")
# adds intensity_mean_p21_nucleus_norm
```

---

## When no function fits

Hand-roll with matplotlib only then, and match the lab style:

```python
import matplotlib.pyplot as plt
from omero_screen_plots.colors import COLOR   # the lab palette (an Enum)
```

Use figure sizes in cm (1 inch = 2.54 cm), the palette above, per-plate
medians when there are repeats, and save as PDF with `save_fig`. Say in the
answer that this plot is custom.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `KeyError` on `cell_cycle` or a `_norm` column | The plate was run without the EdU/cell-cycle step; cell-cycle plots need it. Use `feature_plot`/`count_plot` instead |
| Empty plot or missing condition | `conditions` must match `df["condition"]` values exactly (case, spaces) |
| No significance marks | Needs three or more plates (`plate_id`) with the same conditions |
| Wrong cell line mixed in | Set `selector_col="cell_line", selector_val="…"` |
| Figure too small/large | `fig_size=(w, h)` is in cm by default (`size_units="cm"`) |
