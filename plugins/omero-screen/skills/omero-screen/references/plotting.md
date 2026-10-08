# omero-screen-plots — Visualisation API

Publication-ready statistical plots for IF cell cycle data from omero-screen.

Docs: https://hocheggerlab.github.io/omero-screen/
Package: `packages/omero-screen-plots/`

---

## Setup

```python
# Load data first
from cellview.api import cellview_load_data
df, variable_names = cellview_load_data(12345)

# Filter to cell line of interest
df_rpe = df[df["cell_line"] == "RPE"]
conditions = ["control", "drug_10nM", "drug_100nM"]
```

---

## Plot Types

### Combined Plots (recommended starting point)

#### Cell cycle analysis panel
```python
from omero_screen_plots import combplot_cellcycle

fig, axes = combplot_cellcycle(
    df=df_rpe,
    conditions=conditions,
    selector_col="cell_line",   # optional filter column
    selector_val="RPE",          # optional filter value
    save=True,                   # save to file
    file_format="svg",           # "svg" | "pdf" | "png"
    filename="cellcycle_analysis"
)
```
Produces a 2×2 grid: scatter (DAPI vs EdU), stacked bar (phase %), individual phase bars, count plot.

#### Feature analysis panel
```python
from omero_screen_plots import combplot_feature

fig, axes = combplot_feature(
    df=df_rpe,
    conditions=conditions,
    feature="intensity_mean_p21_nucleus",
    threshold=5000,        # horizontal threshold line
    cell_number=3000,      # subsample N cells for scatter plot
    norm_control="control", # normalise to this condition
    save=True,
    file_format="svg"
)
```
Produces: feature distribution plot, normalised plot, threshold scatter.

---

### Individual Plot Types

#### Cell cycle stacked barplot
```python
from omero_screen_plots import cellcycle_stacked

fig, ax = cellcycle_stacked(
    df=df_rpe,
    conditions=conditions
)
```

#### Cell cycle phase quantification (2×2 grid)
```python
from omero_screen_plots import cellcycle_plot

fig, axes = cellcycle_plot(
    df=df_rpe,
    conditions=conditions
)
```

#### Feature box/violin plot
```python
from omero_screen_plots import feature_plot

fig, ax = feature_plot(
    df=df_rpe,
    feature="area_nucleus",
    conditions=conditions,
    plot_type="violin"  # "box" | "violin"
)
```

#### Feature plot with threshold
```python
from omero_screen_plots import feature_plot_norm

fig, axes = feature_plot_norm(
    df=df_rpe,
    feature="intensity_mean_p21_nucleus",
    conditions=conditions,
    threshold=5000,
    norm_control="control"
)
```

#### Cell count analysis
```python
from omero_screen_plots import count_plot

fig, ax = count_plot(
    df=df_rpe,
    conditions=conditions,
    norm_control="control",  # normalise to this condition
    absolute=False           # True for absolute counts
)
```

#### Histogram / distribution
```python
from omero_screen_plots import histogram_plot

fig, ax = histogram_plot(
    df=df_rpe,
    feature="integrated_int_DAPI",
    conditions=conditions,
    log_scale=True,  # log x-axis
    kde=True         # overlay KDE
)
```

#### Scatter plot (cell cycle coloured)
```python
from omero_screen_plots import scatter_plot

fig, ax = scatter_plot(
    df=df_rpe,
    x_feature="integrated_int_DAPI",
    y_feature="intensity_mean_EdU_nucleus",
    conditions=conditions,
    color_by="cell_cycle"  # colour points by cell cycle phase
)
```

#### Classification results
```python
from omero_screen_plots import classification_plot

fig, ax = classification_plot(
    df=df_rpe,
    classifier_col="classifier_micronuclei_v1",
    conditions=conditions
)
```

---

## Normalisation

Intensity data is normalised to the histogram mode (DNA content = 1.0 for G1 cells):

```python
from omero_screen_plots.normalise import normalize_by_mode

# Normalise a column
df["dapi_norm"] = normalize_by_mode(df["integrated_int_DAPI"])

# Per-condition normalisation
df["dapi_norm"] = df.groupby("condition")["integrated_int_DAPI"].transform(
    normalize_by_mode
)
```

This is applied automatically inside the cell cycle plot functions. For custom analysis:

```python
from omero_screen_plots.normalise import find_intensity_mode

mode = find_intensity_mode(df["integrated_int_DAPI"])
df["dapi_norm"] = df["integrated_int_DAPI"] / mode
```

---

## Statistical Analysis

When ≥3 biological replicates exist in the data, plots automatically run statistical tests and mark significance:

```python
from omero_screen_plots.stats import plate_stats

# Get plate-level aggregated statistics
stats_df = plate_stats(
    df=df_rpe,
    feature="intensity_mean_p21_nucleus",
    conditions=conditions,
    groupby="plate_id"
)
```

Significance annotations follow standard conventions: `ns`, `*`, `**`, `***`, `****`.

---

## Saving Figures

```python
# Save as SVG (recommended for publication — vector format)
fig.savefig("figure1.svg", bbox_inches="tight", dpi=300)

# Save as PDF
fig.savefig("figure1.pdf", bbox_inches="tight")

# Using the built-in save parameter
combplot_cellcycle(df=df, conditions=conditions, save=True, file_format="svg",
                   filename="my_figure", output_dir="/path/to/figures/")
```

---

## Available Feature Columns

Standard features from the pipeline (per compartment: `_nucleus`, `_cell`, `_cytoplasm`):

```
area_{compartment}
intensity_mean_{channel}_{compartment}
intensity_max_{channel}_{compartment}
intensity_min_{channel}_{compartment}
integrated_int_DAPI     ← sum of DAPI, key for cell cycle
cell_cycle              ← G1 | S | G2 | Polyploid | SubG1
cell_cycle_detailed     ← adds Mitotic subdivision
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `KeyError: cell_cycle` | Run the pipeline with EdU channel; cell cycle requires DNA content + S-phase marker |
| Empty plots | Check `conditions` list matches actual values in `df["condition"]` |
| Statistical test fails | Need ≥3 plates in the experiment for plate-level stats |
| Normalisation looks wrong | Check `integrated_int_DAPI` is present; single-cell values, not already normalised |
| Figures too small | Increase `figsize` parameter or adjust `rcParams` after import |
