# Combined Scatter + Cell-cycle Plot Example

This document illustrates how to create a **combination plot** (scatter plot aligned with cell-cycle distribution) using the `comb_plot` helper from `omero_screen_plots`.

---

## 1. Install the package

```bash
pip install omero-screen-plots  # or `pip install -e .` from the repo root
```

---

## 2. Download example data (Zenodo)

```bash
mkdir -p data && cd data
wget https://zenodo.org/record/15728770/files/sample_plate_data.csv?download=1 -O sample_plate_data.csv
cd ..
```

> You only have to download the dataset once. Add the file to `.gitignore` if you do not wish to track large data files in Git.

---

## 3. Quick start

```python
import pandas as pd
from omero_screen_plots.combplot import comb_plot

# DataFrame with one row per single cell or per measurement
csv_path = "data/sample_plate_data.csv"
df = pd.read_csv(csv_path)

# Conditions that will appear on the x-axis
conditions = ["ctr", "cdk1i", "cdk2i", "cdk4i"]

comb_plot(
    df=df,
    conditions=conditions,
    feature_col="intensity_mean_yH2AX_nucleus",  # y-axis of upper scatter plot
    feature_y_lim=6000,
    condition_col="condition",  # column that matches the `conditions` list
    selector_col="cell_line",   # optional subdivision of the dataset
    selector_val="RPE1wt",
    title="Combined plot example – RPE1 (WT)",
    cell_number=1000,           # subsample to 1k cells for performance
    save=True,
    path="combplot_example.png",
)
```

The function creates **two aligned axes**:

1. Upper scatter plot – each point represents a cell and is coloured by cell-cycle phase.
2. Lower stacked bar plot – summarises the proportion of cells in each phase.

Example output (truncated for size):

![combplot example](../images/combplot.png)

---

## 4. Extra options

* **`cell_number`** – down-sample the number of points (pass `None` to plot all).
* **`feature_y_lim`** – specify y-axis limits for the scatter plot.
* **Stats overlay** – when at least three imaging plates are available, `omero_screen_plots` performs a t-test and annotates p-values.

---

### Reproducible research tip

For publications or sharing, consider exporting the final figure as a PDF instead of PNG for better vector quality:

```python
comb_plot(..., path="combplot_example.pdf", save=True)
```

---

## 5. See also

* `cellcycle_standard.md` – canonical bar plot.
* `feature_plot_example.md` – expression/feature distribution across treatments.

---

© 2025 • Helfrid Hochegger – MIT License
