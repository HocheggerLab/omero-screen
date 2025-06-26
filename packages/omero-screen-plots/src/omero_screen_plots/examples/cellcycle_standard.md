# Cell Cycle Plot – Standard Example

This example demonstrates how to generate a standard cell-cycle bar plot (grouped by phase) with the `omero-screen-plots` package.

---

## 1. Requirements

```bash
#install from the repo root via
uv pip install -e .
```

The example was tested with **Python 3.12** and `omero-screen-plots` ≥ 0.1.3-alpha.

> **Note** `omero-screen-plots` relies on `pandas`, `seaborn`, `matplotlib`, and `scipy`. These dependencies are installed automatically with the package.

---

## 2. Retrieve the sample dataset

The dataset used below is available on **Zenodo** under DOI `10.5281/zenodo.15728770`.

```bash
# Download the CSV file (~ few MB)
mkdir -p data && cd data

# wget is optional – use curl if you prefer.
wget https://zenodo.org/records/15728770/files/omero-screen-plots-sampledata.csv?download=1 -O sample_plate_data.csv
cd ..
```

If you prefer, you can download the data through the Zenodo web interface and place the file in a local `data/` folder.

---

## 3. Minimal code example

```python
import pandas as pd
from omero_screen_plots.plots.cellcycle.standard import cellcycle_plot

# 1. Load the data
csv_path = "data/sample_plate_data.csv"
df = pd.read_csv(csv_path)

# 2. Define the experimental conditions that should appear on the x-axis
#    (order matters – change it to suit your experiment)
conditions = [
    "ctr",  # negative control
    "cdk1i",
    "cdk2i",
    "cdk4i",
]

# 3. Generate the plot
cellcycle_plot(
    df=df,
    conditions=conditions,
    condition_col="condition",   # column containing treatment labels
    selector_col="cell_line",    # split the data by cell-line if desired
    selector_val="RPE1wt",       # choose the value to plot
    title="Cell-cycle distribution – RPE1 (WT)",
    save=True,                    # save the figure instead of displaying
    path="cellcycle_standard.png",
)
```

Running the code above will create a file named `cellcycle_standard.png` similar to:

![Cell-cycle standard plot](../images/cellcycle.png)

---

## 4. Next steps

* Explore the full API in the interactive Jupyter notebook `cellcycle_standard.ipynb` located in the same folder.
* Have a look at other example files (e.g. `combplot_example.md`) to learn about the rest of the plotting helpers.

---

## 5. Troubleshooting

* **ValueError: Unknown column names** – double-check that `condition_col` and `selector_col` match the column names in your CSV file.
* **Empty/blank plot** – confirm that `selector_val` exists in the chosen `selector_col`.

---

© 2025 • Helfrid Hochegger – Released under the MIT License
