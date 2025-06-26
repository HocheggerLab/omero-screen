# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%

import pandas as pd

# Import the new classes
from omero_screen_plots import cellcycle_standard_plot

# %%
df = pd.read_csv("data/sample_plate_data.csv")

conditions = [
    "palb:0.0 c604:0",
    "palb:0.0 c604:1",
    "palb:0.75 c604:0",
    "palb:0.75 c604:1",
]

# %%
# 1) Minimal example

fig = cellcycle_standard_plot(
    data=df,
    conditions=conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    title="Cell cycle plot example",
    figsize=(10, 10),
    phases=["G1", "S", "G2/M", "Polyploid"],
)

# %%
