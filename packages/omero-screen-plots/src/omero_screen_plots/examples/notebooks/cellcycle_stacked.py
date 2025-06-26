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
from pathlib import Path

import pandas as pd

# Import the new classes
from omero_screen_plots import cellcycle_stacked_plot

# %%
df = pd.read_csv("data/sample_plate_data.csv")

conditions = [
    "palb:0.0 c604:0",
    "palb:0.0 c604:1",
    "palb:0.75 c604:0",
    "palb:0.75 c604:1",
]

# %%
path = Path("./images/cellcycle")
path.mkdir(parents=True, exist_ok=True)


# %%
# 1) Minimal example

fig = cellcycle_stacked_plot(
    data=df,
    conditions=conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    title="Cell cycle plot example",
    figsize=(6, 4),
    show_error_bars=False,
    output_path=path,
    save=True,
    filename="cellcycle_stacked_minimal",
    file_format="png",  # default is pdf
    dpi=300,  # default is 300
)

# %%
# 2) Larger example with grouping
conditions = [
    "palb:0.0 c604:0",
    "palb:0.0 c604:1",
    "palb:0.375 c604:0",
    "palb:0.375 c604:1",
    "palb:0.75 c604:0",
    "palb:0.75 c604:1",
    "palb:1.5 c604:0",
    "palb:1.5 c604:1",
]

df = pd.read_csv("data/sample_plate_data.csv")

fig = cellcycle_stacked_plot(
    data=df,
    conditions=conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    title="Cell cycle plot example",
    figsize=(8, 4),
    show_error_bars=False,
    group_size=2,
    between_group_gap=1,
    output_path=str(path),
    save=True,
    filename="cellcycle_stacked_grouped",
    file_format="png",  # default is pdf
)

# %%
# 3) Integrating plost using ax
from matplotlib import pyplot as plt

from omero_screen_plots.utils import save_fig

conditions_0 = [
    "palb:0.0 c604:0",
    "palb:0.375 c604:0",
    "palb:0.75 c604:0",
    "palb:1.5 c604:0",
]

conditions_1 = [
    "palb:0.0 c604:1",
    "palb:0.375 c604:1",
    "palb:0.75 c604:1",
    "palb:1.5 c604:1",
]


n_rows = len(conditions)
fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(1, 2))


for idx, (ax, cond_list) in enumerate(zip(axs, conditions, strict=False)):
    cellcycle_stacked_plot(
        data=df,
        conditions=conditions,
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        title=f"Cell cycle plot example {idx + 1}",
        ax=ax,
        show_x_label=(
            idx == n_rows - 1
        ),  # only bottom subplot shows tick labels
        show_error_bars=True,
    )
save_fig(
    fig,
    path,
    "cellcycle_stacked_multiple_axes",
    fig_extension="png",
    tight_layout=True,
)


# %%
# All arguments of the cellcycle_stacked_plot function


# data: DataFrame containing cell cycle data with required columns:
#       - 'cell_cycle': Cell cycle phase for each cell
#       - 'plate_id': Plate/replicate identifier
#       - 'experiment': Unique cell identifier
#       - condition_col: Column containing experimental conditions
#       - selector_col: Column for data selection (e.g., cell_line)
# conditions: List of experimental conditions to plot

# # Data filtering arguments
# condition_col: Name of column containing experimental conditions
# selector_col: Name of column for data filtering (e.g., 'cell_line')
# selector_val: Value to filter by in selector_col (e.g., 'RPE-1')

# # Plot appearance arguments
# title: Overall plot title. If None, auto-generated from selector_val
# colors: Custom color palette. If None, uses default from config
# figsize: Figure size as (width, height) in inches. If None, uses default
# phases: List of cell cycle phases to plot. If None, uses default order ["SubG1", "G1", "S", "G2/M", "Polyploid"]

# # Stacked plot specific arguments
# reverse_stack: If True, reverse the stacking order of phases
# show_legend: Whether to show the phase legend
# legend_position: Legend position ("right", "bottom", "top", "left")
# group_size: If >0, arrange conditions into visual groups of this size
# within_group_spacing: Spacing between conditions inside a group
# between_group_gap: Extra space between consecutive groups
# bar_width: Optional bar width for each bar
# show_error_bars: If True, draw standard-deviation error bars on each segment
# error_bar_capsize: Size of the error-bar caps
# error_bar_color: Color of the error-bar lines

# # Integration arguments
# ax: Optional matplotlib axes to plot on. If provided, creates subplot
# show_x_label: Whether to show the x-axis label

# # Output arguments
# save: Whether to save the figure to file
# output_path: Directory or full path for saving. Required if save=True
# filename: Specific filename. If None, auto-generated based on parameters

# # Save quality arguments
# dpi: Resolution for saved figure (dots per inch)
# file_format: File format ('pdf', 'png', 'svg', etc.)
# tight_layout: Whether to apply tight layout before saving

# **kwargs: Additional arguments passed to the base class
