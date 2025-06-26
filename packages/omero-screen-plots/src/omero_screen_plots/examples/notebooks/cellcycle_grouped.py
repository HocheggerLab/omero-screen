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

# %% [markdown]
# ## Cell Cycle Grouped Plot Examples
#
# This script demonstrates the cellcycle_grouped_plot() function, which creates grouped
# bar charts showing individual replicates within each condition. This visualization is ideal for:
#
# - Replicate variability analysis: See individual replicate performance
# - Outlier detection: Identify replicates that deviate from the norm
# - Experimental consistency: Assess reproducibility across replicates
# - Detailed analysis: Most granular view of cell cycle distributions
#
# Key Features:
# - Individual replicate bars within each condition group
# - Stacked by cell cycle phases
# - Configurable group organization
# - Optional group boxes for visual separation
# - Full integration with matplotlib subplots∑

# %% [markdown]
#

# %%
# Import required libraries
from pathlib import Path

import pandas as pd

# Import the grouped plot function
from omero_screen_plots.plots.cellcycle.grouped import cellcycle_grouped_plot

# %%
data_path = Path("data/sample_plate_data.csv")
cell_data = pd.read_csv(data_path)

# %%
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


# %%
for cell_line in cell_data.cell_line.unique():
    fig = cellcycle_grouped_plot(
        # REQUIRED PARAMETERS
        data=cell_data,
        conditions=conditions,  # Available conditions from data
        # BASE CLASS ARGUMENTS (explicitly showing defaults)
        condition_col="condition",  # Default column name
        selector_col="cell_line",  # Default selector column
        selector_val=cell_line,  # Required for filtering
        title=None,  # Auto-generated: "Cell Cycle Analysis - RPE1wt (Individual Replicates)"
        colors=None,  # Uses package default color palette
        figsize=(5, 3),  # Uses config default size
        # CELLCYCLE GROUPED PLOT SPECIFIC ARGUMENTS (showing defaults)
        phases=None,  # Uses ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]
        group_size=2,  # Number of conditions per visual group
        n_repeats=3,  # Number of replicates to show per condition
        repeat_offset=0.18,  # Spacing between replicate bars
        bar_width=None,  # Auto-calculated (repeat_offset * 1.05)
        show_group_boxes=True,  # Draw boxes around condition groups
        show_legend=True,  # Show phase legend
        # INTEGRATION ARGUMENTS
        ax=None,  # Create own figure
        # OUTPUT ARGUMENTS (showing defaults)
        save=False,  # Don't save by default
        output_path=None,  # No default save path
        filename=None,  # Auto-generated if saving
        # SAVE QUALITY ARGUMENTS (showing defaults)
        dpi=300,  # Standard resolution
        format="pdf",  # Default file format
        tight_layout=True,  # Apply tight layout
    )

# %%
