CellCycleStacked API
===================

.. currentmodule:: omero_screen_plots.cellcycleplot_api

The cellcycle_stacked module provides unified visualization for cell cycle phase distributions using stacked bar plots. These plots display the percentage of cells in each cell cycle phase (G1, S, G2/M, etc.) as stacked bars, with options for summary statistics or individual triplicate display. The function replaces separate cellcycle_stacked and cellcycle_grouped functions with a single, configurable interface.

Main Functions
--------------

.. autofunction:: cellcycle_stacked

Examples
--------

Basic Stacked Plot
~~~~~~~~~~~~~~~~~~

Create a basic stacked bar plot showing cell cycle phase proportions with error bars::

    from omero_screen_plots import cellcycle_stacked
    import pandas as pd

    df = pd.read_csv("data.csv")
    fig, ax = cellcycle_stacked(
        df=df,
        conditions=['control', 'cond01', 'cond02', 'cond03'],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        title="cellcycle stacked default",
        fig_size=(7, 5),
        save=True,
        file_format="svg"
    )

.. image:: ../_static/cellcycle_stacked_default.svg

Stacked Plot without Error Bars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove error bars for a cleaner appearance::

    fig, ax = cellcycle_stacked(
        df=df,
        conditions=['control', 'cond01', 'cond02', 'cond03'],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        title="cellcycle stacked no errorbars",
        show_error_bars=False,
        fig_size=(7, 5)
    )

.. image:: ../_static/cellcycle_stacked_no_errorbars.svg

Triplicates with Boxes
~~~~~~~~~~~~~~~~~~~~~~

Show individual triplicate data with boxes around grouped replicates::

    fig, ax = cellcycle_stacked(
        df=df,
        conditions=['control', 'cond01', 'cond02', 'cond03'],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        title="cellcycle stacked triplicates",
        show_triplicates=True,
        group_size=2,  # Group conditions in pairs
        fig_size=(7, 6)
    )

.. image:: ../_static/cellcycle_stacked_triplicates.svg

DNA Content Terminology
~~~~~~~~~~~~~~~~~~~~~~~~

Use DNA content naming convention instead of cell cycle phases::

    fig, ax = cellcycle_stacked(
        df=df,
        conditions=["control", "cond01", "cond02"],
        selector_val="MCF10A",
        cc_phases=False,  # Use DNA content terminology (<2N, 2N, S, 4N, >4N)
        title="cellcycle stacked DNA content"
    )

.. image:: ../_static/cellcycle_stacked_DNA_content.svg

Custom Phase Selection
~~~~~~~~~~~~~~~~~~~~~~

Display only specific cell cycle phases and exclude others::

    fig, ax = cellcycle_stacked(
        df=df,
        conditions=["control", "cond01", "cond02"],
        selector_val="MCF10A",
        phase_order=["G1", "S", "G2/M"],  # Exclude Sub-G1 and Polyploid
        title="cellcycle stacked custom phases"
    )

.. image:: ../_static/cellcycle_stacked_custom_phases.svg

Plot without Legend
~~~~~~~~~~~~~~~~~~~

Remove legend for use in subplots or when creating custom legends::

    fig, ax = cellcycle_stacked(
        df=df,
        conditions=["control", "cond01"],
        selector_val="MCF10A",
        show_legend=False,  # Remove legend
        title="cellcycle stacked no legend"
    )

.. image:: ../_static/cellcycle_stacked_no_legend.svg

Triplicates without Boxes
~~~~~~~~~~~~~~~~~~~~~~~~~

Show individual triplicate bars but without the surrounding boxes::

    fig, ax = cellcycle_stacked(
        df=df,
        conditions=["control", "cond01"],
        selector_val="MCF10A",
        show_triplicates=True,  # Show individual bars
        show_boxes=False,       # But don't draw boxes
        title="cellcycle stacked triplicates no boxes"
    )

.. image:: ../_static/cellcycle_stacked_triplicates_no_boxes.svg

Custom Color Scheme
~~~~~~~~~~~~~~~~~~~

Use custom colors for the cell cycle phases::

    custom_colors = [
        "#FFB6C1",  # Light pink for Sub-G1
        "#87CEEB",  # Sky blue for G1
        "#98FB98",  # Pale green for S
        "#F0E68C",  # Khaki for G2/M
        "#DDA0DD",  # Plum for Polyploid
    ]

    fig, ax = cellcycle_stacked(
        df=df,
        conditions=["control", "cond01", "cond02"],
        selector_val="MCF10A",
        colors=custom_colors,
        title="cellcycle stacked custom colors"
    )

.. image:: ../_static/cellcycle_stacked_custom_colors.svg

Advanced Grouping with Triplicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine grouping with triplicate mode for complex experimental layouts::

    fig, ax = cellcycle_stacked(
        df=df,
        conditions=['control', 'cond01', 'cond02', 'cond03'],
        selector_val="MCF10A",
        show_triplicates=True,     # Individual bars per plate
        show_boxes=True,          # Boxes around triplicates
        group_size=2,             # Group in pairs
        within_group_spacing=0.01, # Space between bars within group
        between_group_gap=0.1,    # Gap between groups
        repeat_offset=0.15,       # Closer spacing for triplicates
        title="cellcycle stacked advanced grouping",
        fig_size=(8, 6)          # Wider figure for complex layout
    )

.. image:: ../_static/cellcycle_stacked_advanced_grouping.svg

Combined Analysis with Other Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create multi-panel figures combining cell cycle analysis with feature analysis::

    import matplotlib.pyplot as plt
    from omero_screen_plots import feature_norm_plot
    from omero_screen_plots.utils import save_fig

    fig, axes = plt.subplots(nrows=2, figsize=(2, 5))

    # Feature normalization plot
    feature_norm_plot(
        df=df,
        feature="area_cell",
        conditions=['control', 'cond01', 'cond02', 'cond03'],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        show_triplicates=True,
        group_size=2,
        within_group_spacing=0.1,
        between_group_gap=0.2,
        x_label=False,
        axes=axes[0],
    )
    axes[0].set_title("feature norm plot", fontsize=8)

    # Cell cycle stacked plot
    cellcycle_stacked(
        df=df,
        conditions=['control', 'cond01', 'cond02', 'cond03'],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        show_triplicates=True,
        group_size=2,
        within_group_spacing=0.1,  # Match feature plot spacing
        between_group_gap=0.2,      # Match feature plot spacing
        axes=axes[1],
    )
    axes[1].set_title("cellcycle stacked plot", fontsize=8)

    fig.suptitle("Combined Analysis", fontsize=8, weight="bold")
    save_fig(fig, "output/", "combined_analysis", fig_extension="svg")

.. image:: ../_static/cellcycle_stacked_combined.svg

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

The cellcycle_stacked function offers extensive customization options:

**Display Modes**:

- **Summary mode** (default): Aggregated bars with error bars showing mean ± SEM across replicates
- **Triplicate mode**: Individual bars for each replicate/plate with optional box outlines
- **Mixed grouping**: Combine conditions into groups with custom spacing

**Phase Options**:

- **Cell cycle terminology** (default): G1, S, G2/M phases with Sub-G1 and Polyploid phases
- **DNA content terminology**: <2N, 2N, S, 4N, >4N based on DNA content measurements
- **Custom phase selection**: Specify exactly which phases to display and their order
- **Automatic phase detection**: Uses available phases from the data automatically

**Visual Customization**:

- **Error bars**: Toggle error bars on summary plots (mean ± SEM)
- **Boxes**: Draw boxes around triplicate groups for visual organization
- **Legend**: Control legend display and positioning
- **Colors**: Use default phase colors or specify custom color schemes
- **Spacing**: Fine-tune within-group and between-group spacing

**Layout & Export**:

- **Figure size**: Control dimensions in cm or inches
- **Grouping**: Organize conditions into visual groups
- **Subplot integration**: Use with existing axes for multi-panel figures
- **Export formats**: Save as SVG, PDF, PNG with custom resolution

**Data Requirements**:

- **Required columns**: 'cell_cycle', 'plate_id', and condition column
- **Cell cycle phases**: Detected automatically from data
- **Replicates**: Uses plate_id to identify biological replicates
- **Statistical analysis**: Requires ≥3 plates for error bar calculations

This unified function replaces the need for separate cellcycle_stacked and cellcycle_grouped functions, providing all functionality through configuration options while maintaining backward compatibility.
