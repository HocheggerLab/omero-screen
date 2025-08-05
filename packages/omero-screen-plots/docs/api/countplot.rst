CountPlot API
=============

.. currentmodule:: omero_screen_plots.countplot_api

The count plot module provides visualization for cell count analysis across experimental conditions. Count plots support both normalized and absolute count display, statistical significance testing, and flexible grouping layouts for comparative analysis.

Main Functions
--------------

.. autofunction:: count_plot

Examples
--------

Basic Normalized Count Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a standard normalized count plot showing relative counts compared to control::

    from omero_screen_plots import count_plot
    import pandas as pd

    df = pd.read_csv("data.csv")
    fig, ax = count_plot(
        df=df,
        norm_control="control",
        conditions=["control", "cond01", "cond02", "cond03"],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        title="count plot basic",
        fig_size=(6, 4),
        save=True,
        file_format="svg"
    )

.. image:: ../_static/count_plot_basic.svg

Absolute Count Plot
~~~~~~~~~~~~~~~~~~~

Display raw cell counts without normalization to see actual numbers::

    from omero_screen_plots import count_plot, PlotType

    fig, ax = count_plot(
        df=df,
        norm_control="control",
        conditions=["control", "cond01", "cond02", "cond03"],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        plot_type=PlotType.ABSOLUTE,
        title="count plot absolute",
        fig_size=(6, 4),
        save=True,
        file_format="svg"
    )

.. image:: ../_static/count_plot_absolute.svg

Grouped Layout
~~~~~~~~~~~~~~

Group conditions for better visual organization and within-group statistical comparisons::

    fig, ax = count_plot(
        df=df,
        norm_control="control",
        conditions=["control", "cond01", "cond02", "cond03"],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        group_size=3,
        within_group_spacing=0.2,
        between_group_gap=0.8,
        title="count plot grouped",
        fig_size=(6, 4)
    )

.. image:: ../_static/count_plot_grouped.svg

Combined Subplot Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Create multi-panel figures comparing normalized and absolute counts::

    import matplotlib.pyplot as plt
    from omero_screen_plots.utils import save_fig

    fig, axes = plt.subplots(2, 1, figsize=(2, 4))
    fig.suptitle("count plot with axes", fontsize=8, weight="bold")

    # Normalized counts
    count_plot(
        df=df,
        norm_control="control",
        conditions=["control", "cond01", "cond02", "cond03"],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        axes=axes[0],
        plot_type=PlotType.NORMALISED,
        x_label=False
    )
    axes[0].set_title("MCF10A Norm Counts", fontsize=8, weight="bold")

    # Absolute counts
    count_plot(
        df=df,
        norm_control="control",
        conditions=["control", "cond01", "cond02", "cond03"],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        plot_type=PlotType.ABSOLUTE,
        axes=axes[1]
    )
    axes[1].set_title("MCF10A Abs. Counts", fontsize=8, weight="bold")

    save_fig(fig, "output/", "count_plot_with_axes", fig_extension="svg")

.. image:: ../_static/count_plot_with_axes.svg

Configuration Options
~~~~~~~~~~~~~~~~~~~~

The count_plot function supports extensive customization:

**Plot Types**:

- **NORMALISED** (default): Shows counts relative to control condition
- **ABSOLUTE**: Shows raw cell counts without normalization
- **Statistical analysis**: Automatic significance testing when ≥3 plates are present

**Data Processing**:

- **Cell counting**: Groups by (plate_id, condition, well) and counts cells per well
- **Mean calculation**: Computes mean count per condition across wells within each plate
- **Normalization**: Optional normalization relative to control condition

**Layout Options**:

- **Grouping**: Organize conditions into groups with custom spacing
- **Bar styling**: Customizable colors using the COLOR enum
- **Statistical marks**: Significance levels shown as *, **, *** annotations
- **Axes control**: Use existing matplotlib axes or create new figures

**Export & Styling**:

- **Figure size**: Control dimensions in cm or inches
- **File formats**: Save as SVG, PDF, or PNG with custom DPI
- **Labels**: Control axis labels, titles, and condition names
- **Spacing**: Customize within-group and between-group spacing
