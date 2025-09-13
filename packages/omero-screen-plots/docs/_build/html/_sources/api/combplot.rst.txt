Combined Plot API
=================

.. currentmodule:: omero_screen_plots

The combined plot creates comprehensive visualizations with marginal distributions for exploring relationships between features with integrated histogram and scatter plot components.

Main Functions
--------------

.. autofunction:: comb_plot

Examples
--------

Basic Combined Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Create a combined scatter plot with marginal histograms::

    from omero_screen_plots import comb_plot
    import pandas as pd

    df = pd.read_csv("data.csv")
    fig = comb_plot(
        df=df,
        x_feature="area_cell",
        y_feature="intensity_mean_p21_nucleus",
        conditions=['control', 'cond01', 'cond02', 'cond03'],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        plot_type="scatter",
        title="Combined Feature Analysis",
        save=True,
        file_format="svg"
    )

Cell Cycle Combination
~~~~~~~~~~~~~~~~~~~~~~

Combine cell cycle scatter plot with DNA content and EdU marginal distributions::

    fig = comb_plot(
        df=df,
        x_feature="integrated_int_DAPI_norm",
        y_feature="intensity_mean_EdU_nucleus_norm",
        conditions=['control', 'treatment'],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        plot_type="scatter",
        title="Cell Cycle Combined Analysis"
    )
