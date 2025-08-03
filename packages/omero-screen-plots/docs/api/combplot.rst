CombPlot API
============

.. currentmodule:: omero_screen_plots.combplot

The combplot module creates combined visualizations with marginal distributions for exploring relationships between features.

Main Functions
--------------

.. autofunction:: comb_plot

.. autofunction:: combplot_simple

.. autofunction:: histogram_plot

.. autofunction:: scatter_plot

.. autofunction:: scatter_plot_feature

Examples
--------

Combined plot with automatic styling::

    from omero_screen_plots.combplot import comb_plot

    comb_plot(
        data_path="sample_plate_data.csv",
        plot_type="scatter",
        conditions=["DMSO", "Nutlin"],
        condition_col="condition",
        x_feature="intensity_integrated_dapi_nucleus",
        y_feature="intensity_mean_edu_nucleus",
        output_path="output/"
    )

Simple combined plot::

    from omero_screen_plots.combplot import combplot_simple

    combplot_simple(
        data_path="sample_plate_data.csv",
        plot_type="histogram",
        conditions=["DMSO", "Nutlin", "Etop", "Noc"],
        condition_col="condition",
        x_feature="area_nucleus",
        output_path="output/",
        fig_id="nuclear_area_distribution"
    )

Histogram plot::

    from omero_screen_plots.combplot import histogram_plot
    import pandas as pd

    df = pd.read_csv("sample_plate_data.csv")
    fig, ax = histogram_plot(
        df=df,
        conditions=["DMSO", "Nutlin"],
        condition_col="condition",
        x_feature="intensity_mean_p21_nucleus",
        bins=50
    )

Scatter plot with feature relationship::

    from omero_screen_plots.combplot import scatter_plot_feature

    scatter_plot_feature(
        data_path="sample_plate_data.csv",
        conditions=["DMSO", "Nutlin"],
        condition_col="condition",
        x_feature="intensity_integrated_dapi_nucleus",
        y_feature="intensity_mean_edu_nucleus",
        output_path="output/"
    )
