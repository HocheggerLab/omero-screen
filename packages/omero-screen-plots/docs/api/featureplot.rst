FeaturePlot API
===============

.. currentmodule:: omero_screen_plots.featureplot

The featureplot module provides flexible visualization options for comparing quantitative features across experimental conditions.

Main Functions
--------------

.. autofunction:: feature_plot

.. autofunction:: feature_plot_simple

.. autofunction:: feature_threshold_plot

.. autofunction:: draw_violin_or_box

Examples
--------

Basic box plot::

    from omero_screen_plots.featureplot import feature_plot
    import pandas as pd

    df = pd.read_csv("data.csv")
    fig, ax = feature_plot(
        df=df,
        y_feature="intensity_mean_p21_nucleus",
        conditions=["Control", "Treatment"],
        condition_col="condition",
        plot_type="box"
    )

Simple feature plot with automatic styling::

    from omero_screen_plots.featureplot import feature_plot_simple

    feature_plot_simple(
        data_path="data.csv",
        y_feature="area_nucleus",
        conditions=["DMSO", "Drug1", "Drug2"],
        condition_col="condition",
        output_path="output/",
        fig_id="nuclear_area"
    )

Threshold-based analysis::

    from omero_screen_plots.featureplot import feature_threshold_plot

    feature_threshold_plot(
        data_path="data.csv",
        feature="intensity_mean_p21_nucleus",
        threshold_value=1000,
        conditions=["Control", "Treatment"],
        condition_col="condition",
        output_path="output/"
    )
