Normalise API
=============

.. currentmodule:: omero_screen_plots.normalise

The normalise module provides data normalization utilities for immunofluorescence data.

Main Functions
--------------

.. autofunction:: normalize_by_mode

.. autofunction:: normalize_and_plot

.. autofunction:: plot_normalization_result

.. autofunction:: set_threshold_categories

Helper Functions
----------------

.. autofunction:: find_intensity_mode

Examples
--------

Normalize by mode::

    from omero_screen_plots.normalise import normalize_by_mode
    import pandas as pd

    df = pd.read_csv("sample_plate_data.csv")
    normalized_df = normalize_by_mode(
        df=df,
        feature="intensity_mean_p21_nucleus",
        condition_col="condition",
        control_condition="DMSO"
    )

Normalize and plot results::

    from omero_screen_plots.normalise import normalize_and_plot

    normalize_and_plot(
        data_path="sample_plate_data.csv",
        feature="intensity_mean_p21_nucleus",
        condition_col="condition",
        control_condition="DMSO",
        output_path="output/"
    )

Set threshold categories::

    from omero_screen_plots.normalise import set_threshold_categories

    df_with_categories = set_threshold_categories(
        df=df,
        feature="intensity_mean_p21_nucleus",
        low_threshold=500,
        high_threshold=2000
    )
