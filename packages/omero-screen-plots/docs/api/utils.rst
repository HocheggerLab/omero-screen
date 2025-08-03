Utility Functions
=================

.. currentmodule:: omero_screen_plots.utils

This module provides utility functions used across the plotting package.

Functions
---------

.. autofunction:: save_fig

.. autofunction:: scale_data

.. autofunction:: selector_val_filter

.. autofunction:: get_repeat_points

.. autofunction:: show_repeat_points

.. autofunction:: select_datapoints

.. autofunction:: grouped_x_positions

Examples
--------

Scaling immunofluorescence data::

    from omero_screen_plots.utils import scale_data
    import pandas as pd

    # Scale intensity data to 16-bit range
    df = pd.read_csv("data.csv")
    df_scaled = scale_data(
        df,
        scale_col="intensity_mean_p21_nucleus",
        scale_min=1.0,  # 1st percentile
        scale_max=99.0  # 99th percentile
    )

Saving figures with consistent formatting::

    from omero_screen_plots.utils import save_fig
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # ... create your plot ...

    save_fig(
        fig=fig,
        path=Path("output"),
        fig_id="my_analysis",
        fig_extension="pdf",
        resolution=300
    )

Filtering data by condition::

    from omero_screen_plots.utils import selector_val_filter

    # Filter for specific experimental condition
    filtered_df = selector_val_filter(
        df=data,
        selector_col="cell_line",
        selector_val="HeLa",
        condition_col="treatment",
        conditions=["DMSO", "Nutlin"]
    )
