CountPlot API
=============

.. currentmodule:: omero_screen_plots.countplot

The countplot module provides functions for analyzing and visualizing cell counts per well or condition.

Main Functions
--------------

.. autofunction:: count_plot

.. autofunction:: norm_count

Enums
-----

.. autoclass:: PlotType
   :members:
   :undoc-members:

Examples
--------

Basic count plot::

    from omero_screen_plots.countplot import count_plot

    count_plot(
        data_path="sample_plate_data.csv",
        conditions=["DMSO", "Nutlin", "Etop", "Noc"],
        condition_col="condition",
        output_path="output/"
    )

Normalized count analysis::

    from omero_screen_plots.countplot import norm_count
    import pandas as pd

    df = pd.read_csv("sample_plate_data.csv")
    normalized_df = norm_count(
        df=df,
        condition_col="condition",
        control_condition="DMSO"
    )
