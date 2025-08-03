CellCyclePlot API
=================

.. currentmodule:: omero_screen_plots.cellcycleplot

The cellcycleplot module analyzes and visualizes cell cycle distribution based on DNA content and S-phase markers.

Main Functions
--------------

.. autofunction:: cellcycle_plot

.. autofunction:: cellcycle_stacked

.. autofunction:: cellcycle_grouped

Helper Functions
----------------

.. autofunction:: cc_phase

.. autofunction:: prop_pivot

.. autofunction:: plot_triplicate_bars

.. autofunction:: draw_triplicate_boxes

Examples
--------

Basic cell cycle analysis::

    from omero_screen_plots.cellcycleplot import cellcycle_plot

    cellcycle_plot(
        data_path="sample_plate_data.csv",
        conditions=["DMSO", "Nutlin"],
        condition_col="condition",
        dapi_col="intensity_integrated_dapi_nucleus",
        edu_col="intensity_mean_edu_nucleus",
        output_path="output/"
    )

Stacked bar plot::

    from omero_screen_plots.cellcycleplot import cellcycle_stacked

    cellcycle_stacked(
        data_path="sample_plate_data.csv",
        conditions=["DMSO", "Nutlin", "Etop", "Noc"],
        condition_col="condition",
        dapi_col="intensity_integrated_dapi_nucleus",
        edu_col="intensity_mean_edu_nucleus",
        output_path="output/",
        normalise=True
    )

Grouped cell cycle analysis::

    from omero_screen_plots.cellcycleplot import cellcycle_grouped

    cellcycle_grouped(
        data_path="sample_plate_data.csv",
        conditions=["WT_Control", "WT_Drug", "KO_Control", "KO_Drug"],
        condition_col="condition",
        dapi_col="intensity_integrated_dapi_nucleus",
        edu_col="intensity_mean_edu_nucleus",
        output_path="output/"
    )
