"""Module for handling OMERO screen plots.

This package provides a shared functionality for plotting OMERO screen data.

The omero_screen_plots package provides plotting and analysis tools for OMERO screen data, including cell cycle, classification, count, feature, and synergy plots.

"""

__version__ = "0.1.2"


# Import user-facing plot functions
from omero_screen_plots.cellcycleplot_api import (
    cellcycle_plot,
    cellcycle_stacked,
)
from omero_screen_plots.classificationplot_api import classification_plot
from omero_screen_plots.combplot import comb_plot
from omero_screen_plots.combplot_api import (
    combplot_cellcycle,
    combplot_feature,
)
from omero_screen_plots.countplot_api import count_plot
from omero_screen_plots.countplot_factory import PlotType
from omero_screen_plots.featureplot_api import feature_norm_plot, feature_plot
from omero_screen_plots.histogramplot_api import histogram_plot
from omero_screen_plots.scatterplot_api import scatter_plot
from omero_screen_plots.utils import (
    save_fig,
)

__all__ = [
    "cellcycle_plot",
    "cellcycle_stacked",
    "classification_plot",
    "comb_plot",
    "combplot_cellcycle",
    "combplot_feature",
    "count_plot",
    "feature_plot",
    "feature_norm_plot",
    "histogram_plot",
    "PlotType",
    "save_fig",
    "scatter_plot",
]
