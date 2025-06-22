"""Module for handling OMERO screen plots.

This package provides a shared functionality for plotting OMERO screen data.

The omero_screen_plots package provides plotting and analysis tools for OMERO screen data, including cell cycle, classification, count, feature, and synergy plots.

"""

__version__ = "0.1.2"

from .base import OmeroCombPlots, OmeroPlots

# For backward compatibility, also expose the old function-based interface
from .cellcycleplot import cellcycle_plot
from .colors import COLOR
from .combplot import histogram_plot, scatter_plot
from .countplot import count_plot
from .featureplot import feature_plot
from .plots import (
    BaseCellCyclePlot,
    # Combined plot classes
    BaseCombPlot,
    CellCycleGroupedPlot,
    CellCyclePlot,
    CellCycleScatterPlot,
    CellCycleStackedPlot,
    FeatureScatterPlot,
    FullCombPlot,
    HistogramPlot,
    SimpleCombPlot,
)

# Import user-facing plot functions
from .plots.cellcycle import (
    cellcycle_grouped_plot,
    cellcycle_stacked_plot,
    cellcycle_standard_plot,
)
from .plots.combplots import (
    cellcycle_scatter_plot,
    feature_scatter_plot,
    full_combplot,
    simple_combplot,
)
from .plots.combplots import histogram_plot as new_histogram_plot

__all__ = [
    "COLOR",
    "OmeroPlots",
    "OmeroCombPlots",
    # Cell cycle plot classes
    "BaseCellCyclePlot",
    "CellCyclePlot",
    "CellCycleStackedPlot",
    "CellCycleGroupedPlot",
    # Combined plot classes
    "BaseCombPlot",
    "HistogramPlot",
    "CellCycleScatterPlot",
    "FeatureScatterPlot",
    "SimpleCombPlot",
    "FullCombPlot",
    # User-facing cell cycle plot functions
    "cellcycle_standard_plot",
    "cellcycle_stacked_plot",
    "cellcycle_grouped_plot",
    # User-facing combined plot functions
    "new_histogram_plot",
    "cellcycle_scatter_plot",
    "feature_scatter_plot",
    "simple_combplot",
    "full_combplot",
    # Legacy functions
    "cellcycle_plot",
    "feature_plot",
    "histogram_plot",
    "scatter_plot",
    "count_plot",
]
