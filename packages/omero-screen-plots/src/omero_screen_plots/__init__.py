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
from .featureplot import feature_plot as legacy_feature_plot
from .featureplot import grouped_feature_plot as legacy_grouped_feature_plot
from .featureplot import (
    grouped_stacked_threshold_barplot as legacy_grouped_stacked_threshold_barplot,
)
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

# Import new modular feature plot functions
from .plots.featureplots import (
    # Classes for advanced usage
    FeaturePlot,
    GroupedFeaturePlot,
    GroupedStackedThresholdBarplot,
    feature_plot,
    grouped_feature_plot,
    grouped_stacked_threshold_barplot,
)

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
    # Feature plot classes
    "FeaturePlot",
    "GroupedFeaturePlot",
    "GroupedStackedThresholdBarplot",
    # User-facing feature plot functions
    "feature_plot",
    "grouped_feature_plot",
    "grouped_stacked_threshold_barplot",
    # Legacy functions
    "cellcycle_plot",
    "legacy_feature_plot",
    "legacy_grouped_feature_plot",
    "legacy_grouped_stacked_threshold_barplot",
    "histogram_plot",
    "scatter_plot",
    "count_plot",
]
