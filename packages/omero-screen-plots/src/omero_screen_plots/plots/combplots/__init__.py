"""Combined plots (combplots) submodule.

This module provides comprehensive combined plotting functionality for OMERO screen data,
including individual plot components and complete multi-panel figures.

Available plot types:
- Histogram plots: DAPI intensity distributions
- Cell cycle scatter plots: DAPI vs EdU with phase annotations
- Feature scatter plots: DAPI vs any feature with threshold coloring
- Simple combined plots: 2-row layout (histogram + scatter + bar)
- Full combined plots: 3-row layout (histogram + cell cycle + feature)

Classes:
    BaseCombPlot: Base class for all combined plots
    BaseHistogramPlot: Base class for histogram components
    BaseScatterPlot: Base class for scatter plot components
    BaseCellCycleScatter: Base class for cell cycle scatter plots
    BaseFeatureScatter: Base class for feature scatter plots

    HistogramPlot: Individual histogram plot
    CellCycleScatterPlot: Individual cell cycle scatter plot
    FeatureScatterPlot: Individual feature scatter plot
    SimpleCombPlot: 2-row combined plot
    FullCombPlot: 3-row combined plot

User-facing functions:
    histogram_plot(): Create DAPI histogram plots
    cellcycle_scatter_plot(): Create cell cycle scatter plots
    feature_scatter_plot(): Create feature scatter plots
    simple_combplot(): Create 2-row combined plots
    full_combplot(): Create 3-row combined plots
"""

# Import base classes
from .base import (
    BaseCellCycleScatter,
    BaseCombPlot,
    BaseFeatureScatter,
    BaseHistogramPlot,
    BaseScatterPlot,
)
from .cellcycle_scatter import CellCycleScatterPlot, cellcycle_scatter_plot
from .feature_scatter import FeatureScatterPlot, feature_scatter_plot
from .full import FullCombPlot, full_combplot

# Import individual plot classes and functions
from .histogram import HistogramPlot, histogram_plot

# Import combined plot classes and functions
from .simple import SimpleCombPlot, simple_combplot

__all__ = [
    # Base classes
    "BaseCombPlot",
    "BaseHistogramPlot",
    "BaseScatterPlot",
    "BaseCellCycleScatter",
    "BaseFeatureScatter",
    # Individual plot classes
    "HistogramPlot",
    "CellCycleScatterPlot",
    "FeatureScatterPlot",
    # Combined plot classes
    "SimpleCombPlot",
    "FullCombPlot",
    # User-facing functions
    "histogram_plot",
    "cellcycle_scatter_plot",
    "feature_scatter_plot",
    "simple_combplot",
    "full_combplot",
]
