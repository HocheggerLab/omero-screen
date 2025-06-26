"""Plot modules for OMERO screen data.

This package contains specific plot implementations that inherit from
the base classes defined in the base module.
"""

# Import from cellcycle submodule
from .cellcycle import (
    BaseCellCyclePlot,
    CellCycleGroupedPlot,
    # Legacy aliases
    CellCyclePlot,
    CellCycleStackedPlot,
    GroupedCellCyclePlot,
    StackedCellCyclePlot,
    StandardCellCyclePlot,
)

# Import from combplots submodule
from .combplots import (
    BaseCellCycleScatter,
    BaseCombPlot,
    BaseFeatureScatter,
    BaseHistogramPlot,
    BaseScatterPlot,
    CellCycleScatterPlot,
    FeatureScatterPlot,
    FullCombPlot,
    HistogramPlot,
    # Legacy aliases
    SimpleCombPlot,
    StandardCombPlot,
)

__all__ = [
    # Cell cycle plots
    "BaseCellCyclePlot",
    "StandardCellCyclePlot",
    "StackedCellCyclePlot",
    "GroupedCellCyclePlot",
    # Legacy aliases
    "CellCyclePlot",
    "CellCycleStackedPlot",
    "CellCycleGroupedPlot",
    # Combined plots
    "BaseCombPlot",
    "BaseHistogramPlot",
    "BaseScatterPlot",
    "BaseCellCycleScatter",
    "BaseFeatureScatter",
    "HistogramPlot",
    "CellCycleScatterPlot",
    "FeatureScatterPlot",
    "StandardCombPlot",
    "FullCombPlot",
    # Legacy aliases
    "SimpleCombPlot",
]
