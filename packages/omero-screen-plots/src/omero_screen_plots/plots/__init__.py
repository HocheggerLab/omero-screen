"""Plot modules for OMERO screen data.

This package contains specific plot implementations that inherit from
the base classes defined in the base module.
"""

# Import from cellcycle submodule
from .cellcycle import (
    BaseCellCyclePlot,
    CellCycleGroupedPlot,
    CellCyclePlot,
    CellCycleStackedPlot,
)

# Legacy import for backward compatibility
from .cellcycle.standard import CellCyclePlot as CellCyclePlotLegacy

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
    SimpleCombPlot,
)

__all__ = [
    # Cell cycle plots
    "BaseCellCyclePlot",
    "CellCyclePlot",
    "CellCycleStackedPlot",
    "CellCycleGroupedPlot",
    "CellCyclePlotLegacy",  # For backward compatibility
    # Combined plots
    "BaseCombPlot",
    "BaseHistogramPlot",
    "BaseScatterPlot",
    "BaseCellCycleScatter",
    "BaseFeatureScatter",
    "HistogramPlot",
    "CellCycleScatterPlot",
    "FeatureScatterPlot",
    "SimpleCombPlot",
    "FullCombPlot",
]
