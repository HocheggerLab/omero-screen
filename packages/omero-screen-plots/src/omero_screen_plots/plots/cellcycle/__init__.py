"""Cell cycle plotting submodule.

This submodule contains different cell cycle plot implementations:
- Standard: 2x2 subplot grid showing each phase separately
- Stacked: Stacked bar plot showing phase proportions
- Grouped: Grouped stacked bars with individual replicates
"""

from .base import BaseCellCyclePlot
from .grouped import GroupedCellCyclePlot, cellcycle_grouped_plot
from .stacked import StackedCellCyclePlot, cellcycle_stacked_plot
from .standard import StandardCellCyclePlot, cellcycle_standard_plot

__all__ = [
    # Base classes
    "BaseCellCyclePlot",
    # Plot classes
    "StandardCellCyclePlot",
    "StackedCellCyclePlot",
    "GroupedCellCyclePlot",
    # User-facing functions
    "cellcycle_standard_plot",
    "cellcycle_stacked_plot",
    "cellcycle_grouped_plot",
    # Legacy aliases for backward compatibility
    "CellCyclePlot",
    "CellCycleStackedPlot",
    "CellCycleGroupedPlot",
]

# Backward compatibility aliases
CellCyclePlot = StandardCellCyclePlot
CellCycleStackedPlot = StackedCellCyclePlot
CellCycleGroupedPlot = GroupedCellCyclePlot
