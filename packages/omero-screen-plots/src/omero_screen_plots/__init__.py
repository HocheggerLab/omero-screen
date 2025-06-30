"""Module for handling OMERO screen plots.

This package provides a shared functionality for plotting OMERO screen data.

The omero_screen_plots package provides plotting and analysis tools for OMERO screen data, including cell cycle, classification, count, feature, and synergy plots.

"""

__version__ = "0.1.2"


# Import user-facing plot functions

from omero_screen_plots.cellcycleplot import (
    cellcycle_grouped,
    cellcycle_plot,
    cellcycle_stacked,
)

__all__ = [
    "cellcycle_plot",
    "cellcycle_stacked",
    "cellcycle_grouped",
]
