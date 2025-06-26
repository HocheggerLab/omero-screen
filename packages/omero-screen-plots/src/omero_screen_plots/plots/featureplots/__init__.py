"""Feature plots module.

This module provides various plot types for visualizing feature data:
- Standard feature plots (box/violin plots per condition)
- Grouped feature plots (conditions organized in visual groups)
- Threshold-based stacked bar plots

The module follows a modular architecture where each plot type is implemented
as a class inheriting from base classes that provide common functionality.
"""

from .base import BaseFeaturePlot
from .grouped import GroupedFeaturePlot, grouped_feature_plot
from .stacked import (
    StackedFeaturePlot,
    grouped_stacked_threshold_barplot,
)
from .standard import StandardFeaturePlot, feature_plot

__all__ = [
    # Base classes
    "BaseFeaturePlot",
    # Plot classes
    "StandardFeaturePlot",
    "GroupedFeaturePlot",
    "StackedFeaturePlot",
    # User-facing functions
    "feature_plot",
    "grouped_feature_plot",
    "grouped_stacked_threshold_barplot",
    # Legacy aliases for backward compatibility
    "FeaturePlot",
    "GroupedStackedThresholdBarplot",
]

# Backward compatibility aliases
FeaturePlot = StandardFeaturePlot
GroupedStackedThresholdBarplot = StackedFeaturePlot
