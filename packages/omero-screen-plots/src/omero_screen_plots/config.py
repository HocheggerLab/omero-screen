"""Configuration module for OMERO screen plots.

This module centralizes all styling, constants, and configuration
for consistent plot generation across all plot types.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# Figure dimensions in cm converted to inches
CM_TO_INCHES = 1 / 2.54


class PlotConfig:
    """Central configuration for all OMERO plots."""

    # Default figure dimensions (converted from cm to inches)
    DEFAULT_WIDTH = 9 * CM_TO_INCHES  # 9 cm
    DEFAULT_HEIGHT = 6 * CM_TO_INCHES  # 6 cm

    # Feature plot specific dimensions
    FEATURE_WIDTH = 9 * CM_TO_INCHES  # 9 cm
    FEATURE_HEIGHT = 4 * CM_TO_INCHES  # 4 cm

    # Cell cycle plot dimensions
    CELLCYCLE_WIDTH = 9 * CM_TO_INCHES  # 9 cm
    CELLCYCLE_HEIGHT = 6 * CM_TO_INCHES  # 6 cm

    # Default save parameters
    DEFAULT_DPI = 300
    DEFAULT_FORMAT = "pdf"
    DEFAULT_FACECOLOR = "white"
    DEFAULT_EDGECOLOR = "white"

    # Statistical significance thresholds
    P_VALUE_THRESHOLDS = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: "ns"}

    def __init__(self) -> None:
        """Initialize plot configuration with style setup."""
        self._setup_style()
        self._setup_colors()

    def _setup_style(self) -> None:
        """Setup matplotlib style."""
        current_dir = Path(__file__).parent
        style_path = (current_dir / "../../hhlab_style01.mplstyle").resolve()

        if style_path.exists():
            plt.style.use(style_path)
        else:
            # Fallback to default style if custom style not found
            plt.style.use("default")

        # Disable pandas chained assignment warnings
        pd.options.mode.chained_assignment = None

    def _setup_colors(self) -> None:
        """Setup color palette from matplotlib style."""
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        self.colors = prop_cycle.by_key()["color"]

    def get_figure_size(
        self, plot_type: str = "default"
    ) -> tuple[float, float]:
        """Get figure size for specific plot type."""
        size_map = {
            "default": (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT),
            "feature": (self.FEATURE_WIDTH, self.FEATURE_HEIGHT),
            "cellcycle": (self.CELLCYCLE_WIDTH, self.CELLCYCLE_HEIGHT),
        }
        return size_map.get(plot_type, size_map["default"])

    def get_save_params(self, **kwargs: Any) -> dict[str, Any]:
        """Get parameters for saving figures."""
        params = {
            "dpi": self.DEFAULT_DPI,
            "format": self.DEFAULT_FORMAT,
            "facecolor": self.DEFAULT_FACECOLOR,
            "edgecolor": self.DEFAULT_EDGECOLOR,
        }
        params.update(kwargs)
        return params


# Global configuration instance
CONFIG: PlotConfig = PlotConfig()
