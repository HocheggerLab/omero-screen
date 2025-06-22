"""Base classes for OMERO screen plots.

This module defines the base classes that all specific plot types inherit from,
providing common functionality for data validation, styling, and figure management.
"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .config import CM_TO_INCHES, CONFIG
from .stats import calculate_pvalues, get_significance_marker


class OmeroPlots(ABC):
    """Base class for all OMERO screen plots.

    This class provides common functionality including:
    - Data validation and preprocessing
    - Style and configuration management
    - Figure creation and management
    - Statistical analysis utilities
    - Save/show functionality

    All specific plot types should inherit from this class.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: List[str],
        condition_col: str = "condition",
        selector_col: Optional[str] = "cell_line",
        selector_val: Optional[str] = None,
        title: Optional[str] = None,
        colors: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the base plot.

        Args:
            data: DataFrame containing the plot data
            conditions: List of conditions to plot
            condition_col: Column name containing conditions
            selector_col: Column name for data selection (e.g., cell_line)
            selector_val: Value to select from selector_col
            title: Plot title
            colors: Custom color palette
            figsize: Figure size as (width, height) in inches
            ax: Optional axes to plot on (for integration into larger figures)
            **kwargs: Additional configuration options
        """
        self.data = self._validate_and_filter_data(
            data, conditions, condition_col, selector_col, selector_val
        )
        self.conditions = conditions
        self.condition_col = condition_col
        self.selector_col = selector_col
        self.selector_val = selector_val
        self.title = title
        self.colors = colors or CONFIG.colors
        self.figsize = self._cm_to_inches(figsize) or CONFIG.get_figure_size(
            self.plot_type
        )
        self.config = CONFIG

        # Figure management
        self.ax = ax
        self.fig: Optional[Figure] = None
        self._owns_figure = ax is None  # Track if we created the figure
        self._single_ax_mode = not self._owns_figure

        # Additional configuration
        self.plot_config = kwargs

    @property
    @abstractmethod
    def plot_type(self) -> str:
        """Return the plot type identifier for configuration."""

    def _cm_to_inches(
        self, figsize_cm: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Convert figsize from cm to inches."""
        if figsize_cm is None:
            return None
        return (figsize_cm[0] * CM_TO_INCHES, figsize_cm[1] * CM_TO_INCHES)

    def _validate_and_filter_data(
        self,
        data: pd.DataFrame,
        conditions: List[str],
        condition_col: str,
        selector_col: Optional[str],
        selector_val: Optional[str],
    ) -> pd.DataFrame:
        """Validate and filter input data."""
        if data.empty:
            raise ValueError("Input data is empty")

        if condition_col not in data.columns:
            raise ValueError(
                f"Condition column '{condition_col}' not found in data"
            )

        # Filter by conditions
        filtered_data = data[data[condition_col].isin(conditions)].copy()

        if filtered_data.empty:
            raise ValueError(f"No data found for conditions: {conditions}")

        # Apply selector filter if specified
        if selector_col and selector_val:
            if selector_col not in filtered_data.columns:
                raise ValueError(
                    f"Selector column '{selector_col}' not found in data"
                )
            filtered_data = filtered_data[
                filtered_data[selector_col] == selector_val
            ].copy()

            if filtered_data.empty:
                raise ValueError(
                    f"No data found for {selector_col}='{selector_val}'"
                )
        elif selector_col and selector_val is None:
            raise ValueError(
                "selector_val must be provided when selector_col is specified"
            )

        return filtered_data

    def _setup_figure(self) -> None:
        """Setup figure and axes if not provided."""
        if self.ax is None:
            self.fig = plt.figure(figsize=self.figsize)
            # Don't create an Axes object here, let the subclass do it
        else:
            self.fig = cast(Figure, self.ax.figure)

    def _apply_title(self) -> None:
        """Apply title to the plot if specified."""
        if self.title and self.ax:
            self.ax.set_title(self.title, fontsize=6, weight="regular")

    def _apply_figure_title(self, fig: Optional[Figure] = None) -> None:
        """Apply title as figure suptitle for multi-plot layouts.

        Args:
            fig: Figure to apply title to. If None, uses self.fig
        """
        if self.title:
            if figure := fig or self.fig:
                y_pos = 1.16 if "\n" in self.title else 1.08
                figure.suptitle(
                    self.title, fontsize=7, weight="regular", y=y_pos, x=0.5
                )

    def calculate_statistics(self, column: str) -> List[float]:
        """Calculate p-values for statistical significance."""
        try:
            return calculate_pvalues(
                self.data, self.conditions, self.condition_col, column
            )
        except Exception as e:
            warnings.warn(f"Could not calculate statistics for {column}: {e}")
            return []

    def add_significance_markers(self, column: str, y_max: float) -> None:
        """Add significance markers to the plot."""
        if not self.ax:
            return

        pvalues = self.calculate_statistics(column)
        for i, condition in enumerate(self.conditions[1:], start=1):
            p_value = pvalues[i - 1]
            significance = get_significance_marker(p_value)

            # Position the significance marker
            x_pos = i
            y_pos = y_max * 1.05

            self.ax.text(
                x_pos,
                y_pos,
                significance,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    @abstractmethod
    def generate(self) -> Figure:
        """Generate the plot. Must be implemented by subclasses."""

    def save(
        self,
        path: Union[str, Path],
        filename: Optional[str] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the figure to file.

        Args:
            path: Directory path or full file path
            filename: Filename (if path is directory)
            tight_layout: Whether to apply tight layout
            **kwargs: Additional parameters for savefig
        """
        if not self.fig:
            raise ValueError("No figure to save. Call generate() first.")

        path = Path(path)

        if filename:
            save_path = path / filename
        elif path.suffix:
            save_path = path
        else:
            # Default filename based on plot type
            default_name = f"{self.plot_type}_plot.{CONFIG.DEFAULT_FORMAT}"
            save_path = path / default_name

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply tight layout if requested and we own the figure
        if tight_layout and self._owns_figure:
            self.fig.tight_layout()

        # Get save parameters
        save_params = CONFIG.get_save_params(**kwargs)

        print(f"Saving {self.plot_type} plot to {save_path}")
        self.fig.savefig(save_path, **save_params)

    def show(self) -> None:
        """Display the figure."""
        if not self.fig:
            raise ValueError("No figure to show. Call generate() first.")
        plt.show()

    def close(self) -> None:
        """Close the figure if we own it."""
        if self._owns_figure and self.fig:
            plt.close(self.fig)


class OmeroCombPlots:
    """Handle multiple plots in grid layouts.

    This class manages the creation of composite figures containing
    multiple OMERO plots arranged in a grid layout.
    """

    def __init__(
        self,
        plots: List[OmeroPlots],
        layout: Optional[Tuple[int, int]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        titles: Optional[List[Optional[str]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize composite plot manager.

        Args:
            plots: List of OmeroPlots instances to combine
            layout: Grid layout as (rows, cols). If None, auto-calculated
            figsize: Overall figure size. If None, auto-calculated
            titles: Individual subplot titles
            **kwargs: Additional configuration for subplots
        """
        self.plots = plots
        self.layout = layout or self._calculate_layout(len(plots))
        self.figsize = figsize or self._calculate_figsize()
        self.titles = titles or [None] * len(plots)
        self.subplot_config = kwargs

        self.fig: Optional[Figure] = None
        self.axes: List[Axes] = []

    def _calculate_layout(self, n_plots: int) -> Tuple[int, int]:
        """Calculate optimal grid layout for n plots."""
        if n_plots == 1:
            return (1, 1)
        elif n_plots == 2:
            return (1, 2)
        elif n_plots <= 4:
            return (2, 2)
        elif n_plots <= 6:
            return (2, 3)
        else:
            # For larger numbers, try to keep roughly square
            cols = int(n_plots**0.5) + 1
            rows = (n_plots + cols - 1) // cols
            return (rows, cols)

    def _calculate_figsize(self) -> Tuple[float, float]:
        """Calculate figure size based on layout and individual plot sizes."""
        rows, cols = self.layout

        # Use the first plot's dimensions as reference
        if self.plots:
            plot_width, plot_height = self.plots[0].figsize
        else:
            plot_width, plot_height = CONFIG.get_figure_size()

        # Add some padding between subplots
        total_width = cols * plot_width * 1.2
        total_height = rows * plot_height * 1.2

        return (total_width, total_height)

    def generate_grid(self) -> Figure:
        """Create subplot grid with multiple plots."""
        rows, cols = self.layout

        # Create figure and subplots
        self.fig, axs = plt.subplots(
            rows, cols, figsize=self.figsize, **self.subplot_config
        )

        # Ensure axes is a flat list
        axes: List[Axes]
        if isinstance(axs, np.ndarray):
            axes = axs.flatten().tolist()
        else:
            axes = [axs]

        self.axes = axes

        # Generate each plot on its assigned axes
        for i, plot in enumerate(self.plots):
            if i < len(axes):
                # Set the axes for the plot and generate
                plot.ax = axes[i]
                plot.fig = self.fig
                plot._owns_figure = False

                # Apply individual title if provided
                if self.titles[i]:
                    plot.title = self.titles[i]

                plot.generate()

        # Hide unused subplots
        for i in range(len(self.plots), len(axes)):
            axes[i].set_visible(False)

        assert self.fig is not None
        return self.fig

    def save_composite(
        self,
        path: Union[str, Path],
        filename: Optional[str] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the composite figure."""
        if not self.fig:
            raise ValueError("No figure to save. Call generate_grid() first.")

        path = Path(path)

        if filename:
            save_path = path / filename
        elif path.suffix:
            save_path = path
        else:
            save_path = path / f"composite_plot.{CONFIG.DEFAULT_FORMAT}"

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if tight_layout:
            self.fig.tight_layout()

        save_params = CONFIG.get_save_params(**kwargs)

        print(f"Saving composite plot to {save_path}")
        self.fig.savefig(save_path, **save_params)

    def show(self) -> None:
        """Display the composite figure."""
        if not self.fig:
            raise ValueError("No figure to show. Call generate_grid() first.")
        plt.show()

    def close(self) -> None:
        """Close the composite figure."""
        if self.fig:
            plt.close(self.fig)
