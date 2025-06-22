"""Base classes for combined plots (combplots).

This module provides the base classes that all combplot types inherit from,
providing common functionality for data validation, styling, and subplot management.
"""

import warnings
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ...base import OmeroPlots


class BaseCombPlot(OmeroPlots):
    """Base class for all combined plot types.

    This class extends OmeroPlots and provides common functionality for:
    - Data normalization and preprocessing
    - Subplot grid management
    - Color and styling consistency
    - Cell sampling for performance
    - Common plot elements (axes, labels, etc.)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        cell_number: Optional[int] = None,
        dapi_col: str = "integrated_int_DAPI_norm",
        edu_col: str = "intensity_mean_EdU_nucleus_norm",
        **kwargs: Any,
    ) -> None:
        """Initialize base combined plot.

        Args:
            data: DataFrame containing the plot data
            conditions: List of conditions to plot
            cell_number: Optional cell sampling limit per condition
            dapi_col: Column name for DAPI intensity (DNA content)
            edu_col: Column name for EdU intensity
            **kwargs: Additional arguments passed to OmeroPlots
        """
        super().__init__(data, conditions, **kwargs)

        self.cell_number = cell_number
        self.dapi_col = dapi_col
        self.edu_col = edu_col

        # Validate required columns
        self._validate_combplot_columns()

        # Calculate data ranges for consistent scaling
        self._calculate_data_ranges()

    def _validate_combplot_columns(self) -> None:
        """Validate that required columns exist in the data."""
        required_cols = [self.dapi_col, self.edu_col, "cell_cycle"]
        missing_cols = [
            col for col in required_cols if col not in self.data.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _calculate_data_ranges(self) -> None:
        """Calculate consistent data ranges for all subplots."""
        # DAPI (DNA content) range
        self.dapi_min = 1.0
        self.dapi_max = 16.0

        # EdU range
        self.edu_min = self.data[self.edu_col].quantile(0.01) * 0.8
        self.edu_max = self.data[self.edu_col].quantile(0.99) * 1.5

    def get_condition_data(self, condition: str) -> pd.DataFrame:
        """Get data for a specific condition with optional sampling.

        Args:
            condition: The condition to filter for

        Returns:
            DataFrame filtered to the condition, optionally sampled
        """
        condition_data = self.data[
            self.data[self.condition_col] == condition
        ].copy()

        if self.cell_number and len(condition_data) >= self.cell_number:
            condition_data = condition_data.sample(
                n=self.cell_number, random_state=42
            )

        return condition_data

    def setup_subplot_grid(
        self,
        n_rows: int,
        n_cols: int,
        height_ratios: Optional[list[float]] = None,
        width_ratios: Optional[list[float]] = None,
        **grid_kwargs: Any,
    ) -> tuple[Figure, GridSpec]:
        """Setup subplot grid layout.

        Args:
            n_rows: Number of rows in the grid
            n_cols: Number of columns in the grid
            height_ratios: Relative heights of rows
            width_ratios: Relative widths of columns
            **grid_kwargs: Additional GridSpec parameters

        Returns:
            Tuple of (figure, gridspec)
        """
        self._setup_figure()

        # Default grid parameters
        grid_params = {
            "height_ratios": height_ratios,
            "width_ratios": width_ratios,
            "hspace": 0.05,
            "wspace": 0.25,
        }
        grid_params |= grid_kwargs

        # Remove None values
        grid_params = {k: v for k, v in grid_params.items() if v is not None}

        gs = GridSpec(n_rows, n_cols, **grid_params)  # type: ignore[arg-type]
        return self.fig, gs  # type: ignore

    def add_common_x_label(
        self,
        fig: Figure,
        label: str = "norm. DNA content",
        force_label: bool = True,
        row_position: str = "bottom",
        target_axes: Optional[list[Axes]] = None,
    ) -> None:
        """Add common x-axis label to the figure.

        Args:
            fig: The matplotlib figure
            label: Label text for x-axis
            force_label: Whether to force add the label (for standalone plots)
            row_position: Position relative to subplot rows ("bottom", "middle")
            target_axes: Specific axes to center the label under. If None, auto-detect.
        """
        if not force_label:
            return
        if target_axes:
            # Find the bounds of the target axes
            x_positions = []
            y_positions = []

            for ax in target_axes:
                if ax.get_visible():
                    bbox = ax.get_position()
                    x_positions.extend([bbox.x0, bbox.x0 + bbox.width])
                    y_positions.append(bbox.y0)

            if x_positions and y_positions:
                # Center horizontally across the target axes
                x_center = (min(x_positions) + max(x_positions)) / 2

                # Position below the lowest axis
                lowest_y = min(y_positions)

                # Check if any axis has visible x-ticks
                has_xticks = any(
                    ax.xaxis.get_visible()
                    for ax in target_axes
                    if ax.get_visible()
                )
                offset = 0.12 if has_xticks else 0.08
                y_position = lowest_y - offset

                fig.text(x_center, y_position, label, ha="center", fontsize=6)
            else:
                # Fallback
                fig.text(0.5, -0.07, label, ha="center", fontsize=6)
        else:
            # Auto-detect mode - use simple approach
            # Get all axes in the figure
            all_axes = fig.get_axes()

            # Find axes that are likely scatter plots (taller than histograms)
            scatter_axes = []
            for ax in all_axes:
                if ax.get_visible():
                    bbox = ax.get_position()
                    # Check if this is likely a scatter plot (has reasonable height)
                    if (
                        bbox.height > 0.2
                    ):  # Scatter plots are taller than histograms
                        scatter_axes.append(ax)

            if scatter_axes:
                # Use these axes for positioning
                self.add_common_x_label(
                    fig, label, force_label, row_position, scatter_axes
                )
            else:
                # Fallback to default positioning
                fig.text(0.5, -0.07, label, ha="center", fontsize=6)

    def _should_show_x_label(
        self, n_conditions: int, is_part_of_combplot: bool = False
    ) -> Union[bool, str]:
        """Determine whether to show x-axis label based on context.

        Args:
            n_conditions: Number of conditions being plotted
            is_part_of_combplot: Whether this is part of a larger combined plot

        Returns:
            False if no label, "individual" for single plot, "common" for multiple plots
        """
        if is_part_of_combplot:
            # No x-label when part of a larger combplot
            return False
        elif n_conditions == 1:
            # Single plot: show x-label on the individual axis
            return "individual"
        else:
            # Multiple plots: show common x-label below all subplots
            return "common"

    def save(
        self,
        path: Union[str, Path],
        filename: Optional[str] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the combined plot.

        Args:
            path: Path to save location
            filename: Optional filename. If None, generates descriptive name
            tight_layout: Whether to apply tight layout
            **kwargs: Additional save parameters
        """
        if filename is None:
            selector_part = (
                f"_{self.selector_val}" if self.selector_val else ""
            )
            filename = f"{self.plot_type}{selector_part}.png"

        # Default to PNG for combplots (better for complex scatter plots)
        save_params = {"format": "png", "dpi": 300, **kwargs}

        # Combplots manage their own layout, so default to False unless overridden
        if "tight_layout" not in save_params:
            save_params["tight_layout"] = False

        super().save(path, filename, **save_params)

    def create_histogram(
        self,
        ax: Axes,
        data: pd.DataFrame,
        condition_index: int,
        show_individual_xlabel: bool = False,
    ) -> None:
        """Create histogram of DAPI intensity.

        Args:
            ax: Matplotlib axes to plot on
            data: Data for this condition
            condition_index: Index of condition (for styling)
            show_individual_xlabel: Whether to show x-label on this individual axis
        """
        import seaborn as sns
        from matplotlib import ticker

        # Use seaborn for histogram (exactly like original working code)
        sns.histplot(data=data, x=self.dapi_col, ax=ax, color=self.colors[-1])

        # Configure axes (context-aware x-axis labeling)
        ax.set_xscale("log", base=2)
        ax.set_xlim(1, 16)  # Use original fixed limits

        # Handle x-axis label and visibility based on context
        if show_individual_xlabel:
            # Single plot: show x-label and ticks on this axis
            ax.set_xlabel("norm. DNA content", fontsize=6)
            ax.xaxis.set_visible(True)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: str(int(x)))
            )
            ax.set_xticks([2, 4, 8])
        else:
            # Multiple plots or part of combplot: no individual x-label
            ax.set_xlabel("")
            ax.xaxis.set_visible(False)

        # Y-axis configuration
        if condition_index == 0:
            ax.set_ylabel("Freq.", fontsize=6)
        else:
            ax.yaxis.set_visible(False)

        ax.tick_params(axis="both", which="major", labelsize=6)

    def create_cellcycle_scatter(
        self,
        ax: Axes,
        data: pd.DataFrame,
        condition_index: int,
        total_conditions: int,
    ) -> None:
        """Create cell cycle scatter plot (DAPI vs EdU).

        Args:
            ax: Matplotlib axes to plot on
            data: Data for this condition
            condition_index: Index of condition
            total_conditions: Total number of conditions
        """
        import seaborn as sns

        # Create scatter plot with cell cycle phase coloring (match original order)
        phases = ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]
        sns.scatterplot(
            data=data,
            x=self.dapi_col,
            y=self.edu_col,
            hue="cell_cycle",
            palette=self.colors[: len(phases)],
            hue_order=phases,
            s=2,
            alpha=1,
            ax=ax,
        )

        # Remove legend from individual plots
        ax.legend().remove()

        # Setup axes using scatter plot configuration
        self._setup_scatter_axes(
            ax,
            condition_index,
            total_conditions,
            self.edu_col,
            self.edu_min,
            self.edu_max,
        )

    def _setup_scatter_axes(
        self,
        ax: Axes,
        condition_index: int,
        total_conditions: int,
        y_col: str,
        y_min: float,
        y_max: float,
    ) -> None:
        """Setup common scatter plot axes configuration.

        Args:
            ax: Matplotlib axes
            condition_index: Index of current condition
            total_conditions: Total number of conditions
            y_col: Y-axis column name
            y_min: Y-axis minimum
            y_max: Y-axis maximum
        """
        from matplotlib import ticker

        # X-axis (DAPI) configuration
        ax.set_xscale("log")
        ax.set_xlim(self.dapi_min, self.dapi_max)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: str(int(x)))
        )
        ax.set_xticks([2, 4, 8])
        ax.set_xlabel("")

        # Y-axis configuration
        ax.set_ylim(y_min, y_max)

        # Show y-label only on leftmost plot
        if condition_index == 0:
            y_label = self._format_y_label(y_col)
            ax.set_ylabel(y_label, fontsize=6)
        else:
            ax.yaxis.set_visible(False)

        if "edu" in y_col.lower():
            ax.set_yscale("log", base=2)
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: str(int(x)))
            )

        # Add reference lines
        ax.axvline(x=2, color="gray", linestyle="--", linewidth=0.5)  # G1/S
        ax.axvline(x=4, color="gray", linestyle="--", linewidth=0.5)  # G2/M
        ax.axhline(y=4, color="gray", linestyle="--", linewidth=0.5)  # EdU Pos

        # General styling
        ax.grid(False)
        ax.tick_params(axis="both", which="major", labelsize=6)

    def _format_y_label(self, y_col: str) -> str:
        """Format y-axis label from column name.

        Args:
            y_col: Column name

        Returns:
            Formatted label
        """
        if "edu" in y_col.lower():
            return "norm. EdU int."
        else:
            # Extract meaningful part from column name
            parts = y_col.split("_")
            if len(parts) >= 3:
                return f"{parts[2]} norm."
            return y_col.replace("_", " ").title()


class BaseHistogramPlot(BaseCombPlot):
    """Base class for histogram subplot components."""

    @property
    def plot_type(self) -> str:
        """Return the type of plot."""
        return "histogram"

    def create_histogram(
        self,
        ax: Axes,
        data: pd.DataFrame,
        condition_index: int,
        show_individual_xlabel: bool = False,
    ) -> None:
        """Create histogram of DAPI intensity.

        Args:
            ax: Matplotlib axes to plot on
            data: Data for this condition
            condition_index: Index of condition (for styling)
            show_individual_xlabel: Whether to show x-label on this individual axis
        """
        # Use seaborn for histogram (exactly like original working code)
        sns.histplot(data=data, x=self.dapi_col, ax=ax, color=self.colors[-1])

        # Configure axes (context-aware x-axis labeling)
        ax.set_xscale("log", base=2)
        ax.set_xlim(1, 16)  # Use original fixed limits

        # Handle x-axis label and visibility based on context
        if show_individual_xlabel:
            # Single plot: show x-label and ticks on this axis
            ax.set_xlabel("norm. DNA content", fontsize=6)
            ax.xaxis.set_visible(True)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: str(int(x)))
            )
            ax.set_xticks([2, 4, 8])
        else:
            # Multiple plots or part of combplot: no individual x-label
            ax.set_xlabel("")
            ax.xaxis.set_visible(False)

        # Y-axis configuration
        if condition_index == 0:
            ax.set_ylabel("Freq.", fontsize=6)
        else:
            ax.yaxis.set_visible(False)

        ax.tick_params(axis="both", which="major", labelsize=6)


class BaseScatterPlot(BaseCombPlot):
    """Base class for scatter plot subplot components."""

    def setup_scatter_axes(
        self,
        ax: Axes,
        condition_index: int,
        total_conditions: int,
        y_col: str,
        y_min: float,
        y_max: float,
    ) -> None:
        """Setup common scatter plot axes configuration.

        Args:
            ax: Matplotlib axes
            condition_index: Index of current condition
            total_conditions: Total number of conditions
            y_col: Y-axis column name
            y_min: Y-axis minimum
            y_max: Y-axis maximum
        """
        # X-axis (DAPI) configuration
        ax.set_xscale("log")
        ax.set_xlim(self.dapi_min, self.dapi_max)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: str(int(x)))
        )
        ax.set_xticks([2, 4, 8])
        ax.set_xlabel("")

        # Y-axis configuration
        ax.set_ylim(y_min, y_max)

        # Show y-label only on leftmost plot
        if condition_index == 0 or self._single_ax_mode:
            y_label = self._format_y_label(y_col)
            ax.set_ylabel(y_label, fontsize=6)
        else:
            ax.yaxis.set_visible(False)

        if "edu" in y_col.lower():
            ax.set_yscale("log", base=2)
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: str(int(x)))
            )

        # Add reference lines
        ax.axvline(x=2, color="gray", linestyle="--", linewidth=0.5)  # G1/S
        ax.axvline(x=4, color="gray", linestyle="--", linewidth=0.5)  # G2/M
        ax.axhline(y=4, color="gray", linestyle="--", linewidth=0.5)  # EdU Pos

        # General styling
        ax.grid(False)
        ax.tick_params(axis="both", which="major", labelsize=6)

    def _format_y_label(self, y_col: str) -> str:
        """Format y-axis label from column name.

        Args:
            y_col: Column name

        Returns:
            Formatted label
        """
        if "edu" in y_col.lower():
            return "norm. EdU int."
        # Extract meaningful part from column name
        parts = y_col.split("_")
        return f"{parts[2]} norm." if len(parts) >= 3 else y_col

    def add_kde_overlay(
        self, ax: Axes, data: pd.DataFrame, x_col: str, y_col: str
    ) -> None:
        """Add KDE density overlay to scatter plot.

        Args:
            ax: Matplotlib axes
            data: Data to plot
            x_col: X-axis column
            y_col: Y-axis column
        """
        try:
            sns.kdeplot(
                data=data,
                x=x_col,
                y=y_col,
                fill=True,
                alpha=0.3,
                cmap="rocket_r",
                ax=ax,
            )
        except (ValueError, KeyError, IndexError) as e:
            warnings.warn(f"Could not add KDE overlay: {e}", stacklevel=2)


class BaseCellCycleScatter(BaseScatterPlot):
    """Base class for cell cycle scatter plots (DAPI vs EdU)."""

    @property
    def plot_type(self) -> str:
        """Return the type of plot."""
        return "cellcycle_scatter"

    def create_cellcycle_scatter(
        self,
        ax: Axes,
        data: pd.DataFrame,
        condition_index: int,
        total_conditions: int,
    ) -> None:
        """Create cell cycle scatter plot.

        Args:
            ax: Matplotlib axes to plot on
            data: Data for this condition
            condition_index: Index of condition
            total_conditions: Total number of conditions
        """
        # Cell cycle phases and colors
        phases = ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]

        # Create scatter plot
        sns.scatterplot(
            data=data,
            x=self.dapi_col,
            y=self.edu_col,
            hue="cell_cycle",
            hue_order=phases,
            palette=self.colors[: len(phases)],
            s=2,
            alpha=1,
            ax=ax,
        )

        # Remove legend (will be handled separately)
        ax.legend().remove()

        # Setup axes
        self.setup_scatter_axes(
            ax,
            condition_index,
            total_conditions,
            self.edu_col,
            self.edu_min,
            self.edu_max,
        )

        # Add KDE overlay
        self.add_kde_overlay(ax, data, self.dapi_col, self.edu_col)


class BaseFeatureScatter(BaseScatterPlot):
    """Base class for feature scatter plots (DAPI vs any feature)."""

    @property
    def plot_type(self) -> str:
        """Return the type of plot."""
        return "feature_scatter"

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        feature_col: str,
        feature_threshold: float,
        **kwargs: Any,
    ) -> None:
        """Initialize feature scatter plot.

        Args:
            data: DataFrame containing data
            conditions: List of conditions
            feature_col: Column name for the feature to plot
            feature_threshold: Threshold value for color categorization
            **kwargs: Additional arguments
        """
        super().__init__(data, conditions, **kwargs)
        self.feature_col = feature_col
        self.feature_threshold = feature_threshold

        # Calculate feature range
        self.feature_min = data[feature_col].quantile(0.01) * 0.8
        self.feature_max = data[feature_col].quantile(0.99) * 1.5

    def create_feature_scatter(
        self,
        ax: Axes,
        data: pd.DataFrame,
        condition_index: int,
        total_conditions: int,
    ) -> None:
        """Create feature scatter plot.

        Args:
            ax: Matplotlib axes to plot on
            data: Data for this condition
            condition_index: Index of condition
            total_conditions: Total number of conditions
        """
        # Create binary categories based on threshold
        data = data.copy()
        data.loc[:, "threshold_category"] = data[self.feature_col].apply(
            lambda x: "below" if x < self.feature_threshold else "above"
        )

        # Create scatter plot with threshold-based coloring
        sns.scatterplot(
            data=data,
            x=self.dapi_col,
            y=self.feature_col,
            hue="threshold_category",
            palette={
                "below": "#87CEEB",
                "above": "#4169E1",
            },  # Light blue, blue
            hue_order=["below", "above"],
            s=2,
            alpha=1,
            ax=ax,
        )

        # Remove legend
        ax.legend().remove()

        # Setup axes
        self.setup_scatter_axes(
            ax,
            condition_index,
            total_conditions,
            self.feature_col,
            self.feature_min,
            self.feature_max,
        )
