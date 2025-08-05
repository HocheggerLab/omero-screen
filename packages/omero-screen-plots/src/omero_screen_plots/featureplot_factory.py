"""Feature plot factory with unified configuration and base class architecture."""

import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.base import BasePlotConfig
from omero_screen_plots.colors import COLOR
from omero_screen_plots.stats import set_significance_marks_adaptive
from omero_screen_plots.utils import (
    convert_size_to_inches,
    finalize_plot_with_title,
    grouped_x_positions,
    prepare_plot_data,
    save_fig,
    set_y_limits,
    show_repeat_points_adaptive,
)

# Suppress matplotlib and seaborn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


@dataclass
class FeaturePlotConfig(BasePlotConfig):
    """Configuration for feature plots."""

    # Figure settings
    fig_size: tuple[float, float] = (5, 5)
    size_units: str = "cm"
    dpi: int = 300

    # Save settings
    save: bool = False
    file_format: str = "pdf"
    tight_layout: bool = False
    path: Path | None = None

    # Display settings
    title: str | None = None
    colors: list[str] = field(default_factory=list)

    # Feature plot specific settings
    scale: bool = False
    ymax: float | tuple[float, float] | None = None
    group_size: int = 1
    within_group_spacing: float = 0.2
    between_group_gap: float = 0.5
    show_x_labels: bool = True
    rotation: int = 45

    # Plot style settings
    plot_style: str = "standard"  # "standard", "simple", "threshold"
    violin: bool = False  # Use violin plots instead of box plots
    show_scatter: bool = True  # Show scatter points overlay
    threshold: float = 0.0  # for threshold plots
    legend: tuple[str, list[str]] | None = None  # for plots with legend
    show_significance: bool = True
    show_repeat_points: bool = True


class BaseFeaturePlot:
    """Base class for feature plots with common functionality."""

    PLOT_TYPE_NAME = "feature"

    def __init__(self, config: FeaturePlotConfig | None = None):
        """Initialize with configuration."""
        self.config = config or FeaturePlotConfig()
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self._axes_provided: bool = False
        self._filename: str | None = None

    def create_plot(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str = "condition",
        selector_col: str | None = None,
        selector_val: str | None = None,
        axes: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Create complete feature plot.

        Args:
            df: Input dataframe
            feature: Feature column to plot
            conditions: List of conditions to plot
            condition_col: Column name containing conditions
            selector_col: Optional column for filtering
            selector_val: Value to filter by if selector_col provided
            axes: Optional existing axes to plot on

        Returns:
            Tuple of (Figure, Axes)
        """
        # Validate inputs
        self._validate_inputs(
            df,
            feature,
            conditions,
            condition_col,
            selector_col,
            selector_val,
        )

        # Filter and process data
        processed_data = self._process_data(
            df,
            feature,
            conditions,
            condition_col,
            selector_col,
            selector_val,
        )

        # Create figure
        self._create_figure(axes)

        # Get x positions for plotting
        x_positions = self._get_x_positions(conditions)

        # Build plot (delegated to subclasses) - pass x_positions
        self._build_plot(
            processed_data, feature, conditions, condition_col, x_positions
        )

        # Add statistical elements (repeat points and significance marks)
        self._add_statistical_elements(
            processed_data, feature, conditions, condition_col, x_positions
        )

        # Format axes
        self._format_axes(feature, conditions, x_positions)

        # Finalize
        self._finalize_plot(feature, selector_val)

        # Save if configured
        self._save_figure()

        assert self.fig is not None and self.ax is not None, (
            "Figure and axes should be created"
        )
        return self.fig, self.ax

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
    ) -> None:
        """Validate all inputs with improved error messages."""
        if df.empty:
            raise ValueError("Input dataframe is empty")

        # Check required columns
        required_cols = ["plate_id"]
        if missing_cols := set(required_cols) - set(df.columns):
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        # Check feature column
        if feature not in df.columns:
            raise ValueError(
                f"Feature column '{feature}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Check condition column
        if condition_col not in df.columns:
            raise ValueError(
                f"Condition column '{condition_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Check conditions exist
        available_conditions = set(df[condition_col].unique())
        if missing_conditions := set(conditions) - available_conditions:
            raise ValueError(
                f"Conditions not found in data: {missing_conditions}. "
                f"Available conditions: {sorted(available_conditions)}"
            )

        # Check selector parameters
        if selector_col and selector_col not in df.columns:
            raise ValueError(
                f"Selector column '{selector_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        if selector_col and not selector_val:
            available_values = sorted(df[selector_col].unique())
            raise ValueError(
                f"selector_val must be provided when selector_col is specified. "
                f"Available values in '{selector_col}': {available_values}"
            )

        if (
            selector_col
            and selector_val
            and selector_val not in df[selector_col].unique()
        ):
            available_values = sorted(df[selector_col].unique())
            raise ValueError(
                f"Value '{selector_val}' not found in column '{selector_col}'. "
                f"Available values: {available_values}"
            )

    def _process_data(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
    ) -> pd.DataFrame:
        """Process data for feature plots using existing utility."""
        # Use the existing prepare_plot_data utility function
        processed_data = prepare_plot_data(
            df,
            feature,
            conditions,
            condition_col,
            selector_col,
            selector_val,
            self.config.scale,
        )

        if processed_data.empty:
            raise ValueError(
                "No data remaining after filtering and processing"
            )

        return processed_data

    def _create_figure(self, axes: Axes | None) -> None:
        """Create or use existing figure."""
        if axes:
            self.fig = axes.figure  # type: ignore[assignment]
            self.ax = axes
            self._axes_provided = True
        else:
            fig_inches = convert_size_to_inches(
                self.config.fig_size, self.config.size_units
            )
            self.fig, self.ax = plt.subplots(figsize=fig_inches)
            self._axes_provided = False

    @abstractmethod
    def _build_plot(
        self,
        data: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        x_positions: list[float],
    ) -> None:
        """Build the specific feature plot type. Implemented by subclasses."""

    def _get_x_positions(self, conditions: list[str]) -> list[float]:
        """Get x positions for plotting based on grouping configuration."""
        if self.config.group_size > 1:
            return grouped_x_positions(
                len(conditions),
                group_size=self.config.group_size,
                within_group_spacing=self.config.within_group_spacing,
                between_group_gap=self.config.between_group_gap,
            )
        else:
            return list(range(len(conditions)))

    def _calculate_median_data(
        self,
        data: pd.DataFrame,
        feature: str,
        condition_col: str,
    ) -> pd.DataFrame:
        """Calculate median values per plate and condition."""
        return (
            data.groupby(["plate_id", condition_col])[feature]
            .median()
            .reset_index()
        )

    def _sample_data_for_display(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
    ) -> pd.DataFrame:
        """Sample data points for display to avoid overplotting."""
        from omero_screen_plots.utils import select_datapoints

        return select_datapoints(data, conditions, condition_col)

    def _get_plate_colors(self, n_plates: int) -> list[str]:
        """Get colors for different plates, cycling through available colors."""
        # Default plate colors using COLOR enum for better differentiation
        if not self.config.colors:
            # Use BLUE, YELLOW, PINK for plate differentiation
            plate_colors = [
                COLOR.BLUE.value,
                COLOR.YELLOW.value,
                COLOR.PINK.value,
            ]
        else:
            plate_colors = self.config.colors

        # Repeat colors if more plates than colors
        return [plate_colors[i % len(plate_colors)] for i in range(n_plates)]

    def _add_statistical_elements(
        self,
        data: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        x_positions: list[float],
    ) -> None:
        """Add repeat points and significance marks when appropriate."""
        if (
            not self.config.show_repeat_points
            and not self.config.show_significance
        ):
            return

        # Calculate median data for repeat points using the common method
        df_median = self._calculate_median_data(data, feature, condition_col)

        assert self.ax is not None

        # Show statistical elements for all group sizes (adaptive functions handle positioning)
        if self.config.show_repeat_points:
            show_repeat_points_adaptive(
                df_median,
                conditions,
                condition_col,
                feature,
                self.ax,
                self.config.group_size,
                x_positions,
            )

        if self.config.show_significance and data.plate_id.nunique() >= 3:
            # Use the exact same approach for both box and violin plots
            y_top = self.ax.get_ylim()[1]

            set_significance_marks_adaptive(
                self.ax,
                df_median,
                conditions,
                condition_col,
                feature,
                y_top,
                group_size=self.config.group_size,
                x_positions=x_positions,
            )

    def _format_axes(
        self, feature: str, conditions: list[str], x_positions: list[float]
    ) -> None:
        """Format axes labels and limits."""
        assert self.ax is not None

        # Set y-axis limits if specified
        set_y_limits(self.ax, self.config.ymax)

        # Set y-axis label
        self.ax.set_ylabel(feature.replace("_", " ").title())
        self.ax.set_xlabel("")

        # Set x-axis ticks and labels
        self.ax.set_xticks(x_positions)

        if self.config.show_x_labels:
            self.ax.set_xticklabels(
                conditions, rotation=self.config.rotation, ha="right"
            )
        else:
            self.ax.set_xticklabels([])

    def _finalize_plot(self, feature: str, selector_val: str | None) -> None:
        """Finalize plot with title."""
        # Generate default title using feature name
        default_title = (
            f"{feature} {selector_val}" if selector_val else feature
        )

        # Use provided title, config title, or default
        title = self.config.title or default_title

        # Use utility function for consistent formatting
        assert self.fig is not None
        self._filename = finalize_plot_with_title(
            self.fig, title, default_title, self._axes_provided
        )

    def _save_figure(self) -> None:
        """Save figure if configured."""
        if self.config.save and self.config.path:
            assert self.fig is not None
            save_fig(
                self.fig,
                self.config.path,
                self._filename or "featureplot",
                tight_layout=self.config.tight_layout,
                fig_extension=self.config.file_format,
                resolution=self.config.dpi,
            )


class StandardFeaturePlot(BaseFeaturePlot):
    """Unified feature plot supporting box/violin plots with optional scatter points."""

    PLOT_TYPE_NAME = "feature"

    def _build_plot(
        self,
        data: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        x_positions: list[float],
    ) -> None:
        """Build unified feature plot with box/violin plots and optional scatter points."""
        assert self.ax is not None

        # Create the base plots (box or violin)
        self._create_base_plots(
            data, feature, conditions, condition_col, x_positions
        )

        # Ensure consistent y-axis limits for both plot types
        self._standardize_y_limits(data, feature)

        # Add scatter points if enabled
        if self.config.show_scatter:
            sampled_data = self._sample_data_for_display(
                data, conditions, condition_col
            )
            self._add_scatter_points(
                sampled_data, feature, conditions, condition_col, x_positions
            )

        # Add legend if specified
        if self.config.legend:
            self._add_legend()

    def _create_base_plots(
        self,
        data: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        x_positions: list[float],
    ) -> None:
        """Create base plots (box or violin) at specified positions."""
        from omero_screen_plots.utils import (
            create_standard_boxplot,
            create_standard_violin,
        )

        assert self.ax is not None

        # Get base plot color
        base_color = COLOR.DARKGREY.value

        # Create plots at specified positions
        for idx, condition in enumerate(conditions):
            cond_data = data[data[condition_col] == condition]
            if not cond_data.empty:
                if self.config.violin:
                    create_standard_violin(
                        self.ax,
                        cond_data[feature].values,
                        x_positions[idx],
                        color=base_color,
                    )
                else:
                    create_standard_boxplot(
                        self.ax,
                        cond_data[feature].values,
                        x_positions[idx],
                        color=base_color,
                    )

    def _add_scatter_points(
        self,
        sampled_data: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        x_positions: list[float],
    ) -> None:
        """Add scatter points with jitter for plate differentiation."""
        assert self.ax is not None

        # Return early if no sampled data
        if sampled_data.empty:
            return

        # Create condition to x position mapping
        cond_to_x = dict(zip(conditions, x_positions, strict=False))
        plate_colors = self._get_plate_colors(sampled_data.plate_id.nunique())

        # Add scatter points with jitter for all plates
        plate_ids = sampled_data.plate_id.unique()
        for idx, plate_id in enumerate(plate_ids):
            plate_data = sampled_data[sampled_data.plate_id == plate_id]
            for condition in conditions:
                cond_plate_data = plate_data[
                    plate_data[condition_col] == condition
                ]
                if not cond_plate_data.empty:
                    x_base = cond_to_x[condition]
                    y_values = cond_plate_data[feature].values
                    # Add jitter for visibility
                    x_jittered = x_base + np.random.uniform(
                        -0.1, 0.1, size=len(y_values)
                    )
                    self.ax.scatter(
                        x_jittered,
                        y_values,
                        color=plate_colors[idx],
                        alpha=0.8,
                        s=7,
                        edgecolor=None,
                        linewidth=0.5,
                        zorder=3,
                    )

    def _standardize_y_limits(self, data: pd.DataFrame, feature: str) -> None:
        """Ensure consistent y-axis limits for both box and violin plots."""
        assert self.ax is not None

        # Check if user provided custom y-limits
        if self.config.ymax is not None:
            if isinstance(self.config.ymax, tuple):
                # User provided (y_min, y_max) tuple
                y_bottom, y_top = self.config.ymax
            else:
                # User provided only y_max, calculate y_min from data
                data_min = data[feature].quantile(0.01)  # 1st percentile
                data_range = self.config.ymax - data_min
                y_bottom = data_min - (data_range * 0.05)
                y_top = self.config.ymax

                # Ensure non-negative bottom if data is all positive
                if data_min >= 0:
                    y_bottom = max(0, y_bottom)

        else:
            # Use percentiles to handle outliers better
            data_min = data[feature].quantile(0.01)  # 1st percentile
            data_max = data[feature].quantile(0.99)  # 99th percentile
            data_range = data_max - data_min

            # Add padding: 5% below min, 20% above max (to leave room for significance marks)
            y_bottom = data_min - (data_range * 0.05)
            y_top = data_max + (data_range * 0.20)

            # Ensure non-negative bottom if data is all positive
            if data_min >= 0:
                y_bottom = max(0, y_bottom)

        self.ax.set_ylim(y_bottom, y_top)

    def _add_legend(self) -> None:
        """Add legend to the plot if configured."""
        assert self.ax is not None

        if self.config.legend:
            legend_title, legend_labels = self.config.legend
            from matplotlib.patches import Patch

            # Create legend handles using plate colors
            plate_colors = self._get_plate_colors(len(legend_labels))
            handles = [
                Patch(
                    facecolor=plate_colors[i % len(plate_colors)], label=label
                )
                for i, label in enumerate(legend_labels)
            ]

            self.ax.legend(
                handles=handles,
                title=legend_title,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
