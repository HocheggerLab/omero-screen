"""Feature plot factory with unified configuration and base class architecture."""

import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.base import BasePlotBuilder, BasePlotConfig
from omero_screen_plots.colors import COLOR
from omero_screen_plots.stats import set_significance_marks_adaptive
from omero_screen_plots.utils import (
    finalize_plot_with_title,
    grouped_x_positions,
    prepare_plot_data,
    set_y_limits,
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
    violin: bool = False  # Use violin plots instead of box plots
    show_scatter: bool = True  # Show scatter points overlay
    threshold: float = 1.5  # for threshold plots (default 1.5x mode)
    show_legend: bool = True  # show default plate legend
    show_significance: bool = True
    show_repeat_points: bool = True

    # Normalization settings (for NormFeaturePlot)
    normalize_by_plate: bool = True  # Normalize within each plate
    save_norm_qc: bool = False  # Save normalization QC plots

    # Triplicate settings (for NormFeaturePlot)
    show_triplicates: bool = False  # Show individual triplicate bars
    show_error_bars: bool = True  # Show error bars on summary bars
    repeat_offset: float = 0.18  # Offset between triplicate bars
    max_repeats: int = 3  # Maximum number of repeats to show
    show_boxes: bool = True  # Draw boxes around triplicates


class BaseFeaturePlot(BasePlotBuilder):
    """Base class for feature plots with common functionality."""

    PLOT_TYPE_NAME = "feature"
    config: FeaturePlotConfig  # Type annotation for mypy

    def __init__(self, config: FeaturePlotConfig | None = None):
        """Initialize with configuration."""
        super().__init__(config or FeaturePlotConfig())
        self._axes_provided: bool = False

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
        self._setup_figure(axes)

        # Get x positions for plotting
        x_positions = self._get_x_positions(conditions)

        # Build plot (delegated to subclasses) - pass x_positions
        self.build_plot(
            processed_data,
            feature=feature,
            conditions=conditions,
            condition_col=condition_col,
            x_positions=x_positions,
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
        self._save_plot()

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

    def _setup_figure(self, axes: Axes | None) -> None:
        """Setup figure using parent's create_figure method."""
        if axes:
            self.fig = axes.figure  # type: ignore[assignment]
            self.ax = axes
            self._axes_provided = True
        else:
            # Use parent's create_figure method
            self.create_figure(axes)
            self._axes_provided = False

    @abstractmethod
    def build_plot(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> "BasePlotBuilder":
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

    def _add_repeat_points_with_shapes(
        self,
        df_median: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        feature: str,
        x_positions: list[float],
    ) -> None:
        """Add repeat points with different shapes for each plate. Override in subclasses."""
        # Default implementation - can be overridden by subclasses
        from omero_screen_plots.utils import show_repeat_points_adaptive

        assert self.ax is not None
        show_repeat_points_adaptive(
            df_median,
            conditions,
            condition_col,
            feature,
            self.ax,
            self.config.group_size,
            x_positions,
        )

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
            # Use custom plate-specific markers instead of generic function
            self._add_repeat_points_with_shapes(
                df_median, conditions, condition_col, feature, x_positions
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

    def _save_plot(self) -> None:
        """Save figure using parent's save_figure method."""
        # Use parent's save_figure method with our filename
        self.save_figure(self._filename or "featureplot")


class StandardFeaturePlot(BaseFeaturePlot):
    """Unified feature plot supporting box/violin plots with optional scatter points."""

    PLOT_TYPE_NAME = "feature"

    def build_plot(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> "BasePlotBuilder":
        """Build unified feature plot with box/violin plots and optional scatter points."""
        # Extract required parameters from kwargs
        feature = kwargs["feature"]
        conditions = kwargs["conditions"]
        condition_col = kwargs["condition_col"]
        x_positions = kwargs["x_positions"]

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

        # Add default plate legend if requested and we have repeat points and multiple plates
        if (
            self.config.show_legend
            and self.config.show_repeat_points
            and self._has_multiple_plates(data)
        ):
            self._add_plate_legend(data)

        return self

    def _add_repeat_points_with_shapes(
        self,
        df_median: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        feature: str,
        x_positions: list[float],
    ) -> None:
        """Add repeat points with different shapes for each plate."""
        assert self.ax is not None

        # Define marker shapes for each plate
        markers = ["s", "o", "^"]  # square, circle, triangle
        plate_ids = sorted(df_median["plate_id"].unique())

        # Create condition to x position mapping
        cond_to_x = dict(zip(conditions, x_positions, strict=False))

        # Add jitter width for spacing
        jitter_width = 0.07

        # Plot points for each plate with different shapes
        for plate_idx, plate_id in enumerate(
            plate_ids[:3]
        ):  # Limit to 3 plates
            plate_data = df_median[df_median["plate_id"] == plate_id]
            marker = markers[plate_idx % len(markers)]

            for condition in conditions:
                cond_plate_data = plate_data[
                    plate_data[condition_col] == condition
                ]
                if not cond_plate_data.empty:
                    x_base = cond_to_x[condition]
                    y_values = cond_plate_data[feature].values

                    # Add jitter for visibility when multiple plates
                    if len(plate_ids) > 1:
                        x_jittered = np.random.normal(
                            x_base, jitter_width, size=len(y_values)
                        )
                    else:
                        x_jittered = np.array([x_base] * len(y_values))

                    self.ax.scatter(
                        x_jittered,
                        y_values,
                        marker=marker,
                        color="lightgray",
                        s=25,  # size
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.8,
                        zorder=3,
                    )

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
                        np.asarray(cond_data[feature].values, dtype=float),
                        x_positions[idx],
                        color=base_color,
                    )
                else:
                    create_standard_boxplot(
                        self.ax,
                        np.asarray(cond_data[feature].values, dtype=float),
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

    def _has_multiple_plates(self, data: pd.DataFrame) -> bool:
        """Check if data has multiple plates."""
        return bool(data["plate_id"].nunique() > 1)

    def _add_plate_legend(self, data: pd.DataFrame) -> None:
        """Add default legend showing plate symbols."""
        assert self.ax is not None

        plate_ids = sorted(data["plate_id"].unique())
        # Limit to 3 plates maximum for legend
        plate_ids = plate_ids[:3]

        # Define marker shapes for each plate
        markers = ["s", "o", "^"]  # square, circle, triangle

        from matplotlib.lines import Line2D

        # Create legend handles
        handles = []
        for i, plate_id in enumerate(plate_ids):
            marker = markers[i % len(markers)]
            handle = Line2D(
                [0],
                [0],
                marker=marker,
                color="lightgray",
                markerfacecolor="lightgray",
                markersize=6,
                linestyle="None",
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=str(plate_id),
            )
            handles.append(handle)

        # Add legend
        self.ax.legend(
            handles=handles,
            title="Plate ID",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )


class NormFeaturePlot(BaseFeaturePlot):
    """Normalized feature plot with threshold-based stacked bars and optional triplicates."""

    PLOT_TYPE_NAME = "norm_feature"

    def __init__(self, config: FeaturePlotConfig | None = None):
        """Initialize with configuration."""
        super().__init__(config)
        # Store normalized data and threshold for later use
        self._normalized_data: pd.DataFrame | None = None
        self._threshold: float = 1.5  # Default threshold

    def _get_norm_colors(self) -> list[str]:
        """Get colors for positive/negative categories from config."""
        if self.config.colors and len(self.config.colors) >= 2:
            return self.config.colors[:2]  # Use first two colors
        else:
            # Default to green scheme
            return [
                COLOR.OLIVE.value,
                COLOR.LIGHT_GREEN.value,
            ]  # positive, negative

    def _process_data(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
    ) -> pd.DataFrame:
        """Process data with normalization before threshold application."""
        # Import normalize_by_mode locally to avoid circular imports
        from omero_screen_plots.normalise import normalize_by_mode
        from omero_screen_plots.utils import selector_val_filter

        # First filter the data without scaling
        filtered_data = selector_val_filter(
            df, selector_col, selector_val, condition_col, conditions
        )
        assert filtered_data is not None, "No data found"

        # Normalize by mode within plates (or globally based on config)
        normalize_by_plate = getattr(self.config, "normalize_by_plate", True)
        group_column = "plate_id" if normalize_by_plate else None

        # Check if we should save normalization QC plots
        save_norm_qc = getattr(self.config, "save_norm_qc", False)

        if save_norm_qc and self.config.path:
            # Use normalize_and_plot to generate QC documentation
            from omero_screen_plots.normalise import normalize_and_plot

            # Generate QC filename
            qc_filename = f"{feature}_norm_qc"

            self._normalized_data = normalize_and_plot(
                filtered_data,
                feature,
                group_column,
                plot=True,
                save=True,
                path=self.config.path / qc_filename,
            )
        else:
            # Just normalize without QC plots
            self._normalized_data = normalize_by_mode(
                filtered_data, feature, group_column, suffix="_norm"
            )

        # Use threshold from config (defaults to 1.5)
        self._threshold = getattr(self.config, "threshold", 1.5)

        # Apply threshold and calculate proportions
        return self._calculate_threshold_proportions(
            self._normalized_data, feature, conditions, condition_col
        )

    def _calculate_threshold_proportions(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
    ) -> pd.DataFrame:
        """Calculate positive/negative proportions based on threshold."""
        normalized_feature = f"{feature}_norm"

        # Create threshold categories
        df = df.copy()
        df["threshold_category"] = np.where(
            df[normalized_feature] > self._threshold, "positive", "negative"
        )

        # Calculate proportions per plate/condition
        proportions = []
        for condition in conditions:
            for plate_id in sorted(df["plate_id"].unique()):
                plate_cond_data = df[
                    (df[condition_col] == condition)
                    & (df["plate_id"] == plate_id)
                ]

                if not plate_cond_data.empty:
                    total_cells = len(plate_cond_data)
                    positive_cells = len(
                        plate_cond_data[
                            plate_cond_data["threshold_category"] == "positive"
                        ]
                    )
                    positive_percent = (positive_cells / total_cells) * 100
                    negative_percent = 100 - positive_percent

                    proportions.extend(
                        [
                            {
                                "plate_id": plate_id,
                                condition_col: condition,
                                "category": "positive",
                                "percent": positive_percent,
                                "count": positive_cells,
                                "total": total_cells,
                            },
                            {
                                "plate_id": plate_id,
                                condition_col: condition,
                                "category": "negative",
                                "percent": negative_percent,
                                "count": total_cells - positive_cells,
                                "total": total_cells,
                            },
                        ]
                    )

        return pd.DataFrame(proportions)

    def build_plot(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> "BasePlotBuilder":
        """Build normalized feature plot with stacked bars."""
        # Extract required parameters from kwargs
        # feature is passed but not used in normalized plots (already processed in data)
        conditions = kwargs["conditions"]
        condition_col = kwargs["condition_col"]
        x_positions = kwargs["x_positions"]

        assert self.ax is not None

        # Check if we should show triplicates
        show_triplicates = getattr(self.config, "show_triplicates", False)

        if show_triplicates:
            self._build_triplicate_plot(
                data, conditions, condition_col, x_positions
            )
        else:
            self._build_summary_plot(
                data, conditions, condition_col, x_positions
            )

        # Add legend
        self._add_threshold_legend()

        return self

    def _build_summary_plot(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        x_positions: list[float],
    ) -> None:
        """Build summary plot with mean values and error bars."""
        assert self.ax is not None

        # Calculate mean and std for each condition and category
        summary_stats = (
            data.groupby([condition_col, "category"])["percent"]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Get colors from configuration
        colors = self._get_norm_colors()  # positive, negative
        categories = ["positive", "negative"]  # Order matters for stacking

        # Create stacked bars
        bottoms = [0] * len(x_positions)
        for cat_idx, category in enumerate(categories):
            cat_data = summary_stats[summary_stats["category"] == category]

            means = []
            stds = []
            for condition in conditions:
                cond_data = cat_data[cat_data[condition_col] == condition]
                if not cond_data.empty:
                    means.append(cond_data["mean"].iloc[0])
                    stds.append(cond_data["std"].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)

            # Check if we should show error bars
            show_error_bars = getattr(self.config, "show_error_bars", True)

            self.ax.bar(
                x_positions,
                means,
                width=0.5,
                bottom=bottoms,
                color=colors[cat_idx],
                label=f"{category.title()}",
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                yerr=stds if show_error_bars else None,
                capsize=3,
                error_kw={"linewidth": 1},
            )

            # Update bottoms for stacking
            bottoms = [bottoms[i] + means[i] for i in range(len(means))]

    def _build_triplicate_plot(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        x_positions: list[float],
    ) -> None:
        """Build triplicate plot showing individual plate data."""
        assert self.ax is not None

        # Get colors from configuration
        colors = self._get_norm_colors()  # positive, negative
        categories = ["positive", "negative"]  # Order matters for stacking

        # Get triplicate settings
        repeat_offset = getattr(self.config, "repeat_offset", 0.18)
        max_repeats = getattr(self.config, "max_repeats", 3)

        # Get available plate IDs (up to max_repeats)
        plate_ids = sorted(data["plate_id"].unique())[:max_repeats]

        # Plot bars for each condition and plate
        for cond_idx, condition in enumerate(conditions):
            base_x = x_positions[cond_idx]

            # Calculate x positions for ALL max_repeats slots (consistent spacing)
            if max_repeats == 1:
                rep_x_positions = [base_x]
            else:
                rep_x_positions = [
                    base_x + (rep_idx - (max_repeats - 1) / 2) * repeat_offset
                    for rep_idx in range(max_repeats)
                ]

            # Draw bars only for plates that exist, but maintain consistent x positions
            for rep_idx in range(max_repeats):
                if rep_idx < len(plate_ids):
                    # Plate exists - draw the bar
                    plate_id = plate_ids[rep_idx]
                    rep_x = rep_x_positions[rep_idx]
                    plate_data = data[
                        (data[condition_col] == condition)
                        & (data["plate_id"] == plate_id)
                    ]

                    if not plate_data.empty:
                        # Create stacked bar for this replicate
                        y_bottom = 0
                        for cat_idx, category in enumerate(categories):
                            cat_data = plate_data[
                                plate_data["category"] == category
                            ]
                            if not cat_data.empty:
                                percent_val = cat_data["percent"].iloc[0]
                                self.ax.bar(
                                    rep_x,
                                    percent_val,
                                    width=repeat_offset,  # Full width so bars touch
                                    bottom=y_bottom,
                                    color=colors[cat_idx],
                                    edgecolor="white",
                                    linewidth=0.5,
                                    alpha=0.9,
                                )
                                y_bottom += percent_val
                # If rep_idx >= len(plate_ids): plate doesn't exist - leave empty space

        # Draw boxes around triplicates if requested
        show_boxes = getattr(self.config, "show_boxes", True)
        if show_boxes:
            self._draw_triplicate_boxes(
                data,
                conditions,
                condition_col,
                x_positions,
                plate_ids,
                repeat_offset,
            )

    def _draw_triplicate_boxes(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        x_positions: list[float],
        plate_ids: list[str],
        repeat_offset: float,
    ) -> None:
        """Draw boxes around triplicate bars."""
        assert self.ax is not None

        from matplotlib.patches import Rectangle

        y_min = 0
        for cond_idx, condition in enumerate(conditions):
            base_x = x_positions[cond_idx]

            # Get condition data to check if we have data
            cond_data = data[data[condition_col] == condition]
            if cond_data.empty:
                continue

            # Calculate triplicate x positions - always use max_repeats for consistent box size
            max_repeats = getattr(self.config, "max_repeats", 3)
            if max_repeats == 1:
                trip_xs = [base_x]
            else:
                # Always calculate positions for max_repeats, not actual number of plates
                trip_xs = [
                    base_x + (rep_idx - (max_repeats - 1) / 2) * repeat_offset
                    for rep_idx in range(max_repeats)
                ]

            # Calculate maximum height for this condition's triplicates
            y_max_box = max(
                (
                    cond_data[cond_data["plate_id"] == plate_id][
                        "percent"
                    ].sum()
                    for plate_id in plate_ids
                    if not cond_data[cond_data["plate_id"] == plate_id].empty
                ),
                default=100,  # Default to 100% if no data
            )

            # Calculate box bounds
            left = min(trip_xs) - repeat_offset / 2
            right = max(trip_xs) + repeat_offset / 2

            # Draw rectangle (following cellcycle approach exactly)
            rect = Rectangle(
                (left, y_min),
                width=right - left,
                height=y_max_box - y_min,
                linewidth=0.5,
                edgecolor="black",
                facecolor="none",
                zorder=10,
            )
            self.ax.add_patch(rect)

    def _add_threshold_legend(self) -> None:
        """Add legend for positive/negative categories."""
        assert self.ax is not None

        from matplotlib.patches import Patch

        colors = self._get_norm_colors()  # positive, negative
        categories = ["Positive", "Negative"]

        legend_handles = [
            Patch(facecolor=colors[i], edgecolor="white", label=category)
            for i, category in enumerate(categories)
        ]

        self.ax.legend(
            handles=legend_handles,
            title=f"Threshold: {self._threshold:.1f}x",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    def _format_axes(
        self, feature: str, conditions: list[str], x_positions: list[float]
    ) -> None:
        """Format axes labels and limits for normalized plot."""
        assert self.ax is not None

        # Set y-axis label and limits
        self.ax.set_ylabel("% of Population")
        self.ax.set_ylim(0, 110)

        # Set x-axis
        self.ax.set_xlabel("")
        self.ax.set_xticks(x_positions)

        if self.config.show_x_labels:
            self.ax.set_xticklabels(
                conditions, rotation=self.config.rotation, ha="right"
            )
        else:
            self.ax.set_xticklabels([])

    def _finalize_plot(self, feature: str, selector_val: str | None) -> None:
        """Finalize plot with title."""
        # Generate default title
        default_title = (
            f"{feature} threshold analysis {selector_val}"
            if selector_val
            else f"{feature} threshold analysis"
        )

        # Use provided title, config title, or default
        title = self.config.title or default_title

        # Use utility function for consistent formatting
        assert self.fig is not None
        self._filename = finalize_plot_with_title(
            self.fig, title, default_title, self._axes_provided
        )

    def _add_statistical_elements(
        self,
        data: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        x_positions: list[float],
    ) -> None:
        """Add statistical elements for proportion data using existing functions."""
        if not self.config.show_significance or len(conditions) < 2:
            return

        # Get unique plates to check if we have enough replicates
        n_plates = data["plate_id"].nunique()
        if n_plates < 3:
            return  # Need at least 3 replicates for meaningful statistics

        assert self.ax is not None

        # Filter for positive category only (that's what we're comparing)
        positive_data = data[data["category"] == "positive"]

        # Use the existing set_significance_marks_adaptive function
        # It will use calculate_pvalues internally with the "percent" column
        y_max = 105  # Position marks at 105% for percentage plots

        set_significance_marks_adaptive(
            self.ax,
            positive_data,
            conditions,
            condition_col,
            "percent",  # This is the column with proportion data
            y_max,
            group_size=self.config.group_size,
            x_positions=x_positions,
        )
