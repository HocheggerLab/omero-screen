"""Simplified count plot implementation with single class architecture."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.colors import COLOR
from omero_screen_plots.stats import set_significance_marks_adaptive
from omero_screen_plots.utils import (
    convert_size_to_inches,
    finalize_plot_with_title,
    grouped_x_positions,
    save_fig,
    show_repeat_points_adaptive,
)


class PlotType(Enum):
    """Enumeration for plot types: normalised and absolute."""

    NORMALISED = "normalised"
    ABSOLUTE = "absolute"


@dataclass
class CountPlotConfig:
    """Configuration for count plots."""

    # Figure settings
    fig_size: tuple[float, float] = (7, 7)
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

    # Count plot specific settings
    plot_type: PlotType = PlotType.NORMALISED
    group_size: int = 1
    within_group_spacing: float = 0.2
    between_group_gap: float = 0.5
    show_x_labels: bool = True
    rotation: int = 45


class CountPlot:
    """Simplified count plot implementation combining config, processing, and plotting."""

    PLOT_TYPE_NAME = "count"

    def __init__(self, config: CountPlotConfig | None = None):
        """Initialize with configuration."""
        self.config = config or CountPlotConfig()
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self._axes_provided: bool = False
        self._filename: str | None = None

    def create_plot(
        self,
        df: pd.DataFrame,
        norm_control: str,
        conditions: list[str],
        condition_col: str = "condition",
        selector_col: str | None = None,
        selector_val: str | None = None,
        axes: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Create complete count plot.

        Args:
            df: Input dataframe
            norm_control: Control condition for normalization
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
            norm_control,
            conditions,
            condition_col,
            selector_col,
            selector_val,
        )

        # Filter and process data (optimized to avoid unnecessary copying)
        processed_data = self._process_data(
            df,
            norm_control,
            conditions,
            condition_col,
            selector_col,
            selector_val,
        )

        # Create figure
        self._create_figure(axes)

        # Build plot
        self._build_plot(processed_data, conditions, condition_col)

        # Finalize
        self._finalize_plot(selector_val)

        # Save if configured
        self._save_figure()

        assert self.fig is not None and self.ax is not None, (
            "Figure and axes should be created"
        )
        return self.fig, self.ax

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        norm_control: str,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
    ) -> None:
        """Validate all inputs with improved error messages."""
        if df.empty:
            raise ValueError("Input dataframe is empty")

        # Check required columns
        required_cols = ["plate_id", "well", "experiment"]
        if missing_cols := set(required_cols) - set(df.columns):
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
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

        # Check norm_control is in conditions
        if norm_control not in conditions:
            raise ValueError(
                f"Normalization control '{norm_control}' must be in conditions list: {conditions}"
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
        norm_control: str,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
    ) -> pd.DataFrame:
        """Process data for count plots with optimized DataFrame operations."""
        # Filter by conditions and selector in one step to avoid copying
        mask = df[condition_col].isin(conditions)
        if selector_col and selector_val:
            mask &= df[selector_col] == selector_val

        filtered_df = df.loc[mask]

        if filtered_df.empty:
            raise ValueError("No data remaining after filtering")

        # Count experiments per well
        well_counts = filtered_df.groupby(
            ["plate_id", condition_col, "well"]
        ).size()
        well_counts = well_counts.reset_index()
        well_counts.columns = list(well_counts.columns[:-1]) + ["well_count"]

        # Calculate mean count across wells with same condition
        grouped = (
            well_counts.groupby(["plate_id", condition_col], as_index=False)
            .agg({"well_count": "mean"})
            .rename(columns={"well_count": "count"})
        )

        # Create pivot for normalization
        pivot_df = grouped.pivot(
            index="plate_id", columns=condition_col, values="count"
        )

        # Validate norm_control exists in pivot
        if norm_control not in pivot_df.columns:
            available_conditions = list(pivot_df.columns)
            raise ValueError(
                f"Control condition '{norm_control}' not found in processed data. "
                f"Available conditions after processing: {available_conditions}"
            )

        # Calculate normalized values
        norm_control_values = pivot_df[norm_control]
        if (norm_control_values == 0).any():
            raise ValueError(
                f"Cannot normalize: control condition '{norm_control}' has zero values"
            )

        normalized_df = pivot_df.div(norm_control_values, axis=0)

        # Reshape back to long format efficiently
        count_df = pivot_df.reset_index().melt(
            id_vars="plate_id", value_name="count", var_name=condition_col
        )
        norm_df = normalized_df.reset_index().melt(
            id_vars="plate_id",
            value_name="normalized_count",
            var_name=condition_col,
        )

        # Merge and return
        return pd.merge(
            count_df,
            norm_df[["plate_id", condition_col, "normalized_count"]],
            on=["plate_id", condition_col],
            how="left",
        )

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

    def _build_plot(
        self, data: pd.DataFrame, conditions: list[str], condition_col: str
    ) -> None:
        """Build the count plot with bars and points."""
        # Determine which column to plot
        count_col = (
            "normalized_count"
            if self.config.plot_type == PlotType.NORMALISED
            else "count"
        )

        # Get x positions for grouping
        x_positions = (
            grouped_x_positions(
                len(conditions),
                group_size=self.config.group_size,
                within_group_spacing=self.config.within_group_spacing,
                between_group_gap=self.config.between_group_gap,
            )
            if self.config.group_size > 1
            else [float(i) for i in range(len(conditions))]
        )

        # Create bars with proper positioning
        if self.config.group_size > 1:
            # Manual bar creation for grouped layout
            for idx, condition in enumerate(conditions):
                cond_data = data[data[condition_col] == condition]
                if not cond_data.empty:
                    assert self.ax is not None
                    self.ax.bar(
                        x_positions[idx],
                        cond_data[count_col].mean(),
                        width=0.6,
                        color=COLOR.BLUE.value,
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.8,
                    )
        else:
            # Use seaborn for standard layout
            sns.barplot(
                data=data,
                x=condition_col,
                y=count_col,
                order=conditions,
                color=COLOR.BLUE.value,
                ax=self.ax,
            )

        # Add individual points
        assert self.ax is not None
        show_repeat_points_adaptive(
            data,
            conditions,
            condition_col,
            count_col,
            self.ax,
            self.config.group_size,
            x_positions,
        )

        # Add significance marks if enough replicates
        if data.plate_id.nunique() >= 3:
            assert self.ax is not None
            set_significance_marks_adaptive(
                self.ax,
                data,
                conditions,
                condition_col,
                count_col,
                self.ax.get_ylim()[1],
                group_size=self.config.group_size,
                x_positions=x_positions,
            )

        # Format axes
        self._format_axes(conditions, count_col, x_positions)

    def _format_axes(
        self, conditions: list[str], count_col: str, x_positions: list[float]
    ) -> None:
        """Format axes labels and ticks."""
        # Set y-axis label
        assert self.ax is not None
        self.ax.set_ylabel(count_col.replace("_", " ").title())
        self.ax.set_xlabel("")

        # Set x-axis ticks
        self.ax.set_xticks(x_positions)

        if self.config.show_x_labels:
            self.ax.set_xticklabels(
                conditions, rotation=self.config.rotation, ha="right"
            )
        else:
            self.ax.set_xticklabels([])

    def _finalize_plot(self, selector_val: str | None) -> None:
        """Finalize plot with title."""
        # Generate default title using class attribute
        default_title = (
            f"{self.PLOT_TYPE_NAME}s {selector_val}"
            if selector_val
            else f"{self.PLOT_TYPE_NAME}s"
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
                self._filename or "countplot",
                tight_layout=self.config.tight_layout,
                fig_extension=self.config.file_format,
                resolution=self.config.dpi,
            )
