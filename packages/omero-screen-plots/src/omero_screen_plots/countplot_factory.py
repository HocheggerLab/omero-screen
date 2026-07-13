"""Simplified count plot implementation with single class architecture."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.base import BasePlotBuilder, BasePlotConfig
from omero_screen_plots.colors import COLOR
from omero_screen_plots.stats import (
    annotate_significance,
    compute_significance,
)
from omero_screen_plots.utils import (
    finalize_plot_with_title,
    grouped_x_positions,
    show_repeat_points_adaptive,
)


class PlotType(Enum):
    """Enumeration for plot types: normalised and absolute."""

    NORMALISED = "normalised"
    ABSOLUTE = "absolute"


@dataclass
class CountPlotConfig(BasePlotConfig):
    """Configuration for count plots.

    Figure/save/display fields are inherited unchanged from ``BasePlotConfig``.
    """

    # Count plot specific settings
    plot_type: PlotType = PlotType.NORMALISED
    show_triplicates: bool = False
    group_size: int = 1
    within_group_spacing: float = 0.2
    between_group_gap: float = 0.5
    show_x_labels: bool = True
    rotation: int = 45

    # Triplicate-specific settings (kept in sync with the feature and
    # cell-cycle plots so triplicate bars look identical across plot types).
    repeat_offset: float = 0.18
    max_repeats: int = 3
    bar_alpha: float = 0.9
    bar_edgecolor: str = "white"
    bar_linewidth: float = 0.2
    show_boxes: bool = True
    box_linewidth: float = 0.5
    box_color: str = "black"


class CountPlot(BasePlotBuilder):
    """Simplified count plot implementation combining config, processing, and plotting."""

    PLOT_TYPE_NAME = "count"
    REQUIRED_COLUMNS = ("plate_id", "well", "experiment")
    config: CountPlotConfig  # Type annotation for mypy

    def __init__(self, config: CountPlotConfig | None = None):
        """Initialize with configuration."""
        super().__init__(config or CountPlotConfig())

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

        # Create figure (base handles axes-provided vs new figure + the flag)
        self.create_figure(axes)

        # Build plot
        self.build_plot(
            processed_data, conditions=conditions, condition_col=condition_col
        )

        # Finalize
        self._finalize_plot(selector_val)

        # Save if configured
        self._save_plot()

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
        # Count-specific: normalization control must be one of the conditions.
        # Checked before the selector checks to match the historical ordering.
        if norm_control not in conditions:
            raise ValueError(
                f"Normalization control '{norm_control}' must be in conditions list: {conditions}"
            )

        self._validate_common(
            df, conditions, condition_col, selector_col, selector_val
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

    def build_plot(
        self, data: pd.DataFrame, **kwargs: Any
    ) -> "BasePlotBuilder":
        """Build the count plot with bars and points."""
        # Extract required parameters from kwargs
        conditions = kwargs.get("conditions")
        condition_col = kwargs.get("condition_col")
        assert conditions is not None and condition_col is not None
        # Determine which column to plot
        count_col = (
            "normalized_count"
            if self.config.plot_type == PlotType.NORMALISED
            else "count"
        )

        # Get x positions for grouping. Use grouped_x_positions whenever bars
        # are drawn manually (grouped, or triplicate mode) so between_group_gap
        # controls the spacing; fall back to integer positions only for the
        # seaborn-drawn single-bar layout (group_size==1, no triplicates), which
        # places bars categorically and must stay aligned with the overlays.
        x_positions = (
            grouped_x_positions(
                len(conditions),
                group_size=self.config.group_size,
                within_group_spacing=self.config.within_group_spacing,
                between_group_gap=self.config.between_group_gap,
            )
            if self.config.group_size > 1 or self.config.show_triplicates
            else [float(i) for i in range(len(conditions))]
        )

        assert self.ax is not None

        if self.config.show_triplicates:
            self._build_triplicate_bars(
                data, conditions, condition_col, count_col, x_positions
            )
        elif self.config.group_size > 1:
            # Manual bar creation for grouped layout
            for idx, condition in enumerate(conditions):
                cond_data = data[data[condition_col] == condition]
                if not cond_data.empty:
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

        # Add individual repeat points (mean bar modes only)
        if not self.config.show_triplicates:
            show_repeat_points_adaptive(
                data,
                conditions,
                condition_col,
                count_col,
                self.ax,
                self.config.group_size,
                x_positions,
            )

        # Add significance marks if enough replicates (both mean-bar and
        # triplicate modes; stars sit above each condition's group centre).
        if data.plate_id.nunique() >= 3:
            medians_df, results = compute_significance(
                data,
                conditions,
                condition_col,
                count_col,
                group_size=self.config.group_size,
                paired=self.config.paired,
            )
            annotate_significance(
                self.ax, results, x_positions, self.ax.get_ylim()[1] * 0.93
            )
            self._record_stats(
                medians_df,
                results,
                value_label=count_col.replace("_", " ").title(),
                condition_col=condition_col,
                value_col=count_col,
            )

        # Format axes
        self._format_axes(conditions, count_col, x_positions)

        return self

    def _build_triplicate_bars(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        count_col: str,
        x_positions: list[float],
    ) -> None:
        """Draw one bar per replicate plate, matching the feature/cell-cycle plots.

        All replicate bars share a single colour and touch each other within a
        condition (``repeat_offset`` spacing, ``repeat_offset`` width); the
        replicates are grouped by a surrounding box rather than by a per-plate
        alpha gradient, so the layout is identical to the other triplicate plots.
        """
        assert self.ax is not None
        repeat_offset = self.config.repeat_offset
        max_repeats = self.config.max_repeats

        # Available plates, capped at max_repeats for consistent spacing.
        plate_ids = sorted(data["plate_id"].unique())[:max_repeats]

        for cond_idx, condition in enumerate(conditions):
            base_x = x_positions[cond_idx]
            rep_x_positions = self._repeat_x_positions(base_x)
            for rep_idx, plate_id in enumerate(plate_ids):
                cond_data = data[
                    (data[condition_col] == condition)
                    & (data["plate_id"] == plate_id)
                ]
                if not cond_data.empty:
                    self.ax.bar(
                        rep_x_positions[rep_idx],
                        cond_data[count_col].iloc[0],
                        width=repeat_offset,
                        color=COLOR.BLUE.value,
                        edgecolor=self.config.bar_edgecolor,
                        linewidth=self.config.bar_linewidth,
                        alpha=self.config.bar_alpha,
                    )

        # Draw boxes around each condition's triplicate group.
        if self.config.show_boxes and len(plate_ids) > 1:
            self._draw_triplicate_boxes(
                data, conditions, condition_col, count_col, x_positions
            )

    def _repeat_x_positions(self, base_x: float) -> list[float]:
        """Return the centred x positions for ``max_repeats`` replicate bars."""
        max_repeats = self.config.max_repeats
        if max_repeats == 1:
            return [base_x]
        return [
            base_x
            + (rep_idx - (max_repeats - 1) / 2) * self.config.repeat_offset
            for rep_idx in range(max_repeats)
        ]

    def _draw_triplicate_boxes(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        count_col: str,
        x_positions: list[float],
    ) -> None:
        """Draw a box around each condition's triplicate bars."""
        from matplotlib.patches import Rectangle

        assert self.ax is not None
        half_width = self.config.repeat_offset / 2
        for cond_idx, condition in enumerate(conditions):
            cond_data = data[data[condition_col] == condition]
            if cond_data.empty:
                continue
            rep_x_positions = self._repeat_x_positions(x_positions[cond_idx])
            left = min(rep_x_positions) - half_width
            right = max(rep_x_positions) + half_width
            height = cond_data[count_col].max()
            self.ax.add_patch(
                Rectangle(
                    (left, 0),
                    width=right - left,
                    height=height,
                    linewidth=self.config.box_linewidth,
                    edgecolor=self.config.box_color,
                    facecolor="none",
                    zorder=10,
                )
            )

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
            self.fig, title, default_title, self.axes_provided
        )

    def _save_plot(self) -> None:
        """Save figure using parent's save_figure method."""
        # Use parent's save_figure method with our filename
        self.save_figure(self._filename or "countplot")
