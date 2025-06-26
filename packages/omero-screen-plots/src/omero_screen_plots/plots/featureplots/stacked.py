"""Stacked threshold barplot implementation.

This module provides the StackedFeaturePlot class that creates stacked bar plots
showing the proportion of cells above/below a threshold for each condition, with visual grouping.
"""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...colors import COLOR
from .base import BaseFeaturePlot


class StackedFeaturePlot(BaseFeaturePlot):
    """Stacked barplot showing threshold-based proportions.

    Creates stacked bar plots showing the percentage of cells above/below
    a threshold value for each condition, with conditions organized in visual groups.
    """

    @property
    def plot_type(self) -> str:
        """Return the type of plot."""
        return "feature_threshold_stacked"

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        feature: str = "feature",
        threshold: float = 0.0,
        group_size: int = 2,
        within_group_spacing: float = 0.5,
        between_group_gap: float = 0.75,
        colors: Any = COLOR,
        repeat_offset: float = 0.18,
        x_label: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize grouped stacked threshold barplot.

        Args:
            data: DataFrame containing feature data
            conditions: List of conditions to plot
            feature: Column name of the feature to plot
            threshold: Threshold value for categorization
            group_size: Number of conditions per visual group
            within_group_spacing: Spacing between conditions within a group
            between_group_gap: Gap between groups
            colors: Color object or list for positive/negative categories
            repeat_offset: Offset for replicate bars
            x_label: Whether to show x-axis labels
            **kwargs: Additional arguments passed to base classes
        """
        # Initialize both parent classes
        super().__init__(data, conditions, feature, **kwargs)

        self.threshold = threshold
        self.group_size = group_size
        self.within_group_spacing = within_group_spacing
        self.between_group_gap = between_group_gap
        self.colors_config = colors
        self.repeat_offset = repeat_offset
        self.x_label = x_label

        # Validate parameters
        if group_size < 1:
            raise ValueError("Group size must be at least 1")

        # Set colors for positive/negative
        if hasattr(colors, "GREY") and hasattr(colors, "BLUE"):
            self.neg_color = colors.GREY.value
            self.pos_color = colors.BLUE.value
        else:
            # Fallback colors
            self.neg_color = "#808080"
            self.pos_color = "#0000FF"

    def generate(self) -> Figure:
        """Generate the grouped stacked threshold barplot.

        Returns:
            Figure containing the plot
        """
        # Setup figure and axis
        if self.ax is None:
            # Create new figure with subplot - this is where figsize gets applied
            from typing import cast

            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.fig = fig
            self.ax = ax
        else:
            # Use provided axis
            from typing import cast

            self.fig = cast(Figure, self.ax.figure)

        # Calculate threshold percentages for all conditions
        threshold_data = self._calculate_replicate_percentages()

        # Calculate group positions
        x_positions = self.get_grouped_positions(
            self.group_size,
            within_group_spacing=self.within_group_spacing,
            between_group_gap=self.between_group_gap,
        )
        self._x_positions = x_positions

        # Track bar positions for legend
        self.bar_positions: list[tuple[Any, Any]] = []

        # Plot each condition
        for i, (condition, base_x) in enumerate(
            zip(self.conditions, x_positions, strict=False)
        ):
            condition_data = threshold_data[
                threshold_data[self.condition_col] == condition
            ]
            if self.ax is not None:
                self._plot_condition_bars(self.ax, condition_data, base_x, i)

        # Configure axes
        self._configure_axes(x_positions)

        # Add legend
        self._add_legend()

        # Apply title using base class helper
        self._apply_title()

        if self.fig is not None:
            return self.fig
        raise ValueError("No figure available")

    def _calculate_replicate_percentages(self) -> pd.DataFrame:
        """Calculate percentages for each replicate separately.

        Returns:
            DataFrame with columns: condition, plate_id, category, percentage
        """
        results = []

        # Check if we have replicates
        has_replicates = "plate_id" in self.data.columns

        for condition in self.conditions:
            condition_data = self.get_feature_data(condition)

            if condition_data.empty:
                continue

            if has_replicates:
                # Calculate per replicate
                for plate_id in condition_data["plate_id"].unique():
                    plate_data = condition_data[
                        condition_data["plate_id"] == plate_id
                    ]
                    total_cells = len(plate_data)

                    if total_cells > 0:
                        above_threshold = (
                            plate_data[self.feature] > self.threshold
                        ).sum()
                        below_threshold = total_cells - above_threshold

                        results.append(
                            {
                                self.condition_col: condition,
                                "plate_id": plate_id,
                                "category": "pos",
                                "percentage": (above_threshold / total_cells)
                                * 100,
                            }
                        )
                        results.append(
                            {
                                self.condition_col: condition,
                                "plate_id": plate_id,
                                "category": "neg",
                                "percentage": (below_threshold / total_cells)
                                * 100,
                            }
                        )
            else:
                # No replicates, calculate overall
                total_cells = len(condition_data)
                if total_cells > 0:
                    above_threshold = (
                        condition_data[self.feature] > self.threshold
                    ).sum()
                    below_threshold = total_cells - above_threshold

                    results.append(
                        {
                            self.condition_col: condition,
                            "plate_id": "all",
                            "category": "pos",
                            "percentage": (above_threshold / total_cells)
                            * 100,
                        }
                    )
                    results.append(
                        {
                            self.condition_col: condition,
                            "plate_id": "all",
                            "category": "neg",
                            "percentage": (below_threshold / total_cells)
                            * 100,
                        }
                    )

        return pd.DataFrame(results)

    def _plot_condition_bars(
        self,
        ax: Axes,
        condition_data: pd.DataFrame,
        base_x: float,
        condition_index: int,
    ) -> None:
        """Plot stacked bars for a single condition.

        Args:
            ax: Matplotlib axis
            condition_data: Data for this condition
            base_x: Base x position for this condition
            condition_index: Index of the condition
        """
        if condition_data.empty:
            return

        # Get unique replicates
        replicates = condition_data["plate_id"].unique()
        n_replicates = len(replicates)

        # Calculate bar positions
        if n_replicates == 1:
            x_positions = [base_x]
            bar_width = 0.3
        else:
            # Spread replicates around base position
            offsets = np.linspace(
                -self.repeat_offset * (n_replicates - 1) / 2,
                self.repeat_offset * (n_replicates - 1) / 2,
                n_replicates,
            )
            x_positions = [base_x + offset for offset in offsets]
            bar_width = 0.15

        # Plot each replicate
        for rep_idx, (replicate, x_pos) in enumerate(
            zip(replicates, x_positions, strict=False)
        ):
            rep_data = condition_data[condition_data["plate_id"] == replicate]

            # Get positive and negative percentages
            pos_data = rep_data[rep_data["category"] == "pos"]
            neg_data = rep_data[rep_data["category"] == "neg"]

            pos_pct = (
                pos_data["percentage"].values[0] if len(pos_data) > 0 else 0
            )
            neg_pct = (
                neg_data["percentage"].values[0] if len(neg_data) > 0 else 0
            )

            # Create stacked bars
            # Bottom (negative) bar
            neg_bar = ax.bar(
                x_pos,
                neg_pct,
                width=bar_width,
                color=self.neg_color,
                edgecolor="black",
                linewidth=0.5,
            )

            # Top (positive) bar
            pos_bar = ax.bar(
                x_pos,
                pos_pct,
                bottom=neg_pct,
                width=bar_width,
                color=self.pos_color,
                edgecolor="black",
                linewidth=0.5,
            )

            # Store first bar of first condition for legend
            if condition_index == 0 and rep_idx == 0:
                self.bar_positions = [(neg_bar[0], pos_bar[0])]

    def _configure_axes(self, x_positions: list[float]) -> None:
        """Configure plot axes.

        Args:
            x_positions: X-positions for all conditions
        """
        # Setup x-axis
        if self.ax is not None:
            self.setup_grouped_x_axis(self.ax, x_positions, self.x_label)

            # Setup y-axis
            self.ax.set_ylabel("Percentage (%)", fontsize=6)
            self.ax.set_ylim(0, 105)  # Leave room at top

            # Add horizontal line at 50%
            self.ax.axhline(
                y=50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
            )

            # Add vertical lines to separate groups
            self._add_group_separators(x_positions)

            # General styling
            self.ax.tick_params(axis="both", which="major", labelsize=6)
            self.ax.grid(False)

    def _add_group_separators(self, x_positions: list[float]) -> None:
        """Add vertical lines to separate groups visually.

        Args:
            x_positions: X-positions for all conditions
        """
        n_groups = (
            len(self.conditions) + self.group_size - 1
        ) // self.group_size

        for i in range(1, n_groups):
            # Find position between groups
            last_in_prev_group = min(
                i * self.group_size - 1, len(x_positions) - 1
            )
            first_in_next_group = min(
                i * self.group_size, len(x_positions) - 1
            )

            if first_in_next_group < len(x_positions):
                separator_x = (
                    x_positions[last_in_prev_group]
                    + x_positions[first_in_next_group]
                ) / 2
            else:
                separator_x = x_positions[last_in_prev_group] + 0.5

            if self.ax is not None:
                self.ax.axvline(
                    x=separator_x,
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.5,
                )

    def _add_legend(self) -> None:
        """Add legend showing positive/negative categories."""
        # Create legend handles
        from matplotlib.patches import Patch

        handles = [
            Patch(
                color=self.pos_color,
                label=f"{self.feature} > {self.threshold}",
            ),
            Patch(
                color=self.neg_color,
                label=f"{self.feature} ≤ {self.threshold}",
            ),
        ]

        if self.ax is not None:
            self.ax.legend(
                handles,
                [str(handle.get_label()) for handle in handles],
                loc="upper right",
                bbox_to_anchor=(1.1, 1),
                fontsize=6,
            )

    def save(
        self,
        path: Union[str, Path],
        filename: Optional[str] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the threshold barplot.

        Args:
            path: Path to save location
            filename: Optional filename. If None, generates descriptive name
            tight_layout: Whether to apply tight layout
            **kwargs: Additional save parameters
        """
        if filename is None:
            # Generate descriptive filename
            selector_part = (
                f"_{self.selector_val}" if self.selector_val else ""
            )
            filename = (
                f"feature_threshold_stacked_{self.feature}{selector_part}.png"
            )

        super().save(path, filename, tight_layout=tight_layout, **kwargs)


def grouped_stacked_threshold_barplot(
    ax: Optional[Axes],
    data: pd.DataFrame,
    conditions: list[str],
    group_size: int = 2,
    within_group_spacing: float = 0.5,
    between_group_gap: float = 0.75,
    feature: str = "feature",
    threshold: float = 0.0,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    colors: Any = COLOR,
    repeat_offset: float = 0.18,
    figsize: tuple[float, float] = (3.5, 1.5),
    x_label: bool = True,
    title: Optional[str] = None,
    save: bool = True,
    path: Optional[Path] = None,
) -> Figure:
    """Create a grouped stacked barplot showing proportion above/below threshold.

    This function creates stacked bar plots where each bar shows the percentage
    of cells above (positive) and below (negative) a threshold value, with
    conditions organized in visual groups.

    Args:
        ax: Optional matplotlib axis. If None, creates new figure
        data: DataFrame containing the data
        conditions: List of conditions to include
        group_size: Number of conditions per visual group (default: 2)
        within_group_spacing: Spacing between conditions within a group
        between_group_gap: Gap between groups
        feature: Name of the feature column to threshold
        threshold: Threshold value for categorization (default: 0.0)

        # Data filtering
        condition_col: Name of the condition column
        selector_col: Optional column for filtering
        selector_val: Value to filter by in selector_col

        # Plot configuration
        colors: Color configuration object with GREY and BLUE attributes
        repeat_offset: Horizontal offset between replicate bars
        figsize: Figure size as (width, height) if creating new figure
        x_label: Whether to show x-axis labels
        title: Plot title

        # Output options
        save: Whether to save the figure
        path: Directory to save if save=True

    Examples:
        Basic threshold plot:
        >>> grouped_stacked_threshold_barplot(
        ...     ax=None,
        ...     data=df,
        ...     feature='GFP_intensity',
        ...     threshold=100,
        ...     conditions=['Control', 'Drug1', 'Drug2', 'Drug3'],
        ...     group_size=2
        ... )

        With custom threshold, title, and size:
        >>> grouped_stacked_threshold_barplot(
        ...     ax=None,
        ...     data=df,
        ...     feature='EdU_positive',
        ...     threshold=0.5,
        ...     conditions=['0h', '6h', '12h', '24h'],
        ...     title='EdU Incorporation Over Time',
        ...     group_size=2,
        ...     figsize=(8, 4)
        ... )

        Integrated into existing figure:
        >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
        >>> grouped_stacked_threshold_barplot(
        ...     ax=ax2,
        ...     data=df,
        ...     feature='mitotic_index',
        ...     threshold=0.05,
        ...     conditions=['DMSO', 'Taxol', 'Nocodazole'],
        ...     save=False
        ... )
    """
    # Create plot instance
    plot = StackedFeaturePlot(
        data=data,
        conditions=conditions,
        feature=feature,
        threshold=threshold,
        group_size=group_size,
        within_group_spacing=within_group_spacing,
        between_group_gap=between_group_gap,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        ax=ax,
        figsize=figsize,
        title=title,
        colors=colors,
        repeat_offset=repeat_offset,
        x_label=x_label,
    )

    # Generate the plot
    fig = plot.generate()

    # Save if requested
    if save:
        if path is None:
            raise ValueError("path is required when save=True")
        plot.save(path)

    # If using provided axis, don't close the figure
    if ax is None:
        plt.show()

    return fig
