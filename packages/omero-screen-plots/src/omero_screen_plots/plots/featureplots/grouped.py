"""Grouped feature plot implementation.

This module provides the GroupedFeaturePlot class that creates grouped box or violin plots
for a feature, allowing comparison of conditions within visual groups.
"""

from pathlib import Path
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...featureplot import draw_violin_or_box
from ...stats import set_grouped_significance_marks
from .base import BaseFeaturePlot


class GroupedFeaturePlot(BaseFeaturePlot):
    """Grouped feature plot with conditions organized in visual groups.

    Creates box or violin plots grouped visually to facilitate comparison
    within groups while maintaining overview across all conditions.
    """

    @property
    def plot_type(self) -> str:
        """Return the type of plot."""
        return "feature_grouped"

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        feature: str,
        group_size: int = 2,
        within_group_spacing: float = 0.5,
        between_group_gap: float = 0.75,
        violin: bool = False,
        color: str = "#B19CD9",  # Default lavender color
        show_replicates: bool = True,
        show_significance: bool = True,
        legend: Optional[tuple[str, list[str]]] = None,
        x_label: bool = True,
        y_label: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize grouped feature plot.

        Args:
            data: DataFrame containing feature data
            conditions: List of conditions to plot
            feature: Column name of the feature to plot
            group_size: Number of conditions per visual group
            within_group_spacing: Spacing between conditions within a group
            between_group_gap: Gap between groups
            violin: Whether to use violin plot instead of box plot
            color: Color for the plots
            show_replicates: Whether to show replicate median points with distinct markers
            show_significance: Whether to show significance markers
            legend: Optional legend configuration
            x_label: Whether to show x-axis labels
            y_label: Custom y-axis label
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(data, conditions, feature, **kwargs)

        self.group_size = group_size
        self.within_group_spacing = within_group_spacing
        self.between_group_gap = between_group_gap
        self.violin = violin
        self.color = color
        self.show_replicates = show_replicates
        self.show_significance = show_significance
        self.legend = legend
        self.x_label = x_label
        self.y_label = y_label

        # Validate group size
        if group_size < 1:
            raise ValueError("Group size must be at least 1")

    def generate(self) -> Figure:
        """Generate the grouped feature plot.

        Returns:
            Figure containing the plot
        """
        # Create new figure with proper figsize

        # Setup figure and axis - directly like standard feature plot
        if self.ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.fig = fig
            self.ax = ax
        else:
            self.fig = cast(Figure, self.ax.figure)

        # Calculate group positions using the user-defined spacing parameters
        x_positions = self.get_grouped_positions(
            self.group_size,
            within_group_spacing=self.within_group_spacing,
            between_group_gap=self.between_group_gap,
        )
        # Store for later use (e.g. significance markers)
        self._x_positions = x_positions

        # Plot each condition
        for i, (condition, x_pos) in enumerate(
            zip(self.conditions, x_positions, strict=False)
        ):
            if self.ax is not None:
                self._plot_condition(self.ax, condition, x_pos, i)

        # Setup axes
        self._configure_axes(x_positions)

        # Add replicate median points if requested
        if self.show_replicates:
            self._add_replicate_points(x_positions)

        # Add significance markers if requested
        if self.show_significance and self._has_sufficient_replicates():
            self._add_significance_markers()

        # Add legend if specified
        if self.legend:
            self._add_legend()

        # Add replicate legend if showing replicates
        if self.show_replicates and self.ax is not None:
            self._add_replicate_legend()

        # Apply title using base class helper
        self._apply_figure_title()

        if self.fig is not None:
            return self.fig
        raise ValueError("No figure available")

    def _plot_condition(
        self,
        ax: Axes,
        condition: str,
        x_position: float,
        condition_index: int,
    ) -> None:
        """Plot data for a single condition.

        Args:
            ax: Matplotlib axis
            condition: Condition name
            x_position: X-axis position for this condition
            condition_index: Index of the condition
        """
        # Get data for this condition
        condition_data = self.get_feature_data(condition)

        if condition_data.empty:
            # Add a text marker for empty data
            ax.text(
                x_position,
                ax.get_ylim()[0],
                "No data",
                ha="center",
                va="bottom",
                fontsize=5,
                rotation=45,
            )
            return

        # Select data points
        data_points = condition_data[self.feature].dropna().tolist()

        # Draw the plot
        draw_violin_or_box(
            ax, data_points, x_position, self.color, self.violin
        )

    def _configure_axes(self, x_positions: list[float]) -> None:
        """Configure plot axes.

        Args:
            x_positions: X-positions for all conditions
        """
        # Setup x-axis
        if self.ax is not None:
            self.setup_grouped_x_axis(self.ax, x_positions, self.x_label)

            # Setup y-axis
            if self.y_label is not None:
                ylabel = self.y_label
            else:
                ylabel = self.format_feature_label()

            self.setup_feature_axis(
                self.ax, ylabel=ylabel, ylim=getattr(self, "ymax", None)
            )

        # Add vertical lines to separate groups
        self._add_group_separators(x_positions)

    def _add_replicate_points(self, x_positions: list[float]) -> None:
        """Add replicate median points to the plot.

        Args:
            x_positions: X-positions for all conditions
        """
        # Check if we have plate_id column for replicates
        if "plate_id" not in self.data.columns:
            return

        # Calculate median per plate_id per condition
        df_median = (
            self.data.groupby(["plate_id", self.condition_col])[self.feature]
            .median()
            .reset_index()
        )

        # Define markers for different replicates
        markers = ["o", "s", "^"]  # circle, square, triangle
        plate_ids = sorted(df_median["plate_id"].unique())
        plate_id_to_marker = {
            pid: markers[i % len(markers)] for i, pid in enumerate(plate_ids)
        }

        # Add points for each condition
        for condition, x_pos in zip(
            self.conditions, x_positions, strict=False
        ):
            cond_medians = df_median[
                df_median[self.condition_col] == condition
            ]

            if cond_medians.empty:
                continue

            # Add small jitter to avoid overlapping points
            jitter = np.random.uniform(-0.05, 0.05, size=len(cond_medians))
            x = np.full(len(cond_medians), x_pos) + jitter
            y = cond_medians[self.feature].values

            # Plot each replicate with its marker
            for xi, yi, pid in zip(
                x, y, cond_medians["plate_id"], strict=False
            ):
                marker = plate_id_to_marker[pid]
                if self.ax is not None:
                    self.ax.scatter(
                        xi,
                        yi,
                        color="black",
                        edgecolor="white",
                        s=20,
                        zorder=5,
                        marker=marker,
                    )

    def _add_replicate_legend(self) -> None:
        """Add legend for replicate markers."""
        from matplotlib.lines import Line2D

        # Check if we have plate_id data
        if "plate_id" not in self.data.columns:
            return

        # Get unique plate IDs
        plate_ids = sorted(self.data["plate_id"].unique())
        n_replicates = len(plate_ids)

        if n_replicates == 0:
            return

        # Define markers and labels
        markers = ["o", "s", "^"]  # circle, square, triangle
        marker_labels = ["rep1", "rep2", "rep3"]

        # Create legend handles for available replicates
        handles = []
        for i in range(min(n_replicates, len(markers))):
            handle = Line2D(
                [0],
                [0],
                marker=markers[i],
                color="black",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=6,
                linestyle="None",
                label=marker_labels[i],
            )
            handles.append(handle)

        # Add legend
        if self.ax is not None:
            self.ax.legend(
                handles=handles,
                title="median",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
                fontsize=6,
                title_fontsize=6,
            )

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

    def _add_significance_markers(self) -> None:
        """Add statistical significance markers between conditions."""
        try:
            if self.ax is not None:
                y_max = self.ax.get_ylim()[1]

                # Recompute x-positions in case they are not stored (e.g. when
                # this method is called independently). Fall back to stored
                # value if available.
                x_positions = getattr(self, "_x_positions", None)
                if x_positions is None:
                    x_positions = self.get_grouped_positions(
                        self.group_size,
                        within_group_spacing=self.within_group_spacing,
                        between_group_gap=self.between_group_gap,
                    )

                set_grouped_significance_marks(
                    self.ax,
                    self.data,
                    self.conditions,
                    self.condition_col,
                    self.feature,
                    y_max,
                    self.group_size,
                    x_positions=x_positions,
                )
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not add significance markers: {e}")

    def _has_sufficient_replicates(self) -> bool:
        """Check if there are sufficient replicates for statistics.

        Returns:
            True if sufficient replicates exist
        """
        # Check if we have plate_id column for replicates
        if "plate_id" not in self.data.columns:
            return False

        # Check number of unique plates
        n_replicates = self.data["plate_id"].nunique()
        return n_replicates >= 3  # type: ignore[no-any-return]

    def _add_legend(self) -> None:
        """Add legend to the plot."""
        if self.legend:
            title, labels = self.legend
            # Create proxy artists for legend
            from matplotlib.patches import Patch

            # Use single color for grouped plots
            handles = [
                Patch(color=self.color, label=label) for label in labels
            ]

            if self.ax is not None:
                self.ax.legend(
                    handles,
                    labels,
                    title=title,
                    loc="upper right",
                    bbox_to_anchor=(1.1, 1),
                    fontsize=6,
                    title_fontsize=6,
                )

    def save(
        self,
        path: Union[str, Path],
        filename: Optional[str] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the grouped feature plot.

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
            plot_type = "violin" if self.violin else "box"
            filename = f"feature_grouped_{plot_type}_{self.feature}{selector_part}.png"

        super().save(path, filename, tight_layout=tight_layout, **kwargs)


def grouped_feature_plot(
    ax: Optional[Axes],
    data: pd.DataFrame,
    feature: str,
    conditions: list[str],
    group_size: int = 2,
    within_group_spacing: float = 0.5,
    between_group_gap: float = 0.75,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = "",
    ymax: Optional[Union[float, tuple[float, float]]] = None,
    legend: Optional[tuple[str, list[str]]] = None,
    figsize: Optional[tuple[float, float]] = None,
    x_label: bool = True,
    y_label: Optional[str] = None,
    violin: bool = False,
    color: str = "#B19CD9",
    show_replicates: bool = True,
    title: Optional[str] = None,
    save: bool = False,
    path: Optional[Path] = None,
) -> Figure:
    """Create a grouped feature plot for comparing conditions in visual groups.

    This function creates box or violin plots with conditions grouped visually,
    making it easier to compare within groups while maintaining an overview.

    Args:
        ax: Optional matplotlib axis. If None, creates new figure
        data: DataFrame containing the data
        feature: Name of the feature column to plot
        conditions: List of conditions to include
        group_size: Number of conditions per visual group (default: 2)
        within_group_spacing: Spacing between conditions within a group
        between_group_gap: Gap between groups

        # Data filtering
        condition_col: Name of the condition column
        selector_col: Optional column for filtering
        selector_val: Value to filter by in selector_col

        # Plot configuration
        ymax: Maximum y-axis value or tuple (min, max)
        legend: Optional legend as (title, list_of_labels)
        figsize: Figure size as (width, height) if creating new figure
        x_label: Whether to show x-axis labels
        y_label: Custom y-axis label. If None, auto-generated
        violin: If True, creates violin plots instead of box plots
        color: Color for the plots
        show_replicates: Whether to show replicate median points with distinct markers
        title: Plot title

        # Output options
        save: Whether to save the figure
        path: Directory to save if save=True

    Examples:
        Create grouped box plots:
        >>> grouped_feature_plot(
        ...     ax=None,
        ...     data=df,
        ...     feature='intensity_mean_GFP',
        ...     conditions=['DMSO', 'Drug1', 'Drug2', 'Drug3'],
        ...     group_size=2,
        ...     title='GFP Expression by Treatment Group',
        ...     figsize=(8, 4)
        ... )

        Integrate into existing figure:
        >>> fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        >>> grouped_feature_plot(
        ...     ax=ax,
        ...     data=df,
        ...     feature='area_nucleus',
        ...     conditions=['WT_ctrl', 'WT_treat', 'KO_ctrl', 'KO_treat'],
        ...     group_size=2,
        ...     color='#2E8B57'
        ... )

        With violin plots and custom styling:
        >>> grouped_feature_plot(
        ...     ax=None,
        ...     data=df,
        ...     feature='EdU_intensity',
        ...     conditions=['0h', '6h', '12h', '24h', '48h', '72h'],
        ...     group_size=3,
        ...     violin=True,
        ...     y_label='EdU Intensity (a.u.)',
        ...     legend=('Time', ['Early', 'Mid', 'Late'])
        ... )
    """
    # Use provided figsize or calculate based on number of conditions - same as standard
    if figsize is None:
        width = len(conditions) * 1.5 + 0.5
        height = 1.5  # Default height
        figsize = (width, height)

    # Create plot instance
    plot = GroupedFeaturePlot(
        data=data,
        conditions=conditions,
        feature=feature,
        group_size=group_size,
        within_group_spacing=within_group_spacing,
        between_group_gap=between_group_gap,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        ax=ax,
        title=title,
        colors=[color],  # Pass as list for base class compatibility
        figsize=figsize,  # Pass through to base class
        violin=violin,
        color=color,
        show_replicates=show_replicates,
        x_label=x_label,
        y_label=y_label,
        legend=legend,
        ymax=ymax,
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
