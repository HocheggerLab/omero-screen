"""Standard feature plot implementation.

This module provides the FeaturePlot class that creates individual box or violin plots
for a feature across different conditions, with optional significance testing.
"""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...stats import set_significance_marks

# No longer need select_datapoints import
from .base import BaseFeaturePlot


class StandardFeaturePlot(BaseFeaturePlot):
    """Standard feature plot with box or violin plots.

    Creates individual plots for each condition showing:
    - Box plot or violin plot of feature distribution
    - Individual data points overlaid (optional)
    - Statistical significance markers (if sufficient replicates)

    This provides a detailed view of feature distributions across conditions.
    """

    @property
    def plot_type(self) -> str:
        """Return the type of plot."""
        return "feature_standard"

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        feature: str,
        violin: bool = False,
        show_points: bool = False,
        show_replicates: bool = True,
        show_significance: bool = True,
        show_xlabels: bool = True,
        ymax: Optional[Union[float, tuple[float, float]]] = None,
        legend: Optional[tuple[str, list[str]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize standard feature plot.

        Args:
            data: DataFrame containing feature data
            conditions: List of conditions to plot
            feature: Column name of the feature to plot
            violin: Whether to use violin plot instead of box plot
            show_points: Whether to show individual data points (default: False)
            show_replicates: Whether to show replicate median points (default: True)
            show_significance: Whether to show significance markers
            show_xlabels: Whether to show condition names as x-axis labels (default: True)
            ymax: Y-axis maximum value. Can be a single value or tuple (min, max)
            legend: Optional legend configuration as (title, labels)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(data, conditions, feature, **kwargs)

        self.violin = violin
        self.show_points = show_points
        self.show_replicates = show_replicates
        self.show_significance = show_significance
        self.show_xlabels = show_xlabels
        self.ymax = ymax
        self.legend = legend

    def generate(self) -> Figure:
        """Generate the feature plot.

        Returns:
            Figure containing the plot

        Raises:
            ValueError: If ax parameter was provided (not supported for multi-condition plots)
        """
        # Check if we're creating a new figure or using provided axis
        if len(self.conditions) > 1 and self.ax is not None:
            raise ValueError(
                "Cannot use provided axis for multiple conditions. "
                "Use GroupedFeaturePlot for single-axis integration."
            )

        # For single condition, we can use provided axis
        if len(self.conditions) == 1 and self.ax is not None:
            self._plot_condition(
                self.ax, self.conditions[0], 0, is_single_plot=True
            )

            # Apply axis-level title from base class
            self._apply_title()

            fig = self.ax.figure
            if fig is not None:
                from matplotlib.figure import Figure

                if isinstance(fig, Figure):
                    return fig
            raise ValueError("No figure available")

        # Create new figure for multiple conditions
        n_conditions = len(self.conditions)
        self.fig, axes = plt.subplots(
            1, n_conditions, figsize=self.figsize, sharey=True
        )

        # Handle single vs multiple subplots
        if n_conditions == 1:
            axes = [axes]

        # Plot each condition
        for i, (ax, condition) in enumerate(
            zip(axes, self.conditions, strict=False)
        ):
            self._plot_condition(ax, condition, i, is_single_plot=False)

        # Apply figure-level title using base class helper
        self._apply_figure_title()

        # Add replicate points if requested
        if self.show_replicates:
            self._add_replicate_points(axes)

        # Add significance markers if requested
        if self.show_significance and len(self.conditions) > 1:
            self._add_multi_condition_significance(axes)

        # Add legend if specified or if showing replicates (for multi-condition plots)
        if self.legend:
            self._add_legend(axes[-1])
        elif self.show_replicates and len(self.conditions) > 1:
            self._add_replicate_legend(axes[-1])

        # Adjust layout
        plt.tight_layout()

        return self.fig

    def _plot_condition(
        self,
        ax: Axes,
        condition: str,
        condition_index: int,
        is_single_plot: bool = False,
    ) -> None:
        """Plot data for a single condition.

        Args:
            ax: Matplotlib axis to plot on
            condition: Condition name
            condition_index: Index of the condition (for styling)
            is_single_plot: Whether this is a standalone plot (not part of multi-condition figure)
        """
        # Get data for this condition
        condition_data = self.get_feature_data(condition)

        if condition_data.empty:
            ax.text(
                0.5,
                0.5,
                f"No data for {condition}"
                if is_single_plot
                else f"No data\n{condition}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            if not is_single_plot:
                ax.set_title(condition, fontsize=6)
            return

        # Prepare data for plotting
        plot_data = condition_data[self.feature].dropna().tolist()

        # Create plot
        self._create_plot(ax, plot_data, condition_index)

        # Add significance markers if requested (only for multi-condition plots)
        if (
            self.show_significance
            and condition_index > 0
            and not is_single_plot
        ):
            self._add_significance(ax, condition_index)

        # Setup axis - let setup_feature_axis handle y-axis for all cases
        ax.set_xticks([0])
        if self.show_xlabels:
            fontsize = None if is_single_plot else 6
            ax.set_xticklabels(
                [condition], rotation=45, ha="right", fontsize=fontsize
            )
        else:
            ax.set_xticklabels([])
        ax.set_xlabel("")

        # Setup y-axis - only show y-label on leftmost plot for multi-condition
        show_ylabel = is_single_plot or condition_index == 0
        if show_ylabel:
            self.setup_feature_axis(ax, ylim=self.ymax)
        else:
            # For non-leftmost plots, just set limits and remove ylabel
            self.setup_feature_axis(ax, ylim=self.ymax, ylabel="")

        # Add replicate legend if this is a single plot
        if is_single_plot and self.show_replicates and not self.legend:
            self._add_replicate_legend(ax)

    def _create_plot(
        self, ax: Axes, data: list[float], color_index: int
    ) -> None:
        """Create the actual box or violin plot.

        Args:
            ax: Matplotlib axis
            data: Data points to plot
            color_index: Index for color selection
        """
        # Select color
        color = self.colors[color_index % len(self.colors)]

        # Position for plot
        x_pos = 0

        if self.violin:
            # Create violin plot
            parts = ax.violinplot(
                [data],
                positions=[x_pos],
                showmeans=True,
                showmedians=True,
            )

            # Style violin plot
            bodies: Any = parts.get("bodies", [])
            if hasattr(bodies, "__iter__"):
                for pc in bodies:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)

            for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
                if partname in parts:
                    parts[partname].set_color(color)
                    parts[partname].set_linewidth(1)
        else:
            # Create box plot
            bp = ax.boxplot(
                [data],
                positions=[x_pos],
                patch_artist=True,
                showfliers=False,
            )

            # Style box plot
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)

            for element in ["whiskers", "caps", "medians"]:
                for item in bp[element]:
                    item.set_color(color)
                    item.set_linewidth(1)

        # Add individual points if requested (limit to reasonable number)
        if self.show_points:
            import numpy as np

            # Limit to max 100 points to avoid visual clutter
            max_points = 100
            if len(data) > max_points:
                # Sample random points
                np.random.seed(42)  # For reproducibility
                indices = np.random.choice(
                    len(data), max_points, replace=False
                )
                points_to_show = [data[i] for i in sorted(indices)]  # type: ignore
            else:
                points_to_show = data

            n_points = len(points_to_show)
            jitter = np.random.normal(0, 0.04, n_points)
            x_points = np.full(n_points, x_pos) + jitter

            ax.scatter(
                x_points,
                points_to_show,
                color="black",
                s=8,
                alpha=0.6,
                zorder=10,
            )

        # Set x-axis properties
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([x_pos])

    def _add_significance(self, ax: Axes, condition_index: int) -> None:
        """Add significance markers comparing to first condition.

        Args:
            ax: Matplotlib axis
            condition_index: Index of current condition
        """
        try:
            # Check if we have plate_id column for replicate-based statistics
            if "plate_id" not in self.data.columns:
                return

            # Calculate median per plate_id per condition (same as legacy code)
            df_median = (
                self.data.groupby(["plate_id", self.condition_col])[
                    self.feature
                ]
                .median()
                .reset_index()
            )

            # Get y-max for significance markers
            y_max = ax.get_ylim()[1]

            # Use the stats module function with median data (like legacy code)
            set_significance_marks(
                ax,
                df_median,
                [self.conditions[0], self.conditions[condition_index]],
                self.condition_col,
                self.feature,
                y_max,
            )
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not add significance markers: {e}")

    def _add_multi_condition_significance(self, axes: list[Axes]) -> None:
        """Add significance markers comparing conditions across multiple plots.

        Args:
            axes: List of matplotlib axes
        """
        try:
            # Check if we have plate_id column for replicate-based statistics
            if "plate_id" not in self.data.columns:
                return

            # Calculate median per plate_id per condition (same as legacy code)
            df_median = (
                self.data.groupby(["plate_id", self.condition_col])[
                    self.feature
                ]
                .median()
                .reset_index()
            )

            from ...stats import calculate_pvalues, get_significance_marker

            # Calculate p-values comparing each condition to the first using median data
            pvalues = calculate_pvalues(
                df_median, self.conditions, self.condition_col, self.feature
            )

            # Add significance markers to each condition plot (skip first)
            for ax, p_value in zip(axes[1:], pvalues, strict=False):
                significance = get_significance_marker(p_value)

                # Get y-max for positioning - use same approach as legacy code
                y_max = ax.get_ylim()[1]

                # Add significance marker at top center of plot
                # Position at y_max like the legacy set_significance_marks function
                ax.annotate(
                    significance,
                    xy=(0, y_max),
                    xycoords="data",
                    ha="center",
                    va="bottom",
                    fontsize=6,  # Match legacy fontsize
                    fontweight="bold",
                )

        except (ValueError, KeyError, ImportError) as e:
            print(
                f"Warning: Could not add multi-condition significance markers: {e}"
            )

    def _add_replicate_points(self, axes: list[Axes]) -> None:
        """Add replicate median points to all axes.

        Args:
            axes: List of matplotlib axes
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

        # Add points to each axis
        for ax, condition in zip(axes, self.conditions, strict=False):
            cond_medians = df_median[
                df_median[self.condition_col] == condition
            ]

            if cond_medians.empty:
                continue

            # Position at center of each condition's plot
            x_base = 0  # Center position for single condition plots

            # Add small jitter to avoid overlapping points
            import numpy as np

            jitter = np.random.uniform(-0.03, 0.03, size=len(cond_medians))
            x = np.full(len(cond_medians), x_base) + jitter
            y = cond_medians[self.feature].values

            # Plot each replicate with its marker
            for xi, yi, pid in zip(
                x, y, cond_medians["plate_id"], strict=False
            ):
                marker = plate_id_to_marker[pid]
                ax.scatter(
                    xi,
                    yi,
                    color="black",
                    edgecolor="white",
                    s=18,
                    zorder=4,
                    marker=marker,
                )

    def _add_replicate_legend(self, ax: Axes) -> None:
        """Add legend for replicate markers.

        Args:
            ax: Matplotlib axis to add legend to
        """
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
        ax.legend(
            handles=handles,
            title="median",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            fontsize=6,
            title_fontsize=6,
        )

    def _add_legend(self, ax: Axes) -> None:
        """Add legend to the plot.

        Args:
            ax: Axis to add legend to
        """
        if self.legend:
            title, labels = self.legend
            # Create proxy artists for legend
            from matplotlib.patches import Patch

            colors_to_use = self.colors[: len(labels)]
            handles = [
                Patch(color=color, label=label)
                for color, label in zip(colors_to_use, labels, strict=False)
            ]

            ax.legend(
                handles,
                labels,
                title=title,
                loc="upper right",
                bbox_to_anchor=(1.2, 1),
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
        """Save the feature plot.

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
            filename = f"feature_{plot_type}_{self.feature}{selector_part}.pdf"

        super().save(path, filename, tight_layout=tight_layout, **kwargs)


def feature_plot(
    data: pd.DataFrame,
    feature: str,
    conditions: list[str],
    # Base class arguments
    ymax: Optional[Union[float, tuple[float, float]]] = None,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = "",
    title: Optional[str] = "",
    colors: Optional[list[str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    scale: bool = False,
    # FeaturePlot specific arguments
    legend: Optional[tuple[str, list[str]]] = None,
    violin: bool = False,
    show_replicates: bool = True,
    show_significance: bool = False,
    show_xlabels: bool = True,
    # Output arguments
    save: bool = True,
    path: Optional[Path] = None,
) -> Figure:
    """Create a standard feature plot showing distributions across conditions.

    This is the main user-facing function for creating feature plots.
    It creates box plots or violin plots for a single feature across multiple conditions.

    Args:
        data: DataFrame containing the data with feature and condition columns
        feature: Name of the feature column to plot
        conditions: List of conditions to include in the plot

        # Data filtering and processing
        ymax: Maximum y-axis value. Can be a single value or tuple (min, max)
        condition_col: Name of the column containing conditions
        selector_col: Optional column for additional filtering
        selector_val: Value to filter by in selector_col
        scale: Whether to scale the feature data

        # Plot appearance
        title: Plot title. If empty, auto-generated from feature name
        colors: List of colors to use. If None, uses default palette
        figsize: Figure size as (width, height) in inches. If None, auto-calculated
        legend: Optional legend as (title, list_of_labels)
        violin: If True, creates violin plots instead of box plots
        show_replicates: Whether to show replicate median points (default: True)
        show_significance: Whether to show statistical significance markers (default: False)
        show_xlabels: Whether to show condition names as x-axis labels (default: True)

        # Output options
        save: Whether to save the figure
        path: Directory to save the figure. Required if save=True

    Returns:
        matplotlib.figure.Figure: The generated figure

    Examples:
        Basic box plot:
        >>> fig = feature_plot(
        ...     data=df,
        ...     feature='intensity_mean_GFP',
        ...     conditions=['Control', 'Treatment'],
        ...     selector_col='cell_line',
        ...     selector_val='HeLa'
        ... )

        Violin plot with scaling and custom size:
        >>> fig = feature_plot(
        ...     data=df,
        ...     feature='area_nucleus',
        ...     conditions=['DMSO', 'Drug1', 'Drug2'],
        ...     violin=True,
        ...     scale=True,
        ...     ymax=2.0,
        ...     figsize=(8, 3)
        ... )

        With custom styling and legend:
        >>> fig = feature_plot(
        ...     data=df,
        ...     feature='intensity_integrated_EdU',
        ...     conditions=['0h', '24h', '48h'],
        ...     colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
        ...     legend=('Time', ['0 hours', '24 hours', '48 hours']),
        ...     title='EdU Incorporation Over Time'
        ... )
    """
    # Auto-generate title if not provided
    if not title:
        title = f"{feature.replace('_', ' ').title()}"
        if selector_val:
            title += f" - {selector_val}"

    # Use provided figsize or calculate based on number of conditions
    if figsize is None:
        width = len(conditions) * 1.5 + 0.5
        height = 1.5  # Default height
        figsize = (width, height)

    # Set default color to PURPLE if not provided
    if colors is None:
        from ...colors import COLOR

        colors = [COLOR.PURPLE.value]

    # Create the plot instance
    plot = StandardFeaturePlot(
        data=data,
        conditions=conditions,
        feature=feature,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        title=title,
        colors=colors,
        figsize=figsize,
        scale=scale,
        violin=violin,
        show_replicates=show_replicates,
        show_significance=show_significance,
        show_xlabels=show_xlabels,
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

    return fig
