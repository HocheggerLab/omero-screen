"""Grouped cell cycle plot implementation.

This module provides the CellCycleGroupedPlot class that creates a grouped stacked
bar plot showing individual replicates as separate bars within condition groups.
This provides the most detailed view of variability between replicates.
"""

from typing import List, Optional

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ...utils import grouped_x_positions
from .base import BaseCellCyclePlot


class CellCycleGroupedPlot(BaseCellCyclePlot):
    """Grouped stacked bar plot with individual replicates.

    Creates a grouped bar chart where:
    - Conditions are separated into groups along x-axis
    - Within each group, individual replicates are shown as separate bars
    - Each replicate bar is stacked by cell cycle phase
    - Groups are visually separated with boxes or spacing

    This visualization is ideal for:
    - Showing replicate-to-replicate variability
    - Identifying outlier replicates
    - Detailed analysis of experimental consistency
    """

    @property
    def plot_type(self) -> str:
        return "cellcycle_grouped"

    def __init__(
        self,
        data,
        conditions: List[str],
        phases: Optional[List[str]] = None,
        group_size: int = 2,
        n_repeats: int = 3,
        repeat_offset: float = 0.18,
        bar_width: Optional[float] = None,
        show_group_boxes: bool = True,
        show_legend: bool = True,
        **kwargs,
    ):
        """Initialize grouped cell cycle plot.

        Args:
            data: DataFrame containing cell cycle data
            conditions: List of conditions to plot
            phases: List of cell cycle phases
            group_size: Number of conditions per group
            n_repeats: Number of replicates to show per condition
            repeat_offset: Spacing between replicate bars
            bar_width: Width of individual bars (auto-calculated if None)
            show_group_boxes: Whether to draw boxes around condition groups
            show_legend: Whether to show phase legend
            **kwargs: Additional arguments passed to base class
        """
        # Default phases for grouped plot
        default_phases = ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]
        if phases is None:
            phases = default_phases

        super().__init__(data, conditions, phases=phases, **kwargs)

        self.group_size = group_size
        self.n_repeats = n_repeats
        self.repeat_offset = repeat_offset
        self.bar_width = bar_width or repeat_offset * 1.05
        self.show_group_boxes = show_group_boxes
        self.show_legend = show_legend

        # Get available replicates
        self.repeat_ids = self._get_repeat_ids()

    def _get_repeat_ids(self) -> List[str]:
        """Get list of available replicate IDs, limited to n_repeats."""
        available_ids = sorted(self.processed_data["plate_id"].unique())
        return available_ids[: self.n_repeats]

    def generate(self) -> Figure:
        """Generate the grouped stacked bar plot.

        Returns:
            Figure containing the grouped bar plot
        """
        # Setup figure
        self._setup_figure()

        # Create axis if we own the figure but don't have one
        if self.ax is None and self._owns_figure:
            self.ax = self.fig.add_subplot(111)

        # Calculate x positions for groups
        x_base_positions = grouped_x_positions(
            len(self.conditions),
            group_size=self.group_size,
            within_group_spacing=0.6,
            between_group_gap=0.7,
        )

        # Plot replicate bars
        self._plot_replicate_bars(x_base_positions)

        # Customize axes
        self._customize_axes(x_base_positions)

        # Draw group boxes if requested
        if self.show_group_boxes:
            self._draw_group_boxes(x_base_positions)

        # Add legend if requested
        if self.show_legend:
            self._add_phase_legend()

        # Apply title
        self._apply_title()

        return self.fig  # type: ignore

    def _plot_replicate_bars(self, x_base_positions: List[float]) -> None:
        """Plot stacked bars for each replicate.

        Args:
            x_base_positions: X-axis positions for each condition group
        """
        for cond_idx, condition in enumerate(self.conditions):
            for rep_idx, plate_id in enumerate(self.repeat_ids):
                # Calculate x position for this replicate
                x_pos = (
                    x_base_positions[cond_idx]
                    + (rep_idx - 1) * self.repeat_offset
                )

                # Get data for this condition and replicate
                replicate_data = self.processed_data[
                    (self.processed_data[self.condition_col] == condition)
                    & (self.processed_data["plate_id"] == plate_id)
                ]

                if not replicate_data.empty:
                    self._plot_single_replicate_bar(x_pos, replicate_data)

    def _plot_single_replicate_bar(self, x_pos: float, replicate_data) -> None:
        """Plot a stacked bar for a single replicate.

        Args:
            x_pos: X position for the bar
            replicate_data: Data for this specific replicate
        """
        assert self.ax is not None, "Axis is not set"
        # Convert to pivot table for easy access
        pivot = replicate_data.set_index("cell_cycle")["percent"]

        y_bottom = 0
        for i, phase in enumerate(self.phases):
            value = pivot.get(phase, 0)
            color = self.colors[i % len(self.colors)]

            self.ax.bar(
                x_pos,
                value,
                width=self.bar_width,
                bottom=y_bottom,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=phase
                if x_pos == 0
                else "",  # Only label first occurrence
            )
            y_bottom += value

    def _customize_axes(self, x_base_positions: List[float]) -> None:
        """Customize axis appearance and labels.

        Args:
            x_base_positions: X-axis positions for condition labels
        """
        assert self.ax is not None, "Axis is not set"
        # Set x-axis
        self.ax.set_xticks(x_base_positions)
        self.ax.set_xticklabels(self.conditions, rotation=45, ha="right")
        self.ax.set_xlabel("")

        # Set y-axis
        self.ax.set_ylabel("% of population")
        self.ax.set_ylim(0, 100)

        # Add grid
        self.ax.grid(True, alpha=0.3, axis="y")
        self.ax.set_axisbelow(True)

    def _draw_group_boxes(self, x_base_positions: List[float]) -> None:
        """Draw boxes around replicate groups.

        Args:
            x_base_positions: X-axis positions for each condition
        """
        assert self.ax is not None, "Axis is not set"
        for cond_idx, condition in enumerate(self.conditions):
            x_center = x_base_positions[cond_idx]

            # Calculate box dimensions
            box_width = len(self.repeat_ids) * self.repeat_offset
            box_height = 100  # Full height
            box_x = x_center - box_width / 2

            # Create rectangle
            rect = Rectangle(
                (box_x, 0),
                box_width,
                box_height,
                fill=False,
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
            )
            self.ax.add_patch(rect)

    def _add_phase_legend(self) -> None:
        """Add legend for cell cycle phases."""
        assert self.ax is not None, "Axis is not set"
        # Create legend handles
        handles = []
        for i, phase in enumerate(self.phases):
            color = self.colors[i % len(self.colors)]
            handle = Rectangle((0, 0), 1, 1, color=color, label=phase)
            handles.append(handle)

        # Add legend
        self.ax.legend(
            handles=list(reversed(handles)),
            labels=list(reversed(self.phases)),
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    def add_replicate_labels(self) -> None:
        """Add replicate ID labels below x-axis."""
        if not self.fig:
            raise ValueError("Must call generate() before adding labels")

    def get_replicate_variability(self) -> dict:
        """Calculate coefficient of variation for each condition and phase.

        Returns:
            Dictionary with conditions as keys and phase CVs as values
        """
        variability = {}

        for condition in self.conditions:
            condition_data = {}
            condition_subset = self.processed_data[
                self.processed_data[self.condition_col] == condition
            ]

            for phase in self.phases:
                phase_data = condition_subset[
                    condition_subset.cell_cycle == phase
                ]["percent"]

                if len(phase_data) > 1:
                    cv = (
                        phase_data.std() / phase_data.mean()
                        if phase_data.mean() > 0
                        else 0
                    )
                    condition_data[phase] = cv
                else:
                    condition_data[phase] = 0

            variability[condition] = condition_data

        return variability

    def highlight_outliers(self, z_threshold: float = 2.0) -> None:
        """Highlight replicate bars that are outliers.

        Args:
            z_threshold: Z-score threshold for outlier detection
        """
        # This would add visual indicators for outlier replicates
        # Implementation depends on specific requirements

    def save(self, path, filename: Optional[str] = None, **kwargs):
        """Save the grouped cell cycle plot.

        Args:
            path: Path to save location
            filename: Optional filename. If None, generates descriptive name
            **kwargs: Additional save parameters
        """
        if filename is None:
            selector_part = (
                f"_{self.selector_val}" if self.selector_val else ""
            )
            filename = f"cellcycle_grouped{selector_part}.pdf"

        super().save(path, filename, **kwargs)


def cellcycle_grouped_plot(
    data: pd.DataFrame,
    conditions: List[str],
    # Base class arguments
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    # CellCycleGroupedPlot specific arguments
    phases: Optional[List[str]] = None,
    group_size: int = 2,
    n_repeats: int = 3,
    repeat_offset: float = 0.18,
    bar_width: Optional[float] = None,
    show_group_boxes: bool = True,
    show_legend: bool = True,
    # Integration arguments
    ax: Optional[Axes] = None,
    # Output arguments
    save: bool = False,
    output_path: Optional[str] = None,
    filename: Optional[str] = None,
    # Additional matplotlib/save arguments
    dpi: int = 300,
    format: str = "pdf",
    tight_layout: bool = True,
    **kwargs,
) -> Figure:
    """Create a grouped cell cycle plot showing individual replicates.

    This is the main user-facing function for creating grouped cell cycle plots.
    It combines all functionality from the base classes and provides a simple interface
    for generating publication-ready grouped bar charts that show replicate variability.

    Args:
        data: DataFrame containing cell cycle data with required columns:
              - 'cell_cycle': Cell cycle phase for each cell
              - 'plate_id': Plate/replicate identifier
              - 'experiment': Unique cell identifier
              - condition_col: Column containing experimental conditions
              - selector_col: Column for data selection (e.g., cell_line)
        conditions: List of experimental conditions to plot

        # Data filtering arguments
        condition_col: Name of column containing experimental conditions
        selector_col: Name of column for data filtering (e.g., 'cell_line')
        selector_val: Value to filter by in selector_col (e.g., 'RPE-1')

        # Plot appearance arguments
        title: Overall plot title. If None, auto-generated from selector_val
        colors: Custom color palette. If None, uses default from config
        figsize: Figure size as (width, height) in inches. If None, uses default
        phases: List of cell cycle phases to plot. If None, uses default order

        # Grouped plot specific arguments
        group_size: Number of conditions per visual group
        n_repeats: Number of replicates to show per condition
        repeat_offset: Spacing between replicate bars within each condition
        bar_width: Width of individual replicate bars. If None, auto-calculated
        show_group_boxes: Whether to draw boxes around condition groups
        show_legend: Whether to show the phase legend

        # Integration arguments
        ax: Optional matplotlib axes to plot on. If provided, creates subplot

        # Output arguments
        save: Whether to save the figure to file
        output_path: Directory or full path for saving. Required if save=True
        filename: Specific filename. If None, auto-generated based on parameters

        # Save quality arguments
        dpi: Resolution for saved figure (dots per inch)
        format: File format ('pdf', 'png', 'svg', etc.)
        tight_layout: Whether to apply tight layout before saving

        **kwargs: Additional arguments passed to the base class

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If required arguments are missing or invalid
        FileNotFoundError: If output_path doesn't exist when save=True

    Examples:
        Basic usage:
        >>> fig = cellcycle_grouped_plot(
        ...     data=cell_cycle_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_val='RPE-1'
        ... )

        Integration with subplots:
        >>> fig, ax = plt.subplots()
        >>> fig = cellcycle_grouped_plot(
        ...     data=cell_cycle_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_val='RPE-1',
        ...     ax=ax
        ... )

        Full customization:
        >>> fig = cellcycle_grouped_plot(
        ...     data=cell_cycle_data,
        ...     conditions=['Control', 'CDK4i', 'CDK6i'],
        ...     condition_col='condition',
        ...     selector_col='cell_line',
        ...     selector_val='RPE-1',
        ...     title='Grouped Cell Cycle Analysis with Replicates',
        ...     colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        ...     figsize=(14, 8),
        ...     phases=['Polyploid', 'G2/M', 'S', 'G1'],
        ...     group_size=3,
        ...     n_repeats=4,
        ...     repeat_offset=0.2,
        ...     bar_width=0.15,
        ...     show_group_boxes=True,
        ...     show_legend=True,
        ...     save=True,
        ...     output_path='figures/',
        ...     filename='grouped_cellcycle.pdf',
        ...     dpi=600,
        ...     format='pdf'
        ... )
    """
    from pathlib import Path

    # Validate required arguments
    if data.empty:
        raise ValueError("Input data cannot be empty")

    if not conditions:
        raise ValueError("At least one condition must be specified")

    if save and not output_path:
        raise ValueError("output_path is required when save=True")

    # Auto-generate title if not provided
    if title is None and selector_val:
        title = f"Cell Cycle Analysis - {selector_val} (Individual Replicates)"
    elif title is None:
        title = "Cell Cycle Analysis (Individual Replicates)"

    # Create the plot instance
    plot = CellCycleGroupedPlot(
        data=data,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        title=title,
        colors=colors,
        figsize=figsize,
        phases=phases,
        group_size=group_size,
        n_repeats=n_repeats,
        repeat_offset=repeat_offset,
        bar_width=bar_width,
        show_group_boxes=show_group_boxes,
        show_legend=show_legend,
        ax=ax,
        **kwargs,
    )

    # Generate the plot
    try:
        fig = plot.generate()

        # Save if requested (only if we own the figure)
        if save and plot._owns_figure:
            save_path = Path(output_path)

            # Auto-generate filename if not provided
            if filename is None:
                # Create descriptive filename
                selector_part = f"_{selector_val}" if selector_val else ""
                group_part = f"_g{group_size}" if group_size != 2 else ""
                reps_part = f"_r{n_repeats}" if n_repeats != 3 else ""
                filename = f"cellcycle_grouped{selector_part}{group_part}{reps_part}.{format}"

            # Ensure filename has correct extension
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"

            plot.save(
                path=save_path,
                filename=filename,
                tight_layout=tight_layout,
                dpi=dpi,
                format=format,
            )

            print(f"Grouped cell cycle plot saved to: {save_path / filename}")
        elif save:
            print(
                "Warning: Cannot save when using provided axis. Save the parent figure manually."
            )

        return fig

    except Exception as e:
        # Clean up resources in case of error
        if plot._owns_figure:
            plot.close()
        raise e

    finally:
        # Note: We don't automatically close the figure here because the user
        # might want to further customize it. User should call plt.close(fig)
        # or plot.close() when done (if they own the figure).
        pass
