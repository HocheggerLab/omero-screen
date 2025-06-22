"""Stacked cell cycle plot implementation.

This module provides the CellCycleStackedPlot class that creates a stacked bar plot
showing the relative proportions of all cell cycle phases in a single chart.
This view is excellent for comparing overall cell cycle distributions between conditions.
"""

from typing import List, Optional

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import BaseCellCyclePlot


class CellCycleStackedPlot(BaseCellCyclePlot):
    """Stacked bar plot for cell cycle phase proportions.

    Creates a single stacked bar chart where:
    - Each bar represents one experimental condition
    - Different colors represent different cell cycle phases
    - Heights show the percentage of cells in each phase
    - Total height of each bar equals 100%

    This visualization is ideal for:
    - Comparing overall cell cycle distributions
    - Identifying major shifts in cell cycle profiles
    - Integration into larger figure layouts
    """

    @property
    def plot_type(self) -> str:
        return "cellcycle_stacked"

    def __init__(
        self,
        data,
        conditions: List[str],
        phases: Optional[List[str]] = None,
        reverse_stack: bool = False,
        show_legend: bool = True,
        legend_position: str = "right",
        **kwargs,
    ):
        """Initialize stacked cell cycle plot.

        Args:
            data: DataFrame containing cell cycle data
            conditions: List of conditions to plot
            phases: List of cell cycle phases. If None, uses default order
            reverse_stack: If True, reverse the stacking order
            show_legend: Whether to show the phase legend
            legend_position: Legend position ("right", "bottom", "top", "left")
            **kwargs: Additional arguments passed to base class
        """
        # Default phases for stacked plot (common order)
        default_phases = ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]
        if phases is None:
            phases = default_phases

        super().__init__(data, conditions, phases=phases, **kwargs)

        self.reverse_stack = reverse_stack
        self.show_legend = show_legend
        self.legend_position = legend_position

        # Adjust stacking order if requested
        if self.reverse_stack:
            self.phases = list(reversed(self.phases))

    def generate(self) -> Figure:
        """Generate the stacked bar plot.

        Returns:
            Figure containing the stacked bar plot
        """
        # Setup figure
        self._setup_figure()

        # Create axis if we own the figure but don't have one
        if self.ax is None and self._owns_figure:
            self.ax = self.fig.add_subplot(111)

        assert self.ax is not None, "Axis is not set"
        # Get mean percentages for plotting
        mean_data = self.get_mean_percentages()

        # Ensure we have all requested conditions
        mean_data = mean_data.reindex(self.conditions, fill_value=0)

        # Create stacked bar plot
        bottom = None
        legend_handles = []

        for i, phase in enumerate(self.phases):
            if phase not in mean_data.columns:
                continue

            color = self.colors[i % len(self.colors)]

            bars = self.ax.bar(
                range(len(self.conditions)),
                mean_data[phase],
                bottom=bottom,
                label=phase,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

            # Track handles for legend
            legend_handles.append(bars[0])

            # Update bottom for next stack level
            if bottom is None:
                bottom = mean_data[phase].copy()
            else:
                bottom += mean_data[phase]

        # Customize plot
        self._customize_axes()

        # Add legend if requested
        if self.show_legend:
            self._add_legend(legend_handles)

        # Apply title
        self._apply_title()

        return self.fig  # type: ignore

    def _customize_axes(self) -> None:
        """Customize axis appearance and labels."""
        assert self.ax is not None, "Axis is not set"
        # Set x-axis
        self.ax.set_xticks(range(len(self.conditions)))
        self.ax.set_xticklabels(self.conditions, rotation=45, ha="right")

        # Set y-axis
        self.ax.set_ylabel("% of population")
        self.ax.set_ylim(0, 100)

        # Add grid for better readability
        self.ax.grid(True, alpha=0.3, axis="y")
        self.ax.set_axisbelow(True)

        # Remove top and right spines for cleaner look
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

    def _add_legend(self, handles) -> None:
        """Add legend to the plot.

        Args:
            handles: List of matplotlib artists for legend
        """
        assert self.ax is not None, "Axis is not set"
        legend_kwargs = {
            "handles": list(reversed(handles)),
            "labels": list(reversed(self.phases)),
            "frameon": True,
            "fancybox": True,
            "shadow": True,
            "framealpha": 0.9,
        }

        if self.legend_position == "right":
            legend_kwargs |= {"bbox_to_anchor": (1.05, 1), "loc": "upper left"}
        elif self.legend_position == "bottom":
            legend_kwargs |= {
                "bbox_to_anchor": (0.5, -0.15),
                "loc": "upper center",
                "ncol": min(len(self.phases), 3),
            }
        elif self.legend_position == "top":
            legend_kwargs |= {
                "bbox_to_anchor": (0.5, 1.15),
                "loc": "lower center",
                "ncol": min(len(self.phases), 3),
            }
        else:  # left
            legend_kwargs |= {
                "bbox_to_anchor": (-0.05, 1),
                "loc": "upper right",
            }

        self.ax.legend(**legend_kwargs)

    def get_phase_contributions(self) -> dict:
        """Get the contribution of each phase as percentages.

        Returns:
            Dictionary with phase names as keys and their mean contributions as values
        """
        mean_data = self.get_mean_percentages()
        total_cells = mean_data.sum(axis=1).mean()  # Average across conditions

        return {
            phase: mean_data[phase].mean()
            if phase in mean_data.columns
            else 0.0
            for phase in self.phases
        }

    def add_percentage_labels(self, min_percentage: float = 5.0) -> None:
        """Add percentage labels to bars.

        Args:
            min_percentage: Minimum percentage to show label for (avoids clutter)
        """
        assert self.ax is not None, "Axis is not set"
        if not self.fig:
            raise ValueError("Must call generate() before adding labels")

        mean_data = self.get_mean_percentages()

        for i, condition in enumerate(self.conditions):
            y_bottom = 0
            for phase in self.phases:
                if phase not in mean_data.columns:
                    continue

                value = mean_data.loc[condition, phase]
                assert isinstance(value, float), "Value is not a float"
                if value >= min_percentage:
                    y_center = y_bottom + value / 2
                    self.ax.text(
                        i,
                        y_center,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white" if value > 15 else "black",
                    )
                y_bottom += value

    def save(self, path, filename: Optional[str] = None, **kwargs):
        """Save the stacked cell cycle plot.

        Args:
            path: Path to save location
            filename: Optional filename. If None, generates descriptive name
            **kwargs: Additional save parameters
        """
        if filename is None:
            selector_part = (
                f"_{self.selector_val}" if self.selector_val else ""
            )
            filename = f"cellcycle_stacked{selector_part}.pdf"

        super().save(path, filename, **kwargs)


def cellcycle_stacked_plot(
    data: pd.DataFrame,
    conditions: List[str],
    # Base class arguments
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    # CellCycleStackedPlot specific arguments
    phases: Optional[List[str]] = None,
    reverse_stack: bool = False,
    show_legend: bool = True,
    legend_position: str = "right",
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
    """Create a stacked cell cycle plot showing phase proportions.

    This is the main user-facing function for creating stacked cell cycle plots.
    It combines all functionality from the base classes and provides a simple interface
    for generating publication-ready stacked bar charts.

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

        # Stacked plot specific arguments
        reverse_stack: If True, reverse the stacking order of phases
        show_legend: Whether to show the phase legend
        legend_position: Legend position ("right", "bottom", "top", "left")

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
        >>> fig = cellcycle_stacked_plot(
        ...     data=cell_cycle_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_val='RPE-1'
        ... )

        Integration with subplots:
        >>> fig, ax = plt.subplots()
        >>> fig = cellcycle_stacked_plot(
        ...     data=cell_cycle_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_val='RPE-1',
        ...     ax=ax
        ... )

        Full customization:
        >>> fig = cellcycle_stacked_plot(
        ...     data=cell_cycle_data,
        ...     conditions=['Control', 'CDK4i', 'CDK6i'],
        ...     condition_col='condition',
        ...     selector_col='cell_line',
        ...     selector_val='RPE-1',
        ...     title='Stacked Cell Cycle Distribution',
        ...     colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        ...     figsize=(10, 6),
        ...     phases=['Polyploid', 'G2/M', 'S', 'G1'],
        ...     reverse_stack=False,
        ...     show_legend=True,
        ...     legend_position='right',
        ...     save=True,
        ...     output_path='figures/',
        ...     filename='stacked_cellcycle.pdf',
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
        title = f"Cell Cycle Distribution - {selector_val}"
    elif title is None:
        title = "Cell Cycle Distribution"

    # Create the plot instance
    plot = CellCycleStackedPlot(
        data=data,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        title=title,
        colors=colors,
        figsize=figsize,
        phases=phases,
        reverse_stack=reverse_stack,
        show_legend=show_legend,
        legend_position=legend_position,
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
                stack_part = "_reversed" if reverse_stack else ""
                filename = (
                    f"cellcycle_stacked{selector_part}{stack_part}.{format}"
                )

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

            print(f"Stacked cell cycle plot saved to: {save_path / filename}")
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
