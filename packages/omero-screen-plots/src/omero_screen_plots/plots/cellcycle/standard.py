"""Standard cell cycle plot implementation.

This module provides the CellCyclePlot class that creates a 2x2 subplot grid
showing the percentage of cells in each cell cycle phase (G1, S, G2/M, Polyploid)
as separate bar plots with individual data points and significance testing.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from ...utils import show_repeat_points
from .base import BaseCellCyclePlot


class CellCyclePlot(BaseCellCyclePlot):
    """Standard cell cycle plot with 2x2 subplot grid.

    Creates individual bar plots for each cell cycle phase, showing:
    - Bar plot of mean percentages per condition
    - Individual replicate data points overlaid
    - Statistical significance markers (if sufficient replicates)

    This is the most detailed view of cell cycle data, allowing comparison
    of individual phases across conditions.
    """

    @property
    def plot_type(self) -> str:
        return "cellcycle_standard"

    def __init__(
        self,
        data,
        conditions: List[str],
        phases: Optional[List[str]] = None,
        show_significance: bool = True,
        show_points: bool = True,
        **kwargs,
    ):
        """Initialize standard cell cycle plot.

        Args:
            data: DataFrame containing cell cycle data
            conditions: List of conditions to plot
            phases: List of cell cycle phases to plot (max 4 for 2x2 grid)
            show_significance: Whether to show significance markers
            show_points: Whether to show individual replicate points
            **kwargs: Additional arguments passed to base class
        """
        # Limit phases to 4 for 2x2 grid
        if phases and len(phases) > 4:
            phases = phases[:4]

        super().__init__(data, conditions, phases=phases, **kwargs)

        self.show_significance = show_significance
        self.show_points = show_points

        # Validate that we have at most 4 phases for the 2x2 grid
        if len(self.phases) > 4:
            self.phases = self.phases[:4]

    def generate(self) -> Figure:
        """Generate the 2x2 cell cycle plot.

        Returns:
            Figure containing the 2x2 subplot grid

        Raises:
            ValueError: If ax parameter was provided (incompatible with subplots)
        """
        # Standard plot requires creating subplots, can't use provided ax
        if self.ax is not None:
            raise ValueError(
                "CellCyclePlot creates a 2x2 subplot grid and cannot use a provided axis. "
                "Use CellCycleStackedPlot for single-axis integration."
            )

        # Create 2x2 subplot grid
        self.fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

        # Plot each phase
        for i, phase in enumerate(self.phases):
            if i >= len(ax_list):
                break

            axis = ax_list[i]
            self._plot_phase(axis, phase, i)

        # Hide unused subplots
        for i in range(len(self.phases), len(ax_list)):
            ax_list[i].set_visible(False)

        # Apply overall title if specified
        if self.title:
            self.fig.suptitle(self.title, fontweight="bold", y=1.05)

        # Adjust layout
        plt.tight_layout()
        if self.title:
            plt.subplots_adjust(top=0.92)  # Make room for suptitle

        return self.fig

    def _plot_phase(self, ax, phase: str, phase_index: int) -> None:
        """Plot data for a single cell cycle phase.

        Args:
            ax: Matplotlib axis to plot on
            phase: Cell cycle phase name
            phase_index: Index of the phase (for color selection)
        """
        # Get data for this phase
        phase_data = self.get_phase_data(phase)

        if phase_data.empty:
            ax.text(
                0.5,
                0.5,
                f"No data for {phase}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(phase, fontweight="bold")
            return

        # Select color (offset by 1 to avoid using first color)
        color_idx = (phase_index + 1) % len(self.colors)
        phase_color = self.colors[color_idx]

        # Create bar plot
        sns.barplot(
            data=phase_data,
            x=self.condition_col,
            y="percent",
            color=phase_color,
            order=self.conditions,
            ax=ax,
        )

        # Add individual data points if requested
        if self.show_points and callable(show_repeat_points):
            try:
                show_repeat_points(
                    df=phase_data,
                    conditions=self.conditions,
                    condition_col=self.condition_col,
                    y_col="percent",
                    ax=ax,
                )
            except Exception as e:
                print(f"Warning: Could not add repeat points for {phase}: {e}")

        # Add significance markers if requested and sufficient replicates
        if self.show_significance and self.has_sufficient_replicates():
            self.add_significance_markers_to_axis(ax, phase_data, "percent")

        # Customize axis
        ax.set_title(phase, fontweight="bold", fontsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("% of population")

        # Rotate x-axis labels if they're long
        max_label_length = max(len(str(cond)) for cond in self.conditions)
        if max_label_length > 6:
            ax.tick_params(axis="x", rotation=45)

        # Set y-axis limits to ensure consistency and room for significance markers
        y_max = (
            max(100, phase_data["percent"].max() * 1.2)
            if not phase_data.empty
            else 100
        )
        ax.set_ylim(0, y_max)

    def save(self, path, filename: Optional[str] = None, **kwargs):
        """Save the cell cycle plot.

        Args:
            path: Path to save location
            filename: Optional filename. If None, generates descriptive name
            **kwargs: Additional save parameters
        """
        if filename is None:
            # Generate descriptive filename
            selector_part = (
                f"_{self.selector_val}" if self.selector_val else ""
            )
            filename = f"cellcycle_standard{selector_part}.pdf"

        super().save(path, filename, **kwargs)


def cellcycle_standard_plot(
    data: pd.DataFrame,
    conditions: List[str],
    # Base class arguments
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    # CellCyclePlot specific arguments
    phases: Optional[List[str]] = None,
    show_significance: bool = True,
    show_points: bool = True,
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
    """Create a standard cell cycle plot with 2x2 subplot grid.

    This is the main user-facing function for creating standard cell cycle plots.
    It combines all functionality from the base classes and provides a simple interface.

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
        phases: List of cell cycle phases to plot (max 4). If None, uses ['G1', 'S', 'G2/M', 'Polyploid']

        # Plot features arguments
        show_significance: Whether to show statistical significance markers
        show_points: Whether to show individual replicate data points

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
        >>> fig = cellcycle_standard_plot(
        ...     data=cell_cycle_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_col='cell_line',
        ...     selector_val='RPE-1'
        ... )

        Save to file:
        >>> fig = cellcycle_standard_plot(
        ...     data=cell_cycle_data,
        ...     conditions=['Control', 'CDK4i', 'CDK6i'],
        ...     selector_col='cell_line',
        ...     selector_val='RPE-1',
        ...     title='Cell Cycle Analysis - RPE-1',
        ...     save=True,
        ...     output_path='figures/',
        ...     filename='rpe1_cellcycle.pdf'
        ... )

        Custom styling:
        >>> fig = cellcycle_standard_plot(
        ...     data=cell_cycle_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_col='cell_line',
        ...     selector_val='HeLa',
        ...     colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        ...     figsize=(10, 8),
        ...     show_significance=False,
        ...     phases=['G1', 'S', 'G2/M']
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
        title = f"Cell Cycle Analysis - {selector_val}"
    elif title is None:
        title = "Cell Cycle Analysis"

    # Create the plot instance
    plot = CellCyclePlot(
        data=data,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        title=title,
        colors=colors,
        figsize=figsize,
        phases=phases,
        show_significance=show_significance,
        show_points=show_points,
        **kwargs,
    )

    # Generate the plot
    try:
        fig = plot.generate()

        # Save if requested
        if save:
            save_path = Path(output_path)

            # Auto-generate filename if not provided
            if filename is None:
                # Create descriptive filename
                selector_part = f"_{selector_val}" if selector_val else ""
                phase_part = f"_{len(plot.phases)}phases" if phases else ""
                filename = (
                    f"cellcycle_standard{selector_part}{phase_part}.{format}"
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

            print(f"Cell cycle plot saved to: {save_path / filename}")

        return fig

    except Exception as e:
        # Clean up resources in case of error
        plot.close()
        raise e

    finally:
        # Note: We don't automatically close the figure here because the user
        # might want to further customize it. User should call plt.close(fig)
        # or plot.close() when done.
        pass
