"""Histogram plot for DAPI intensity distribution.

This module provides the HistogramPlot class that creates histograms showing
the distribution of normalized DAPI intensity (DNA content) for different conditions.
"""

from typing import List, Optional

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import BaseHistogramPlot


class HistogramPlot(BaseHistogramPlot):
    """Histogram plot for DAPI intensity distribution.

    Creates histograms showing the distribution of DNA content across conditions.
    This visualization is useful for:
    - Assessing DNA content distributions
    - Identifying population shifts
    - Quality control of cell cycle data
    """

    @property
    def plot_type(self) -> str:
        return "histogram"

    def generate(self) -> Figure:
        """Generate histogram plot.

        Returns:
            Figure containing the histogram plot
        """
        n_conditions = len(self.conditions)

        # If an axis is provided, use it (subplot integration mode)
        if self.ax is not None:
            # Single axis provided - plot first condition only
            if n_conditions > 1:
                import warnings

                warnings.warn(
                    f"Multiple conditions provided but only one axis. Plotting only '{self.conditions[0]}'."
                )

            condition = self.conditions[0]
            data = self.get_condition_data(condition)

            # Plot on provided axis with individual x-label (since it's a subplot)
            self.create_histogram(
                self.ax, data, 0, show_individual_xlabel=True
            )

            # Add title to the individual axis
            if self.title:
                self.ax.set_title(self.title, size=8, weight="bold")
            else:
                self.ax.set_title(condition, size=8, weight="regular")

            return self.ax.figure

        # No axis provided - create our own figure with subplots
        # Determine x-axis labeling strategy
        label_strategy = self._should_show_x_label(
            n_conditions, is_part_of_combplot=False
        )

        # Setup figure and grid
        fig, gs = self.setup_subplot_grid(
            n_rows=1,
            n_cols=n_conditions,
            height_ratios=None,
            width_ratios=None,
        )

        # Create histogram for each condition
        for i, condition in enumerate(self.conditions):
            ax = fig.add_subplot(gs[0, i])
            data = self.get_condition_data(condition)

            # Create histogram with context-aware x-labeling
            show_individual_xlabel = label_strategy == "individual"
            self.create_histogram(ax, data, i, show_individual_xlabel)

            # Add condition title
            ax.set_title(condition, size=6, weight="regular")

        # Add common x-axis label only for multiple conditions
        if label_strategy == "common":
            self.add_common_x_label(fig, force_label=True)

        # Apply title using centralized method
        self._apply_figure_title(fig)

        return fig

    def save(self, path, filename: Optional[str] = None, **kwargs):
        """Save the histogram plot.

        Args:
            path: Path to save location
            filename: Optional filename. If None, generates descriptive name
            **kwargs: Additional save parameters
        """
        if filename is None:
            selector_part = (
                f"_{self.selector_val}" if self.selector_val else ""
            )
            filename = f"histogram{selector_part}.png"

        super().save(path, filename, **kwargs)


def histogram_plot(
    data: pd.DataFrame,
    conditions: List[str],
    # Base class arguments
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    # Histogram specific arguments
    cell_number: Optional[int] = None,
    dapi_col: str = "integrated_int_DAPI_norm",
    # Integration arguments
    ax: Optional[Axes] = None,
    # Output arguments
    save: bool = False,
    output_path: Optional[str] = None,
    filename: Optional[str] = None,
    # Additional matplotlib/save arguments
    dpi: int = 300,
    format: str = "png",
    tight_layout: bool = True,
    **kwargs,
) -> Figure:
    """Create a histogram plot of DAPI intensity distribution.

    This is the main user-facing function for creating histogram plots showing
    DNA content distributions across experimental conditions.

    Args:
        data: DataFrame containing DAPI intensity data with required columns:
              - dapi_col: Normalized DAPI intensity values
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

        # Histogram specific arguments
        cell_number: Optional limit on number of cells per condition (for performance)
        dapi_col: Column name for DAPI intensity values

        # Integration arguments
        ax: Optional matplotlib axes to plot on. If provided, creates subplot

        # Output arguments
        save: Whether to save the figure to file
        output_path: Directory or full path for saving. Required if save=True
        filename: Specific filename. If None, auto-generated based on parameters

        # Save quality arguments
        dpi: Resolution for saved figure (dots per inch)
        format: File format ('png', 'pdf', 'svg', etc.)
        tight_layout: Whether to apply tight layout before saving

        **kwargs: Additional arguments passed to the base class

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If required arguments are missing or invalid
        FileNotFoundError: If output_path doesn't exist when save=True

    Examples:
        Basic usage:
        >>> fig = histogram_plot(
        ...     data=cell_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_val='RPE-1'
        ... )

        With cell sampling for performance:
        >>> fig = histogram_plot(
        ...     data=cell_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_val='RPE-1',
        ...     cell_number=5000
        ... )

        High-quality output:
        >>> fig = histogram_plot(
        ...     data=cell_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_val='RPE-1',
        ...     save=True,
        ...     output_path='figures/',
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
        title = f"DAPI Intensity Distribution - {selector_val}"
    elif title is None:
        title = "DAPI Intensity Distribution"

    # Create the plot instance
    plot = HistogramPlot(
        data=data,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        title=title,
        colors=colors,
        figsize=figsize,
        cell_number=cell_number,
        dapi_col=dapi_col,
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
                sample_part = f"_n{cell_number}" if cell_number else ""
                filename = f"histogram{selector_part}{sample_part}.{format}"

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

            print(f"Histogram plot saved to: {save_path / filename}")
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
