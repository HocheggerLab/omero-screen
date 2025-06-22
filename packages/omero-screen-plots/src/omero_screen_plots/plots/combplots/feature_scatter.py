"""Feature scatter plot for DAPI vs any feature analysis.

This module provides the FeatureScatterPlot class that creates scatter plots showing
the relationship between DNA content (DAPI) and any specified feature with
threshold-based color coding.
"""

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import BaseFeatureScatter


class FeatureScatterPlot(BaseFeatureScatter):
    """Feature scatter plot for DAPI vs feature analysis.

    Creates scatter plots showing the relationship between DNA content and
    any specified feature, with threshold-based color coding. This visualization is ideal for:
    - Feature correlation with cell cycle
    - Threshold-based population analysis
    - Quality control of experimental measurements
    """

    @property
    def plot_type(self) -> str:
        """Return the type of plot."""
        return "feature_scatter"

    def generate(self) -> Figure:
        """Generate feature scatter plot.

        Returns:
            Figure containing the scatter plot
        """
        n_conditions = len(self.conditions)

        # If an axis is provided, use it (subplot integration mode)
        if self.ax is not None:
            # Single axis provided - plot first condition only
            if n_conditions > 1:
                import warnings

                warnings.warn(
                    f"Multiple conditions provided but only one axis. Plotting only '{self.conditions[0]}'.",
                    stacklevel=2,
                )

            condition = self.conditions[0]
            data = self.get_condition_data(condition)

            # Plot on provided axis with x-label (since it's a subplot)
            self.create_feature_scatter(self.ax, data, 0, 1)

            # For provided axis, always show x-axis labels and ticks
            self.ax.set_xlabel("norm. DNA content", fontsize=6)
            self.ax.xaxis.set_visible(True)

            # Add title to the individual axis
            if self.title:
                self.ax.set_title(self.title, size=8, weight="bold")
            else:
                feature_name = (
                    self.feature_col.split("_")[-1]
                    if "_" in self.feature_col
                    else self.feature_col
                )
                self.ax.set_title(
                    f"{feature_name.title()} vs DNA - {condition}",
                    size=8,
                    weight="regular",
                )

            return self.ax.figure  # type: ignore[return-value]

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

        # Create scatter plot for each condition
        for i, condition in enumerate(self.conditions):
            ax = fig.add_subplot(gs[0, i])
            data = self.get_condition_data(condition)

            # Create feature scatter plot
            self.create_feature_scatter(ax, data, i, n_conditions)

            # Add condition title
            ax.set_title(condition, size=6, weight="regular")

            # Handle x-axis labeling based on context
            if label_strategy == "individual":
                # Single plot: show x-label
                ax.set_xlabel("norm. DNA content", fontsize=6)
                ax.xaxis.set_visible(True)
            else:
                # Multiple plots: no individual x-label
                ax.set_xlabel("")

        # Add common x-axis label only for multiple conditions
        if label_strategy == "common":
            self.add_common_x_label(fig, force_label=True)

        # Apply title using centralized method
        self._apply_figure_title(fig)

        return fig

    def save(
        self,
        path: Union[str, Path],
        filename: Optional[str] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the feature scatter plot.

        Args:
            path: Path to save location
            filename: Optional filename. If None, generates descriptive name
            tight_layout: Whether to apply tight layout
            **kwargs: Additional save parameters
        """
        if filename is None:
            selector_part = (
                f"_{self.selector_val}" if self.selector_val else ""
            )
            feature_part = (
                f"_{self.feature_col.split('_')[-1]}"
                if "_" in self.feature_col
                else f"_{self.feature_col}"
            )
            filename = f"feature_scatter{selector_part}{feature_part}.png"

        super().save(path, filename, tight_layout=tight_layout, **kwargs)


def feature_scatter_plot(
    data: pd.DataFrame,
    conditions: list[str],
    feature_col: str,
    feature_threshold: float,
    # Base class arguments
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[list[str]] = None,
    figsize: Optional[tuple[int, int]] = None,
    # Feature scatter specific arguments
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
    file_format: str = "png",
    tight_layout: bool = True,
    **kwargs: Any,
) -> Figure:
    """Create a feature scatter plot of DAPI vs any feature.

    This is the main user-facing function for creating feature scatter plots showing
    the relationship between DNA content and any specified feature, with threshold-based coloring.

    Args:
        data: DataFrame containing cell data with required columns:
              - dapi_col: Normalized DAPI intensity (DNA content)
              - feature_col: The feature column to analyze
              - condition_col: Column containing experimental conditions
              - selector_col: Column for data selection (e.g., cell_line)
        conditions: List of experimental conditions to plot
        feature_col: Column name for the feature to plot against DAPI
        feature_threshold: Threshold value for color categorization

        # Data filtering arguments
        condition_col: Name of column containing experimental conditions
        selector_col: Name of column for data filtering (e.g., 'cell_line')
        selector_val: Value to filter by in selector_col (e.g., 'RPE-1')

        # Plot appearance arguments
        title: Overall plot title. If None, auto-generated from selector_val
        colors: Custom color palette. If None, uses default from config
        figsize: Figure size as (width, height) in inches. If None, uses default

        # Feature scatter specific arguments
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
        file_format: File format ('png', 'pdf', 'svg', etc.)
        tight_layout: Whether to apply tight layout before saving

        **kwargs: Additional arguments passed to the base class

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If required arguments are missing or invalid
        FileNotFoundError: If output_path doesn't exist when save=True

    Examples:
        Basic usage:
        >>> fig = feature_scatter_plot(
        ...     data=cell_data,
        ...     conditions=['Control', 'Treatment'],
        ...     feature_col='intensity_mean_yH2AX_nucleus',
        ...     feature_threshold=3.0,
        ...     selector_val='RPE-1'
        ... )

        Custom threshold and styling:
        >>> fig = feature_scatter_plot(
        ...     data=cell_data,
        ...     conditions=['Control', 'Treatment'],
        ...     feature_col='area_nucleus',
        ...     feature_threshold=5000,
        ...     selector_val='RPE-1',
        ...     title='Nuclear Area vs DNA Content',
        ...     figsize=(12, 6)
        ... )

        Performance optimized with saving:
        >>> fig = feature_scatter_plot(
        ...     data=cell_data,
        ...     conditions=['Control', 'Treatment'],
        ...     feature_col='intensity_mean_yH2AX_nucleus',
        ...     feature_threshold=3.0,
        ...     selector_val='RPE-1',
        ...     cell_number=5000,
        ...     save=True,
        ...     output_path='figures/',
        ...     dpi=600
        ... )
    """
    # Validate required arguments
    if data.empty:
        raise ValueError("Input data cannot be empty")

    if not conditions:
        raise ValueError("At least one condition must be specified")

    if feature_col not in data.columns:
        raise ValueError(f"Feature column '{feature_col}' not found in data")

    if save and not output_path:
        raise ValueError("output_path is required when save=True")

    # Auto-generate title if not provided
    if title is None and selector_val:
        feature_name = (
            feature_col.split("_")[-1] if "_" in feature_col else feature_col
        )
        title = f"{feature_name.title()} vs DNA Content - {selector_val}"
    elif title is None:
        feature_name = (
            feature_col.split("_")[-1] if "_" in feature_col else feature_col
        )
        title = f"{feature_name.title()} vs DNA Content"

    # Create the plot instance
    plot = FeatureScatterPlot(
        data=data,
        conditions=conditions,
        feature_col=feature_col,
        feature_threshold=feature_threshold,
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
            save_path = Path(output_path) if output_path else Path(".")

            # Auto-generate filename if not provided
            if filename is None:
                # Create descriptive filename
                selector_part = f"_{selector_val}" if selector_val else ""
                feature_part = (
                    f"_{feature_col.split('_')[-1]}"
                    if "_" in feature_col
                    else f"_{feature_col}"
                )
                sample_part = f"_n{cell_number}" if cell_number else ""
                threshold_part = f"_t{feature_threshold}"
                filename = f"feature_scatter{selector_part}{feature_part}{threshold_part}{sample_part}.{format}"

            # Ensure filename has correct extension
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"

            plot.save(
                path=save_path,
                filename=filename,
                tight_layout=tight_layout,
                dpi=dpi,
                format=file_format,
            )

            print(f"Feature scatter plot saved to: {save_path / filename}")
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
