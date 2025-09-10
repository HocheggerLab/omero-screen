"""High-level API for scatter plots."""

from pathlib import Path
from typing import Literal

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.colors import COLOR
from omero_screen_plots.scatterplot_factory import (
    ScatterPlot,
    ScatterPlotConfig,
)


def scatter_plot(
    df: pd.DataFrame,
    conditions: str | list[str],
    condition_col: str = "condition",
    selector_col: str | None = None,
    selector_val: str | None = None,
    # Plot features
    x_feature: str = "integrated_int_DAPI_norm",
    y_feature: str = "intensity_mean_EdU_nucleus_norm",
    # Data sampling
    cell_number: int | None = 3000,
    # Hue settings
    hue: str | None = None,  # Will auto-detect cell_cycle
    hue_order: list[str] | None = None,
    palette: list[str] | dict[str, str] | None = None,
    # Scale settings
    x_scale: Literal["linear", "log"]
    | None = None,  # Auto-detect based on features
    x_scale_base: int = 2,
    y_scale: Literal["linear", "log"]
    | None = None,  # Auto-detect based on features
    y_scale_base: int = 2,
    # Axis limits
    x_limits: tuple[float, float] | None = None,  # Auto-set for DNA content
    y_limits: tuple[float, float] | None = None,
    # Axis ticks
    x_ticks: list[float] | None = None,
    y_ticks: list[float] | None = None,
    # Scatter settings
    size: float = 2,
    alpha: float = 1.0,
    # KDE overlay
    kde_overlay: bool | None = None,  # Auto-detect based on features
    kde_cmap: str = "rocket_r",
    kde_alpha: float = 0.1,
    # Reference lines
    vline: float | None = None,  # Auto-set for DNA/EdU
    hline: float | None = None,  # Auto-set for DNA/EdU
    line_style: str = "--",
    line_color: str = "black",
    # Display settings
    grid: bool = False,
    show_title: bool = False,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    show_legend: bool = False,
    legend_loc: str = "best",
    legend_title: str | None = None,
    # Threshold settings
    threshold: float | None = None,
    # Figure settings
    fig_size: tuple[float, float] | None = None,
    size_units: str = "cm",
    dpi: int = 300,
    # Save settings
    save: bool = False,
    path: Path | None = None,
    file_format: str = "pdf",
    tight_layout: bool = False,
    # Axes settings
    axes: Axes | None = None,
) -> tuple[Figure, Axes | list[Axes]]:
    """Create a scatter plot with flexible configuration.

    This function provides a high-level API for creating scatter plots with
    support for multiple conditions, log scales, KDE overlays, and threshold-based coloring.

    Args:
        df: DataFrame containing the data
        conditions: Single condition string or list of conditions
        condition_col: Column containing condition labels
        selector_col: Optional column for additional filtering
        selector_val: Optional value for selector_col filtering

        # Plot features
        x_feature: Column name for x-axis (default: DNA content)
        y_feature: Column name for y-axis (default: EdU intensity)

        # Data sampling
        cell_number: Number of cells to sample per condition (None = all)

        # Hue settings
        hue: Column name for color mapping (auto-detects cell_cycle)
        hue_order: Order for hue categories
        palette: Colors for hue categories

        # Scale settings
        x_scale: Scale type for x-axis ("linear" or "log", auto-detected)
        x_scale_base: Base for log scale on x-axis
        y_scale: Scale type for y-axis ("linear" or "log", auto-detected)
        y_scale_base: Base for log scale on y-axis

        # Axis settings
        x_limits: Limits for x-axis (auto-set for DNA content)
        y_limits: Limits for y-axis
        x_ticks: Custom x-axis tick positions
        y_ticks: Custom y-axis tick positions

        # Scatter settings
        size: Size of scatter points
        alpha: Transparency of points

        # KDE overlay
        kde_overlay: Whether to add KDE overlay (auto for DNA vs EdU)
        kde_cmap: Colormap for KDE
        kde_alpha: Transparency of KDE

        # Reference lines
        vline: Vertical reference line position (auto for DNA)
        hline: Horizontal reference line position (auto for EdU)
        line_style: Style for reference lines
        line_color: Color for reference lines

        # Display settings
        grid: Whether to show grid
        show_title: Whether to show title
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        show_legend: Whether to show legend
        legend_loc: Legend location
        legend_title: Legend title

        # Threshold settings
        threshold: Y-value threshold for blue/red coloring

        # Figure settings
        fig_size: Figure size in size_units
        size_units: Units for figure size ("cm" or "inches")
        dpi: Resolution for saved figures

        # Save settings
        save: Whether to save the figure
        path: Directory to save figure
        file_format: Format for saved figure
        tight_layout: Whether to use tight layout

        # Axes settings
        axes: Existing matplotlib Axes to plot on

    Returns:
        tuple: (Figure, Axes or list of Axes)

    Examples:
        Basic cell cycle plot with auto-detection:
        >>> fig, ax = scatter_plot(df, "control")

        Multiple conditions with threshold coloring:
        >>> fig, axes = scatter_plot(
        ...     df,
        ...     ["control", "treatment"],
        ...     y_feature="intensity_mean_p21_nucleus",
        ...     threshold=5000
        ... )

        Custom features with log scales:
        >>> fig, ax = scatter_plot(
        ...     df,
        ...     "control",
        ...     x_feature="area_cell",
        ...     y_feature="intensity_mean_p21_nucleus",
        ...     x_scale="log",
        ...     y_scale="log"
        ... )
    """
    # Prepare KDE parameters dictionary
    kde_params = {
        "fill": True,
        "alpha": kde_alpha,
        "cmap": kde_cmap,
    }

    # Prepare threshold colors dictionary
    threshold_colors = {
        "below": COLOR.BLUE.value,  # Blue for below threshold
        "above": "#DC143C",  # Red for above threshold
    }

    # Calculate dynamic figure size if not provided
    if fig_size is None:
        if isinstance(conditions, str):
            # Single condition: square 7x7 cm (matching BasePlotConfig default)
            fig_size = (7, 7)
        else:
            # Multiple conditions: 7cm per condition width, 7cm height
            n_conditions = len(conditions)
            fig_size = (7 * n_conditions, 7)

    # Create configuration object
    config = ScatterPlotConfig(
        # Plot features
        x_feature=x_feature,
        y_feature=y_feature,
        # Data sampling
        cell_number=cell_number,
        # Hue settings
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        # Scale settings - smart defaults based on features
        x_scale=x_scale or "log",  # Default to log scale for x (usually DNA)
        x_scale_base=x_scale_base,
        y_scale=y_scale
        or ("log" if "EdU" in y_feature else "linear"),  # Log only for EdU
        y_scale_base=y_scale_base,
        # Axis settings
        x_limits=x_limits,
        y_limits=y_limits,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        # Scatter settings
        size=size,
        alpha=alpha,
        # KDE overlay
        kde_overlay=kde_overlay,
        kde_params=kde_params,
        # Reference lines
        vline=vline,
        hline=hline,
        line_style=line_style,
        line_color=line_color,
        # Display settings
        grid=grid,
        show_title=show_title,
        title=title,
        x_label=x_label,
        y_label=y_label,
        show_legend=show_legend,
        legend_loc=legend_loc,
        legend_title=legend_title,
        # Threshold settings
        threshold=threshold,
        threshold_colors=threshold_colors,
        # Figure settings
        fig_size=fig_size,
        size_units=size_units,
        dpi=dpi,
        # Save settings
        save=save,
        path=path,
        file_format=file_format,
        tight_layout=tight_layout,
    )

    # Create plot using class-based approach like other APIs
    plot = ScatterPlot(config)
    return plot.create_plot(
        df=df,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        axes=axes,
    )
