"""Histogram plot API with backward compatibility."""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.colors import COLOR
from omero_screen_plots.histogramplot_factory import (
    HistogramPlot,
    HistogramPlotConfig,
)


def histogram_plot(
    df: pd.DataFrame,
    feature: str,
    conditions: str | list[str],
    condition_col: str = "condition",
    selector_col: Optional[str] = None,
    selector_val: Optional[str] = None,
    axes: Optional[Axes] = None,
    title: Optional[str] = None,
    show_title: bool = False,
    colors: list[str] | None = None,
    fig_size: tuple[float, float]
    | None = None,  # Now None by default for dynamic sizing
    size_units: str = "cm",
    bins: int | str = 100,
    log_scale: bool = False,
    log_base: float = 2,
    x_limits: tuple[float, float] | None = None,
    normalize: bool = False,
    kde_overlay: bool = False,
    kde_smoothing: float = 0.8,
    kde_params: dict[str, Any] | None = None,
    show_x_labels: bool = True,
    rotation: int = 0,
    save: bool = False,
    path: Optional[Path] = None,
    tight_layout: bool = False,
    file_format: str = "pdf",
    dpi: int = 300,
) -> tuple[Figure, Axes | list[Axes]]:
    """Create histogram plot(s) for single or multiple conditions.

    This function creates histograms using seaborn.histplot with support for
    log scaling, custom binning, normalization, and KDE overlays.

    - If conditions is a string: Creates single histogram for that condition
    - If conditions is a list: Creates subplots with one histogram per condition

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data to plot.
    feature : str
        The column name to create histogram for.
    conditions : str | list[str]
        Single condition (string) or multiple conditions (list) to plot.
    condition_col : str, default="condition"
        The column name for condition labels.
    selector_col : Optional[str], default=None
        The column name for additional filtering.
    selector_val : Optional[str], default=None
        The value to filter by if selector_col is provided.
    axes : Optional[Axes], default=None
        The matplotlib axes to plot on. Only valid for single condition (string).
        For multiple conditions, creates subplots automatically.
    title : Optional[str], default=None
        The title of the plot. If None, auto-generates from feature and condition.
    show_title : bool, default=False
        Whether to display the title. When True, positions title at y-axis start.
    colors : list[str] | None, default=None
        Colors to use for the histogram. Uses first color in list.
    fig_size : tuple[float, float] | None, default=None
        The size of the figure in the specified units. If None, uses dynamic defaults:
        - Single condition: (4, 4) cm
        - Multiple conditions: (4 * num_conditions, 4) cm
    size_units : str, default="cm"
        The units for figure size ("cm" or "inches").
    bins : int | str, default=100
        Number of bins or binning strategy. Default is 100 bins.
        Can be int for exact number or str for methods like "auto", "sturges", etc.
        For multiple conditions with int bins, calculates unified bin edges across
        all conditions for consistent visualization.
    log_scale : bool, default=False
        Whether to use logarithmic scaling for x-axis.
    log_base : float, default=2
        Base for logarithmic scaling (only used if log_scale=True).
    x_limits : tuple[float, float] | None, default=None
        X-axis limits as (min, max). If None, uses data range.
    normalize : bool, default=False
        Whether to normalize histogram to show density instead of counts.
    kde_overlay : bool, default=False
        Whether to overlay KDE curves. When True, shows only KDE lines (no histograms)
        in a single plot with different colors for each condition.
    kde_smoothing : float, default=0.8
        KDE smoothing factor (bw_adjust parameter). Lower = smoother curves.
        Typical range: 0.5 (very smooth) to 2.0 (more detailed).
    kde_params : dict[str, Any] | None, default=None
        Additional KDE parameters. Common options:
        - bw_method: 'scott' (default), 'silverman', or float for manual bandwidth
        - gridsize: int (default 200) - higher = smoother curves
        - cut: float (default 2) - how far to extend beyond data range
    show_x_labels : bool, default=True
        Whether to show x-axis tick labels.
    rotation : int, default=0
        Rotation angle for x-axis tick labels.
    save : bool, default=False
        Whether to save the plot to file.
    path : Optional[Path], default=None
        Directory path for saving the plot (required if save=True).
    tight_layout : bool, default=False
        Whether to use tight layout when saving.
    file_format : str, default="pdf"
        File format for saved plot ("pdf", "png", "svg", etc.).
    dpi : int, default=300
        Resolution for saved plot in dots per inch.

    Returns:
    -------
    tuple[Figure, Axes | list[Axes]]
        The matplotlib Figure and Axes object(s). Returns:
        - (Figure, Axes) for single condition (string input)
        - (Figure, list[Axes]) for multiple conditions (list input)

    Raises:
    ------
    ValueError
        If required parameters are missing or invalid.
        If specified columns are not found in dataframe.
        If no data remains after filtering.

    Examples:
    --------
    >>> # Basic histogram for one condition
    >>> fig, ax = histogram_plot(
    ...     df=df,
    ...     feature="integrated_int_DAPI_norm",
    ...     condition="control"
    ... )

    >>> # Log-scale histogram with KDE overlay
    >>> fig, ax = histogram_plot(
    ...     df=df,
    ...     feature="integrated_int_DAPI_norm",
    ...     condition="control",
    ...     log_scale=True,
    ...     log_base=2,
    ...     kde_overlay=True,
    ...     x_limits=(1, 16)
    ... )

    >>> # Histogram with custom styling
    >>> fig, ax = histogram_plot(
    ...     df=df,
    ...     feature="area_nucleus",
    ...     condition="treatment1",
    ...     colors=["#1f77b4"],
    ...     title="Nucleus Area Distribution - Treatment 1",
    ...     bins=50,
    ...     normalize=True
    ... )

    >>> # Multiple histograms using subplots
    >>> fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    >>> conditions = ["control", "treatment1", "treatment2"]
    >>> for i, condition in enumerate(conditions):
    ...     histogram_plot(
    ...         df=df,
    ...         feature="integrated_int_DAPI_norm",
    ...         condition=condition,
    ...         axes=axes[i],
    ...         title=f"Histogram: {condition}"
    ...     )

    Notes:
    -----
    This function creates histograms for single conditions. For multiple histograms:
    - Use this function multiple times with different axes
    - Use combplot functionality for complex layouts
    - Create subplots manually and pass axes parameter

    The function is designed for flexibility - it can integrate with existing plots
    by accepting an axes parameter, or create standalone plots when axes=None.

    For DNA content histograms (common in cell cycle analysis), consider:
    - Using log_scale=True with log_base=2
    - Setting x_limits based on expected DNA content range
    - Using normalize=True for comparing different sample sizes
    """
    # Set default colors if None provided
    if colors is None:
        if isinstance(conditions, str):
            # Single condition: use only blue
            colors = [COLOR.BLUE.value]
        else:
            # Multiple conditions: use blue for all unless KDE overlay is enabled
            if kde_overlay:
                # With KDE overlay, need different colors to distinguish conditions on single plot
                colors = [
                    COLOR.BLUE.value,
                    COLOR.YELLOW.value,
                    COLOR.PINK.value,
                    COLOR.LIGHT_GREEN.value,
                ]
            else:
                # Without KDE, use blue for all (since each has its own subplot)
                colors = [COLOR.BLUE.value]

    # Set default KDE parameters
    if kde_params is None:
        kde_params = {}

    # Calculate dynamic figure size if not provided
    if fig_size is None:
        if isinstance(conditions, str):
            # Single condition: square 4x4 cm
            fig_size = (4, 4)
        else:
            # Multiple conditions: 4cm per condition width, 4cm height
            n_conditions = len(conditions)
            fig_size = (4 * n_conditions, 4)

    # Create configuration object
    config = HistogramPlotConfig(
        fig_size=fig_size,
        size_units=size_units,
        dpi=dpi,
        save=save,
        file_format=file_format,
        tight_layout=tight_layout,
        path=path,
        title=title,
        show_title=show_title,
        colors=colors,
        bins=bins,
        log_scale=log_scale,
        log_base=log_base,
        x_limits=x_limits,
        normalize=normalize,
        kde_overlay=kde_overlay,
        kde_smoothing=kde_smoothing,
        kde_params=kde_params,
        show_x_labels=show_x_labels,
        rotation=rotation,
    )

    # Create histogram plot and delegate to factory
    plot = HistogramPlot(config)
    return plot.create_plot(
        df=df,
        feature=feature,
        conditions=conditions,  # Pass through as-is (string or list)
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        axes=axes,
    )
