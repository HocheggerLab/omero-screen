"""Feature plot API with backward compatibility."""

from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.colors import COLOR
from omero_screen_plots.featureplot_factory import (
    FeaturePlotConfig,
    StandardFeaturePlot,
)


def feature_plot(
    df: pd.DataFrame,
    feature: str,
    conditions: list[str],
    axes: Optional[Axes] = None,
    x_label: bool = True,
    ymax: float | tuple[float, float] | None = None,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = "",
    title: Optional[str] = "",
    colors: list[str] | None = None,
    fig_size: tuple[float, float] = (5, 5),
    size_units: str = "cm",
    scale: bool = False,
    violin: bool = False,
    show_scatter: bool = True,
    legend: bool = True,
    group_size: int = 1,
    within_group_spacing: float = 0.2,
    between_group_gap: float = 0.5,
    save: bool = True,
    path: Optional[Path] = None,
    tight_layout: bool = False,
    file_format: str = "pdf",
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """Plot a unified feature plot with box/violin plots and optional scatter points.

    This is a backward-compatible wrapper around the new class-based architecture.
    The feature plot can show either boxplots or violin plots, with optional
    scatter points overlaid, plus median points and significance marks. Statistical
    analysis is shown for all group sizes.

    Parameters
    ----------
    Data Filtering
    ^^^^^^^^^^^^^^
    df : pd.DataFrame
        The dataframe to plot containing single-cell measurements.
    feature : str
        The feature column to plot.
    conditions : list[str]
        The conditions to plot.
    condition_col : str, default="condition"
        The column name for the conditions.
    selector_col : Optional[str], default="cell_line"
        The column name for the selector.
    selector_val : Optional[str], default=""
        The value of the selector.

    Display Options
    ^^^^^^^^^^^^^^^
    violin : bool, default=False
        Whether to use violin plots instead of box plots.
    show_scatter : bool, default=True
        Whether to show scatter points overlay.
    ymax : float | tuple[float, float] | None, default=None
        The y-axis maximum value.
    scale : bool, default=False
        Whether to scale the data.

    Grouping & Layout
    ^^^^^^^^^^^^^^^^^
    group_size : int, default=1
        The number of conditions to group.
    within_group_spacing : float, default=0.2
        The spacing between conditions within a group.
    between_group_gap : float, default=0.5
        The gap between groups.

    Styling & Colors
    ^^^^^^^^^^^^^^^^
    title : Optional[str], default=""
        The title of the plot.
    colors : list[str], default=COLORS
        The colors to use for the plot.
    legend : bool, default=True
        Whether to show the default plate legend.
    x_label : bool, default=True
        Whether to show the x-label.
    axes : Optional[Axes], default=None
        The axis to plot on.

    Layout & Sizing
    ^^^^^^^^^^^^^^^
    fig_size : tuple[float, float], default=(5, 5)
        The size of the figure.
    size_units : str, default="cm"
        The units of the figure size.

    Save Options
    ^^^^^^^^^^^^
    save : bool, default=True
        Whether to save the plot.
    path : Optional[Path], default=None
        The path to save the plot.
    file_format : str, default="pdf"
        The format of the saved figure.
    tight_layout : bool, default=False
        Whether to use tight layout.
    dpi : int, default=300
        The resolution of the saved figure.

    Returns:
    -------
    tuple[Figure, Axes]
        The figure and axes objects.
    """
    # Set default colors if None provided
    if colors is None:
        colors = [COLOR.BLUE.value, COLOR.YELLOW.value, COLOR.PINK.value]

    # Create configuration object
    config = FeaturePlotConfig(
        fig_size=fig_size,
        size_units=size_units,
        dpi=dpi,
        save=save,
        file_format=file_format,
        tight_layout=tight_layout,
        path=path,
        title=title if title else None,
        colors=colors,
        scale=scale,
        ymax=ymax,
        violin=violin,
        show_scatter=show_scatter,
        show_legend=legend,
        group_size=group_size,
        within_group_spacing=within_group_spacing,
        between_group_gap=between_group_gap,
        show_x_labels=x_label,
        rotation=45,
        show_significance=True,
        show_repeat_points=True,
    )

    # Use StandardFeaturePlot class
    plot = StandardFeaturePlot(config)
    return plot.create_plot(
        df=df,
        feature=feature,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        axes=axes,
    )


def feature_norm_plot(
    df: pd.DataFrame,
    feature: str,
    conditions: list[str],
    axes: Optional[Axes] = None,
    x_label: bool = True,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = "",
    title: Optional[str] = "",
    color_scheme: str = "green",
    legend: bool = False,  # Default False for norm plots since they have their own legend
    fig_size: tuple[float, float] = (8, 6),
    size_units: str = "cm",
    normalize_by_plate: bool = True,
    threshold: float = 1.5,
    save_norm_qc: bool = False,
    show_triplicates: bool = False,
    show_error_bars: bool = True,
    show_boxes: bool = True,
    group_size: int = 1,
    within_group_spacing: float = 0.2,
    between_group_gap: float = 0.5,
    save: bool = True,
    path: Optional[Path] = None,
    tight_layout: bool = False,
    file_format: str = "pdf",
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """Create a normalized feature plot with threshold-based stacked bars.

    This plot normalizes the feature data by mode (peak=1.0) and displays the
    proportion of cells above/below a threshold as stacked bars. Optionally
    shows individual triplicate data with boxes.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing single-cell measurements.
    feature : str
        The feature column to normalize and plot.
    conditions : list[str]
        The conditions to plot.
    axes : Optional[Axes], default=None
        The axis to plot on.
    x_label : bool, default=True
        Whether to show the x-label.
    condition_col : str, default="condition"
        The column name for the conditions.
    selector_col : Optional[str], default="cell_line"
        The column name for the selector.
    selector_val : Optional[str], default=""
        The value of the selector.
    title : Optional[str], default=""
        The title of the plot.
    color_scheme : str, default="green"
        Color scheme for positive/negative categories. Options:
        - "green": Olive (positive) and Light Green (negative)
        - "blue": Blue (positive) and Light Blue (negative)
        - "purple": Purple (positive) and Lavender (negative)
        Invalid values default to "green".
    legend : bool, default=False
        Whether to show the default plate legend. Default False since
        norm plots have their own threshold legend.
    fig_size : tuple[float, float], default=(8, 6)
        The size of the figure.
    size_units : str, default="cm"
        The units of the figure size.
    normalize_by_plate : bool, default=True
        Whether to normalize within each plate.
    threshold : float, default=1.5
        The threshold value (times mode).
    save_norm_qc : bool, default=False
        Whether to save normalization QC plots. QC plots show the intensity
        distributions before/after normalization to document the process.
        Saved to the same path as the main plot with suffix "_norm_qc".
    show_triplicates : bool, default=False
        Whether to show individual triplicate bars.
    show_error_bars : bool, default=True
        Whether to show error bars on summary bars.
    show_boxes : bool, default=True
        Whether to draw boxes around triplicates.
    group_size : int, default=1
        The number of conditions to group.
    within_group_spacing : float, default=0.2
        The spacing between conditions within a group.
    between_group_gap : float, default=0.5
        The gap between groups.
    save : bool, default=True
        Whether to save the plot.
    path : Optional[Path], default=None
        The path to save the plot.
    tight_layout : bool, default=False
        Whether to use tight layout.
    file_format : str, default="pdf"
        The format of the saved figure.
    dpi : int, default=300
        The resolution of the saved figure.

    Returns:
    -------
    tuple[Figure, Axes]
        The figure and axes objects.

    Examples:
    --------
    >>> # Basic normalized feature plot with default green scheme
    >>> fig, ax = feature_norm_plot(
    ...     df=df,
    ...     feature="intensity_mean_p21_nucleus",
    ...     conditions=["control", "treatment1", "treatment2"],
    ... )

    >>> # Blue color scheme with individual triplicates
    >>> fig, ax = feature_norm_plot(
    ...     df=df,
    ...     feature="intensity_mean_p21_nucleus",
    ...     conditions=["control", "treatment1", "treatment2"],
    ...     color_scheme="blue",
    ...     show_triplicates=True,
    ...     threshold=2.0,  # Custom threshold at 2x mode
    ... )

    >>> # Purple color scheme for publication
    >>> fig, ax = feature_norm_plot(
    ...     df=df,
    ...     feature="intensity_mean_p21_nucleus",
    ...     conditions=["control", "treatment1", "treatment2"],
    ...     color_scheme="purple",
    ... )
    """
    # Set colors based on color scheme
    color_schemes = {
        "green": [
            COLOR.OLIVE.value,
            COLOR.LIGHT_GREEN.value,
        ],  # positive, negative
        "blue": [
            COLOR.BLUE.value,
            COLOR.LIGHT_BLUE.value,
        ],  # positive, negative
        "purple": [
            COLOR.PURPLE.value,
            COLOR.LAVENDER.value,
        ],  # positive, negative
    }

    # Use specified scheme or default to green
    colors = color_schemes.get(color_scheme.lower(), color_schemes["green"])

    # Create configuration object
    config = FeaturePlotConfig(
        fig_size=fig_size,
        size_units=size_units,
        dpi=dpi,
        save=save,
        file_format=file_format,
        tight_layout=tight_layout,
        path=path,
        title=title if title else None,
        colors=colors,
        show_legend=legend,
        normalize_by_plate=normalize_by_plate,
        threshold=threshold,
        save_norm_qc=save_norm_qc,
        show_triplicates=show_triplicates,
        show_error_bars=show_error_bars,
        show_boxes=show_boxes,
        group_size=group_size,
        within_group_spacing=within_group_spacing,
        between_group_gap=between_group_gap,
        show_x_labels=x_label,
        rotation=45,
        repeat_offset=0.18,
        max_repeats=3,
    )

    # Use NormFeaturePlot class
    from omero_screen_plots.featureplot_factory import NormFeaturePlot

    plot = NormFeaturePlot(config)
    return plot.create_plot(
        df=df,
        feature=feature,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        axes=axes,
    )
