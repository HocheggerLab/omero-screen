"""Cell cycle plot API with backward compatibility."""

from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.cellcycleplot_factory import (
    StackedCellCyclePlot,
    StackedCellCyclePlotConfig,
    StandardCellCyclePlot,
    StandardCellCyclePlotConfig,
)
from omero_screen_plots.colors import COLOR


def cellcycle_plot(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: str | None = "cell_line",
    selector_val: str | None = None,
    title: str | None = None,
    fig_size: tuple[float, float] = (6, 6),
    size_units: str = "cm",
    colors: list[str] | None = None,
    save: bool = True,
    path: Optional[Path] = None,
    tight_layout: bool = False,
    file_format: str = "pdf",
    dpi: int = 300,
    show_significance: bool = True,
    show_repeat_points: bool = True,
    rotation: int = 45,
    cc_phases: bool = True,
    show_subG1: bool = False,
    show_plate_legend: bool = False,
) -> tuple[Figure, list[Axes]]:
    """Plot cell cycle phases in a 2x2 subplot grid with statistical analysis.

    This is a backward-compatible wrapper around the new class-based architecture.
    Creates a 2x2 subplot grid showing each cell cycle phase (G1, S, G2/M, Polyploid)
    separately with repeat points and significance marks.

    Note: Unlike other plot functions, this returns (Figure, list of Axes) because
    it creates multiple subplots that cannot be collapsed into a single axis.

    Parameters
    ----------
    Data Filtering
    ^^^^^^^^^^^^^^
    df : pd.DataFrame
        DataFrame containing cell cycle data with required columns:
        'cell_cycle', 'plate_id', and condition_col.
    conditions : list[str]
        List of condition names to plot.
    condition_col : str, default="condition"
        Column name for experimental condition.
    selector_col : str | None, default="cell_line"
        Column name for selector (e.g., cell line).
    selector_val : str | None, default=None
        Value to filter selector_col by.

    Display Options
    ^^^^^^^^^^^^^^^
    cc_phases : bool, default=True
        If True, use cell cycle terminology {Sub-G1, G1, S, G2/M, Polyploid}.
        If False, use DNA content terminology {<2N, 2N, S, 4N, >4N}.
    show_subG1 : bool, default=False
        Whether to include Sub-G1/<2N phase in the plot.
    show_significance : bool, default=True
        Whether to show significance marks (requires ≥3 plates).
    show_repeat_points : bool, default=True
        Whether to show individual repeat points.
    show_plate_legend : bool, default=False
        Whether to show legend with different shapes for each plate_id.

    Styling & Colors
    ^^^^^^^^^^^^^^^^
    title : str | None, default=None
        Plot title. If None, generates default title.
    fig_size : tuple[float, float], default=(6, 6)
        Dimensions of the figure.
    size_units : str, default="cm"
        Units of the figure size ("cm" or "inch").
    colors : list[str] | None, default=None
        List of colors for plotting. If None, uses default phase colors:
        G1=PINK, S=LIGHT_BLUE, G2/M=YELLOW, Polyploid=BLUE.
    rotation : int, default=45
        Rotation angle for x-axis labels.
    tight_layout : bool, default=False
        Whether to use tight layout.

    Save Options
    ^^^^^^^^^^^^
    save : bool, default=True
        Whether to save the figure.
    path : Optional[Path], default=None
        Path to save the figure.
    file_format : str, default="pdf"
        Format of the figure ("pdf", "png", "svg", etc.).
    dpi : int, default=300
        Resolution of the figure.

    Returns:
    -------
    tuple[Figure, list]
        The figure object and list of 4 axes objects [top-left, top-right, bottom-left, bottom-right].

    Examples:
    --------
    >>> fig, axes = cellcycle_plot(
    ...     df=data,
    ...     conditions=["control", "treatment"],
    ...     selector_val="MCF10A",
    ...     save=False
    ... )
    >>> plt.show()

    >>> # Custom colors
    >>> fig, axes = cellcycle_plot(
    ...     df=data,
    ...     conditions=["ctrl", "drug1", "drug2"],
    ...     colors=["red", "green", "blue", "purple"],
    ...     title="Custom Cell Cycle Analysis"
    ... )
    """
    # Set default colors if None provided - use specific phase colors
    if colors is None:
        colors = [
            COLOR.PINK.value,  # G1
            COLOR.LIGHT_BLUE.value,  # S
            COLOR.YELLOW.value,  # G2/M
            COLOR.BLUE.value,  # Polyploid
        ]

    # Create configuration object
    config = StandardCellCyclePlotConfig(
        fig_size=fig_size,
        size_units=size_units,
        dpi=dpi,
        save=save,
        file_format=file_format,
        tight_layout=tight_layout,
        path=path,
        title=title,
        colors=colors,
        show_significance=show_significance,
        show_repeat_points=show_repeat_points,
        rotation=rotation,
        cc_phases=cc_phases,
        show_subG1=show_subG1,
        show_plate_legend=show_plate_legend,
    )

    # Use StandardCellCyclePlot class
    plot = StandardCellCyclePlot(config)
    return plot.create_plot(
        df=df,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
    )


def cellcycle_stacked(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: str | None = "cell_line",
    selector_val: str | None = None,
    axes: Axes | None = None,
    title: str | None = None,
    fig_size: tuple[float, float] = (6, 6),
    size_units: str = "cm",
    colors: list[str] | None = None,
    save: bool = False,
    path: Path | None = None,
    tight_layout: bool = False,
    file_format: str = "pdf",
    dpi: int = 300,
    # Display options
    show_triplicates: bool = False,
    show_error_bars: bool = True,
    show_boxes: bool = True,
    # Phase options
    cc_phases: bool = True,
    phase_order: list[str] | None = None,
    # Grouping options
    group_size: int = 1,
    within_group_spacing: float = 0.2,
    between_group_gap: float = 0.5,
    # Bar options
    bar_width: float = 0.5,
    repeat_offset: float = 0.18,
    max_repeats: int = 3,
    # Axis options
    x_label: bool = True,
    rotation: int = 45,
    # Legend options
    show_legend: bool = True,
) -> tuple[Figure, Axes]:
    """Create a stacked barplot for cell cycle phase proportions with flexible display modes.

    This is a unified wrapper around the new StackedCellCyclePlot class that can create
    either summary stacked bars (with error bars) or individual triplicate bars (with boxes).

    Parameters
    ----------
    Data Filtering
    ^^^^^^^^^^^^^^
    df : pd.DataFrame
        DataFrame containing cell cycle data with required columns:
        'cell_cycle', 'plate_id', and condition_col.
    conditions : list[str]
        List of condition names to plot.
    condition_col : str, default="condition"
        Column name for experimental condition.
    selector_col : str | None, default="cell_line"
        Column name for selector (e.g., cell line).
    selector_val : str | None, default=None
        Value to filter selector_col by.

    Display Options
    ^^^^^^^^^^^^^^^
    show_triplicates : bool, default=False
        If True, show individual bars for each replicate/plate with boxes.
        If False, show summary bars with optional error bars.
    show_error_bars : bool, default=True
        Whether to show error bars (only applies when show_triplicates=False).
    show_boxes : bool, default=True
        Whether to draw boxes around triplicates (only applies when show_triplicates=True).
    cc_phases : bool, default=True
        If True, use cell cycle terminology {Sub-G1, G1, S, G2/M, Polyploid}.
        If False, use DNA content terminology {<2N, 2N, S, 4N, >4N}.
    phase_order : list[str] | None, default=None
        Custom list of cell cycle phases to plot. If None, uses all available phases.
    axes : Axes | None, default=None
        Matplotlib axis. If None, a new figure is created.
    x_label : bool, default=True
        Whether to show the x-axis labels.
    rotation : int, default=45
        Rotation angle for x-axis labels.
    show_legend : bool, default=True
        Whether to show the legend with cell cycle phase colors.

    Grouping & Layout
    ^^^^^^^^^^^^^^^^^
    group_size : int, default=1
        Number of conditions per group on the x-axis (1 = no grouping).
    within_group_spacing : float, default=0.2
        Spacing between bars within a group.
    between_group_gap : float, default=0.5
        Spacing between groups.
    bar_width : float, default=0.5
        Width of each bar.
    repeat_offset : float, default=0.18
        Offset between replicate bars (only applies when show_triplicates=True).
    max_repeats : int, default=3
        Maximum number of replicates to show (only applies when show_triplicates=True).

    Styling & Colors
    ^^^^^^^^^^^^^^^^
    title : str | None, default=None
        Plot title.
    fig_size : tuple[float, float], default=(6, 6)
        Dimensions of the figure.
    size_units : str, default="cm"
        Units of the figure size ("cm" or "inch").
    colors : list[str] | None, default=None
        List of colors for plotting phases.
    tight_layout : bool, default=False
        Whether to use tight layout.

    Save Options
    ^^^^^^^^^^^^
    save : bool, default=False
        Whether to save the figure.
    path : Path | None, default=None
        Path to save the figure.
    file_format : str, default="pdf"
        Format of the figure.
    dpi : int, default=300
        Resolution of the figure.

    Returns:
    -------
    tuple[Figure, Axes]
        The figure and axes objects.

    Examples:
    --------
    >>> # Simple summary stacked plot with error bars
    >>> fig, ax = cellcycle_stacked(
    ...     df=data,
    ...     conditions=["control", "treatment"],
    ...     selector_val="MCF10A"
    ... )

    >>> # Triplicate stacked plot with boxes
    >>> fig, ax = cellcycle_stacked(
    ...     df=data,
    ...     conditions=["control", "treatment"],
    ...     show_triplicates=True,
    ...     selector_val="MCF10A"
    ... )

    >>> # Grouped stacked plot with custom phases
    >>> fig, ax = cellcycle_stacked(
    ...     df=data,
    ...     conditions=["ctrl", "drug1", "drug2", "drug3"],
    ...     group_size=2,  # Group in pairs
    ...     cc_phases=True,  # Use cell cycle terminology
    ...     phase_order=["G1", "S", "G2/M"]  # Exclude Sub-G1 and Polyploid
    ... )

    >>> # Plot without legend (for subplots or custom legends)
    >>> fig, ax = cellcycle_stacked(
    ...     df=data,
    ...     conditions=["control", "treatment"],
    ...     show_legend=False
    ... )
    """
    # Don't set default colors here - let the factory handle it
    # Only pass colors if explicitly provided by the user

    # Create configuration with explicit parameters to avoid typing issues
    config = StackedCellCyclePlotConfig(
        fig_size=fig_size,
        size_units=size_units,
        dpi=dpi,
        save=save,
        file_format=file_format,
        tight_layout=tight_layout,
        path=path,
        title=title,
        colors=colors or [],
        # Display options
        show_triplicates=show_triplicates,
        show_error_bars=show_error_bars,
        show_boxes=show_boxes,
        # Phase options
        cc_phases=cc_phases,
        phase_order=phase_order,
        # Grouping options
        group_size=group_size,
        within_group_spacing=within_group_spacing,
        between_group_gap=between_group_gap,
        # Bar options
        bar_width=bar_width,
        repeat_offset=repeat_offset,
        max_repeats=max_repeats,
        # Axis options
        rotation=rotation if x_label else 0,
        # Legend options
        show_legend=show_legend,
    )

    # Use StackedCellCyclePlot class
    plot = StackedCellCyclePlot(config)
    return plot.create_plot(
        df=df,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        axes=axes,
    )


# Backward compatibility alias
def cellcycle_grouped(
    df: pd.DataFrame,
    conditions: list[str],
    group_size: int = 2,
    condition_col: str = "condition",
    selector_col: str | None = "cell_line",
    selector_val: str | None = None,
    phases: list[str] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    x_label: bool = True,
    fig_size: tuple[float, float] = (6, 6),
    size_units: str = "cm",
    colors: list[str] | None = None,
    save: bool = True,
    path: Path | None = None,
    tight_layout: bool = False,
    file_format: str = "pdf",
    dpi: int = 300,
    repeat_offset: float = 0.18,
    within_group_spacing: float = 0.2,
    between_group_gap: float = 0.5,
) -> tuple[Figure, Axes]:
    """Create a grouped stacked barplot with triplicates boxed.

    DEPRECATED: Use cellcycle_stacked with show_triplicates=True instead.

    This function is maintained for backward compatibility and calls
    cellcycle_stacked with show_triplicates=True.
    """
    import warnings

    warnings.warn(
        "cellcycle_grouped is deprecated. Use cellcycle_stacked with show_triplicates=True instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return cellcycle_stacked(
        df=df,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        axes=ax,
        title=title,
        fig_size=fig_size,
        size_units=size_units,
        colors=colors,
        save=save,
        path=path,
        tight_layout=tight_layout,
        file_format=file_format,
        dpi=dpi,
        show_triplicates=True,  # Key difference - enable triplicate mode
        show_boxes=True,
        group_size=group_size,
        repeat_offset=repeat_offset,
        within_group_spacing=within_group_spacing,
        between_group_gap=between_group_gap,
        phase_order=phases,
        x_label=x_label,
    )
