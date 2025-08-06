"""Cell cycle plot API with backward compatibility."""

from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.cellcycleplot_factory import (
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
    title : str | None, default=None
        Plot title. If None, generates default title.
    fig_size : tuple[float, float], default=(6, 6)
        Dimensions of the figure.
    size_units : str, default="cm"
        Units of the figure size ("cm" or "inch").
    colors : list[str] | None, default=None
        List of colors for plotting. If None, uses default phase colors:
        G1=PINK, S=LIGHT_BLUE, G2/M=YELLOW, Polyploid=BLUE.
    save : bool, default=True
        Whether to save the figure.
    path : Optional[Path], default=None
        Path to save the figure.
    tight_layout : bool, default=False
        Whether to use tight layout.
    file_format : str, default="pdf"
        Format of the figure ("pdf", "png", "svg", etc.).
    dpi : int, default=300
        Resolution of the figure.
    show_significance : bool, default=True
        Whether to show significance marks (requires ≥3 plates).
    show_repeat_points : bool, default=True
        Whether to show individual repeat points.
    rotation : int, default=45
        Rotation angle for x-axis labels.
    cc_phases : bool, default=True
        If True, use cell cycle terminology {Sub-G1, G1, S, G2/M, Polyploid}.
        If False, use DNA content terminology {<2N, 2N, S, 4N, >4N}.
    show_subG1 : bool, default=False
        Whether to include Sub-G1/<2N phase in the plot.
    show_plate_legend : bool, default=False
        Whether to show legend with different shapes for each plate_id.

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
