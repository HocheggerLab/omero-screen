"""Count plot API with backward compatibility."""

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.colors import COLOR
from omero_screen_plots.countplot_factory import (
    CountPlot,
    CountPlotConfig,
    PlotType,
)


def count_plot(
    df: pd.DataFrame,
    norm_control: str,
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: str | None = "cell_line",
    selector_val: str | None = None,
    plot_type: PlotType = PlotType.NORMALISED,
    title: str | None = None,
    colors: Any = COLOR,
    save: bool = False,
    dpi: int = 300,
    tight_layout: bool = False,
    file_format: str = "pdf",
    path: Path | None = None,
    fig_size: tuple[float, float] = (7, 7),
    size_units: str = "cm",
    axes: Axes | None = None,
    group_size: int = 1,
    within_group_spacing: float = 0.2,
    between_group_gap: float = 0.5,
    x_label: bool = True,
) -> tuple[Figure, Axes]:
    """Plot normalized or absolute counts with optional grouping.

    This is a backward-compatible wrapper around the simplified architecture.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to plot containing single-cell measurements.
    norm_control : str
        The control condition to normalize by (required even for absolute plots).
    conditions : list[str]
        The conditions to plot.
    condition_col : str, default="condition"
        The column name for the conditions.
    selector_col : str | None, default="cell_line"
        The column name for the selector.
    selector_val : str | None, default=None
        The value of the selector.
    plot_type : PlotType, default=PlotType.NORMALISED
        The type of plot to create (NORMALISED or ABSOLUTE).
    title : str | None, default=None
        The title of the plot.
    colors : Any, default=COLOR
        The colors to use for the plot.
    save : bool, default=False
        Whether to save the plot.
    dpi : int, default=300
        The resolution of the plot.
    tight_layout : bool, default=False
        Whether to use tight layout.
    file_format : str, default="pdf"
        The format of the plot.
    path : Path | None, default=None
        The path to save the plot.
    fig_size : tuple[float, float], default=(7, 7)
        The size of the plot.
    size_units : str, default="cm"
        The units of the plot size.
    axes : Axes | None, default=None
        The axis to plot on.
    group_size : int, default=1
        The number of conditions to group together.
    within_group_spacing : float, default=0.2
        The spacing between conditions within a group.
    between_group_gap : float, default=0.5
        The gap between groups.
    x_label : bool, default=True
        Whether to show x-axis labels.

    Returns:
    -------
    tuple[Figure, Axes]
        The figure and axes objects.
    """
    # Create configuration object
    config = CountPlotConfig(
        fig_size=fig_size,
        size_units=size_units,
        dpi=dpi,
        save=save,
        file_format=file_format,
        tight_layout=tight_layout,
        path=path,
        title=title,
        colors=colors
        if isinstance(colors, list)
        else [colors.value]
        if hasattr(colors, "value")
        else [colors],
        plot_type=plot_type,
        group_size=group_size,
        within_group_spacing=within_group_spacing,
        between_group_gap=between_group_gap,
        show_x_labels=x_label,
        rotation=45,
    )

    # Use simplified CountPlot class
    plot = CountPlot(config)
    return plot.create_plot(
        df=df,
        norm_control=norm_control,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        axes=axes,
    )
