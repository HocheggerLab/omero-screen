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
    legend: Optional[tuple[str, list[str]]] = None,
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
    df : pd.DataFrame
        The dataframe to plot containing single-cell measurements.
    feature : str
        The feature column to plot.
    conditions : list[str]
        The conditions to plot.
    axes : Optional[Axes], default=None
        The axis to plot on.
    x_label : bool, default=True
        Whether to show the x-label.
    ymax : float | tuple[float, float] | None, default=None
        The y-axis maximum value.
    condition_col : str, default="condition"
        The column name for the conditions.
    selector_col : Optional[str], default="cell_line"
        The column name for the selector.
    selector_val : Optional[str], default=""
        The value of the selector.
    title : Optional[str], default=""
        The title of the plot.
    colors : list[str], default=COLORS
        The colors to use for the plot.
    fig_size : tuple[float, float], default=(5, 5)
        The size of the figure.
    size_units : str, default="cm"
        The units of the figure size.
    scale : bool, default=False
        Whether to scale the data.
    violin : bool, default=False
        Whether to use violin plots instead of box plots.
    show_scatter : bool, default=True
        Whether to show scatter points overlay.
    legend : Optional[tuple[str, list[str]]], default=None
        Legend configuration as (title, labels) tuple.
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
        legend=legend,
        group_size=group_size,
        within_group_spacing=within_group_spacing,
        between_group_gap=between_group_gap,
        show_x_labels=x_label,
        rotation=45,
        plot_style="standard",
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
