"""Module for count plotting."""

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.colors import COLOR
from omero_screen_plots.stats import set_significance_marks_adaptive
from omero_screen_plots.utils import (
    finalize_plot_with_title,
    grouped_x_positions,
    save_fig,
    setup_figure,
    show_repeat_points_adaptive,
)

pd.options.mode.chained_assignment = None


class PlotType(Enum):
    """Enumeration for plot types: normalised and absolute."""

    NORMALISED = "normalised"
    ABSOLUTE = "absolute"


def norm_count(
    df: pd.DataFrame, norm_control: str, condition: str = "condition"
) -> pd.DataFrame:
    """Normalize count by control condition and return both raw and normalized counts."""
    # First count experiments per well
    well_counts = (
        df.groupby(["plate_id", condition, "well"])["experiment"]
        .count()  # Count experiments per well
        .reset_index()
        .rename(columns={"experiment": "well_count"})
    )

    # Then calculate mean count across wells with same condition
    grouped = (
        well_counts.groupby(["plate_id", condition])["well_count"]
        .mean()  # Average the counts across wells
        .reset_index()
        .rename(columns={"well_count": "count"})
    )

    # Rest of the function remains the same
    pivot_df = grouped.pivot(
        index="plate_id", columns=condition, values="count"
    )
    normalized_df = pivot_df.div(pivot_df[norm_control], axis=0)

    count_df = pivot_df.reset_index().melt(
        id_vars="plate_id", value_name="count", var_name=condition
    )
    norm_df = normalized_df.reset_index().melt(
        id_vars="plate_id", value_name="normalized_count", var_name=condition
    )

    return pd.merge(
        count_df,
        norm_df[["plate_id", condition, "normalized_count"]],
        on=["plate_id", condition],
    )


def prepare_count_data(
    df: pd.DataFrame,
    norm_control: str,
    conditions: list[str],
    condition_col: str,
    selector_col: Optional[str],
    selector_val: Optional[str],
) -> pd.DataFrame:
    """Prepare count data for plotting.

    Args:
        df: Input dataframe
        norm_control: Control condition for normalization
        conditions: List of conditions to include
        condition_col: Column containing condition values
        selector_col: Optional column for filtering
        selector_val: Optional value to filter by

    Returns:
        Processed count data with normalized values
    """
    from omero_screen_plots.utils import selector_val_filter

    df_filtered = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df_filtered is not None, "No data found"
    return norm_count(df_filtered, norm_control, condition=condition_col)


def count_plot(
    df: pd.DataFrame,
    norm_control: str,
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    plot_type: PlotType = PlotType.NORMALISED,
    title: Optional[str] = None,
    colors: Any = COLOR,
    save: bool = False,
    dpi: int = 300,
    tight_layout: bool = False,
    file_format: str = "pdf",
    path: Optional[Path] = None,
    fig_size: tuple[float, float] = (7, 7),
    size_units: str = "cm",
    axes: Optional[Axes] = None,
    group_size: int = 1,
    within_group_spacing: float = 0.2,
    between_group_gap: float = 0.5,
    x_label: bool = True,
) -> tuple[Figure, Axes]:
    """Plot normalized or absolute counts with optional grouping.

    Parameters:
    df: pd.DataFrame - the dataframe to plot
    norm_control: str - the control condition to normalize by
    conditions: list[str] - the conditions to plot
    condition_col: str - the column name for the conditions
    selector_col: Optional[str] - the column name for the selector
    selector_val: Optional[str] - the value of the selector
    plot_type: PlotType - the type of plot to create
    title: Optional[str] - the title of the plot
    colors: list[str] - the colors to use for the plot
    save: bool - whether to save the plot
    dpi: int - the resolution of the plot
    tight_layout: bool - whether to use tight layout
    file_format: str - the format of the plot
    path: Optional[Path] - the path to save the plot
    fig_size: tuple[float, float] - the size of the plot
    size_units: str - the units of the plot size
    axes: Optional[Axes] - the axis to plot on
    group_size: int - the number of conditions to group together
    within_group_spacing: float - the spacing between conditions within a group
    between_group_gap: float - the gap between groups
    x_label: bool - whether to show x-axis labels
    """
    count_col = (
        "normalized_count" if plot_type == PlotType.NORMALISED else "count"
    )

    # Prepare data and setup figure
    counts = prepare_count_data(
        df, norm_control, conditions, condition_col, selector_col, selector_val
    )
    fig, ax = setup_figure(axes, fig_size, size_units)
    axes_provided = axes is not None

    # Get x positions for grouping
    x_positions = (
        grouped_x_positions(
            len(conditions),
            group_size=group_size,
            within_group_spacing=within_group_spacing,
            between_group_gap=between_group_gap,
        )
        if group_size > 1
        else list(range(len(conditions)))
    )
    # Create bars with proper positioning
    if group_size > 1:
        # Manual bar creation for grouped layout
        for idx, condition in enumerate(conditions):
            cond_data = counts[counts[condition_col] == condition]
            if not cond_data.empty:
                ax.bar(
                    x_positions[idx],
                    cond_data[count_col].mean(),
                    width=0.6,
                    color=COLOR.BLUE.value,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.8,
                )
    else:
        # Use seaborn for standard layout
        sns.barplot(
            data=counts,
            x=condition_col,
            y=count_col,
            order=conditions,
            color=COLOR.BLUE.value,
            ax=ax,
        )

    show_repeat_points_adaptive(
        counts,
        conditions,
        condition_col,
        count_col,
        ax,
        group_size,
        x_positions,
    )
    if counts.plate_id.nunique() >= 3:
        set_significance_marks_adaptive(
            ax,
            counts,
            conditions,
            condition_col,
            count_col,
            ax.get_ylim()[1],
            group_size=group_size,
            x_positions=x_positions,
        )

    # Format labels and finalize plot
    if not title:
        title = f"counts {selector_val}"

    # Custom label formatting for count plots
    count_feature = (
        "normalized_count" if plot_type == PlotType.NORMALISED else "count"
    )
    ax.set_ylabel(count_feature)
    ax.set_xlabel("")
    ax.set_xticks(x_positions)

    if x_label:
        ax.set_xticklabels(conditions, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])

    file_name = finalize_plot_with_title(
        fig, title, count_feature, axes_provided
    )

    if save and path:
        save_fig(
            fig,
            path,
            file_name,
            tight_layout=tight_layout,
            fig_extension=file_format,
            resolution=dpi,
        )
    return fig, ax
