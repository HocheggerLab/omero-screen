"""Module for feature plotting.

This module provides two different feature plots.
1) feature plot where the feature is plotted seperately for each condition.
2) grouped feature plot where the feature is plotted seperately for each condition.

The grouped feature plot is the most complex and is the one that is used in the paper.


"""

import warnings
from pathlib import Path  # noqa: E402
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from omero_screen_plots.stats import (
    set_significance_marks,
)
from omero_screen_plots.utils import (
    COLORS,
    convert_size_to_inches,
    create_standard_boxplot,
    create_standard_violin,
    finalize_plot_with_title,
    format_plot_labels,
    grouped_x_positions,
    prepare_plot_data,
    save_fig,
    select_datapoints,
    selector_val_filter,
    set_y_limits,
    setup_figure,
    show_repeat_points,
)

warnings.filterwarnings("ignore", category=UserWarning)


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
    colors: list[str] = COLORS,
    fig_size: tuple[float, float] = (5, 5),
    size_units: str = "cm",
    scale: bool = False,
    group_size: int = 1,
    within_group_spacing: float = 0.2,
    between_group_gap: float = 0.5,
    save: bool = True,
    path: Optional[Path] = None,
    tight_layout: bool = False,
    file_format: str = "pdf",
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """Plot a feature plot.

    The feature plot is a boxenplot with swarmplot points overlaid.
    Also showing the median points repeat points and significance marks when group_size = 1.
    When group_size > 1, only boxplots and scatterplots are shown without statistical analysis.

    Parameters:
    df: pd.DataFrame - the dataframe to plot
    feature: str - the feature to plot
    conditions: list[str] - the conditions to plot
    axes: Optional[Axes] - the axis to plot on
    x_label: bool - whether to show the x-label
    ymax: float | tuple[float, float] | None - the y-axis maximum value
    condition_col: str - the column name for the conditions
    selector_col: Optional[str] - the column name for the selector
    selector_val: Optional[str] - the value of the selector
    title: Optional[str] - the title of the plot
    colors: list[str] - the colors to use for the plot
    fig_size: tuple[float, float] - the size of the figure
    size_units: str - the units of the figure size
    scale: bool - whether to scale the data
    group_size: int - the number of conditions to group
    within_group_spacing: float - the spacing between conditions within a group
    between_group_gap: float - the gap between groups
    save: bool - whether to save the plot
    path: Optional[Path] - the path to save the plot
    tight_layout: bool - whether to use tight layout
    file_format: str - the format of the saved figure
    dpi: int - the resolution of the saved figure
    """
    if size_units == "cm":
        fig_size = convert_size_to_inches(fig_size, size_units)
    # Prepare data and setup figure
    df_filtered = prepare_plot_data(
        df,
        feature,
        conditions,
        condition_col,
        selector_col,
        selector_val,
        scale,
    )
    fig, ax = setup_figure(axes, fig_size, size_units)
    axes_provided = axes is not None

    # Get x positions
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

    # Use grouped positions only if group_size > 1
    if group_size > 1:
        # Map conditions to x positions
        cond_to_x = dict(zip(conditions, x_positions, strict=False))

        # Create boxplots manually at grouped positions
        df_sampled = select_datapoints(df_filtered, conditions, condition_col)
        for idx, condition in enumerate(conditions):
            cond_data = df_filtered[df_filtered[condition_col] == condition]
            if not cond_data.empty:
                create_standard_boxplot(
                    ax,
                    cond_data[feature].values,
                    x_positions[idx],
                    color=colors[-1],
                )

        # Add swarmplot points
        color_list = [colors[2], colors[3], colors[4], colors[5]]
        plate_ids = df_filtered.plate_id.unique()
        for idx, plate_id in enumerate(plate_ids):
            plate_data = df_sampled[df_sampled.plate_id == plate_id]
            for condition in conditions:
                cond_plate_data = plate_data[
                    plate_data[condition_col] == condition
                ]
                if not cond_plate_data.empty:
                    x_base = cond_to_x[condition]
                    y_values = cond_plate_data[feature].values
                    # Add slight jitter for visibility
                    x_jittered = x_base + np.random.uniform(
                        -0.1, 0.1, size=len(y_values)
                    )
                    ax.scatter(
                        x_jittered,
                        y_values,
                        color=color_list[idx % len(color_list)],
                        alpha=0.8,
                        s=7,
                        edgecolor=None,
                        linewidth=0.5,
                        zorder=3,
                    )
    else:
        # Use standard seaborn plotting when no grouping
        x_positions = list(range(len(conditions)))
        sns.boxenplot(
            data=df_filtered,
            x=condition_col,
            y=feature,
            color=colors[-1],
            order=conditions,
            showfliers=False,
            ax=ax,
        )
        color_list = [colors[2], colors[3], colors[4], colors[5]]
        plate_ids = df_filtered.plate_id.unique()
        df_sampled = select_datapoints(df_filtered, conditions, condition_col)
        for idx, plate_id in enumerate(plate_ids):
            plate_data = df_sampled[df_sampled.plate_id == plate_id]
            sns.swarmplot(
                data=plate_data,
                x=condition_col,
                y=feature,
                color=color_list[idx],  # Use color from palette
                alpha=1,
                size=2,
                edgecolor="white",
                dodge=True,
                order=conditions,
                ax=ax,
            )
    set_y_limits(ax, ymax)
    df_median = (
        df_filtered.groupby(["plate_id", condition_col])[feature]
        .median()
        .reset_index()
    )

    # Only show repeat points and significance marks when group_size = 1
    if group_size == 1:
        show_repeat_points(df_median, conditions, condition_col, feature, ax)
        if len(df.plate_id.unique()) >= 3:
            set_significance_marks(
                ax,
                df_median,
                conditions,
                condition_col,
                feature,
                ax.get_ylim()[1],
            )

    # Format labels and finalize plot
    format_plot_labels(ax, feature, conditions, x_positions, x_label)
    file_name = finalize_plot_with_title(fig, title, feature, axes_provided)

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


def feature_plot_simple(
    df: pd.DataFrame,
    feature: str,
    conditions: list[str],
    ymax: float | tuple[float, float] | None = None,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = "",
    title: Optional[str] = "",
    axes: Optional[Axes] = None,
    x_label: bool = True,
    colors: list[str] = COLORS,
    scale: bool = False,
    legend: Optional[
        tuple[str, list[str]]
    ] = None,  # (legend_title, [label, ...])
    fig_size: tuple[float, float] = (5, 5),
    size_units: str = "cm",
    violin: bool = False,
    save: bool = True,
    path: Optional[Path] = None,
    group_size: int = 1,
    within_group_spacing: float = 0.2,
    between_group_gap: float = 0.5,
) -> tuple[Figure, Axes]:
    """Plot a feature plot.

    Optionally add a legend: legend=(legend_title, [label, ...])
    Colors are assigned automatically from the COLORS list.
    Parameters:
    df: pd.DataFrame - the dataframe to plot
    feature: str - the feature to plot
    conditions: list[str] - the conditions to plot
    ymax: float | tuple[float, float] | None - the y-axis maximum value
    condition_col: str - the column name for the conditions
    selector_col: Optional[str] - the column name for the selector
    selector_val: Optional[str] - the value of the selector
    title: Optional[str] - the title of the plot
    axes: Optional[Axes] - the axis to plot on
    x_label: bool - whether to show the x-label
    colors: list[str] - the colors to use for the plot
    scale: bool - whether to scale the data
    legend: Optional[tuple[str, list[str]]] - the legend for the plot
    fig_size: tuple[float, float] - the size of the figure
    size_units: str - the units of the figure size
    violin: bool - whether to use a violin plot
    save: bool - whether to save the plot
    path: Optional[Path] - the path to save the plot
    """
    # Prepare data and setup figure
    df_filtered = prepare_plot_data(
        df,
        feature,
        conditions,
        condition_col,
        selector_col,
        selector_val,
        scale,
    )
    fig, ax = setup_figure(axes, fig_size, size_units)
    axes_provided = axes is not None

    # Get x positions based on grouping
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

    df_sampled = select_datapoints(df_filtered, conditions, condition_col)
    for idx, condition in enumerate(conditions):
        plate_data = df_sampled[df_sampled[condition_col] == condition]
        if not plate_data.empty:
            if violin:
                create_standard_violin(
                    ax,
                    plate_data[feature].values,
                    x_positions[idx],
                    color=colors[-1],
                )
            else:
                create_standard_boxplot(
                    ax,
                    plate_data[feature].values,
                    x_positions[idx],
                    color=colors[-1],
                )
    set_y_limits(ax, ymax)
    df_median = (
        df_filtered.groupby(["plate_id", condition_col])[feature]
        .median()
        .reset_index()
    )

    # Only show repeat points and significance marks when group_size = 1
    if group_size == 1:
        show_repeat_points(df_median, conditions, condition_col, feature, ax)
        if len(df.plate_id.unique()) >= 3:
            set_significance_marks(
                ax,
                df_median,
                conditions,
                condition_col,
                feature,
                ax.get_ylim()[1],
            )

    # Format labels and finalize plot
    format_plot_labels(ax, feature, conditions, x_positions, x_label)
    file_name = finalize_plot_with_title(fig, title, feature, axes_provided)
    # Add legend if provided
    if legend is not None:
        legend_title, legend_labels = legend
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=COLORS[i % len(COLORS)], label=label)
            for i, label in enumerate(legend_labels)
        ]
        ax.legend(
            handles=handles,
            title=legend_title,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
    if save and path:
        save_fig(
            fig,
            path,
            file_name,
            tight_layout=False,
            fig_extension="pdf",
        )
    # After plotting, format y-axis tick labels to two digits
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

    return fig, ax


def draw_violin_or_box(
    ax: Axes, dat: list[float], xpos: float, color: str, violin: bool
) -> None:
    """Draw a violin or boxplot at the given position with the given color."""
    if violin:
        create_standard_violin(ax, dat, xpos, color=color, alpha=0.9)
    else:
        create_standard_boxplot(ax, dat, xpos, color=color, linewidth=1.5)


# def featureplot_threshold(
#     ax: Optional[Axes],
#     df: pd.DataFrame,
#     conditions: list[str],
#     feature: str,
#     threshold: float,
#     condition_col: str = "condition",
#     selector_col: Optional[str] = "cell_line",
#     selector_val: Optional[str] = "",
#     dimensions: tuple[float, float] = (width, height),
# ) -> None:
#     """Plot a feature plot with a threshold."""
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(dimensions[0], dimensions[1]))
#     # Filter DataFrame using selector_val_filter, as in feature_plot
#     df_filtered = selector_val_filter(
#         df, selector_col, selector_val, condition_col, conditions
#     )
#     assert df_filtered is not None, "No data found"
#     df_filtered["threshold"] = np.where(df[feature] > threshold, "pos", "neg")


def feature_threshold_plot(
    ax: Optional[Axes],
    df: pd.DataFrame,
    conditions: list[str],
    group_size: int = 2,
    feature: str = "feature",
    threshold: float = 0.0,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    colors: list[str] = COLORS,
    repeat_offset: float = 0.18,
    dimensions: tuple[float, float] = (10, 10),
    x_label: bool = True,
    title: Optional[str] = None,
    save: bool = True,
    path: Optional[Path] = None,
) -> None:
    """Create a grouped stacked barplot for thresholded feature (pos/neg) proportions.

    Args:
        ax: Matplotlib axis.
        df: DataFrame containing feature data.
        conditions: List of condition names to plot.
        group_size: Number of groups for x-axis grouping.
        feature: Feature column name.
        threshold: Threshold value for categorization.
        condition_col: Column name for experimental condition.
        selector_col: Column name for selector (e.g., cell line).
        selector_val: Value to filter selector_col by.
        colors: List of colors for plotting.
        repeat_offset: Offset for repeat bars.
        dimensions: Dimensions of the figure.
        x_label: Whether to show the x-axis label.
        title: Plot title.
        save: Whether to save the figure.
        path: Path to save the figure.
    """
    feature_name = feature.split("_")[2]
    if ax is None:
        fig, ax = plt.subplots(figsize=dimensions)
    else:
        fig = ax.figure  # type: ignore[assignment]
    # Filter DataFrame
    df1 = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df1 is not None
    # Create thresholded column
    df1["threshold"] = np.where(df1[feature] > threshold, "pos", "neg")
    n_repeats = 3
    repeat_ids = sorted(df1["plate_id"].unique())[:n_repeats]
    n_conditions = len(conditions)
    x_base_positions = grouped_x_positions(
        n_conditions,
        group_size=group_size,
        within_group_spacing=0.6,
        between_group_gap=0.7,
    )
    # For each condition and replicate, count pos/neg and plot stacked bars as percentage
    for cond_idx, cond in enumerate(conditions):
        for rep_idx, plate_id in enumerate(repeat_ids):
            xpos = x_base_positions[cond_idx] + (rep_idx - 1) * repeat_offset
            plate_data = df1[
                (df1[condition_col] == cond) & (df1["plate_id"] == plate_id)
            ]
            if not plate_data.empty:
                counts = plate_data["threshold"].value_counts()
                total = counts.get("neg", 0) + counts.get("pos", 0)
                if total == 0:
                    neg_pct = pos_pct = 0
                else:
                    neg_pct = counts.get("neg", 0) / total * 100
                    pos_pct = counts.get("pos", 0) / total * 100
                ax.bar(
                    xpos,
                    pos_pct,
                    width=repeat_offset * 1.05,
                    color=colors[-1],
                    edgecolor="white",
                    linewidth=0.7,
                    label="pos" if cond_idx == 0 and rep_idx == 0 else None,
                )
                ax.bar(
                    xpos,
                    neg_pct,
                    width=repeat_offset * 1.05,
                    bottom=pos_pct,
                    color=colors[-2],
                    edgecolor="white",
                    linewidth=0.7,
                    label="neg" if cond_idx == 0 and rep_idx == 0 else None,
                )
    ax.set_xticks(x_base_positions)
    if x_label:
        ax.set_xticklabels(conditions, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    # Draw triplicate boxes
    from matplotlib.patches import Rectangle

    n_repeats = len(repeat_ids)
    bar_width = repeat_offset * 1.05
    y_min = 0
    # Always 100% for the box height
    y_max_box = 100
    for cond_idx, _cond in enumerate(conditions):
        trip_xs = [
            x_base_positions[cond_idx] + (rep_idx - 1) * repeat_offset
            for rep_idx in range(n_repeats)
        ]
        left = min(trip_xs) - bar_width / 2
        right = max(trip_xs) + bar_width / 2
        rect = Rectangle(
            (left, y_min),
            width=right - left,
            height=y_max_box - y_min,
            linewidth=0.5,
            edgecolor="black",
            facecolor="none",
            zorder=10,
        )
        ax.add_patch(rect)
    # Add legend for pos/neg
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=colors[-2],
            edgecolor="white",
            label=f"{feature_name}-",
        ),
        Patch(
            facecolor=colors[-1],
            edgecolor="white",
            label=f"{feature_name}+",
        ),
    ]
    dummy_label = " " * 20
    legend_handles.append(
        Patch(facecolor="none", edgecolor="none", label=dummy_label)
    )
    ax.legend(
        handles=legend_handles,
        title="",
        bbox_to_anchor=(0.95, 1),
        loc="upper left",
        frameon=False,
    )
    if title:
        ax.set_title(title, fontsize=8, weight="bold", y=1.1)
    plt.tight_layout()
    if save and path and title and fig is not None:
        from matplotlib.figure import Figure as MplFigure

        if isinstance(fig, MplFigure):
            file_name = title.replace(" ", "_")
            save_fig(
                fig,
                path,
                file_name,
                tight_layout=False,
                fig_extension="pdf",
            )
