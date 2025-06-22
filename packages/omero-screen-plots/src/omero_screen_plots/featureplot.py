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
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from omero_screen_plots import COLOR
from omero_screen_plots.stats import (
    set_grouped_significance_marks,
    set_significance_marks,
)
from omero_screen_plots.utils import (
    grouped_x_positions,
    save_fig,
    scale_data,
    select_datapoints,
    selector_val_filter,
    show_repeat_points,
)

warnings.filterwarnings("ignore", category=UserWarning)
width = 9 / 2.54  # 9 cm
height = 4 / 2.54  # 4.5 cm
current_dir = Path(__file__).parent
style_path = (current_dir / "../../hhlab_style01.mplstyle").resolve()
plt.style.use(style_path)
prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]


def feature_plot(
    df: pd.DataFrame,
    feature: str,
    conditions: list[str],
    ymax: float | tuple[float, float] | None = None,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = "",
    title: Optional[str] = "",
    colors: list[str] = COLORS,
    scale: bool = False,
    save: bool = True,
    path: Optional[Path] = None,
    legend: Optional[
        tuple[str, list[str]]
    ] = None,  # (legend_title, [label, ...])
    height: float = height,
    violin: bool = False,
) -> None:
    """Plot a feature plot.

    Optionally add a legend: legend=(legend_title, [label, ...])
    Colors are assigned automatically from the COLORS list.
    Parameters:
    df: pd.DataFrame
    feature: str
    conditions: list[str]
    ymax: float | tuple[float, float] | None
    condition_col: str
    selector_col: Optional[str]
    """
    df_filtered = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df_filtered is not None, "No data found"
    if scale:
        df_filtered = scale_data(df_filtered, feature)

    fig, ax = plt.subplots(figsize=(height, height))
    color_list = [colors[2], colors[3], colors[4], colors[5]]
    plate_ids = df_filtered.plate_id.unique()
    df_sampled = select_datapoints(df_filtered, conditions, condition_col)
    for idx, plate_id in enumerate(plate_ids):
        plate_data = df_sampled[df_sampled.plate_id == plate_id]
        if violin:
            vp = ax.violinplot(
                [plate_data[feature]],
                positions=[idx],
                widths=0.5,
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )
            bodies = (
                vp["bodies"]
                if isinstance(vp["bodies"], list)
                else [vp["bodies"]]
            )
            for body in bodies:
                body.set_facecolor(color_list[idx])
                # body.set_edgecolor("black")
                body.set_alpha(0.75)
                # xbody.set_linewidth(5)
            if "cmedians" in vp:
                vp["cmedians"].set_color(color_list[idx])
                vp["cmedians"].set_linewidth(2)
        else:
            ax.boxplot(
                [plate_data[feature]],
                positions=[idx],
                widths=0.5,
                showfliers=False,
                patch_artist=True,
                boxprops={
                    "facecolor": color_list[idx],
                    "edgecolor": "black",
                    "linewidth": 1.5,
                    "alpha": 0.75,
                },
                medianprops={"color": color_list[idx], "linewidth": 2},
                whiskerprops={"color": "black", "linewidth": 1.2},
                capprops={"color": "black", "linewidth": 1.2},
            )
    if ymax:
        if isinstance(ymax, tuple):
            ax.set_ylim(ymax[0], ymax[1])  # unpack tuple into min and max
        else:
            ax.set_ylim(
                0, ymax
            )  # assume 0 as minimum if single value provided
    df_median = (
        df_filtered.groupby(["plate_id", condition_col])[feature]
        .median()
        .reset_index()
    )

    show_repeat_points(df_median, conditions, condition_col, feature, ax)
    if len(df.plate_id.unique()) >= 3:
        set_significance_marks(
            ax, df_median, conditions, condition_col, feature, ax.get_ylim()[1]
        )
    ax.set_ylabel(feature)
    ax.set_xlabel("")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    if not title:
        title = feature
    fig.suptitle(title, fontsize=7, weight="bold", x=0, y=1.05, ha="left")
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
    file_name = title.replace(" ", "_")
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


def draw_violin_or_box(
    ax: Axes, dat: list[float], xpos: float, color: str, violin: bool
) -> None:
    """Draw a violin or boxplot at the given position with the given color."""
    if violin:
        vp = ax.violinplot(
            [dat],
            positions=[xpos],
            widths=0.5,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        bodies = (
            vp["bodies"] if isinstance(vp["bodies"], list) else [vp["bodies"]]
        )
        for body in bodies:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.9)
            body.set_linewidth(0.5)
        if "cmedians" in vp:
            vp["cmedians"].set_color(color)
            vp["cmedians"].set_linewidth(2)
    else:
        ax.boxplot(
            [dat],
            positions=[xpos],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops={
                "facecolor": color,
                "edgecolor": "black",
                "linewidth": 1.5,
                "alpha": 0.75,
            },
            medianprops={"color": color, "linewidth": 2},
            whiskerprops={"color": "black", "linewidth": 1.2},
            capprops={"color": "black", "linewidth": 1.2},
        )


def grouped_feature_plot(
    ax: Optional[Axes],
    df: pd.DataFrame,
    feature: str,
    conditions: list[str],
    group_size: int = 2,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = "",
    ymax: float | tuple[float, float] | None = None,
    legend: Optional[tuple[str, list[str]]] = None,
    dimensions: tuple[float, float] = (width, height),
    x_label: bool = True,
    y_label: Optional[str] = None,
    violin: bool = False,
    color=COLOR.LAVENDER.value,
    title: Optional[str] = None,
    save: bool = False,
    path: Optional[Path] = None,
) -> None:
    """Plot grouped boxplots for a feature, grouping conditions for comparison.

    Optionally add a legend: legend=(legend_title, [label, ...])
    Colors are assigned automatically from the COLORS list.

    Parameters:
    df: pd.DataFrame
    feature: str
    conditions: list[str]
    group_size: int
    condition_col: str
    selector_col: Optional[str]
    selector_val: Optional[str]
    ymax: float | tuple[float, float] | None
    legend: Optional[tuple[str, list[str]]]
    dim: tuple[float, float]
    x_label: bool
    height: float
    violin: bool
    title: str
    save: bool
    path: Optional[Path]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(dimensions[0], dimensions[1]))
    # Filter DataFrame using selector_val_filter, as in feature_plot
    df_filtered = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df_filtered is not None, "No data found"

    # Scale the feature data by 100 for percentage-style y-axis
    # (we need to work on a copy to avoid modifying the caller's DataFrame)
    df_filtered = df_filtered.copy()
    df_filtered[feature] = df_filtered[feature] * 100
    # After scaling the data we continue with the usual plotting workflow
    x_positions = grouped_x_positions(
        len(conditions), group_size=group_size, between_group_gap=0.75
    )
    data_for_plot = [
        df_filtered[df_filtered[condition_col] == cond][feature]
        for cond in conditions
    ]
    for i, (dat, xpos) in enumerate(
        zip(data_for_plot, x_positions, strict=False)
    ):
        draw_violin_or_box(ax, dat, xpos, color, violin)

    # Overlay repeat points (medians per plate_id per condition) with different markers
    df_median = (
        df_filtered.groupby(["plate_id", condition_col])[feature]
        .median()
        .reset_index()
    )
    # Map condition to x-position
    cond_to_x = dict(zip(conditions, x_positions, strict=False))
    markers = ["o", "s", "^"]  # circle, square, triangle
    plate_ids = df_median["plate_id"].unique()
    plate_id_to_marker = {
        pid: markers[i % len(markers)] for i, pid in enumerate(plate_ids)
    }
    scatter_handles = {}
    for cond in conditions:
        cond_medians = df_median[df_median[condition_col] == cond]
        x_base = np.full(len(cond_medians), cond_to_x[cond])
        jitter = np.random.uniform(-0.05, 0.05, size=len(cond_medians))
        x = x_base + jitter
        y = cond_medians[feature].values
        for i, (xi, yi, pid) in enumerate(
            zip(x, y, cond_medians["plate_id"], strict=False)
        ):
            marker = plate_id_to_marker[pid]
            handle = ax.scatter(
                xi,
                yi,
                color="black",
                edgecolor="white",
                s=18,
                zorder=4,
                marker=marker,
            )
            # Store one handle per marker for legend
            if marker not in scatter_handles:
                scatter_handles[marker] = handle
    ax.set_xticks(x_positions)
    if x_label:
        ax.set_xticklabels(conditions, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_xlabel("")
    if y_label is None:
        ax.set_ylabel(f"{feature} ×100")
    else:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, fontsize=8, weight="bold", y=1.1)
    # Add legend for repeat medians with marker shapes
    from matplotlib.lines import Line2D

    marker_labels = ["rep1", "rep2", "rep3"]
    handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            color="black",
            markerfacecolor="black",
            markeredgecolor="white",
            markersize=6,
            linestyle="None",
            label=label,
        )
        for m, label in zip(markers, marker_labels, strict=False)
    ]
    ax.legend(
        handles=handles,
        title="median",
        bbox_to_anchor=(0.95, 1),
        loc="upper left",
        frameon=False,
    )
    if ymax:
        if isinstance(ymax, tuple):
            ax.set_ylim(ymax[0], ymax[1])
        else:
            ax.set_ylim(0, ymax)
    plt.tight_layout()

    # Display the y-axis ticks as whole percentage numbers (0-100)
    def percent_formatter(x, pos):  # noqa: D401 – simple lambda replacement
        return f"{x:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
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
    # Annotate pairwise significance for each group
    set_grouped_significance_marks(
        ax,
        df_median,
        conditions,
        condition_col,
        feature,
        ax.get_ylim()[1],
        group_size=group_size,
        x_positions=x_positions,
    )


def featureplot_threshold(
    ax: Axes,
    df: pd.DataFrame,
    conditions: list[str],
    feature: str,
    threshold: float,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = "",
) -> None:
    """Plot a feature plot with a threshold."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(dimensions[0], dimensions[1]))
    # Filter DataFrame using selector_val_filter, as in feature_plot
    df_filtered = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df_filtered is not None, "No data found"
    df_filtered["threshold"] = np.where(df[feature] > threshold, "pos", "neg")


def grouped_stacked_threshold_barplot(
    ax: Optional[Axes],
    df: pd.DataFrame,
    conditions: list[str],
    group_size: int = 2,
    feature: str = "feature",
    threshold: float = 0.0,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    colors=COLOR,
    repeat_offset: float = 0.18,
    dimensions: tuple[float, float] = (width, height),
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
        fig = ax.figure
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
                    color=COLOR.OLIVE.value,
                    edgecolor="white",
                    linewidth=0.7,
                    label="pos" if cond_idx == 0 and rep_idx == 0 else None,
                )
                ax.bar(
                    xpos,
                    neg_pct,
                    width=repeat_offset * 1.05,
                    bottom=pos_pct,
                    color=COLOR.LIGHT_GREEN.value,
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
    for cond_idx, cond in enumerate(conditions):
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
            facecolor=COLOR.LIGHT_GREEN.value,
            edgecolor="white",
            label=f"{feature_name}-",
        ),
        Patch(
            facecolor=COLOR.OLIVE.value,
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
