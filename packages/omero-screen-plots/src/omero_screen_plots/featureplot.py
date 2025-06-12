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
height = 3 / 2.54  # 2 cm
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
                body.set_edgecolor("black")
                body.set_alpha(0.75)
                body.set_linewidth(1.5)
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
            body.set_alpha(0.75)
            body.set_linewidth(1.5)
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
    df: pd.DataFrame,
    feature: str,
    conditions: list[str],
    group_size: int = 2,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = "",
    ymax: float | tuple[float, float] | None = None,
    legend: Optional[tuple[str, list[str]]] = None,
    x_label: bool = True,
    height: float = height,
    violin: bool = False,
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
    x_label: bool
    height: float
    violin: bool
    title: str
    save: bool
    path: Optional[Path]
    """
    # Filter DataFrame using selector_val_filter, as in feature_plot
    df_filtered = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df_filtered is not None, "No data found"

    x_positions = grouped_x_positions(
        len(conditions), group_size=group_size, between_group_gap=0.75
    )
    data_for_plot = [
        df_filtered[df_filtered[condition_col] == cond][feature]
        for cond in conditions
    ]
    fig, ax = plt.subplots(figsize=(height * 3, height * 1.5))
    for i, (dat, xpos) in enumerate(
        zip(data_for_plot, x_positions, strict=False)
    ):
        color_idx = i % group_size  # alternate colors within group
        draw_violin_or_box(ax, dat, xpos, COLORS[color_idx], violin)

    # Overlay repeat points (medians per plate_id per condition)
    df_median = (
        df_filtered.groupby(["plate_id", condition_col])[feature]
        .median()
        .reset_index()
    )
    # Map condition to x-position
    cond_to_x = dict(zip(conditions, x_positions, strict=False))
    for cond in conditions:
        cond_medians = df_median[df_median[condition_col] == cond]
        x_base = np.full(len(cond_medians), cond_to_x[cond])
        jitter = np.random.uniform(-0.05, 0.05, size=len(cond_medians))
        x = x_base + jitter
        y = cond_medians[feature].values
        ax.scatter(
            x,
            y,
            color="black",
            edgecolor="white",
            s=18,
            zorder=4,
            label="Repeat medians" if cond == conditions[0] else None,
        )
    ax.set_xticks(x_positions)
    if x_label:
        ax.set_xticklabels(conditions, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel(feature)
    if title:
        ax.set_title(title, fontsize=8, weight="bold", y=1.1)
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
    if ymax:
        if isinstance(ymax, tuple):
            ax.set_ylim(ymax[0], ymax[1])
        else:
            ax.set_ylim(0, ymax)
    plt.tight_layout()
    if save and path and title:
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
    plt.show()
