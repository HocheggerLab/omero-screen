"""Module for classification plotting."""

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Rectangle

from omero_screen_plots import COLOR
from omero_screen_plots.utils import save_fig, selector_val_filter

current_dir = Path(__file__).parent
style_path = (current_dir / "../../hhlab_style01.mplstyle").resolve()
plt.style.use(style_path)
prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]
pd.options.mode.chained_assignment = None


def quantify_classification(
    df: pd.DataFrame, condition_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify classification results for each condition and return summary statistics."""
    df_class = (
        df.groupby(
            ["plate_id", "cell_line", "well_id", condition_col, "Class"]
        )["experiment"]
        .count()
        .reset_index()
        .rename(columns={"experiment": "class count"})
    )
    df_class["percentage"] = (
        df_class["class count"]
        / df_class.groupby(["plate_id", "cell_line", "well_id"])[
            "class count"
        ].transform("sum")
        * 100
    )
    df_class_mean = (
        df_class.groupby(["plate_id", "cell_line", condition_col, "Class"])[
            "percentage"
        ]
        .mean()
        .reset_index()
    )
    if len(df.plate_id.unique()) > 1:
        df_class_std = (
            df_class.groupby(
                ["plate_id", "cell_line", condition_col, "Class"]
            )["percentage"]
            .std()
            .reset_index()
        )
    else:
        # Copy df_class_mean structure but set only percentage values to 0
        df_class_std = df_class_mean.copy()
        df_class_std["percentage"] = 0
    return df_class_mean, df_class_std


height = 3 / 2.54  # 2 cm


def plot_classification(
    df: pd.DataFrame,
    classes: list[str],
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: str | None = "cell_line",
    selector_val: str | None = None,
    y_lim: tuple[int, int] = (0, 100),
    title: str | None = None,
    colors: list[str] = COLORS,
    save: bool = True,
    path: Path | None = None,
    class_col: str = "Class",
) -> None:
    """Plot classification results for the given classes and conditions.

    Args:
        df: DataFrame containing classification data.
        classes: List of class names to plot.
        conditions: List of condition names to plot.
        condition_col: Column name for experimental condition.
        selector_col: Column name for selector (e.g., cell line).
        selector_val: Value to filter selector_col by.
        y_lim: y-axis limits.
        title: Plot title.
        colors: List of colors for plotting.
        save: Whether to save the figure.
        path: Path to save the figure.
        class_col: Column name for class/category (default 'Class').
    """
    df1 = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df1 is not None, "df1 is None"
    assert len(df1) > 0, "df1 is empty"
    df_class_mean, df_class_std = quantify_classification(df1, condition_col)
    # Set categorical dtype to enforce order
    df_class_mean[condition_col] = pd.Categorical(
        df_class_mean[condition_col], categories=conditions, ordered=True
    )
    df_class_std[condition_col] = pd.Categorical(
        df_class_std[condition_col], categories=conditions, ordered=True
    )
    plot_data = df_class_mean.pivot_table(
        index=condition_col,
        columns=class_col,
        values="percentage",
        observed=False,
    ).reset_index()
    std_data = df_class_std.pivot_table(
        index=condition_col,
        columns=class_col,
        values="percentage",
        observed=False,
    ).reset_index()
    yerr = std_data[classes].values.T
    fig, ax = plt.subplots(figsize=(height, height))
    plot_data.plot(
        x=condition_col,
        y=classes,
        kind="bar",
        stacked=True,
        yerr=yerr,
        width=0.75,
        legend=False,
        ax=ax,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=7
    )
    ax.set_xlabel("")
    ax.set_ylabel("% of total cells")
    ax.set_ylim(y_lim)
    ax.legend(
        fontsize=7,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    ax.get_legend().set_title("")
    if not title:
        title = f"Classification Analysis {selector_val}"
    fig.suptitle(title, fontsize=8, weight="bold", x=0, y=1.05, ha="left")
    if save and path:
        save_fig(
            fig,
            path,
            title,
            tight_layout=True,
            fig_extension="pdf",
        )


def grouped_stacked_classification_barplot(
    ax: Optional[Axes],
    df: pd.DataFrame,
    classes: list[str],
    conditions: list[str],
    group_size: int = 2,
    condition_col: str = "condition",
    selector_col: str | None = "cell_line",
    selector_val: str | None = None,
    colors: Any = COLOR,
    repeat_offset: float = 0.18,
    dimensions: tuple[float, float] = (6, 4),
    x_label: bool = True,
    title: str | None = None,
    save: bool = True,
    path: Path | None = None,
    class_col: str = "Class",
) -> None:
    """Grouped stacked barplot for classification results (classes stacked, repeats grouped, boxed by condition).

    Args:
        ax: Matplotlib axis.
        df: DataFrame containing classification data.
        classes: List of class names to plot.
        conditions: List of condition names to plot.
        group_size: Number of groups for x-axis grouping.
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
        class_col: Column name for class/category (default 'Class').
    """
    if len(classes) == 2:
        colors = [COLOR.LIGHT_GREEN.value, COLOR.OLIVE.value]
    elif len(classes) == 3:
        colors = [COLOR.BLUE.value, COLOR.LIGHT_BLUE.value, COLOR.GREY.value]
    else:
        colors = [color.value for color in COLOR]

    if ax is None:
        fig, ax = plt.subplots(figsize=dimensions)
    else:
        fig = ax.figure  # type: ignore[assignment]
    df1 = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df1 is not None and len(df1) > 0
    # Get mean percentage per plate_id, condition, class
    df_class = (
        df1.groupby(["plate_id", condition_col, class_col])["experiment"]
        .count()
        .reset_index()
        .rename(columns={"experiment": "class count"})
    )
    df_class["percentage"] = (
        df_class["class count"]
        / df_class.groupby(["plate_id", condition_col])[
            "class count"
        ].transform("sum")
        * 100
    )
    n_repeats = 3
    repeat_ids = sorted(df_class["plate_id"].unique())[:n_repeats]
    n_conditions = len(conditions)
    from omero_screen_plots.utils import grouped_x_positions

    x_base_positions = grouped_x_positions(
        n_conditions,
        group_size=group_size,
        within_group_spacing=0.6,
        between_group_gap=0.7,
    )
    bar_width = repeat_offset * 1.05
    # Plot bars: for each condition, for each repeat, stack classes
    for cond_idx, cond in enumerate(conditions):
        for rep_idx, plate_id in enumerate(repeat_ids):
            xpos = x_base_positions[cond_idx] + (rep_idx - 1) * repeat_offset
            plate_data = df_class[
                (df_class[condition_col] == cond)
                & (df_class["plate_id"] == plate_id)
            ]
            if not plate_data.empty:
                bottoms = 0
                for i, cls in enumerate(classes):
                    val = plate_data[plate_data[class_col] == cls][
                        "percentage"
                    ].sum()
                    ax.bar(
                        xpos,
                        val,
                        width=bar_width,
                        bottom=bottoms,
                        color=colors[i],
                        edgecolor="white",
                        linewidth=0.7,
                        label=cls if cond_idx == 0 and rep_idx == 0 else None,
                    )
                    bottoms += val
    ax.set_xticks(x_base_positions)
    if x_label:
        ax.set_xticklabels(conditions, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    # Draw triplicate boxes
    y_min = 0
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
    # Add legend for classes
    legend_handles = [
        Patch(facecolor=colors[i % len(colors)], edgecolor="white", label=cls)
        for i, cls in enumerate(classes)
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
