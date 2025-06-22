"""Plot a combined histogram and scatter plot from omero screen cell cycle data."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from omero_screen_plots.cellcycleplot import prop_pivot
from omero_screen_plots.colors import COLOR
from omero_screen_plots.utils import save_fig, selector_val_filter

current_dir = Path(__file__).parent
style_path = (current_dir / "../../hhlab_style01.mplstyle").resolve()
plt.style.use(style_path)
prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]
COLOR_ENUM = COLOR
pd.options.mode.chained_assignment = None


# Functions to plot histogram and scatter plots
def histogram_plot(
    ax: Axes, i: int, data: pd.DataFrame, colors: list[str] = COLORS
) -> None:
    """Plot a histogram of the integrated DAPI intensity.

    Parameters
    ----------
    ax : Axes
        The axes on which to plot the histogram.
    i : int
        The index of the histogram (used for labeling).
    data : pd.DataFrame
        The data containing the integrated DAPI intensity.
    colors : list[str]
        A list of colors to use for the histogram.

    Returns:
    -------
    None
        This function does not return a value.
    """
    sns.histplot(
        data=data, x="integrated_int_DAPI_norm", ax=ax, color=COLORS[-1]
    )
    ax.set_xlabel("")
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, 16)
    ax.xaxis.set_visible(False)
    if i == 0:
        ax.set_ylabel("Freq.", fontsize=6)
    else:
        ax.yaxis.set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=6)


def scatter_plot(
    ax: Axes,
    i: int,
    data: pd.DataFrame,
    conditions: list[str],
    colors: list[str],
) -> None:
    """Plot a scatter plot of the integrated DAPI intensity vs. the mean EdU intensity.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to plot the scatter plot.
    i : int
        The index of the scatter plot (used for labeling).
    data : pd.DataFrame
        The data containing the integrated DAPI intensity and mean EdU intensity.
    conditions : list[str]
        The conditions to use for the scatter plot.
    colors : list[str]
        A list of colors to use for the scatter plot.

    Returns:
    -------
    None
        This function does not return a value.
    """
    phases = ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]
    sns.scatterplot(
        data=data,
        x="integrated_int_DAPI_norm",
        y="intensity_mean_EdU_nucleus_norm",
        hue="cell_cycle",
        hue_order=phases,
        palette=colors[: len(phases)],
        s=2,
        alpha=1,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.grid(False)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: str(int(x)))
    )
    ax.set_xticks([2, 4, 8])
    ax.set_xlim(1, 16)
    if i == len(conditions):
        ax.set_ylabel("norm. EdU int.", fontsize=6)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: str(int(x)))
        )
    else:
        ax.yaxis.set_visible(False)
    ax.legend().remove()
    ax.set_xlabel("")
    ax.axvline(x=3, color="black", linestyle="--")
    ax.axhline(y=3, color="black", linestyle="--")
    sns.kdeplot(
        data=data,
        x="integrated_int_DAPI_norm",
        y="intensity_mean_EdU_nucleus_norm",
        fill=True,
        alpha=0.3,
        cmap="rocket_r",
        ax=ax,
    )
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.set_xlabel("")


def scatter_plot_feature(
    ax: Axes,
    i: int,
    data: pd.DataFrame,
    conditions: list[str],
    col: str,
    y_lim: float,
    colors: list[str] = COLORS,
) -> None:
    """Plot a scatter plot of the integrated DAPI intensity vs. a specified column.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to plot the scatter plot.
    i : int
        The index of the scatter plot (used for labeling).
    data : pd.DataFrame
        The data containing the integrated DAPI intensity and the specified column.
    conditions : list[str]
        The conditions to use for the scatter plot.
    col : str
        The column to plot against the integrated DAPI intensity.
    y_lim : float
        The threshold value for color categorization.
    colors : list[str]
        A list of colors to use for the scatter plot.
    """
    # Create binary categories based on threshold
    data.loc[:, "threshold_category"] = data[col].apply(
        lambda x: "below" if x < y_lim else "above"
    )

    sns.scatterplot(
        data=data,
        x="integrated_int_DAPI_norm",
        y=col,
        hue="threshold_category",
        palette={
            "below": COLOR_ENUM.LIGHT_BLUE.value,
            "above": COLOR_ENUM.BLUE.value,
        },
        hue_order=["below", "above"],
        s=2,
        alpha=1,
        ax=ax,
    )
    ax.set_xscale("log")
    # ax.set_yscale("log", base=2)
    ax.grid(False)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: str(int(x)))
    )
    ax.set_xticks([2, 4, 8])
    ax.set_xlim(1, 16)

    # Set specific y-axis ticks and labels
    # ax.set_yticks([2000, 4000, 8000, 16000])
    # ax.set_yticklabels(["2", "4", "8", "16"])

    if i == len(conditions) * 2:
        y_label = f"{col.split('_')[2]} norm."
        ax.set_ylabel(y_label, fontsize=6)
        # Custom y-axis formatter to remove zeros
        # ax.yaxis.set_major_formatter(
        #     ticker.FuncFormatter(lambda y, _: f"{y / 1000:g}")
        # )
    else:
        ax.yaxis.set_visible(False)

    ax.legend().remove()
    ax.set_xlabel("")
    ax.axvline(x=3, color="black", linestyle="--")
    ax.axhline(y=3, color="black", linestyle="--")

    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.set_xlabel("")


# Define figure size in inches
width = 10 / 2.54  # 10 cm
height = 7 / 2.54  # 4 cm


def comb_plot(
    df: pd.DataFrame,
    conditions: list[str],
    feature_col: str,
    feature_y_lim: float,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: str | None = None,
    cell_number: int | None = None,
    colors: list[str] = COLORS,
    width: float = 10 / 2.54,
    height: float = 7 / 2.54,
    save: bool = True,
    path: Path | None = None,
) -> None:
    """Plot a combined histogram and scatter plot."""
    col_number = len(conditions)
    df1 = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df1 is not None  # tells type checker df1 is definitely not None
    condition_list = conditions * 3

    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(3, col_number, height_ratios=[1, 3, 3], hspace=0.05)
    ax_list = [(i, j) for i in range(3) for j in range(col_number)]
    y_max = df["intensity_mean_EdU_nucleus_norm"].quantile(0.99) * 1.5
    y_min = df["intensity_mean_EdU_nucleus_norm"].quantile(0.01) * 0.8
    y_max_col = df[feature_col].quantile(0.99) * 1.5
    y_min_col = df[feature_col].quantile(0.01) * 0.8

    for i, pos in enumerate(ax_list):
        data = df1[df1[condition_col] == condition_list[i]]
        if cell_number and len(data) >= cell_number:
            data_red = pd.DataFrame(
                data.sample(n=cell_number, random_state=42)
            )
        else:
            data_red = pd.DataFrame(data)
        ax = fig.add_subplot(gs[pos[0], pos[1]])

        if i < len(conditions):
            histogram_plot(ax, i, data_red, colors)
            ax.set_title(f"{condition_list[i]}", size=6, weight="regular")
        elif i < 2 * len(conditions):
            scatter_plot(ax, i, data_red, conditions, colors)
            ax.set_ylim(y_min, y_max)
        else:
            scatter_plot_feature(
                ax, i, data_red, conditions, feature_col, feature_y_lim, colors
            )
            ax.set_ylim(y_min_col, y_max_col)

        ax.grid(visible=False)

    # Set common x-axis label
    fig.text(0.5, -0.07, "norm. DNA content", ha="center", fontsize=6)
    if not title:
        title = f"combplot{selector_val}"
    fig.suptitle(title, fontsize=8, weight="bold", x=0, y=1.05, ha="left")
    figure_title = title.replace(" ", "_")
    if save and path:
        save_fig(
            fig,
            path,
            figure_title,
            tight_layout=False,
            fig_extension="png",  # the scatterplots can cause problems in illustrator when saved as vector compatible pdfs
        )


def _plot_hist_and_scatter(
    fig: Figure,
    gs: GridSpec,
    df1: pd.DataFrame,
    conditions: list[str],
    cell_number: int | None,
    y_min: float,
    y_max: float,
    colors: list[str],
    condition_col: str,
) -> None:
    """Plot histogram and scatter subplots for each condition.

    Args:
        fig (Figure): The matplotlib figure object to plot on.
        gs (GridSpec): The GridSpec layout for subplot arrangement.
        df1 (pd.DataFrame): Filtered dataframe containing the data to plot.
        conditions (list[str]): List of condition names to plot.
        cell_number (int | None): Number of cells to sample per condition (if not None).
        y_min (float): Minimum y-axis value for scatter plots.
        y_max (float): Maximum y-axis value for scatter plots.
        colors (list[str]): List of colors for plotting.
        condition_col (str): Name of the column indicating experimental condition.

    Returns:
        None
    """
    col_number = len(conditions)
    condition_list = conditions * 2
    ax_list = [(i, j) for i in range(2) for j in range(col_number)]
    for i, pos in enumerate(ax_list):
        data = df1[df1[condition_col] == condition_list[i]]
        if cell_number and len(data) >= cell_number:
            data_red = data.sample(n=cell_number, random_state=42)
        else:
            data_red = data
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        if i < len(conditions):
            histogram_plot(ax, i, data_red, colors)
            ax.set_title(f"{condition_list[i]}", size=6, weight="regular")
        else:
            scatter_plot(ax, i, data_red, conditions, colors)
            ax.set_ylim(y_min, y_max)
        ax.grid(visible=False)


def _plot_stacked_bar(
    fig: Figure,
    gs: GridSpec,
    df1: pd.DataFrame,
    condition_col: str,
    conditions: list[str],
    H3: bool,
    colors: list[str],
) -> Axes:
    """Plot a stacked barplot of cell cycle phases as the last column in the grid.

    Args:
        fig (Figure): The matplotlib figure object to plot on.
        gs (GridSpec): The GridSpec layout for subplot arrangement.
        df1 (pd.DataFrame): Filtered dataframe containing the data to plot.
        condition_col (str): Name of the column indicating experimental condition.
        conditions (list[str]): List of condition names to plot.
        H3 (bool): Whether to use H3 cell cycle phases.
        colors (list[str]): List of colors for plotting.

    Returns:
        Axes: The matplotlib Axes object for the stacked barplot.
    """
    ax_bar = fig.add_subplot(gs[:, -1])
    df_mean, df_std = prop_pivot(df1, condition_col, conditions, H3)
    df_mean.plot(
        kind="bar",
        stacked=True,
        yerr=df_std,
        width=0.75,
        ax=ax_bar,
        color=colors[: df_mean.shape[1]],
    )
    ax_bar.set_ylim(0, 110)
    ax_bar.set_xticklabels(conditions, rotation=30, ha="right")
    ax_bar.set_xlabel("")
    if H3:
        legend = ax_bar.legend(
            ["Sub-G1", "G1", "S", "G2", "M", "Polyploid"],
            title="CellCyclePhase",
        )
        ax_bar.set_ylabel("% of population")
    else:
        legend = ax_bar.legend(
            ["Sub-G1", "G1", "S", "G2/M", "Polyploid"], title="CellCyclePhase"
        )
    handles, labels = ax_bar.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    legend.remove()
    box = ax_bar.get_position()
    ax_bar.set_position((box.x0, box.y0, box.width * 0.8, box.height))
    legend = ax_bar.legend(
        handles,
        labels,
        title="CellCyclePhase",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=6,
        title_fontsize=7,
        frameon=False,
    )
    ax_bar.set_ylabel("% of population")
    ax_bar.grid(False)
    ax_bar.set_title("Cell cycle phases", fontsize=6, y=1.05)
    return ax_bar


def combplot_simple(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: str | None = None,
    cell_number: int | None = None,
    colors: list[str] = COLORS,
    save: bool = True,
    path: Path | None = None,
    H3: bool = False,
) -> None:
    """Plot a combined histogram, scatter plot, and a stacked barplot as the last column."""
    col_number = len(conditions)
    df1 = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df1 is not None  # tells type checker df1 is definitely not None
    fig = plt.figure(figsize=(width + 2, height * 2 / 3))  # wider for barplot
    gs = GridSpec(
        2,
        col_number + 1,
        height_ratios=[1, 3],
        width_ratios=[1] * col_number + [0.7],
        hspace=0.05,
        wspace=0.25,
    )
    y_max = df["intensity_mean_EdU_nucleus_norm"].quantile(0.99) * 1.5
    y_min = df["intensity_mean_EdU_nucleus_norm"].quantile(0.01) * 0.8

    _plot_hist_and_scatter(
        fig,
        gs,
        df1,
        conditions,
        cell_number,
        y_min,
        y_max,
        colors,
        condition_col,
    )
    _plot_stacked_bar(fig, gs, df1, condition_col, conditions, H3, colors)

    # Set common x-axis label
    fig.text(0.5, -0.07, "norm. DNA content", ha="center", fontsize=6)
    if not title:
        title = f"combplot_simple{selector_val}"
    fig.suptitle(title, fontsize=8, weight="bold", x=0, y=1.05, ha="left")
    figure_title = title.replace(" ", "_")
    if save and path:
        save_fig(
            fig,
            path,
            figure_title,
            tight_layout=False,
            fig_extension="png",
        )
