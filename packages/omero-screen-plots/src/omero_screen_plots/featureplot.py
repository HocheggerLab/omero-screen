"""Module for feature plotting."""

import warnings
from pathlib import Path  # noqa: E402
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from omero_screen_plots.stats import set_significance_marks
from omero_screen_plots.utils import (
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
) -> None:
    """Plot a feature plot."""
    df_filtered = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df_filtered is not None, "No data found"
    if scale:
        df_filtered = scale_data(df_filtered, feature)

    fig, ax = plt.subplots(figsize=(height, height))
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
    file_name = title.replace(" ", "_")
    if save and path:
        save_fig(
            fig,
            path,
            file_name,
            tight_layout=False,
            fig_extension="pdf",
        )
