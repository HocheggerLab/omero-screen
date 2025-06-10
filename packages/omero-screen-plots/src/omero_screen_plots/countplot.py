"""Module for count plotting."""

from enum import Enum
from pathlib import Path
from typing import Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.stats import set_significance_marks
from omero_screen_plots.utils import (
    save_fig,
    selector_val_filter,
    show_repeat_points,
)

current_dir = Path(__file__).parent
style_path = (current_dir / "../../hhlab_style01.mplstyle").resolve()
plt.style.use(style_path)
prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]
pd.options.mode.chained_assignment = None

height = 3 / 2.54  # 2 cm


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


def count_plot(
    df: pd.DataFrame,
    norm_control: str,
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    plot_type: PlotType = PlotType.NORMALISED,
    title: Optional[str] = None,
    colors: list[str] = COLORS,
    save: bool = True,
    path: Optional[Path] = None,
    ax: Optional[Axes] = None,
) -> None:
    """Plot normalized counts."""
    count_col = (
        "normalized_count" if plot_type == PlotType.NORMALISED else "count"
    )

    # Create figure only if we don't have an existing axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(height, height))
    else:
        fig = fig = cast(Figure, ax.figure)

    df1 = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df1 is not None
    counts = norm_count(df1, norm_control, condition=condition_col)
    sns.barplot(
        data=counts,
        x=condition_col,
        y=count_col,
        order=conditions,
        color=colors[-1],
        ax=ax,
    )
    ax.set_xticks(range(len(conditions)))  #
    ax.set_xticklabels(conditions, rotation=45, ha="right")

    show_repeat_points(counts, conditions, condition_col, count_col, ax)
    if df1.plate_id.nunique() >= 3:
        set_significance_marks(
            ax,
            counts,
            conditions,
            condition_col,
            count_col,
            ax.get_ylim()[1],
        )
    ax.set_xlabel("")
    if not title:
        title = f"counts {selector_val}"
    file_name = title.replace(" ", "_")
    fig.suptitle(title, fontsize=7, weight="bold", x=0, y=1.05, ha="left")
    if save and path:
        save_fig(
            fig,
            path,
            file_name,
            tight_layout=False,
            fig_extension="pdf",
        )
