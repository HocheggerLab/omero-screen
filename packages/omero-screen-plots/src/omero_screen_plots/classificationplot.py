"""Module for classification plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
) -> None:
    """Plot classification results for the given classes and conditions."""
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
        columns="Class",
        values="percentage",
        observed=False,
    ).reset_index()
    std_data = df_class_std.pivot_table(
        index=condition_col,
        columns="Class",
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
