"""Module for cell cycle plotting."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from omero_screen_plots.stats import set_significance_marks
from omero_screen_plots.utils import (
    save_fig,
    selector_val_filter,
    show_repeat_points,
)

# Define figure size in inches
width = 10 / 2.54  # 10 cm
height = 7 / 2.54  # 4 cm


current_dir = Path(__file__).parent
style_path = (current_dir / "../../hhlab_style01.mplstyle").resolve()
plt.style.use(style_path)
prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]
pd.options.mode.chained_assignment = None


def cc_phase(df: pd.DataFrame, condition: str = "condition") -> pd.DataFrame:
    """Calculate the percentage of cells in each cell cycle phase for each condition."""
    return (
        (
            df.groupby(["plate_id", "cell_line", condition, "cell_cycle"])[
                "experiment"
            ].count()
            / df.groupby(["plate_id", "cell_line", condition])[
                "experiment"
            ].count()
            * 100
        )
        .reset_index()
        .rename(columns={"experiment": "percent"})
    )


def cellcycle_plot(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: str | None = "cell_line",
    selector_val: str | None = None,
    title: str | None = None,
    colors: list[str] = COLORS,
    save: bool = True,
    path: Path | None = None,
) -> None:
    """Plot the cell cycle phases for each condition."""
    print(f"Plotting cell cycle quantifications for {selector_val}")
    df1 = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df1 is not None
    df1 = cc_phase(df1, condition=condition_col)
    fig, ax = plt.subplots(2, 2, figsize=(height * 0.7, height))
    ax_list = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
    cellcycle = ["G1", "S", "G2/M", "Polyploid"]
    for i, phase in enumerate(cellcycle):
        axes = ax_list[i]
        df_phase = df1[
            (df1.cell_cycle == phase) & (df1[condition_col].isin(conditions))
        ]
        # y_max = df_phase["percent"].max() * 1.2

        sns.barplot(
            data=df_phase,
            x=condition_col,
            y="percent",
            color=colors[i + 1],
            order=conditions,
            ax=axes,
        )
        show_repeat_points(
            df=df_phase,
            conditions=conditions,
            condition_col=condition_col,
            y_col="percent",
            ax=axes,
        )
        if df1.plate_id.nunique() >= 3:
            set_significance_marks(
                axes,
                df_phase,
                conditions,
                condition_col,
                "percent",
                axes.get_ylim()[1],
            )
        axes.set_title(f"{phase}", fontsize=6, y=1.05)
        if i in [1, 3]:
            axes.set_ylabel(None)
        if i in [0, 1]:
            axes.set_xticklabels([])
        else:
            # Set the tick positions
            axes.set_xticks(range(len(conditions)))  #
            axes.set_xticklabels(conditions, rotation=45, ha="right")
        axes.set_xlabel(None)
        # Get the y-max for positioning significance markers
    if not title:
        title = f"Cellcycle Analysis {selector_val}"
    fig.suptitle(title, fontsize=8, weight="bold", x=0, y=1, ha="left")
    fig_title = title.replace(" ", "_")
    if save and path:
        save_fig(
            fig,
            path,
            fig_title,
            tight_layout=False,
            fig_extension="pdf",
        )


def prop_pivot(
    df: pd.DataFrame, condition: str, conditions: list[str], H3: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function to pivot the cell cycle proportion dataframe and get the mean and std of each cell cycle phase."""
    df_prop = cc_phase(df, condition=condition)

    # Define the desired order of cell cycle phases
    cc_phases = (
        ["Sub-G1", "G1", "S", "G2/M", "Polyploid"]
        if not H3
        else ["Sub-G1", "G1", "S", "G2", "M", "Polyploid"]
    )

    df_prop1 = df_prop.copy()
    df_prop1[condition] = pd.Categorical(
        df_prop1[condition], categories=conditions, ordered=True
    )
    # Make cell_cycle a categorical with ordered phases
    df_prop1["cell_cycle"] = pd.Categorical(
        df_prop1["cell_cycle"], categories=cc_phases, ordered=True
    )

    df_mean = (
        df_prop1.groupby([condition, "cell_cycle"], observed=False)["percent"]
        .mean()
        .sort_index(level=condition)
        .reset_index()
        .pivot_table(columns=["cell_cycle"], index=[condition], observed=False)
    )
    df_mean.columns = df_mean.columns.droplevel(0)
    df_mean = df_mean[cc_phases]
    if len(df_prop1.plate_id.unique()) > 1:
        df_std = (
            df_prop1.groupby([condition, "cell_cycle"], observed=False)[
                "percent"
            ]
            .std()
            .sort_index(level=condition)
            .reset_index()
            .pivot_table(
                columns=["cell_cycle"], index=[condition], observed=False
            )
        )
        df_std.columns = df_std.columns.droplevel(0)
        df_std = df_std[cc_phases]
    else:
        df_std = pd.DataFrame(0, index=df_mean.index, columns=df_mean.columns)
    return df_mean, df_std


def stacked_barplot(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    H3: bool = False,
    title: str | None = None,
    colors: list[str] = COLORS,
    save: bool = True,
    path: Path | None = None,
) -> None:
    """Create a stacked barplot for cell cycle phase proportions."""
    df1 = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df1 is not None  # tells type checker df1 is definitely not None
    df_mean, df_std = prop_pivot(df1, condition_col, conditions, H3)
    fig, ax = plt.subplots()
    df_mean.plot(kind="bar", stacked=True, yerr=df_std, width=0.75, ax=ax)
    ax.set_ylim(0, 110)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_xlabel("")  # Remove the x-axis label)
    if H3:
        legend = ax.legend(
            ["Sub-G1", "G1", "S", "G2", "M", "Polyploid"],
            title="CellCyclePhase",
        )
        ax.set_ylabel("% of population")
    else:
        legend = ax.legend(
            ["Sub-G1", "G1", "S", "G2/M", "Polyploid"], title="CellCyclePhase"
        )
    # Get current handles and labels
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    # Clear the current legend
    legend.remove()
    # Create a new legend with the reversed handles and labels
    legend = ax.legend(
        handles,
        labels,
        title="CellCyclePhase",
        bbox_to_anchor=(1.25, 1),  # Position legend to the right
        loc="upper right",
    )
    frame = legend.get_frame()
    frame.set_alpha(0.5)
    ax.set_ylabel("% of population")
    ax.grid(False)
    if not title:
        title = f"stackedbarplot_{selector_val}"
    fig.suptitle(title, fontsize=8, weight="bold", x=0, y=1.05, ha="left")
    fig_title = title.replace(" ", "_")
    if save and path:
        save_fig(
            fig,
            path,
            fig_title,
            tight_layout=False,
            fig_extension="pdf",
        )
