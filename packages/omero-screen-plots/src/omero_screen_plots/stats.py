"""Module for statistical analysis functions."""

import pandas as pd
from matplotlib.axes import Axes
from scipy import stats  # type: ignore


def calculate_pvalues(
    df: pd.DataFrame, conditions: list[str], condition_col: str, column: str
) -> list[float]:
    """Calculate p-values for each condition against the first condition."""
    df2 = df[df[condition_col].isin(conditions)]
    count_list = [
        df2[df2[condition_col] == condition][column].tolist()
        for condition in conditions
    ]
    return [
        stats.ttest_ind(count_list[0], data).pvalue  # type: ignore
        for data in count_list[1:]
    ]


def get_significance_marker(p: float) -> str:
    """Get the significance marker for a p-value."""
    match p:
        case p if p > 0.05:
            return "ns"
        case p if p > 0.01:
            return "*"
        case p if p > 0.001:
            return "**"
        case _:
            return "***"


def set_significance_marks(
    axes: Axes,
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    y_max: float,
) -> None:
    """Set the significance marks on the axes."""
    pvalues = calculate_pvalues(df, conditions, condition_col, y_col)
    for i, _ in enumerate(conditions[1:], start=1):
        p_value = pvalues[i - 1]  # Adjust index for p-values list
        significance = get_significance_marker(p_value)

        # Find the midpoint of the bar
        x = i

        y = y_max

        # Annotate the significance marker
        axes.annotate(
            significance,
            xy=(x, y),
            xycoords="data",
            ha="center",
            va="bottom",
            fontsize=6,
        )


def set_grouped_significance_marks(
    axes: Axes,
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    y_max: float,
    group_size: int = 2,
    x_positions: list[float] | None = None,
) -> None:
    """Sets significance marks on the axes.

    For each group of group_size in conditions, perform pairwise t-tests between adjacent conditions in the group
    and annotate the significance above the midpoint between the two columns being compared.
    Optionally, provide x_positions for custom x-axis placement.
    Parameters:
    axes: Axes
    df: pd.DataFrame
    conditions: list[str]
    condition_col: str
    y_col: str
    y_max: float
    """
    n = len(conditions)
    if x_positions is None:
        x_positions = [float(i) for i in range(n)]
    elif not isinstance(x_positions, list):
        x_positions = list(x_positions)
    for group_start in range(0, n, group_size):
        group_conds = conditions[group_start : group_start + group_size]
        group_xs = x_positions[group_start : group_start + group_size]
        for i in range(len(group_conds) - 1):
            cond1 = group_conds[i]
            cond2 = group_conds[i + 1]
            x1 = group_xs[i]
            x2 = group_xs[i + 1]
            # Get data for each condition
            data1 = df[df[condition_col] == cond1][y_col]
            data2 = df[df[condition_col] == cond2][y_col]
            # Perform t-test (TtestResult object)
            ttest = stats.ttest_ind(data1, data2)
            p_value = float(ttest.pvalue)
            significance = get_significance_marker(p_value)
            # Annotate at midpoint
            x_mid = (x1 + x2) / 2
            axes.annotate(
                significance,
                xy=(x_mid, y_max),
                xycoords="data",
                ha="center",
                va="bottom",
                fontsize=6,
            )
