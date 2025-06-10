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
        stats.ttest_ind(count_list[0], data).pvalue for data in count_list[1:]
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
