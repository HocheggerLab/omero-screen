"""Module for statistical analysis functions."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from omero_screen.config import get_logger
from scipy import stats

logger = get_logger(__name__)


def calculate_pvalues(
    df: pd.DataFrame, conditions: list[str], condition_col: str, column: str
) -> list[float]:
    """Calculate p-values for each condition against the first condition."""
    df2 = df[df[condition_col].isin(conditions)]
    count_list = [
        df2[df2[condition_col] == condition][column].tolist()
        for condition in conditions
    ]
    logger.debug("count_list: %s", count_list)

    pvalues = []
    for data in count_list[1:]:
        try:
            # Check for sufficient variance and sample size
            if len(count_list[0]) < 2 or len(data) < 2:
                logger.warning(
                    "Insufficient sample size for t-test, setting p-value to 1.0"
                )
                pvalues.append(1.0)
            elif np.var(count_list[0]) == 0 and np.var(data) == 0:
                # Both groups have zero variance - no meaningful difference
                pvalues.append(1.0)
            else:
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning,
                        message=".*catastrophic cancellation.*",
                    )
                    p_value = stats.ttest_ind(count_list[0], data).pvalue
                    pvalues.append(p_value)
        except (ValueError, RuntimeError, stats.LinAlgError) as e:
            logger.warning(
                "Error in t-test calculation: %s, setting p-value to 1.0", e
            )
            pvalues.append(1.0)

    return pvalues


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
    logger.info("pvalues: %s", pvalues)
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


def calculate_grouped_pvalues(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    column: str,
    group_size: int = 2,
) -> list[tuple[int, float]]:
    """Calculate p-values within each group against the first condition of that group.

    Args:
        df: DataFrame containing the data
        conditions: List of all conditions
        condition_col: Column name containing condition values
        column: Column name for the data to compare
        group_size: Number of conditions per group

    Returns:
        List of tuples (condition_index, p_value) for non-reference conditions
    """
    df_filtered = df[df[condition_col].isin(conditions)]
    results = []

    for group_start in range(0, len(conditions), group_size):
        group_conditions = conditions[group_start : group_start + group_size]

        if len(group_conditions) < 2:
            continue

        # Get reference data (first condition in group)
        reference_condition = group_conditions[0]
        reference_data = df_filtered[
            df_filtered[condition_col] == reference_condition
        ][column].tolist()

        # Compare other conditions in group to reference
        for i, condition in enumerate(group_conditions[1:], start=1):
            condition_data = df_filtered[
                df_filtered[condition_col] == condition
            ][column].tolist()

            if len(reference_data) > 0 and len(condition_data) > 0:
                try:
                    # Check for sufficient variance and sample size
                    if len(reference_data) < 2 or len(condition_data) < 2:
                        logger.warning(
                            "Insufficient sample size for t-test: %s vs %s",
                            reference_condition,
                            condition,
                        )
                        p_value = 1.0
                    elif (
                        np.var(reference_data) == 0
                        and np.var(condition_data) == 0
                    ):
                        # Both groups have zero variance - no meaningful difference
                        p_value = 1.0
                    else:
                        import warnings

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                category=RuntimeWarning,
                                message=".*catastrophic cancellation.*",
                            )
                            p_value = stats.ttest_ind(
                                reference_data, condition_data
                            ).pvalue

                    global_index = group_start + i
                    results.append((global_index, p_value))
                except (ValueError, RuntimeError, stats.LinAlgError) as e:
                    logger.warning(
                        "Error in t-test calculation for %s vs %s: %s",
                        reference_condition,
                        condition,
                        e,
                    )
                    global_index = group_start + i
                    results.append((global_index, 1.0))
            else:
                logger.warning(
                    "No data found for comparison: %s vs %s",
                    reference_condition,
                    condition,
                )

    return results


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


def set_grouped_within_significance_marks(
    axes: Axes,
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    y_max: float,
    group_size: int = 2,
    x_positions: list[float] | None = None,
) -> None:
    """Set significance marks comparing conditions within each group to the group's first condition.

    Args:
        axes: Matplotlib axes object
        df: DataFrame containing the data
        conditions: List of all conditions
        condition_col: Column name containing condition values
        y_col: Column name for the data to compare
        y_max: Y-axis maximum for positioning marks
        group_size: Number of conditions per group
        x_positions: Optional custom x-axis positions
    """
    if x_positions is None:
        x_positions = [float(i) for i in range(len(conditions))]
    elif not isinstance(x_positions, list):
        x_positions = list(x_positions)

    # Get p-values for within-group comparisons
    pvalue_results = calculate_grouped_pvalues(
        df, conditions, condition_col, y_col, group_size
    )

    logger.info("grouped pvalues: %s", pvalue_results)

    # Annotate significance marks
    for condition_index, p_value in pvalue_results:
        significance = get_significance_marker(p_value)
        x = x_positions[condition_index]

        axes.annotate(
            significance,
            xy=(x, y_max),
            xycoords="data",
            ha="center",
            va="bottom",
            fontsize=6,
        )


def set_significance_marks_adaptive(
    axes: Axes,
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    y_max: float,
    group_size: int = 1,
    x_positions: list[float] | None = None,
) -> None:
    """Adaptively set significance marks based on group_size.

    - If group_size = 1: Use traditional comparison (all vs first condition)
    - If group_size > 1: Use within-group comparison (each condition vs group reference)

    Args:
        axes: Matplotlib axes object
        df: DataFrame containing the data
        conditions: List of all conditions
        condition_col: Column name containing condition values
        y_col: Column name for the data to compare
        y_max: Y-axis maximum for positioning marks
        group_size: Number of conditions per group
        x_positions: Optional custom x-axis positions
    """
    if group_size == 1:
        # Traditional behavior: compare all to first condition
        set_significance_marks(
            axes, df, conditions, condition_col, y_col, y_max
        )
    else:
        # New behavior: compare within groups to group reference
        set_grouped_within_significance_marks(
            axes,
            df,
            conditions,
            condition_col,
            y_col,
            y_max,
            group_size,
            x_positions,
        )
