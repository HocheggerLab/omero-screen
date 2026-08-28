"""Statistical analysis for the plots.

The convention across every plot is a **replicate-level** comparison:

1. collapse the per-cell data to **one median per repeat** (``plate_id``) per
   condition, then
2. compare each condition to the **first condition in its group** with a
   Student's t-test — **paired** (``scipy.stats.ttest_rel``, the default,
   aligned on the plates present in both conditions) or, optionally, unpaired
   (``ttest_ind``).

``group_size == 1`` means every condition is compared to the overall first
condition (the control); ``group_size > 1`` compares each condition to the first
condition of its own group.

The compute step (:func:`compute_significance`) returns both the per-repeat
medians and the test results so callers can annotate the axes
(:func:`annotate_significance`) and export both tables to CSV
(:func:`write_stats_csv`).
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes
from scipy import stats


@dataclass
class StatResult:
    """One condition-vs-reference comparison.

    Attributes:
        reference: The reference condition (first in the group).
        condition: The condition compared against the reference.
        condition_index: Index of ``condition`` within the full conditions list
            (used to position the on-plot marker).
        n_pairs: Number of repeats used in the test (paired: shared plates).
        p_value: The t-test p-value.
        significance: Star marker derived from ``p_value`` (``ns``/``*``/...).
        test: Which test produced the value
            (``paired_t``/``paired_logratio``/``unpaired_t``/
            ``ns_insufficient``).
        effect: Relative change of the condition's mean repeat-median against
            the reference's, as a fraction (``0.25`` = +25%). ``nan`` when the
            reference median is zero or the comparison could not be made.
            The p-value says whether a shift reproduces; ``effect`` says how
            big it is. With three repeats those answers diverge sharply --
            a 2% shift measured on 10,000 cells per well reproduces almost
            perfectly -- so callers can gate the on-plot marker on ``effect``
            via ``min_effect``.
    """

    reference: str
    condition: str
    condition_index: int
    n_pairs: int
    p_value: float
    significance: str
    test: str
    effect: float = float("nan")


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


def _paired_logratio_pvalue(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> tuple[float, int, str]:
    """Paired one-sample t-test on the per-repeat log fold-change.

    For matched reference/condition medians ``a``/``b`` (same repeats, same
    order) this tests whether ``L_i = ln(b_i / a_i)`` differs from 0 — i.e. a
    paired Student's t on the log scale. ``ln`` turns the multiplicative
    plate-to-plate baseline (×k) into an additive offset (+ln k) that the
    pairing removes, so the test tracks effect size rather than control
    stability (see the stats-improvement note).

    Non-positive or non-finite pairs are dropped (``ln`` undefined); a warning
    is logged. Requires >=2 surviving pairs.
    """
    mask = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    if not mask.all():
        logger.warning(
            f"log-ratio: dropping {int((~mask).sum())} non-positive/NaN "
            "pair(s) before the test"
        )
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        logger.warning(
            "log-ratio t-test needs >=2 valid shared repeats; got "
            f"{n}, setting p-value to 1.0"
        )
        return 1.0, n, "ns_insufficient"
    log_ratio = np.log(b / a)
    if np.all(log_ratio == 0):
        # No change on any plate -> not significant.
        return 1.0, n, "paired_logratio"
    if np.var(log_ratio) == 0:
        # Perfectly consistent non-zero fold-change -> maximally significant.
        return 0.0, n, "paired_logratio"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*catastrophic cancellation.*",
        )
        p = float(stats.ttest_1samp(log_ratio, 0.0).pvalue)
    return (p if not np.isnan(p) else 1.0), n, "paired_logratio"


def _pvalue_from_medians(
    ref: pd.Series,
    cond: pd.Series,
    paired: bool,
    *,
    normalise_within_plate: bool = False,
) -> tuple[float, int, str]:
    """Compute a t-test p-value between two per-repeat median series.

    Args:
        ref: Reference condition medians, indexed by repeat (plate_id).
        cond: Compared condition medians, indexed by repeat (plate_id).
        paired: If True, align on shared repeats and use ``ttest_rel``;
            otherwise use ``ttest_ind`` on all medians.
        normalise_within_plate: If True (and ``paired``), test the per-repeat
            log fold-change ``ln(cond/ref)`` against 0 instead of the absolute
            difference — removes a multiplicative plate baseline for
            ratio-scale readouts (counts/intensities/area). Ignored (with a
            warning) when ``paired`` is False.

    Returns:
        Tuple of (p_value, n, test_name).
    """
    if paired:
        shared = ref.index.intersection(cond.index)
        a = ref.loc[shared].to_numpy(dtype=float)
        b = cond.loc[shared].to_numpy(dtype=float)
        n = len(shared)
        if n < 2:
            logger.warning(
                "Paired t-test needs >=2 shared repeats; got "
                f"{n}, setting p-value to 1.0"
            )
            return 1.0, n, "ns_insufficient"
        if normalise_within_plate:
            return _paired_logratio_pvalue(a, b)
        diff = b - a
        if np.all(diff == 0):
            # Identical series - no difference, ttest_rel would return NaN.
            return 1.0, n, "paired_t"
        if np.var(diff) == 0:
            # Perfectly consistent non-zero difference across repeats -> the
            # maximally significant case (ttest_rel t -> inf).
            return 0.0, n, "paired_t"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*catastrophic cancellation.*",
            )
            p = float(stats.ttest_rel(a, b).pvalue)
        return (p if not np.isnan(p) else 1.0), n, "paired_t"

    if normalise_within_plate:
        logger.warning(
            "normalise_within_plate requires paired=True (needs same-plate "
            "reference); ignoring and using the unpaired absolute-value test."
        )
    a = ref.to_numpy(dtype=float)
    b = cond.to_numpy(dtype=float)
    n = min(len(a), len(b))
    if len(a) < 2 or len(b) < 2:
        logger.warning(
            "Unpaired t-test needs >=2 repeats per group, setting p-value to 1.0"
        )
        return 1.0, n, "ns_insufficient"
    if np.var(a) == 0 and np.var(b) == 0:
        # Both constant: significant only if the constants differ.
        return (1.0 if np.mean(a) == np.mean(b) else 0.0), n, "unpaired_t"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*catastrophic cancellation.*",
        )
        p = float(stats.ttest_ind(a, b).pvalue)
    return (p if not np.isnan(p) else 1.0), n, "unpaired_t"


def compute_significance(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    value_col: str,
    *,
    group_size: int = 1,
    repeat_col: str = "plate_id",
    paired: bool = True,
    normalise_within_plate: bool = False,
) -> tuple[pd.DataFrame, list[StatResult]]:
    """Compute replicate-level significance for a plot.

    Aggregates ``value_col`` to one median per ``repeat_col`` per condition,
    then compares each condition to the first condition of its group.

    Args:
        df: Data containing ``condition_col``, ``value_col`` and ``repeat_col``.
            May be per-cell or already per-repeat; it is reduced to per-repeat
            medians either way.
        conditions: Ordered conditions to compare.
        condition_col: Column holding the condition labels.
        value_col: Column holding the measurement to test.
        group_size: Conditions per group (1 = all vs the overall first).
        repeat_col: Column identifying a biological repeat (default plate_id).
        paired: Use a paired t-test aligned on shared repeats (default True).
        normalise_within_plate: If True (and ``paired``), test the per-repeat
            log fold-change ``ln(cond/ref)`` against 0 rather than the absolute
            difference. Use for positive ratio-scale readouts with a
            multiplicative batch effect (counts/intensities/area); do **not**
            use for bounded proportions (cell-cycle %, classification).

    Returns:
        Tuple of (per-repeat medians DataFrame, list of StatResult).
    """
    df2 = df[df[condition_col].isin(conditions)]
    if df2.empty or repeat_col not in df2.columns:
        return pd.DataFrame(columns=[repeat_col, condition_col, value_col]), []

    medians = (
        df2.groupby([repeat_col, condition_col])[value_col]
        .median()
        .reset_index()
    )
    by_cond = {
        cond: medians[medians[condition_col] == cond].set_index(repeat_col)[
            value_col
        ]
        for cond in conditions
    }

    results: list[StatResult] = []
    # group_size <= 1 means a single group spanning all conditions (everything
    # vs the overall first/control); >1 splits into consecutive windows, each
    # compared to its own first condition.
    gs = group_size if group_size and group_size > 1 else len(conditions)
    for group_start in range(0, len(conditions), gs):
        group = conditions[group_start : group_start + gs]
        if len(group) < 2:
            continue
        reference = group[0]
        ref_series = by_cond.get(reference)
        for offset, condition in enumerate(group[1:], start=1):
            idx = group_start + offset
            cond_series = by_cond.get(condition)
            if (
                ref_series is None
                or cond_series is None
                or len(ref_series) == 0
                or len(cond_series) == 0
            ):
                p, n, test = 1.0, 0, "ns_insufficient"
                effect = float("nan")
            else:
                p, n, test = _pvalue_from_medians(
                    ref_series,
                    cond_series,
                    paired,
                    normalise_within_plate=normalise_within_plate,
                )
                # Mean of the per-repeat medians on each side; relative to the
                # reference so the number is comparable across readouts with
                # different units.
                ref_mean = float(ref_series.mean())
                cond_mean = float(cond_series.mean())
                effect = (
                    (cond_mean - ref_mean) / ref_mean
                    if ref_mean
                    else float("nan")
                )
            results.append(
                StatResult(
                    reference=reference,
                    condition=condition,
                    condition_index=idx,
                    n_pairs=n,
                    p_value=p,
                    significance=get_significance_marker(p),
                    test=test,
                    effect=effect,
                )
            )

    logger.info(
        f"significance ({'paired' if paired else 'unpaired'}): "
        f"{[(r.condition, round(r.p_value, 4), r.significance) for r in results]}"
    )
    return medians, results


def annotate_significance(
    axes: Axes,
    results: list[StatResult],
    x_positions: list[float],
    y: float,
    *,
    fontsize: int = 6,
    show_ns: bool = True,
    min_effect: float | None = None,
    color: str | None = None,
) -> None:
    """Draw significance markers from ``results`` at the given x positions.

    ``show_ns=False`` draws only the star markers. ``min_effect`` additionally
    requires ``|effect| >= min_effect`` before a marker is drawn (``0.10`` =
    10% change against the reference), so a shift that reproduces perfectly
    but is too small to matter goes unmarked.

    ``color`` tints the markers, so a caller stacking several rows of stars
    can key each row to the series it belongs to.

    Neither option touches the computation: every comparison, its exact
    p-value and its effect size are still written to the stats CSV. They
    suppress markers only, and a pre-specified threshold applied uniformly
    must be stated in the figure legend to be honest.
    """
    for result in results:
        if not show_ns and result.significance == "ns":
            continue
        if (
            min_effect is not None
            and result.significance != "ns"
            and not (abs(result.effect) >= min_effect)
        ):
            continue
        if 0 <= result.condition_index < len(x_positions):
            axes.annotate(
                result.significance,
                xy=(x_positions[result.condition_index], y),
                xycoords="data",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=color if color else "black",
            )


def stats_results_to_dataframe(
    results: list[StatResult],
    *,
    value_label: str,
    extra: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert StatResults into a tidy p-value table for CSV export."""
    rows: list[dict[str, object]] = []
    for result in results:
        row: dict[str, object] = {"feature": value_label}
        if extra:
            row.update(extra)
        row.update(
            {
                "reference": result.reference,
                "condition": result.condition,
                "n_pairs": result.n_pairs,
                "p_value": result.p_value,
                "effect": result.effect,
                "significance": result.significance,
                "test": result.test,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def medians_to_dataframe(
    medians: pd.DataFrame,
    *,
    repeat_col: str,
    condition_col: str,
    value_col: str,
    value_label: str,
    extra: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert the per-repeat medians frame into a tidy table for CSV export."""
    out = medians.rename(
        columns={
            repeat_col: "plate_id",
            condition_col: "condition",
            value_col: "value",
        }
    ).copy()
    out.insert(0, "feature", value_label)
    extra_cols = list(extra.keys()) if extra else []
    for key, val in (extra or {}).items():
        out[key] = val
    return out[["feature", *extra_cols, "plate_id", "condition", "value"]]


def write_stats_csv(
    df: pd.DataFrame, path: Path, fig_id: str, kind: str
) -> None:
    """Write a stats/medians table next to a saved figure.

    Args:
        df: The table to write.
        path: Output directory (same as the figure's).
        fig_id: Figure id (sanitised title); the file is ``{fig_id}_{kind}.csv``.
        kind: ``"stats"`` (p-values) or ``"medians"``.
    """
    if df.empty:
        return
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / f"{fig_id}_{kind}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {kind} table to {out_path}")


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers (compute + annotate). Existing callers ignore
# the return value; they now receive the StatResult list so the plot builders
# can capture it for CSV export.
# ---------------------------------------------------------------------------


def calculate_pvalues(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    column: str,
    *,
    repeat_col: str = "plate_id",
    paired: bool = True,
) -> list[float]:
    """Calculate p-values for each condition against the first condition."""
    _, results = compute_significance(
        df,
        conditions,
        condition_col,
        column,
        group_size=1,
        repeat_col=repeat_col,
        paired=paired,
    )
    return [r.p_value for r in results]


def calculate_grouped_pvalues(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    column: str,
    group_size: int = 2,
    *,
    repeat_col: str = "plate_id",
    paired: bool = True,
) -> list[tuple[int, float]]:
    """Calculate within-group p-values against each group's first condition."""
    _, results = compute_significance(
        df,
        conditions,
        condition_col,
        column,
        group_size=group_size,
        repeat_col=repeat_col,
        paired=paired,
    )
    return [(r.condition_index, r.p_value) for r in results]


def set_significance_marks(
    axes: Axes,
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    y_max: float,
    *,
    repeat_col: str = "plate_id",
    paired: bool = True,
) -> list[StatResult]:
    """Annotate significance (all conditions vs the first) and return results."""
    _, results = compute_significance(
        df,
        conditions,
        condition_col,
        y_col,
        group_size=1,
        repeat_col=repeat_col,
        paired=paired,
    )
    x_positions = [float(i) for i in range(len(conditions))]
    annotate_significance(axes, results, x_positions, y_max)
    return results


def set_grouped_within_significance_marks(
    axes: Axes,
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    y_max: float,
    group_size: int = 2,
    x_positions: list[float] | None = None,
    *,
    repeat_col: str = "plate_id",
    paired: bool = True,
) -> list[StatResult]:
    """Annotate within-group significance and return results."""
    if x_positions is None:
        x_positions = [float(i) for i in range(len(conditions))]
    elif not isinstance(x_positions, list):
        x_positions = list(x_positions)
    _, results = compute_significance(
        df,
        conditions,
        condition_col,
        y_col,
        group_size=group_size,
        repeat_col=repeat_col,
        paired=paired,
    )
    annotate_significance(axes, results, x_positions, y_max)
    return results


def set_significance_marks_adaptive(
    axes: Axes,
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    y_max: float,
    group_size: int = 1,
    x_positions: list[float] | None = None,
    *,
    repeat_col: str = "plate_id",
    paired: bool = True,
) -> list[StatResult]:
    """Annotate significance, adapting to group_size, and return results."""
    if x_positions is None:
        x_positions = [float(i) for i in range(len(conditions))]
    elif not isinstance(x_positions, list):
        x_positions = list(x_positions)
    _, results = compute_significance(
        df,
        conditions,
        condition_col,
        y_col,
        group_size=group_size,
        repeat_col=repeat_col,
        paired=paired,
    )
    annotate_significance(axes, results, x_positions, y_max)
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
    *,
    repeat_col: str = "plate_id",
    paired: bool = True,
) -> list[StatResult]:
    """Annotate significance between adjacent conditions within each group.

    Legacy helper (sequential pairwise within a group). Uses the same
    replicate-level paired test and annotates at the midpoint between the two
    columns being compared.
    """
    n = len(conditions)
    if x_positions is None:
        x_positions = [float(i) for i in range(n)]
    elif not isinstance(x_positions, list):
        x_positions = list(x_positions)

    df2 = df[df[condition_col].isin(conditions)]
    medians = (
        df2.groupby([repeat_col, condition_col])[y_col].median().reset_index()
        if not df2.empty and repeat_col in df2.columns
        else pd.DataFrame(columns=[repeat_col, condition_col, y_col])
    )
    by_cond = {
        cond: medians[medians[condition_col] == cond].set_index(repeat_col)[
            y_col
        ]
        for cond in conditions
    }

    results: list[StatResult] = []
    for group_start in range(0, n, group_size):
        group = conditions[group_start : group_start + group_size]
        group_xs = x_positions[group_start : group_start + group_size]
        for i in range(len(group) - 1):
            left, right = by_cond.get(group[i]), by_cond.get(group[i + 1])
            if (
                left is None
                or right is None
                or len(left) == 0
                or len(right) == 0
            ):
                p, npairs, test = 1.0, 0, "ns_insufficient"
            else:
                p, npairs, test = _pvalue_from_medians(left, right, paired)
            significance = get_significance_marker(p)
            x_mid = (group_xs[i] + group_xs[i + 1]) / 2
            axes.annotate(
                significance,
                xy=(x_mid, y_max),
                xycoords="data",
                ha="center",
                va="bottom",
                fontsize=6,
            )
            results.append(
                StatResult(
                    reference=group[i],
                    condition=group[i + 1],
                    condition_index=group_start + i + 1,
                    n_pairs=npairs,
                    p_value=p,
                    significance=significance,
                    test=test,
                )
            )
    return results
