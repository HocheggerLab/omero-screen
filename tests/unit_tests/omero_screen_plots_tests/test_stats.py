"""Tests for the replicate-level statistics and CSV export.

Covers the paired/unpaired t-test on per-plate medians, the first-in-group
comparison convention, shared-plate alignment, and the per-figure
``{title}_stats.csv`` / ``{title}_medians.csv`` export.
"""

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from omero_screen_plots.cellcycleplot_api import cellcycle_stacked
from omero_screen_plots.classificationplot_api import classification_plot
from omero_screen_plots.countplot_api import count_plot
from omero_screen_plots.countplot_factory import PlotType
from omero_screen_plots.featureplot_api import feature_plot
from omero_screen_plots.stats import (
    compute_significance,
    medians_to_dataframe,
    stats_results_to_dataframe,
    write_stats_csv,
)

CONDITIONS = ["ctrl", "c2", "c3"]


def _per_plate_frame(values_by_cond: dict[str, list[float]]) -> pd.DataFrame:
    """Build a per-plate medians frame: one value per (plate, condition)."""
    rows = []
    for cond, vals in values_by_cond.items():
        for plate, v in enumerate(vals, start=1):
            rows.append({"plate_id": plate, "condition": cond, "value": v})
    return pd.DataFrame(rows)


class TestComputeSignificance:
    """Replicate-level p-value computation."""

    def test_group_size_1_is_all_vs_first(self):
        """group_size=1 compares every condition to the overall first."""
        df = _per_plate_frame(
            {"ctrl": [10, 11, 12], "c2": [10, 11, 12], "c3": [20, 21, 22]}
        )
        _, results = compute_significance(
            df, CONDITIONS, "condition", "value", group_size=1, paired=True
        )
        # Two comparisons, both against the first condition (ctrl).
        assert [r.condition for r in results] == ["c2", "c3"]
        assert all(r.reference == "ctrl" for r in results)

    def test_null_data_is_ns(self):
        """Identical per-plate medians yield ns for every comparison."""
        df = _per_plate_frame(
            {"ctrl": [10, 11, 12], "c2": [10, 11, 12], "c3": [10, 11, 12]}
        )
        _, results = compute_significance(
            df, CONDITIONS, "condition", "value", paired=True
        )
        assert all(r.significance == "ns" for r in results)

    def test_consistent_shift_is_significant(self):
        """A consistent per-plate shift is significant; no shift is ns."""
        df = _per_plate_frame(
            {"ctrl": [10, 11, 12], "c2": [10, 11, 12], "c3": [30, 31, 32]}
        )
        _, results = compute_significance(
            df, CONDITIONS, "condition", "value", paired=True
        )
        by_cond = {r.condition: r for r in results}
        assert by_cond["c2"].significance == "ns"
        assert by_cond["c3"].p_value < 0.05
        assert by_cond["c3"].test == "paired_t"

    def test_paired_uses_only_shared_plates(self):
        """Paired test intersects on plates present in both conditions."""
        # c3 is missing plate 3 -> only plates 1,2 are shared with ctrl.
        df = pd.DataFrame(
            [
                {"plate_id": 1, "condition": "ctrl", "value": 10.0},
                {"plate_id": 2, "condition": "ctrl", "value": 11.0},
                {"plate_id": 3, "condition": "ctrl", "value": 12.0},
                {"plate_id": 1, "condition": "c3", "value": 20.0},
                {"plate_id": 2, "condition": "c3", "value": 21.0},
            ]
        )
        _, results = compute_significance(
            df, ["ctrl", "c3"], "condition", "value", paired=True
        )
        assert results[0].n_pairs == 2
        assert results[0].test == "paired_t"

    def test_insufficient_shared_pairs_is_ns(self):
        """Fewer than two shared plates -> ns_insufficient, p=1.0."""
        df = pd.DataFrame(
            [
                {"plate_id": 1, "condition": "ctrl", "value": 10.0},
                {"plate_id": 2, "condition": "ctrl", "value": 11.0},
                {"plate_id": 9, "condition": "c3", "value": 20.0},
            ]
        )
        _, results = compute_significance(
            df, ["ctrl", "c3"], "condition", "value", paired=True
        )
        assert results[0].test == "ns_insufficient"
        assert results[0].p_value == 1.0

    def test_group_size_2_compares_within_group(self):
        """group_size>1 compares each condition to its own group's first."""
        conds = ["a0", "a1", "b0", "b1"]
        df = _per_plate_frame(
            {
                "a0": [10, 11, 12],
                "a1": [10, 11, 12],
                "b0": [50, 51, 52],
                "b1": [10, 11, 12],
            }
        )
        _, results = compute_significance(
            df, conds, "condition", "value", group_size=2, paired=True
        )
        refs = {r.condition: r.reference for r in results}
        # a1 vs a0 (group 1), b1 vs b0 (group 2) — not vs the overall first.
        assert refs == {"a1": "a0", "b1": "b0"}

    def test_unpaired_toggle_uses_ttest_ind(self):
        """paired=False routes through the unpaired t-test."""
        df = _per_plate_frame(
            {"ctrl": [10, 11, 12], "c2": [10, 11, 12], "c3": [30, 31, 32]}
        )
        _, results = compute_significance(
            df, CONDITIONS, "condition", "value", paired=False
        )
        assert all(r.test in ("unpaired_t", "ns_insufficient") for r in results)


class TestLogRatioSignificance:
    """Paired log-ratio test (``normalise_within_plate``) for ratio-scale data.

    Tests a paired one-sample t on ``L_i = ln(cond_i / ref_i)`` — removes a
    multiplicative plate baseline so significance tracks effect size rather
    than control stability (see the 2026-07-21 stats-improvement note).
    """

    def test_flag_off_matches_default(self):
        """Flag off is identical to the default paired_t path (parity)."""
        df = _per_plate_frame(
            {"ctrl": [1000, 800, 1200], "c2": [520, 430, 610]}
        )
        _, off = compute_significance(
            df, ["ctrl", "c2"], "condition", "value",
            paired=True, normalise_within_plate=False,
        )
        _, default = compute_significance(
            df, ["ctrl", "c2"], "condition", "value", paired=True,
        )
        assert off[0].p_value == default[0].p_value
        assert off[0].test == default[0].test == "paired_t"

    def test_equivalent_to_paired_t_on_log(self):
        """Log-ratio p equals a paired t-test on the logged medians."""
        from scipy import stats as sp

        df = _per_plate_frame(
            {"ctrl": [1000, 800, 1200], "c2": [520, 430, 610]}
        )
        _, res = compute_significance(
            df, ["ctrl", "c2"], "condition", "value",
            paired=True, normalise_within_plate=True,
        )
        assert res[0].test == "paired_logratio"
        manual = sp.ttest_rel(
            np.log([520.0, 430.0, 610.0]), np.log([1000.0, 800.0, 1200.0])
        ).pvalue
        assert res[0].p_value == pytest.approx(manual)

    def test_equal_effect_consistent_across_baseline_noise(self):
        """Same effect -> consistent significance despite control noise."""
        stable = _per_plate_frame(
            {"ctrl": [1000, 990, 1010], "c2": [530, 525, 535]}
        )
        noisy = _per_plate_frame(
            {"ctrl": [1000, 700, 1300], "c2": [530, 371, 689]}
        )
        _, rs = compute_significance(
            stable, ["ctrl", "c2"], "condition", "value",
            paired=True, normalise_within_plate=True,
        )
        _, rn = compute_significance(
            noisy, ["ctrl", "c2"], "condition", "value",
            paired=True, normalise_within_plate=True,
        )
        assert rs[0].significance != "ns"
        assert rn[0].significance != "ns"

    def test_effect_size_monotonicity(self):
        """A larger fold-change gives a smaller p (with matched noise)."""
        small = _per_plate_frame(  # ~10% drop, noisy fold-change
            {"ctrl": [1000, 800, 1200], "c2": [880, 730, 1100]}
        )
        large = _per_plate_frame(  # ~50% drop, similar relative noise
            {"ctrl": [1000, 800, 1200], "c2": [480, 420, 620]}
        )
        _, rsm = compute_significance(
            small, ["ctrl", "c2"], "condition", "value",
            paired=True, normalise_within_plate=True,
        )
        _, rlg = compute_significance(
            large, ["ctrl", "c2"], "condition", "value",
            paired=True, normalise_within_plate=True,
        )
        assert rlg[0].p_value <= rsm[0].p_value

    def test_nonpositive_pairs_dropped(self):
        """Non-positive medians are dropped before the log (kept pairs used)."""
        df = _per_plate_frame(
            {"ctrl": [1000, 0, 1200], "c2": [500, 400, 600]}
        )
        _, res = compute_significance(
            df, ["ctrl", "c2"], "condition", "value",
            paired=True, normalise_within_plate=True,
        )
        assert res[0].n_pairs == 2  # plate 2 (ctrl=0) dropped
        assert res[0].test == "paired_logratio"

    def test_insufficient_valid_pairs_is_ns(self):
        """Fewer than two valid pairs after dropping -> ns_insufficient."""
        df = _per_plate_frame(
            {"ctrl": [1000, 0, 0], "c2": [500, 400, 600]}
        )
        _, res = compute_significance(
            df, ["ctrl", "c2"], "condition", "value",
            paired=True, normalise_within_plate=True,
        )
        assert res[0].test == "ns_insufficient"
        assert res[0].p_value == 1.0

    def test_ignored_when_unpaired(self):
        """normalise_within_plate needs pairing; falls back to unpaired_t."""
        df = _per_plate_frame(
            {"ctrl": [1000, 800, 1200], "c2": [500, 400, 600]}
        )
        _, res = compute_significance(
            df, ["ctrl", "c2"], "condition", "value",
            paired=False, normalise_within_plate=True,
        )
        assert res[0].test == "unpaired_t"


class TestStatsTables:
    """Tidy-table converters and CSV writer."""

    def test_results_dataframe_columns(self):
        """The p-value table has the expected tidy columns (+extra)."""
        df = _per_plate_frame(
            {"ctrl": [10, 11, 12], "c2": [30, 31, 32]}
        )
        _, results = compute_significance(
            df, ["ctrl", "c2"], "condition", "value"
        )
        out = stats_results_to_dataframe(
            results, value_label="feat", extra={"phase": "G1"}
        )
        assert list(out.columns) == [
            "feature",
            "phase",
            "reference",
            "condition",
            "n_pairs",
            "p_value",
            "significance",
            "test",
        ]

    def test_medians_dataframe_columns(self):
        """The medians table has the expected tidy columns."""
        medians = _per_plate_frame(
            {"ctrl": [10, 11, 12], "c2": [30, 31, 32]}
        ).rename(columns={"value": "feat"})
        out = medians_to_dataframe(
            medians,
            repeat_col="plate_id",
            condition_col="condition",
            value_col="feat",
            value_label="feat",
        )
        assert list(out.columns) == [
            "feature",
            "plate_id",
            "condition",
            "value",
        ]

    def test_write_stats_csv(self, tmp_path):
        """write_stats_csv writes {fig_id}_{kind}.csv."""
        df = pd.DataFrame({"a": [1, 2]})
        write_stats_csv(df, tmp_path, "myfig", "stats")
        assert (tmp_path / "myfig_stats.csv").exists()

    def test_write_stats_csv_skips_empty(self, tmp_path):
        """An empty table writes nothing."""
        write_stats_csv(pd.DataFrame(), tmp_path, "myfig", "stats")
        assert not (tmp_path / "myfig_stats.csv").exists()


@pytest.fixture
def feature_df():
    """Three plates; c3 carries a consistent shift, c2 does not."""
    rng = np.random.default_rng(0)
    rows = []
    for plate in (1, 2, 3):
        for cond in CONDITIONS:
            shift = 2000 if cond == "c3" else 0
            for _ in range(200):
                rows.append(
                    {
                        "plate_id": plate,
                        "well": "A1",
                        "experiment": "e",
                        "condition": cond,
                        "cell_line": "HeLa",
                        "intensity_mean_EdU_nucleus": float(
                            rng.normal(3000 + shift, 600)
                        ),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def classification_df():
    """Three plates; c3 shifts the mito fraction, the others are flat."""
    rng = np.random.default_rng(1)
    rows = []
    wid = 0
    for plate in (1, 2, 3):
        for cond in CONDITIONS:
            wid += 1
            p = [0.15, 0.8, 0.05] if cond != "c3" else [0.4, 0.55, 0.05]
            for cls in rng.choice(["mito", "inter", "apop"], size=300, p=p):
                rows.append(
                    {
                        "plate_id": plate,
                        "condition": cond,
                        "cell_line": "HeLa",
                        "well_id": wid,
                        "experiment": "e",
                        "Class": cls,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def cellcycle_df():
    """Three plates of cell-cycle phase calls; c3 shifts toward G2/M."""
    rng = np.random.default_rng(2)
    rows = []
    for plate in (1, 2, 3):
        for cond in CONDITIONS:
            p = (
                [0.5, 0.2, 0.3]
                if cond != "c3"
                else [0.3, 0.2, 0.5]
            )
            for phase in rng.choice(["G1", "S", "G2/M"], size=300, p=p):
                rows.append(
                    {
                        "plate_id": plate,
                        "well": "A1",
                        "well_id": plate * 10,
                        "experiment": "e",
                        "condition": cond,
                        "cell_line": "HeLa",
                        "cell_cycle": phase,
                    }
                )
    return pd.DataFrame(rows)


_STATS_COLS = [
    "feature",
    "reference",
    "condition",
    "n_pairs",
    "p_value",
    "significance",
    "test",
]


def _markers(ax) -> list[str]:
    """Significance markers drawn on the axes."""
    valid = {"ns", "*", "**", "***"}
    return [t.get_text() for t in ax.texts if t.get_text() in valid]


class TestOnPlotSignificance:
    """Where significance stars are (and aren't) drawn."""

    @patch("matplotlib.pyplot.show")
    def test_count_triplicate_shows_significance(
        self, mock_show, feature_df
    ):
        """Count plot annotates significance even in triplicate mode."""
        _, ax = count_plot(
            feature_df,
            norm_control="ctrl",
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            plot_type=PlotType.ABSOLUTE,
            show_triplicates=True,
            save=False,
        )
        # One marker per non-reference condition.
        assert len(_markers(ax)) == 2

    @patch("matplotlib.pyplot.show")
    def test_classification_multiclass_has_no_onplot_stars(
        self, mock_show, classification_df
    ):
        """Multi-class classification draws no stars (ambiguous), CSV aside."""
        _, ax = classification_plot(
            classification_df,
            classes=["mito", "inter", "apop"],
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            show_triplicates=True,
            save=False,
        )
        assert _markers(ax) == []

    @patch("matplotlib.pyplot.show")
    def test_classification_single_class_shows_stars(
        self, mock_show, classification_df
    ):
        """A single-class classification annotates significance."""
        _, ax = classification_plot(
            classification_df,
            classes=["mito"],
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            show_triplicates=True,
            save=False,
        )
        assert len(_markers(ax)) == 2


def _bar_span(ax) -> float:
    """Horizontal span of the filled bar centres."""
    xs = [
        p.get_x() + p.get_width() / 2
        for p in ax.patches
        if p.get_facecolor()[3] > 0
    ]
    return max(xs) - min(xs)


class TestBetweenGroupGap:
    """between_group_gap controls spacing in triplicate mode at group_size=1."""

    @patch("matplotlib.pyplot.show")
    def test_count_triplicate_gap_widens_layout(self, mock_show, feature_df):
        """A larger between_group_gap spreads count triplicate clusters."""
        import matplotlib.pyplot as plt

        spans = []
        for gap in (0.2, 1.5):
            _, ax = plt.subplots()
            count_plot(
                feature_df,
                norm_control="ctrl",
                conditions=CONDITIONS,
                condition_col="condition",
                selector_col=None,
                plot_type=PlotType.ABSOLUTE,
                show_triplicates=True,
                group_size=1,
                between_group_gap=gap,
                axes=ax,
            )
            spans.append(_bar_span(ax))
            plt.close()
        assert spans[1] > spans[0]

    @patch("matplotlib.pyplot.show")
    def test_classification_triplicate_gap_widens_layout(
        self, mock_show, classification_df
    ):
        """A larger between_group_gap spreads classification triplicate clusters."""
        import matplotlib.pyplot as plt

        spans = []
        for gap in (0.2, 1.5):
            _, ax = plt.subplots()
            classification_plot(
                classification_df,
                classes=["mito", "inter", "apop"],
                conditions=CONDITIONS,
                condition_col="condition",
                selector_col=None,
                show_triplicates=True,
                group_size=1,
                between_group_gap=gap,
                axes=ax,
            )
            spans.append(_bar_span(ax))
            plt.close()
        assert spans[1] > spans[0]


class TestCsvExport:
    """End-to-end: saving a figure writes the two sibling CSVs."""

    @patch("matplotlib.pyplot.show")
    def test_feature_plot_writes_both_csvs(
        self, mock_show, feature_df, tmp_path
    ):
        """feature_plot with save=True writes stats and medians CSVs."""
        feature_plot(
            feature_df,
            feature="intensity_mean_EdU_nucleus",
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            save=True,
            path=tmp_path,
            title="edu",
        )
        stats = pd.read_csv(tmp_path / "edu_stats.csv")
        medians = pd.read_csv(tmp_path / "edu_medians.csv")
        assert list(stats.columns) == _STATS_COLS
        # One row per non-reference condition.
        assert set(stats["condition"]) == {"c2", "c3"}
        # 3 plates x 3 conditions of per-plate medians.
        assert len(medians) == 9

    @patch("matplotlib.pyplot.show")
    def test_no_csv_when_not_saving(self, mock_show, feature_df, tmp_path):
        """No CSVs are written when save=False."""
        feature_plot(
            feature_df,
            feature="intensity_mean_EdU_nucleus",
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            save=False,
            path=tmp_path,
            title="edu",
        )
        assert not list(tmp_path.glob("*.csv"))

    @patch("matplotlib.pyplot.show")
    def test_count_plot_writes_csvs(self, mock_show, feature_df, tmp_path):
        """count_plot with save=True writes stats and medians CSVs."""
        count_plot(
            feature_df,
            norm_control="ctrl",
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            plot_type=PlotType.ABSOLUTE,
            save=True,
            path=tmp_path,
            title="counts",
        )
        assert (tmp_path / "counts_stats.csv").exists()
        assert (tmp_path / "counts_medians.csv").exists()

    @patch("matplotlib.pyplot.show")
    def test_classification_exports_all_classes(
        self, mock_show, classification_df, tmp_path
    ):
        """Classification CSV covers every class, not just the annotated one."""
        classification_plot(
            classification_df,
            classes=["mito", "inter", "apop"],
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            show_triplicates=True,
            stats_class="mito",
            save=True,
            path=tmp_path,
            title="classes",
        )
        stats = pd.read_csv(tmp_path / "classes_stats.csv")
        # Every class appears in the stats CSV, not just the annotated one.
        assert set(stats["feature"]) == {"mito", "inter", "apop"}
        assert (tmp_path / "classes_medians.csv").exists()

    @patch("matplotlib.pyplot.show")
    def test_save_stats_on_shared_axes_writes_csv_not_figure(
        self, mock_show, feature_df, tmp_path
    ):
        """save_stats=True on a provided axes exports CSVs but no figure PDF."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        feature_plot(
            feature_df,
            feature="intensity_mean_EdU_nucleus",
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            axes=ax,
            save_stats=True,
            path=tmp_path,
            title="edu",
        )
        assert (tmp_path / "edu_stats.csv").exists()
        assert (tmp_path / "edu_medians.csv").exists()
        # Embedded in a provided axes -> the sub-plot must not write a figure.
        assert not list(tmp_path.glob("*.pdf"))

    @patch("matplotlib.pyplot.show")
    def test_save_stats_without_path_writes_nothing(
        self, mock_show, feature_df, tmp_path
    ):
        """save_stats=True but no path -> nothing written (warns)."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        feature_plot(
            feature_df,
            feature="intensity_mean_EdU_nucleus",
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            axes=ax,
            save_stats=True,
            path=None,
            title="edu",
        )
        assert not list(tmp_path.glob("*.csv"))

    @patch("matplotlib.pyplot.show")
    def test_stacked_cellcycle_exports_per_phase(
        self, mock_show, cellcycle_df, tmp_path
    ):
        """Stacked cell cycle exports per-phase stats even without stars."""
        cellcycle_stacked(
            cellcycle_df,
            conditions=CONDITIONS,
            condition_col="condition",
            selector_col=None,
            save=True,
            path=tmp_path,
            title="cc",
        )
        stats = pd.read_csv(tmp_path / "cc_stats.csv")
        # One feature column entry per cell-cycle phase present.
        assert {"G1", "S", "G2/M"}.issubset(set(stats["feature"]))
        assert (tmp_path / "cc_medians.csv").exists()
