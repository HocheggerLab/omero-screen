"""Unit tests for omero_screen.alignment_qc.

These exercise the pure metric functions without an OMERO connection, using
small hand-built DataFrames and arrays where the correct answer is known.
"""

import numpy as np
import pandas as pd
import pytest

from omero_screen import alignment_qc as qc

# Helpers -------------------------------------------------------------------


def _sample_df():
    """Sample alignment table: one well with 3 fields + one blank frame."""
    return pd.DataFrame(
        {
            "plate": [2, 2, 2, 2],
            "well": ["A1", "A1", "A1", "A1"],
            "sample": [0, 1, 2, 3],
            "image_id": [10, 11, 12, 13],
            "x": [3.0, 4.0, 0.0, 0.0],  # last is a blank frame
            "y": [4.0, 3.0, 0.0, 0.0],
        }
    )


def _agg_df():
    """Aggregated table: 3 master cells, one repeat plate (index 0).

    Master centroids at (10,10),(20,20),(30,30). Repeat centroids stored as the
    already-aligned coordinates (master - residual). Third master cell is
    unmatched (repeat NaN); an extra repeat-only row is unmatched from master.
    """
    return pd.DataFrame(
        {
            "well": ["A1", "A1", "A1", "A1"],
            "image_id": [10, 10, 10, np.nan],
            "centroid-1": [10.0, 20.0, 30.0, np.nan],
            "centroid-0": [10.0, 20.0, 30.0, np.nan],
            "image_id.0": [100, 100, np.nan, 100],
            "centroid-1.0": [11.0, 22.0, np.nan, 50.0],
            "centroid-0.0": [10.0, 20.0, np.nan, 50.0],
        }
    )


# repeat_indices / well_to_grid --------------------------------------------


def test_repeat_indices():
    df = pd.DataFrame(columns=["centroid-1.0", "centroid-1.1", "centroid-1"])
    assert qc.repeat_indices(df) == [0, 1]


@pytest.mark.parametrize(
    "well,expected",
    [("A1", (0, 0)), ("B3", (1, 2)), ("H12", (7, 11)), ("AA1", (26, 0))],
)
def test_well_to_grid(well, expected):
    assert qc.well_to_grid(well) == expected


def test_well_to_grid_bad():
    with pytest.raises(ValueError):
        qc.well_to_grid("not-a-well")


# shift_summary -------------------------------------------------------------


def test_shift_summary_magnitude_and_drop_zero():
    out = qc.shift_summary(_sample_df(), pixel_size_um=0.5)
    # blank (0,0) frame dropped -> 2 rows remain
    assert len(out) == 2
    # (3,4) -> 5 px -> 2.5 um
    assert out["magnitude_px"].tolist() == [5.0, 5.0]
    assert out["magnitude_um"].tolist() == [2.5, 2.5]


def test_shift_summary_keep_zero():
    out = qc.shift_summary(_sample_df(), drop_zero=False)
    assert len(out) == 4


# per_well_agreement --------------------------------------------------------


def test_per_well_agreement_mean_and_residual():
    out = qc.per_well_agreement(_sample_df(), iqr=0.0)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_fields"] == 2
    # mean of (3,4) and (4,3) = (3.5, 3.5)
    assert row["mean_x"] == pytest.approx(3.5)
    assert row["mean_y"] == pytest.approx(3.5)
    # each field is sqrt(0.5) from the mean -> rms = sqrt(0.5)
    assert row["rms_residual_px"] == pytest.approx(np.sqrt(0.5))


def test_per_well_agreement_perfect():
    df = pd.DataFrame(
        {
            "plate": [2, 2],
            "well": ["A1", "A1"],
            "sample": [0, 1],
            "image_id": [1, 2],
            "x": [5.0, 5.0],
            "y": [5.0, 5.0],
        }
    )
    out = qc.per_well_agreement(df, iqr=0.0)
    assert out.iloc[0]["rms_residual_px"] == pytest.approx(0.0)


# matched_residuals ---------------------------------------------------------


def test_matched_residuals_before_after_per_well():
    alignment = pd.DataFrame(
        {"plate": [2], "well": ["A1"], "x": [1.0], "y": [0.0]}
    )
    out = qc.matched_residuals(_agg_df(), alignment, pixel_size_um=2.0)
    # two matched cells (third master unmatched, extra repeat unmatched)
    assert len(out) == 2
    # cell 1: after = (10-11, 10-10) = (-1,0) -> 1.0 ; before = after - shift(1,0) = (-2,0) -> 2.0
    r0 = out.iloc[0]
    assert r0["residual_after_px"] == pytest.approx(1.0)
    assert r0["residual_before_px"] == pytest.approx(2.0)
    assert r0["residual_after_um"] == pytest.approx(2.0)
    # cell 2: after = (20-22,0) = (-2,0) -> 2.0 ; before = (-3,0) -> 3.0
    r1 = out.iloc[1]
    assert r1["residual_after_px"] == pytest.approx(2.0)
    assert r1["residual_before_px"] == pytest.approx(3.0)


def test_matched_residuals_empty_when_no_match():
    agg = _agg_df().copy()
    agg["centroid-1.0"] = np.nan  # nothing matched
    alignment = pd.DataFrame(
        {"plate": [2], "well": ["A1"], "x": [1.0], "y": [0.0]}
    )
    out = qc.matched_residuals(agg, alignment)
    assert out.empty


# match_rate ----------------------------------------------------------------


def test_match_rate():
    out = qc.match_rate(_agg_df())
    row = out.iloc[0]
    assert row["n_master"] == 3
    assert row["n_repeat"] == 3
    assert row["n_matched"] == 2
    # min(3,3) = 3 -> 2/3
    assert row["match_fraction"] == pytest.approx(2 / 3)


# binary_overlap ------------------------------------------------------------


def test_binary_overlap_shift_recovers_perfect():
    # A 5x5 mask with a 2x2 block; repeat is the same block shifted by (x=1,y=0)
    base = np.zeros((6, 6), dtype=int)
    base[2:4, 2:4] = 1
    # repeat shifted right by 1 column (x=+1): block at columns 3:5
    rep = np.zeros((6, 6), dtype=int)
    rep[2:4, 3:5] = 1
    scores = qc.binary_overlap(base, rep, shift_xy=(1.0, 0.0))
    # after undoing the +1 x shift, footprints coincide -> dice 1.0
    assert scores["dice_after"] == pytest.approx(1.0)
    assert scores["iou_after"] == pytest.approx(1.0)
    # before: 2x2 vs 2x2 overlapping in one column -> inter=2, dice=2*2/8=0.5
    assert scores["dice_before"] == pytest.approx(0.5)


def test_binary_overlap_empty_masks():
    z = np.zeros((4, 4), dtype=int)
    scores = qc.binary_overlap(z, z, shift_xy=(0.0, 0.0))
    assert np.isnan(scores["dice_after"])


# summarise -----------------------------------------------------------------


def test_summarise_combines_metrics():
    alignment = pd.DataFrame(
        {"plate": [2], "well": ["A1"], "x": [1.0], "y": [0.0]}
    )
    shift_df = qc.shift_summary(_sample_df())
    agreement_df = qc.per_well_agreement(_sample_df(), iqr=0.0)
    residual_df = qc.matched_residuals(_agg_df(), alignment)
    match_df = qc.match_rate(_agg_df())
    summary = qc.summarise(shift_df, agreement_df, residual_df, match_df)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["repeat"] == 0
    assert row["match_fraction"] == pytest.approx(2 / 3)
    assert row["median_residual_after_px"] == pytest.approx(1.5)


# across_replicate_stats ----------------------------------------------------


def _replicate_long():
    """Three biological replicates, one repeat round, differing medians."""
    rows = []
    # replicate medians will be 2, 4, 6 -> mean 4, sd 2
    for master, val in [(10, 2.0), (20, 4.0), (30, 6.0)]:
        for _ in range(50):  # many cells per replicate (must NOT inflate n)
            rows.append({"master": master, "repeat": 0, "value": val})
    return pd.DataFrame(rows)


def test_across_replicate_stats_uses_replicate_n_not_cells():
    out = qc.across_replicate_stats(_replicate_long(), "value", by="repeat")
    assert len(out) == 1
    row = out.iloc[0]
    # n is the number of replicates (3), not the 150 cells
    assert row["n_replicates"] == 3
    assert row["mean_of_medians"] == pytest.approx(4.0)
    assert row["sd"] == pytest.approx(2.0)
    assert row["sem"] == pytest.approx(2.0 / np.sqrt(3))
    assert row["replicate_medians"] == [2.0, 4.0, 6.0]


def test_across_replicate_stats_pooled():
    out = qc.across_replicate_stats(_replicate_long(), "value", by=None)
    assert len(out) == 1
    assert out.iloc[0]["n_replicates"] == 3


# plot_superplot ------------------------------------------------------------


def test_plot_superplot_runs():
    df = _replicate_long()
    df = pd.concat([df, df.assign(repeat=1, value=df["value"] + 1)])
    fig = qc.plot_superplot(df, "value", group_col="repeat", master_col="master")
    # two rounds -> two x ticks
    assert len(fig.axes[0].get_xticks()) == 2


def test_plot_superplot_ymax_clips_view_not_stats():
    df = _replicate_long().copy()
    df.loc[df.index[0], "value"] = 9999.0  # a gross outlier
    fig = qc.plot_superplot(df, "value", ymax=10.0)
    assert fig.axes[0].get_ylim()[1] == pytest.approx(10.0)


def test_plot_qc_panel_runs():
    n = 40
    masters = [10, 20, 30]
    shift = pd.concat(
        pd.DataFrame(
            {"repeat": r, "master": m, "magnitude_px": 3.0}
            for r in (0, 1)
            for m in masters
        )
        for _ in range(n)
    )
    resid = pd.concat(
        pd.DataFrame(
            {
                "repeat": r,
                "master": m,
                "residual_after_px": 0.5,
                "residual_before_px": 5.0,
            }
            for r in (0, 1)
            for m in masters
        )
        for _ in range(n)
    )
    overlap = pd.concat(
        pd.DataFrame(
            {"repeat": r, "master": m, "dice_after": 0.9}
            for r in (0, 1)
            for m in masters
        )
        for _ in range(n)
    )
    fig = qc.plot_qc_panel(shift, resid, overlap)
    assert len(fig.axes) == 3
