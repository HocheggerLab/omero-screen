"""Per-cell fate walker on synthetic PIP-FUCCI tracks."""

import pandas as pd
from cellview.tracks.fate import FateParams, follow_cells

G1 = {"pip": 1000.0, "geminin": 100.0}
S = {"pip": 50.0, "geminin": 400.0}
G2 = {"pip": 1000.0, "geminin": 800.0}


def _rows(tid, parent, frames, y, phase, x0=100.0, dx=10.0, area=400.0):
    """One repaired track in a fixed phase, moving ``dx`` px per frame."""
    return [
        {
            "track_id": tid,
            "parent_track_id": parent,
            "timepoint": t,
            "area": area,
            "y": y,
            "x": x0 + dx * i,
            "dna": 1000.0,
            **phase,
        }
        for i, t in enumerate(frames)
    ]


def _cycling_cell(tid, y):
    """G1 (0-9) -> S (10-19) -> G2 (20-29) -> two G1 daughters from 30."""
    rows = (
        _rows(tid, 0, range(10), y, G1)
        + _rows(tid, 0, range(10, 20), y, S, x0=200)
        + _rows(tid, 0, range(20, 30), y, G2, x0=300)
    )
    rows += _rows(tid + 1, tid, range(30, 41), y - 20, G1, x0=400)
    rows += _rows(tid + 2, tid, range(30, 41), y + 20, G1, x0=400)
    return rows


def _walk(rows, **kw):
    return follow_cells(pd.DataFrame(rows), 0, 40, FateParams(**kw), shape=(5000, 5000))


def test_cycling_cell_divides_after_g1_s_g2() -> None:
    """A full cycle ends in division with the phases in order."""
    cells, states = _walk(_cycling_cell(10, 500) + _cycling_cell(20, 1500))
    cell = cells.iloc[0]
    assert cell.outcome == "divided" and cell.outcome_frame == 30
    assert cell.start_phase == "G1" and not cell.excluded
    phases = states[states.cell == cell.cell].phase
    runs = phases[phases.ne(phases.shift())].tolist()
    assert runs == ["G1", "S", "G2"]
    assert "phase_order" not in cell.review_flags


def test_stationary_object_without_reporter_is_debris() -> None:
    """Debris neither moves nor expresses a reporter."""
    debris = _rows(99, 0, range(41), 3000, {"pip": 0.0, "geminin": 0.0}, dx=0.0)
    cells, _ = _walk(_cycling_cell(10, 500) + _cycling_cell(20, 1500) + debris)
    row = cells.set_index("start_track").loc[99]
    assert row.outcome == "debris" and row.excluded and row.review_flags == ""


def test_short_gap_is_bridged_and_flagged() -> None:
    """A track that breaks for one frame continues on its matching founder."""
    broken = _rows(50, 0, range(15), 2500, G1) + _rows(51, 0, range(17, 41), 2500, G1, x0=270)
    cells, _ = _walk(_cycling_cell(10, 500) + _cycling_cell(20, 1500) + broken)
    row = cells.set_index("start_track").loc[50]
    assert row.segments == "50 51"
    assert "gap" in row.review_flags
    assert row.outcome == "no_mitosis"


def test_track_ending_with_nothing_nearby_is_lost() -> None:
    """A cell that cannot be followed is censored and flagged."""
    lost = _rows(60, 0, range(15), 3500, G1)
    cells, _ = _walk(_cycling_cell(10, 500) + _cycling_cell(20, 1500) + lost)
    row = cells.set_index("start_track").loc[60]
    assert row.outcome == "lost" and row.censored and "lost" in row.review_flags


def test_brief_both_low_history_is_unclear_not_negative() -> None:
    """Early S has both reporters low; a short history is not called negative."""
    dim = _rows(70, 0, range(8), 4000, {"pip": 50.0, "geminin": 100.0})
    cells, _ = _walk(_cycling_cell(10, 500) + _cycling_cell(20, 1500) + dim)
    row = cells.set_index("start_track").loc[70]
    assert row.outcome != "no_reporter"
    assert "reporter_unclear" in row.review_flags
