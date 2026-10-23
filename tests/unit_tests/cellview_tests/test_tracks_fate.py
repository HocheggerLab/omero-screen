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
    return follow_cells(
        pd.DataFrame(rows), 0, 40, FateParams(**kw), shape=(5000, 5000)
    )


def test_cycling_cell_divides_after_g1_s_g2() -> None:
    """A full cycle ends in division with the phases in order."""
    cells, states = _walk(_cycling_cell(10, 500) + _cycling_cell(20, 1500))
    cell = cells.iloc[0]
    assert cell.outcome == "divided" and cell.outcome_frame == 30
    assert cell.start_phase == "G1" and not cell.excluded
    phases = states[states.cell == cell.cell].phase
    runs = phases[phases.ne(phases.shift())].tolist()
    assert runs == ["G1", "S", "G2"]
    assert "geminin_jump" not in cell.review_flags


def test_stationary_object_without_reporter_is_debris() -> None:
    """Debris neither moves nor expresses a reporter."""
    debris = _rows(
        99, 0, range(41), 3000, {"pip": 0.0, "geminin": 0.0}, dx=0.0
    )
    cells, _ = _walk(_cycling_cell(10, 500) + _cycling_cell(20, 1500) + debris)
    row = cells.set_index("start_track").loc[99]
    assert row.outcome == "debris" and row.excluded and row.review_flags == ""


def test_short_gap_is_bridged_and_flagged() -> None:
    """A track that breaks for one frame continues on its matching founder."""
    broken = _rows(50, 0, range(15), 2500, G1) + _rows(
        51, 0, range(17, 41), 2500, G1, x0=270
    )
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
    assert (
        row.outcome == "lost" and row.censored and "lost" in row.review_flags
    )


def test_brief_both_low_history_is_unclear_not_negative() -> None:
    """Early S has both reporters low; a short history is not called negative."""
    dim = _rows(70, 0, range(8), 4000, {"pip": 50.0, "geminin": 100.0})
    cells, _ = _walk(_cycling_cell(10, 500) + _cycling_cell(20, 1500) + dim)
    row = cells.set_index("start_track").loc[70]
    assert row.outcome != "no_reporter"
    assert "reporter_unclear" in row.review_flags


def test_geminin_fall_without_split_is_flagged() -> None:
    """Geminin never falls along one cell outside mitosis: a fall is a bad link."""
    rows = _rows(5, 0, range(20), 500, G2) + _rows(
        5, 0, range(20, 41), 500, {"pip": 1000.0, "geminin": 450.0}, x0=300
    )
    cells, _ = _walk(rows + _cycling_cell(10, 1500))
    cell = cells.set_index("start_track").loc[5]
    assert "geminin_jump" in cell.review_flags
    assert cell.outcome == "no_mitosis"


def test_one_frame_geminin_spike_is_not_an_exit() -> None:
    """A G1 nucleus that compacts for one frame neither exits nor jumps."""
    rows = _rows(5, 0, range(41), 500, G1)
    rows[10]["geminin"] = 900.0
    cells, _ = _walk(rows + _cycling_cell(10, 1500))
    cell = cells.set_index("start_track").loc[5]
    assert cell.outcome == "no_mitosis"
    assert "geminin_jump" not in cell.review_flags


def _ending(tid, y, tail, x0=1000.0):
    """A G1 cell followed to t20, then the ``tail`` rows (area, dna) until it ends."""
    rows = _rows(tid, 0, range(20), y, G1, x0=x0)
    for i, (area, dna) in enumerate(tail):
        rows += _rows(tid, 0, [20 + i], y, G1, x0=x0 + 200 + 10 * i, area=area)
        rows[-1]["dna"] = dna
    return rows


def test_pyknotic_end_is_death() -> None:
    """Small and bright for three frames at the end: the nucleus died."""
    rows = _ending(5, 500, [(150.0, 2500.0)] * 4)
    cells, _ = _walk(rows + _cycling_cell(10, 1500))
    cell = cells.set_index("start_track").loc[5]
    assert cell.outcome == "death" and cell.outcome_frame == 20


def test_dim_shrinking_end_is_lost_not_death() -> None:
    """A track ending on a small dim mask (speck, partial mask) is lost."""
    rows = _ending(5, 500, [(150.0, 400.0)] * 4)
    cells, _ = _walk(rows + _cycling_cell(10, 1500))
    assert cells.set_index("start_track").loc[5].outcome == "lost"


def test_pyknotic_end_at_the_border_left_the_field() -> None:
    """At the image border a small mask is the nucleus leaving the field."""
    rows = _ending(5, 10, [(150.0, 2500.0)] * 4)
    cells, _ = _walk(rows + _cycling_cell(10, 1500))
    assert cells.set_index("start_track").loc[5].outcome == "left_field"


def test_track_heading_out_of_the_field_left_it() -> None:
    """A track that ends at the border after moving towards it left the
    field: censored, but not flagged lost or edge, so it is not queued."""
    out = _rows(60, 0, range(15), 3500, G1, x0=4800, dx=12)  # ends at x 4968
    cells, _ = _walk(_cycling_cell(10, 500) + out)
    row = cells.set_index("start_track").loc[60]
    assert row.outcome == "left_field" and row.censored
    assert "lost" not in row.review_flags and "edge" not in row.review_flags


def test_track_ending_near_the_border_but_not_heading_out_is_lost() -> None:
    """Moving along the border and then vanishing is a tracking failure."""
    along = [
        {**r, "x": 4970.0, "y": 1000.0 + 10 * r["timepoint"]}
        for r in _rows(60, 0, range(15), 0, G1)
    ]
    cells, _ = _walk(_cycling_cell(10, 500) + along)
    assert cells.set_index("start_track").loc[60].outcome == "lost"


def test_switch_onto_a_neighbour_is_flagged_and_proposed() -> None:
    """A raw track that jumps onto a small dim object while the real nucleus
    carries on under a new id is cut at the jump, and the neighbour proposed."""
    import json

    # Track 70: a G1 nucleus for 10 frames, then (same raw id) a quarter-size,
    # PIP-dark object. Track 71 appears at frame 10 next to where 70 was,
    # with 70's area and reporters, and runs to the end.
    good = _rows(70, 0, range(10), 2000, G1, area=600.0)
    wrong = _rows(
        70,
        0,
        range(10, 41),
        2000,
        {"pip": 50.0, "geminin": 100.0},
        x0=200,
        area=150.0,
    )
    real = _rows(71, 0, range(10, 41), 2010, G1, x0=200, area=600.0)
    cells, states = _walk(
        _cycling_cell(10, 500) + _cycling_cell(20, 1500) + good + wrong + real
    )
    row = cells.set_index("start_track").loc[70]
    assert "switch" in row.review_flags
    assert row.outcome == "lost" and row.outcome_frame == 10
    assert states[states.cell == row.cell].timepoint.max() == 9
    props = json.loads(row.proposals)
    assert props[-1]["op"] == "link" and props[-1]["track"] == 71
    assert props[-1]["frame"] == 10 and props[-1]["confidence"] == "uncertain"


def test_mis_segmented_frame_without_a_neighbour_is_not_a_switch() -> None:
    """A one-frame shrink with nothing else nearby is not a switch."""
    rows = _rows(80, 0, range(12), 2500, G1, area=600.0)
    rows[6]["area"] = 150.0
    rows += _rows(80, 0, range(12, 41), 2500, G1, x0=220, area=600.0)
    cells, _ = _walk(_cycling_cell(10, 500) + _cycling_cell(20, 1500) + rows)
    row = cells.set_index("start_track").loc[80]
    assert "switch" not in row.review_flags and row.outcome == "no_mitosis"


def test_unlinked_daughters_make_a_division() -> None:
    """A G2 track that ends beside two new half-size, geminin-low founders divided."""
    import json

    mother = _rows(90, 0, range(20), 3200, G2, area=800.0)
    d1 = _rows(91, 0, range(21, 41), 3185, G1, x0=300, area=380.0)
    d2 = _rows(92, 0, range(21, 41), 3215, G1, x0=300, area=380.0)
    cells, _ = _walk(
        _cycling_cell(10, 500) + _cycling_cell(20, 1500) + mother + d1 + d2
    )
    row = cells.set_index("start_track").loc[90]
    assert row.outcome == "divided" and row.outcome_frame == 21
    assert (
        "daughter_pair" in row.review_flags
        and "outcome_review" in row.review_flags
    )
    props = json.loads(row.proposals)
    assert {p["track"] for p in props if p["op"] == "set_parent"} == {91, 92}


def test_bridges_are_proposed_with_confidence() -> None:
    """A direct restart with one candidate is a clear bridge proposal; a
    skipped frame makes it uncertain."""
    import json

    restart = _rows(50, 0, range(15), 2500, G1) + _rows(
        51, 0, range(15, 41), 2500, G1, x0=250
    )
    cells, _ = _walk(
        _cycling_cell(10, 500) + _cycling_cell(20, 1500) + restart
    )
    (p,) = json.loads(cells.set_index("start_track").loc[50].proposals)
    assert p["op"] == "link" and p["track"] == 51 and p["frame"] == 15
    assert p["confidence"] == "clear"

    skipped = _rows(50, 0, range(15), 2500, G1) + _rows(
        51, 0, range(17, 41), 2500, G1, x0=270
    )
    cells, _ = _walk(
        _cycling_cell(10, 500) + _cycling_cell(20, 1500) + skipped
    )
    (p,) = json.loads(cells.set_index("start_track").loc[50].proposals)
    assert p["frame"] == 17 and p["confidence"] == "uncertain"


def test_jump_onto_a_pre_existing_neighbour_is_flagged_not_acted_on() -> None:
    """A matching neighbour that already existed before the step is only a
    suspect: the track is not cut and no relink is proposed."""
    import json

    good = _rows(70, 0, range(10), 2000, G1, area=600.0)
    wrong = _rows(
        70,
        0,
        range(10, 41),
        2000,
        {"pip": 50.0, "geminin": 100.0},
        x0=200,
        area=150.0,
    )
    neighbour = _rows(71, 0, range(0, 41), 2010, G1, x0=100, area=600.0)
    cells, states = _walk(
        _cycling_cell(10, 500)
        + _cycling_cell(20, 1500)
        + good
        + wrong
        + neighbour
    )
    row = cells.set_index("start_track").loc[70]
    assert (
        "jump_suspect" in row.review_flags
        and "switch" not in row.review_flags.split()
    )
    assert states[states.cell == row.cell].timepoint.max() == 40
    assert all(p["op"] != "link" for p in json.loads(row.proposals))
