"""Per-cell fate walker on synthetic PIP-FUCCI tracks."""

import pandas as pd
from omero_screen_track.fate import FateParams, follow_cells

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
    """A track that ends at the border, heading out, left the field.

    Censored, but not flagged lost or edge, so it is not queued.
    """
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
    """A raw track that jumps onto a small dim object is cut at the jump.

    The real nucleus carries on under a new id, which is proposed.
    """
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
    """A direct restart with one candidate is a clear bridge proposal.

    A skipped frame makes it uncertain.
    """
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
    """A matching neighbour that existed before the step is only a suspect.

    The track is not cut and no relink is proposed.
    """
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
    neighbour = _rows(71, 0, range(41), 2010, G1, x0=100, area=600.0)
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


def _switching_cell():
    """Track 70 jumps at frame 10 onto a small dark object.

    New track 71 carries the cell on (the setting of the switch test above).
    """
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
    return _cycling_cell(10, 500) + _cycling_cell(20, 1500) + good + wrong + real


def _walk_confirmed(rows, confirmed):
    return follow_cells(
        pd.DataFrame(rows),
        0,
        40,
        FateParams(),
        shape=(5000, 5000),
        confirmed=confirmed,
    )


def test_unconfirmed_switch_still_cuts() -> None:
    """A reviewer edit at another frame does not protect the switch."""
    cells, states = _walk_confirmed(_switching_cell(), {70: {30}})
    row = cells.set_index("start_track").loc[70]
    assert "switch" in row.review_flags.split()
    assert row.outcome == "lost" and row.outcome_frame == 10
    assert states[states.cell == row.cell].timepoint.max() == 9


def test_switch_at_a_reviewer_link_does_not_cut() -> None:
    """A switch where the reviewer fixed the cell's identity is theirs.

    Within one frame of it, the track is kept whole and nothing is flagged
    or proposed.
    """
    import json

    for frame in (9, 10, 11):
        cells, states = _walk_confirmed(_switching_cell(), {70: {frame}})
        row = cells.set_index("start_track").loc[70]
        flags = row.review_flags.split()
        assert "switch" not in flags and "jump_suspect" not in flags
        assert row.outcome != "lost" and row.last_frame == 40
        assert states[states.cell == row.cell].timepoint.max() == 40
        assert all(p["op"] != "link" for p in json.loads(row.proposals))


def test_switch_two_frames_from_a_reviewer_link_still_cuts() -> None:
    """The tolerance is one frame."""
    cells, _ = _walk_confirmed(_switching_cell(), {70: {12}})
    row = cells.set_index("start_track").loc[70]
    assert "switch" in row.review_flags.split() and row.outcome_frame == 10


# --- H2B-Tub walker (#87) ----------------------------------------------------


def _h2b_rows(tid, parent, frames, y, content, x0=100.0, dx=10.0, area=400.0,
              h2b_max=1000.0):
    """One track on an H2B plate; ``dna`` is set so that dna * area = content."""
    return [
        {
            "track_id": tid,
            "parent_track_id": parent,
            "timepoint": t,
            "area": area,
            "y": y,
            "x": x0 + dx * i,
            "dna": content / area,
            "h2b": 500.0,
            "h2b_max": h2b_max,
        }
        for i, t in enumerate(frames)
    ]


def _h2b_cycle(tid, y, condensed=True):
    """G1 (0-9) -> S (10-19) -> G2 (20-29) -> two half-content daughters from 30."""
    rows = (
        _h2b_rows(tid, 0, range(10), y, 1000.0)
        + _h2b_rows(tid, 0, range(10, 20), y, 1400.0, x0=200)
        + _h2b_rows(tid, 0, range(20, 30), y, 2000.0, x0=300)
    )
    if condensed:
        rows[-1]["h2b_max"] = 3000.0
    rows += _h2b_rows(tid + 1, tid, range(30, 41), y - 20, 1000.0, x0=400, area=200.0)
    rows += _h2b_rows(tid + 2, tid, range(30, 41), y + 20, 1000.0, x0=400, area=200.0)
    return rows


def _h2b_walk(rows, **kw):
    from omero_screen_track.fate import H2BTubWalker

    return follow_cells(
        pd.DataFrame(rows), 0, 40, FateParams(**kw), shape=(5000, 5000),
        walker=H2BTubWalker(),
    )


def _h2b_population(*extra):
    """Mostly G1 nuclei so the well's content peak sits at G1 (1000)."""
    rows = [r for k in range(6) for r in _h2b_rows(200 + k, 0, range(41), 3000 + 100 * k, 1000.0)]
    for e in extra:
        rows += e
    return rows


def test_h2b_cycle_divides_with_content_phases() -> None:
    """Content G1 -> S -> G2, then a condensed split: divided, no flags."""
    cells, states = _h2b_walk(_h2b_population(_h2b_cycle(10, 500)))
    cell = cells.set_index("start_track").loc[10]
    assert cell.outcome == "divided" and cell.outcome_frame == 30
    phases = states[states.cell == cell.cell].phase
    assert phases[phases.ne(phases.shift())].tolist() == ["G1", "S", "G2"]
    assert cell.review_flags == ""


def test_h2b_division_without_condensation_is_flagged() -> None:
    """A split the H2B maximum does not show is kept but flagged for review."""
    cells, _ = _h2b_walk(_h2b_population(_h2b_cycle(10, 500, condensed=False)))
    flags = cells.set_index("start_track").loc[10].review_flags
    assert "division_unconfirmed" in flags and "outcome_review" in flags


def test_h2b_content_fall_without_split_is_a_dna_jump() -> None:
    """DNA content never halves along one cell outside mitosis."""
    rows = _h2b_rows(5, 0, range(20), 500, 2000.0) + _h2b_rows(
        5, 0, range(20, 41), 500, 1000.0, x0=300
    )
    cells, _ = _h2b_walk(_h2b_population(rows))
    cell = cells.set_index("start_track").loc[5]
    assert "dna_jump" in cell.review_flags and "geminin_jump" not in cell.review_flags


def test_h2b_bridge_needs_matching_content() -> None:
    """A founder with twice the content is not the cell's continuation."""
    g1 = _h2b_rows(50, 0, range(15), 2500, 1000.0)
    wrong = _h2b_rows(51, 0, range(16, 41), 2500, 2000.0, x0=260)
    cells, _ = _h2b_walk(_h2b_population(g1 + wrong))
    assert cells.set_index("start_track").loc[50].segments == "50"
    right = _h2b_rows(52, 0, range(16, 41), 2500, 1050.0, x0=260)
    cells, _ = _h2b_walk(_h2b_population(g1 + right))
    assert cells.set_index("start_track").loc[50].segments == "50 52"


def test_h2b_unlinked_daughters_make_a_division() -> None:
    """A G2-content track that ends beside two half-content founders divided."""
    rows = _h2b_rows(5, 0, range(30), 500, 2000.0)
    rows += _h2b_rows(6, 0, range(31, 41), 480, 1000.0, x0=400, area=200.0)
    rows += _h2b_rows(7, 0, range(31, 41), 520, 1000.0, x0=400, area=200.0)
    cells, _ = _h2b_walk(_h2b_population(rows))
    cell = cells.set_index("start_track").loc[5]
    assert cell.outcome == "divided" and "daughter_pair" in cell.review_flags


def test_seeds_follow_cells_from_their_own_frame() -> None:
    """Seeded daughters are followed from birth; nobody else is."""
    from omero_screen_track.fate import H2BTubWalker

    rows = _h2b_population(_h2b_cycle(10, 500))
    cells, states = follow_cells(
        pd.DataFrame(rows), 0, 40, FateParams(), shape=(5000, 5000),
        walker=H2BTubWalker(), seeds=[(11, 30)],
    )
    assert cells.start_track.tolist() == [11]
    assert cells.start_frame.tolist() == [30]
    assert states.timepoint.min() == 30


def test_resolve_walker_by_channels() -> None:
    """FUCCI first, else H2B with tubulin, else none."""
    from omero_screen_track.fate import (
        FucciWalker,
        H2BTubWalker,
        resolve_walker,
    )

    assert isinstance(resolve_walker(["pip", "geminin", "H2B", "Tub"], 10), FucciWalker)
    walker = resolve_walker(["H2B_rfp", "tub_cell", "DAPI"], 10)
    assert isinstance(walker, H2BTubWalker) and walker.h2b == "h2b_rfp"
    assert resolve_walker(["H2B", "DAPI"], 10) is None
    assert resolve_walker(["DAPI", "Tub"], 10) is None


def _bright_births(rows):
    """Newborn daughters (from frame 30) start bright: condensed chromatin."""
    for r in rows:
        if r["parent_track_id"] and r["timepoint"] == 30:
            r["h2b"] = 900.0
    return rows


def test_birth_seeds_one_daughter_of_each_confirmed_division() -> None:
    """Only condensed divisions in the window seed; one daughter each."""
    from omero_screen_track.fate import H2BTubWalker, birth_seeds

    rows = _bright_births(
        _h2b_population(
            _h2b_cycle(10, 500),
            _h2b_cycle(20, 1500, condensed=False),
        )
    )
    seeds = birth_seeds(pd.DataFrame(rows), (25, 35), H2BTubWalker())
    assert len(seeds) == 1
    (tid, frame), = seeds
    assert tid in (11, 12) and frame == 30
    assert birth_seeds(pd.DataFrame(rows), (31, 40), H2BTubWalker()) == []
    both = birth_seeds(pd.DataFrame(rows), (25, 35), H2BTubWalker(), sisters=True)
    assert both == [(11, 30), (12, 30)]


def test_birth_seeds_skip_daughters_that_do_not_look_newborn() -> None:
    """A dim daughter, or one from a fragment-length mother, is not a birth."""
    from omero_screen_track.fate import FateParams, H2BTubWalker, birth_seeds

    dim = _h2b_population(_h2b_cycle(10, 500))
    assert birth_seeds(pd.DataFrame(dim), (25, 35), H2BTubWalker()) == []
    rows = _bright_births(_h2b_population(_h2b_cycle(10, 500)))
    strict = FateParams(min_mother_frames=31)
    assert birth_seeds(pd.DataFrame(rows), (25, 35), H2BTubWalker(), strict) == []


def test_birth_seeds_use_the_mothers_end_not_a_late_child() -> None:
    """A child track that starts long after the mother ended is no birth."""
    from omero_screen_track.fate import H2BTubWalker, birth_seeds

    rows = _bright_births(_h2b_population(_h2b_cycle(10, 500)))
    for r in rows:  # daughter 12 only appears at frame 36
        if r["track_id"] == 12 and r["timepoint"] < 36:
            r["track_id"] = -1
    rows = [r for r in rows if r["track_id"] != -1]
    seeds = birth_seeds(pd.DataFrame(rows), (25, 35), H2BTubWalker(), sisters=True)
    assert seeds == [(11, 30)]
    assert birth_seeds(pd.DataFrame(rows), (36, 40), H2BTubWalker(), sisters=True) == []
