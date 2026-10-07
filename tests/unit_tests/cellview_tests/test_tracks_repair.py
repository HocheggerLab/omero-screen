"""Lineage repair on synthetic Trackastra output, one test per rule."""

import pandas as pd
import pytest
from cellview.tracks.repair import RepairParams, apply_repair, repair_lineage


def _track(tid, parent, frames, y, x, area=400.0, marker=1000.0, dx=0.0):
    """Rows for one raw track moving ``dx`` px per frame along x."""
    return [
        {
            "track_id_raw": tid,
            "parent_track_id_raw": parent,
            "timepoint": t,
            "area": area,
            "y": y,
            "x": x + dx * i,
            "marker": marker,
        }
        for i, t in enumerate(frames)
    ]


def _det(*tracks) -> pd.DataFrame:
    return pd.DataFrame([row for tr in tracks for row in tr])


def _divisions(result) -> int:
    return len({p for p in result.parents.values() if p})


def test_geminin_drop_is_a_real_mitosis() -> None:
    """Daughters losing geminin keep their division."""
    det = _det(
        _track(1, 0, range(10), 100, 100),
        _track(2, 1, range(10, 20), 100, 70, area=240, marker=100),
        _track(3, 1, range(10, 20), 100, 130, area=240, marker=100),
    )
    r = repair_lineage(det)
    assert r.parents[2] == 1 and r.parents[3] == 1
    assert set(r.events.rule) == {"mitosis"}


def test_touching_pieces_without_geminin_drop_are_merged() -> None:
    """Two touching pieces that conserve area are one nucleus."""
    det = _det(
        _track(1, 0, range(10), 100, 100),
        _track(2, 1, range(10, 20), 100, 95, area=200),
        _track(3, 1, range(10, 12), 100, 105, area=200),
    )
    r = repair_lineage(det)
    assert set(r.assignment.values()) == {1}
    assert _divisions(r) == 0
    tracks = apply_repair(det, r, ("marker",))
    assert tracks.loc[tracks.timepoint == 10, "area"].item() == pytest.approx(
        400
    )
    assert tracks.loc[tracks.timepoint == 10, "n_pieces"].item() == 2


def test_mislinked_neighbour_becomes_a_founder() -> None:
    """A far daughter without a geminin drop is a different cell."""
    det = _det(
        _track(1, 0, range(10), 100, 100),
        _track(2, 1, range(10, 20), 100, 102),
        _track(3, 1, range(10, 20), 100, 200),
    )
    r = repair_lineage(det)
    assert r.assignment[2] == 1
    assert r.assignment[3] == 3 and r.parents[3] == 0
    assert "neighbour" in set(r.events.rule)


def test_restart_next_to_an_ended_track_is_joined() -> None:
    """A founder starting where a childless track just ended continues it."""
    det = _det(
        _track(1, 0, range(10), 100, 100),
        _track(2, 0, range(10, 20), 100, 103),
    )
    r = repair_lineage(det)
    assert r.assignment[2] == 1
    assert set(r.events.rule) == {"restart"}


def test_second_mitosis_inside_refractory_period_is_rejected() -> None:
    """One mitosis called twice within hours keeps only the first division."""
    det = _det(
        _track(1, 0, range(10), 100, 100),
        _track(2, 1, range(10, 13), 100, 70, area=240, marker=100),
        _track(3, 1, range(10, 30), 100, 130, area=240, marker=100),
        _track(4, 2, range(13, 30), 100, 68, area=240, marker=10),
        _track(5, 2, range(13, 30), 100, 20, area=240, marker=10),
    )
    r = repair_lineage(det, RepairParams(refractory_hours=8))
    assert _divisions(r) == 1
    refractory = r.events[r.events.rule != "mitosis"]
    assert refractory.refractory.any()


def test_without_marker_only_fragments_are_merged() -> None:
    """With no mitotic marker, geometry alone cannot reject a division."""
    det = _det(
        _track(1, 0, range(10), 100, 100),
        _track(2, 1, range(10, 20), 100, 70, area=240),
        _track(3, 1, range(10, 20), 100, 130, area=240),
    ).drop(columns="marker")
    r = repair_lineage(det)
    assert _divisions(r) == 1
    assert set(r.events.rule) == {"kept"}


def test_missing_column_raises() -> None:
    """The detections table must carry every required column."""
    with pytest.raises(ValueError, match="area"):
        repair_lineage(_det(_track(1, 0, range(3), 1, 1)).drop(columns="area"))


def _rows(tid, parent, spec, marker=1000.0):
    """Rows from ``{frame: (area, y, x)}``."""
    return [
        {
            "track_id_raw": tid,
            "parent_track_id_raw": parent,
            "timepoint": t,
            "area": a,
            "y": y,
            "x": x,
            "marker": marker,
        }
        for t, (a, y, x) in spec.items()
    ]


def test_short_piece_of_a_nucleus_is_folded_back() -> None:
    """A nucleus cut in two for two frames: the piece's own track is merged."""
    host = {t: (400.0, 100.0, 100.0) for t in range(20)}
    host.update({10: (220.0, 100.0, 95.0), 11: (220.0, 100.0, 95.0)})
    det = _det(
        _rows(1, 0, host),
        _rows(2, 0, {10: (180.0, 100.0, 110.0), 11: (180.0, 100.0, 110.0)}),
    )
    r = repair_lineage(det)
    assert r.assignment[2] == 1
    assert "piece" in set(r.events.rule)
    tracks = apply_repair(det, r)
    assert tracks.loc[tracks.timepoint == 10, "area"].item() == pytest.approx(
        400
    )


def test_short_track_away_from_any_nucleus_is_kept() -> None:
    """A short track that touches nothing is not a piece."""
    det = _det(
        _track(1, 0, range(20), 100, 100),
        _track(2, 0, range(10, 12), 100, 300),
    )
    r = repair_lineage(det)
    assert r.assignment[2] == 2


def test_metaphase_half_with_the_daughters_returns_them_to_the_mother() -> (
    None
):
    """The mother is cut in two at metaphase and Trackastra hands the
    daughters to the half that started a new track; the division is restored
    on the mother and judged by geminin."""
    mother = {t: (400.0, 100.0, 100.0) for t in range(9)}
    mother[9] = (150.0, 100.0, 95.0)
    det = _det(
        _rows(1, 0, mother),
        _rows(2, 0, {9: (150.0, 100.0, 107.0)}),
        _track(3, 2, range(10, 20), 100, 80, area=240, marker=100),
        _track(4, 2, range(10, 20), 100, 120, area=240, marker=100),
    )
    r = repair_lineage(det)
    assert r.assignment[2] == 1
    assert r.parents[3] == 1 and r.parents[4] == 1
    assert {"piece", "mitosis"} <= set(r.events.rule)


def test_piece_of_a_growing_daughter_is_judged_on_the_area_after() -> None:
    """After mitosis a daughter nucleus grows fast: the area before the cut is
    too small, the area after it matches."""
    host = {t: (200.0, 100.0, 100.0) for t in range(10, 12)}
    host.update({12: (270.0, 100.0, 95.0), 13: (520.0, 100.0, 100.0)})
    host.update({t: (520.0, 100.0, 100.0) for t in range(14, 20)})
    det = _det(
        _rows(1, 0, host),
        _rows(2, 0, {12: (160.0, 100.0, 110.0)}),
    )
    r = repair_lineage(det)
    assert r.assignment[2] == 1
