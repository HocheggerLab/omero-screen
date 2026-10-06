"""Replayable curation: edit operations, the log, undo and validation."""

import json

import pandas as pd
import pytest
from cellview.tracks.edit import (
    Curated,
    EditError,
    EditLog,
    base_lineage,
    replay,
)
from cellview.tracks.repair import repair_lineage


def _curated(spec: dict[int, tuple[range, int]], parents: dict[int, int] | None = None) -> Curated:
    """Build a lineage from ``{label: (frames, track_id)}``."""
    tracks = {(t, label): tid for label, (frames, tid) in spec.items() for t in frames}
    tids = set(tracks.values())
    par = {tid: 0 for tid in tids} | (parents or {})
    return Curated(tracks=tracks, parents=par, next_id=max(tids) + 1)


def _frames(cur: Curated, tid: int) -> list[int]:
    return [t for t, _ in cur.detections(tid)]


def test_link_joins_a_founder_across_a_gap() -> None:
    """A lost cell continues as a new track that starts after a gap."""
    cur = _curated({1: (range(0, 10), 1), 5: (range(12, 20), 5)})
    cur.link((0, 1), 12, 5)
    assert _frames(cur, 1) == [*range(0, 10), *range(12, 20)]
    assert 5 not in cur.parents


def test_link_takes_only_the_tail_of_an_existing_track() -> None:
    """Linking into the middle of another track leaves its earlier part alone."""
    cur = _curated({1: (range(0, 10), 1), 5: (range(0, 20), 5)})
    cur.link((0, 1), 12, 5)
    assert _frames(cur, 1)[-1] == 19 and 12 in _frames(cur, 1)
    assert _frames(cur, 5) == list(range(0, 12))


def test_link_moves_the_cells_own_tail_to_a_new_track() -> None:
    """If the cell already had detections from that frame, they become a new cell."""
    cur = _curated({1: (range(0, 20), 1), 5: (range(12, 20), 5)})
    cur.link((0, 1), 12, 5)
    labels_of_1 = {label for t, label in cur.detections(1) if t >= 12}
    assert labels_of_1 == {5}
    new = cur.tracks[(15, 1)]
    assert new not in (1, 5) and cur.parents[new] == 0


def test_daughters_follow_the_linked_tail() -> None:
    """Daughters of the absorbed track become daughters of the cell."""
    cur = _curated(
        {1: (range(0, 10), 1), 5: (range(12, 20), 5), 6: (range(20, 25), 6)},
        parents={6: 5},
    )
    cur.link((0, 1), 12, 5)
    assert cur.parents[6] == 1


def test_unlink_makes_a_founder_and_refuses_an_empty_cut() -> None:
    """Unlink cuts the track; cutting past its end is an error."""
    cur = _curated({1: (range(0, 20), 1)})
    cur.unlink((0, 1), 10)
    tail = cur.tracks[(15, 1)]
    assert tail != 1 and cur.parents[tail] == 0 and _frames(cur, 1) == list(range(10))
    with pytest.raises(EditError):
        cur.unlink((0, 1), 50)


def test_set_parent_rejects_cycles_and_overlap() -> None:
    """A division must go forward in time and cannot loop."""
    cur = _curated({1: (range(0, 10), 1), 2: (range(10, 20), 2), 3: (range(5, 20), 3)})
    cur.set_parent((10, 2), (0, 1))
    assert cur.parents[2] == 1
    with pytest.raises(EditError, match="ancestor"):
        cur.set_parent((0, 1), (10, 2))
    with pytest.raises(EditError, match="end before"):
        cur.set_parent((5, 3), (0, 1))


def test_swap_exchanges_identities_from_a_frame() -> None:
    """After two cells cross, swap gives each its own tail back."""
    cur = _curated({1: (range(0, 20), 1), 2: (range(0, 20), 2)})
    cur.swap((0, 1), (0, 2), 10)
    assert cur.tracks[(15, 1)] == 2 and cur.tracks[(15, 2)] == 1
    assert cur.tracks[(5, 1)] == 1


def test_annotations_and_bad_event_kind() -> None:
    """Events are stored per cell; unknown kinds are refused."""
    cur = _curated({1: (range(0, 20), 1)})
    cur.annotate("event", (0, 1), {"kind": "death", "frame": 18})
    cur.annotate("set_outcome", (0, 1), {"outcome": "death"})
    assert cur.annotations[1]["events"] == [{"kind": "death", "frame": 18}]
    with pytest.raises(EditError):
        cur.annotate("event", (0, 1), {"kind": "explosion", "frame": 3})


def test_unknown_anchor_is_an_error() -> None:
    """Anchors must name an existing detection."""
    with pytest.raises(EditError, match="No nucleus"):
        _curated({1: (range(5), 1)}).resolve((3, 99))


def test_log_replay_is_deterministic_and_undo_reverts(tmp_path) -> None:
    """Replaying the same log twice gives the same lineage; undo appends a revert."""
    base = _curated({1: (range(0, 10), 1), 5: (range(12, 20), 5)})
    log = EditLog(tmp_path / "edits.jsonl")
    log.append("link", "C2", {"cell": [0, 1], "frame": 12, "label": 5}, author="agent",
               confirmed_by="human", reason="2.4 diameters, markers continuous", base=base)
    log.append("event", "C2", {"cell": [0, 1], "kind": "mitosis", "frame": 19}, base=base)
    a, b = replay(base, log, "C2"), replay(base, log, "C2")
    assert a.tracks == b.tracks and a.annotations == b.annotations
    assert a.tracks[(15, 5)] == 1

    undo = log.undo("C2", base=base)
    assert undo.op == "revert" and undo.args == {"target": "e0002"}
    after = replay(base, log, "C2")
    assert "events" not in after.annotations.get(1, {})
    assert after.tracks[(15, 5)] == 1  # the link survives
    assert len(log.entries()) == 3  # nothing deleted
    first = json.loads((tmp_path / "edits.jsonl").read_text().splitlines()[0])
    assert first["author"] == "agent" and first["confirmed_by"] == "human" and first["v"] == 1


def test_invalid_edit_is_not_written(tmp_path) -> None:
    """Validation against the base happens before anything reaches the log."""
    base = _curated({1: (range(0, 10), 1)})
    log = EditLog(tmp_path / "edits.jsonl")
    with pytest.raises(EditError):
        log.append("link", "C2", {"cell": [0, 1], "frame": 12, "label": 77}, base=base)
    assert log.entries() == []


def test_edits_for_other_wells_are_ignored(tmp_path) -> None:
    """One log per plate; replay applies only the well asked for."""
    base = _curated({1: (range(0, 10), 1), 5: (range(12, 20), 5)})
    log = EditLog(tmp_path / "edits.jsonl")
    log.append("link", "C3", {"cell": [0, 1], "frame": 12, "label": 5})
    assert replay(base, log, "C2").tracks[(15, 5)] == 5


def test_base_lineage_from_repair() -> None:
    """The repair's assignment becomes the starting lineage, keyed by raw label."""
    det = pd.DataFrame(
        [
            {"track_id_raw": 1, "parent_track_id_raw": 0, "timepoint": t, "area": 400.0, "y": 100.0, "x": 100.0}
            for t in range(10)
        ]
        + [
            {"track_id_raw": 2, "parent_track_id_raw": 0, "timepoint": t, "area": 400.0, "y": 100.0, "x": 103.0}
            for t in range(10, 20)
        ]
    )
    result = repair_lineage(det)
    cur = base_lineage(det, result)
    assert cur.tracks[(15, 2)] == 1  # restart joined by the repair
    assert cur.next_id == 3
    table = cur.label_table()
    assert set(table.columns) == {"timepoint", "label", "track_id", "parent_track_id"}


def test_absorb_and_drop_move_single_detections() -> None:
    """Absorb adds a piece to the cell over a frame range; drop removes it from any track."""
    cur = _curated({1: (range(0, 10), 1), 7: (range(3, 6), 7)})
    assert cur.absorb((0, 1), (3, 5), 7) == 3
    assert {cur.tracks[(t, 7)] for t in range(3, 6)} == {1}
    assert 7 not in cur.parents  # the emptied track is gone
    assert cur.drop((4, 4), 7) == 1
    assert (4, 7) not in cur.tracks and cur.tracks[(3, 7)] == 1
    with pytest.raises(EditError):
        cur.absorb((0, 1), (20, 25), 7)


def test_mask_add_replaces_raw_labels_and_joins_cell() -> None:
    """A reviewer mask replaces the raw labels it covers and belongs to the anchored cell."""
    from cellview.tracks.cells import apply_extras

    cur = _curated({1: (range(0, 10), 1), 7: (range(0, 10), 7)})
    cur.mask_add((0, 1), 5, 900, {"area": 120.0, "y": 1.0, "x": 2.0, "pip": 10.0, "patch": "masks/C2/t0005_L900.npz"}, [7])
    cur.mask_add(None, 5, 901, {"area": 80.0, "y": 3.0, "x": 4.0, "pip": 20.0}, [])
    assert cur.tracks[(5, 900)] == 1 and (5, 7) in cur.removed and (5, 7) not in cur.tracks
    assert cur.parents[cur.tracks[(5, 901)]] == 0  # second part is a new founder
    det = pd.DataFrame({"timepoint": [5, 5], "label": [1, 7], "track_id_raw": [1, 7], "parent_track_id_raw": [0, 0],
                        "area": [100.0, 100.0], "y": [0.0, 9.0], "x": [0.0, 9.0], "pip": [1.0, 1.0]})
    out = apply_extras(det, cur)
    assert set(out.label) == {1, 900, 901}
    assert out.set_index("label").loc[900, "pip"] == 10.0
    with pytest.raises(EditError):
        cur.mask_add((0, 1), 5, 900, {"area": 1.0}, [])
