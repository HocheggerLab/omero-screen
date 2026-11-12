"""Replayable curation: edit operations, the log, undo and validation."""

import json

import pandas as pd
import pytest
from omero_screen_track.edit import (
    Curated,
    Edit,
    EditError,
    EditLog,
    base_lineage,
    confirmed_frames,
    replay,
)
from omero_screen_track.repair import repair_lineage


def _curated(
    spec: dict[int, tuple[range, int]], parents: dict[int, int] | None = None
) -> Curated:
    """Build a lineage from ``{label: (frames, track_id)}``."""
    tracks = {
        (t, label): tid
        for label, (frames, tid) in spec.items()
        for t in frames
    }
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
    assert (
        tail != 1
        and cur.parents[tail] == 0
        and _frames(cur, 1) == list(range(10))
    )
    with pytest.raises(EditError):
        cur.unlink((0, 1), 50)


def test_set_parent_rejects_cycles_and_overlap() -> None:
    """A division must go forward in time and cannot loop."""
    cur = _curated(
        {1: (range(0, 10), 1), 2: (range(10, 20), 2), 3: (range(5, 20), 3)}
    )
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
    log.append(
        "link",
        "C2",
        {"cell": [0, 1], "frame": 12, "label": 5},
        author="agent",
        confirmed_by="human",
        reason="2.4 diameters, markers continuous",
        base=base,
    )
    log.append(
        "event",
        "C2",
        {"cell": [0, 1], "kind": "mitosis", "frame": 19},
        base=base,
    )
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
    assert (
        first["author"] == "agent"
        and first["confirmed_by"] == "human"
        and first["v"] == 1
    )


def test_invalid_edit_is_not_written(tmp_path) -> None:
    """Validation against the base happens before anything reaches the log."""
    base = _curated({1: (range(0, 10), 1)})
    log = EditLog(tmp_path / "edits.jsonl")
    with pytest.raises(EditError):
        log.append(
            "link", "C2", {"cell": [0, 1], "frame": 12, "label": 77}, base=base
        )
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
            {
                "track_id_raw": 1,
                "parent_track_id_raw": 0,
                "timepoint": t,
                "area": 400.0,
                "y": 100.0,
                "x": 100.0,
            }
            for t in range(10)
        ]
        + [
            {
                "track_id_raw": 2,
                "parent_track_id_raw": 0,
                "timepoint": t,
                "area": 400.0,
                "y": 100.0,
                "x": 103.0,
            }
            for t in range(10, 20)
        ]
    )
    result = repair_lineage(det)
    cur = base_lineage(det, result)
    assert cur.tracks[(15, 2)] == 1  # restart joined by the repair
    assert cur.next_id == 3
    table = cur.label_table()
    assert set(table.columns) == {
        "timepoint",
        "label",
        "track_id",
        "parent_track_id",
    }


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
    from omero_screen_track.cells import apply_extras

    cur = _curated({1: (range(0, 10), 1), 7: (range(0, 10), 7)})
    cur.mask_add(
        (0, 1),
        5,
        900,
        {
            "area": 120.0,
            "y": 1.0,
            "x": 2.0,
            "pip": 10.0,
            "patch": "masks/C2/t0005_L900.npz",
        },
        [7],
    )
    cur.mask_add(
        None, 5, 901, {"area": 80.0, "y": 3.0, "x": 4.0, "pip": 20.0}, []
    )
    assert (
        cur.tracks[(5, 900)] == 1
        and (5, 7) in cur.removed
        and (5, 7) not in cur.tracks
    )
    assert (
        cur.parents[cur.tracks[(5, 901)]] == 0
    )  # second part is a new founder
    det = pd.DataFrame(
        {
            "timepoint": [5, 5],
            "label": [1, 7],
            "track_id_raw": [1, 7],
            "parent_track_id_raw": [0, 0],
            "area": [100.0, 100.0],
            "y": [0.0, 9.0],
            "x": [0.0, 9.0],
            "pip": [1.0, 1.0],
        }
    )
    out = apply_extras(det, cur)
    assert set(out.label) == {1, 900, 901}
    assert out.set_index("label").loc[900, "pip"] == 10.0
    with pytest.raises(EditError):
        cur.mask_add((0, 1), 5, 900, {"area": 1.0}, [])


def test_extend_writes_all_or_nothing_and_batch_undo(tmp_path) -> None:
    """A batch is validated as a whole; one undo reverts every batch entry."""
    base = _curated({1: (range(0, 10), 1), 5: (range(0, 20), 5)})
    log = EditLog(tmp_path / "edits.jsonl")
    log.append("note", "C2", {"cell": [0, 1], "text": "before"}, base=base)
    with pytest.raises(EditError):
        log.extend(
            [
                ("unlink", "C2", {"cell": [0, 1], "frame": 5}, ""),
                ("link", "C2", {"cell": [0, 99], "frame": 12, "label": 5}, ""),
            ],
            base=base,
        )
    assert len(log.entries()) == 1
    done = log.extend(
        [
            ("unlink", "C2", {"cell": [0, 1], "frame": 5, "batch": "b1"}, "x"),
            ("unlink", "C2", {"cell": [0, 1], "frame": 3, "batch": "b1"}, "x"),
        ],
        base=base,
    )
    assert [e.id for e in done] == ["e0002", "e0003"]
    assert _frames(replay(base, log, "C2"), 1) == [0, 1, 2]
    log.undo("C2", base=base)
    assert _frames(replay(base, log, "C2"), 1) == list(range(0, 10))
    log.undo("C2", base=base)  # the note, written before the batch
    assert not replay(base, log, "C2").annotations.get(1, {}).get("notes")


def test_confirmed_frames_are_live_links_and_swaps_on_the_replayed_lineage() -> (
    None
):
    """Links (the anchored cell) and swaps (both cells) of the well, human or
    agent, give their frame; reverted edits, other ops and wells do not.
    """
    base = _curated(
        {
            1: (range(10), 1),
            5: (range(10, 30), 5),
            2: (range(30), 2),
            3: (range(30), 3),
        }
    )
    edits = [
        Edit(
            "e0001",
            "link",
            "C2",
            {"cell": [0, 1], "frame": 10, "label": 5},
            author="agent",
        ),
        Edit(
            "e0002",
            "swap",
            "C2",
            {"cell": [0, 2], "other": [0, 3], "frame": 20},
        ),
        Edit("e0003", "link", "C2", {"cell": [0, 2], "frame": 25, "label": 5}),
        Edit("e0004", "revert", "C2", {"target": "e0003"}),
        Edit("e0005", "unlink", "C2", {"cell": [0, 3], "frame": 28}),
        Edit("e0006", "link", "C3", {"cell": [0, 1], "frame": 12, "label": 5}),
    ]
    cur = replay(base, edits, "C2")
    assert confirmed_frames(cur, edits, "C2") == {1: {10}, 2: {20}, 3: {20}}


def _merged_neighbour() -> Curated:
    """Cell (track 1, label 1) t0–9; neighbour (track 2, label 2) t0–9. At t5 the
    neighbour's mask 2 also covers the cell, which has no own mask there.
    """
    base = _curated({1: (range(10), 1), 2: (range(10), 2)})
    del base.tracks[(5, 1)]
    return base


def test_split_old_form_replays_with_the_other_part_orphaned() -> None:
    """Old logs: part 1 replaces, part 2 (cell None, replaces []) is a new track;
    the neighbour's track loses t5. Replayed exactly as before.
    """
    base = _merged_neighbour()
    edits = [
        Edit(
            "e0001",
            "mask_add",
            "C2",
            {
                "cell": [0, 1],
                "frame": 5,
                "label": 900,
                "measurements": {"area": 1.0},
                "replaces": [2],
            },
        ),
        Edit(
            "e0002",
            "mask_add",
            "C2",
            {
                "cell": None,
                "frame": 5,
                "label": 901,
                "measurements": {"area": 1.0},
                "replaces": [],
            },
        ),
    ]
    cur = replay(base, edits, "C2")
    assert cur.tracks[(5, 900)] == 1
    orphan = cur.tracks[(5, 901)]
    assert orphan not in (1, 2) and cur.parents[orphan] == 0
    assert 5 not in _frames(cur, 2)


def test_split_new_form_keeps_the_other_part_in_its_owner() -> None:
    """New logs: part 2 first, anchored on the cut nucleus, joins the neighbour;
    part 1 then replaces it in the cell. The neighbour's track is unbroken.
    """
    base = _merged_neighbour()
    edits = [
        Edit(
            "e0001",
            "mask_add",
            "C2",
            {
                "cell": [5, 2],
                "frame": 5,
                "label": 901,
                "measurements": {"area": 1.0},
                "replaces": [],
                "batch": "s",
            },
        ),
        Edit(
            "e0002",
            "mask_add",
            "C2",
            {
                "cell": [0, 1],
                "frame": 5,
                "label": 900,
                "measurements": {"area": 1.0},
                "replaces": [2],
                "batch": "s",
            },
        ),
    ]
    cur = replay(base, edits, "C2")
    assert cur.tracks[(5, 900)] == 1 and cur.tracks[(5, 901)] == 2
    assert _frames(cur, 2) == list(range(10)) and _frames(cur, 1) == list(
        range(10)
    )
    assert (5, 2) in cur.removed and (5, 2) not in cur.tracks


def test_cell_scoped_undo_skips_other_cells_and_reverts_the_batch(
    tmp_path,
) -> None:
    """Undo with ``concerns_cell`` reverts the cell's latest edit (and its whole
    batch), never another cell's; with nothing left it refuses, no fallback.
    """
    from omero_screen_track.edit import concerns_cell

    base = _merged_neighbour()
    base.tracks.update({(t, 3): 3 for t in range(10)})
    base.parents[3] = 0
    base.next_id = 4
    log = EditLog(tmp_path / "edits.jsonl")
    log.extend(
        [
            (
                "mask_add",
                "C2",
                {
                    "cell": [5, 2],
                    "frame": 5,
                    "label": 901,
                    "measurements": {"area": 1.0},
                    "replaces": [],
                    "batch": "s",
                },
                "",
            ),
            (
                "mask_add",
                "C2",
                {
                    "cell": [0, 1],
                    "frame": 5,
                    "label": 900,
                    "measurements": {"area": 1.0},
                    "replaces": [2],
                    "batch": "s",
                },
                "",
            ),
        ],
        base=base,
    )
    log.append("note", "C2", {"cell": [0, 3], "text": "other cell"}, base=base)
    log.append("unlink", "C2", {"cell": [0, 3], "frame": 7}, base=base)

    def undo_cell(anchor):
        return log.undo(
            "C2",
            base=base,
            concerns=concerns_cell(replay(base, log, "C2"), anchor),
        )

    # The cell (anchor at t2, a frame other than its edits' anchor) reaches its split.
    undo_cell((2, 1))
    cur = replay(base, log, "C2")
    assert (
        not cur.extras and _frames(cur, 3)[-1] == 6
    )  # cell 3's unlink untouched
    with pytest.raises(EditError, match="Nothing to undo for this cell"):
        undo_cell((2, 1))
    undo_cell((0, 3))  # the unlink, then the note
    undo_cell((0, 3))
    assert not replay(base, log, "C2").annotations.get(3, {}).get("notes")
    # Without a predicate the whole well is still the scope.
    with pytest.raises(EditError, match="well C2"):
        log.undo("C2", base=base)


def test_concerns_cell_matches_anchors_that_resolve_to_the_cell() -> None:
    """Anchors equal to the cell's or resolving to its track concern it."""
    from omero_screen_track.edit import concerns_cell

    cur = _curated({1: (range(10), 1), 2: (range(10), 2)})
    concerns = concerns_cell(cur, (0, 1))
    assert concerns(Edit("e1", "note", "C2", {"cell": [7, 1], "text": ""}))
    assert concerns(
        Edit("e2", "swap", "C2", {"cell": [0, 2], "other": [3, 1], "frame": 4})
    )
    assert not concerns(Edit("e3", "note", "C2", {"cell": [0, 2], "text": ""}))
    assert not concerns(
        Edit("e4", "mask_add", "C2", {"cell": None, "frame": 1, "label": 5})
    )
    assert concerns(Edit("e5", "note", "C2", {"cell": [0, 1], "text": ""}))


def _mask(cell, frame, label, replaces, batch):
    return {
        "cell": cell,
        "frame": frame,
        "label": label,
        "measurements": {"area": 1.0},
        "replaces": replaces,
        "batch": batch,
    }


def test_split_part_cut_again_leaves_extras_and_undo_restores_it(
    tmp_path,
) -> None:
    """A part of a split (an extra) can be split again: the second split replaces
    the extra, which leaves ``extras`` (not ``removed``); anchors naming it still
    resolve; undoing the second split brings the first split's parts back.
    """
    from omero_screen_track.edit import concerns_cell

    base = _merged_neighbour()
    log = EditLog(tmp_path / "edits.jsonl")
    first = [
        ("mask_add", "C2", _mask([5, 2], 5, 901, [], "s1"), ""),
        ("mask_add", "C2", _mask([0, 1], 5, 900, [2], "s1"), ""),
    ]
    log.extend(first, base=base)
    # The cell's part 900 was still two nuclei: cut it, the rest a new track.
    second = [
        ("mask_add", "C2", _mask([0, 1], 5, 902, [900], "s2"), ""),
        ("mask_add", "C2", _mask(None, 5, 903, [], "s2"), ""),
    ]
    log.extend(second, base=base)
    cur = replay(base, log, "C2")
    assert set(cur.extras) == {(5, 901), (5, 902), (5, 903)}
    assert (5, 900) not in cur.removed and (5, 900) not in cur.tracks
    assert cur.tracks[(5, 902)] == 1 and cur.tracks[(5, 901)] == 2
    assert cur.tracks[(5, 903)] not in (1, 2)
    assert cur.resolve((5, 900)) == 1  # aliased to the new part
    assert _frames(cur, 1) == list(range(10))

    # Cutting the neighbour's part: its other part stays in the neighbour.
    third = [
        ("mask_add", "C2", _mask([5, 901], 5, 905, [], "s3"), ""),
        ("mask_add", "C2", _mask([0, 1], 5, 904, [901], "s3"), ""),
    ]
    log.extend(third, base=base)
    cur = replay(base, log, "C2")
    assert (5, 901) not in cur.extras
    assert cur.tracks[(5, 905)] == 2 and cur.tracks[(5, 904)] == 1
    assert _frames(cur, 2) == list(range(10))

    for _ in range(2):  # the third split, then the second
        log.undo(
            "C2",
            base=base,
            concerns=concerns_cell(replay(base, log, "C2"), (0, 1)),
        )
    cur = replay(base, log, "C2")
    assert set(cur.extras) == {(5, 900), (5, 901)}
    assert cur.tracks[(5, 900)] == 1 and cur.tracks[(5, 901)] == 2


def test_swap_until_exchanges_only_a_crossing() -> None:
    """A swap with until gives the cells their own identities back after it."""
    cur = _curated({1: (range(0, 20), 1), 2: (range(0, 20), 2)}, parents={})
    kid = cur._new_track()
    cur.tracks[(25, 3)] = kid
    cur.parents[kid] = 1
    cur.swap((0, 1), (0, 2), 10, until=12)
    assert [cur.tracks[(t, 1)] for t in (9, 10, 12, 13)] == [1, 2, 2, 1]
    assert [cur.tracks[(t, 2)] for t in (9, 10, 12, 13)] == [2, 1, 1, 2]
    assert cur.parents[kid] == 1  # daughters stay
    with pytest.raises(EditError, match="until"):
        cur.swap((0, 1), (0, 2), 10, until=9)
    with pytest.raises(EditError, match="Neither"):
        cur.swap((0, 1), (0, 2), 30, until=31)


def test_ranged_swap_confirms_both_ends() -> None:
    base = _curated({2: (range(30), 2), 3: (range(30), 3)})
    edits = [
        Edit(
            "e0001",
            "swap",
            "C2",
            {"cell": [0, 2], "other": [0, 3], "frame": 10, "until": 12},
        )
    ]
    cur = replay(base, edits, "C2")
    assert confirmed_frames(cur, edits, "C2") == {2: {10, 13}, 3: {10, 13}}


def test_set_parent_split_cuts_a_mother_that_runs_on() -> None:
    """The tracker carried the mother (track 1) into one daughter at t10; the
    other daughter (track 2) is born at t10."""
    cur = _curated({1: (range(0, 20), 1), 2: (range(10, 20), 2)})
    grandkid = cur._new_track()
    cur.tracks[(25, 7)] = grandkid
    cur.parents[grandkid] = 1
    with pytest.raises(EditError, match="split"):
        cur.set_parent((10, 2), (0, 1))
    cur.set_parent((10, 2), (0, 1), split=True)
    sister = cur.resolve((10, 1))
    assert sister not in (1, 2)
    assert _frames(cur, 1) == list(range(10))
    assert cur.parents[2] == 1 and cur.parents[sister] == 1
    assert cur.parents[grandkid] == sister  # the tail keeps its daughters
    with pytest.raises(EditError, match="begin"):
        cur.set_parent((10, 2), (10, 1), split=True)


def test_set_parent_split_via_the_log() -> None:
    base = _curated({1: (range(0, 20), 1), 2: (range(10, 20), 2)})
    edits = [
        Edit(
            "e0001",
            "set_parent",
            "C2",
            {"cell": [10, 2], "parent": [0, 1], "split": True},
        )
    ]
    cur = replay(base, edits, "C2")
    assert cur.parents[cur.resolve((10, 2))] == 1
    assert cur.parents[cur.resolve((15, 1))] == 1
