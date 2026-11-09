"""Track view model: lineage table, layout, filters and colours on a toy well."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omero_screen_track.edit import Curated, Edit
from omero_screen_napari.review import track_view_model as m


def _toy() -> tuple[pd.DataFrame, Curated]:
    """A small well.

    * track 1 (label 5) t0–4 divides into 2 (label 6, t5–8) and
      3 (label 7, t5–9 with no mask at t7);
    * track 4 (label 9) t0–9: G1, S at t4–6, G2 from t7; at t3 a reviewer
      mask (label 900) replaced label 9;
    * track 10 (label 11) t2–3, a fragment.
    """
    rows = []
    for t in range(5):
        rows.append((t, 5, 100.0, 10.0, 1.0))
    for t in range(5, 9):
        rows.append((t, 6, 100.0, 10.0, 1.0))
    for t in (5, 6, 8, 9):
        rows.append((t, 7, 100.0, 10.0, 1.0))
    for t in range(10):
        pip = 1.0 if t in (4, 5, 6) else 10.0
        gem = 20.0 if t >= 7 else 1.0
        rows.append((t, 900 if t == 3 else 9, 120.0, pip, gem))
    for t in (2, 3):
        rows.append((t, 11, 30.0, 10.0, 1.0))
    det = pd.DataFrame(
        rows, columns=["timepoint", "label", "area", "pip", "geminin"]
    )
    det["y"], det["x"] = 5.0, 5.0
    tracks = {
        (int(t), int(lab)): {5: 1, 6: 2, 7: 3, 9: 4, 900: 4, 11: 10}[lab]
        for t, lab in zip(det["timepoint"], det["label"], strict=True)
    }
    curated = Curated(
        tracks=tracks,
        parents={1: 0, 2: 1, 3: 1, 4: 0, 10: 0},
        extras={(3, 900): {"area": 120.0}},
        removed={(3, 9)},
        aliases={(3, 9): (3, 900)},
        next_id=901,
    )
    return det, curated


def _edits() -> list[Edit]:
    return [
        Edit("e0001", "link", "C2", {"cell": [0, 9], "frame": 3, "label": 900}),
        Edit("e0002", "note", "C2", {"cell": [0, 9], "text": "x"}),
        Edit("e0003", "link", "C2", {"cell": [0, 5], "frame": 2, "label": 5}),
        Edit("e0004", "revert", "C2", {"target": "e0003"}),
        Edit("e0005", "link", "C3", {"cell": [0, 5], "frame": 1, "label": 5}),
    ]


def _records() -> dict[str, m.ReviewRecord]:
    return {
        "C2-t0-L5": m.ReviewRecord(
            verdict="accept",
            outcome="divided",
            reviewed_by="agent+human",
            author="human",
            time="2026-10-01T10:00:00",
        ),
        "C2-t0-L9": m.ReviewRecord(
            verdict="reject",
            reviewed_by="agent",
            author="agent",
            time="2026-10-01T10:00:00",
        ),
    }


@pytest.fixture
def table() -> m.LineageTable:
    """The toy well with verdicts, queue items and edits."""
    det, curated = _toy()
    records = _records()
    queue = {
        "C2-t0-L5": {"reason": "lost", "outcome": "lost"},
        "C2-t2-L11": {"reason": "", "outcome": "lost"},
    }
    return m.build_lineage_table(
        det,
        curated,
        well="C2",
        edits=_edits(),
        records=records,
        queue=queue,
        window=(1, 9),
    )


# -- table -------------------------------------------------------------------------


def test_nodes_one_per_track_and_frame(table) -> None:
    """Every detection is a node; the reviewer mask is one, flagged as such."""
    nodes = table.nodes.set_index(["track", "frame"])
    assert len(nodes) == 5 + 4 + 4 + 10 + 2
    assert nodes.loc[(4, 3), "is_extra"]
    assert nodes.loc[(4, 3), "label"] == 900
    assert not nodes.loc[(4, 2), "is_extra"]


def test_gap_and_division_edges(table, monkeypatch) -> None:
    """A missing frame is a gap edge; daughters hang off the mother's last frame."""
    e = table.edges
    gap = e[e["kind"] == "gap"]
    assert gap[["track", "f0", "f1"]].values.tolist() == [[3, 6, 8]]
    # The reviewer says divided, but the daughters (4 frames) are too short
    # to be drawn as branches.
    assert set(e.loc[e["track"].isin([2, 3]) & (e["f0"] == 4), "kind"]) == {
        "split"
    }
    monkeypatch.setattr(m, "MIN_DAUGHTER_FRAMES", 4)
    det, curated = _toy()
    e = m.build_lineage_table(
        det, curated, well="C2", records=_records()
    ).edges
    div = e[e["kind"] == "division"].sort_values("track")
    assert div[["track", "source", "f0", "f1"]].values.tolist() == [
        [2, 1, 4, 5],
        [3, 1, 4, 5],
    ]
    assert table.nodes.set_index(["track", "frame"]).loc[(3, 8), "after_gap"]


def test_track_table(table) -> None:
    """Span, gaps, parent and review fields per track."""
    tr = table.tracks
    assert tr.loc[3, ["start", "end", "n_frames", "n_gaps"]].tolist() == [
        5,
        9,
        4,
        1,
    ]
    assert tr.loc[2, "parent"] == 1
    assert tr.loc[1, "verdict"] == "accept"
    assert tr.loc[1, "outcome"] == "divided"  # the reviewer's
    assert tr.loc[1, "item_id"] == "C2-t0-L5"
    assert tr.loc[1, "queued"] and tr.loc[1, "flags"] == "lost"
    assert tr.loc[4, "excluded"]
    assert tr.loc[10, "outcome"] == "lost"  # the queue's automatic call
    assert tr.loc[10, "queued"] and tr.loc[10, "flags"] == ""
    assert tr.loc[2, "outcome"] == ""


def test_edits_flag_frames_and_density(table) -> None:
    """Live edits of the well count; reverted ones and other wells do not."""
    nodes = table.nodes.set_index(["track", "frame"])
    assert nodes.loc[(4, 3), "edited"]
    assert nodes.loc[(4, 3), "confirmed"]
    assert not nodes.loc[(1, 2), "edited"]
    tr = table.tracks
    assert tr.loc[4, "edits"] == 2  # link + note
    assert tr.loc[1, "edits"] == 0
    assert tr.loc[4, "edits_per_h"] == pytest.approx(2 / (10 * 20 / 60))


def test_events(table) -> None:
    """Window start, division (mother's last frame), S and G2 entry."""
    tr = table.tracks
    assert tr.loc[1, "ev_window"] == 1
    assert np.isnan(tr.loc[2, "ev_window"])  # starts after the window start
    assert tr.loc[1, "ev_division"] == 4
    assert np.isnan(tr.loc[4, "ev_division"])
    assert tr.loc[4, "ev_s"] == 4
    assert tr.loc[4, "ev_g2"] == 7
    assert np.isnan(tr.loc[1, "ev_s"])


def test_walker_phases_match_fate_walker() -> None:
    """The vectorised call is the fate walker's call_phases, track by track."""
    from omero_screen_track.fate import FateParams, Thresholds, call_phases

    det, _ = _toy()
    nodes = det[det["label"].isin([9, 900])].rename(
        columns={"timepoint": "frame"}
    )
    nodes = nodes.assign(track=4).reset_index(drop=True)
    pip_high, gem_low = 8.0, 2.0
    ours = m.walker_phases(nodes, pip_high, gem_low)
    theirs = call_phases(
        nodes.set_index("frame"),
        Thresholds(pip_high, gem_low, 0.0, 100.0),
        FateParams(),
    )
    assert ours.tolist() == theirs.tolist()


def test_slow_annotations_replace_phases_and_outcomes(table) -> None:
    """HMM phases and walker outcomes, when given, take over."""
    det, curated = _toy()
    slow = m.SlowAnnotations(
        fates=pd.DataFrame(
            {
                "segments": ["2"],
                "outcome": ["death"],
                "outcome_frame": [8.0],
                "excluded": [False],
                "review_flags": ["edge"],
            }
        ),
        phases=pd.DataFrame(
            {"track": [4, 4], "frame": [0, 1], "phase": ["E", "X"]}
        ),
        boundaries=pd.DataFrame(
            {"track": [4], "s_entry": [0.0], "g2_entry": [8.0], "exit": [0]}
        ),
    )
    t2 = m.build_lineage_table(det, curated, well="C2", slow=slow)
    assert t2.has_hmm
    nodes = t2.nodes.set_index(["track", "frame"])
    assert nodes.loc[(4, 0), "phase"] == "E"
    assert nodes.loc[(4, 5), "phase"] == "S"  # not called by the HMM: walker
    assert nodes.loc[(4, 5), "phase_walker"] == "S"
    assert t2.tracks.loc[2, "outcome"] == "death"
    assert t2.tracks.loc[2, "flags"] == "edge"
    assert t2.tracks.loc[4, "ev_exit"] == 0
    assert t2.tracks.loc[4, "ev_g2"] == 8
    assert np.isnan(t2.tracks.loc[4, "ev_s"])  # at the first frame: before


# -- layout ------------------------------------------------------------------------


def test_layout_mother_centred_over_daughters(table) -> None:
    """Daughters take adjacent columns; the mother sits between them."""
    lay = m.layout(table)
    x = lay.tracks["x"]
    assert abs(x[2] - x[3]) == 1
    assert x[1] == pytest.approx((x[2] + x[3]) / 2)
    assert len({x[1], x[4], x[10]}) == 3
    n = lay.nodes.set_index(["track", "frame"])
    assert n.loc[(3, 8), "y"] == 8  # absolute frame
    div = lay.edges[lay.edges["kind"].isin(["division", "split"])]
    assert set(div["x0"]) == {x[1]}
    assert set(div["x1"]) == {x[2], x[3]}


def test_align_division_puts_divisions_on_one_row(table) -> None:
    """Division alignment: the mother's last node at y = 0; others parked."""
    lay = m.layout(table, align="division")
    n = lay.nodes.set_index(["track", "frame"])
    assert n.loc[(1, 4), "y"] == 0
    assert n.loc[(2, 5), "y"] == 1  # the daughters move with the tree
    tr = lay.tracks
    assert not tr.loc[1, "parked"] and tr.loc[4, "parked"]
    assert lay.parked_from is not None
    assert tr.loc[4, "x"] >= lay.parked_from > tr.loc[[1, 2, 3], "x"].max()
    hidden = m.layout(table, align="division", park=False)
    assert set(hidden.tracks.index) == {1, 2, 3}


@pytest.mark.parametrize(
    ("align", "track", "frame"),
    [("window", 4, 1), ("s", 4, 4), ("g2", 4, 7), ("frame", 4, 0)],
)
def test_each_alignment_anchors_its_event(table, align, track, frame) -> None:
    """The event frame of an aligned track lands on y = 0."""
    n = m.layout(table, align=align).nodes.set_index(["track", "frame"])
    assert n.loc[(track, frame), "y"] == 0


def test_filters(table) -> None:
    """Accepted keeps the reviewed tree; min length drops fragments."""
    acc = m.layout(table, filters=m.Filters(verdict="accepted"))
    assert set(acc.tracks.index) == {1, 2, 3}
    alone = m.layout(
        table,
        filters=m.Filters(verdict="accepted", with_descendants=False),
    )
    assert set(alone.tracks.index) == {1}
    rej = m.layout(table, filters=m.Filters(verdict="reject"))
    assert set(rej.tracks.index) == {4}
    longer = m.layout(table, filters=m.Filters(min_length=3))
    assert 10 not in longer.tracks.index
    both = m.layout(table, filters=m.Filters(reviewed_by="both"))
    assert set(both.tracks.index) == {1, 2, 3}
    agent = m.layout(table, filters=m.Filters(reviewed_by="agent"))
    assert set(agent.tracks.index) == {4}
    flagged = m.layout(table, filters=m.Filters(flagged_only=True))
    assert set(flagged.tracks.index) == {1, 2, 3}
    none = m.layout(table, filters=m.Filters(verdict="none"))
    assert 1 not in none.tracks.index and 10 in none.tracks.index
    audited = m.layout(table, filters=m.Filters(audited_only=True))
    assert audited.tracks.empty


def test_daughter_kept_without_mother_becomes_a_root(table) -> None:
    """A filter that drops the mother leaves the daughter as its own tree."""
    lay = m.layout(table, filters=m.Filters(min_length=4))
    assert 2 in lay.tracks.index and 1 in lay.tracks.index
    lay = m.layout(
        table.__class__(
            nodes=table.nodes,
            edges=table.edges,
            tracks=table.tracks.assign(
                n_frames=table.tracks["n_frames"].where(
                    table.tracks.index != 1, 1
                )
            ),
        ),
        filters=m.Filters(min_length=2),
    )
    assert 1 not in lay.tracks.index
    assert (lay.edges["kind"] != "division").all()


def test_sort_orders_trees_and_is_stable(table) -> None:
    """Trees follow the founder's key; ties keep founder id order."""

    def order(sort: str) -> list[int]:
        tr = m.layout(table, sort=sort).tracks
        roots = tr[tr["parent"] == 0].sort_values("x")
        return roots.index.tolist()

    assert order("track") == [1, 4, 10]
    assert order("length") == [4, 1, 10]
    assert order("edits") == [4, 1, 10]
    assert order("outcome") == [1, 10, 4]  # divided, lost, then none
    assert order("length") == order("length")


def test_unknown_options_raise(table) -> None:
    """Unknown alignment, sort or colour mode is an error."""
    with pytest.raises(ValueError, match="alignment"):
        m.layout(table, align="mitosis")
    with pytest.raises(ValueError, match="sort"):
        m.layout(table, sort="age")
    with pytest.raises(ValueError, match="colour"):
        m.node_categories(table.nodes, table.tracks, "rainbow")


# -- colours -----------------------------------------------------------------------


def test_node_categories(table) -> None:
    """Every node gets a category of the mode's palette."""
    nodes, tracks = table.nodes, table.tracks
    for mode in m.COLOUR_MODES:
        cats = m.node_categories(nodes, tracks, mode)
        assert len(cats) == len(nodes)
        assert set(cats) <= set(m.PALETTES[mode])
    key = list(zip(nodes["track"], nodes["frame"], strict=True))
    flags = dict(zip(key, m.node_categories(nodes, tracks, "flags"), strict=False))
    assert flags[(4, 3)] == "mask"
    assert flags[(3, 8)] == "after gap"
    assert flags[(1, 0)] == ""
    verdict = dict(zip(key, m.node_categories(nodes, tracks, "verdict"), strict=False))
    assert verdict[(1, 0)] == "accept" and verdict[(2, 5)] == ""
    edits = dict(zip(key, m.node_categories(nodes, tracks, "edits"), strict=False))
    assert edits[(4, 0)] == "≤1/h" and edits[(1, 0)] == "0"


def test_edit_density_bins() -> None:
    """Densities fall in the palette's bins, in order."""
    assert [
        m.edit_density_bin(v) for v in (0, 0.2, 0.9, 1.5, 3)
    ] == list(m.PALETTES["edits"])


# -- review records ------------------------------------------------------------------


def test_review_records_merge_queues(tmp_path: Path) -> None:
    """A human verdict beats a later agent one; the audit is recorded apart."""

    def write(name: str, decisions: list[dict]) -> None:
        (tmp_path / name).write_text(json.dumps({"decisions": decisions}))

    write(
        "decisions.json",
        [
            {"id": "C2-t0-L5", "verdict": "accept", "author": "human",
             "time": "2026-10-01T10:00:00", "outcome": "divided"},
            {"id": "C2-t0-L9", "verdict": "unsure", "author": "agent",
             "time": "2026-10-01T10:00:00"},
        ],
    )
    write(
        "audit_decisions.json",
        [
            {"id": "C2-t0-L5", "verdict": "reject", "author": "agent",
             "time": "2026-10-02T10:00:00"},
            {"id": "C2-t0-L9", "verdict": "correct", "author": "agent",
             "time": "2026-10-02T10:00:00", "outcome": "lost"},
        ],
    )
    rec = m.review_records(tmp_path)
    assert rec["C2-t0-L5"].verdict == "accept"
    assert rec["C2-t0-L5"].reviewed_by == "agent+human"
    assert rec["C2-t0-L5"].audit == "reject"
    assert rec["C2-t0-L9"].verdict == "correct"
    assert rec["C2-t0-L9"].outcome == "lost"
    assert rec["C2-t0-L9"].reviewed_by == "agent"
    assert m.review_records(tmp_path / "missing") == {}


# -- divisions ----------------------------------------------------------------------


def _fates(segments: str, outcome: str, frame: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segments": [segments],
            "outcome": [outcome],
            "outcome_frame": [frame],
            "excluded": [False],
            "review_flags": [""],
        }
    )


def test_tracker_link_is_no_division_if_the_walker_says_so() -> None:
    """A parent link the walker does not call a division is a faint split."""
    det, curated = _toy()
    slow = m.SlowAnnotations(fates=_fates("1", "no_mitosis", 9.0))
    t = m.build_lineage_table(det, curated, well="C2", slow=slow)
    assert not t.tracks.loc[1, "divided"]
    assert np.isnan(t.tracks.loc[1, "ev_division"])
    assert t.tracks.loc[1, "outcome"] == "no_mitosis"
    assert set(t.edges.loc[t.edges["track"] != t.edges["source"], "kind"]) == {"split"}
    lay = m.layout(t, align="division", park=False)
    assert lay.tracks.empty  # nothing to align on


def test_walker_division_dates_and_draws_the_division(monkeypatch) -> None:
    """The walker's division is dated at the mother's last frame."""
    monkeypatch.setattr(m, "MIN_DAUGHTER_FRAMES", 4)
    det, curated = _toy()
    slow = m.SlowAnnotations(fates=_fates("1", "divided", 5.0))
    t = m.build_lineage_table(det, curated, well="C2", slow=slow)
    assert t.tracks.loc[1, "divided"]
    assert t.tracks.loc[1, "ev_division"] == 4
    assert set(t.edges.loc[t.edges["track"] != t.edges["source"], "kind"]) == {"division"}


def test_walker_outcome_belongs_to_the_last_segment(monkeypatch) -> None:
    """A cell followed through 4 then 1: only 1 divided."""
    monkeypatch.setattr(m, "MIN_DAUGHTER_FRAMES", 4)
    det, curated = _toy()
    slow = m.SlowAnnotations(fates=_fates("4 1", "divided", 5.0))
    t = m.build_lineage_table(det, curated, well="C2", slow=slow)
    assert t.tracks.loc[1, "divided"] and not t.tracks.loc[4, "divided"]


def test_unfollowed_tracks_need_two_long_daughters(monkeypatch) -> None:
    """Without a walker call a link counts if two daughters are long enough."""
    det, curated = _toy()
    t = m.build_lineage_table(det, curated, well="C2")
    assert not t.tracks.loc[1, "divided"]  # daughters of 4 frames
    monkeypatch.setattr(m, "MIN_DAUGHTER_FRAMES", 4)
    t = m.build_lineage_table(det, curated, well="C2")
    assert t.tracks.loc[1, "divided"]
    assert t.tracks.loc[1, "outcome"] == "divided"


def test_pooled_reference_is_used(table) -> None:
    """A reference passed in is used as is, and reported as pooled."""
    from omero_screen_track.phases import WellReference

    det, curated = _toy()
    ref = WellReference(bleed=0.0, pip_high=10.0, gem_high=20.0)
    inputs = m.WellInputs(
        det=det,
        curated=curated,
        well="C2",
        edits=[],
        records={},
        queue={},
        window=(0, 9),
        interval_minutes=20.0,
        reference=ref,
    )
    slow = m.slow_annotations(inputs, walker=False, min_frames=6)
    assert slow.pooled and slow.reference is ref
    assert slow.phases is not None
    assert set(slow.phases["track"]) == {4}  # the only track of ≥ 6 frames
    t = m.table_from_inputs(inputs, slow)
    assert t.pooled and t.has_hmm


def test_hmm_only_inside_the_window() -> None:
    """HMM phases are called on the analysis window only."""
    det, curated = _toy()
    inputs = m.WellInputs(
        det=det,
        curated=curated,
        well="C2",
        edits=[],
        records={},
        queue={},
        window=(0, 6),
        interval_minutes=20.0,
    )
    slow = m.slow_annotations(inputs, walker=False, min_frames=6)
    assert not slow.pooled  # no plate: fitted on the well
    assert slow.phases is not None and slow.phases["frame"].max() <= 6


# -- review directories ----------------------------------------------------------------


def _review_dir(
    path: Path, wells: list[str], plate: int = 5054, decided: float = 0
) -> Path:
    path.mkdir(parents=True)
    (path / "edits.jsonl").write_text("")
    (path / "correction.json").write_text(
        json.dumps({"plate_id": plate, "wells": wells, "hash": "x"})
    )
    if decided:
        d = path / "decisions.json"
        d.write_text(json.dumps({"decisions": []}))
        import os

        os.utime(d, (decided, decided))
    return path


def test_discover_review_dirs(tmp_path: Path) -> None:
    """Siblings of the plate, newest review first per well; one level only."""
    c2 = _review_dir(tmp_path / "review_agent", ["C2"], decided=200)
    _review_dir(tmp_path / "review", ["C2"], decided=100)
    c3 = _review_dir(tmp_path / "review_agent_c3", ["C3"], decided=50)
    _review_dir(tmp_path / "other_plate", ["C3"], plate=1, decided=900)
    _review_dir(c2 / "_superseded", ["C4"], decided=999)
    (tmp_path / "not_a_review").mkdir()
    found = m.discover_review_dirs(c3, 5054)
    assert [d.path.name for d in found["C2"]] == ["review_agent", "review"]
    assert [d.path for d in found["C3"]] == [c3]
    assert "C4" not in found


def test_plate_loader_choices(tmp_path: Path) -> None:
    """The session's well is interactive; others read-only, from their dirs."""
    from types import SimpleNamespace

    from omero_screen_napari.review_queue import ReviewItem

    _review_dir(tmp_path / "review_agent", ["C2"], decided=200)
    c3 = _review_dir(tmp_path / "review_agent_c3", ["C3"], decided=50)
    session = SimpleNamespace(
        queue_path=c3 / "queue.json",
        queue=SimpleNamespace(
            items=(ReviewItem(id="C3-t72-L1", well="C3", frame=72),)
        ),
    )
    loader = m.PlateLoader(
        5054, session=session, wells=["C2", "C3", "C4"]  # ty: ignore[invalid-argument-type]
    )
    assert loader.interactive("C3")
    assert not loader.interactive("C2")
    assert loader.review_dir("C2") == tmp_path / "review_agent"
    assert loader.review_dir("C4") is None
    assert loader.watch_paths("C4") == []
    loader.set_review_dir("C3", None)
    assert not loader.interactive("C3")
    assert loader.review_dir("C3") is None
