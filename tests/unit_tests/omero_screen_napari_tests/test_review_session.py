"""ReviewSession: search, lineage and edit previews, on a toy lineage."""

import json
from pathlib import Path

import pandas as pd
import pytest
from omero_screen_track.edit import Curated
from omero_screen_napari.review.session import ReviewSession, _span
from omero_screen_napari.review_queue import (
    QueueError,
    ReviewItem,
    write_queue,
)


def _det() -> pd.DataFrame:
    """Track 1 (label 5) t0–3 divides into 2 (label 6) and 3 (label 7) at t4;
    a neighbour, track 4 (label 9), runs t0–5.
    """
    rows = []
    for t in range(4):
        rows.append((t, 5, 10.0, 10.0 + t))
    for t in range(4, 6):
        rows += [(t, 6, 8.0, 14.0), (t, 7, 12.0, 14.0)]
    for t in range(6):
        rows.append((t, 9, 40.0, 40.0))
    det = pd.DataFrame(rows, columns=["timepoint", "label", "y", "x"])
    det["area"] = 100.0
    det["geminin"] = 1.0
    return det


def _curated() -> Curated:
    tracks = {}
    for t in range(4):
        tracks[(t, 5)] = 1
    for t in range(4, 6):
        tracks[(t, 6)] = 2
        tracks[(t, 7)] = 3
    for t in range(6):
        tracks[(t, 9)] = 4
    return Curated(tracks=tracks, parents={1: 0, 2: 1, 3: 1, 4: 0}, next_id=5)


@pytest.fixture
def session(tmp_path: Path, monkeypatch) -> ReviewSession:
    q = tmp_path / "queue.json"
    write_queue(q, 1, [ReviewItem(id="C2-t0-L5", well="C2", frame=0)])
    s = ReviewSession(q)
    det, curated = _det(), _curated()
    monkeypatch.setattr(s.source, "well", lambda well: (det, curated))
    return s


def test_find_returns_queue_item_for_same_track(session) -> None:
    """A search hitting a queued cell's track opens that queue item."""
    assert session.find("C2", 2, 5).id == "C2-t0-L5"
    assert not session.adhoc


def test_find_by_label_opens_adhoc_item(session) -> None:
    """A label not in the queue opens an ad-hoc item usable like a queue item."""
    item = session.find("C2", 3, 9)
    assert item.id == "C2-t3-L9"
    assert session.item(item.id) is item
    assert session.find("C2", 3, 9) is item


def test_find_by_track_id(session) -> None:
    """A number that is no label in the frame is read as a track id."""
    assert session.find("C2", 5, 3).id == "C2-t5-L7"


def test_find_unknown_raises(session) -> None:
    """Nothing in the frame matches: a clear error."""
    with pytest.raises(KeyError, match="frame 2"):
        session.find("C2", 2, 77)


def test_lineage_has_mother_and_daughters(session) -> None:
    """The mother's track comes with both daughters' tracks."""
    roles = {
        t["track_id"]: (t["role"], t["parent"])
        for t in session.lineage("C2-t0-L5")
    }
    assert roles == {
        1: ("cell", 0),
        2: ("daughter", 1),
        3: ("daughter", 1),
    }


def test_preview_link_names_frames_and_source(session) -> None:
    """Linking the neighbour from t4 says what is taken from whom."""
    lines, after = session.preview(
        "C2-t0-L5", "link", {"frame": 4, "label": 9}
    )
    assert "gains t4–5 from track 4" in lines
    assert after.timepoint.max() == 5
    # nothing was written
    assert not session.edits_path.exists()


def test_preview_unlink_moves_the_tail(session) -> None:
    """Unlinking says which frames go to a new track."""
    lines, after = session.preview("C2-t0-L5", "unlink", {"frame": 2})
    assert "loses t2–3 to a new track" in lines
    assert after.timepoint.max() == 1


def test_preview_annotation(session) -> None:
    """Annotations change no frames; the summary repeats the arguments."""
    lines, _ = session.preview(
        "C2-t0-L5", "event", {"kind": "mitosis", "frame": 3}
    )
    assert lines == ["event: kind mitosis, frame 3"]


def test_span() -> None:
    """Frames compress to runs."""
    assert _span([4]) == "t4"
    assert _span([4, 5, 6, 9]) == "t4–6, t9"


def _queue_with_proposals(tmp_path: Path) -> Path:
    q = tmp_path / "queue.json"
    item = ReviewItem(
        id="C2-t0-L5",
        well="C2",
        frame=0,
        proposals=(
            {
                "op": "link",
                "args": {"frame": 4, "label": 6},
                "reason": "bridge: gap 1, 1 candidate",
                "confidence": "clear",
            },
            {
                "op": "link",
                "args": {"frame": 5, "label": 7},
                "reason": "switch at t5",
                "confidence": "uncertain",
            },
        ),
    )
    write_queue(q, 1, [item])
    return q


def test_walker_proposals_are_seeded_once(tmp_path: Path, monkeypatch) -> None:
    """Clear suggestions from the queue are applied on load, uncertain ones wait;
    reopening the session does not duplicate them.
    """
    applied = []
    monkeypatch.setattr(
        ReviewSession,
        "edit",
        lambda self, item, op, args, **kw: applied.append((op, args))
        or type("E", (), {"id": "e0001"})(),
    )
    q = _queue_with_proposals(tmp_path)
    s = ReviewSession(q)
    props = s.proposals()
    assert [p.status for p in props] == ["applied", "pending"]
    assert all(p.author == "walker" for p in props)
    assert applied == [("link", {"frame": 4, "label": 6})]
    assert s.status()["pending_proposals"] == 1
    ReviewSession(q)
    assert len(s.proposals()) == 2


def test_failed_walker_proposal_is_recorded_not_raised(
    tmp_path: Path, monkeypatch
) -> None:
    def boom(self, *a, **k):
        raise ValueError("no such nucleus")

    monkeypatch.setattr(ReviewSession, "edit", boom)
    s = ReviewSession(_queue_with_proposals(tmp_path))
    first = s.proposals()[0]
    assert first.status == "failed" and "no such nucleus" in first.note


def test_applied_proposal_can_be_confirmed_or_reverted(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        ReviewSession,
        "edit",
        lambda self, item, op, args, **kw: type("E", (), {"id": "e0001"})(),
    )
    reverted = []
    monkeypatch.setattr(
        ReviewSession,
        "_revert",
        lambda self, p, author="human": reverted.append(p.id),
    )
    s = ReviewSession(_queue_with_proposals(tmp_path))
    applied, pending = s.proposals()
    assert s.confirm_applied("C2-t0-L5")[0].id == applied.id
    assert s.proposals()[0].status == "confirmed"
    s2 = ReviewSession(_queue_with_proposals(tmp_path / "b"))
    a2 = s2.proposals()[0]
    assert s2.resolve_proposal(a2.id, False, note="wrong").status == "reverted"
    assert reverted == [a2.id]
    with pytest.raises(ValueError, match="already"):
        s2.resolve_proposal(a2.id, True)


def test_queue_on_another_correction_is_refused(tmp_path: Path) -> None:
    q = tmp_path / "queue.json"
    write_queue(q, 1, [ReviewItem(id="C2-t0-L5", well="C2", frame=0)])
    data = json.loads(q.read_text())
    data["correction"] = "a" * 64
    q.write_text(json.dumps(data))
    (tmp_path / "correction.json").write_text(json.dumps({"hash": "b" * 64}))
    with pytest.raises(QueueError, match="another|new correction"):
        ReviewSession(q)
    (tmp_path / "correction.json").write_text(json.dumps({"hash": "a" * 64}))
    assert ReviewSession(q).queue.correction == "a" * 64


def test_second_queue_keeps_its_own_verdicts(tmp_path: Path) -> None:
    """An audit queue beside the review shares the edit log but not the verdicts."""
    from omero_screen_napari.review_queue import (
        decisions_path,
        latest_decisions,
    )

    write_queue(
        tmp_path / "queue.json",
        1,
        [ReviewItem(id="C2-t0-L5", well="C2", frame=0)],
    )
    write_queue(
        tmp_path / "audit.json",
        1,
        [ReviewItem(id="C2-t0-L5", well="C2", frame=0)],
    )
    review, audit = (
        ReviewSession(tmp_path / "queue.json"),
        ReviewSession(tmp_path / "audit.json"),
    )
    assert review.edits_path == audit.edits_path
    assert audit.proposals_path.name == "audit_proposals.json"
    review.verdict("C2-t0-L5", "correct", outcome="divided")
    audit.verdict("C2-t0-L5", "accept", outcome="divided")
    assert (
        decisions_path(tmp_path / "audit.json").name == "audit_decisions.json"
    )
    assert (
        latest_decisions(decisions_path(tmp_path / "queue.json"))[
            "C2-t0-L5"
        ].verdict
        == "correct"
    )
    assert audit.decided()["C2-t0-L5"].verdict == "accept"


def test_undo_reverts_only_this_cells_edits(session, monkeypatch) -> None:
    """Shift-Z walks back through the cell's own edits, never a neighbour's."""
    from omero_screen_track.edit import EditError, EditLog

    monkeypatch.setattr(session.source, "base", lambda well: _curated())
    session.edit("C2-t0-L5", "note", {"text": "mine"})
    session.find("C2", 0, 9)
    session.edit("C2-t0-L9", "note", {"text": "neighbour"})
    entry = session.undo("C2-t0-L5")
    assert entry.args["target"] == "e0001"
    with pytest.raises(EditError, match="Nothing to undo for this cell"):
        session.undo("C2-t0-L5")
    live = {e.args.get("target") for e in EditLog(session.edits_path)}
    assert "e0002" not in live


@pytest.fixture
def two_items(tmp_path: Path, monkeypatch) -> ReviewSession:
    """The dividing cell and its neighbour (track 4, label 9) both queued."""
    q = tmp_path / "queue.json"
    write_queue(
        q,
        1,
        [
            ReviewItem(id="C2-t0-L5", well="C2", frame=0),
            ReviewItem(id="C2-t0-L9", well="C2", frame=0),
        ],
    )
    s = ReviewSession(q)
    det, curated = _det(), _curated()
    monkeypatch.setattr(s.source, "well", lambda well: (det, curated))
    return s


def test_link_taking_frames_from_a_decided_item_is_reported(two_items) -> None:
    link = {"frame": 4, "label": 9}
    assert two_items.taken_from_decided("C2-t0-L5", "link", link) == []
    two_items.verdict("C2-t0-L9", "correct")
    assert two_items.taken_from_decided("C2-t0-L5", "link", link) == [
        "takes t4–5 from C2-t0-L9, which already has a verdict (correct)"
    ]
    # annotations and edits that do not apply are not checked
    assert two_items.taken_from_decided("C2-t0-L5", "note", {"text": "x"}) == []
    bad = {"frame": 4, "label": 99}
    assert two_items.taken_from_decided("C2-t0-L5", "link", bad) == []


def test_applied_proposal_carries_the_warning(two_items, monkeypatch) -> None:
    monkeypatch.setattr(
        ReviewSession,
        "edit",
        lambda self, item, op, args, **kw: type("E", (), {"id": "e0001"})(),
    )
    two_items.verdict("C2-t0-L9", "correct")
    prop = two_items.propose(
        "C2-t0-L5", "link", {"frame": 4, "label": 9}, "re-emerges", "clear"
    )
    assert prop.status == "applied"
    assert prop.note.startswith("WARNING: takes t4–5 from C2-t0-L9")
