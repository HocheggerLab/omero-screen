"""ReviewSession: search, lineage and edit previews, on a toy lineage."""

from pathlib import Path

import pandas as pd
import pytest

from cellview.tracks.edit import Curated
from omero_screen_napari.review.session import ReviewSession, _span
from omero_screen_napari.review_queue import ReviewItem, write_queue


def _det() -> pd.DataFrame:
    """Track 1 (label 5) t0–3 divides into 2 (label 6) and 3 (label 7) at t4;
    a neighbour, track 4 (label 9), runs t0–5."""
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
