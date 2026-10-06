"""Review queue I/O and the Track review widget's decision loop."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omero_screen_napari.review_queue import (
    Decision,
    PathPoint,
    QueueError,
    ReviewItem,
    decisions_path,
    latest_decisions,
    read_queue,
    record_decision,
    write_queue,
)


def _item(item_id: str = "C2-1", frame: int = 10) -> ReviewItem:
    path = (
        PathPoint(9, 100.0, 200.0, 5),
        PathPoint(10, 102.0, 201.0, 0),
        PathPoint(11, 104.0, 203.0, 7),
    )
    return ReviewItem(
        id=item_id,
        well="C2",
        frame=frame,
        reason="lost",
        question="where?",
        outcome="lost",
        path=path,
    )


def test_queue_round_trip(tmp_path: Path) -> None:
    q = tmp_path / "queue.json"
    write_queue(q, 5054, [_item(), _item("C2-2", 20)], focus="C2-2")
    queue = read_queue(q)
    assert queue.plate_id == 5054
    assert queue.focus == "C2-2"
    assert [i.id for i in queue.items] == ["C2-1", "C2-2"]
    assert queue.items[0].path[1] == PathPoint(10, 102.0, 201.0, 0)
    assert "not_a_cell" in queue.outcomes


def test_point_at_uses_nearest_earlier_frame() -> None:
    item = _item()
    assert item.point_at(10).label == 0
    assert item.point_at(50).t == 11
    assert item.point_at(0).t == 9  # before the path: first point


@pytest.mark.parametrize(
    "payload",
    [
        {"items": []},
        {"plate_id": 1, "items": [{"id": "a"}]},
        {
            "plate_id": 1,
            "items": [
                {"id": "a", "well": "C2", "frame": 1},
                {"id": "a", "well": "C2", "frame": 2},
            ],
        },
    ],
)
def test_malformed_queue_raises(tmp_path: Path, payload: dict) -> None:
    q = tmp_path / "queue.json"
    q.write_text(json.dumps(payload))
    with pytest.raises(QueueError):
        read_queue(q)


def test_decisions_append_and_latest_wins(tmp_path: Path) -> None:
    d = decisions_path(tmp_path / "queue.json")
    record_decision(d, Decision(id="C2-1", verdict="unsure"))
    record_decision(
        d, Decision(id="C2-1", verdict="reject", outcome="death", frames=[12])
    )
    record_decision(d, Decision(id="C2-2", verdict="accept"))
    latest = latest_decisions(d)
    assert latest["C2-1"].verdict == "reject"
    assert latest["C2-1"].frames == [12]
    assert latest["C2-1"].time
    assert len(json.loads(d.read_text())["decisions"]) == 3


def test_unknown_verdict_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        record_decision(
            tmp_path / "decisions.json", Decision(id="x", verdict="maybe")
        )


@pytest.fixture(autouse=True)
def isolated_settings():
    """Never let a test overwrite the reviewer's real 'last queue' setting."""
    settings = MagicMock()
    settings.value.return_value = ""
    with patch(
        "omero_screen_napari._review_widget.QSettings", return_value=settings
    ):
        yield settings


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_manifest_points_at_widget_class() -> None:
    """napari injects ``napari_viewer`` into classes, not into bare functions."""
    from importlib.resources import files

    import yaml

    manifest = yaml.safe_load(
        files("omero_screen_napari").joinpath("napari.yaml").read_text()
    )
    cmd = next(
        c
        for c in manifest["contributions"]["commands"]
        if c["id"].endswith("track_review_widget")
    )
    assert cmd["python_name"].endswith(":TrackReviewWidget")


def _queue(tmp_path: Path) -> Path:
    q = tmp_path / "queue.json"
    write_queue(q, 5054, [_item("C2-t72-L1", 80), _item("C2-t72-L2", 90)])
    return q


def _viewer() -> MagicMock:
    viewer = MagicMock()
    viewer.dims.current_step = (85, 0, 0)
    viewer.layers.__contains__.return_value = False
    viewer.layers.__iter__.return_value = iter([])
    return viewer


def _widget(tmp_path: Path):
    from omero_screen_napari._review_widget import TrackReviewWidget
    from omero_screen_napari.review.session import ReviewSession

    q = _queue(tmp_path)
    viewer = _viewer()
    patches = [
        patch.object(TrackReviewWidget, "_ensure_well"),
        patch.object(ReviewSession, "path", side_effect=RuntimeError("no db")),
        patch.object(ReviewSession, "breaks", return_value=[86]),
        patch("omero_screen_napari._review_widget.notifications"),
    ]
    for p in patches:
        p.start()
    widget = TrackReviewWidget(napari_viewer=viewer)
    widget.load_queue(q)
    return widget, viewer, q, patches


def test_widget_opens_first_item_and_records_verdicts(
    qapp, tmp_path: Path
) -> None:
    """Loading selects the first open item; a verdict is saved and advances."""
    widget, viewer, q, patches = _widget(tmp_path)
    try:
        assert widget.current.id == "C2-t72-L1"
        viewer.dims.set_current_step.assert_called_with(0, 80)
        widget.note.setText("looks fine")
        widget.decide("accept")
        assert (
            latest_decisions(decisions_path(q))["C2-t72-L1"].note
            == "looks fine"
        )
        assert widget.current.id == "C2-t72-L2"
    finally:
        for p in patches:
            p.stop()


def test_pick_and_candidate_link_go_through_the_session(
    qapp, tmp_path: Path
) -> None:
    """Clicking a nucleus or pressing a candidate number writes a link edit."""
    import numpy as np
    import pandas as pd

    from omero_screen_napari.review.session import ReviewSession

    widget, viewer, q, patches = _widget(tmp_path)
    try:
        nuclei = np.zeros((100, 50, 50), dtype=np.uint32)
        nuclei[85, 30:36, 40:46] = 3132
        widget._nuclei, widget._pixel_size = nuclei, 0.5
        with patch.object(ReviewSession, "edit") as edit:
            assert widget.pick_at((85, 10.0, 5.0)) == 0  # background
            assert widget.pick_at((85, 33 * 0.5, 43 * 0.5)) == 3132
            edit.assert_called_with(
                "C2-t72-L1", "link", {"frame": 85, "label": 3132}, reason=""
            )
            widget.cands = pd.DataFrame(
                {
                    "rank": [1, 2],
                    "timepoint": [86, 87],
                    "label": [7, 9],
                    "y": [1.0, 2.0],
                    "x": [1.0, 2.0],
                }
            )
            widget.link_candidate(2)
            edit.assert_called_with(
                "C2-t72-L1",
                "link",
                {"frame": 87, "label": 9},
                reason="candidate 2",
            )
    finally:
        for p in patches:
            p.stop()


def test_session_proposals_change_nothing_until_confirmed(
    tmp_path: Path,
) -> None:
    """An agent proposal is stored pending; confirming applies it as an agent edit."""
    from omero_screen_napari.review.session import ReviewSession

    session = ReviewSession(_queue(tmp_path))
    with patch.object(ReviewSession, "edit") as edit:
        prop = session.propose(
            "C2-t72-L1",
            "link",
            {"frame": 86, "label": 7},
            "2.1 diameters, markers continuous",
        )
        edit.assert_not_called()
        assert session.status()["pending_proposals"] == 1
        session.resolve_proposal(prop.id, confirm=True)
        edit.assert_called_once_with(
            "C2-t72-L1",
            "link",
            {"frame": 86, "label": 7},
            author="agent",
            confirmed_by="human",
            reason="2.1 diameters, markers continuous",
        )
    assert session.proposals()[0].status == "confirmed"
    with pytest.raises(ValueError):
        session.resolve_proposal(prop.id, confirm=False)


def test_session_status_counts_per_well(tmp_path: Path) -> None:
    """Status reports reviewed items per well."""
    from omero_screen_napari.review.session import ReviewSession

    session = ReviewSession(_queue(tmp_path))
    session.verdict("C2-t72-L1", "reject", outcome="debris")
    st = session.status()
    assert st["reviewed"] == 1 and st["per_well"]["C2"] == {
        "queued": 2,
        "reviewed": 1,
    }
