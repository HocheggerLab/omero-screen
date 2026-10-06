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


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_widget_records_decision_and_follows_focus(
    qapp, tmp_path: Path
) -> None:
    from omero_screen_napari._review_widget import TrackReviewWidget

    q = tmp_path / "queue.json"
    write_queue(q, 5054, [_item(), _item("C2-2", 20)], focus="C2-2")
    viewer = MagicMock()
    viewer.dims.current_step = (11, 0, 0)
    viewer.layers.__contains__.return_value = False
    with (
        patch.object(TrackReviewWidget, "_ensure_well"),
        patch.object(TrackReviewWidget, "_draw"),
    ):
        # napari injects the viewer by this keyword when opened from the menu.
        widget = TrackReviewWidget(napari_viewer=viewer)
        widget.load_queue(q)
        assert widget.current is not None and widget.current.id == "C2-2"
        viewer.dims.set_current_step.assert_called_with(0, 20)

        widget.items.setCurrentRow(0)
        widget._mark_frame()
        widget.decide("reject")

    latest = latest_decisions(decisions_path(q))
    assert latest["C2-1"].verdict == "reject"
    assert latest["C2-1"].frames == [11]
    assert latest["C2-1"].outcome == "lost"


def test_manifest_points_at_widget_class() -> None:
    """napari injects ``napari_viewer`` into classes, not into bare functions."""
    from importlib.resources import files

    import yaml

    manifest = yaml.safe_load(files("omero_screen_napari").joinpath("napari.yaml").read_text())
    cmd = next(c for c in manifest["contributions"]["commands"] if c["id"].endswith("track_review_widget"))
    assert cmd["python_name"].endswith(":TrackReviewWidget")
