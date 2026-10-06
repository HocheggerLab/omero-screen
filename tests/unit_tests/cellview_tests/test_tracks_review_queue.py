"""Review-queue building: sampling, audit, overrides from curation."""

import json

import pandas as pd
import pytest
from cellview.tracks import review_queue as rq
from cellview.tracks.edit import Curated, EditLog
from cellview.tracks.fate import apply_annotations


def _det(n_cells: int = 6, frames: int = 30) -> pd.DataFrame:
    """Moving G1 cells, one raw track each, plus a stationary debris object."""
    rows, mid = [], 0
    for c in range(n_cells):
        for t in range(frames):
            mid += 1
            rows.append(
                (
                    mid,
                    t,
                    c + 1,
                    0,
                    400.0,
                    100.0 + 300 * c,
                    100.0 + 10 * t,
                    1000.0,
                    100.0,
                    1000.0,
                )
            )
    for t in range(frames):
        mid += 1
        rows.append((mid, t, 99, 0, 400.0, 3000.0, 3000.0, 0.0, 0.0, 1000.0))
    df = pd.DataFrame(
        rows,
        columns=[
            "measurement_id",
            "timepoint",
            "track_id_raw",
            "parent_track_id_raw",
            "area",
            "y",
            "x",
            "pip",
            "geminin",
            "spydna",
        ],
    )
    df["label"] = df["track_id_raw"]
    return df


@pytest.fixture
def fake_well(monkeypatch):
    monkeypatch.setattr(rq, "load_well", lambda conn, plate, well: _det())


def test_sampling_is_reproducible_and_excludes_debris(fake_well) -> None:
    """Same seed, same sample; excluded objects are never sampled."""
    spec = rq.QueueSpec(start=5, stop=25, sample=3, audit=1.0, seed=7)
    a, items_a = rq.build_well(None, 1, "C2", spec)
    b, _ = rq.build_well(None, 1, "C2", spec)
    assert list(a.id) == list(b.id) and len(a) == 3
    assert 99 not in set(a.label)
    assert all(i["id"].startswith("C2-t5-L") for i in items_a)
    assert set(a.reason) <= {"audit"} | set(
        rq.QUESTIONS
    )  # unflagged cells all audited at audit=1


def test_no_audit_queues_only_flagged(fake_well) -> None:
    """With audit=0, unflagged cells are sampled but not queued."""
    sample, items = rq.build_well(
        None, 1, "C2", rq.QueueSpec(start=5, stop=25, sample=6, audit=0.0)
    )
    assert len(items) == int((sample.n_flags > 0).sum())


def test_queue_files_written_and_shuffled(fake_well, tmp_path) -> None:
    """queue.json is readable by the widget format; sample.csv lists every sampled cell."""
    q, s, table = rq.build_queue(
        None,
        1,
        ["C2", "C3"],
        rq.QueueSpec(start=5, stop=25, sample=2, audit=1.0),
        tmp_path,
    )
    data = json.loads(q.read_text())
    assert data["plate_id"] == 1 and len(data["items"]) == 4
    assert {i["well"] for i in data["items"]} == {"C2", "C3"}
    assert len(pd.read_csv(s)) == 4


def test_curated_annotations_override_the_walker() -> None:
    """A curated death event replaces the automatic outcome; exclude excludes."""
    cells = pd.DataFrame(
        {
            "cell": [1, 2],
            "segments": ["5", "6"],
            "outcome": ["lost", "no_mitosis"],
            "outcome_frame": [20, 30],
            "censored": [True, True],
            "excluded": [False, False],
        }
    )
    cur = Curated(
        tracks={},
        parents={},
        annotations={
            5: {"events": [{"kind": "death", "frame": 18}]},
            6: {"excluded": "debris"},
        },
    )
    out = apply_annotations(cells, cur).set_index("cell")
    assert (
        out.loc[1, "outcome"] == "death"
        and out.loc[1, "outcome_frame"] == 18
        and not out.loc[1, "censored"]
    )
    assert out.loc[2, "excluded"] and out.loc[2, "outcome"] == "debris"
    assert out.curated.all()


def test_edit_log_is_replayed_before_sampling(fake_well, tmp_path) -> None:
    """Exclusions in the log remove cells from the eligible pool."""
    log = EditLog(tmp_path / "edits.jsonl")
    for lab in range(1, 6):
        log.append("exclude", "C2", {"cell": [5, lab], "reason": "debris"})
    sample, _ = rq.build_well(
        None, 1, "C2", rq.QueueSpec(start=5, stop=25, sample=10), log
    )
    assert list(sample.label) == [6]
