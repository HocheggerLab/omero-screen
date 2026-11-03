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
    assert a.attrs["exclusions"]["debris_hidden"] == 1
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


def test_stationary_cells_with_reporter_raise_a_warning() -> None:
    """A well where many non-moving objects still express a reporter is flagged."""
    cells = pd.DataFrame(
        {
            "outcome": ["divided"] * 8 + ["stationary"] * 2,
            "excluded": [False] * 8 + [True] * 2,
        }
    )
    summary = rq.exclusion_summary(cells, warn_fraction=0.05)
    assert summary["stationary"] == 2 and summary["stationary_fraction"] == 0.2
    assert summary["stationary_warning"]
    calm = rq.exclusion_summary(
        cells.assign(outcome=["divided"] * 8 + ["debris"] * 2)
    )
    assert calm["debris"] == 2 and not calm["stationary_warning"]


def test_queue_records_exclusions(fake_well, tmp_path) -> None:
    """queue.json carries the per-well exclusion counts."""
    q, _, table = rq.build_queue(
        None, 1, ["C2"], rq.QueueSpec(start=5, stop=25, sample=2), tmp_path
    )
    data = json.loads(q.read_text())
    # The stationary, reporter-negative object is hidden before the walker.
    assert data["exclusions"]["C2"]["debris_hidden"] == 1
    assert table.attrs["exclusions"]["C2"]["start_cells"] == 6


def test_queue_carries_walker_proposals_as_raw_label_edits(
    monkeypatch,
) -> None:
    """A bridged gap becomes a link proposal naming the raw label of the next track."""
    det = _det(n_cells=3)
    # Break cell 2's track: frames 10-29 get a new raw id, 50, with frame 10 missing.
    broken = det["track_id_raw"].eq(2) & det["timepoint"].ge(10)
    det.loc[broken, ["track_id_raw", "label"]] = 50
    det = det[~(det["label"].eq(50) & det["timepoint"].eq(10))]
    monkeypatch.setattr(rq, "load_well", lambda conn, plate, well: det)
    _, items = rq.build_well(
        None, 1, "C2", rq.QueueSpec(start=5, stop=25, sample=10, audit=0.0)
    )
    (item,) = [i for i in items if i["id"] == "C2-t5-L2"]
    assert "gap" in item["reason"]
    (p,) = item["proposals"]
    assert p["op"] == "link" and p["args"] == {"frame": 11, "label": 50}
    assert p["confidence"] == "uncertain"  # a gap of 2 frames


def test_edit_log_on_another_correction_is_refused(tmp_path) -> None:
    """The correction sidecar guards an edit log against a re-run correction."""
    rq.record_correction(tmp_path, 1, ["C2"], "a" * 64)
    rq.check_correction(tmp_path, "a" * 64)
    with pytest.raises(ValueError, match="another correction|old correction"):
        rq.check_correction(tmp_path, "b" * 64)
    rq.check_correction(
        tmp_path, None
    )  # no stored correction: nothing to compare


def test_imported_lineage_is_the_identity() -> None:
    """The review starts from the lineage CellView holds, parents included."""
    from cellview.tracks.edit import imported_lineage

    det = _det(n_cells=2)
    det.loc[det["label"].eq(2), "parent_track_id_raw"] = 1
    cur = imported_lineage(det)
    assert cur.tracks[(3, 2)] == 2 and cur.parents == {1: 0, 2: 1, 99: 0}
    assert cur.next_id == 100


def test_audit_queue_draws_n_analysed_cells_without_proposals(
    fake_well, tmp_path
) -> None:
    """The final audit is N random non-excluded cells, end to end, nothing auto-applied."""
    out = rq.build_audit(
        None, 1, ["C2"], 3, tmp_path / "audit.json", start=5, stop=25, seed=4
    )
    data = json.loads(out.read_text())
    items = data["items"]
    assert len(items) == 3 and len({i["id"] for i in items}) == 3
    assert all(
        i["reason"] == "final_audit" and i["proposals"] == [] for i in items
    )
    assert all(i["frame"] == 5 for i in items)
    assert "C2-t5-L99" not in {i["id"] for i in items}  # debris never drawn
    again = json.loads(
        rq.build_audit(
            None, 1, ["C2"], 3, tmp_path / "b.json", start=5, stop=25, seed=4
        ).read_text()
    )
    assert [i["id"] for i in again["items"]] == [i["id"] for i in items]
