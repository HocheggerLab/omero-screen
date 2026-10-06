"""repair_plate reads raw track columns from CellView and writes curated ones."""

import duckdb
import pytest
from cellview.tracks.plate import MarkerNotFoundError, repair_plate


@pytest.fixture
def conn(tmp_path):
    """A minimal CellView-shaped database with one fragmented nucleus."""
    c = duckdb.connect(str(tmp_path / "cv.duckdb"))
    c.execute("create table repeats (repeat_id int, plate_id int)")
    c.execute("create table conditions (condition_id int, repeat_id int, well varchar)")
    c.execute(
        'create table measurements (measurement_id int, condition_id int, timepoint int, '
        'area_nucleus float, "centroid-0-nuc" float, "centroid-1-nuc" float, '
        "intensity_mean_Geminin_nucleus float, Geminin_background float, "
        "track_id int, track_id_raw int, parent_track_id int, parent_track_id_raw int)"
    )
    c.execute("insert into repeats values (1, 5054)")
    c.execute("insert into conditions values (1, 1, 'C2')")
    rows, mid = [], 0
    def add(tid, parent, frames, x, area):
        nonlocal mid
        for t in frames:
            mid += 1
            rows.append((mid, 1, t, area, 100.0, x, 1050.0, 50.0, tid, tid, parent, parent))
    add(1, 0, range(10), 100.0, 400.0)
    add(2, 1, range(10, 20), 95.0, 200.0)
    add(3, 1, range(10, 12), 105.0, 200.0)
    c.executemany("insert into measurements values (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    yield c
    c.close()


def test_dry_run_reports_without_writing(conn) -> None:
    """A dry run counts the change but leaves track_id untouched."""
    (s,) = repair_plate(conn, 5054, dry_run=True)
    assert (s.tracks_before, s.tracks_after, s.divisions_before, s.divisions_after) == (3, 1, 1, 0)
    assert conn.execute("select count(distinct track_id) from measurements").fetchone()[0] == 3


def test_repair_writes_curated_columns_and_keeps_raw(conn, tmp_path) -> None:
    """Curated ids are written; raw ids and the events CSV are kept."""
    events = tmp_path / "events.csv"
    repair_plate(conn, 5054, events_path=events)
    assert conn.execute("select distinct track_id from measurements").fetchall() == [(1,)]
    assert conn.execute("select max(parent_track_id) from measurements").fetchone()[0] == 0
    assert conn.execute("select count(distinct track_id_raw) from measurements").fetchone()[0] == 3
    assert "fragment" in events.read_text()


def test_plate_without_marker_is_refused(conn) -> None:
    """No geminin channel, no repair."""
    with pytest.raises(MarkerNotFoundError):
        repair_plate(conn, 5054, marker="Cdt1")


def test_curate_plate_replays_log_and_writes(conn, tmp_path) -> None:
    """Replay = repair from raw columns + edits; --write stores curated ids."""
    from cellview.tracks.edit import EditLog
    from cellview.tracks.plate import curate_plate, well_bases

    conn.execute("insert into measurements values (99,1,25,400,100,300,1050,50,9,9,0,0)")
    base = well_bases(conn, 5054)["C2"][1]
    log = EditLog(tmp_path / "edits.jsonl")
    log.append("link", "C2", {"cell": [0, 1], "frame": 25, "label": 9}, base=base)
    curated = curate_plate(conn, 5054, log, write=True)
    assert curated["C2"].tracks[(25, 9)] == 1
    assert conn.execute("select track_id from measurements where measurement_id = 99").fetchone()[0] == 1
    # Replaying again on the raw columns gives the same answer.
    assert curate_plate(conn, 5054, log)["C2"].tracks == curated["C2"].tracks
