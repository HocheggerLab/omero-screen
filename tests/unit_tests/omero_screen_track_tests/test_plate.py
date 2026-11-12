"""repair_plate reads raw track columns from CellView and writes curated ones."""

import duckdb
import pytest
from omero_screen_track.plate import NotPipFucciError, repair_plate


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
        "intensity_mean_pip_nucleus float, "
        "track_id int, track_id_raw int, parent_track_id int, parent_track_id_raw int)"
    )
    c.execute("insert into repeats values (1, 5054)")
    c.execute("insert into conditions values (1, 1, 'C2')")
    rows, mid = [], 0
    def add(tid, parent, frames, x, area):
        nonlocal mid
        for t in frames:
            mid += 1
            rows.append((mid, 1, t, area, 100.0, x, 1050.0, 50.0, 800.0, tid, tid, parent, parent))
    add(1, 0, range(10), 100.0, 400.0)
    add(2, 1, range(10, 20), 95.0, 200.0)
    add(3, 1, range(10, 12), 105.0, 200.0)
    c.executemany("insert into measurements values (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
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


def test_plate_without_pip_is_refused(conn) -> None:
    """Geminin alone is not PIP-FUCCI: no repair."""
    conn.execute("update measurements set intensity_mean_pip_nucleus = null")
    with pytest.raises(NotPipFucciError):
        repair_plate(conn, 5054)


def test_pip_column_of_another_plate_does_not_count(conn) -> None:
    """A pip column with values only for another plate leaves this one non-FUCCI."""
    conn.execute("update measurements set intensity_mean_pip_nucleus = null")
    conn.execute("insert into repeats values (2, 9999)")
    conn.execute("insert into conditions values (2, 2, 'D2')")
    conn.execute(
        "insert into measurements values (500,2,0,400,1,1,1000,50,800,1,1,0,0)"
    )
    with pytest.raises(NotPipFucciError):
        repair_plate(conn, 5054)


def test_curate_plate_replays_log_and_writes(conn, tmp_path) -> None:
    """Replay = repair from raw columns + edits; --write stores curated ids."""
    from omero_screen_track.edit import EditLog
    from omero_screen_track.plate import curate_plate, well_bases

    conn.execute("insert into measurements values (99,1,25,400,100,300,1050,50,800,9,9,0,0)")
    base = well_bases(conn, 5054)["C2"][1]
    log = EditLog(tmp_path / "edits.jsonl")
    log.append("link", "C2", {"cell": [0, 1], "frame": 25, "label": 9}, base=base)
    curated = curate_plate(conn, 5054, log, write=True)
    assert curated["C2"].tracks[(25, 9)] == 1
    assert conn.execute("select track_id from measurements where measurement_id = 99").fetchone()[0] == 1
    # Replaying again on the raw columns gives the same answer.
    assert curate_plate(conn, 5054, log)["C2"].tracks == curated["C2"].tracks


@pytest.fixture
def h2b_conn(tmp_path):
    """A non-FUCCI plate with an H2B reporter: one mitosis, one mislink."""
    c = duckdb.connect(str(tmp_path / "h2b.duckdb"))
    c.execute("create table repeats (repeat_id int, plate_id int)")
    c.execute("create table conditions (condition_id int, repeat_id int, well varchar)")
    c.execute(
        'create table measurements (measurement_id int, condition_id int, timepoint int, '
        'area_nucleus float, "centroid-0-nuc" float, "centroid-1-nuc" float, '
        "intensity_mean_H2B_nucleus float, intensity_max_H2B_nucleus float, "
        "H2B_background float, "
        "track_id int, track_id_raw int, parent_track_id int, parent_track_id_raw int)"
    )
    c.execute("insert into repeats values (1, 5061)")
    c.execute("insert into conditions values (1, 1, 'B2')")
    rows, mid = [], 0

    def add(tid, parent, frames, x, area, peak=None):
        nonlocal mid
        for t in frames:
            mid += 1
            mx = (peak or {}).get(t, 1000.0)
            rows.append((mid, 1, t, area, 100.0, x, 500.0, mx, 50.0, tid, tid, parent, parent))

    # Track 1 divides at 20 (condensed, half-size daughters 2 and 3).
    add(1, 0, range(20), 100.0, 400.0, {19: 3000.0})
    add(2, 1, range(20, 30), 80.0, 200.0, {20: 2500.0})
    add(3, 1, range(20, 30), 120.0, 200.0, {20: 2500.0})
    # Track 4 is mislinked to a whole, uncondensed neighbour (6).
    add(4, 0, range(20), 300.0, 400.0)
    add(5, 4, range(20, 30), 302.0, 400.0)
    add(6, 4, range(20, 30), 400.0, 390.0)
    c.executemany("insert into measurements values (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    yield c
    c.close()


def test_h2b_plate_is_repaired_by_condensation(h2b_conn, tmp_path) -> None:
    """Without geminin, H2B decides: the mitosis stays, the mislink goes."""
    events = tmp_path / "events.csv"
    (s,) = repair_plate(h2b_conn, 5061, events_path=events)
    assert (s.divisions_before, s.divisions_after) == (2, 1)
    parents = dict(
        h2b_conn.execute(
            "select distinct track_id_raw, parent_track_id from measurements"
        ).fetchall()
    )
    assert parents[2] == 1 and parents[3] == 1
    assert parents[6] == 0
    text = events.read_text()
    assert "mitosis" in text and "neighbour" in text


def test_h2b_curated_well_uses_the_maximum(h2b_conn) -> None:
    """The curated-lineage path loads the H2B maximum and repairs the same way."""
    from omero_screen_track.plate import curated_well

    _, curated = curated_well(h2b_conn, 5061, "B2")
    divided = {p for p in curated.parents.values() if p}
    assert len(divided) == 1
