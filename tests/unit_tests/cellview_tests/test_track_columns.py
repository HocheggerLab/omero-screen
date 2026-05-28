"""Track-column migration tests for the measurements importer.

Trackastra writes ``track_id``, ``track_id_raw``, ``parent_track_id``, and
``parent_track_id_raw`` columns to the omero-screen ``final_data.csv``.
CellView must (a) declare them in the static schema for fresh databases and
(b) ALTER them onto legacy databases that pre-date tracking.
"""

import duckdb
import pandas as pd
import pytest

from cellview.db.db import CellViewDB
from cellview.importers.measurements import _TRACK_COLUMNS, MeasurementsManager
from cellview.utils.state import CellViewStateCore


TRACK_COL_NAMES = {
    "track_id",
    "track_id_raw",
    "parent_track_id",
    "parent_track_id_raw",
}


def _measurements_columns(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(measurements)").fetchall()}


def _legacy_measurements_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a minimal measurements table missing the track columns."""
    conn.execute(
        """
        CREATE SEQUENCE measurement_id_seq;
        CREATE TABLE measurements (
            measurement_id INTEGER PRIMARY KEY DEFAULT nextval('measurement_id_seq'),
            condition_id INTEGER,
            image_id INTEGER NOT NULL,
            timepoint INTEGER NOT NULL,
            label VARCHAR NOT NULL,
            area_nucleus FLOAT,
            "centroid-0-nuc" FLOAT,
            "centroid-1-nuc" FLOAT
        );
        """
    )


def test_track_column_constant_matches_schema() -> None:
    """The migration whitelist must cover every name the importer emits."""
    assert _TRACK_COLUMNS == TRACK_COL_NAMES


def test_static_schema_includes_track_columns(tmp_path) -> None:
    """A fresh CellView database declares the track columns up front."""
    db = CellViewDB(db_path=tmp_path / "fresh.duckdb")
    db.connect()
    db.create_tables()
    assert db.conn is not None
    cols = _measurements_columns(db.conn)
    assert TRACK_COL_NAMES.issubset(cols)


class TestDynamicMigration:
    """ALTER TABLE adds track columns to a legacy measurements table."""

    @pytest.fixture
    def legacy_conn(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(":memory:")
        _legacy_measurements_table(conn)
        return conn

    @pytest.fixture
    def manager(self, legacy_conn) -> MeasurementsManager:
        state = CellViewStateCore(ui=None)  # type: ignore[arg-type]
        return MeasurementsManager(db_conn=legacy_conn, state=state)  # type: ignore[arg-type]

    def test_legacy_table_starts_without_track_columns(self, legacy_conn) -> None:
        assert TRACK_COL_NAMES.isdisjoint(_measurements_columns(legacy_conn))

    def test_dynamic_add_creates_track_columns(self, manager, legacy_conn) -> None:
        manager._ensure_dynamic_columns_exist(
            [
                "intensity_mean_DAPI_nucleus",  # already-supported branch
                "track_id",
                "track_id_raw",
                "parent_track_id",
                "parent_track_id_raw",
            ]
        )
        cols = _measurements_columns(legacy_conn)
        assert TRACK_COL_NAMES.issubset(cols)

    def test_dynamic_add_types_are_integer(self, manager, legacy_conn) -> None:
        manager._ensure_dynamic_columns_exist(list(TRACK_COL_NAMES))
        types = {
            row[1]: row[2]
            for row in legacy_conn.execute("PRAGMA table_info(measurements)").fetchall()
            if row[1] in TRACK_COL_NAMES
        }
        assert set(types) == TRACK_COL_NAMES
        assert all(t == "INTEGER" for t in types.values())

    def test_repeated_call_is_idempotent(self, manager, legacy_conn) -> None:
        """Re-running migration after a partial import must not raise."""
        manager._ensure_dynamic_columns_exist(list(TRACK_COL_NAMES))
        manager._ensure_dynamic_columns_exist(list(TRACK_COL_NAMES))
        assert TRACK_COL_NAMES.issubset(_measurements_columns(legacy_conn))


def test_set_classifier_does_not_rename_track_columns() -> None:
    """Track columns must not be caught by _set_classifier's legacy detection.

    The legacy branch renames any unprefixed column to ``classifier_<name>``;
    without the allowlist entry, ``track_id`` etc would land in the DB as
    ``classifier_track_id`` and miss the napari Tracks widget.
    """
    from cellview.utils.state import CellViewStateCore

    state = CellViewStateCore(ui=None)  # type: ignore[arg-type]
    # Skip DB writes: short-circuit by hitting the empty-classifier early return
    # while still exercising the legacy-detection branch.
    state.df = pd.DataFrame(
        {
            "label": ["1"],
            "area_nucleus": [10.0],
            "centroid-0-nuc": [5.0],
            "centroid-1-nuc": [5.0],
            "intensity_mean_DAPI_nucleus": [100.0],
            "track_id": [1],
            "track_id_raw": [1],
            "parent_track_id": [0],
            "parent_track_id_raw": [0],
        }
    )
    state.repeat_id = 1
    state.db_conn = duckdb.connect(":memory:")
    # _set_classifier returns early when no classifier cols found and the
    # allowlist covers every remaining column — which is exactly what we want.
    state._set_classifier()
    # Track columns must keep their original names — no classifier_ prefix.
    assert "track_id" in state.df.columns
    assert "parent_track_id" in state.df.columns
    assert not any(c.startswith("classifier_") for c in state.df.columns)


def test_insert_with_track_columns_round_trip(tmp_path) -> None:
    """End-to-end: dynamic migration + INSERT round-trips track values."""
    conn = duckdb.connect(":memory:")
    _legacy_measurements_table(conn)
    state = CellViewStateCore(ui=None)  # type: ignore[arg-type]
    manager = MeasurementsManager(db_conn=conn, state=state)  # type: ignore[arg-type]

    df = pd.DataFrame(
        {
            "image_id": [1, 1, 1, 1],
            "timepoint": [0, 1, 2, 2],
            "label": ["1", "2", "3", "4"],
            "area_nucleus": [10.0, 11.0, 5.0, 6.0],
            "track_id": [1, 2, 3, 4],
            "track_id_raw": [1, 2, 3, 4],
            "parent_track_id": [0, 0, 2, 2],
            "parent_track_id_raw": [0, 0, 2, 2],
        }
    )
    manager._ensure_dynamic_columns_exist(list(df.columns))
    conn.register("temp_df", df)
    conn.execute(
        'INSERT INTO measurements (image_id, timepoint, label, area_nucleus, '
        'track_id, track_id_raw, parent_track_id, parent_track_id_raw) '
        "SELECT image_id, timepoint, label, area_nucleus, track_id, "
        "track_id_raw, parent_track_id, parent_track_id_raw FROM temp_df"
    )
    rows = conn.execute(
        "SELECT track_id, parent_track_id FROM measurements ORDER BY track_id"
    ).fetchall()
    assert rows == [(1, 0), (2, 0), (3, 2), (4, 2)]
