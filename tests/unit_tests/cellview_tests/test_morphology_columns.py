"""Morphology-column handling in the measurements importer.

omero-screen can emit per-segment geometry features named
``{feature}_{segment}`` (e.g. ``solidity_nucleus``, ``area_convex_cell``,
``perimeter_cyto``). These are numeric measurements and must:

* not be misfiled as classifier outputs by ``_set_classifier``'s legacy
  (unprefixed) detection — that path is for *string* class labels only; and
* be added to the measurements table as FLOAT by the importer, so new
  regionprops features need no schema change.
"""

import duckdb
import pandas as pd
import pytest

from cellview.importers.measurements import MeasurementsManager
from cellview.utils.state import CellViewStateCore

MORPHOLOGY_COLS = [
    "area_convex_nucleus",
    "solidity_nucleus",
    "eccentricity_cell",
    "perimeter_cyto",
    "equivalent_diameter_area_nucleus",
    "axis_major_length_cell",
]


def _legacy_measurements_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a minimal measurements table without the morphology columns."""
    conn.execute(
        """
        CREATE SEQUENCE measurement_id_seq;
        CREATE TABLE measurements (
            measurement_id INTEGER PRIMARY KEY DEFAULT nextval('measurement_id_seq'),
            condition_id INTEGER,
            image_id INTEGER NOT NULL,
            timepoint INTEGER NOT NULL,
            label VARCHAR NOT NULL,
            area_nucleus FLOAT
        );
        """
    )


def _col_types(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    return {
        row[1]: row[2]
        for row in conn.execute("PRAGMA table_info(measurements)").fetchall()
    }


class TestSetClassifierIgnoresNumericMorphology:
    """``_set_classifier`` classifies by dtype, not by name."""

    def _state(self, df: pd.DataFrame) -> CellViewStateCore:
        state = CellViewStateCore(ui=None)  # type: ignore[arg-type]
        state.df = df
        state.repeat_id = 1
        state.db_conn = duckdb.connect(":memory:")
        return state

    def test_numeric_morphology_not_renamed(self) -> None:
        """Numeric geometry columns stay as-is — never become classifier_*."""
        state = self._state(
            pd.DataFrame(
                {
                    "label": ["1"],
                    "area_nucleus": [10.0],
                    "centroid-0-nuc": [5.0],
                    "centroid-1-nuc": [5.0],
                    "intensity_mean_DAPI_nucleus": [100.0],
                    "solidity_nucleus": [0.95],
                    "area_convex_cell": [120.0],
                    "perimeter_cyto": [44.0],
                    "equivalent_diameter_area_nucleus": [8.0],
                }
            )
        )
        state._set_classifier()
        assert not any(c.startswith("classifier_") for c in state.df.columns)
        for col in (
            "solidity_nucleus",
            "area_convex_cell",
            "perimeter_cyto",
            "equivalent_diameter_area_nucleus",
        ):
            assert col in state.df.columns

    def test_non_numeric_unknown_column_still_becomes_classifier(self) -> None:
        """A genuine (string) classifier output is still detected and renamed."""
        state = self._state(
            pd.DataFrame(
                {
                    "label": ["1"],
                    "area_nucleus": [10.0],
                    "centroid-0-nuc": [5.0],
                    "centroid-1-nuc": [5.0],
                    "intensity_mean_DAPI_nucleus": [100.0],
                    "solidity_nucleus": [0.95],  # numeric -> measurement
                    "mitotic": ["positive"],  # string -> classifier
                }
            )
        )
        # _set_classifier writes the model name into the repeats table.
        state.db_conn.execute(
            "CREATE TABLE repeats (repeat_id INTEGER, classifier TEXT)"
        )
        state.db_conn.execute("INSERT INTO repeats VALUES (1, NULL)")
        state._set_classifier()
        assert "classifier_mitotic" in state.df.columns
        assert "solidity_nucleus" in state.df.columns  # untouched
        assert "classifier_solidity_nucleus" not in state.df.columns


class TestDynamicMorphologyColumns:
    """``_ensure_dynamic_columns_exist`` adds numeric morphology as FLOAT."""

    @pytest.fixture
    def legacy_conn(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(":memory:")
        _legacy_measurements_table(conn)
        return conn

    @pytest.fixture
    def manager(self, legacy_conn) -> MeasurementsManager:
        state = CellViewStateCore(ui=None)  # type: ignore[arg-type]
        # The importer consults the dataframe's dtypes to classify columns.
        state.df = pd.DataFrame(
            {col: pd.Series([1.0]) for col in MORPHOLOGY_COLS}
        )
        return MeasurementsManager(db_conn=legacy_conn, state=state)  # type: ignore[arg-type]

    def test_morphology_columns_added_as_float(
        self, manager, legacy_conn
    ) -> None:
        manager._ensure_dynamic_columns_exist(MORPHOLOGY_COLS)
        types = _col_types(legacy_conn)
        for col in MORPHOLOGY_COLS:
            assert col in types, f"{col} not added to schema"
            assert types[col] == "FLOAT"

    def test_round_trip_insert(self, legacy_conn) -> None:
        """End-to-end: dynamic add + INSERT round-trips morphology values."""
        state = CellViewStateCore(ui=None)  # type: ignore[arg-type]
        df = pd.DataFrame(
            {
                "image_id": [1, 1],
                "timepoint": [0, 0],
                "label": ["1", "2"],
                "area_nucleus": [10.0, 11.0],
                "solidity_nucleus": [0.95, 0.88],
                "area_convex_cell": [120.0, 130.0],
            }
        )
        state.df = df
        manager = MeasurementsManager(db_conn=legacy_conn, state=state)  # type: ignore[arg-type]
        manager._ensure_dynamic_columns_exist(list(df.columns))
        legacy_conn.register("temp_df", df)
        legacy_conn.execute(
            "INSERT INTO measurements "
            "(image_id, timepoint, label, area_nucleus, solidity_nucleus, area_convex_cell) "
            "SELECT image_id, timepoint, label, area_nucleus, solidity_nucleus, area_convex_cell "
            "FROM temp_df"
        )
        rows = legacy_conn.execute(
            "SELECT solidity_nucleus, area_convex_cell FROM measurements ORDER BY label"
        ).fetchall()
        assert rows == [(pytest.approx(0.95), 120.0), (pytest.approx(0.88), 130.0)]
