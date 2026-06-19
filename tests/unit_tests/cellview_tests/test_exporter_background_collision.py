"""Regression tests for ``*_background`` column collisions on export.

Background columns are per-image measurement readouts and live in the
``measurements`` table. On older imports they could also be misclassified as
per-well condition variables and stored in ``condition_variables`` too. When
that happens the exporters' measurements ⋈ conditions merge disambiguates the
duplicate column with ``_x``/``_y`` (pandas) or ``_right`` (polars) suffixes.

These tests cover both halves of the fix:

* the export-side guard drops the condition-side duplicate so already-polluted
  DBs export cleanly, and
* the import-side ``_identify_variable_columns`` no longer classifies
  ``*_background`` columns as condition variables in the first place.
"""

from __future__ import annotations

import pandas as pd
import pytest
from cellview.exporters.db_to_pandas import export_pandas_df
from cellview.exporters.db_to_polars import export_polars_lf
from cellview.importers.conditions import ConditionManager
from cellview.utils.state import CellViewStateCore


def _seed_polluted_plate(db, plate_id: int) -> None:
    """Seed a plate where ``DAPI_background`` lives in BOTH tables.

    Mimics the on-disk state of a DB imported before the
    ``_identify_variable_columns`` fix: the background value sits in the
    measurements table (the source of truth) and is *also* duplicated into
    condition_variables as a per-well constant.
    """
    conn = db.connect()
    project_id = conn.execute("SELECT nextval('project_id_seq')").fetchone()[0]
    conn.execute(
        "INSERT INTO projects (project_id, project_name) VALUES (?, ?)",
        [project_id, f"proj_{plate_id}"],
    )
    experiment_id = conn.execute(
        "SELECT nextval('experiment_id_seq')"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO experiments (experiment_id, project_id, experiment_name) "
        "VALUES (?, ?, ?)",
        [experiment_id, project_id, f"exp_{plate_id}"],
    )
    repeat_id = conn.execute("SELECT nextval('repeat_id_seq')").fetchone()[0]
    conn.execute(
        "INSERT INTO repeats "
        "(repeat_id, experiment_id, plate_id, date, channel_0, nucleus_channel) "
        "VALUES (?, ?, ?, CURRENT_DATE, 'DAPI', 'DAPI')",
        [repeat_id, experiment_id, plate_id],
    )
    condition_id = conn.execute(
        "SELECT nextval('condition_id_seq')"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO conditions "
        "(condition_id, repeat_id, well, well_id, cell_line) "
        "VALUES (?, ?, 'A1', 'A1', 'HeLa')",
        [condition_id, repeat_id],
    )

    # Background column is added dynamically on real imports; do the same here.
    conn.execute(
        'ALTER TABLE measurements ADD COLUMN "DAPI_background" FLOAT'
    )
    measurement_id = conn.execute(
        "SELECT nextval('measurement_id_seq')"
    ).fetchone()[0]
    conn.execute(
        'INSERT INTO measurements '
        '(measurement_id, condition_id, image_id, timepoint, label, '
        'area_nucleus, "centroid-0-nuc", "centroid-1-nuc", '
        'intensity_min_DAPI_nucleus, intensity_mean_DAPI_nucleus, '
        'intensity_max_DAPI_nucleus, integrated_int_DAPI_norm, '
        '"DAPI_background") '
        'VALUES (?, ?, 1, 1, \'cell_1\', 100.5, 10.0, 20.0, '
        '150.5, 200.5, 250.5, 2.0, 42.0)',
        [measurement_id, condition_id],
    )

    # The pollution: same column duplicated into condition_variables.
    conn.execute(
        "INSERT INTO condition_variables "
        "(condition_id, variable_name, variable_value) VALUES (?, ?, ?)",
        [condition_id, "DAPI_background", "42.0"],
    )


class TestExportGuardDropsBackgroundDuplicate:
    """The export guard fixes already-polluted DBs without a migration."""

    def test_pandas_no_suffix_collision(self, db):
        """Pandas export drops the condition-side background duplicate."""
        _seed_polluted_plate(db, plate_id=501)
        df, variable_names = export_pandas_df(501, db.conn)
        cols = list(df.columns)
        # No merge-suffix artifacts anywhere.
        assert not any(c.endswith(("_x", "_y")) for c in cols), cols
        # Background survives exactly once, from the measurements table.
        assert cols.count("DAPI_background") == 1
        assert df["DAPI_background"].iloc[0] == pytest.approx(42.0)
        # And it is not advertised as a condition variable.
        assert "DAPI_background" not in variable_names

    def test_polars_no_suffix_collision(self, db):
        """Polars export drops the condition-side background duplicate."""
        _seed_polluted_plate(db, plate_id=502)
        lf, variable_names = export_polars_lf(502, db.conn)
        cols = lf.collect().columns
        assert not any(c.endswith(("_right",)) for c in cols), cols
        assert cols.count("DAPI_background") == 1
        assert "DAPI_background" not in variable_names

    def test_pandas_and_polars_handle_background_identically(self, db):
        """Both exporters surface background once, suffix-free."""
        # Both exporters must surface background exactly once and free of
        # collision suffixes. (Full column-set equality is not expected: the
        # pandas exporter additionally drops all-null columns.)
        _seed_polluted_plate(db, plate_id=503)
        pd_cols = list(export_pandas_df(503, db.conn)[0].columns)
        pl_cols = list(export_polars_lf(503, db.conn)[0].collect().columns)
        assert pd_cols.count("DAPI_background") == 1
        assert pl_cols.count("DAPI_background") == 1
        assert not any(c.endswith(("_x", "_y", "_right")) for c in pd_cols)
        assert not any(c.endswith(("_x", "_y", "_right")) for c in pl_cols)


class TestImportExcludesBackground:
    """Root-cause fix: background is never stored as a condition variable."""

    def _make_manager(self, df: pd.DataFrame) -> ConditionManager:
        state = CellViewStateCore(ui=None)  # type: ignore[arg-type]
        state.df = df
        # db_conn is unused by _identify_variable_columns.
        return ConditionManager(db_conn=None, state=state)  # type: ignore[arg-type]

    def test_background_not_classified_as_variable(self):
        """``*_background`` columns are excluded from condition variables."""
        # Two wells, single image each → background is per-well constant and
        # would otherwise be picked up as a condition variable.
        df = pd.DataFrame(
            {
                "well": ["A1", "A2"],
                "well_id": [1, 2],
                "image_id": [1, 2],
                "cell_line": ["HeLa", "HeLa"],
                "sirna": ["ctrl", "p53"],
                "DAPI_background": [42.0, 37.0],
                "EdU_background": [5.0, 6.0],
            }
        )
        variable_cols = self._make_manager(df)._identify_variable_columns()
        assert "DAPI_background" not in variable_cols
        assert "EdU_background" not in variable_cols
        # Genuine experimental condition is still detected.
        assert "sirna" in variable_cols
