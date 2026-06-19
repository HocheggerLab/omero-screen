"""Round-trip tests for the DAPI → nucleus_channel rehydration on export.

Verifies that the pandas and polars exporters rename the canonical
``*_DAPI_*`` measurement columns back to the actual fluorophore name
stored on ``repeats.nucleus_channel``.
"""

from __future__ import annotations

import pytest
from cellview.exporters.db_to_pandas import export_pandas_df
from cellview.exporters.db_to_polars import export_polars_lf


def _seed_plate(
    db,
    plate_id: int,
    nucleus_channel: str,
    channel_0: str | None = None,
) -> None:
    """Seed a minimal plate with one measurement row.

    The canonical ``*_DAPI_*`` measurement columns are populated regardless
    of fluorophore — that is the on-disk contract the exporter rehydrates.
    """
    conn = db.connect()
    project_id = conn.execute(
        "SELECT nextval('project_id_seq')"
    ).fetchone()[0]
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
    repeat_id = conn.execute(
        "SELECT nextval('repeat_id_seq')"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO repeats "
        "(repeat_id, experiment_id, plate_id, date, channel_0, nucleus_channel) "
        "VALUES (?, ?, ?, CURRENT_DATE, ?, ?)",
        [
            repeat_id,
            experiment_id,
            plate_id,
            channel_0 or nucleus_channel,
            nucleus_channel,
        ],
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
    measurement_id = conn.execute(
        "SELECT nextval('measurement_id_seq')"
    ).fetchone()[0]
    conn.execute(
        'INSERT INTO measurements '
        '(measurement_id, condition_id, image_id, timepoint, label, '
        'area_nucleus, "centroid-0-nuc", "centroid-1-nuc", '
        'intensity_min_DAPI_nucleus, intensity_mean_DAPI_nucleus, '
        'intensity_max_DAPI_nucleus, integrated_int_DAPI_norm) '
        'VALUES (?, ?, 1, 1, \'cell_1\', 100.5, 10.0, 20.0, '
        '150.5, 200.5, 250.5, 2.0)',
        [measurement_id, condition_id],
    )


class TestPandasExporterRehydration:
    def test_dapi_plate_exports_unchanged(self, db):
        _seed_plate(db, plate_id=101, nucleus_channel="DAPI")
        df, _ = export_pandas_df(101, db.conn)
        cols = set(df.columns)
        assert "intensity_mean_DAPI_nucleus" in cols
        assert "integrated_int_DAPI_norm" in cols
        # No Hoechst columns leaked in.
        assert not any("Hoechst" in c for c in cols)

    def test_hoechst_plate_rehydrates(self, db):
        _seed_plate(db, plate_id=202, nucleus_channel="Hoechst")
        df, _ = export_pandas_df(202, db.conn)
        cols = set(df.columns)
        assert "intensity_mean_Hoechst_nucleus" in cols
        assert "intensity_min_Hoechst_nucleus" in cols
        assert "intensity_max_Hoechst_nucleus" in cols
        assert "integrated_int_Hoechst_norm" in cols
        # Canonical DAPI columns are renamed away.
        assert "intensity_mean_DAPI_nucleus" not in cols
        assert "integrated_int_DAPI_norm" not in cols
        # nucleus_channel column still present on the export for traceability.
        assert "nucleus_channel" in cols
        assert (df["nucleus_channel"] == "Hoechst").all()

    def test_h2b_rfp_underscore_channel(self, db):
        _seed_plate(db, plate_id=303, nucleus_channel="H2B_RFP")
        df, _ = export_pandas_df(303, db.conn)
        assert "intensity_mean_H2B_RFP_nucleus" in df.columns
        assert "intensity_mean_DAPI_nucleus" not in df.columns


class TestPolarsExporterRehydration:
    def test_dapi_plate_exports_unchanged(self, db):
        _seed_plate(db, plate_id=104, nucleus_channel="DAPI")
        lf, _ = export_polars_lf(104, db.conn)
        df = lf.collect()
        cols = set(df.columns)
        assert "intensity_mean_DAPI_nucleus" in cols
        assert "integrated_int_DAPI_norm" in cols

    def test_hoechst_plate_rehydrates(self, db):
        _seed_plate(db, plate_id=205, nucleus_channel="Hoechst")
        lf, _ = export_polars_lf(205, db.conn)
        df = lf.collect()
        cols = set(df.columns)
        assert "intensity_mean_Hoechst_nucleus" in cols
        assert "integrated_int_Hoechst_norm" in cols
        assert "intensity_mean_DAPI_nucleus" not in cols


@pytest.mark.parametrize("channel", ["DAPI", "Hoechst", "H2B_RFP"])
def test_pandas_and_polars_agree(db, channel):
    """Pandas and Polars exporters produce the same nucleus column names."""
    plate_id = 999
    _seed_plate(db, plate_id=plate_id, nucleus_channel=channel)
    pd_df, _ = export_pandas_df(plate_id, db.conn)
    pl_df, _ = export_polars_lf(plate_id, db.conn)
    pl_cols = set(pl_df.collect().columns)
    expected_nucleus = f"intensity_mean_{channel}_nucleus"
    assert expected_nucleus in pd_df.columns
    assert expected_nucleus in pl_cols
