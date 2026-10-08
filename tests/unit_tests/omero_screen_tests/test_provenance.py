"""Provenance record written on each processed plate (#29)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omero_screen import default_config
from omero_screen.constants import OmeroScreenNS
from omero_screen.provenance import record_provenance, run_provenance


@pytest.fixture
def metadata() -> SimpleNamespace:
    """Plate metadata with two cell lines at 10x."""
    return SimpleNamespace(
        well_data={"cell_line": ["RPE-1", "RPE-1", "HeLa"]}, pixel_size=1.2
    )


def test_record_names_versions_models_and_stitching(metadata, monkeypatch):
    """Versions, the models per cell line, stitching and settings are recorded."""
    monkeypatch.setattr(default_config, "MODEL_DICT", {})
    monkeypatch.setattr(default_config, "MODEL_OVERRIDE", "cp3:cyto3")
    monkeypatch.setenv("OMERO_SCREEN_INFERENCE_MODEL", "micronuclei")
    record = run_provenance(metadata, stitch_mode=True)
    assert record["version omero-screen"] not in ("", "not installed")
    assert record["nucleus model"] == "cp3:cyto3"
    assert (
        record["cell model RPE-1"] == record["cell model HeLa"] == "cp3:cyto3"
    )
    assert "overlap_x" in record["stitch parameters"]
    assert record["mode"] == "stitched"
    assert record["OMERO_SCREEN_INFERENCE_MODEL"] == "micronuclei"
    assert "PASSWORD" not in " ".join(record)


def test_record_replaces_the_previous_annotation(metadata):
    """The previous provenance annotation is deleted before the new one is added."""
    conn = MagicMock()
    with (
        patch("omero_utils.map_anns.delete_map_annotations") as delete,
        patch("omero_utils.map_anns.add_map_annotations") as add,
    ):
        record_provenance(conn, 7, {"version omero-screen": "1.0"})
    plate = conn.getObject.return_value
    delete.assert_called_once_with(conn, plate, ns=OmeroScreenNS.PROVENANCE)
    add.assert_called_once_with(
        conn,
        plate,
        {"version omero-screen": "1.0"},
        ns=OmeroScreenNS.PROVENANCE,
    )
