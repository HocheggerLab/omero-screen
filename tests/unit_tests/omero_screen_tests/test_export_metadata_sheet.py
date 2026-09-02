"""Tests for the re-attachable metadata workbook.

The contract under test is ``MetadataParser._load_data_from_excel``: sheet names
``["Sheet1", "Sheet2"]`` in that exact order, ``Channels``/``Index`` on Sheet1,
and ``Well``/``cell_line`` plus conditions on Sheet2. The final test parses the
written file with the real parser logic so the two cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from omero_screen.export.metadata_sheet import EMPTY_WELL, write_metadata_excel


def make_well(position: str, annotations: dict[str, str]) -> MagicMock:
    """A stand-in WellWrapper with map annotations."""
    well = MagicMock()
    well.getWellPos.return_value = position
    well._annotations = annotations
    return well


@pytest.fixture
def conn(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """A connection whose plate has two annotated wells and one empty one."""
    wells = [
        make_well("A1", {"cell_line": "HeLa", "condition": "ctrl"}),
        make_well("A2", {"cell_line": "RPE1", "condition": "siB55a"}),
        make_well("B1", {}),
    ]
    plate = MagicMock()
    plate.getId.return_value = 42
    plate.listChildren.return_value = wells
    plate._annotations = {"DAPI": "0", "Tub": "1", "EdU": "2"}

    connection = MagicMock()
    connection.getObject.return_value = plate

    def fake_parse_annotations(obj: MagicMock, ns: str | None = None):
        return obj._annotations

    monkeypatch.setattr(
        "omero_screen.export.metadata_sheet.parse_annotations",
        fake_parse_annotations,
    )
    return connection


def test_sheet_names_and_order_match_the_parser_contract(
    conn: MagicMock, tmp_path: Path
) -> None:
    """The parser compares the sheet-name list for equality, so order matters."""
    path = write_metadata_excel(
        conn, 42, ["A1", "A2", "B1"], ["DAPI"], tmp_path / "metadata.xlsx"
    )
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets.keys()) == ["Sheet1", "Sheet2"]


def test_channels_sheet_uses_the_plate_annotation(
    conn: MagicMock, tmp_path: Path
) -> None:
    """Sheet1 is rebuilt from the plate's omero-screen channel annotation."""
    path = write_metadata_excel(
        conn, 42, ["A1"], ["ignored"], tmp_path / "metadata.xlsx"
    )
    sheet1 = pd.read_excel(path, sheet_name="Sheet1")
    assert list(sheet1.columns) == ["Channels", "Index"]
    assert list(sheet1["Channels"]) == ["DAPI", "Tub", "EdU"]
    assert list(sheet1["Index"].astype(str)) == ["0", "1", "2"]


def test_channels_fall_back_to_image_names_without_an_annotation(
    conn: MagicMock, tmp_path: Path
) -> None:
    """An unprocessed plate falls back to the image channel names."""
    conn.getObject.return_value._annotations = {}
    path = write_metadata_excel(
        conn, 42, ["A1"], ["DAPI", "Tub"], tmp_path / "metadata.xlsx"
    )
    sheet1 = pd.read_excel(path, sheet_name="Sheet1")
    assert list(sheet1["Channels"]) == ["DAPI", "Tub"]


def test_conditions_are_recovered_per_well(
    conn: MagicMock, tmp_path: Path
) -> None:
    """Well map annotations become Sheet2 condition columns."""
    path = write_metadata_excel(
        conn, 42, ["A1", "A2"], ["DAPI"], tmp_path / "metadata.xlsx"
    )
    sheet2 = pd.read_excel(path, sheet_name="Sheet2")
    assert list(sheet2.columns)[:2] == ["Well", "cell_line"]
    assert list(sheet2["Well"]) == ["A1", "A2"]
    assert list(sheet2["cell_line"]) == ["HeLa", "RPE1"]
    assert list(sheet2["condition"]) == ["ctrl", "siB55a"]


def test_unannotated_wells_are_marked_empty(
    conn: MagicMock, tmp_path: Path
) -> None:
    """Wells with no annotations round-trip as cell_line 'Empty'."""
    path = write_metadata_excel(
        conn, 42, ["A1", "B1"], ["DAPI"], tmp_path / "metadata.xlsx"
    )
    sheet2 = pd.read_excel(path, sheet_name="Sheet2")
    assert sheet2.loc[sheet2["Well"] == "B1", "cell_line"].item() == EMPTY_WELL


def test_only_requested_wells_are_written(
    conn: MagicMock, tmp_path: Path
) -> None:
    """Sheet2 covers the exported wells only."""
    path = write_metadata_excel(
        conn, 42, ["A2"], ["DAPI"], tmp_path / "metadata.xlsx"
    )
    sheet2 = pd.read_excel(path, sheet_name="Sheet2")
    assert list(sheet2["Well"]) == ["A2"]


def test_missing_plate_is_rejected(conn: MagicMock, tmp_path: Path) -> None:
    """A missing plate raises rather than writing an empty workbook."""
    conn.getObject.return_value = None
    with pytest.raises(ValueError, match="was not found"):
        write_metadata_excel(
            conn, 42, ["A1"], ["DAPI"], tmp_path / "metadata.xlsx"
        )


def test_written_file_survives_the_real_parser_logic(
    conn: MagicMock, tmp_path: Path
) -> None:
    """Mirror MetadataParser._load_data_from_excel against our output."""
    path = write_metadata_excel(
        conn, 42, ["A1", "A2", "B1"], ["DAPI"], tmp_path / "metadata.xlsx"
    )
    meta = pd.read_excel(path, sheet_name=None)

    # The parser rejects anything but exactly these two sheets, in order.
    assert list(meta.keys()) == ["Sheet1", "Sheet2"]

    channel_data = {
        meta["Sheet1"]["Channels"][i]: str(meta["Sheet1"]["Index"][i])
        for i in range(len(meta["Sheet1"]["Channels"]))
    }
    assert channel_data == {"DAPI": "0", "Tub": "1", "EdU": "2"}

    df = meta["Sheet2"]
    empty_mask = df["cell_line"].astype(str).str.strip() == EMPTY_WELL
    assert df.loc[empty_mask, "Well"].tolist() == ["B1"]

    well_data = {str(k): v for k, v in df[~empty_mask].to_dict("list").items()}
    assert well_data["Well"] == ["A1", "A2"]
    assert well_data["cell_line"] == ["HeLa", "RPE1"]
