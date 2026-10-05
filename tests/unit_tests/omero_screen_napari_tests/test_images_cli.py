"""The ``omero-screen-images`` CLI: option parsing, cell selection, end to end."""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import patch

import click
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import pytest
from click.testing import CliRunner

from omero_screen_napari.images_cli import (
    _parse_grid,
    _parse_limits,
    _resolve_wells,
    _restrict_to_cells,
    cli,
)

PLATE_ROWS = pl.DataFrame(
    {
        "well": ["A1", "A1", "B2", "B2"],
        "image_id": [10, 10, 20, 20],
        "label": [1, 2, 1, 2],
        "timepoint": [0, 0, 0, 0],
        "classifier_nuclei4": ["normal", "micronuclei", "normal", "normal"],
    }
)


# ---------------------------------------------------------------------- #
# Pure helpers                                                           #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("grid", "expected"), [("5x5", (5, 5)), ("2X3", (2, 3)), ("4×1", (4, 1))]
)
def test_parse_grid(grid, expected):
    assert _parse_grid(grid) == expected


@pytest.mark.parametrize("grid", ["5", "0x3", "axb", "2x3x4"])
def test_parse_grid_rejects(grid):
    with pytest.raises(click.BadParameter):
        _parse_grid(grid)


def test_parse_limits():
    assert _parse_limits(("DAPI=200:12000", "NHS ester=0:500")) == {
        "DAPI": (200, 12000),
        "NHS ester": (0, 500),
    }


@pytest.mark.parametrize("spec", ["DAPI", "DAPI=1", "=1:2", "DAPI=5:5", "DAPI=a:b"])
def test_parse_limits_rejects(spec):
    with pytest.raises(click.BadParameter):
        _parse_limits((spec,))


def test_restrict_to_cells_matches_keys_as_strings():
    # A CSV round trip reads labels as strings; CellView has integers.
    cells = pl.DataFrame({"image_id": ["10", "20"], "label": ["2", "1"]})
    kept = _restrict_to_cells(PLATE_ROWS.lazy(), cells).collect()
    assert kept.select("image_id", "label").rows() == [(10, 2), (20, 1)]
    assert kept.columns == PLATE_ROWS.columns


def test_restrict_to_cells_uses_timepoint_when_given():
    rows = PLATE_ROWS.with_columns(pl.Series("timepoint", [0, 1, 0, 1]))
    cells = pl.DataFrame({"image_id": [10, 10], "label": [1, 2], "timepoint": [0, 0]})
    kept = _restrict_to_cells(rows.lazy(), cells).collect()
    assert kept["label"].to_list() == [1]


def test_restrict_to_cells_without_selection_is_a_no_op():
    lf = PLATE_ROWS.lazy()
    assert _restrict_to_cells(lf, None) is lf


def test_resolve_wells():
    assert _resolve_wells("b2, a1", ["A1", "B2"], None, 1) == ["B2", "A1"]
    assert _resolve_wells("All", ["A1", "B2"], None, 1) == ["A1", "B2"]
    cells = pl.DataFrame({"image_id": [1], "label": [1]})
    assert _resolve_wells(None, ["B2"], cells, 1) == ["B2"]
    with pytest.raises(click.UsageError):
        _resolve_wells(None, ["A1"], None, 1)
    with pytest.raises(click.BadParameter, match="C3"):
        _resolve_wells("A1,C3", ["A1", "B2"], None, 1)


# ---------------------------------------------------------------------- #
# End to end, with the loaders and the renderer mocked                   #
# ---------------------------------------------------------------------- #


@pytest.fixture
def harness():
    """Patch CellView, the well loader and the renderer; record calls."""
    record: dict = {"loads": [], "builds": []}

    def fake_load(plate_id, wells, *, omero_data, timepoint, connection):
        record["loads"].append(list(wells))
        omero_data.plate_id = plate_id
        omero_data.plate_name = "plate"
        omero_data.channel_data = {"DAPI": "0", "Tub": "1"}
        omero_data.intensities = {0: (0, 1000), 1: (0, 2000)}
        omero_data.pixel_size = (1.2, 1.2)
        omero_data.well_pos_list = list(wells)
        omero_data.plate_data = PLATE_ROWS.lazy()
        return omero_data

    def fake_build(od, well_settings, **_kwargs):
        rows = od.plate_data.filter(pl.col("well") == well_settings.well).collect()
        record["builds"].append((well_settings, rows, dict(od.intensities)))
        print("library noise on stdout")
        od.selected_images = [0] * min(rows.height, 4)
        od.cropped_images = []
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.plot([0, 1])
        return fig

    with (
        patch(
            "omero_screen_napari.plate_cache._load_plate_data_from_cellview",
            return_value=PLATE_ROWS.lazy(),
        ),
        patch(
            "omero_screen_napari.well_context.well_source",
            return_value=record.setdefault("source", "fields"),
        ) as source,
        patch(
            "omero_screen_napari.well_context.load_well_context",
            side_effect=fake_load,
        ),
        patch(
            "omero_screen_napari.gallery_export.build_gallery_figure",
            side_effect=fake_build,
        ),
    ):
        record["source_patch"] = source
        yield record


def _run(*args):
    return CliRunner().invoke(cli, ["gallery", *args], catch_exceptions=False)


def test_gallery_json_manifest_is_clean_stdout(harness, tmp_path):
    result = _run(
        "1", "--wells", "A1,B2", "--channels", "DAPI", "--grid", "2x2",
        "--classifier-column", "classifier_nuclei4", "--class", "micronuclei",
        "--seed", "4", "--out", str(tmp_path), "--fmt", "png", "--json",
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads(result.stdout)  # only the manifest on stdout
    assert manifest["command"] == "gallery"
    assert manifest["source"] == "fields"
    assert manifest["seed"] == 4
    settings = manifest["settings"]
    assert settings["classifier_column"] == "classifier_nuclei4"
    assert settings["classifier_filter"] == "micronuclei"
    assert (settings["rows"], settings["columns"]) == (2, 2)
    assert settings["channels"] == ["DAPI"]
    assert set(manifest["wells"]) == {"A1", "B2"}
    assert (tmp_path / "A1.png").exists()


def test_field_plate_loads_one_well_at_a_time(harness, tmp_path):
    result = _run("1", "--wells", "A1,B2", "--channels", "DAPI", "--out", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert harness["loads"] == [["A1"], ["B2"]]


def test_zarr_plate_loads_all_wells_once(harness, tmp_path):
    harness["source_patch"].return_value = "zarr"
    result = _run("1", "--wells", "A1,B2", "--channels", "DAPI", "--out", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert harness["loads"] == [["A1", "B2"]]


def test_limits_override_loaded_intensities(harness, tmp_path):
    result = _run(
        "1", "--wells", "A1", "--channels", "DAPI,Tub",
        "--limits", "Tub=5:50", "--out", str(tmp_path), "--json",
    )
    assert result.exit_code == 0, result.output
    assert harness["builds"][0][2] == {0: (0, 1000), 1: (5, 50)}
    assert json.loads(result.stdout)["intensities"]["1"] == [5, 50]


def test_cells_file_restricts_rows_and_derives_wells(harness, tmp_path):
    cells = tmp_path / "cells.csv"
    pl.DataFrame({"image_id": [20], "label": [2]}).write_csv(cells)
    result = _run(
        "1", "--cells", str(cells), "--channels", "DAPI",
        "--out", str(tmp_path / "out"), "--json",
    )
    assert result.exit_code == 0, result.output
    assert harness["loads"] == [["B2"]]
    rows = harness["builds"][0][1]
    assert rows.select("image_id", "label").rows() == [(20, 2)]
    assert json.loads(result.stdout)["cells_file"] == str(cells)


def test_unknown_channel_is_a_usage_error(harness, tmp_path):
    result = _run("1", "--wells", "A1", "--channels", "GFP", "--out", str(tmp_path))
    assert result.exit_code == 2
    assert "GFP" in result.output
    assert "DAPI" in result.output


def test_class_needs_a_column(harness, tmp_path):
    result = _run(
        "1", "--wells", "A1", "--channels", "DAPI", "--class", "micronuclei",
        "--out", str(tmp_path),
    )
    assert result.exit_code == 2
    assert "--classifier-column" in result.output


def test_no_gallery_written_exits_nonzero(harness, tmp_path):
    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        return_value=None,
    ):
        result = _run("1", "--wells", "A1", "--channels", "DAPI", "--out", str(tmp_path))
    assert result.exit_code == 1


# ---------------------------------------------------------------------- #
# Import weight                                                          #
# ---------------------------------------------------------------------- #


def test_render_path_imports_no_napari_or_qt():
    """The CLI and every module it renders with stay napari- and Qt-free."""
    code = (
        "import sys\n"
        "import omero_screen_napari.images_cli\n"
        "import omero_screen_napari.gallery_export\n"
        "import omero_screen_napari.well_context\n"
        "import omero_screen_napari.plate_cache\n"
        "import omero_screen_napari.zarr_cache.display\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if m.split('.')[0] in {'napari', 'qtpy', 'PyQt5', 'PyQt6',\n"
        "                                    'PySide2', 'PySide6', 'magicgui'})\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""
