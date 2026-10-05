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
import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from omero_screen_napari.images_cli import (
    _parse_grid,
    _read_plan,
    _slug,
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
        # Fields (N, Y, X, C): DAPI 100..400, Tub 1000..4000.
        omero_data.images = np.stack(
            [
                np.linspace(100, 400, 64).reshape(1, 8, 8),
                np.linspace(1000, 4000, 64).reshape(1, 8, 8),
            ],
            axis=-1,
        )
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
    # Sampling pass for the shared limits (last to first, leaving A1
    # loaded), then B2 again to render it.
    assert harness["loads"] == [["B2"], ["A1"], ["B2"]]


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
    # DAPI pooled from the fields (not the loaded 0-1000), Tub as given.
    dapi_lo, dapi_hi = harness["builds"][0][2][0]
    assert 100 <= dapi_lo < 110 and 390 < dapi_hi <= 400
    assert harness["builds"][0][2][1] == (5, 50)
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


def test_background_is_kept_by_default(harness, tmp_path):
    _run("1", "--wells", "A1", "--channels", "DAPI", "--out", str(tmp_path))
    assert harness["builds"][0][0].no_background is False
    _run(
        "1", "--wells", "A1", "--channels", "DAPI", "--blank-background",
        "--out", str(tmp_path),
    )
    assert harness["builds"][1][0].no_background is True


# ---------------------------------------------------------------------- #
# well                                                                   #
# ---------------------------------------------------------------------- #


@pytest.fixture
def well_harness():
    """A zarr plate with two cached wells; render_wells mocked."""
    calls: dict = {}

    def fake_render(wells, load_well, settings, out_dir):
        calls["wells"] = list(wells)
        calls["settings"] = settings
        calls["inputs"] = [load_well(w) for w in wells]
        return {
            "limits": {"DAPI": [1, 2]},
            "wells": {w: {"exported": True, "file": f"{w}.png"} for w in wells},
        }

    with (
        patch(
            "omero_screen_napari.well_context.well_source", return_value="zarr"
        ),
        patch(
            "omero_screen_napari.zarr_cache.plate_info",
            return_value={
                "channel_names": ["DAPI", "Tub"],
                "plate_name": "p",
                "pixel_size_um": 1.2,
                "well_metadata": {"A1": {"cell_line": "RPE-1"}},
            },
        ),
        patch(
            "omero_screen_napari.zarr_cache.cached_wells",
            return_value=["A1", "B2"],
        ),
        patch(
            "omero_screen_napari.well_overview.zarr_well_input",
            side_effect=lambda p, w, info, t: __import__(
                "omero_screen_napari.well_overview", fromlist=["WellInput"]
            ).WellInput(w, None, f"cap {w}", 1.2),
        ),
        patch(
            "omero_screen_napari.well_overview.render_wells",
            side_effect=fake_render,
        ),
    ):
        yield calls


def _run_well(*args):
    return CliRunner().invoke(cli, ["well", *args], catch_exceptions=False)


def test_well_defaults_to_all_channels_whole_well(well_harness, tmp_path):
    result = _run_well("1", "--wells", "All", "--out", str(tmp_path), "--json")
    assert result.exit_code == 0, result.output
    settings = well_harness["settings"]
    assert well_harness["wells"] == ["A1", "B2"]
    assert settings.layers == ["DAPI", "Tub"]
    assert (settings.zoom, settings.center) == (1, (0.5, 0.5))
    manifest = json.loads(result.stdout)
    assert manifest["command"] == "well"
    assert manifest["source"] == "zarr"
    assert (tmp_path / "well_overview.json").exists()


def test_well_options_reach_the_renderer(well_harness, tmp_path):
    result = _run_well(
        "1", "--wells", "a1", "--layers", "DAPI,nuclei_masks", "--zoom", "4",
        "--center", "0.25,0.75", "--limits", "DAPI=10:20", "--no-caption",
        "--out", str(tmp_path),
    )
    assert result.exit_code == 0, result.output
    settings = well_harness["settings"]
    assert settings.layers == ["DAPI", "nuclei_masks"]
    assert (settings.zoom, settings.center) == (4, (0.25, 0.75))
    assert settings.limits == {"DAPI": (10, 20)}
    assert well_harness["inputs"][0].caption is None


@pytest.mark.parametrize(
    "args",
    [
        ("--wells", "C3"),
        ("--wells", "A1", "--zoom", "3"),
        ("--wells", "A1", "--center", "2,0"),
        ("--wells", "A1", "--limits", "GFP=1:2"),
    ],
)
def test_well_rejects_bad_options(well_harness, tmp_path, args):
    result = _run_well("1", *args, "--out", str(tmp_path))
    assert result.exit_code == 2


# ---------------------------------------------------------------------- #
# batch                                                                  #
# ---------------------------------------------------------------------- #


def _plan(tmp_path, text):
    path = tmp_path / "plan.csv"
    path.write_text(text)
    return path


def test_read_plan_groups_by_plate_and_render(tmp_path):
    plan = _plan(
        tmp_path,
        "plate_id,well,render,note\n"
        "5108,b2,well,x\n5108,G5,Well,\n5108,G5,gallery:micronuclei,\n"
        "5108,G5,well,dup\n42,A1,gallery,\n",
    )
    assert _read_plan(plan) == {
        (5108, "well"): ["B2", "G5"],
        (5108, "gallery:micronuclei"): ["G5"],
        (42, "gallery"): ["A1"],
    }


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("plate,well,render\n1,A1,well\n", "missing column"),
        ("plate_id,well,render\nx,A1,well\n", "not a number"),
        ("plate_id,well,render\n1,A1,montage\n", "montage"),
        ("plate_id,well,render\n1,A1,gallery:\n", "class name"),
        ("plate_id,well,render\n1,,well\n", "no well"),
        ("plate_id,well,render\n", "no rows"),
    ],
)
def test_read_plan_rejects(tmp_path, text, match):
    with pytest.raises(click.BadParameter, match=match):
        _read_plan(_plan(tmp_path, text))


def test_slug():
    assert _slug("gallery:micro nuclei/x") == "gallery_micro_nuclei_x"
    assert _slug("well") == "well"


@pytest.fixture
def batch_harness():
    calls: list = []

    def fake_gallery(**kwargs):
        calls.append(("gallery", kwargs))
        out = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        if kwargs["plate_id"] == 13:
            raise click.ClickException("Plate 13 has no CellView rows.")
        return out / "gallery_export.json", len(kwargs["wells"].split(",")), 1

    def fake_wells(**kwargs):
        calls.append(("well", kwargs))
        out = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        return out / "well_overview.json", len(kwargs["wells"].split(",")), 1

    with (
        patch(
            "omero_screen_napari.images_cli._render_gallery",
            side_effect=fake_gallery,
        ),
        patch(
            "omero_screen_napari.images_cli._render_wells",
            side_effect=fake_wells,
        ),
    ):
        yield calls


def test_batch_runs_each_group_with_shared_options(batch_harness, tmp_path):
    plan = _plan(
        tmp_path,
        "plate_id,well,render\n5108,B2,well\n5108,G5,well\n"
        "5108,G5,gallery:micronuclei\n13,A1,gallery\n",
    )
    result = CliRunner().invoke(
        cli,
        [
            "batch", str(plan), "--channels", "DAPI", "--classifier-column",
            "classifier_nuclei4", "--zoom", "2", "--limits", "DAPI=1:2",
            "--no-labels", "--out", str(tmp_path / "out"), "--json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads(result.stdout)
    runs = {(r["plate_id"], r["render"]): r for r in manifest["runs"]}
    assert runs[(5108, "well")]["written"] == 2
    assert runs[(5108, "well")]["out"] == "5108_well"
    assert runs[(5108, "gallery:micronuclei")]["manifest"] == (
        "5108_gallery_micronuclei/gallery_export.json"
    )
    assert runs[(13, "gallery")]["error"] == "Plate 13 has no CellView rows."

    kinds = {kind: kw for kind, kw in batch_harness if kw["plate_id"] == 5108}
    assert kinds["well"]["wells"] == "B2,G5"
    assert kinds["well"]["zoom"] == "2"
    assert kinds["well"]["caption"] is False
    assert kinds["gallery"]["class_value"] == "micronuclei"
    assert kinds["gallery"]["classifier_column"] == "classifier_nuclei4"
    assert kinds["gallery"]["title"] is False
    assert kinds["gallery"]["limit_specs"] == ("DAPI=1:2",)
    plain = [kw for kind, kw in batch_harness if kw["plate_id"] == 13][0]
    assert plain["classifier_column"] == ""  # plain gallery: no class filter


@pytest.mark.parametrize(
    ("rows", "args", "hint"),
    [
        ("1,A1,gallery", [], "--channels"),
        ("1,A1,gallery:mn", ["--channels", "DAPI"], "--classifier-column"),
    ],
)
def test_batch_requires_gallery_options(batch_harness, tmp_path, rows, args, hint):
    plan = _plan(tmp_path, f"plate_id,well,render\n{rows}\n")
    result = CliRunner().invoke(cli, ["batch", str(plan), *args])
    assert result.exit_code == 2
    assert hint in result.output


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
        "import omero_screen_napari.well_overview\n"
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
