"""Batch gallery export: well resolution, file output, manifest, isolation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import pytest

from omero_screen_napari.gallery_export import (
    MANIFEST_NAME,
    available_wells,
    export_galleries,
)
from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.omero_data import OmeroData


@pytest.fixture
def omero_data():
    mock = MagicMock(spec=OmeroData)
    mock.plate_id = 3868
    mock.plate_name = "test-plate"
    mock.well_pos_list = ["A1", "B2"]
    mock.well_metadata_list = [{"cell_line": "RPE-1"}, {"cell_line": "RPE-1"}]
    mock.channel_data = {"DAPI": "0", "Tub": "1"}
    mock.intensities = {0: (0, 1000)}
    mock.pixel_size = (1.18, 1.18)
    mock.plate_data = pl.DataFrame({"well": ["A1", "B2"]}).lazy()
    mock.cropped_images = ["keep"]
    mock.cropped_labels = ["keep"]
    mock.cropped_cell_meta = ["keep"]
    mock.selected_images = ["keep"]
    mock.selected_cell_meta = ["keep"]
    return mock


@pytest.fixture
def user_data():
    ud = UserData()
    ud.channels = ["DAPI", "Tub"]
    ud.segmentation = "nucleus"
    ud.cellcycle = "All"
    ud.crop_size = 50
    ud.rows = 2
    ud.columns = 2
    return ud


def _fake_figure(*_args, **_kwargs):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1])
    return fig


# ---------------------------------------------------------------------- #
# Well resolution                                                        #
# ---------------------------------------------------------------------- #


def test_available_wells_prefers_zarr_cache(omero_data):
    omero_data.plate_data = pl.DataFrame(
        {"well": ["A1", "A2", "B2"]}
    ).lazy()
    with patch(
        "omero_screen_napari.zarr_cache.cached_wells",
        return_value=["B2", "A1", "A2"],
    ):
        # Cached wells win over the (smaller) set loaded in the viewer.
        assert available_wells(omero_data) == ["A1", "A2", "B2"]


def test_available_wells_skips_wells_absent_from_cellview(omero_data):
    omero_data.plate_data = pl.DataFrame({"well": ["A1"]}).lazy()
    with patch(
        "omero_screen_napari.zarr_cache.cached_wells",
        return_value=["A1", "A2"],
    ):
        assert available_wells(omero_data) == ["A1"]


def test_available_wells_falls_back_to_loaded_wells(omero_data):
    with patch(
        "omero_screen_napari.zarr_cache.cached_wells", return_value=[]
    ):
        assert available_wells(omero_data) == ["A1", "B2"]


# ---------------------------------------------------------------------- #
# Export                                                                 #
# ---------------------------------------------------------------------- #


def test_exports_one_file_per_well(tmp_path, omero_data, user_data):
    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_fake_figure,
    ):
        written = export_galleries(
            tmp_path,
            wells=["A1", "B2"],
            omero_data=omero_data,
            user_data=user_data,
        )

    assert [p.name for p in written] == ["A1.pdf", "B2.pdf"]
    assert all(p.exists() for p in written)


def test_format_and_wellname_filenames(tmp_path, omero_data, user_data):
    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_fake_figure,
    ):
        written = export_galleries(
            tmp_path,
            wells=["C3"],
            fmt="png",
            omero_data=omero_data,
            user_data=user_data,
        )

    assert written[0].name == "C3.png"


def test_manifest_records_settings_and_wells(tmp_path, omero_data, user_data):
    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_fake_figure,
    ):
        export_galleries(
            tmp_path,
            wells=["A1"],
            seed=7,
            omero_data=omero_data,
            user_data=user_data,
        )

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["plate_id"] == 3868
    assert manifest["seed"] == 7
    assert manifest["settings"]["crop_size"] == 50
    assert manifest["settings"]["channels"] == ["DAPI", "Tub"]
    # ``well`` is per-file, not a run setting.
    assert "well" not in manifest["settings"]
    assert manifest["wells"]["A1"]["exported"] is True
    assert manifest["wells"]["A1"]["file"] == "A1.pdf"
    assert manifest["wells"]["A1"]["well_metadata"] == {"cell_line": "RPE-1"}


def test_well_settings_carry_the_requested_well(
    tmp_path, omero_data, user_data
):
    seen = []

    def _capture(_omero, settings, **_kwargs):
        seen.append(settings.well)
        return _fake_figure()

    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_capture,
    ):
        export_galleries(
            tmp_path,
            wells=["A1", "B2"],
            omero_data=omero_data,
            user_data=user_data,
        )

    assert seen == ["A1", "B2"]
    # The singleton's own well is untouched by the batch run.
    assert user_data.well == ""


def test_one_failing_well_does_not_abort_the_run(
    tmp_path, omero_data, user_data
):
    def _fail_on_b2(_omero, settings, **_kwargs):
        if settings.well == "B2":
            raise ValueError("well B2 is not loaded")
        return _fake_figure()

    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_fail_on_b2,
    ):
        written = export_galleries(
            tmp_path,
            wells=["A1", "B2", "C3"],
            omero_data=omero_data,
            user_data=user_data,
        )

    assert [p.name for p in written] == ["A1.pdf", "C3.pdf"]
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["wells"]["B2"]["exported"] is False
    assert "not loaded" in manifest["wells"]["B2"]["reason"]


def test_well_without_crops_is_recorded_not_written(
    tmp_path, omero_data, user_data
):
    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        return_value=None,
    ):
        written = export_galleries(
            tmp_path,
            wells=["A1"],
            omero_data=omero_data,
            user_data=user_data,
        )

    assert written == []
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["wells"]["A1"] == {
        "exported": False,
        "reason": "no crops",
    }


def test_interactive_crop_pool_is_restored(tmp_path, omero_data, user_data):
    def _clobber(*_args, **_kwargs):
        omero_data.cropped_images = ["batch"]
        omero_data.selected_images = ["batch"]
        return _fake_figure()

    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_clobber,
    ):
        export_galleries(
            tmp_path,
            wells=["A1"],
            omero_data=omero_data,
            user_data=user_data,
        )

    assert omero_data.cropped_images == ["keep"]
    assert omero_data.selected_images == ["keep"]


def test_output_directory_is_created(tmp_path, omero_data, user_data):
    target = tmp_path / "figures" / "plate3868"
    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_fake_figure,
    ):
        export_galleries(
            target,
            wells=["A1"],
            omero_data=omero_data,
            user_data=user_data,
        )

    assert (target / "A1.pdf").exists()


def test_show_title_override_reaches_the_well_settings(
    tmp_path, omero_data, user_data
):
    # Interactive default is a title; an export for figure placement
    # overrides it without touching the singleton.
    user_data.show_title = True
    seen = []

    def _capture(_omero, settings, **_kwargs):
        seen.append(settings.show_title)
        return _fake_figure()

    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_capture,
    ):
        export_galleries(
            tmp_path,
            wells=["A1"],
            show_title=False,
            omero_data=omero_data,
            user_data=user_data,
        )

    assert seen == [False]
    assert user_data.show_title is True


def test_show_title_none_keeps_current_setting(
    tmp_path, omero_data, user_data
):
    user_data.show_title = True
    seen = []

    def _capture(_omero, settings, **_kwargs):
        seen.append(settings.show_title)
        return _fake_figure()

    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_capture,
    ):
        export_galleries(
            tmp_path,
            wells=["A1"],
            omero_data=omero_data,
            user_data=user_data,
        )

    assert seen == [True]


def test_manifest_records_effective_show_title(
    tmp_path, omero_data, user_data
):
    user_data.show_title = True
    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_fake_figure,
    ):
        export_galleries(
            tmp_path,
            wells=["A1"],
            show_title=False,
            omero_data=omero_data,
            user_data=user_data,
        )

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["settings"]["show_title"] is False


def test_progress_callback_reports_each_well(
    tmp_path, omero_data, user_data
):
    ticks = []
    with patch(
        "omero_screen_napari.gallery_export.build_gallery_figure",
        side_effect=_fake_figure,
    ):
        export_galleries(
            tmp_path,
            wells=["A1", "B2"],
            on_progress=lambda w, i, n: ticks.append((w, i, n)),
            omero_data=omero_data,
            user_data=user_data,
        )

    assert ticks == [("A1", 0, 2), ("B2", 1, 2)]


# ---------------------------------------------------------------------- #
# Widget wiring                                                          #
# ---------------------------------------------------------------------- #


def test_parse_export_wells():
    from omero_screen_napari._gallery_widget import _parse_export_wells

    # "All" / blank means let the exporter resolve every well.
    assert _parse_export_wells("All") is None
    assert _parse_export_wells("all") is None
    assert _parse_export_wells("  ") is None
    assert _parse_export_wells("A1, B2") == ["A1", "B2"]
    assert _parse_export_wells("A1,,B2 ") == ["A1", "B2"]


# ---------------------------------------------------------------------- #
# Validation                                                             #
# ---------------------------------------------------------------------- #


def test_no_plate_loaded_raises(omero_data, user_data, tmp_path):
    omero_data.plate_id = 0
    with pytest.raises(ValueError, match="No plate loaded"):
        export_galleries(
            tmp_path, omero_data=omero_data, user_data=user_data
        )


def test_no_channels_selected_raises(omero_data, user_data, tmp_path):
    user_data.channels = []
    with pytest.raises(ValueError, match="No channels selected"):
        export_galleries(
            tmp_path, omero_data=omero_data, user_data=user_data
        )


def test_no_resolvable_wells_raises(omero_data, user_data, tmp_path):
    omero_data.well_pos_list = []
    omero_data.plate_data = pl.DataFrame({"well": []}).lazy()
    with patch(
        "omero_screen_napari.zarr_cache.cached_wells", return_value=[]
    ):
        with pytest.raises(ValueError, match="No wells to export"):
            export_galleries(
                tmp_path, omero_data=omero_data, user_data=user_data
            )
