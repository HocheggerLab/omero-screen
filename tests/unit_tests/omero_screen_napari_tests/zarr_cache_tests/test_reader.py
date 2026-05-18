"""Reader: open_plate, read_well, cached_wells, plate_info."""

from __future__ import annotations

import pytest

from omero_screen_napari.zarr_cache import (
    PlateZarrWriter,
    cached_wells,
    open_plate,
    plate_info,
    plate_zarr_path,
    read_well,
)


def _build_two_well_plate(plate_id, synth_well_data, well_meta=None):
    image, nuc, cell = synth_well_data(h=256, w=256, c=2)
    w = PlateZarrWriter(
        plate_id=plate_id,
        plate_name="test",
        channel_names=["DAPI", "Tub"],
        pixel_size_um=0.65,
        n_timepoints=1,
    )
    w.ensure_plate(all_wells=["A1", "B2"], well_metadata=well_meta)
    w.write_well("A1", image, nuc, cell)
    w.write_well("B2", image, nuc, cell)
    return plate_zarr_path(plate_id)


def test_open_plate_missing_raises():
    with pytest.raises(FileNotFoundError):
        open_plate(99999)


def test_read_well_returns_full_pyramid(synth_well_data):
    _build_two_well_plate(200, synth_well_data)
    data = read_well(200, "A1")
    assert len(data["image"]) == 3  # 3 pyramid levels
    assert len(data["nuclei"]) == 3
    assert len(data["cells"]) == 3
    assert data["channel_names"] == ["DAPI", "Tub"]
    assert data["pixel_size_um"] == 0.65


def test_cached_wells_returns_only_written(synth_well_data):
    # Plate advertises A1, B2 but we'll only write A1.
    image, nuc, _ = synth_well_data(h=256, w=256)
    w = PlateZarrWriter(
        plate_id=201,
        plate_name="t",
        channel_names=["DAPI"],
        pixel_size_um=1.0,
        n_timepoints=1,
    )
    w.ensure_plate(all_wells=["A1", "B2"])
    w.write_well("A1", image, nuc, None)
    assert cached_wells(201) == ["A1"]


def test_cached_wells_returns_empty_when_plate_absent():
    assert cached_wells(404) == []


def test_plate_info_returns_well_metadata(synth_well_data):
    _build_two_well_plate(
        202,
        synth_well_data,
        well_meta={
            "A1": {"cell_line": "U2OS", "condition": "ctrl"},
            "B2": {"cell_line": "RPE", "condition": "drug1"},
        },
    )
    info = plate_info(202)
    assert info["well_metadata"]["A1"]["cell_line"] == "U2OS"
    assert info["well_metadata"]["B2"]["condition"] == "drug1"
    assert "A" in info["rows"]
    assert "1" in info["columns"]
    assert info["n_timepoints"] == 1


def test_read_well_for_nucleus_only_store(synth_well_data):
    image, nuc, _ = synth_well_data(h=256, w=256, c=1)
    w = PlateZarrWriter(
        plate_id=203,
        plate_name="t",
        channel_names=["DAPI"],
        pixel_size_um=1.0,
        n_timepoints=1,
    )
    w.ensure_plate(all_wells=["A1"])
    w.write_well("A1", image, nuc, None)  # no cell mask
    data = read_well(203, "A1")
    assert data["nuclei"] is not None
    assert data["cells"] is None
