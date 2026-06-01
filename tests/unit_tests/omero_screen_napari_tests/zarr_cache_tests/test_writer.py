"""Writer: NGFF layout, chunking, idempotent ensure_plate, atomic well swap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from omero_screen_napari.zarr_cache import PlateZarrWriter, plate_zarr_path


# Convenience for asserting array properties.
def _open_grp(path: Path) -> zarr.hierarchy.Group:
    return zarr.open_group(str(path), mode="r")


def _writer(plate_id: int = 100, **kw) -> PlateZarrWriter:
    defaults = dict(
        plate_id=plate_id,
        plate_name="test",
        channel_names=["DAPI", "Tub"],
        pixel_size_um=0.65,
        n_timepoints=1,
    )
    defaults.update(kw)
    return PlateZarrWriter(**defaults)


def test_ensure_plate_writes_full_well_list(synth_well_data):
    w = _writer()
    w.ensure_plate(all_wells=["A1", "A2", "B1"])
    root = _open_grp(plate_zarr_path(100))
    advertised = {x["path"] for x in root.attrs["plate"]["wells"]}
    assert advertised == {"A/1", "A/2", "B/1"}


def test_ensure_plate_is_idempotent(synth_well_data):
    w = _writer()
    w.ensure_plate(all_wells=["A1"])
    # Second call must not crash even with a different well list
    w.ensure_plate(all_wells=["A1", "A2"])
    root = _open_grp(plate_zarr_path(100))
    # Still the original well list since ensure_plate is a no-op.
    advertised = {x["path"] for x in root.attrs["plate"]["wells"]}
    assert advertised == {"A/1"}


def test_ensure_plate_rejects_empty_well_list():
    w = _writer()
    with pytest.raises(ValueError):
        w.ensure_plate(all_wells=[])


def test_omero_screen_attrs_stash_plate_info(synth_well_data):
    w = _writer(plate_id=101, channel_names=["A", "B", "C"], pixel_size_um=0.5)
    w.ensure_plate(all_wells=["A1"], well_metadata={"A1": {"cell_line": "U2OS"}})
    root = _open_grp(plate_zarr_path(101))
    meta = root.attrs["omero_screen"]
    assert meta["plate_id"] == 101
    assert meta["channel_names"] == ["A", "B", "C"]
    assert meta["pixel_size_um"] == 0.5
    assert meta["well_metadata"] == {"A1": {"cell_line": "U2OS"}}


def test_write_well_creates_pyramid_and_labels(synth_well_data):
    image, nuc, cell = synth_well_data(t=1, c=2, h=512, w=512)
    w = _writer(plate_id=102, channel_names=["DAPI", "Tub"])
    w.ensure_plate(all_wells=["A1"])
    w.write_well("A1", image, nuc, cell)

    root = _open_grp(plate_zarr_path(102))
    well_grp = root["A/1/0"]
    levels = sorted(k for k in well_grp.array_keys() if k.isdigit())
    # 3 pyramid levels.
    assert levels == ["0", "1", "2"]
    assert well_grp["0"].shape == (1, 2, 512, 512)
    assert well_grp["1"].shape == (1, 2, 256, 256)
    assert well_grp["2"].shape == (1, 2, 128, 128)
    # Labels group present.
    assert "labels" in well_grp.group_keys()
    assert "nuclei" in well_grp["labels"].group_keys()
    assert "cells" in well_grp["labels"].group_keys()


def test_image_chunk_shape_uses_spatial_chunk(synth_well_data):
    image, nuc, _ = synth_well_data(t=1, c=2, h=512, w=512)
    w = _writer()
    w.ensure_plate(all_wells=["A1"])
    w.write_well("A1", image, nuc, None)
    arr = _open_grp(plate_zarr_path(100))["A/1/0/0"]
    # (T_chunk=1, C_chunk=1, 256, 256) for T=1 fixed-cell plate.
    assert arr.chunks == (1, 1, 256, 256)


def test_timelapse_chunks_one_frame_per_chunk(synth_well_data):
    """T axis must chunk as 1 even for a multi-timepoint well.

    A packed T chunk makes BigDataViewer / Mastodon render only t=0 and black
    thereafter; one frame per chunk also keeps napari time-scrubbing cheap.
    """
    image, nuc, _ = synth_well_data(t=6, c=2, h=512, w=512)
    w = _writer(n_timepoints=6)
    w.ensure_plate(all_wells=["A1"])
    w.write_well("A1", image, nuc, None)
    img = _open_grp(plate_zarr_path(100))["A/1/0/0"]
    lbl = _open_grp(plate_zarr_path(100))["A/1/0/labels/nuclei/0"]
    assert img.shape[0] == 6 and img.chunks[0] == 1
    assert lbl.chunks[0] == 1


def test_write_well_rejects_wrong_image_ndim(synth_well_data):
    w = _writer()
    w.ensure_plate(all_wells=["A1"])
    bad = np.zeros((1, 256, 256), dtype=np.uint16)  # 3-D instead of 4-D
    with pytest.raises(ValueError):
        w.write_well("A1", bad, np.zeros((1, 256, 256), dtype=np.uint32))


def test_write_well_rejects_wrong_label_ndim(synth_well_data):
    image, _, _ = synth_well_data(h=256, w=256)
    w = _writer()
    w.ensure_plate(all_wells=["A1"])
    with pytest.raises(ValueError):
        w.write_well(
            "A1",
            image,
            np.zeros((1, 1, 256, 256), dtype=np.uint32),  # 4-D not 3-D
        )


def test_write_well_rejects_unadvertised_well(synth_well_data):
    image, nuc, _ = synth_well_data(h=256, w=256)
    w = _writer()
    w.ensure_plate(all_wells=["A1"])
    with pytest.raises(ValueError):
        w.write_well("B2", image, nuc, None)  # not in advertised set


def test_write_well_does_not_mutate_sibling_well(synth_well_data):
    """Writing well A2 must not change A1's chunk files."""
    image_a1, nuc_a1, cell_a1 = synth_well_data(h=256, w=256, seed=1)
    image_a2, nuc_a2, cell_a2 = synth_well_data(h=256, w=256, seed=2)
    w = _writer(plate_id=103)
    w.ensure_plate(all_wells=["A1", "A2"])
    w.write_well("A1", image_a1, nuc_a1, cell_a1)

    a1_dir = plate_zarr_path(103) / "A" / "1"
    before = {p.name: p.stat().st_size for p in a1_dir.rglob("*") if p.is_file()}

    w.write_well("A2", image_a2, nuc_a2, cell_a2)
    after = {p.name: p.stat().st_size for p in a1_dir.rglob("*") if p.is_file()}
    assert before == after


def test_omero_channels_metadata_landed_on_image_group(synth_well_data):
    image, nuc, _ = synth_well_data(h=256, w=256, c=2)
    w = _writer(plate_id=104, channel_names=["DAPI", "Tub"])
    w.ensure_plate(all_wells=["A1"])
    w.write_well("A1", image, nuc, None)
    img_grp = _open_grp(plate_zarr_path(104))["A/1/0"]
    assert "omero" in img_grp.attrs
    omero = img_grp.attrs["omero"]
    assert [c["label"] for c in omero["channels"]] == ["DAPI", "Tub"]
    # Window has the 4 required fields.
    for c in omero["channels"]:
        assert set(c["window"]) == {"min", "max", "start", "end"}


def test_close_cleans_tmp_dir(synth_well_data, tmp_path):
    image, nuc, _ = synth_well_data(h=256, w=256)
    w = _writer(plate_id=105)
    w.ensure_plate(all_wells=["A1"])
    w.write_well("A1", image, nuc, None)
    # Even after a successful write, the .tmp dir should be cleaned up
    # on close.
    w.close()
    assert not w.tmp_path.exists()
