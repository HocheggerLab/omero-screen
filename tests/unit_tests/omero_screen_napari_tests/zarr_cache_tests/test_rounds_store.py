"""The 4i store on disk: attrs round-trip, colour assignment, and resolution.

Backward compatibility is the theme. An ordinary single-plate store must keep
its exact pre-4i attrs, and every reader must tolerate its absence -- the
existing caches on disk have no ``rounds`` block and must not need rebuilding.
"""

from __future__ import annotations

import numpy as np
import zarr

from omero_screen_napari.zarr_cache import PlateZarrWriter, plate_zarr_path
from omero_screen_napari.zarr_cache.crop import resolve_to_zarr
from omero_screen_napari.zarr_cache.reader import plate_info
from omero_screen_napari.zarr_cache.registry import ZarrPlateEntry, upsert
from omero_screen_napari.zarr_cache.rounds import (
    RoundGroup,
    build_channel_plan,
)
from omero_screen_napari.zarr_cache.writer import SCHEMA_VERSION

MASTER = 4127


def _rounds_attrs(include_redundant: bool = False):  # type: ignore[no-untyped-def]
    group = RoundGroup(MASTER, (4130, 4131))
    names, attrs, _ = build_channel_plan(
        group,
        {
            MASTER: {"DAPI": "0", "Tub": "1"},
            4130: {"DAPI": "0", "EdU": "1"},
            4131: {"DAPI": "0", "H3P": "1"},
        },
        include_redundant=include_redundant,
    )
    return names, attrs


def _write(plate_id: int, rounds=None, channel_names=None):  # type: ignore[no-untyped-def]
    names = channel_names or ["DAPI", "Tub"]
    writer = PlateZarrWriter(
        plate_id=plate_id,
        plate_name="test",
        channel_names=names,
        pixel_size_um=0.5,
        n_timepoints=1,
        rounds=rounds,
    )
    with writer:
        writer.ensure_plate(all_wells=["A1"])
        n_c = len(names)
        writer.write_well(
            "A1",
            np.random.default_rng(0).integers(
                0, 500, (1, n_c, 64, 64), dtype=np.uint16
            ),
            np.zeros((1, 64, 64), dtype=np.uint32),
        )
    return writer


class TestOrdinaryStoreUnchanged:
    def test_no_rounds_block_written(self) -> None:
        _write(1234)
        root = zarr.open_group(str(plate_zarr_path(1234)), mode="r")
        attrs = root.attrs["omero_screen"]
        assert "rounds" not in attrs
        assert "schema_version" not in attrs

    def test_plate_info_reports_no_rounds(self) -> None:
        _write(1234)
        info = plate_info(1234)
        assert info["rounds"] is None
        assert info["schema_version"] == 1


class TestFourIStore:
    def test_rounds_block_round_trips(self) -> None:
        names, attrs = _rounds_attrs()
        _write(MASTER, rounds=attrs, channel_names=names)
        info = plate_info(MASTER)
        assert info["schema_version"] == SCHEMA_VERSION
        assert info["rounds"]["master_plate_id"] == MASTER
        assert info["rounds"]["member_plate_ids"] == [4130, 4131]
        assert (
            info["rounds"]["shift_convention"] == "master = restain - (x, y)"
        )

    def test_repeated_nuclear_stain_is_dropped(self) -> None:
        """Only the master's DAPI: rounds 2+ re-image it for registration only."""
        names, attrs = _rounds_attrs()
        _write(MASTER, rounds=attrs, channel_names=names)
        assert plate_info(MASTER)["channel_names"] == [
            "DAPI_R1",
            "Tub_R1",
            "EdU_R2",
            "H3P_R3",
        ]

    def test_every_channel_gets_a_distinct_colour(self) -> None:
        """Two identical colours would sum under additive blending."""
        names, attrs = _rounds_attrs(include_redundant=True)
        _write(MASTER, rounds=attrs, channel_names=names)
        root = zarr.open_group(str(plate_zarr_path(MASTER)), mode="r")
        colours = [
            c["color"] for c in root["A/1/0"].attrs["omero"]["channels"]
        ]
        assert len(colours) == 6
        assert len(set(colours)) == 6

    def test_only_round_one_is_active(self) -> None:
        names, attrs = _rounds_attrs()
        _write(MASTER, rounds=attrs, channel_names=names)
        root = zarr.open_group(str(plate_zarr_path(MASTER)), mode="r")
        active = [c["active"] for c in root["A/1/0"].attrs["omero"]["channels"]]
        assert active == [True, True, False, False]


class TestCropResolution:
    def test_own_store_resolves_directly(self) -> None:
        _write(1234)
        handle = resolve_to_zarr(1234)
        assert handle is not None
        assert handle.plate_id == 1234

    def test_restain_id_resolves_to_the_master_store(self) -> None:
        """fetch_crop_from_row reads plate_id off a row; restain rows must hit."""
        names, attrs = _rounds_attrs()
        _write(MASTER, rounds=attrs, channel_names=names)
        upsert(
            ZarrPlateEntry(plate_id=MASTER, member_plate_ids=[4130, 4131])
        )
        handle = resolve_to_zarr(4130)
        assert handle is not None
        assert handle.plate_id == MASTER
        assert handle.path == plate_zarr_path(MASTER)

    def test_unknown_plate_is_none(self) -> None:
        assert resolve_to_zarr(9999) is None

    def test_member_without_a_built_store_is_none(self) -> None:
        """Registered as a member but the master store is gone."""
        upsert(ZarrPlateEntry(plate_id=MASTER, member_plate_ids=[4130]))
        assert resolve_to_zarr(4130) is None
