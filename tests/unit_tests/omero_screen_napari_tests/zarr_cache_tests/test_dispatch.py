"""Backend dispatch helpers in _welldata_widget."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from omero_screen_napari._welldata_widget import _cache_status_tag
from omero_screen_napari.zarr_cache import PlateZarrWriter


def _make_zarr(plate_id: int) -> None:
    """Create a minimal store so plate_zarr_path(...).exists() is true."""
    w = PlateZarrWriter(
        plate_id=plate_id,
        plate_name="t",
        channel_names=["DAPI"],
        pixel_size_um=1.0,
        n_timepoints=1,
    )
    w.ensure_plate(all_wells=["A1"])


def test_tag_empty_when_neither_cache_present(isolated_cache):
    with patch(
        "omero_screen_napari._welldata_widget.is_plate_cached",
        return_value=False,
    ):
        assert _cache_status_tag(404) == ""


def test_tag_zarr_when_zarr_exists(isolated_cache):
    _make_zarr(500)
    with patch(
        "omero_screen_napari._welldata_widget.is_plate_cached",
        return_value=False,
    ):
        assert _cache_status_tag(500) == "zarr"


def test_tag_zarr_overrides_disk(isolated_cache):
    """Zarr presence wins regardless of the diskcache state — the load
    path uses zarr when available."""
    _make_zarr(501)
    with patch(
        "omero_screen_napari._welldata_widget.is_plate_cached",
        return_value=True,
    ):
        assert _cache_status_tag(501) == "zarr"


def test_tag_disk_when_only_diskcache(isolated_cache):
    with patch(
        "omero_screen_napari._welldata_widget.is_plate_cached",
        return_value=True,
    ):
        assert _cache_status_tag(502) == "disk"
