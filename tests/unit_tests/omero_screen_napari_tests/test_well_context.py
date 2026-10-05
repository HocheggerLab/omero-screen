"""Headless well-context loading: zarr vs per-field dispatch and display limits."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.well_context import (
    WellContextError,
    load_well_context,
    pooled_intensities,
)


def _well(values: np.ndarray) -> dict:
    """A read_well-shaped dict: levels of (T, C, Y, X)."""
    level0 = values
    level1 = values[..., ::2, ::2]
    return {"image": [level0, level1]}


# ---------------------------------------------------------------------- #
# pooled_intensities                                                     #
# ---------------------------------------------------------------------- #


def test_pooled_limits_do_not_depend_on_well_order():
    rng = np.random.default_rng(1)
    a = _well(rng.integers(0, 1000, (1, 2, 64, 64)))
    b = _well(rng.integers(500, 5000, (1, 2, 64, 64)))
    assert pooled_intensities([a, b]) == pooled_intensities([b, a])


def test_pooled_limits_span_all_wells():
    dim = _well(np.full((1, 1, 64, 64), 100))
    bright = _well(np.full((1, 1, 64, 64), 4000))
    lo, hi = pooled_intensities([dim, bright])[0]
    assert lo == 100
    assert hi == 4000


def test_flat_channel_gets_full_range():
    flat = _well(np.full((1, 1, 16, 16), 7))
    assert pooled_intensities([flat]) == {0: (0, 65535)}


def test_timepoint_is_clamped_to_last():
    values = np.stack(
        [np.full((1, 16, 16), 10), np.full((1, 16, 16), 20)]
    )  # (T=2, C=1, Y, X)
    values[1, 0, 0, 0] = 30
    lo, hi = pooled_intensities([_well(values)], timepoint=9)[0]
    assert lo == 20


def test_no_wells_gives_no_limits():
    assert pooled_intensities([]) == {}


# ---------------------------------------------------------------------- #
# load_well_context                                                      #
# ---------------------------------------------------------------------- #


def test_no_wells_is_an_error():
    with pytest.raises(WellContextError):
        load_well_context(1, [])


def test_zarr_plate_rejects_uncached_wells():
    with (
        patch("omero_screen_napari.well_context.well_source", return_value="zarr"),
        patch(
            "omero_screen_napari.zarr_cache.cached_wells", return_value=["A1"]
        ),
        pytest.raises(WellContextError, match="B2"),
    ):
        load_well_context(1, ["A1", "B2"])


def test_zarr_plate_populates_and_pools_limits():
    od = OmeroData()
    wells = {
        "A1": _well(np.full((1, 1, 16, 16), 100)),
        "B2": _well(np.full((1, 1, 16, 16), 900)),
    }
    with (
        patch("omero_screen_napari.well_context.well_source", return_value="zarr"),
        patch(
            "omero_screen_napari.zarr_cache.cached_wells",
            return_value=["A1", "B2"],
        ),
        patch(
            "omero_screen_napari.zarr_cache.plate_info",
            return_value={"channel_names": ["DAPI"], "pixel_size_um": 1.2},
        ),
        patch(
            "omero_screen_napari.zarr_cache.read_well",
            side_effect=lambda _p, w: wells[w],
        ),
        patch(
            "omero_screen_napari.zarr_cache.display.populate_omero_data"
        ) as populate,
    ):
        result = load_well_context(7, ["B2", "A1"], omero_data=od)

    assert result is od
    populate.assert_called_once()
    assert populate.call_args.args[:4] == (od, 7, "B2, A1", ["B2", "A1"])
    assert od.intensities == {0: (100, 900)}


def test_stitched_plate_without_zarr_is_refused():
    with (
        patch(
            "omero_screen_napari.well_context.well_source",
            return_value="fields",
        ),
        patch(
            "omero_screen_napari.plate_cache.get_plate_metadata",
            return_value={"label_stitched_mode": True},
        ),
        pytest.raises(WellContextError, match="stitched"),
    ):
        load_well_context(1, ["A1"], connection=MagicMock())


@pytest.mark.parametrize(("timepoint", "time"), [(None, "All"), (0, "1"), (4, "5")])
def test_field_plate_loads_the_requested_timepoint(timepoint, time):
    connection = MagicMock()
    calls = []

    def fake_load(conn, od, plate_id, wells, images, stop, time):
        calls.append((conn, plate_id, wells, images, time))
        yield 1, 2
        yield 2, 2

    with (
        patch(
            "omero_screen_napari.well_context.well_source",
            return_value="fields",
        ),
        patch(
            "omero_screen_napari.plate_cache.get_plate_metadata",
            return_value={"label_stitched_mode": False},
        ),
        patch(
            "omero_screen_napari.plate_cache.load_from_cache",
            side_effect=fake_load,
        ),
    ):
        load_well_context(
            3, ["A1", "B2"], timepoint=timepoint, connection=connection
        )

    assert calls == [(connection, 3, "A1, B2", "All", time)]


def test_unprocessed_plate_is_a_clean_error():
    with (
        patch(
            "omero_screen_napari.well_context.well_source",
            return_value="fields",
        ),
        patch(
            "omero_screen_napari.plate_cache.get_plate_metadata",
            side_effect=ValueError("No MapAnnotations found for plate 1"),
        ),
        pytest.raises(WellContextError, match="processed by omero-screen"),
    ):
        load_well_context(1, ["A1"], connection=MagicMock())
