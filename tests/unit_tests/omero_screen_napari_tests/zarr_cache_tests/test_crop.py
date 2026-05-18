"""Crop API: fetch_crop, fetch_label_crop, fetch_crop_from_row, prepare."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from omero_screen_napari.zarr_cache import (
    PlateZarrWriter,
    fetch_crop,
    fetch_crop_from_row,
    fetch_label_crop,
    prepare,
    resolve_to_zarr,
)


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _build_single_well(
    plate_id: int,
    *,
    h: int = 512,
    w: int = 512,
    c: int = 3,
    cell_centres: list[tuple[int, int, int]] | None = None,
) -> None:
    """Build a one-well plate where each cell is a known square.

    ``cell_centres`` is a list of ``(label_id, cy, cx)``. We paint:
      * The image with a gradient (so crops have non-zero data everywhere)
      * The nuclei label mask with each cell's label_id in a 30×30 square
    """
    image = np.zeros((1, c, h, w), dtype=np.uint16)
    for y in range(h):
        image[0, :, y, :] = y % 256  # bright gradient
    nuc = np.zeros((1, h, w), dtype=np.uint32)
    cell = np.zeros((1, h, w), dtype=np.uint32)
    for label_id, cy, cx in cell_centres or []:
        nuc[0, cy - 10 : cy + 10, cx - 10 : cx + 10] = label_id
        cell[0, cy - 15 : cy + 15, cx - 15 : cx + 15] = label_id

    w_ = PlateZarrWriter(
        plate_id=plate_id,
        plate_name="t",
        channel_names=[f"ch{i}" for i in range(c)],
        pixel_size_um=1.0,
        n_timepoints=1,
    )
    w_.ensure_plate(all_wells=["A1"])
    w_.write_well("A1", image, nuc, cell)


# ---------------------------------------------------------------------- #
# resolve_to_zarr                                                        #
# ---------------------------------------------------------------------- #


def test_resolve_returns_none_for_missing_plate():
    assert resolve_to_zarr(99999) is None


def test_resolve_returns_handle_for_built_plate(synth_well_data):
    _build_single_well(400, cell_centres=[(1, 100, 100)])
    handle = resolve_to_zarr(400)
    assert handle is not None
    assert handle.plate_id == 400
    assert handle.path.exists()


# ---------------------------------------------------------------------- #
# fetch_crop                                                             #
# ---------------------------------------------------------------------- #


def test_fetch_crop_returns_expected_shape():
    _build_single_well(401, c=4, cell_centres=[(7, 256, 256)])
    crop = fetch_crop(401, "A1", label=7, centroid=(256, 256), size=128)
    assert crop.shape == (4, 128, 128)
    assert crop.dtype == np.uint16


def test_fetch_crop_centred_correctly():
    """Place a sentinel pixel at a known location and confirm it lands
    at the crop's centre."""
    h = w = 512
    image = np.zeros((1, 1, h, w), dtype=np.uint16)
    image[0, 0, 300, 250] = 12345  # sentinel
    nuc = np.zeros((1, h, w), dtype=np.uint32)
    writer = PlateZarrWriter(
        plate_id=402,
        plate_name="t",
        channel_names=["DAPI"],
        pixel_size_um=1.0,
        n_timepoints=1,
    )
    writer.ensure_plate(all_wells=["A1"])
    writer.write_well("A1", image, nuc, None)
    crop = fetch_crop(402, "A1", label=1, centroid=(300, 250), size=128)
    # The crop's centre pixel must hold the sentinel value.
    assert int(crop[0, 64, 64]) == 12345


def test_fetch_crop_channel_subset():
    _build_single_well(403, c=4, cell_centres=[(1, 100, 100)])
    full = fetch_crop(403, "A1", label=1, centroid=(100, 100), size=64)
    subset = fetch_crop(403, "A1", label=1, centroid=(100, 100), size=64, channels=[0, 2])
    assert subset.shape == (2, 64, 64)
    np.testing.assert_array_equal(subset[0], full[0])
    np.testing.assert_array_equal(subset[1], full[2])


def test_fetch_crop_zero_pads_at_canvas_edge():
    _build_single_well(404, h=256, w=256, c=2, cell_centres=[(1, 10, 10)])
    crop = fetch_crop(404, "A1", label=1, centroid=(10, 10), size=128)
    assert crop.shape == (2, 128, 128)
    # The top-left half of the crop falls outside the canvas → zero.
    assert int((crop[:, :50, :50] == 0).sum()) > 0


def test_fetch_crop_off_canvas_is_all_zero():
    _build_single_well(405, c=2, cell_centres=[(1, 100, 100)])
    crop = fetch_crop(405, "A1", label=1, centroid=(-500, -500), size=64)
    assert crop.shape == (2, 64, 64)
    assert (crop == 0).all()


def test_fetch_crop_raises_when_plate_missing():
    with pytest.raises(FileNotFoundError):
        fetch_crop(99999, "A1", label=1, centroid=(100, 100), size=64)


def test_fetch_crop_supports_odd_size():
    _build_single_well(406, c=1, cell_centres=[(1, 100, 100)])
    crop = fetch_crop(406, "A1", label=1, centroid=(100, 100), size=65)
    assert crop.shape == (1, 65, 65)


# ---------------------------------------------------------------------- #
# fetch_label_crop                                                       #
# ---------------------------------------------------------------------- #


def test_fetch_label_crop_returns_expected_label_at_centre():
    _build_single_well(410, cell_centres=[(42, 256, 256)])
    mask = fetch_label_crop(
        410, "A1", centroid=(256, 256), size=64, mask_name="nuclei"
    )
    assert mask.shape == (64, 64)
    # Centre 30×30 square is label 42 (the only cell on this canvas).
    assert int(mask[32, 32]) == 42


def test_fetch_label_crop_cells_mask_distinct_from_nuclei():
    """The synth plate paints cells (30×30) larger than nuclei (20×20).
    A crop at the cell edge sees cell label only."""
    _build_single_well(411, cell_centres=[(7, 256, 256)])
    nuc = fetch_label_crop(411, "A1", centroid=(256, 256), size=64, mask_name="nuclei")
    cells = fetch_label_crop(411, "A1", centroid=(256, 256), size=64, mask_name="cells")
    # Both masks centred on the cell agree on label id at the centre.
    assert int(nuc[32, 32]) == 7
    assert int(cells[32, 32]) == 7
    # But the cells mask covers more pixels than the nuclei mask.
    assert int((cells == 7).sum()) > int((nuc == 7).sum())


def test_fetch_label_crop_raises_on_missing_mask_name():
    _build_single_well(412, cell_centres=[(1, 100, 100)])
    with pytest.raises(KeyError):
        fetch_label_crop(412, "A1", centroid=(100, 100), size=64, mask_name="bogus")


# ---------------------------------------------------------------------- #
# fetch_crop_from_row                                                    #
# ---------------------------------------------------------------------- #


def test_fetch_crop_from_row_basic():
    _build_single_well(420, c=2, cell_centres=[(5, 200, 200)])
    row = {
        "plate_id": 420,
        "well": "A1",
        "label": 5,
        "timepoint": 0,
        "centroid_y": 200,
        "centroid_x": 200,
    }
    crop = fetch_crop_from_row(row, size=32)
    assert crop.shape == (2, 32, 32)


def test_fetch_crop_from_row_returns_label_when_mask_requested():
    _build_single_well(421, c=2, cell_centres=[(9, 200, 200)])
    row = {
        "plate_id": 421,
        "well": "A1",
        "label": 9,
        "centroid_y": 200,
        "centroid_x": 200,
    }
    image_crop, label_crop = fetch_crop_from_row(
        row, size=64, mask_name="nuclei"
    )
    assert image_crop.shape == (2, 64, 64)
    assert label_crop.shape == (64, 64)
    assert int(label_crop[32, 32]) == 9


def test_fetch_crop_from_row_accepts_legacy_centroid_columns():
    """Older measurements use 'centroid-0' / 'centroid-1' or
    '-cell'-suffixed variants. The wrapper falls back through aliases."""
    _build_single_well(422, c=1, cell_centres=[(1, 100, 100)])
    row = {
        "plate_id": 422,
        "well": "A1",
        "label": 1,
        "centroid-0-cell": 100,
        "centroid-1-cell": 100,
    }
    crop = fetch_crop_from_row(row, size=32)
    assert crop.shape == (1, 32, 32)


def test_fetch_crop_from_row_defaults_timepoint_to_zero():
    _build_single_well(423, c=1, cell_centres=[(1, 100, 100)])
    row = {
        "plate_id": 423,
        "well": "A1",
        "label": 1,
        "centroid_y": 100,
        "centroid_x": 100,
    }  # no timepoint key
    crop = fetch_crop_from_row(row, size=32)
    assert crop.shape == (1, 32, 32)


# ---------------------------------------------------------------------- #
# prepare                                                                #
# ---------------------------------------------------------------------- #


def test_prepare_reports_ready_for_existing_plate():
    _build_single_well(430, cell_centres=[(1, 100, 100)])
    result = prepare([430], conn=MagicMock())
    assert result == {430: "ready"}


def test_prepare_calls_builder_for_missing_plate():
    """Patches build_plate_zarr to short-circuit OMERO."""
    with patch(
        "omero_screen_napari.zarr_cache.crop.build_plate_zarr",
        return_value=iter(["A1", "B2"]),
    ) as mock_build:
        result = prepare([500], conn=MagicMock())
    assert result == {500: "built"}
    mock_build.assert_called_once()


def test_prepare_reports_failed_when_build_raises():
    def raising_build(plate_id, conn):
        if False:
            yield  # pragma: no cover
        raise RuntimeError("simulated")

    with patch(
        "omero_screen_napari.zarr_cache.crop.build_plate_zarr",
        side_effect=raising_build,
    ):
        result = prepare([501], conn=MagicMock())
    assert result == {501: "failed"}


def test_prepare_continues_after_individual_failure():
    """One bad plate shouldn't abort the rest."""
    _build_single_well(502, cell_centres=[(1, 100, 100)])

    def selective_build(plate_id, conn):
        if plate_id == 503:
            raise RuntimeError("boom")
        return iter(["A1"])

    with patch(
        "omero_screen_napari.zarr_cache.crop.build_plate_zarr",
        side_effect=selective_build,
    ):
        result = prepare([502, 503, 504], conn=MagicMock())
    assert result[502] == "ready"
    assert result[503] == "failed"
    assert result[504] == "built"


def test_prepare_progress_callback_invoked():
    _build_single_well(510, cell_centres=[(1, 100, 100)])
    messages: list[tuple[int, str]] = []
    prepare([510], conn=MagicMock(), progress_cb=lambda p, m: messages.append((p, m)))
    assert messages == [(510, "ready")]
