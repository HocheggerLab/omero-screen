"""Whole-well overviews: view planning, sources, composition, orchestration."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from omero_screen_napari.well_overview import (
    ArrayWellPixels,
    OverviewError,
    OverviewSettings,
    View,
    WellInput,
    ZarrWellPixels,
    _block_mean,
    _caption,
    compose_rgb,
    draw_outlines,
    percentile_limits,
    plan_view,
    render_wells,
    scale_bar_um,
    split_layers,
    stitch_field_well,
)

# ---------------------------------------------------------------------- #
# plan_view                                                              #
# ---------------------------------------------------------------------- #


def test_whole_well_picks_the_closest_downsample():
    # 5378 px at 2000: level 0 every 3rd block (1793 px) beats level 1 x2.
    view = plan_view((5378, 5378), 3, size=2000)
    assert (view.y0, view.y1, view.x0, view.x1) == (0, 5378, 0, 5378)
    assert (view.level, view.step, view.downsample) == (0, 3, 3)


def test_coarser_level_wins_a_tie():
    view = plan_view((8000, 8000), 3, size=2000)  # needs 4: level 2 x1
    assert (view.level, view.step) == (2, 1)


def test_zoom_halves_the_field_of_view_around_the_centre():
    view = plan_view((4000, 4000), 3, zoom=4, size=2000)
    assert (view.y0, view.y1) == (1500, 2500)
    assert view.downsample == 1


def test_zoomed_view_is_kept_inside_the_canvas():
    view = plan_view((4000, 4000), 1, zoom=2, center=(0.0, 1.0), size=4000)
    assert (view.y0, view.y1, view.x0, view.x1) == (0, 2000, 2000, 4000)


def test_downsample_never_exceeds_available_levels():
    view = plan_view((20000, 20000), 1, size=1000)
    assert (view.level, view.step) == (0, 20)


@pytest.mark.parametrize(
    "kwargs", [{"zoom": 3}, {"zoom": 0}, {"center": (1.5, 0.5)}, {"size": 0}]
)
def test_plan_view_rejects(kwargs):
    with pytest.raises(OverviewError):
        plan_view((100, 100), 1, **kwargs)


# ---------------------------------------------------------------------- #
# Reading                                                                #
# ---------------------------------------------------------------------- #


def test_block_mean_averages_and_matches_strided_shape():
    plane = np.arange(25, dtype=np.float32).reshape(5, 5)
    out = _block_mean(plane, 2)
    assert out.shape == plane[::2, ::2].shape == (3, 3)
    assert out[0, 0] == pytest.approx(np.mean([0, 1, 5, 6]))


def _zarr_well(level0: np.ndarray) -> dict:
    """read_well-shaped dict with a 2-level pyramid (T, C, Y, X)."""
    nuclei0 = (level0[:, 0] > 0).astype(np.int32)
    return {
        "image": [level0, level0[..., ::2, ::2]],
        "nuclei": [nuclei0, nuclei0[..., ::2, ::2]],
        "cells": None,
    }


def test_zarr_pixels_read_level_and_masks():
    level0 = np.arange(2 * 2 * 8 * 8).reshape(2, 2, 8, 8)
    pixels = ZarrWellPixels(_zarr_well(level0), timepoint=1)
    view = View(0, 8, 0, 8, level=1, step=1)
    planes = pixels.read(view, [1])
    np.testing.assert_array_equal(planes[0], level0[1, 1, ::2, ::2])
    assert pixels.read_mask(view, "nuclei").shape == (4, 4)
    assert pixels.read_mask(view, "cells") is None


def test_zarr_timepoint_is_clamped():
    level0 = np.zeros((1, 1, 4, 4))
    assert ZarrWellPixels(_zarr_well(level0), timepoint=7)._t == 0


def test_array_pixels_block_mean_and_strided_masks():
    image = np.ones((1, 6, 6))
    mask = np.arange(36).reshape(6, 6)
    pixels = ArrayWellPixels(image, {"nuclei": mask})
    view = View(0, 6, 0, 6, level=0, step=3)
    assert pixels.read(view, [0]).shape == (1, 2, 2)
    np.testing.assert_array_equal(
        pixels.read_mask(view, "nuclei"), mask[::3, ::3]
    )


def test_stitch_field_well_places_fields_and_splits_masks():
    images = np.ones((2, 10, 10, 1), dtype=np.float32)
    labels = np.ones((2, 10, 10, 2), dtype=np.int32)
    with patch(
        "omero_utils.stitching.resolve_stitch_params",
        return_value={
            "overlap_x": 0,
            "overlap_y": 0,
            "translate_x": 0,
            "translate_y": 0,
        },
    ):
        pixels = stitch_field_well(
            images, labels, [(0.0, 0.0), (1.0, 0.0)], pixel_size_um=1.2
        )
    assert pixels.shape_yx == (10, 20)
    assert set(pixels._masks) == {"nuclei", "cells"}


def test_stitch_field_well_needs_positions():
    images = np.ones((2, 10, 10, 1), dtype=np.float32)
    with pytest.raises(OverviewError, match="stage positions"):
        stitch_field_well(images, None, [None, None], pixel_size_um=1.2)


# ---------------------------------------------------------------------- #
# Composition                                                            #
# ---------------------------------------------------------------------- #


def test_percentile_limits_ignore_zeros_and_well_order():
    a = np.concatenate([np.zeros(1000), np.full(1000, 500)])
    b = np.full(1000, 3000)
    assert percentile_limits([a, b]) == percentile_limits([b, a])
    lo, hi = percentile_limits([a, b])
    assert lo == 500  # the zeros (unacquired canvas) do not pull it down
    assert hi == 3000


def test_percentile_limits_flat_or_empty_channel():
    assert percentile_limits([np.full(10, 7)]) == (0, 65535)
    assert percentile_limits([np.zeros(10)]) == (0, 65535)


def test_compose_rgb_tints_and_clips():
    planes = np.array([[[0, 100]], [[100, 100]]])  # (C=2, y=1, x=2)
    rgb = compose_rgb(planes, ["0000FF", "00FF00"], [(0, 100), (0, 100)])
    np.testing.assert_allclose(rgb[0, 0], [0, 1, 0])
    np.testing.assert_allclose(rgb[0, 1], [0, 1, 1])


def test_draw_outlines_paints_boundaries_only():
    labels = np.zeros((7, 7), dtype=int)
    labels[2:5, 2:5] = 1
    out = draw_outlines(np.zeros((7, 7, 3)), labels, "FFFFFF")
    assert out[2, 2].tolist() == [1, 1, 1]
    assert out[3, 3].tolist() == [0, 0, 0]  # interior untouched


@pytest.mark.parametrize(
    ("width_um", "bar"), [(6000, 1000), (800, 100), (300, 50), (12, 2)]
)
def test_scale_bar_is_round(width_um, bar):
    assert scale_bar_um(width_um) == bar


def test_caption_carries_all_metadata():
    assert _caption(
        5108, "G5", {"cell_line": "RPE-1", "PALB uM": 1, "x": ""}
    ) == ("Plate 5108 — G5 | cell_line: RPE-1, PALB uM: 1")
    assert _caption(1, "A1", None) == "Plate 1 — A1"


def test_split_layers():
    assert split_layers(["DAPI", "nuclei_masks"], ["DAPI", "Tub"]) == (
        ["DAPI"],
        ["nuclei"],
    )
    with pytest.raises(OverviewError, match="GFP"):
        split_layers(["GFP"], ["DAPI"])
    with pytest.raises(OverviewError):
        split_layers([], ["DAPI"])


# ---------------------------------------------------------------------- #
# render_wells                                                           #
# ---------------------------------------------------------------------- #


def _inputs():
    rng = np.random.default_rng(0)
    canvases = {
        "A1": rng.integers(100, 1000, (2, 64, 64)),
        "B2": rng.integers(2000, 5000, (2, 64, 64)),
    }
    masks = {"nuclei": np.pad(np.ones((10, 10), int), 27)}  # 64 x 64

    def load(well):
        if well == "C3":
            raise RuntimeError("no fields")
        return WellInput(
            well,
            ArrayWellPixels(canvases[well], masks),
            f"Plate 1 — {well}",
            1.0,
        )

    return load


def test_render_wells_shares_limits_and_records_outcomes(tmp_path):
    settings = OverviewSettings(
        channel_names=["DAPI", "Tub"],
        layers=["DAPI", "Tub", "nuclei_masks"],
        size=32,
        limits={"Tub": (0, 10)},
    )
    result = render_wells(["A1", "B2", "C3"], _inputs(), settings, tmp_path)

    assert (tmp_path / "A1.png").exists() and (tmp_path / "B2.png").exists()
    assert result["wells"]["C3"] == {"exported": False, "reason": "no fields"}
    # One DAPI window for both wells, spanning both; Tub as given.
    lo, hi = result["limits"]["DAPI"]
    assert lo < 1000 and hi > 2000
    assert result["limits"]["Tub"] == [0, 10]
    assert result["limits_source"] == {"DAPI": "pooled", "Tub": "given"}
    a1 = result["wells"]["A1"]
    assert a1["downsample"] == 2 and a1["shape_yx"] == [32, 32]
    assert a1["pixel_size_um"] == 2.0
    assert result["colours"] == {"DAPI": "0000FF", "Tub": "00FF00"}


def test_render_wells_masks_only(tmp_path):
    settings = OverviewSettings(
        channel_names=["DAPI"], layers=["nuclei_masks"], size=64
    )
    result = render_wells(["A1"], _inputs(), settings, tmp_path)
    assert result["wells"]["A1"]["exported"]
    assert result["limits"] == {}


def test_render_wells_rejects_unknown_layers(tmp_path):
    settings = OverviewSettings(channel_names=["DAPI"], layers=["GFP"])
    with pytest.raises(OverviewError):
        render_wells(["A1"], _inputs(), settings, tmp_path)
