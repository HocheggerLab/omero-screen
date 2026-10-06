"""Reviewer masks: watershed split, background-exact measurement, painting patches."""

import numpy as np
import pandas as pd
import pytest

from omero_screen_napari.review.masks import (
    MaskError,
    channel_map,
    load_patch,
    measure,
    paint_patches,
    save_patch,
    split_mask,
)


def test_split_mask_follows_the_seeds() -> None:
    """Two touching discs split along the waist, one part per seed."""
    yy, xx = np.mgrid[0:40, 0:60]
    region = ((yy - 20) ** 2 + (xx - 18) ** 2 < 100) | ((yy - 20) ** 2 + (xx - 40) ** 2 < 100)
    a, b = split_mask(region, [(20, 18), (20, 40)])
    assert a[20, 18] and b[20, 40] and not (a & b).any() and ((a | b) == region).all()
    with pytest.raises(MaskError):
        split_mask(region, [(0, 0), (20, 40)])


def test_measure_recovers_the_pipeline_background() -> None:
    """A new mask's mean equals raw mean minus the frame background implied by a measured nucleus."""
    image = np.full((1, 1, 50, 50), 100.0)  # background 100
    image[0, 0, 5:15, 5:15] = 600.0  # reference nucleus, raw 600 → measured 500
    image[0, 0, 30:40, 30:40] = 400.0  # new nucleus, raw 400 → expect 300
    nuclei = np.zeros((1, 50, 50), np.uint32)
    nuclei[0, 5:15, 5:15] = 3
    det_frame = pd.DataFrame({"label": [3], "area": [100.0], "y": [9.5], "x": [9.5], "pip": [500.0]})
    mask = np.ones((10, 10), bool)
    out = measure(image, nuclei, det_frame, 0, 30, 30, mask, {"pip": 0})
    assert out["area"] == 100 and out["y"] == 34.5 and out["pip"] == pytest.approx(300.0)


def test_patches_round_trip_and_paint(tmp_path) -> None:
    """Saved patches load back and are painted (replaced labels cleared) into a crop."""
    from cellview.tracks.edit import Curated

    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    rel = save_patch(tmp_path / "masks", "C2", 5, 900, 10, 20, mask)
    y0, x0, back = load_patch(tmp_path, rel)
    assert (y0, x0) == (10, 20) and (back == mask).all()
    cur = Curated(tracks={}, parents={}, extras={(5, 900): {"patch": rel}}, removed={(5, 7)})
    crop = np.full((8, 8), 7, np.uint32)
    painted = paint_patches(crop, 5, 8, 18, cur, tmp_path)
    assert painted[3, 3] == 900 and painted[0, 0] == 0
    assert paint_patches(crop, 6, 8, 18, cur, tmp_path)[0, 0] == 7  # other frames untouched


def test_channel_map_prefix() -> None:
    """Detection columns map to image channels by prefix, either way round."""
    assert channel_map(["geminin", "pip", "spyDNA_nucleus", "bf_cell"], ["pip", "geminin", "spydna"]) == {
        "pip": 1, "geminin": 0, "spydna": 2}
