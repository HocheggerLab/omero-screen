"""Following a cell through time: crops stay centred, padding, rendering."""

import numpy as np
import pandas as pd
import pytest

from omero_screen_napari.review.filmstrip import composite, render_filmstrip
from omero_screen_napari.review.follow import follow, match_channels


def _movie():
    """A bright square nucleus jumping 30 px per frame, label 7, plus a gap at t=2."""
    image = np.zeros((4, 3, 120, 120), np.float32)
    nuclei = np.zeros((4, 120, 120), np.uint32)
    for t in range(4):
        y, x = 40, 20 + 30 * t
        image[t, :, y - 4 : y + 4, x - 4 : x + 4] = 1000
        if t != 2:
            nuclei[t, y - 4 : y + 4, x - 4 : x + 4] = 7
    path = pd.DataFrame(
        {
            "timepoint": [0, 1, 2, 3],
            "label": [7, 7, 0, 7],
            "gap": [False, False, True, False],
            "y": [40.0] * 4,
            "x": [20.0, 50.0, 80.0, 110.0],
            "pip": [1.0, 1.0, np.nan, 1.0],
            "geminin": [0.5, 0.5, np.nan, 0.5],
            "area": [64.0, 64.0, np.nan, 64.0],
            "phase": ["G1", "G1", "", "G1"],
        }
    )
    return image, nuclei, path


def test_match_channels_prefix_and_default_skips_brightfield() -> None:
    """spyDNA matches spyDNA_nucleus; the default leaves brightfield out."""
    names = ["geminin", "pip", "spyDNA_nucleus", "bf_cell"]
    assert match_channels(names, ["spyDNA", "PIP"]) == [2, 1]
    assert match_channels(names, None) == [0, 1, 2]
    with pytest.raises(ValueError):
        match_channels(names, ["H2B"])


def test_crops_stay_centred_on_a_moving_cell_and_pad_at_edges() -> None:
    """The nucleus is in the centre of every crop, even next to the border."""
    image, nuclei, path = _movie()
    cell = follow(image, nuclei, ["a", "b", "c"], path, size=32)
    assert cell.images.shape == (4, 3, 32, 32)
    for i in range(4):
        assert cell.images[i, 0, 16, 16] == 1000  # centre pixel is the nucleus
    assert cell.cell_labels == [7, 7, 0, 7]
    assert cell.labels[2].max() == 0  # gap frame: no mask
    assert (
        cell.images[3, 0, :, 30:].sum() == 0
    )  # x=110 crop runs past the edge → zero padding


def test_filmstrip_renders_tiles_and_trace() -> None:
    """One axis per tile plus the trace panel; composite is RGB in [0, 1]."""
    image, nuclei, path = _movie()
    cell = follow(image, nuclei, ["spyDNA", "pip", "geminin"], path, size=32)
    rgb = composite(cell, 0)
    assert rgb.shape == (32, 32, 3) and 0 <= rgb.min() and rgb.max() <= 1
    fig = render_filmstrip(
        cell, title="test", columns=4, events=[{"kind": "mitosis", "frame": 3}]
    )
    assert len(fig.axes) == 5
