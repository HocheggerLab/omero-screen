"""Unit tests for well-loading helpers in ``omero_screen.loops``.

OMERO objects are mocked so these run fast and offline. The focus is the
memory-critical contract: flatfield-corrected field stacks come back as
float32 (not float64), which halves the resident size of the stitched canvas
and everything derived from it on long multi-channel timelapses.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from omero_screen.loops import _load_well_fields


def _mock_well(n_fields: int) -> MagicMock:
    """A WellWrapper mock with ``n_fields`` identical samples at the origin."""
    well = MagicMock()
    well.listChildren.return_value = [object()] * n_fields
    ws = MagicMock()
    ws.getImage.return_value.getId.return_value = 42
    ws.getPosX.return_value.getValue.return_value = 0.0
    ws.getPosY.return_value.getValue.return_value = 0.0
    well.getWellSample.return_value = ws
    return well


@patch("omero_screen.loops.get_image")
def test_load_well_fields_returns_float32(mock_get_image: MagicMock) -> None:
    """Flatfield-corrected stacks are float32, shaped (N_fields, T, Y, X)."""
    # get_image returns (metadata, array) with array shaped (T, Z, Y, X, C).
    array = (np.ones((3, 1, 4, 4, 2)) * 1000).astype(np.uint16)
    mock_get_image.return_value = (None, array)

    metadata = MagicMock()
    metadata.channel_data = {"DAPI": 0, "Tub": 1}
    flatfield = {
        "DAPI": np.ones((4, 4), dtype=np.float32),
        "Tub": np.ones((4, 4), dtype=np.float32),
    }

    stacked, positions, image_ids = _load_well_fields(
        MagicMock(), _mock_well(2), metadata, 1, flatfield
    )

    assert set(stacked) == {"DAPI", "Tub"}
    for arr in stacked.values():
        assert arr.dtype == np.float32  # NOT float64 — the memory fix
        assert arr.shape == (2, 3, 4, 4)  # (N_fields, T, Y, X)
    assert len(positions) == 2
    assert image_ids == [42, 42]
