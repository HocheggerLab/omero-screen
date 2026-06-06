"""Unit tests for well-loading helpers in ``omero_screen.loops``.

OMERO objects are mocked so these run fast and offline. The focus is the
memory-critical contract: flatfield-corrected field stacks come back as
float32 (not float64), which halves the resident size of the stitched canvas
and everything derived from it on long multi-channel timelapses.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from omero_screen.loops import (
    _load_and_stitch_streaming,
    _load_well_fields,
    _stitch_well,
)


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


def _streaming_well(field_data: dict[int, np.ndarray]) -> MagicMock:
    """A well whose samples expose ids/positions/dims for the streaming loader."""
    n_fields = len(field_data)
    ids = list(field_data)
    well = MagicMock()
    well.listChildren.return_value = [object()] * n_fields
    # Distinct stage positions per field so the stitch geometry is non-trivial.
    samples = []
    for i, fid in enumerate(ids):
        s = MagicMock()
        s.getImage.return_value.getId.return_value = fid
        # dims from this field's array (T, Z, Y, X, C)
        t, z, y, x, c = field_data[fid].shape
        for attr, val in (
            ("getSizeT", t),
            ("getSizeZ", z),
            ("getSizeY", y),
            ("getSizeX", x),
            ("getSizeC", c),
        ):
            getattr(s.getImage.return_value, attr).return_value = val
        s.getPosX.return_value.getValue.return_value = float(i * x)
        s.getPosY.return_value.getValue.return_value = 0.0
        samples.append(s)
    well.getWellSample.side_effect = lambda n: samples[n]
    return well


def test_streaming_stitch_matches_nonstreaming() -> None:
    """The streaming canvas is byte-identical to load-all-then-stitch.

    The whole point of the streaming loader is to bound RAM without changing
    the result, so this is the load-bearing test: same fields + positions fed
    to _stitch_well frame-by-frame must equal stitching the full stack at once.
    """
    rng = np.random.default_rng(0)
    # Two fields, T=3, Z=1, Y=X=5, C=2.
    field_data = {
        10: (rng.uniform(50, 4000, (3, 1, 5, 5, 2))).astype(np.uint16),
        11: (rng.uniform(50, 4000, (3, 1, 5, 5, 2))).astype(np.uint16),
    }
    channels = {"DAPI": 0, "Tub": 1}
    flatfield = {
        "DAPI": np.ones((5, 5), dtype=np.float32),
        "Tub": np.ones((5, 5), dtype=np.float32),
    }
    metadata = MagicMock()
    metadata.channel_data = channels

    def whole_field(conn, image_id, **kw):  # type: ignore[no-untyped-def]
        arr = field_data[image_id]
        start = kw.get("start_coords")
        if start is not None:  # streaming: XYZCT, t is index 4
            t = start[4]
            arr = arr[t : t + 1]
        return (None, arr)

    # Non-streaming reference: load all fields, stitch the full stack.
    with patch("omero_screen.loops.get_image", side_effect=whole_field):
        stacked, positions, _ = _load_well_fields(
            MagicMock(), _streaming_well(field_data), metadata, 1, flatfield
        )
        reference = _stitch_well(stacked, positions)

    # Streaming path on the same data.
    with patch("omero_screen.loops.get_image", side_effect=whole_field):
        canvas, _, _, tile_h, tile_w = _load_and_stitch_streaming(
            MagicMock(), _streaming_well(field_data), metadata, flatfield
        )

    assert canvas.dtype == np.float32
    assert canvas.shape == reference.shape
    assert (tile_h, tile_w) == (5, 5)
    np.testing.assert_array_equal(canvas, reference)
