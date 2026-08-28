"""Tests for canvas-offset loading in the zarr builder.

``canvas.csv`` is attached to a well by a ``--stitch`` run. A well without
it must fail the build rather than have its offsets recomputed from stage
positions, even though that recomputation is exact.

The reason is what a missing attachment turns out to mean in practice. On
plate 4054 the two wells lacking one (A2, D1) were exactly the two with a
field whose acquisition failed — a blank image with no stage position —
and both wells' masks were **present but empty**: 21 mask images each,
zero labels in them, against 400-9800 cells per tile in every healthy
well. Mask names and map annotations both report such a well as
segmented, so nothing else in the system notices. Deriving the offsets
would build a zarr of blank labels and hide it; the hard failure is the
only remaining signal.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omero_screen_napari.zarr_cache.builder import _load_canvas_offsets
from omero_utils.message import PlateDataError

MODULE = "omero_screen_napari.zarr_cache.builder"


def _well(n_fields: int = 4, missing_position: int | None = None):
    """A well with ``n_fields`` samples, optionally one lacking a position."""
    samples = []
    for i in range(n_fields):
        ws = MagicMock()
        if i == missing_position:
            ws.getPosX.return_value = None
            ws.getPosY.return_value = None
        else:
            ws.getPosX.return_value.getValue.return_value = -50.2 + i * 0.0013
            ws.getPosY.return_value.getValue.return_value = 35.7
        samples.append(ws)
    well = MagicMock()
    well.listChildren.return_value = samples
    well.countWellSample.return_value = n_fields
    well.getWellPos.return_value = "A2"
    well.getId.return_value = 65788
    return well


def test_attachment_is_used_when_present() -> None:
    """The happy path: offsets come straight from the attachment."""
    frame = pd.DataFrame({"ox": [11, 22, 33, 44], "oy": [55, 66, 77, 88]})
    with (
        patch(f"{MODULE}.get_file_attachments", return_value=[MagicMock()]),
        patch(f"{MODULE}.parse_csv_data", return_value=frame),
    ):
        offsets = _load_canvas_offsets(_well())
    assert offsets.tolist() == [[11, 55], [22, 66], [33, 77], [44, 88]]


def test_missing_attachment_fails_the_build() -> None:
    """No canvas.csv must raise — never silently derive the offsets."""
    with patch(f"{MODULE}.get_file_attachments", return_value=None):
        with pytest.raises(PlateDataError, match="Missing stitched canvas"):
            _load_canvas_offsets(_well())


def test_error_names_the_field_that_has_no_stage_position() -> None:
    """The message must point at the root cause, not just the symptom.

    The original error said only "Missing stitched canvas offsets", which
    sent us looking at the zarr builder when the real problem was a failed
    acquisition and an empty segmentation.
    """
    well = _well(n_fields=21, missing_position=11)
    with patch(f"{MODULE}.get_file_attachments", return_value=None):
        with pytest.raises(PlateDataError) as exc:
            _load_canvas_offsets(well)
    msg = str(exc.value)
    assert "[11]" in msg
    assert "failed segmentation" in msg
    assert "re-run" in msg.lower()


def test_error_suggests_a_stitch_run_when_positions_are_fine() -> None:
    """A well with every position intact just never had a --stitch run."""
    with patch(f"{MODULE}.get_file_attachments", return_value=None):
        with pytest.raises(PlateDataError) as exc:
            _load_canvas_offsets(_well())
    assert "--stitch" in str(exc.value)


def test_unreadable_attachment_fails() -> None:
    """A corrupt CSV is not silently replaced either."""
    with (
        patch(f"{MODULE}.get_file_attachments", return_value=[MagicMock()]),
        patch(f"{MODULE}.parse_csv_data", return_value=None),
    ):
        with pytest.raises(PlateDataError, match="Failed to load"):
            _load_canvas_offsets(_well())


def test_wrong_length_attachment_fails() -> None:
    """A csv whose row count disagrees with the field count is not trusted."""
    with (
        patch(f"{MODULE}.get_file_attachments", return_value=[MagicMock()]),
        patch(
            f"{MODULE}.parse_csv_data",
            return_value=pd.DataFrame({"ox": [0], "oy": [0]}),
        ),
    ):
        with pytest.raises(PlateDataError, match="Incorrect size"):
            _load_canvas_offsets(_well(n_fields=9))
