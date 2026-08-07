"""Tests for mask reuse and the ``--delete`` clean-slate flag.

A re-run reuses whatever masks the dataset already holds and only
recomputes the measurements; ``--delete`` removes them first so
segmentation runs from scratch. Both the stitched and per-field paths
follow that rule.

The stitched path is the interesting one: it used to re-segment the
canvas unconditionally, which is the expensive step. Reuse recomposes the
stored per-field tiles instead — lossless, because they came from a
single canvas-wide segmentation and so carry globally unique label ids.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from omero_screen.loops import _reuse_stitched_masks

MODULE = "omero_screen.loops"


def _tiles(n: int, side: int = 8, fill: int = 1) -> list[np.ndarray]:
    """``n`` per-field tiles of shape (T=1, side, side)."""
    return [np.full((1, side, side), fill, dtype=np.uint16) for _ in range(n)]


def _offsets(n: int, side: int = 8) -> np.ndarray:
    return np.array([[i * side, 0] for i in range(n)], dtype=np.int_)


def _well() -> MagicMock:
    well = MagicMock()
    well.getWellPos.return_value = "A1"
    return well


def test_reuse_recomposes_stored_masks() -> None:
    """The happy path: stored tiles come back as one canvas mask."""
    with (
        patch(
            f"{MODULE}.resolve_stitched_mask_ids",
            return_value=([1, 2], [9, 8]),
        ),
        patch(
            f"{MODULE}.fetch_stitched_field_masks_trange",
            return_value=(_tiles(2), [None, None]),
        ),
    ):
        result = _reuse_stitched_masks(
            MagicMock(), _well(), [0, 1], _offsets(2)
        )
    assert result is not None
    n_mask, c_mask = result
    assert n_mask.shape == (1, 8, 16)
    assert c_mask is None


def test_no_stored_masks_falls_back_to_segmentation() -> None:
    """A plate never segmented in stitched mode returns None, not an error."""
    with patch(
        f"{MODULE}.resolve_stitched_mask_ids",
        side_effect=KeyError("no Stitched_Segmentation_Mask"),
    ):
        assert (
            _reuse_stitched_masks(MagicMock(), _well(), [0, 1], _offsets(2))
            is None
        )


def test_empty_stored_masks_are_rejected() -> None:
    """Present-but-empty masks must not be reused.

    This is the plate 4054 A2/D1 failure: 21 mask images with correct
    names and annotations, zero labels in any of them. Reusing them would
    carry the empty well into a fresh set of measurements, so the well is
    re-segmented instead.
    """
    with (
        patch(
            f"{MODULE}.resolve_stitched_mask_ids",
            return_value=([1, 2], [9, 8]),
        ),
        patch(
            f"{MODULE}.fetch_stitched_field_masks_trange",
            return_value=(_tiles(2, fill=0), [None, None]),
        ),
    ):
        assert (
            _reuse_stitched_masks(MagicMock(), _well(), [0, 1], _offsets(2))
            is None
        )


def test_partial_cell_mask_coverage_is_rejected() -> None:
    """Mixed coverage is ambiguous, so fall back rather than guess."""
    with (
        patch(
            f"{MODULE}.resolve_stitched_mask_ids",
            return_value=([1, 2], [9, 8]),
        ),
        patch(
            f"{MODULE}.fetch_stitched_field_masks_trange",
            return_value=(_tiles(2), [_tiles(1)[0], None]),
        ),
    ):
        assert (
            _reuse_stitched_masks(MagicMock(), _well(), [0, 1], _offsets(2))
            is None
        )


def test_cell_masks_are_recomposed_when_present() -> None:
    """A two-channel run reuses nucleus and cell masks together."""
    with (
        patch(
            f"{MODULE}.resolve_stitched_mask_ids",
            return_value=([1, 2], [9, 8]),
        ),
        patch(
            f"{MODULE}.fetch_stitched_field_masks_trange",
            return_value=(_tiles(2), _tiles(2, fill=5)),
        ),
    ):
        result = _reuse_stitched_masks(
            MagicMock(), _well(), [0, 1], _offsets(2)
        )
    assert result is not None
    n_mask, c_mask = result
    assert c_mask is not None
    assert c_mask.shape == n_mask.shape


def test_download_failure_falls_back_rather_than_aborting() -> None:
    """Reuse is an optimisation; any problem reading it must not kill the run."""
    with (
        patch(
            f"{MODULE}.resolve_stitched_mask_ids",
            return_value=([1, 2], [9, 8]),
        ),
        patch(
            f"{MODULE}.fetch_stitched_field_masks_trange",
            side_effect=RuntimeError("OMERO went away"),
        ),
    ):
        assert (
            _reuse_stitched_masks(MagicMock(), _well(), [0, 1], _offsets(2))
            is None
        )


def test_delete_existing_clears_masks_before_the_run() -> None:
    """``--delete`` must remove masks before any well is processed.

    Ordering is the whole point: if deletion happened after, the reuse
    lookup would still find the old masks and the flag would be a no-op.
    """
    calls: list[str] = []
    with (
        patch(f"{MODULE}.MetadataParser") as meta,
        patch(f"{MODULE}.PlateDataset") as plate_ds,
        patch(f"{MODULE}.get_cell_model", return_value="cyto3"),
        patch(f"{MODULE}.flatfieldcorr", return_value={}),
        patch(f"{MODULE}._print_device_info"),
        patch(
            f"{MODULE}.delete_masks",
            side_effect=lambda *a, **k: calls.append("delete"),
        ),
        patch(
            f"{MODULE}.process_wells",
            side_effect=lambda *a, **k: (
                calls.append("process"),
                (__import__("pandas").DataFrame(),) * 3,
            )[1],
        ),
    ):
        from omero_screen.loops import plate_loop

        meta.return_value.well_data = {"cell_line": ["RPE"]}
        plate_ds.return_value.dataset_id = 7
        plate_loop(
            MagicMock(), 1, segmentation_mode=True, delete_existing=True
        )

    assert calls == ["delete", "process"]


def test_no_delete_flag_leaves_masks_alone() -> None:
    """The default must never destroy data."""
    with (
        patch(f"{MODULE}.MetadataParser") as meta,
        patch(f"{MODULE}.PlateDataset") as plate_ds,
        patch(f"{MODULE}.get_cell_model", return_value="cyto3"),
        patch(f"{MODULE}.flatfieldcorr", return_value={}),
        patch(f"{MODULE}._print_device_info"),
        patch(f"{MODULE}.delete_masks") as delete,
        patch(
            f"{MODULE}.process_wells",
            return_value=(__import__("pandas").DataFrame(),) * 3,
        ),
    ):
        from omero_screen.loops import plate_loop

        meta.return_value.well_data = {"cell_line": ["RPE"]}
        plate_ds.return_value.dataset_id = 7
        plate_loop(MagicMock(), 1, segmentation_mode=True)

    delete.assert_not_called()
