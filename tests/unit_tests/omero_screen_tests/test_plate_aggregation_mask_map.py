"""Tests for ``plate_aggregation._get_mask_map`` mask selection.

The segmentation dataset can hold more than one mask claiming the same
source image: same-named duplicates left by re-runs (before
``upload_masks`` pruned them), and the stitched/legacy pair a plate
analysed both ways carries by design. Selection used to be whatever
``listChildren()`` yielded last, so aggregation could silently use a mask
from an earlier run. These tests pin the precedence down.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from omero_screen.plate_aggregation import _get_mask_map


def _child(name: str, image_id: int) -> MagicMock:
    child = MagicMock()
    child.getName.return_value = name
    child.getId.return_value = image_id
    return child


def _mask_map(children: list[MagicMock]) -> dict[int, MagicMock]:
    conn = MagicMock()
    dataset = MagicMock()
    dataset.listChildren.return_value = children
    conn.getObject.return_value = dataset
    with patch("omero_screen.plate_aggregation.PlateDataset") as plate_dataset:
        plate_dataset.return_value.dataset_id = 7
        return _get_mask_map(conn, 1)


def test_stitched_mask_beats_legacy_regardless_of_order() -> None:
    """A plate analysed per-field then stitched resolves to the stitched mask."""
    legacy = _child("1234_segmentation", 99)
    stitched = _child("1234_stitched_segmentation", 11)
    # Stitched listed first: a lower id must not lose to the legacy mask.
    assert _mask_map([stitched, legacy]) == {1234: stitched}
    assert _mask_map([legacy, stitched]) == {1234: stitched}


def test_newest_wins_among_same_named_duplicates() -> None:
    """Duplicate uploads resolve to the highest id, not to iteration order."""
    old = _child("1234_segmentation", 10)
    new = _child("1234_segmentation", 30)
    assert _mask_map([new, old]) == {1234: new}
    assert _mask_map([old, new]) == {1234: new}


def test_unrelated_and_unparseable_children_are_skipped() -> None:
    """Non-mask images are ignored; a bad name warns instead of raising."""
    mask = _child("1234_segmentation", 10)
    result = _mask_map(
        [
            mask,
            _child("flatfield_correction", 20),
            _child("notanumber_segmentation", 30),
        ]
    )
    assert result == {1234: mask}
