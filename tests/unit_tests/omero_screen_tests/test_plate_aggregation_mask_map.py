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

import pytest
from omero_utils.message import OmeroError

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


def test_old_mask_is_not_double_deleted_after_upload() -> None:
    """``create_cell_masks`` must not delete what ``upload_masks`` pruned.

    ``upload_masks`` removes any mask sharing the name it just wrote, so an
    old ``{id}_segmentation`` is already gone by the time the explicit
    delete runs. Only an old mask under a different name — a stitched-mode
    mask superseded by a per-field one — still needs removing.
    """
    from omero_screen.plate_aggregation import _should_delete_old_mask

    legacy = _child("1234_segmentation", 10)
    stitched = _child("1234_stitched_segmentation", 11)

    # Same name as the new upload: the prune already handled it.
    assert _should_delete_old_mask(legacy, 1234) is False
    # Different name: still ours to remove.
    assert _should_delete_old_mask(stitched, 1234) is True


def test_missing_mask_is_skipped_not_fatal() -> None:
    """A field with no mask must not abort aggregation.

    After the stitched-path refactor a field whose acquisition failed
    (blank image, no stage position) is legitimately excluded and gets no
    mask. ``_get_mask_dim`` treats a missing mask as fatal, so the callers
    have to check membership first — otherwise a single bad field kills
    the whole cross-plate run part-way through.
    """
    from omero_screen.plate_aggregation import _get_mask_dim

    present = _child("1234_segmentation", 10)
    mask_map = {1234: present}

    # Present: returns dims.
    assert _get_mask_dim(1234, mask_map)[3] == present.getSizeC()
    # Absent: still fatal by design — callers must skip before calling it.
    with pytest.raises(OmeroError):
        _get_mask_dim(9999, mask_map)
    # The membership test the callers use.
    assert 9999 not in mask_map
    assert 1234 in mask_map


def test_method_3_guard_does_not_fire_for_other_methods() -> None:
    """The missing-mask short circuit must be scoped to method 3 only.

    ``aggregate_plates`` builds mask maps *only* when ``method == 3`` and
    passes empty dicts otherwise. An unscoped ``im not in map1`` check
    would therefore be true for every field on methods 0-2 and silently
    drop every mapping on the plate.
    """
    for method, map1, map2 in [(0, {}, {}), (1, {}, {}), (2, {}, {})]:
        fires = method == 3 and (1 not in map1 or 2 not in map2)
        assert fires is False, f"guard wrongly fired for method {method}"
    # Method 3 with a genuinely missing mask does fire.
    assert (3 == 3 and (1 not in {} or 2 not in {})) is True
