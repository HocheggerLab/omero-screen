"""Tests for duplicate segmentation-mask handling.

``upload_masks`` used to create a new mask image on every run without
removing the previous one, so re-analysing a plate accumulated same-named
masks in the segmentation dataset. Only the map annotation on the source
image was repointed; anything resolving masks *by name* could then pick a
mask from an earlier run. These tests cover the pruning helper and the
create → repoint → prune ordering that keeps a mask reachable at all times.

Every OMERO object is mocked, so these run offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from omero_utils.images import prune_duplicate_masks, upload_masks


def _child(name: str, image_id: int) -> MagicMock:
    child = MagicMock()
    child.getName.return_value = name
    child.getId.return_value = image_id
    return child


def _conn_with_children(children: list[MagicMock]) -> MagicMock:
    conn = MagicMock()
    dataset = MagicMock()
    dataset.listChildren.return_value = children
    conn.getObject.return_value = dataset
    return conn


def test_prune_keeps_named_survivor_and_deletes_the_rest() -> None:
    """With an explicit keep_id, every other same-named mask goes."""
    conn = _conn_with_children(
        [
            _child("1234_segmentation", 10),
            _child("1234_segmentation", 20),
            _child("1234_segmentation", 30),
        ]
    )

    deleted = prune_duplicate_masks(conn, 7, "1234_segmentation", keep_id=30)

    assert deleted == [10, 20]
    conn.deleteObjects.assert_called_once_with(
        "Image", [10, 20], deleteAnns=True, wait=True
    )


def test_prune_without_keep_id_keeps_the_newest() -> None:
    """Sweeping an existing plate keeps the highest id — the latest upload."""
    conn = _conn_with_children(
        [
            _child("1234_segmentation", 30),
            _child("1234_segmentation", 10),
        ]
    )

    assert prune_duplicate_masks(conn, 7, "1234_segmentation") == [10]


def test_prune_matches_names_exactly() -> None:
    """``_segmentation`` must never match ``_stitched_segmentation``.

    The two coexist by design — a stitched re-run leaves the per-field
    masks in place — so a suffix match here would delete live data.
    """
    conn = _conn_with_children(
        [
            _child("1234_segmentation", 10),
            _child("1234_stitched_segmentation", 11),
            _child("99_segmentation", 12),
        ]
    )

    assert (
        prune_duplicate_masks(conn, 7, "1234_segmentation", keep_id=10) == []
    )
    conn.deleteObjects.assert_not_called()


def test_prune_is_a_no_op_without_duplicates() -> None:
    """A clean dataset triggers no delete call at all."""
    conn = _conn_with_children([_child("1234_segmentation", 10)])

    assert prune_duplicate_masks(conn, 7, "1234_segmentation") == []
    conn.deleteObjects.assert_not_called()


def test_prune_dry_run_reports_without_deleting() -> None:
    """Dry run is what makes a one-off sweep safe to inspect first."""
    conn = _conn_with_children(
        [
            _child("1234_segmentation", 10),
            _child("1234_segmentation", 20),
        ]
    )

    assert prune_duplicate_masks(
        conn, 7, "1234_segmentation", dry_run=True
    ) == [10]
    conn.deleteObjects.assert_not_called()


def test_upload_masks_prunes_previous_upload() -> None:
    """A re-run leaves exactly one mask behind, and repoints before deleting.

    The ordering matters: if the annotation still referenced the old mask
    when it was deleted, a crash in between would leave the source image
    pointing at nothing.
    """
    old = _child("55_segmentation", 100)
    new = _child("55_segmentation", 200)
    conn = _conn_with_children([old, new])
    conn.createImageFromNumpySeq.return_value = new

    source = MagicMock()
    source.getId.return_value = 55

    calls: list[str] = []
    with (
        patch(
            "omero_utils.images.delete_map_annotation",
            side_effect=lambda *a, **k: calls.append("delete_ann"),
        ),
        patch(
            "omero_utils.images.add_map_annotations",
            side_effect=lambda *a, **k: calls.append("add_ann"),
        ),
    ):
        conn.deleteObjects.side_effect = lambda *a, **k: calls.append("delete")
        upload_masks(conn, 7, source, np.zeros((1, 4, 4), dtype=np.uint16))

    assert calls == ["delete_ann", "add_ann", "delete"]
    conn.deleteObjects.assert_called_once_with(
        "Image", [100], deleteAnns=True, wait=True
    )
