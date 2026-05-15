"""Tests for ``omero_utils.images.fetch_stitched_field_masks``.

These run without an OMERO connection — every OMERO object is mocked.
The goal is to validate the annotation lookup, mask reshape (T,Z,Y,X,C →
T,Y,X plus channel split), and the failure path when annotations are
missing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from omero_utils.images import (
    STITCHED_MASK_ANNOTATION_KEY,
    fetch_stitched_field_masks,
)


def _make_well(n_fields: int) -> MagicMock:
    """Return a fake Well whose ``listChildren`` and ``getWellSample`` are
    consistent for ``n_fields`` fields."""
    well = MagicMock()
    well.listChildren.return_value = [MagicMock() for _ in range(n_fields)]

    samples = []
    field_image_ids = list(range(100, 100 + n_fields))
    for fid in field_image_ids:
        img = MagicMock()
        img.getId.return_value = fid
        ws = MagicMock()
        ws.getImage.return_value = img
        samples.append(ws)
    well.getWellSample = lambda n: samples[n]
    well._field_image_ids = field_image_ids
    return well


def _patch_anns_and_image(
    nuclei_only: bool = False, n_fields: int = 2, t: int = 1, h: int = 64
) -> tuple[MagicMock, dict[int, int], dict[int, np.ndarray]]:
    """Patch ``parse_annotations`` and ``get_image`` to return predictable
    mask arrays for each mocked field image."""
    well = _make_well(n_fields)
    # Annotation per field image points to a synthetic mask image id.
    mask_id_lookup = {
        fid: 1000 + fid for fid in well._field_image_ids
    }  # field_image_id → mask_image_id

    # Mask arrays per mask_id: shape (T, Z=1, Y, X, C).
    c = 1 if nuclei_only else 2
    mask_arrays: dict[int, np.ndarray] = {}
    for fid, mid in mask_id_lookup.items():
        arr = np.zeros((t, 1, h, h, c), dtype=np.uint16)
        # Tag each field with a distinct nucleus pixel value so we can
        # confirm ordering in the test.
        arr[..., 0] = fid
        if c == 2:
            arr[..., 1] = fid + 500
        mask_arrays[mid] = arr
    return well, mask_id_lookup, mask_arrays


def test_two_channel_round_trip():
    well, mid_lookup, mask_arrays = _patch_anns_and_image(
        nuclei_only=False, n_fields=2
    )

    def fake_parse(img, ns=None):
        fid = img.getId()
        return {STITCHED_MASK_ANNOTATION_KEY: str(mid_lookup[fid])}

    def fake_get_image(conn, image_id):
        return None, mask_arrays[image_id]

    conn = MagicMock()
    with (
        patch(
            "omero_utils.images.parse_annotations", side_effect=fake_parse
        ),
        patch("omero_utils.images.get_image", side_effect=fake_get_image),
    ):
        nuclei, cells, source_ids = fetch_stitched_field_masks(conn, well)

    assert source_ids == well._field_image_ids
    assert len(nuclei) == 2
    assert len(cells) == 2
    # Per-field tag preserved.
    assert nuclei[0].max() == 100
    assert nuclei[1].max() == 101
    assert cells[0].max() == 600
    assert cells[1].max() == 601
    # Shape: (T, Y, X).
    assert nuclei[0].shape == (1, 64, 64)


def test_nucleus_only_returns_none_for_cells():
    well, mid_lookup, mask_arrays = _patch_anns_and_image(
        nuclei_only=True, n_fields=2
    )

    def fake_parse(img, ns=None):
        return {STITCHED_MASK_ANNOTATION_KEY: str(mid_lookup[img.getId()])}

    def fake_get_image(conn, image_id):
        return None, mask_arrays[image_id]

    conn = MagicMock()
    with (
        patch(
            "omero_utils.images.parse_annotations", side_effect=fake_parse
        ),
        patch("omero_utils.images.get_image", side_effect=fake_get_image),
    ):
        nuclei, cells, _ = fetch_stitched_field_masks(conn, well)

    assert len(nuclei) == 2
    assert cells == [None, None]


def test_missing_annotation_raises_key_error():
    well = _make_well(2)

    with (
        patch(
            "omero_utils.images.parse_annotations", return_value={}
        ),  # no stitched annotation on any field
        patch("omero_utils.images.get_image"),
    ):
        with pytest.raises(KeyError):
            fetch_stitched_field_masks(MagicMock(), well)


def test_rejects_z_greater_than_one():
    well, mid_lookup, mask_arrays = _patch_anns_and_image(n_fields=1)
    # Replace the (T, Z=1, Y, X, C) array with Z=2.
    fid = well._field_image_ids[0]
    mid = mid_lookup[fid]
    mask_arrays[mid] = np.zeros((1, 2, 16, 16, 1), dtype=np.uint16)

    with (
        patch(
            "omero_utils.images.parse_annotations",
            return_value={STITCHED_MASK_ANNOTATION_KEY: str(mid)},
        ),
        patch(
            "omero_utils.images.get_image",
            return_value=(None, mask_arrays[mid]),
        ),
    ):
        with pytest.raises(ValueError, match="Z="):
            fetch_stitched_field_masks(MagicMock(), well)


def test_rejects_unexpected_channel_count():
    well, mid_lookup, mask_arrays = _patch_anns_and_image(n_fields=1)
    fid = well._field_image_ids[0]
    mid = mid_lookup[fid]
    mask_arrays[mid] = np.zeros((1, 1, 16, 16, 3), dtype=np.uint16)  # C=3

    with (
        patch(
            "omero_utils.images.parse_annotations",
            return_value={STITCHED_MASK_ANNOTATION_KEY: str(mid)},
        ),
        patch(
            "omero_utils.images.get_image",
            return_value=(None, mask_arrays[mid]),
        ),
    ):
        with pytest.raises(ValueError, match="C="):
            fetch_stitched_field_masks(MagicMock(), well)
