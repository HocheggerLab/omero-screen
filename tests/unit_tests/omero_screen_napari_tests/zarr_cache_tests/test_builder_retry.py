"""Transient-OMERO-error retry and disk-cached mask reads in the zarr builder.

A single ``Ice::TimeoutException`` in one dask block used to discard a whole
well after hours of work. These tests pin the two mitigations: block loaders
retry connection-level errors on a fresh connection, and label masks go
through the plate-tagged image disk cache so a rebuild reads them locally.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import Ice
import numpy as np
import pytest
from omero_screen_napari.zarr_cache import builder

_MASK_IDS = [20, 21]
_SOURCE_IDS = [10, 11]
_T, _Y, _X = 3, 4, 4


@pytest.fixture(autouse=True)
def _no_backoff() -> Any:
    """Skip the real retry sleeps."""
    with patch.object(builder.time, "sleep"):
        yield


def _masks() -> dict[int, np.ndarray]:
    rng = np.random.default_rng(1)
    return {
        mid: rng.integers(0, 9, (_T, 1, _Y, _X, 2), dtype=np.uint16)
        for mid in _MASK_IDS
    }


def test_retry_recovers_from_transient_error() -> None:
    """A timeout then a success returns the success, after one retry."""
    calls = MagicMock(side_effect=[Ice.TimeoutException(), "ok"])

    @builder._retry_transient_omero
    def load() -> str:
        return calls()  # type: ignore[no-any-return]

    assert load() == "ok"
    assert calls.call_count == 2


def test_retry_gives_up_after_configured_attempts() -> None:
    """Persistent connection loss re-raises once the attempts are spent."""
    calls = MagicMock(side_effect=Ice.ConnectionLostException())

    @builder._retry_transient_omero
    def load() -> None:
        calls()

    with pytest.raises(Ice.ConnectionLostException):
        load()
    assert calls.call_count == builder._OMERO_BLOCK_ATTEMPTS


def test_retry_does_not_mask_data_errors() -> None:
    """A non-connection error (bad data) propagates on the first attempt."""
    calls = MagicMock(side_effect=ValueError("Z=2; expected Z=1"))

    @builder._retry_transient_omero
    def load() -> None:
        calls()

    with pytest.raises(ValueError):
        load()
    assert calls.call_count == 1


def test_label_block_retries_on_a_fresh_connection() -> None:
    """Each attempt of a block opens its own worker connection."""
    masks = _masks()
    attempts = {"n": 0}

    def flaky_get_image(conn, image_id, start=None, end=None, tag=None):  # type: ignore[no-untyped-def]
        if attempts["n"] == 0:
            attempts["n"] += 1
            raise Ice.InvocationTimeoutException()
        return masks[image_id][start:end]

    omero_conn = MagicMock()
    with (
        patch.object(builder, "get_image", flaky_get_image),
        patch.object(
            builder, "recompose_tiles", lambda fields, _offsets: fields[0]
        ),
    ):
        nuc, cell = builder._load_recompose_label_block(
            MagicMock(),
            omero_conn,
            _MASK_IDS,
            _SOURCE_IDS,
            np.zeros((2, 2), dtype=int),
            0,
            2,
            plate_id=5054,
        )

    assert omero_conn.create_conn.call_count == 2
    np.testing.assert_array_equal(nuc, masks[_MASK_IDS[0]][0:2, 0, ..., 0])
    assert cell is not None
    np.testing.assert_array_equal(cell, masks[_MASK_IDS[0]][0:2, 0, ..., 1])


def test_label_block_reads_masks_through_plate_tagged_cache() -> None:
    """Masks are fetched via the cached ``get_image``, tagged by plate."""
    masks = _masks()
    fake = MagicMock(
        side_effect=lambda conn, image_id, start=None, end=None, tag=None: (
            masks[image_id][start:end]
        )
    )
    with (
        patch.object(builder, "get_image", fake),
        patch.object(
            builder, "recompose_tiles", lambda fields, _offsets: fields[0]
        ),
    ):
        builder._load_recompose_label_block(
            MagicMock(),
            None,
            _MASK_IDS,
            _SOURCE_IDS,
            np.zeros((2, 2), dtype=int),
            1,
            3,
            plate_id=5054,
        )

    assert [c.args[1] for c in fake.call_args_list] == _MASK_IDS
    for c in fake.call_args_list:
        assert c.kwargs == {"start": 1, "end": 3, "tag": 5054}
