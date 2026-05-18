"""Shared fixtures for zarr_cache tests.

All tests run against an isolated ``OMERO_SCREEN_CACHE_PATH`` set to a
per-test temp directory, so we never touch the user's real cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the zarr cache to a tmp dir for every test."""
    monkeypatch.setenv("OMERO_SCREEN_CACHE_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def synth_well_data():
    """Factory returning (image_tcyx, nuc_tyx, cell_tyx) of small size.

    The arrays are small enough to write fast (~ms) but large enough
    to exercise chunking (256 spatial chunk size) and pyramid levels
    (3 levels via 2× downscale).
    """

    def _make(
        *,
        t: int = 1,
        c: int = 2,
        h: int = 512,
        w: int = 512,
        seed: int = 0,
        n_cells: int = 5,
    ):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 2**16, size=(t, c, h, w), dtype=np.uint16)
        nuc = np.zeros((t, h, w), dtype=np.uint32)
        cell = np.zeros((t, h, w), dtype=np.uint32)
        # A few square cells so we have non-zero labels.
        for label_id in range(1, n_cells + 1):
            cy, cx = rng.integers(50, h - 50), rng.integers(50, w - 50)
            nuc[:, cy - 10 : cy + 10, cx - 10 : cx + 10] = label_id
            cell[:, cy - 20 : cy + 20, cx - 20 : cx + 20] = label_id
        return image, nuc, cell

    return _make
