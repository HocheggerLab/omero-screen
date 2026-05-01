# ------------------------------------------------------------------------------
# Shared pytest fixtures for the cellclass unit test suite.
# ------------------------------------------------------------------------------

"""Shared fixtures for cellclass unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROP_SIZE: int = 50
NUM_CROPS: int = 3
NUM_CHANNELS: int = 2
CHANNELS: list[str] = ["DAPI", "Tub"]
CLASS_NAMES: list[str] = ["mitosis", "interphase", "apoptosis"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_npy_data(
    tmp_path: Path,
    filename: str,
    labels: list[str],
    crop_size: int = CROP_SIZE,
    num_channels: int = NUM_CHANNELS,
    label_region: tuple[int, int, int, int] = (20, 30, 20, 30),
) -> Path:
    """Create a .npy session file used by create_dataset.

    Args:
        tmp_path: Temporary directory in which to write the file.
        filename: Name of the .npy file.
        labels: List of class label strings, one per crop.
        crop_size: Height and width of each square crop.
        num_channels: Number of image channels (YXC layout).
        label_region: (y0, y1, x0, x1) bounding box for the mask label.

    Returns:
        Path to the written .npy file.

    """
    rng = np.random.default_rng(0)
    n = len(labels)
    y0, y1, x0, x1 = label_region

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for _ in range(n):
        img = rng.random((crop_size, crop_size, num_channels), dtype=np.float32)
        mask = np.zeros((crop_size, crop_size), dtype=np.int32)
        mask[y0:y1, x0:x1] = 1
        images.append(img)
        masks.append(mask)

    d = {
        "data": (images, masks),
        "target": labels,
    }
    path = tmp_path / filename
    np.save(path, d)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def npy_file(tmp_path: Path) -> Path:
    """Single .npy session file with 3 crops, 2 classes, 2 channels (YXC).

    The mask label is centred at rows 20:30, cols 20:30 — always inside the
    centre pixel of a 50×50 crop, so ``_binary_mask`` will find it via the
    centre-pixel path.

    Returns:
        Path to the .npy file.

    """
    return _make_npy_data(
        tmp_path,
        "session.npy",
        labels=["mitosis", "interphase", "apoptosis"],
    )


@pytest.fixture
def npy_file_with_unassigned(tmp_path: Path) -> Path:
    """Like npy_file but one crop carries the 'unassigned' label.

    Returns:
        Path to the .npy file.

    """
    return _make_npy_data(
        tmp_path,
        "session_unassigned.npy",
        labels=["mitosis", "unassigned", "apoptosis"],
    )


@pytest.fixture
def data_dir(tmp_path: Path, npy_file: Path) -> Path:
    """Directory containing one .npy file and a valid metadata.json.

    The metadata declares channels=['DAPI', 'Tub'].

    Args:
        tmp_path: Pytest temporary directory.
        npy_file: The npy_file fixture (ensures the file exists in tmp_path).

    Returns:
        Path to the data directory (same as tmp_path).

    """
    meta = {"user_data": {"channels": CHANNELS}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return tmp_path


@pytest.fixture
def npz_file(data_dir: Path) -> Path:
    """Run create_dataset.run() on data_dir and return the path to rois.npz.

    Args:
        data_dir: Directory fixture containing a .npy file and metadata.json.

    Returns:
        Path to the generated rois.npz file.

    """
    import argparse

    from cellclass.bin.create_dataset import run

    args = argparse.Namespace(
        dir=str(data_dir),
        name="rois",
        out=None,
        ignore=["unassigned"],
        duplicates=False,
        channels=[],
        single_label=True,
    )
    run(args)
    return data_dir / "rois.npz"


@pytest.fixture
def train_txt(tmp_path: Path, npz_file: Path) -> Path:
    """Write a train.txt with 2 uncommented lines and 1 comment line.

    Args:
        tmp_path: Pytest temporary directory.
        npz_file: Path to a valid .npz dataset (ensures it exists).

    Returns:
        Path to the train.txt file.

    """
    content = (
        f"{npz_file} --model efficientnetb3s --lr 1e-3\n"
        "# this is a comment line\n"
        f"{npz_file} --model densenet121 --lr 1e-4\n"
    )
    p = tmp_path / "train.txt"
    p.write_text(content)
    return p
