"""Storage and lazy application of track corrections beside the zarr cache."""

from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from omero_screen.track_correction import CorrectionResult

from omero_screen_napari.zarr_cache import correction


@pytest.fixture
def result() -> CorrectionResult:
    """Frame 0: labels 1 and 2 are one nucleus (track 1), 3 is debris.
    Frame 1: label 4 continues track 1, label 5 is a new track 7.
    """
    table = pd.DataFrame(
        {
            "timepoint": [0, 0, 0, 1, 1],
            "label": [1, 2, 3, 4, 5],
            "track_id": [1, 1, 0, 1, 7],
        }
    )
    return CorrectionResult(
        table=table,
        parents={1: 0, 7: 0},
        events=pd.DataFrame({"rule": ["fragment"], "pass": [1]}),
        debris={3},
    )


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "plate_1.zarr"
    monkeypatch.setattr(correction, "plate_zarr_path", lambda _pid: path)
    return path


def test_save_and_load_round_trip(store: Path, result: CorrectionResult) -> None:
    assert not correction.has_correction(1, "C2")
    out = correction.save_correction(1, "C2", result, pass1={1: 0, 2: 1, 3: 0})
    assert out == store.with_suffix(".corrections")

    loaded = correction.load_correction(1, "C2")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded.table, result.table)
    assert loaded.parents == result.parents
    assert loaded.debris == {3}
    assert correction.first_pass_lineage(1, "C2") == {1: 0, 2: 1, 3: 0}
    assert correction.load_correction(1, "C3") is None


def test_corrected_pyramid_relabels_every_level(result: CorrectionResult) -> None:
    frame0 = np.array([[1, 2], [3, 0]], np.uint32)
    frame1 = np.array([[4, 5], [9, 0]], np.uint32)  # 9: unknown label
    level0 = da.from_array(np.stack([frame0, frame1]), chunks=(1, 1, 2))
    level1 = level0[:, ::2, ::2]

    out = correction.corrected_pyramid([level0, level1], result)

    np.testing.assert_array_equal(
        out[0].compute(), [[[1, 1], [0, 0]], [[1, 7], [0, 0]]]
    )
    np.testing.assert_array_equal(out[1].compute(), [[[1]], [[1]]])


def test_correction_lut(result: CorrectionResult) -> None:
    lut = correction.correction_lut(result, 2)
    assert lut.shape == (2, 6)
    assert lut[0, 2] == 1 and lut[0, 3] == 0 and lut[1, 5] == 7
