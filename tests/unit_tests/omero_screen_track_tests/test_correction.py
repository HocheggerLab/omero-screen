"""The correction runs with any tracker, without Trackastra or the pipeline (#75)."""

import sys

import numpy as np
import pandas as pd
from omero_screen_track.correction import (
    CorrectionParams,
    correct_tracks,
    merge_detections,
)


def _fragment_well() -> tuple[pd.DataFrame, dict[int, int]]:
    """Track 1 splits into fragments 2 and 3 at t=10; 4 is a neighbour; 9 is debris."""
    rows = []
    for t in range(20):
        if t < 10:
            rows.append((t, 1, 100.0, 50.0, 10.0 + 5 * t, 80.0, 80.0))
        else:
            x = 10.0 + 5 * t
            rows.append((t, 2, 50.0, 46.0, x, 80.0, 80.0))
            rows.append((t, 3, 50.0, 54.0, x, 80.0, 80.0))
        rows.append((t, 4, 100.0, 150.0, 10.0 + 4 * t, 80.0, 80.0))
        rows.append((t, 9, 60.0, 300.0, 300.0, 0.0, 0.0))
    det = pd.DataFrame(
        rows,
        columns=["timepoint", "label", "area", "y", "x", "geminin", "pip"],
    )
    return det, {1: 0, 2: 1, 3: 1, 4: 0, 9: 0}


def test_correct_tracks_with_a_plain_tracker() -> None:
    """A tracker function replaces Trackastra; fragments merge, debris hides."""
    det, parents = _fragment_well()
    calls = []

    def identity(images, masks):  # noqa: ANN001, ANN202
        calls.append(masks.shape)
        tracked = merge_detections(
            det, pd.Series(masks.fn(det["label"].to_numpy())), []
        )[["timepoint", "label"]]
        tracked["track"] = tracked["label"]
        return tracked, {int(t): 0 for t in tracked["track"].unique()}

    masks = np.zeros((20, 4, 4), np.uint32)
    res = correct_tracks(
        masks, masks, parents, {}, identity, CorrectionParams(), det
    )

    assert calls == [(20, 4, 4)]
    final = res.table.set_index(["timepoint", "label"])["track_id"]
    assert final[(12, 2)] == final[(12, 3)] == final[(5, 1)]
    assert final[(5, 9)] == 0
    assert res.debris == {9}
    assert res.frame_map(12) == {2: 1, 3: 1, 4: 4, 9: 0}


def test_package_does_not_import_the_pipeline_or_trackastra() -> None:
    """Importing the correction pulls in neither omero_screen nor trackastra."""
    import subprocess

    code = (
        "import sys, omero_screen_track.correction; "
        "print(sorted(m for m in ('trackastra', 'torch', 'omero_screen."
        "track_correction', 'omero_screen.loops') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"
