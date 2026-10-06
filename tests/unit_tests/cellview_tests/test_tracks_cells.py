"""Per-frame cell paths: pieces combined, gaps interpolated, phases called."""

import numpy as np
import pandas as pd
from cellview.tracks.cells import add_phases, cell_frames
from cellview.tracks.edit import Curated


def _det(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "timepoint",
            "label",
            "track_id_raw",
            "area",
            "y",
            "x",
            "pip",
            "geminin",
        ],
    )


def test_pieces_are_combined_and_gaps_interpolated() -> None:
    """Two pieces in one frame make one nucleus; a missing frame becomes a gap row."""
    det = _det(
        [
            (0, 1, 1, 100.0, 10.0, 10.0, 1000.0, 100.0),
            (1, 1, 1, 60.0, 12.0, 12.0, 1000.0, 100.0),
            (1, 2, 2, 40.0, 12.0, 22.0, 500.0, 100.0),
            (3, 1, 1, 100.0, 16.0, 16.0, 1000.0, 100.0),
        ]
    )
    cur = Curated(
        tracks={(0, 1): 1, (1, 1): 1, (1, 2): 1, (3, 1): 1},
        parents={1: 0},
        next_id=3,
    )
    path = cell_frames(cur, det, (0, 1))
    assert list(path.timepoint) == [0, 1, 2, 3]
    f1 = path.set_index("timepoint").loc[1]
    assert f1.area == 100 and f1.n_pieces == 2 and f1.label == 1
    assert f1.x == np.average([12, 22], weights=[60, 40])
    assert f1.pip == np.average([1000, 500], weights=[60, 40])
    gap = path.set_index("timepoint").loc[2]
    assert gap.gap and gap.label == 0 and np.isnan(gap.area) and gap.y == 14.0


def test_phases_follow_the_reporters() -> None:
    """G1 (PIP high, geminin low) then S (PIP low) then G2 (both high)."""
    rows = [(t, 1, 1, 100.0, 0.0, float(t), 1000.0, 100.0) for t in range(10)]
    rows += [
        (t, 1, 1, 100.0, 0.0, float(t), 50.0, 400.0) for t in range(10, 20)
    ]
    rows += [
        (t, 1, 1, 100.0, 0.0, float(t), 1000.0, 800.0) for t in range(20, 30)
    ]
    det = _det(rows)
    cur = Curated(
        tracks={(t, 1): 1 for t in range(30)}, parents={1: 0}, next_id=2
    )
    phases = add_phases(cell_frames(cur, det, (0, 1)), det).phase
    runs = phases[phases.ne(phases.shift())].tolist()
    assert runs == ["G1", "S", "G2"]
