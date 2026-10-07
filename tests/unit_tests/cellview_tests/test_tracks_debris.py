"""Debris: stationary, reporter-negative tracks are hidden, never deleted."""

import pandas as pd
from cellview.tracks.debris import debris_tracks, drop_debris


def _det() -> pd.DataFrame:
    rows = []
    for t in range(20):
        rows.append((t, 1, 0, 100.0, 100.0 + 10 * t, 1000.0, 100.0))  # moving cell
        rows.append((t, 2, 0, 500.0, 500.0, 0.0, 0.0))  # stationary, no reporter: debris
        rows.append((t, 3, 0, 900.0, 900.0, 1000.0, 600.0))  # stationary but expressing: kept
    rows += [(t, 4, 0, 50.0, 50.0, 0.0, 0.0) for t in range(3)]  # too short to judge: kept
    rows += [(t, 5, 2, 520.0, 500.0 + 8 * t, 1000.0, 100.0) for t in range(20, 25)]  # daughter of debris
    return pd.DataFrame(rows, columns=["timepoint", "track_id_raw", "parent_track_id_raw", "y", "x", "pip", "geminin"])


def test_only_stationary_reporter_negative_long_tracks_are_debris() -> None:
    """Moving cells, expressing objects and short tracks are not debris."""
    assert debris_tracks(_det()) == {2}


def test_drop_debris_removes_whole_tracks_and_orphans_daughters() -> None:
    """Debris detections leave the table; a daughter of debris becomes a founder."""
    clean, ids = drop_debris(_det())
    assert ids == {2} and 2 not in set(clean.track_id_raw)
    assert (clean.loc[clean.track_id_raw == 5, "parent_track_id_raw"] == 0).all()
    assert len(_det()) - len(clean) == 20  # the input is not modified
