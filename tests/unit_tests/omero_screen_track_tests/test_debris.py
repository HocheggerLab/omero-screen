"""Debris: stationary, reporter-negative tracks are hidden, never deleted."""

import pandas as pd
from omero_screen_track.debris import corpse_detections, debris_tracks, drop_debris


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
    clean, ids = drop_debris(_det(), corpses=False)
    assert ids == {2} and 2 not in set(clean.track_id_raw)
    assert (clean.loc[clean.track_id_raw == 5, "parent_track_id_raw"] == 0).all()
    assert len(_det()) - len(clean) == 20  # the input is not modified


def _corpse_det() -> pd.DataFrame:
    """Track 7 is a corpse t0-9, then a live cell takes its track (moving,
    expressing); track 8 condenses (small, DNA-bright, no reporter) t10-14 while
    moving; track 9 is a mitotic cell (small, bright, geminin high); track 10
    is a short reporter-negative pause (2 frames)."""
    rows = []
    for t in range(20):
        live = t >= 10
        rows.append((t, 7, 0, 300.0, 300.0 + (10 * t if live else 0), 1000.0 if live else 0.0, 100.0, 500.0, 1700.0))
        pyk = 10 <= t < 15
        rows.append((t, 8, 0, 600.0, 100.0 + 10 * t, 0.0 if pyk else 1000.0, 50.0, 200.0 if pyk else 500.0, 5000.0 if pyk else 1700.0))
        mit = 10 <= t < 14
        rows.append((t, 9, 0, 900.0, 100.0 + 10 * t, 200.0, 1500.0 if mit else 400.0, 200.0 if mit else 500.0, 5000.0 if mit else 1700.0))
        pause = t in (5, 6)
        rows.append((t, 10, 0, 1200.0, 100.0 + (0 if pause else 10 * t), 0.0 if pause else 1000.0, 100.0, 500.0, 1700.0))
    return pd.DataFrame(rows, columns=["timepoint", "track_id_raw", "parent_track_id_raw", "y", "x", "pip", "geminin", "area", "dapi"])


def test_corpses_are_called_frame_by_frame() -> None:
    """A still corpse and a pyknotic nucleus are dead; the live cell that takes
    the corpse's track, a mitosis and a two-frame pause are not."""
    det = _corpse_det()
    dead = det[corpse_detections(det)]
    by_track = dead.groupby("track_id_raw")["timepoint"].apply(list).to_dict()
    assert by_track[7] == list(range(0, 10))
    assert by_track[8] == list(range(10, 15))
    assert 9 not in by_track and 10 not in by_track


def test_drop_debris_removes_corpse_frames_only() -> None:
    """Corpse frames leave the table; the rest of the track stays."""
    clean, _ = drop_debris(_corpse_det())
    assert sorted(clean.loc[clean.track_id_raw == 7, "timepoint"]) == list(range(10, 20))
