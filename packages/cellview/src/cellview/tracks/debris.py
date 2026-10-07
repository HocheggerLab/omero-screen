"""Find debris among tracked objects, over each track's whole life.

Living RPE-1 nuclei move a median ~13 px per 20-min frame at 20x; debris and
old corpses do not move and express no reporter. A raw track is debris if its
median frame-to-frame step is below ``still_px`` *and* neither PIP nor
geminin ever rises to an expressing level (90th percentile below the
well-level thresholds the fate walker uses).

Debris is hidden, never deleted: :func:`drop_debris` removes those detections
from the table the repair, the walker and the review tools work on, while the
zarr masks and CellView rows stay as they were. Short tracks (fewer than
``min_frames`` detections) cannot be judged and are kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def debris_tracks(
    det: pd.DataFrame,
    still_px: float = 3.0,
    min_frames: int = 6,
    pip: str = "pip",
    geminin: str = "geminin",
    pip_low: float = 0.35,
    gem_high: float = 2.0,
) -> set[int]:
    """Raw track ids that are stationary and reporter-negative.

    Args:
        det: Detections with ``track_id_raw, timepoint, y, x`` and the reporter columns.
        still_px: Median step (pixels per frame) below which a track is stationary.
        min_frames: Tracks with fewer detections are not judged.
        pip: PIP reporter column.
        geminin: Geminin reporter column.
        pip_low: PIP is expressed above this fraction of the well's PIP upper
            quartile (as in the fate walker).
        gem_high: Geminin is expressed above this multiple of the well's
            geminin lower quartile.
    """
    if pip not in det or geminin not in det or det.empty:
        return set()
    pip_thr = pip_low * float(det[pip].quantile(0.75))
    gem_thr = gem_high * float(det[geminin].quantile(0.25))
    d = det.sort_values(["track_id_raw", "timepoint"])
    step = np.hypot(
        d.groupby("track_id_raw")["y"].diff(),
        d.groupby("track_id_raw")["x"].diff(),
    )
    stats = (
        d.assign(step=step)
        .groupby("track_id_raw")
        .agg(
            n=("timepoint", "size"),
            step=("step", "median"),
            pip90=(pip, lambda s: s.quantile(0.9)),
            gem90=(geminin, lambda s: s.quantile(0.9)),
        )
    )
    hit = (
        (stats["n"] >= min_frames)
        & (stats["step"] < still_px)
        & (stats["pip90"] <= pip_thr)
        & (stats["gem90"] <= gem_thr)
    )
    return {int(t) for t in stats.index[hit]}


def drop_debris(
    det: pd.DataFrame, **kwargs: float
) -> tuple[pd.DataFrame, set[int]]:
    """``(detections without debris, debris track ids)``.

    Debris tracks are removed whole; a daughter whose parent was debris
    becomes a founder.
    """
    ids = debris_tracks(det, **kwargs)  # type: ignore[arg-type]
    if not ids:
        return det, ids
    clean = det[~det["track_id_raw"].isin(ids)].copy()
    clean.loc[
        clean["parent_track_id_raw"].isin(ids), "parent_track_id_raw"
    ] = 0
    return clean, ids
