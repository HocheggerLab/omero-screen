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

**Corpses** are judged frame by frame (:func:`corpse_detections`), because the
tracker often hands a corpse's track to a live cell that touches it, so the
track as a whole looks alive. A detection is a corpse if it expresses no
reporter (PIP and geminin both below the expressing levels) *and* it either
sits still (median step over the surrounding frames below ``still_px``) or is
pyknotic (area below ``small`` x the well's median, nuclear stain above
``bright`` x the well's median), for at least ``min_run`` consecutive frames
of its raw track. Mitotic cells (geminin high) and new daughters (PIP returns
at once) are not reporter-negative, so they are kept.
"""

from __future__ import annotations

from typing import Any

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


def corpse_detections(
    det: pd.DataFrame,
    pip: str = "pip",
    geminin: str = "geminin",
    dna: str = "dapi",
    pip_low: float = 0.35,
    gem_high: float = 2.0,
    still_px: float = 3.0,
    window: int = 2,
    small: float = 0.6,
    bright: float = 1.5,
    min_run: int = 3,
) -> pd.Series:
    """Boolean per detection: a dead cell (see the module docstring).

    Args:
        det: Detections with ``track_id_raw, timepoint, y, x, area`` and the
            reporter (and, if present, nuclear-stain) columns.
        pip: PIP reporter column.
        geminin: Geminin reporter column.
        dna: Nuclear-stain column (pyknosis is skipped without it).
        pip_low: PIP expressing level, as in :func:`debris_tracks`.
        gem_high: Geminin expressing level, as in :func:`debris_tracks`.
        still_px: Median step (pixels per frame) below which it sits still.
        window: Frames either side over which the step is taken.
        small: Pyknotic area, as a fraction of the well's median area.
        bright: Pyknotic nuclear stain, as a multiple of the well's median.
        min_run: Consecutive frames of a raw track the call must hold for.
    """
    out = pd.Series(False, index=det.index)
    if pip not in det or geminin not in det or det.empty:
        return out
    pip_thr = pip_low * float(det[pip].quantile(0.75))
    gem_thr = gem_high * float(det[geminin].quantile(0.25))
    negative = (det[pip] <= pip_thr) & (det[geminin] <= gem_thr)
    d = det.sort_values(["track_id_raw", "timepoint"])
    g = d.groupby("track_id_raw")
    step = pd.Series(
        np.hypot(g["y"].diff(), g["x"].diff()), index=d.index, dtype=float
    )
    still = (
        step.groupby(d["track_id_raw"])
        .transform(
            lambda s: s.rolling(
                2 * window + 1, center=True, min_periods=2
            ).median()
        )
        .reindex(det.index)
        < still_px
    )
    pyknotic = pd.Series(False, index=det.index)
    if dna in det and det[dna].notna().any():
        pyknotic = (det["area"] < small * float(det["area"].median())) & (
            det[dna] > bright * float(det[dna].median())
        )
    cand = (negative & (still | pyknotic)).reindex(d.index)
    # Keep only runs of at least ``min_run`` consecutive frames per raw track.
    brk = (
        (cand != cand.shift())
        | (d["track_id_raw"] != d["track_id_raw"].shift())
        | (d["timepoint"].diff() != 1)
    ).cumsum()
    run = cand.groupby(brk).transform("size")
    out.loc[d.index] = cand & (run >= min_run)
    return out


def drop_debris(
    det: pd.DataFrame, corpses: bool = True, **kwargs: Any
) -> tuple[pd.DataFrame, set[int]]:
    """``(detections without debris and corpses, debris track ids)``.

    Debris tracks are removed whole; a daughter whose parent was debris
    becomes a founder. With ``corpses`` the detections
    :func:`corpse_detections` calls dead are removed as well (frame by frame),
    so no live track can be linked into them.
    """
    ids = debris_tracks(det, **kwargs)
    clean = det
    if ids:
        clean = det[~det["track_id_raw"].isin(ids)].copy()
        clean.loc[
            clean["parent_track_id_raw"].isin(ids), "parent_track_id_raw"
        ] = 0
    if corpses:
        dead = corpse_detections(clean)
        if dead.any():
            clean = clean[~dead].copy()
    return clean, ids
