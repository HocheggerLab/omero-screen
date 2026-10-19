"""Per-cell fates from repaired PIP-FUCCI tracks.

The unit here is a cell, not a lineage. Every nucleus present at a start frame
is followed forward until its first outcome or until the window closes. When
the cell divides, its record ends: daughters are not followed. Sampling from a
fixed time rather than from a mitosis keeps cells that never divide. Under a
knockdown that blocks proliferation those cells are the phenotype, and a
mitosis-to-mitosis selection would discard them.

PIP-FUCCI phases (Grant et al. 2018):

=====  ==========  ==========
phase  PIP         geminin
=====  ==========  ==========
G1     high        low
S      low         high
G2     high        high
=====  ==========  ==========

Thresholds come from the well itself (:func:`well_thresholds`), so a plate
imaged with different exposure needs no retuning. Signals are smoothed with a
centred rolling median before calling, which removes single-frame segmentation
noise without moving a transition by more than a frame.

Outcomes:

``divided``
    The repaired track ends in a geminin-confirmed mitosis.
``mitotic_exit_no_division``
    Geminin, high for at least ``exit_hold`` frames, collapses but one nucleus
    continues: slippage or failed cytokinesis. A one-frame geminin spike (a
    nucleus compacting) is not a high level to exit from.
``mitotic_arrest``
    Condensed chromatin (high DNA intensity, small area) for at least
    ``arrest_hours`` with no exit by the window end.
``death``
    The nucleus ends pyknotic: for at least ``death_frames`` frames at the end
    of its track it is small and bright (area below ``death_area`` and DNA
    above ``death_dna`` times the cell's own median), away from the image
    border, and not at G2 geminin level (a G2 nucleus condensing is entering
    mitosis). A track that merely ends, or ends shrinking, is ``lost``: on
    5054 dying cells detach and leave no corpse, and a shrinking mask is far
    more often a partial mask, a speck or the image border than a death.
``no_mitosis``
    The window closes before any of the above; the phase at the end is
    recorded in ``end_phase``.
``lost``
    The cell cannot be followed. Treat as censored.
``debris``
    The object does not move (median step under ``still_px`` over the first
    ``still_hours`` of the window) and never expresses a reporter: debris or
    an old corpse, not a living cell. Excluded, not reviewed. Living RPE-1
    nuclei move a median 13 px per 20 min frame at 20x.
``stationary``
    Does not move but expresses a reporter: possibly a cell in mitotic arrest
    or dying. Excluded from the analysis but flagged for review, because in a
    knockdown that may be the phenotype.
``no_reporter``
    Neither PIP nor geminin is expressed in any frame of the cell's history,
    observed for at least ``reporter_hours``: a nucleus the lentiviral reporter
    did not reach. A brief both-low stretch is the early-S dip and does not
    count. It cannot be phase-called and is excluded from the
    analysis (reported per well, not reviewed).

Every cell carries review flags (:data:`FLAGS`) that say why a human should
look at it.

Review flags and outcomes use geminin continuity, not per-frame phase calls.
Geminin accumulates from S to mitosis and is degraded at anaphase, so along one
cell it never falls except in mitosis: a fall without a split marks a wrong
link or a mis-segmented frame (``geminin_jump``). Rises are allowed, because
geminin builds up fast in S. Phases (``phase``, ``start_phase``,
``end_phase``) are still reported, but per-frame calls are unreliable when a
nucleus compacts, and are best made later from the whole track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np
import pandas as pd

#: Reason codes a cell can be flagged with.
FLAGS = (
    "link_ambiguous",  # a bridge had more than one plausible continuation
    "gap",  # a bridge skipped at least one frame
    "fragment",  # nucleus pieces were merged during the window
    "refractory",  # the refractory rule removed a second division
    "geminin_jump",  # geminin fell (or leapt) without a split: wrong link or bad frame
    "outcome_review",  # a rare outcome that decides a phenotype call
    "lost",  # the cell could not be followed to an outcome
    "edge",  # the nucleus came within one diameter of the image border
    "reporter_unclear",  # both reporters low throughout, but observed too briefly to call
)

PHASES = ("G1", "S", "G2")


@dataclass(frozen=True)
class FateParams:
    """Thresholds for :func:`follow_cells`.

    Attributes:
        interval_minutes: Time between frames.
        smooth: Rolling-median window in frames for phase calling.
        pip_low: S phase if smoothed PIP is below this fraction of the well's
            PIP-high reference.
        gem_high: Geminin is high above this multiple of the well's
            geminin-low reference.
        max_gap: Frames a bridge may skip.
        reach: A continuation must start within this multiple of the summed
            equivalent radii.
        area_tol: Allowed area ratio across a bridge (and its inverse).
        marker_tol: Allowed PIP and geminin ratio across a bridge (and inverse).
        exit_drop: Geminin falling below this fraction of its level over the
            preceding ``exit_hold`` frames is a mitotic exit.
        exit_frames: Frames over which that fall must happen.
        exit_hold: Frames geminin must stay high before an exit.
        jump_window: Frames on each side of a step whose geminin medians are
            compared for ``geminin_jump``.
        jump_drop: Flag a fall of geminin by more than this factor ...
        jump_rise: ... or a rise by more than this factor. Geminin is clipped
            at its high threshold first, so G1-level noise never counts.
        condensed_dna: Mitotic chromatin if DNA intensity exceeds this multiple
            of the well median.
        condensed_area: ... and area is below this fraction of the cell's
            median area.
        arrest_hours: Minimum condensed run for a mitotic arrest.
        death_area: Pyknotic if area is below this fraction of the cell's own
            median ...
        death_dna: ... and DNA intensity above this multiple of its own median ...
        death_frames: ... for at least this many frames at the end of the track.
        min_start_area: Start cells smaller than this fraction of the well
            median area are excluded as debris.
        still_px: An object whose median frame-to-frame step over the first
            ``still_hours`` of the window is below this many pixels is
            stationary.
        still_hours: Period over which motion is judged.
        reporter_hours: Minimum observed time with neither reporter expressed
            before a cell is called reporter-negative (longer than the
            early-S dip, when both reporters are low).
    """

    interval_minutes: float = 20.0
    smooth: int = 5
    pip_low: float = 0.35
    gem_high: float = 2.0
    max_gap: int = 3
    reach: float = 3.5
    area_tol: float = 1.6
    marker_tol: float = 2.0
    exit_drop: float = 0.5
    exit_frames: int = 3
    exit_hold: int = 6
    jump_window: int = 3
    jump_drop: float = 1.4
    jump_rise: float = 2.5
    condensed_dna: float = 2.0
    condensed_area: float = 0.6
    arrest_hours: float = 2.0
    death_area: float = 0.6
    death_dna: float = 1.5
    death_frames: int = 3
    min_start_area: float = 0.7
    reporter_hours: float = 24.0
    still_px: float = 3.0
    still_hours: float = 4.0

    def frames(self, hours: float) -> int:
        """Convert hours to frames."""
        return int(round(hours * 60 / self.interval_minutes))


@dataclass(frozen=True)
class Thresholds:
    """Well-level reference levels for phase calling."""

    pip_high: float
    gem_low: float
    dna_median: float
    area_median: float


def well_thresholds(tracks: pd.DataFrame) -> Thresholds:
    """Reference levels from a well's own intensity distributions.

    PIP is high outside S phase and geminin is low in G1, so the upper
    quartile of PIP and the lower quartile of geminin sit in those modes for
    any population that is not fully arrested in S.
    """
    return Thresholds(
        pip_high=float(tracks["pip"].quantile(0.75)),
        gem_low=float(tracks["geminin"].quantile(0.25)),
        dna_median=float(tracks["dna"].median()),
        area_median=float(tracks["area"].median()),
    )


def call_phases(
    cell: pd.DataFrame, thr: Thresholds, params: FateParams
) -> pd.Series:
    """Per-frame PIP-FUCCI phase for one cell, from smoothed signals."""
    pip = (
        cell["pip"].rolling(params.smooth, center=True, min_periods=1).median()
    )
    gem = (
        cell["geminin"]
        .rolling(params.smooth, center=True, min_periods=1)
        .median()
    )
    s_phase = pip < params.pip_low * thr.pip_high
    gem_hi = gem > params.gem_high * thr.gem_low
    return pd.Series(
        np.select([s_phase, gem_hi], ["S", "G2"], "G1"), index=cell.index
    )


def _geminin_jump(
    cell: pd.DataFrame, thr: Thresholds, params: FateParams
) -> int | None:
    """First frame where geminin falls (or leaps) between neighbouring windows.

    Compares the median of ``jump_window`` frames before each step with the
    median after it, on geminin clipped at its high threshold, so G1-level
    noise and one-frame compaction spikes never count.
    """
    gem = (
        cell["geminin"]
        .clip(lower=params.gem_high * thr.gem_low)
        .to_numpy(dtype=float)
    )
    frames = cell.index.to_numpy()
    w = params.jump_window
    for i in range(w, len(gem) - w + 1):
        before, after = np.median(gem[i - w : i]), np.median(gem[i : i + w])
        if (
            before > params.jump_drop * after
            or after > params.jump_rise * before
        ):
            return int(frames[i])
    return None


def _radius(area: float) -> float:
    return float(np.sqrt(area / np.pi))


class _Index:
    """Lookups over a well's repaired tracks."""

    def __init__(self, tracks: pd.DataFrame) -> None:
        # Track ids are integers; the stubs type groupby keys as Scalar.
        self.rows = {
            cast(int, tid): g.sort_values("timepoint").set_index("timepoint")
            for tid, g in tracks.groupby("track_id")
        }
        self.parent = (
            tracks.groupby("track_id")["parent_track_id"].first().to_dict()
        )
        self.divides = {p for p in self.parent.values() if p}
        self.starts: dict[int, list[int]] = {}
        for tid, g in self.rows.items():
            if not self.parent[tid]:
                self.starts.setdefault(int(g.index[0]), []).append(tid)

    def continuations(
        self, tid: int, params: FateParams, claimed: set[int]
    ) -> list[tuple[float, int, int]]:
        """Founders that plausibly continue ``tid``: ``(cost, gap, track)``."""
        last = self.rows[tid].iloc[-1]
        end = int(self.rows[tid].index[-1])
        found = []
        for gap in range(1, params.max_gap + 1):
            for cand in self.starts.get(end + gap, []):
                if cand in claimed or cand == tid:
                    continue
                first = self.rows[cand].iloc[0]
                dist = np.hypot(first.y - last.y, first.x - last.x)
                if dist > params.reach * (
                    _radius(first.area) + _radius(last.area)
                ):
                    continue
                ratios = [
                    first.area / last.area,
                    max(first.pip, 1) / max(last.pip, 1),
                    max(first.geminin, 1) / max(last.geminin, 1),
                ]
                tols = [params.area_tol, params.marker_tol, params.marker_tol]
                if any(
                    r > t or r < 1 / t
                    for r, t in zip(ratios, tols, strict=True)
                ):
                    continue
                cost = dist / (_radius(last.area) + 1) + sum(
                    abs(np.log(r)) for r in ratios
                )
                found.append((float(cost), gap, cand))
        return sorted(found)


def _mitotic_exit(
    cell: pd.DataFrame, thr: Thresholds, params: FateParams
) -> int | None:
    """First frame at which sustained high geminin collapses within one nucleus.

    Geminin must have been above its high threshold for the ``exit_hold``
    frames before the fall, so a one-frame spike is never an exit.
    """
    gem = cell["geminin"].to_numpy(dtype=float)
    frames = cell.index.to_numpy()
    high = params.gem_high * thr.gem_low
    hold = params.exit_hold
    for i in range(hold, len(gem) - params.exit_frames + 1):
        held = gem[i - hold : i]
        if not (held > high).all():
            continue
        before = float(np.median(held))
        after = float(np.median(gem[i : i + params.exit_frames]))
        if after < params.exit_drop * before:
            return int(frames[i])
    return None


def _reporter_status(
    cell: pd.DataFrame, thr: Thresholds, params: FateParams
) -> str:
    """``"positive"``, ``"negative"`` or ``"unclear"`` for one cell's reporters.

    In every phase one reporter is up (PIP outside S, geminin from S to
    mitosis) except a short dip at S entry, when PIP is already degraded and
    geminin has not yet accumulated. Expressing cells on 5054 show that dip in
    ~40% of tracks, usually under 3 h but up to ~20 h in the control, and longer
    under knockdown, where a persistent both-low state may itself be a
    phenotype. So a cell is only called negative if it *never* expresses either
    reporter over its whole observed history (not just the analysis window)
    and was observed for at least ``reporter_hours``. Shorter both-low
    histories are ``unclear``: kept in the analysis and flagged.

    Args:
        cell: Every frame of the cell's tracks, including before the window.
        thr: Well reference levels.
        params: Thresholds.
    """
    pip = (
        cell["pip"].rolling(params.smooth, center=True, min_periods=1).median()
    )
    gem = (
        cell["geminin"]
        .rolling(params.smooth, center=True, min_periods=1)
        .median()
    )
    # Strict comparisons: a well whose lower geminin quartile is 0 must not
    # count a zero signal as expressed.
    expressed = (pip > params.pip_low * thr.pip_high) | (
        gem > params.gem_high * thr.gem_low
    )
    if expressed.any():
        return "positive"
    span = int(cell.index[-1]) - int(cell.index[0]) + 1
    return (
        "negative"
        if span >= params.frames(params.reporter_hours)
        else "unclear"
    )


def _stationary(cell: pd.DataFrame, params: FateParams) -> bool:
    """True if the object barely moves over the start of the window.

    Needs at least half the judging period observed, so a cell lost after a
    frame or two is not called stationary for lack of data.
    """
    n = params.frames(params.still_hours)
    head = cell.iloc[:n]
    if len(head) < max(3, n // 2):
        return False
    steps = np.hypot(head["y"].diff(), head["x"].diff()).dropna()
    return bool(steps.median() < params.still_px)


def _pyknotic_end(
    cell: pd.DataFrame,
    thr: Thresholds,
    params: FateParams,
    shape: tuple[int, int] | None,
) -> int | None:
    """Onset frame of a pyknotic end of the track, or ``None``.

    The last ``death_frames`` or more frames must be small and bright against
    the cell's own medians, the end must lie away from the image border, and
    geminin at the onset must be below its high threshold (a G2 nucleus that
    condenses is entering mitosis, not dying).
    """
    if len(cell) < 2 * params.death_frames:
        return None
    pyknotic = (cell["area"] < params.death_area * cell["area"].median()) & (
        cell["dna"] > params.death_dna * cell["dna"].median()
    )
    # Length of the trailing run of pyknotic frames.
    tail = pyknotic.to_numpy()[::-1]
    run = len(tail) if tail.all() else int(np.argmin(tail))
    if run < params.death_frames:
        return None
    onset = cell.iloc[-run]
    if onset["geminin"] > params.gem_high * thr.gem_low:
        return None
    if shape is not None:
        last = cell.iloc[-1]
        margin = 2 * _radius(thr.area_median)
        if (
            min(last.y, last.x) < margin
            or last.y > shape[0] - margin
            or last.x > shape[1] - margin
        ):
            return None
    return int(cell.index[-run])


def _condensed_run(
    cell: pd.DataFrame, thr: Thresholds, params: FateParams
) -> tuple[int, int] | None:
    """Longest run of condensed-chromatin frames as ``(start, length)``."""
    condensed = (cell["dna"] > params.condensed_dna * thr.dna_median) & (
        cell["area"] < params.condensed_area * cell["area"].median()
    )
    best: tuple[int, int] | None = None
    run_start, run = 0, 0
    for t, c in cast("Iterable[tuple[int, bool]]", condensed.items()):
        if c:
            run_start = int(t) if run == 0 else run_start
            run += 1
            if best is None or run > best[1]:
                best = (run_start, run)
        else:
            run = 0
    return best


def follow_cells(
    tracks: pd.DataFrame,
    start: int,
    stop: int,
    params: FateParams | None = None,
    repair_events: pd.DataFrame | None = None,
    assignment: dict[int, int] | None = None,
    shape: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Follow every cell present at ``start`` until its first outcome or ``stop``.

    Args:
        tracks: Repaired tracks for one well (:func:`~cellview.tracks.repair.apply_repair`)
            with ``pip``, ``geminin`` and ``dna`` columns, background-subtracted.
        start: First frame of the window.
        stop: Last frame of the window.
        params: Thresholds; defaults to :class:`FateParams`.
        repair_events: :attr:`RepairResult.events`, used for the ``fragment``
            and ``refractory`` flags.
        assignment: :attr:`RepairResult.assignment`, mapping event track ids to
            final ones.
        shape: Image height and width in pixels, for the ``edge`` flag.

    Returns:
        ``(cells, states)``: one row per cell with its outcome and flags, and
        one row per cell and frame with the phase call.
    """
    params = params or FateParams()
    thr = well_thresholds(tracks)
    idx = _Index(tracks)
    events_by_track = _events_by_track(repair_events, assignment)

    present = tracks[tracks.timepoint == start]
    present = present[present.area >= params.min_start_area * thr.area_median]
    claimed: set[int] = set(present.track_id)

    cells, states = [], []
    for cell_no, tid in enumerate(sorted(present.track_id), start=1):
        flags: set[str] = set()
        segments = [int(tid)]
        outcome, outcome_frame = None, None
        cur = int(tid)
        while True:
            end = int(idx.rows[cur].index[-1])
            if end >= stop:
                break
            if cur in idx.divides:
                outcome, outcome_frame = "divided", end + 1
                break
            options = idx.continuations(cur, params, claimed)
            if not options:
                outcome, outcome_frame = "lost", end
                break
            if len(options) > 1:
                flags.add("link_ambiguous")
            _, gap, nxt = options[0]
            if gap > 1:
                flags.add("gap")
            claimed.add(nxt)
            segments.append(nxt)
            cur = nxt

        rows = pd.concat([idx.rows[s] for s in segments])
        rows = rows[(rows.index >= start) & (rows.index <= stop)]
        phases = call_phases(rows, thr, params)

        exit_frame = _mitotic_exit(rows, thr, params)
        if exit_frame is not None and (
            outcome_frame is None or exit_frame < outcome_frame - 1
        ):
            outcome, outcome_frame = "mitotic_exit_no_division", exit_frame
        run = _condensed_run(rows, thr, params)
        if (
            outcome is None
            and run
            and run[1] >= params.frames(params.arrest_hours)
        ):
            outcome, outcome_frame = "mitotic_arrest", run[0]
        if outcome == "lost":
            onset = _pyknotic_end(rows, thr, params, shape)
            if onset is not None:
                outcome, outcome_frame = "death", onset
        if outcome is None:
            outcome, outcome_frame = "no_mitosis", int(rows.index[-1])
        history = pd.concat([idx.rows[s] for s in segments])
        reporter = _reporter_status(
            history[history.index <= stop], thr, params
        )

        # Geminin falls in mitosis itself: judge continuity only before it.
        cut = (
            outcome_frame - params.jump_window
            if outcome in ("divided", "mitotic_exit_no_division")
            and outcome_frame is not None
            else stop + 1
        )
        if _geminin_jump(rows[rows.index < cut], thr, params) is not None:
            flags.add("geminin_jump")
        if outcome in ("mitotic_exit_no_division", "mitotic_arrest", "death"):
            flags.add("outcome_review")
        if outcome == "lost":
            flags.add("lost")
        for s in segments:
            for rule, frame in events_by_track.get(s, ()):
                if start <= frame <= rows.index[-1]:
                    flags.add(rule)
        if shape is not None:
            margin = 2 * _radius(thr.area_median)
            near = (
                (rows.y < margin)
                | (rows.x < margin)
                | (rows.y > shape[0] - margin)
                | (rows.x > shape[1] - margin)
            )
            if near.any():
                flags.add("edge")
        if _stationary(rows, params):
            if reporter == "positive":
                outcome, outcome_frame, flags = (
                    "stationary",
                    None,
                    {"outcome_review"},
                )
            else:
                outcome, outcome_frame, flags = "debris", None, set()
        elif reporter == "negative":
            outcome, outcome_frame, flags = "no_reporter", None, set()
        elif reporter == "unclear":
            flags.add("reporter_unclear")

        cells.append(
            {
                "cell": cell_no,
                "start_track": int(tid),
                "segments": " ".join(map(str, segments)),
                "start_phase": phases.iloc[0],
                "outcome": outcome,
                "outcome_frame": outcome_frame,
                "end_phase": phases.iloc[-1],
                "last_frame": int(rows.index[-1]),
                "censored": outcome in ("lost", "no_mitosis"),
                "excluded": outcome in ("debris", "stationary", "no_reporter"),
                "review_flags": " ".join(f for f in FLAGS if f in flags),
                "n_flags": len(flags),
            }
        )
        st = rows[
            ["track_id", "area", "y", "x", "pip", "geminin", "dna"]
        ].copy()
        st["phase"] = phases
        st["cell"] = cell_no
        states.append(st.reset_index())

    return pd.DataFrame(cells), pd.concat(states, ignore_index=True)


def _events_by_track(
    events: pd.DataFrame | None, assignment: dict[int, int] | None
) -> dict[int, list[tuple[str, int]]]:
    """Map final track ids to the review-relevant repair rules and their frames."""
    out: dict[int, list[tuple[str, int]]] = {}
    if events is None or events.empty or assignment is None:
        return out
    # Row attributes are typed Scalar by the stubs; these columns are ints.
    for ev in cast("Iterable[Any]", events.itertuples()):
        rule = "fragment" if ev.rule == "fragment" else None
        if getattr(ev, "refractory", False) is True:
            rule = "refractory"
        if rule:
            key = assignment.get(int(ev.parent), int(ev.parent))
            out.setdefault(key, []).append((rule, int(ev.frame)))
    return out


def apply_annotations(cells: pd.DataFrame, curated: Any) -> pd.DataFrame:
    """Let curated annotations override the walker's automatic calls.

    For every cell, the annotations of the curated tracks it was followed
    through (``segments``) are applied: an ``exclude`` marks the cell excluded;
    ``set_outcome`` replaces the outcome; the first curated ``mitosis``,
    ``death`` or ``slippage`` event inside the cell's span sets the outcome and
    its frame. Overridden cells get ``curated = True``.

    Args:
        cells: Output of :func:`follow_cells`.
        curated: :class:`cellview.tracks.edit.Curated` the tracks came from.
    """
    kinds = {
        "mitosis": "divided",
        "death": "death",
        "slippage": "mitotic_exit_no_division",
    }
    out = cells.copy()
    out["curated"] = False
    for i, row in out.iterrows():
        i = cast(int, i)  # follow_cells builds `cells` with a RangeIndex
        notes: dict[str, Any] = {}
        for seg in str(row["segments"]).split():
            for key, val in curated.annotations.get(int(seg), {}).items():
                if isinstance(val, list):
                    notes.setdefault(key, []).extend(val)
                else:
                    notes[key] = val
        if not notes:
            continue
        events = sorted(notes.get("events", []), key=lambda e: e["frame"])
        if events:
            first = events[0]
            out.at[i, "outcome"] = kinds[first["kind"]]
            out.at[i, "outcome_frame"] = int(first["frame"])
            out.at[i, "censored"] = False
        if "outcome" in notes:
            out.at[i, "outcome"] = notes["outcome"]
            out.at[i, "censored"] = notes["outcome"] in ("lost", "no_mitosis")
        if "excluded" in notes:
            out.at[i, "excluded"] = True
            out.at[i, "outcome"] = notes["excluded"] or "excluded"
        out.at[i, "curated"] = True
    return out
