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
``left_field``
    The track ends within ``edge_radii`` nuclear radii of the image border
    after moving towards it over its last ``edge_frames`` frames: the cell
    left the imaged field. Censored like ``lost``, but not a tracking failure,
    so it is not queued for review (only audited).
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

Walkers (#87)
-------------

The walking itself (bridges, switches, daughter pairs, outcomes, flags,
proposals) is the same for every plate; what a reporter says about a cell is
not. That part sits behind a :class:`Walker`, chosen from the plate's channels
by :func:`resolve_walker`:

:class:`FucciWalker`
    PIP-FUCCI, as described above.
:class:`H2BTubWalker`
    H2B and tubulin, no FUCCI (plate 5061). Continuity comes from the nuclear
    DNA content (``content`` = background-subtracted DNA-dye mean times area),
    which along one cell only rises (doubling through S) and halves at
    division: on 5054, within-cell G2/G1 content is 2.08 (10–90 %: 1.94–2.31)
    with 12 % frame-to-frame noise. A fall without a split is ``dna_jump``.
    Daughters carry about half the mother's content; a division the H2B
    maximum does not show condensing is flagged ``division_unconfirmed``.
    Phases are content relative to the well's G1 peak: approximate, because
    dye uptake varies between cells (snapshot G1/G2 calls ~85 % on 5054);
    make within-cell calls from the whole track later.

Plates with neither get no walker.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np
import pandas as pd
from omero_utils.channels import resolve_h2b, resolve_pip_fucci

#: Reason codes a cell can be flagged with.
FLAGS = (
    "switch",  # the track jumps to a different-looking nucleus while a NEW label continues the cell
    "jump_suspect",  # as switch, but the matching neighbour already existed: flagged, not acted on
    "daughter_pair",  # the track ends with geminin high and two daughter-like nuclei appear: an unlinked division
    "link_ambiguous",  # a bridge had more than one plausible continuation
    "gap",  # a bridge skipped at least one frame
    "fragment",  # nucleus pieces were merged during the window
    "refractory",  # the refractory rule removed a second division
    "geminin_jump",  # geminin fell (or leapt) without a split: wrong link or bad frame
    "dna_jump",  # DNA content fell (or leapt) without a split: wrong link or bad frame
    "division_unconfirmed",  # the division shows no H2B condensation
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
        edge_radii: A track ending within this many nuclear radii of the
            image border ...
        edge_frames: ... after getting closer to it over this many frames
            has ``left_field``.
        switch_history: Frames before a step whose medians are the cell's
            recent values for ``switch`` (long enough that a two-frame
            mis-segmentation does not dominate them).
        switch_window: Frames from the step on whose medians are compared
            with them (a switch is permanent; a bad frame is not).
        switch_area: A step after which area differs by more than this
            factor, or geminin (clipped at its floor) falls by more than
            ``jump_drop`` or rises by more than ``jump_rise``, is a jump. PIP
            is not used: it falls at S entry and climbs in G2 by itself.
            A jump is a *switch* only if another nucleus within ``reach``
            continues the cell within ``area_tol`` and ``marker_tol``;
            otherwise it is a mis-segmented frame.
        daughter_area: Area of a daughter relative to the mother's last
            frame, as ``(low, high)``, for ``daughter_pair``.
        content_s: H2B-Tub walker: S phase from this multiple of the well's
            G1 DNA content ...
        content_g2: ... G2 from this multiple; also the content at which a
            track end can be a mitosis and a small bright end is not a death.
        content_tol: Allowed DNA-content ratio (and inverse) across a bridge
            or switch: within-cell noise is ~12 %, and content cannot double
            within ``max_gap`` frames, so a G1/G2 mix-up is refused.
        daughter_content: A daughter carries less than this fraction of the
            mother's last DNA content.
        condense_rise: A division is confirmed when the H2B maximum around it
            reaches this multiple of the mother's interphase level.
        newborn_area: A seeded daughter starts at most this fraction of the
            mother's interphase area ...
        newborn_h2b: ... with H2B at least this multiple of hers (telophase
            chromatin is compact and bright) ...
        min_mother_frames: ... and the mother was tracked this many frames
            (a 1-3 frame "mother" is a mask fragment, not a cell).
        condense_window: Frames at the end of the mother (and the start of the
            daughters) searched for that peak.
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
    edge_radii: float = 3.0
    edge_frames: int = 3
    min_start_area: float = 0.7
    reporter_hours: float = 24.0
    still_px: float = 3.0
    still_hours: float = 4.0
    switch_history: int = 5
    switch_window: int = 3
    switch_area: float = 2.0
    daughter_area: tuple[float, float] = (0.3, 0.75)
    content_s: float = 1.3
    content_g2: float = 1.75
    content_tol: float = 1.5
    daughter_content: float = 0.75
    condense_rise: float = 1.8
    condense_window: int = 3
    newborn_area: float = 0.6
    newborn_h2b: float = 1.3
    min_mother_frames: int = 6

    def frames(self, hours: float) -> int:
        """Convert hours to frames."""
        return int(round(hours * 60 / self.interval_minutes))


@dataclass(frozen=True)
class Thresholds:
    """Well-level reference levels for phase calling.

    ``content_g1`` and ``h2b_floor`` are the H2B-Tub walker's (the well's G1
    DNA-content peak, and the H2B level below which a nucleus does not
    express it); the FUCCI walker leaves them unset.
    """

    pip_high: float
    gem_low: float
    dna_median: float
    area_median: float
    content_g1: float = float("nan")
    h2b_floor: float = 0.0


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
    cell: pd.DataFrame,
    thr: Thresholds,
    params: FateParams,
    walker: Walker | None = None,
) -> int | None:
    """First frame where the jump marker falls (or leaps) between neighbouring windows.

    Compares the median of ``jump_window`` frames before each step with the
    median after it, on the walker's jump marker (geminin by default)
    clipped at its floor, so G1-level noise and one-frame compaction spikes
    never count.
    """
    walker = walker or FucciWalker()
    floor = walker.floors(thr, params)[walker.jump]
    gem = cell[walker.jump].clip(lower=floor).to_numpy(dtype=float)
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
        self.frames = {
            int(cast(int, t)): g for t, g in tracks.groupby("timepoint")
        }

    def continuations(
        self,
        tid: int,
        params: FateParams,
        claimed: set[int],
        walker: Walker | None = None,
    ) -> list[tuple[float, int, int]]:
        """Founders that plausibly continue ``tid``: ``(cost, gap, track)``.

        A founder must start within ``reach`` and match area (``area_tol``)
        and the walker's markers (``marker_tol``).
        """
        walker = walker or FucciWalker()
        markers = walker.markers
        tol = walker.tolerances(params)
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
                ratios = [first.area / last.area] + [
                    max(first[m], 1) / max(last[m], 1) for m in markers
                ]
                tols = [params.area_tol] + [tol[m] for m in markers]
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


def _clipped(
    rows: pd.DataFrame, thr: Thresholds, params: FateParams, walker: Walker
) -> list[Any]:
    """Area, then the walker's markers clipped at their floors, as arrays."""
    floors = walker.floors(thr, params)
    return [rows["area"].to_numpy(dtype=float)] + [
        rows[m].clip(lower=floors[m]).to_numpy(float) for m in walker.markers
    ]


def _matching_neighbour(
    t: int,
    last: Any,
    ref: Sequence[float],
    exclude: int,
    idx: _Index,
    thr: Thresholds,
    params: FateParams,
    walker: Walker,
) -> int | None:
    """A nucleus in frame ``t`` near ``last`` that continues ``ref``, by cost.

    ``ref`` is the cell's recent area and markers, in :func:`_clipped` order.
    """
    here = idx.frames.get(t)
    if here is None:
        return None
    here = here[here["track_id"] != exclude]
    if here.empty:
        return None
    values = _clipped(here, thr, params, walker)
    area = values[0]
    dist = np.hypot(
        here["y"].to_numpy(float) - last.y, here["x"].to_numpy(float) - last.x
    )
    tol = walker.tolerances(params)
    tols = [params.area_tol] + [tol[m] for m in walker.markers]
    best: tuple[float, int] | None = None
    for k, tid in enumerate(here["track_id"].to_numpy()):
        if dist[k] > params.reach * (_radius(area[k]) + _radius(last.area)):
            continue
        ratios = [v[k] / r for v, r in zip(values, ref, strict=True)]
        if any(
            r > tol or r < 1 / tol for r, tol in zip(ratios, tols, strict=True)
        ):
            continue
        cost = dist[k] / (_radius(last.area) + 1) + sum(
            abs(np.log(r)) for r in ratios
        )
        if best is None or cost < best[0]:
            best = (float(cost), int(tid))
    return best[1] if best else None


def _jumped(
    area_now: float,
    marker_now: float,
    ref_area: float,
    ref_marker: float,
    params: FateParams,
) -> bool:
    """Whether area or the jump marker differ from the reference beyond the switch tolerances."""
    return (
        area_now > params.switch_area * ref_area
        or area_now < ref_area / params.switch_area
        or marker_now < ref_marker / params.jump_drop
        or marker_now > params.jump_rise * ref_marker
    )


def _switch(
    cell: pd.DataFrame,
    idx: _Index,
    thr: Thresholds,
    params: FateParams,
    confirmed: set[int] | frozenset[int] = frozenset(),
    walker: Walker | None = None,
) -> tuple[int, int] | None:
    """First frame where the track jumps to another nucleus: ``(frame, neighbour)``.

    Within an unbroken track, a step after which area or the walker's jump
    marker (geminin; DNA content for H2B-Tub) stays
    changed (medians of ``switch_history`` frames before against
    ``switch_window`` frames from the step on), while a nucleus nearby
    continues the cell's recent values, is a tracking switch: the raw track
    followed the wrong object and the neighbour is the cell. Comparing
    windows, not single frames, keeps a one- or two-frame mis-segmentation
    from counting. Judged only inside a raw track, because bridges are
    judged by :meth:`_Index.continuations` and divisions end the track.

    Steps within one frame of a ``confirmed`` frame ``f`` are skipped: a
    reviewer fixed the cell's identity there (a ``link`` or ``swap`` in the
    edit log), so the jump is theirs, not the tracker's. One frame either
    side, because the reviewer links at the frame they see the right nucleus,
    which can be a frame off the step the medians date (the same tolerance as
    :func:`_new_label`). Later steps are judged against the cell's values
    from ``f + 1`` on only: the frames before it are, by the reviewer's
    word, the same cell looking different, and would make every following
    step read as the same jump again.
    """
    walker = walker or FucciWalker()
    w, h = params.switch_window, params.switch_history
    values = _clipped(cell, thr, params, walker)
    area = values[0]
    jump = values[1 + walker.markers.index(walker.jump)]
    tids = cell["track_id"].to_numpy()
    frames = cell.index.to_numpy()
    for i in range(2, len(cell)):
        if frames[i] != frames[i - 1] + 1 or tids[i] != tids[i - 1]:
            continue
        t = int(frames[i])
        if any(abs(t - f) <= 1 for f in confirmed):
            continue
        lo, hi = max(0, i - h), min(len(cell), i + w)
        since = [f + 1 for f in confirmed if f + 1 < t]
        if since:
            lo = max(lo, int(np.searchsorted(frames, max(since))))
            if i - lo < 2:
                continue
        ref = [float(np.median(v[lo:i])) for v in values]
        ref_jump = float(np.median(jump[lo:i]))
        # The step frame itself must jump (so the switch is dated to the
        # frame it happens in) and the frames after it must stay jumped.
        if not (
            _jumped(float(area[i]), float(jump[i]), ref[0], ref_jump, params)
            and _jumped(
                float(np.median(area[i:hi])),
                float(np.median(jump[i:hi])),
                ref[0],
                ref_jump,
                params,
            )
        ):
            continue
        alt = _matching_neighbour(
            int(frames[i]),
            cell.iloc[i - 1],
            ref,
            int(tids[i]),
            idx,
            thr,
            params,
            walker,
        )
        if alt is not None:
            return int(frames[i]), alt
    return None


def _new_label(switched: tuple[int, int], idx: _Index) -> bool:
    """Whether the switch's neighbour is a track that starts at the switch (±1 frame)."""
    frame, alt = switched
    rows = idx.rows.get(alt)
    return rows is not None and abs(int(rows.index[0]) - frame) <= 1


def _daughter_pair(
    tid: int,
    idx: _Index,
    thr: Thresholds,
    params: FateParams,
    claimed: set[int],
    walker: Walker | None = None,
) -> tuple[int, list[int]] | None:
    """Two daughter-like founders after a track ends looking mitotic.

    The repair links daughters only when it can; a mitosis whose daughters
    were never linked leaves a G2 track that just ends and two new founders
    beside it, each about half its area and daughter-like by the walker's
    markers (geminin fallen; half the DNA content). Returns
    ``(frame, [daughters])``.
    """
    walker = walker or FucciWalker()
    rows = idx.rows[tid]
    last = rows.iloc[-1]
    end = int(rows.index[-1])
    if not walker.ends_mitotic(rows, thr, params):
        return None
    lo, hi = params.daughter_area
    for gap in range(1, params.max_gap + 1):
        found = []
        for cand in idx.starts.get(end + gap, []):
            if cand in claimed or cand == tid:
                continue
            first = idx.rows[cand].iloc[0]
            dist = np.hypot(first.y - last.y, first.x - last.x)
            if dist > params.reach * (
                _radius(first.area) + _radius(last.area)
            ):
                continue
            if not lo <= first.area / last.area <= hi:
                continue
            if not walker.daughter_like(first, last, params):
                continue
            found.append((float(dist), int(cand)))
        if len(found) >= 2:
            found.sort()
            return end + gap, [found[0][1], found[1][1]]
    return None


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
    walker: Walker | None = None,
) -> int | None:
    """Onset frame of a pyknotic end of the track, or ``None``.

    The last ``death_frames`` or more frames must be small and bright against
    the cell's own medians, the end must lie away from the image border, and
    the cell must not be at G2 level at the onset (geminin high; G2 DNA
    content): a G2 nucleus that condenses is entering mitosis, not dying.
    """
    walker = walker or FucciWalker()
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
    if walker.at_g2(onset, thr, params):
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


def _border_distance(
    y: float | pd.Series, x: float | pd.Series, shape: tuple[int, int]
) -> Any:
    """Distance to the nearest image border."""
    return np.minimum(
        np.minimum(y, x), np.minimum(shape[0] - 1 - y, shape[1] - 1 - x)
    )


def _left_field(
    cell: pd.DataFrame,
    thr: Thresholds,
    params: FateParams,
    shape: tuple[int, int] | None,
) -> bool:
    """Whether a lost track ends at the image border, heading out."""
    if shape is None or len(cell) < 2:
        return False
    d = _border_distance(cell["y"], cell["x"], shape).to_numpy()
    radius = _radius(thr.area_median)
    if d[-1] > params.edge_radii * radius:
        return False
    back = d[-1 - min(params.edge_frames, len(d) - 1)]
    return bool(d[-1] < back or d[-1] < radius)


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


# --- walkers ---------------------------------------------------------------


class Walker:
    """What a plate's reporters tell the walker about one cell.

    Subclasses name the marker columns compared across bridges and steps
    (:attr:`markers`), the one whose fall marks a wrong link (:attr:`jump`)
    and how the reporters call phases, mitotic track ends, daughters, exits,
    expression and G2.
    """

    name: ClassVar[str] = ""
    markers: ClassVar[tuple[str, ...]] = ()
    jump: ClassVar[str] = ""
    jump_flag: ClassVar[str] = ""
    state_columns: ClassVar[tuple[str, ...]] = ()

    def prepare(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """The walker's columns added to (or renamed in) a well's tracks."""
        return tracks

    def thresholds(self, tracks: pd.DataFrame) -> Thresholds:
        """Well-level reference levels."""
        raise NotImplementedError

    def floors(self, thr: Thresholds, params: FateParams) -> dict[str, float]:
        """Per marker, the level below which values are noise (clipped)."""
        raise NotImplementedError

    def tolerances(self, params: FateParams) -> dict[str, float]:
        """Per marker, the ratio (and inverse) allowed across a bridge or step."""
        return dict.fromkeys(self.markers, params.marker_tol)

    def phases(
        self, cell: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> pd.Series:
        """Per-frame G1/S/G2 call."""
        raise NotImplementedError

    def ends_mitotic(
        self, rows: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> bool:
        """Whether a track's end looks like mitosis (for daughter pairs)."""
        raise NotImplementedError

    def daughter_like(self, first: Any, last: Any, params: FateParams) -> bool:
        """Whether a founder's first frame is a daughter of ``last``'s nucleus."""
        raise NotImplementedError

    def mitotic_exit(
        self, rows: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> int | None:
        """Frame of a mitotic exit without division, if the reporters show one."""
        return None

    def reporter_status(
        self, history: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> str:
        """``"positive"``, ``"negative"`` or ``"unclear"``."""
        raise NotImplementedError

    def at_g2(self, row: Any, thr: Thresholds, params: FateParams) -> bool:
        """Whether one frame is at G2 level (then condensing is mitosis, not death)."""
        raise NotImplementedError

    def division_confirmed(
        self,
        mother: pd.DataFrame,
        daughters: list[pd.DataFrame],
        params: FateParams,
    ) -> bool:
        """Whether the reporters show the division (``False`` flags it)."""
        return True

    def newborn(
        self, mother: pd.DataFrame, daughter: pd.DataFrame, params: FateParams
    ) -> bool:
        """Whether a daughter looks newly born (for birth cohorts)."""
        return True


class FucciWalker(Walker):
    """PIP-FUCCI: the walker as described in the module docstring."""

    name = "fucci"
    markers = ("pip", "geminin")
    jump = "geminin"
    jump_flag = "geminin_jump"
    state_columns = ("pip", "geminin", "dna")

    def thresholds(self, tracks: pd.DataFrame) -> Thresholds:
        """PIP-high and geminin-low references (:func:`well_thresholds`)."""
        return well_thresholds(tracks)

    def floors(self, thr: Thresholds, params: FateParams) -> dict[str, float]:
        """PIP clipped at its S-phase level, geminin at its high threshold."""
        return {
            "pip": params.pip_low * thr.pip_high,
            "geminin": params.gem_high * thr.gem_low,
        }

    def phases(
        self, cell: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> pd.Series:
        """PIP-FUCCI phases (:func:`call_phases`)."""
        return call_phases(cell, thr, params)

    def ends_mitotic(
        self, rows: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> bool:
        """Geminin high over the last ``switch_window`` frames."""
        tail = rows["geminin"].to_numpy(float)[-params.switch_window :]
        return float(np.median(tail)) >= params.gem_high * thr.gem_low

    def daughter_like(self, first: Any, last: Any, params: FateParams) -> bool:
        """Geminin fallen below ``exit_drop`` of the mother's."""
        return bool(first.geminin < params.exit_drop * last.geminin)

    def mitotic_exit(
        self, rows: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> int | None:
        """Sustained high geminin collapsing in one nucleus (:func:`_mitotic_exit`)."""
        return _mitotic_exit(rows, thr, params)

    def reporter_status(
        self, history: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> str:
        """Expression of PIP or geminin (:func:`_reporter_status`)."""
        return _reporter_status(history, thr, params)

    def at_g2(self, row: Any, thr: Thresholds, params: FateParams) -> bool:
        """Geminin above its high threshold."""
        return bool(row["geminin"] > params.gem_high * thr.gem_low)


@dataclass(frozen=True)
class H2BTubWalker(Walker):
    """H2B and tubulin, no FUCCI: continuity from DNA content, divisions from H2B.

    Attributes:
        h2b: Column of the H2B channel in the tracks (``load_well`` lower
            case, e.g. ``"h2b"`` or ``"h2b_rfp"``); renamed to ``h2b``.
        h2b_max: Column of its nuclear maximum, if loaded; renamed to
            ``h2b_max``. Without it divisions are not checked.
    """

    h2b: str = "h2b"
    h2b_max: str | None = "h2b_max"
    name: ClassVar[str] = "h2b_tub"
    markers: ClassVar[tuple[str, ...]] = ("content", "h2b")
    jump: ClassVar[str] = "content"
    jump_flag: ClassVar[str] = "dna_jump"
    state_columns: ClassVar[tuple[str, ...]] = ("h2b", "dna", "content")

    def prepare(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """Rename the H2B columns and add ``content`` (DNA mean times area)."""
        out = tracks.rename(
            columns={self.h2b: "h2b"}
            | ({self.h2b_max: "h2b_max"} if self.h2b_max else {})
        )
        if "content" not in out:
            out["content"] = out["dna"] * out["area"]
        return out

    def thresholds(self, tracks: pd.DataFrame) -> Thresholds:
        """G1 content peak (mode of log content) and the H2B expression floor."""
        content = tracks["content"].to_numpy(float)
        content = content[content > 0]
        # The G1 peak: the mode of log content (most cells are in G1).
        hist, edges = np.histogram(np.log(content), bins=60)
        k = int(hist.argmax())
        return Thresholds(
            pip_high=float("nan"),
            gem_low=float("nan"),
            dna_median=float(tracks["dna"].median()),
            area_median=float(tracks["area"].median()),
            content_g1=float(np.exp((edges[k] + edges[k + 1]) / 2)),
            h2b_floor=0.2 * float(tracks["h2b"].median()),
        )

    def floors(self, thr: Thresholds, params: FateParams) -> dict[str, float]:
        """Content clipped at half the G1 peak, H2B at its expression floor."""
        return {"content": 0.5 * thr.content_g1, "h2b": thr.h2b_floor}

    def tolerances(self, params: FateParams) -> dict[str, float]:
        """Content within ``content_tol`` (it is steady); H2B within ``marker_tol``."""
        return {"content": params.content_tol, "h2b": params.marker_tol}

    def phases(
        self, cell: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> pd.Series:
        """Smoothed content against the well's G1 peak (approximate)."""
        level = (
            cell["content"]
            .rolling(params.smooth, center=True, min_periods=1)
            .median()
            / thr.content_g1
        )
        return pd.Series(
            np.select(
                [level >= params.content_g2, level >= params.content_s],
                ["G2", "S"],
                "G1",
            ),
            index=cell.index,
        )

    def ends_mitotic(
        self, rows: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> bool:
        """Content at G2 level over the last ``switch_window`` frames."""
        tail = rows["content"].to_numpy(float)[-params.switch_window :]
        return self.at_g2({"content": float(np.median(tail))}, thr, params)

    def daughter_like(self, first: Any, last: Any, params: FateParams) -> bool:
        """Less than ``daughter_content`` of the mother's content."""
        return bool(first.content < params.daughter_content * last.content)

    def reporter_status(
        self, history: pd.DataFrame, thr: Thresholds, params: FateParams
    ) -> str:
        """H2B expressed in any frame, else negative once observed long enough."""
        h2b = (
            history["h2b"]
            .rolling(params.smooth, center=True, min_periods=1)
            .median()
        )
        if (h2b > thr.h2b_floor).any():
            return "positive"
        span = int(history.index[-1]) - int(history.index[0]) + 1
        return (
            "negative"
            if span >= params.frames(params.reporter_hours)
            else "unclear"
        )

    def at_g2(self, row: Any, thr: Thresholds, params: FateParams) -> bool:
        """Content at ``content_g2`` times the G1 peak or more."""
        return bool(row["content"] >= params.content_g2 * thr.content_g1)

    def division_confirmed(
        self,
        mother: pd.DataFrame,
        daughters: list[pd.DataFrame],
        params: FateParams,
    ) -> bool:
        """H2B condenses: the maximum around the split rises over interphase.

        The peak is taken over the mother's last ``condense_window`` frames
        and the daughters' first ones, against the median of the mother's
        frames before that. Undecidable (no maximum loaded, too short a
        mother) counts as confirmed: the flag is for a visible absence.
        """
        if "h2b_max" not in mother:
            return True
        w = params.condense_window
        base = mother["h2b_max"].to_numpy(float)[:-w][-2 * w :]
        if len(base) < 2:
            return True
        peak = [mother["h2b_max"].to_numpy(float)[-w:].max()] + [
            d["h2b_max"].to_numpy(float)[:w].max() for d in daughters if len(d)
        ]
        return bool(max(peak) >= params.condense_rise * np.median(base))

    def newborn(
        self, mother: pd.DataFrame, daughter: pd.DataFrame, params: FateParams
    ) -> bool:
        """Small and bright against the mother, who was a real, tracked cell.

        The mother's interphase reference is her last six frames before the
        ``condense_window`` frames of mitosis. On 5061 B2 this keeps 30 of
        172 condensation-confirmed seeds; the rest were mask fragments whose
        tracks end within a few frames (median 4.5 against 17.5).
        """
        w = params.condense_window
        if len(mother) < params.min_mother_frames or daughter.empty:
            return False
        base = mother.iloc[:-w].tail(6)
        first = daughter.iloc[0]
        return bool(
            first["area"] <= params.newborn_area * base["area"].median()
            and first["h2b"] >= params.newborn_h2b * base["h2b"].median()
        )


#: Walkers by name.
WALKERS: dict[str, type[Walker]] = {
    FucciWalker.name: FucciWalker,
    H2BTubWalker.name: H2BTubWalker,
}


def resolve_walker(
    channel_names: Iterable[str], n_timepoints: int
) -> Walker | None:
    """The walker for a plate's channels: FUCCI, else H2B with tubulin, else none.

    Args:
        channel_names: The plate's channel names (CellView or OMERO).
        n_timepoints: Frames in the timelapse; a single frame gets no walker.

    Returns:
        :class:`FucciWalker` for PIP-FUCCI; :class:`H2BTubWalker` (with the
        H2B column named as :func:`~omero_screen_track.cells.load_well` names
        it) when there is an H2B channel and a tubulin channel (name
        containing ``tub``); otherwise ``None``.
    """
    names = list(channel_names)
    if resolve_pip_fucci(names, n_timepoints) is not None:
        return FucciWalker()
    h2b = resolve_h2b(names, n_timepoints)
    if h2b is not None and any("tub" in n.lower() for n in names):
        col = h2b.lower()
        return H2BTubWalker(h2b=col, h2b_max=f"{col}_max")
    return None


def birth_seeds(
    tracks: pd.DataFrame,
    window: tuple[int, int],
    walker: Walker | None = None,
    params: FateParams | None = None,
    seed: int = 0,
    sisters: bool = False,
) -> list[tuple[int, int]]:
    """Daughters of the divisions in ``window`` that the reporters confirm.

    For a cohort followed from birth (:func:`follow_cells` ``seeds``). Only
    divisions :meth:`Walker.division_confirmed` accepts give seeds, and only
    daughters :meth:`Walker.newborn` accepts. By default one daughter is
    drawn at random per division, so sisters (whose fates are correlated)
    are not both followed; ``sisters=True`` keeps every one (analyse each
    pair as a cluster).

    Args:
        tracks: A well's tracks, as for :func:`follow_cells`.
        window: First and last birth frame.
        walker: Default :class:`FucciWalker`.
        params: Default :class:`FateParams`.
        seed: Random seed of the daughter draw.
        sisters: Keep every accepted daughter, not one per division.

    Returns:
        ``(daughter track id, birth frame)``, by birth frame.
    """
    walker = walker or FucciWalker()
    params = params or FateParams()
    idx = _Index(walker.prepare(tracks))
    rng = np.random.default_rng(seed)
    kids: dict[int, list[int]] = {}
    for tid, parent in idx.parent.items():
        if parent and parent in idx.rows:
            kids.setdefault(int(parent), []).append(int(cast(int, tid)))
    seeds = []
    for mother, daughters in sorted(kids.items()):
        # A daughter is born when the mother ends: a track that starts later
        # and was linked as a child is not a birth in the window.
        split = int(idx.rows[mother].index[-1]) + 1
        if not window[0] <= split <= window[1]:
            continue
        daughters = [
            d
            for d in daughters
            if int(idx.rows[d].index[0]) - split <= params.max_gap - 1
        ]
        rows = [idx.rows[d] for d in daughters]
        if not walker.division_confirmed(idx.rows[mother], rows, params):
            continue
        ok = sorted(
            d
            for d in daughters
            if walker.newborn(idx.rows[mother], idx.rows[d], params)
        )
        if not ok:
            continue
        picks = ok if sisters else [ok[int(rng.integers(len(ok)))]]
        seeds += [(d, int(idx.rows[d].index[0])) for d in picks]
    return sorted(seeds, key=lambda s: (s[1], s[0]))


def follow_cells(
    tracks: pd.DataFrame,
    start: int,
    stop: int,
    params: FateParams | None = None,
    repair_events: pd.DataFrame | None = None,
    assignment: dict[int, int] | None = None,
    shape: tuple[int, int] | None = None,
    confirmed: dict[int, set[int]] | None = None,
    walker: Walker | None = None,
    seeds: Sequence[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Follow every cell present at ``start`` until its first outcome or ``stop``.

    Args:
        tracks: Repaired tracks for one well (:func:`~omero_screen_track.repair.apply_repair`)
            with ``pip``, ``geminin`` and ``dna`` columns, background-subtracted.
        start: First frame of the window.
        stop: Last frame of the window.
        params: Thresholds; defaults to :class:`FateParams`.
        repair_events: :attr:`RepairResult.events`, used for the ``fragment``
            and ``refractory`` flags.
        assignment: :attr:`RepairResult.assignment`, mapping event track ids to
            final ones.
        shape: Image height and width in pixels, for the ``edge`` flag.
        confirmed: Curated track id to the frames at which a reviewer fixed
            its identity (:func:`~omero_screen_track.edit.confirmed_frames`).
            A switch there is the reviewer's, so it neither cuts the track
            nor is flagged or proposed.
        walker: What the reporters say (:func:`resolve_walker`); default
            :class:`FucciWalker`.
        seeds: ``(track id, frame)`` of the cells to follow, each from its
            own frame (e.g. daughters at birth), instead of every cell
            present at ``start``. The start-area test is skipped for them.

    Returns:
        ``(cells, states)``: one row per cell with its outcome and flags, and
        one row per cell and frame with the phase call.
    """
    params = params or FateParams()
    walker = walker or FucciWalker()
    tracks = walker.prepare(tracks)
    thr = walker.thresholds(tracks)
    idx = _Index(tracks)
    events_by_track = _events_by_track(repair_events, assignment)

    if seeds is None:
        present = tracks[tracks.timepoint == start]
        present = present[
            present.area >= params.min_start_area * thr.area_median
        ]
        seeds = [(int(t), start) for t in sorted(present.track_id)]
    claimed: set[int] = {int(t) for t, _ in seeds}

    cells, states = [], []
    for cell_no, (tid, t0) in enumerate(seeds, start=1):
        flags: set[str] = set()
        segments = [int(tid)]
        proposals: list[dict[str, Any]] = []
        outcome, outcome_frame = None, None
        cur = int(tid)
        while True:
            end = int(idx.rows[cur].index[-1])
            if end >= stop:
                break
            if cur in idx.divides:
                outcome, outcome_frame = "divided", end + 1
                break
            options = idx.continuations(cur, params, claimed, walker)
            if not options:
                pair = _daughter_pair(cur, idx, thr, params, claimed, walker)
                if pair is not None:
                    frame, daughters = pair
                    outcome, outcome_frame = "divided", frame
                    flags.add("daughter_pair")
                    claimed.update(daughters)
                    for d in daughters:
                        proposals.append(
                            {
                                "op": "set_parent",
                                "track": d,
                                "frame": frame,
                                "parent": int(tid),
                                "reason": f"daughter pair at t{frame}: the track ends at G2 level and two half-size nuclei appear",
                                "confidence": "uncertain",
                            }
                        )
                    break
                outcome, outcome_frame = "lost", end
                break
            if len(options) > 1:
                flags.add("link_ambiguous")
            _, gap, nxt = options[0]
            if gap > 1:
                flags.add("gap")
            claimed.add(nxt)
            segments.append(nxt)
            proposals.append(
                {
                    "op": "link",
                    "track": nxt,
                    "frame": int(idx.rows[nxt].index[0]),
                    "reason": f"bridge: gap {gap}, {len(options)} candidate{'s' if len(options) > 1 else ''}",
                    "confidence": "clear"
                    if gap == 1 and len(options) == 1
                    else "uncertain",
                }
            )
            cur = nxt

        rows = pd.concat([idx.rows[s] for s in segments])
        rows = rows[(rows.index >= t0) & (rows.index <= stop)]
        fixed = {f for s in segments for f in (confirmed or {}).get(s, ())}
        switched = _switch(rows, idx, thr, params, fixed, walker)
        if switched is not None and not _new_label(switched, idx):
            # A matching neighbour that already existed is usually a
            # neighbour (on 5054, 19 of 19 such relinks in C4 were wrong: the
            # cell's own geminin fall, an exit, read as a jump). Flag it for
            # review; leave the track and propose nothing.
            flags.add("jump_suspect")
            switched = None
        if switched is not None:
            # The raw track followed the wrong object from here: nothing after
            # it is this cell, and the new label is the likely continuation
            # (the tracker gives the real nucleus a fresh label when it jumps).
            frame, alt = switched
            rows = rows[rows.index < frame]
            outcome, outcome_frame = "lost", frame
            flags.add("switch")
            proposals = [p for p in proposals if p["frame"] < frame]
            proposals.append(
                {
                    "op": "link",
                    "track": alt,
                    "frame": frame,
                    "reason": f"switch at t{frame}: the track jumps to a different-looking nucleus while track {alt} continues the cell",
                    "confidence": "uncertain",
                }
            )
        phases = walker.phases(rows, thr, params)
        if outcome == "divided" and cur in idx.divides:
            kids = [idx.rows[k] for k, p in idx.parent.items() if p == cur]
            if not walker.division_confirmed(idx.rows[cur], kids, params):
                flags.update(("division_unconfirmed", "outcome_review"))

        exit_frame = walker.mitotic_exit(rows, thr, params)
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
        if outcome == "lost" and switched is None:
            onset = _pyknotic_end(rows, thr, params, shape, walker)
            if onset is not None:
                outcome, outcome_frame = "death", onset
            elif _left_field(rows, thr, params, shape):
                outcome = "left_field"
        if outcome is None:
            outcome, outcome_frame = "no_mitosis", int(rows.index[-1])
        history = pd.concat([idx.rows[s] for s in segments])
        reporter = walker.reporter_status(
            history[history.index <= stop], thr, params
        )

        # The jump marker falls in mitosis itself: judge continuity only before it.
        cut = (
            outcome_frame - params.jump_window
            if outcome in ("divided", "mitotic_exit_no_division")
            and outcome_frame is not None
            else stop + 1
        )
        if (
            _geminin_jump(rows[rows.index < cut], thr, params, walker)
            is not None
        ):
            flags.add(walker.jump_flag)
        if outcome in ("mitotic_exit_no_division", "mitotic_arrest", "death"):
            flags.add("outcome_review")
        if "daughter_pair" in flags:
            flags.add("outcome_review")
        if outcome == "lost":
            flags.add("lost")
        for s in segments:
            for rule, frame in events_by_track.get(s, ()):
                if t0 <= frame <= rows.index[-1]:
                    flags.add(rule)
        if shape is not None:
            margin = 2 * _radius(thr.area_median)
            near = (
                (rows.y < margin)
                | (rows.x < margin)
                | (rows.y > shape[0] - margin)
                | (rows.x > shape[1] - margin)
            )
            # A cell that left the field has nothing left to review at the edge.
            if near.any() and outcome != "left_field":
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
                "start_frame": int(t0),
                "segments": " ".join(map(str, segments)),
                "start_phase": phases.iloc[0],
                "outcome": outcome,
                "outcome_frame": outcome_frame,
                "end_phase": phases.iloc[-1],
                "last_frame": int(rows.index[-1]),
                "censored": outcome in ("lost", "left_field", "no_mitosis"),
                "excluded": outcome in ("debris", "stationary", "no_reporter"),
                "review_flags": " ".join(f for f in FLAGS if f in flags),
                "n_flags": len(flags),
                "proposals": json.dumps(proposals),
            }
        )
        st = rows[["track_id", "area", "y", "x", *walker.state_columns]].copy()
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
        curated: :class:`omero_screen_track.edit.Curated` the tracks came from.
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
            out.at[i, "censored"] = notes["outcome"] in (
                "lost",
                "left_field",
                "no_mitosis",
            )
        if "excluded" in notes:
            out.at[i, "excluded"] = True
            out.at[i, "outcome"] = notes["excluded"] or "excluded"
        out.at[i, "curated"] = True
    return out
