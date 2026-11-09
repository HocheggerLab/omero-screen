"""Lineage overview of a well's curated tracks, without Qt.

The model behind the track view (:mod:`omero_screen_napari.review.track_view`),
a TrackScheme-like picture of every curated track of a well: time runs down,
one column per track, daughters branch from their mother at the division.

Three steps, each a pure function so it can be tested on toy lineages:

1. :func:`build_lineage_table` collapses the curated detections onto nodes
   (one per track and frame), edges (``link``, ``gap``, ``division``) and a
   per-track table (span, outcome, verdict, who reviewed it, edit density,
   event frames used for alignment).
2. :func:`layout` filters and sorts the trees and places them: x is a column
   per track (leaves of a tree in depth-first order, a mother centred over
   her daughters), y is the frame minus the tree's anchor (the alignment
   event).
3. :func:`node_categories` gives each node a category for a colour mode;
   :data:`PALETTES` maps categories to colours.

The data come from the same places the Track Review widget and the fate build
read: CellView detections with debris hidden, the stored correction as the
base lineage, the review directory's ``edits.jsonl`` replayed on it, and the
verdicts of every ``*decisions.json`` beside the queue
(:func:`lineage_from_session` gathers them from a
:class:`~omero_screen_napari.review.session.ReviewSession`). Outcomes from the
fate walker and phases from the HMM (:mod:`omero_screen_track.phases`) are slow
(about a second and ten seconds per well), so they are computed separately
(:func:`slow_annotations`) and passed back to :func:`build_lineage_table`;
until then phases are the walker's per-frame G1/S/G2 calls.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from omero_screen_track.edit import Curated, Edit

    from omero_screen_napari.review.cells import CellSource
    from omero_screen_napari.review.session import ReviewSession
    from omero_screen_napari.review_queue import Queue

#: Alignment events: label shown in the widget → track column with the frame.
ALIGNMENTS = {
    "frame": "absolute frame",
    "window": "window start",
    "division": "division",
    "exit": "G2 exit",
    "g2": "G2 entry",
    "s": "S entry",
}
#: Display names of stored keys that are not shown verbatim. PIP-FUCCI shows
#: APC/C activation (geminin falls) without a division and no sign of mitosis,
#: so the paper calls it G2 exit; the stored outcome key
#: ``mitotic_exit_no_division`` stays because saved verdicts use it.
DISPLAY_NAMES = {
    "E": "G2 exit",
    "X": "G1-like after exit",
    "mitotic_exit_no_division": "G2 exit without division",
}


def display_name(key: str) -> str:
    """User-facing name of a phase, outcome or other category key."""
    return DISPLAY_NAMES.get(key, key.replace("_", " "))


#: Sort keys offered by :func:`layout`.
SORTS = ("track", "length", "outcome", "event", "edits")
#: Colour modes offered by :func:`node_categories`.
COLOUR_MODES = ("phase", "verdict", "outcome", "flags", "edits")
#: Verdict filter choices; ``accepted`` is accept or correct, not rejected.
VERDICT_FILTERS = (
    "all",
    "accepted",
    "accept",
    "correct",
    "reject",
    "unsure",
    "any",
    "none",
)
#: Reviewed-by filter choices.
REVIEWER_FILTERS = ("all", "agent", "human", "both", "any", "none")

_GREY = "#bdbcb7"
#: Category → colour per mode. Phases as in the tracking figure (lab
#: palette: G1 pink, S light blue, G2 yellow, G2 exit purple, after exit
#: lavender); outcomes as in its outcome bars.
PALETTES: dict[str, dict[str, str]] = {
    "phase": {
        "G1": "#DC6B83",
        "S": "#75B1CE",
        "G2": "#D8C367",
        "E": "#654875",
        "X": "#C6B2D1",
        "": _GREY,
    },
    "verdict": {
        "accept": "#3a9d5d",
        "correct": "#3b6fb6",
        "reject": "#c0392b",
        "unsure": "#e69b2e",
        "": _GREY,
    },
    "outcome": {
        "divided": "#eb6834",
        "mitotic_exit_no_division": "#654875",
        "death": "#2b2b2b",
        "mitotic_arrest": "#b0367a",
        "no_mitosis": "#8a8984",
        "lost": "#dddcd7",
        "left_field": "#a8c8d8",
        "excluded": "#6b4f3a",
        "stationary": "#7a6a55",
        "no_reporter": "#5c5c5c",
        "debris": "#3d3d3d",
        "": _GREY,
    },
    "flags": {
        "mask": "#d6336c",
        "switch": "#1c7ed6",
        "edited": "#f08c00",
        "after gap": "#2f9e44",
        "": _GREY,
    },
    "edits": {
        "0": "#d9d8d3",
        "≤0.5/h": "#f6d6a8",
        "≤1/h": "#f0a35e",
        "≤2/h": "#d9603b",
        ">2/h": "#9c1f1f",
    },
}

#: Edge kinds: a frame to the next, across frames without a mask, a real
#: division (mother → daughter), and a tracker parent link that is not one.
EDGE_KINDS = ("link", "gap", "division", "split")
#: Frames with a mask a daughter needs to be drawn as a branch of a division.
MIN_DAUGHTER_FRAMES = 6
#: Shortest track (frames with a mask) given HMM phases.
MIN_HMM_FRAMES = 6
_NODE_COLUMNS = [
    "track",
    "frame",
    "label",
    "is_extra",
    "n_pieces",
    "area",
    "pip",
    "geminin",
    "y",
    "x",
    "phase",
    "phase_walker",
    "phase_hmm",
    "edited",
    "confirmed",
    "after_gap",
]


@dataclass(frozen=True)
class ReviewRecord:
    """What the review directory says about one item id.

    Attributes:
        verdict: The verdict that counts (a human's beats an agent's, then
            the latest).
        outcome: The reviewer's outcome with that verdict, if any.
        reviewed_by: ``agent``, ``human`` or ``agent+human`` (any verdict in
            any queue of the directory).
        audit: Verdict from ``audit_decisions.json``, empty if not audited.
        author: Author of the winning verdict.
        time: Time of the winning verdict.
    """

    verdict: str = ""
    outcome: str = ""
    reviewed_by: str = ""
    audit: str = ""
    author: str = ""
    time: str = ""


def _rank(rec: ReviewRecord) -> tuple[bool, str]:
    """A human verdict beats an agent's; then the latest wins."""
    return (rec.author == "human", rec.time)


@dataclass(frozen=True)
class LineageTable:
    """Nodes, edges and tracks of one well.

    Attributes:
        nodes: One row per track and frame with a mask: ``track, frame,
            label`` (raw label of the largest piece), ``is_extra``
            (a reviewer mask), ``n_pieces, area, pip, geminin, y, x``,
            ``phase`` (the HMM's call where it made one, else the walker's),
            ``phase_walker``, ``phase_hmm``,
            ``edited`` (an edit names this frame of this track),
            ``confirmed`` (a link or swap by a reviewer at this frame),
            ``after_gap`` (the previous frame of the track had no mask).
        edges: ``track`` (the track the edge belongs to; the daughter for a
            division), ``source`` (the track at ``f0``), ``f0, f1, kind``
            (one of :data:`EDGE_KINDS`).
        tracks: Indexed by track id: ``parent, start, end, n_frames,
            n_gaps, hours, item_id, queued, flags, outcome, verdict,
            reviewed_by, audit, excluded, edits, edits_per_h, divided`` and
            the event
            frames ``ev_window, ev_division, ev_exit, ev_g2, ev_s`` (NaN if
            the track has no such event).
        well: The well.
        interval_minutes: Time between frames.
        window: ``(start, stop)`` frames of the analysis window.
        has_hmm: Whether ``phase`` comes from the HMM.
        pooled: Whether the HMM used the plate-pooled reference.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame
    tracks: pd.DataFrame
    well: str = ""
    interval_minutes: float = 20.0
    window: tuple[int, int] = (0, 0)
    has_hmm: bool = False
    pooled: bool = False

    def hours(self, frames: float) -> float:
        """Frames to hours."""
        return frames * self.interval_minutes / 60


@dataclass(frozen=True)
class Filters:
    """Which tracks :func:`layout` keeps.

    Attributes:
        verdict: One of :data:`VERDICT_FILTERS`.
        outcome: An outcome, or ``all``.
        reviewed_by: One of :data:`REVIEWER_FILTERS`.
        audited_only: Only tracks with an audit verdict.
        flagged_only: Only tracks the queue builder or the walker flagged.
        min_length: Minimum frames with a mask.
        with_descendants: A kept track brings its daughters (and theirs),
            which are never reviewed themselves, if they are long enough.
    """

    verdict: str = "all"
    outcome: str = "all"
    reviewed_by: str = "all"
    audited_only: bool = False
    flagged_only: bool = False
    min_length: int = 1
    with_descendants: bool = True


@dataclass(frozen=True)
class Layout:
    """Positions of what is drawn.

    Attributes:
        nodes: The kept nodes with ``x, y``.
        edges: The kept edges with ``x0, y0, x1, y1``.
        tracks: The kept tracks with ``x``, ``offset`` (y = frame −
            offset), ``y0, y1`` and ``parked`` (no alignment event; placed
            after the aligned trees).
        n_columns: Columns used.
        parked_from: First column of the parked trees, if any.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame
    tracks: pd.DataFrame
    n_columns: int = 0
    parked_from: float | None = None


@dataclass
class SlowAnnotations:
    """Fate-walker outcomes and HMM phases, computed off the UI thread.

    Attributes:
        fates: :func:`~omero_screen_track.fate.follow_cells` cells with
            curated annotations applied (``segments``, ``outcome``,
            ``outcome_frame``, ``excluded``, ``review_flags``).
        phases: ``track, frame, phase`` from the HMM.
        boundaries: Per track ``s_entry, g2_entry, exit`` median frames.
        reference: The phase reference used (pass it back in
            :class:`WellInputs` to skip refitting it).
        pooled: Whether ``reference`` was fitted on the whole plate.
        errors: What could not be computed, and why.
    """

    fates: pd.DataFrame | None = None
    phases: pd.DataFrame | None = None
    boundaries: pd.DataFrame | None = None
    reference: Any = None
    pooled: bool = False
    errors: list[str] = field(default_factory=list)


# -- review records ------------------------------------------------------------


def review_records(review_dir: Path) -> dict[str, ReviewRecord]:
    """Verdicts per item id over ``decisions.json`` and every ``*_decisions.json``.

    The same rule as the fate build: a human verdict beats an agent's, and
    between verdicts of the same author class the latest ``time`` wins.

    Args:
        review_dir: Directory holding the queue(s) and their decisions.
    """
    review_dir = Path(review_dir)
    paths = [
        review_dir / "decisions.json",
        *sorted(review_dir.glob("*_decisions.json")),
    ]
    best: dict[str, dict[str, Any]] = {}
    authors: dict[str, set[str]] = {}
    audit: dict[str, tuple[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            decisions = json.loads(path.read_text()).get("decisions", [])
        except (OSError, json.JSONDecodeError):
            continue
        for d in decisions:
            author = d.get("author", "human")
            authors.setdefault(d["id"], set()).add(author)
            rank = (author == "human", d.get("time", ""))
            old = best.get(d["id"])
            if old is None or rank >= old["rank"]:
                best[d["id"]] = {**d, "rank": rank}
            if (
                path.name == "audit_decisions.json"
                and d.get("time", "") >= audit.get(d["id"], ("", ""))[0]
            ):
                audit[d["id"]] = (d.get("time", ""), d["verdict"])
    return {
        i: ReviewRecord(
            verdict=str(d.get("verdict", "")),
            outcome=str(d.get("outcome") or ""),
            reviewed_by="+".join(sorted(authors.get(i, set()))),
            audit=audit.get(i, ("", ""))[1],
            author=str(d.get("author", "human")),
            time=str(d.get("time", "")),
        )
        for i, d in best.items()
    }


def _parse_id(item_id: str) -> tuple[str, int, int] | None:
    """``C3-t72-L45`` → ``("C3", 72, 45)``; ``None`` if not a review id."""
    parts = item_id.split("-")
    if len(parts) != 3 or not parts[1].startswith("t"):
        return None
    try:
        return parts[0], int(parts[1][1:]), int(parts[2][1:])
    except ValueError:
        return None


# -- table -----------------------------------------------------------------------


def walker_phases(
    nodes: pd.DataFrame,
    pip_high: float,
    gem_low: float,
    smooth: int = 5,
    pip_low: float = 0.5,
    gem_high: float = 2.0,
) -> pd.Series:
    """The fate walker's per-frame G1/S/G2 call, vectorised over tracks.

    Same rule as :func:`omero_screen_track.fate.call_phases`: a centred rolling
    median over measured frames, S where PIP is below ``pip_low`` × the
    well's PIP-high level, else G2 where geminin is above ``gem_high`` × the
    well's geminin-low level, else G1.

    Args:
        nodes: ``track, frame, pip, geminin`` sorted by track and frame.
        pip_high: Well PIP-high level (upper quartile).
        gem_low: Well geminin-low level (lower quartile).
        smooth: Rolling-median window, frames.
        pip_low: Fraction of ``pip_high`` below which PIP is degraded.
        gem_high: Multiple of ``gem_low`` above which geminin is high.

    Returns:
        Phase per node (index of ``nodes``).
    """
    grouped = nodes.groupby("track", sort=False)
    pip = grouped["pip"].transform(
        lambda s: s.rolling(smooth, center=True, min_periods=1).median()
    )
    gem = grouped["geminin"].transform(
        lambda s: s.rolling(smooth, center=True, min_periods=1).median()
    )
    return pd.Series(
        np.select(
            [pip < pip_low * pip_high, gem > gem_high * gem_low],
            ["S", "G2"],
            "G1",
        ),
        index=nodes.index,
    )


def _edit_frames(edit: Edit) -> list[int]:
    a = edit.args
    if "frames" in a:
        t0, t1 = (int(v) for v in a["frames"])
        return list(range(t0, t1 + 1))
    if "frame" in a:
        return [int(a["frame"])]
    return []


def _edit_track(curated: Curated, edit: Edit) -> int | None:
    """The curated track an edit concerns now, if it still resolves."""
    a = edit.args
    anchors = []
    if a.get("cell") is not None:
        anchors.append(a["cell"])
    elif edit.op == "mask_add":
        anchors.append([a["frame"], a["label"]])
    for anchor in anchors:
        try:
            return curated.resolve((int(anchor[0]), int(anchor[1])))
        except Exception:
            continue
    return None


def live_edits(edits: Iterable[Edit], well: str) -> list[Edit]:
    """The well's entries that replay applies (no reverts, nothing reverted)."""
    edits = list(edits)
    reverted = {e.args.get("target") for e in edits if e.op == "revert"}
    return [
        e
        for e in edits
        if e.well == well and e.op != "revert" and e.id not in reverted
    ]


def _first_entry(
    phases: np.ndarray, frames: np.ndarray, target: str, before: set[str]
) -> float:
    """First frame in ``target`` after a frame in one of ``before``."""
    seen = False
    for ph, t in zip(phases, frames, strict=True):
        if ph in before:
            seen = True
        elif ph == target and seen:
            return float(t)
    return float("nan")


def build_lineage_table(
    det: pd.DataFrame,
    curated: Curated,
    well: str = "",
    edits: Iterable[Edit] = (),
    records: Mapping[str, ReviewRecord] | None = None,
    queue: Mapping[str, Mapping[str, str]] | None = None,
    window: tuple[int, int] | None = None,
    interval_minutes: float = 20.0,
    slow: SlowAnnotations | None = None,
) -> LineageTable:
    """Collapse a well's curated detections onto nodes, edges and tracks.

    Args:
        det: Curated detections (``apply_extras`` applied): ``timepoint,
            label, area`` and, if measured, ``pip, geminin``.
        curated: The well's curated lineage (the edit log replayed).
        well: The well, to pick its entries from ``edits`` and ``records``.
        edits: The edit log (all wells; reverts handled).
        records: Review records by item id (:func:`review_records`).
        queue: Queue items by id: ``{"reason": …, "outcome": …}``.
        window: ``(start, stop)`` of the analysis window; default the
            frames present.
        interval_minutes: Time between frames.
        slow: Walker outcomes and HMM phases, if computed.

    Returns:
        The :class:`LineageTable`.
    """
    from omero_screen_track.edit import confirmed_frames

    records = records or {}
    queue = queue or {}
    slow = slow or SlowAnnotations()
    edits = live_edits(edits, well)

    # -- nodes: detections onto (track, frame) ---------------------------------
    d = det[["timepoint", "label", "area"]].copy()
    for ch in ("pip", "geminin", "y", "x"):
        d[ch] = det.get(ch, np.nan)
    d["track"] = [
        curated.tracks.get((int(t), int(lab)), 0)
        for t, lab in zip(d["timepoint"], d["label"], strict=True)
    ]
    d = d[d["track"] > 0]
    d["frame"] = d["timepoint"].astype(int)
    d["area"] = d["area"].astype(float)
    weighted = ("pip", "geminin", "y", "x")
    for ch in weighted:
        d[ch] = d[ch].astype(float) * d["area"]
    grouped = d.groupby(["track", "frame"], sort=True)
    nodes = grouped[["area", *weighted]].sum(min_count=1)
    for ch in weighted:
        nodes[ch] = nodes[ch] / nodes["area"]
    nodes["n_pieces"] = grouped.size()
    nodes["label"] = (
        d.sort_values("area").groupby(["track", "frame"])["label"].last()
    ).astype(int)
    nodes = nodes.reset_index()
    nodes["is_extra"] = [
        (int(t), int(lab)) in curated.extras
        for t, lab in zip(nodes["frame"], nodes["label"], strict=True)
    ]
    if len(nodes) and nodes["pip"].notna().any():
        nodes["phase_walker"] = walker_phases(
            nodes,
            pip_high=float(nodes["pip"].quantile(0.75)),
            gem_low=float(nodes["geminin"].quantile(0.25)),
        )
        nodes.loc[nodes["pip"].isna(), "phase_walker"] = ""
    else:
        nodes["phase_walker"] = ""
    nodes["phase_hmm"] = ""
    has_hmm = slow.phases is not None and not slow.phases.empty
    if has_hmm:
        assert slow.phases is not None
        hmm = slow.phases.set_index(["track", "frame"])["phase"]
        key = pd.MultiIndex.from_arrays([nodes["track"], nodes["frame"]])
        mapped = pd.Series(hmm.reindex(key).to_numpy(), index=nodes.index)
        nodes["phase_hmm"] = mapped.fillna("").astype(str)
    # The HMM's call where it made one (inside the window, long tracks).
    nodes["phase"] = nodes["phase_hmm"].where(
        nodes["phase_hmm"] != "", nodes["phase_walker"]
    )

    prev = nodes.groupby("track")["frame"].shift()
    nodes["after_gap"] = prev.notna() & (nodes["frame"] - prev > 1)
    edited: set[tuple[int, int]] = set()
    edits_per_track: dict[int, int] = {}
    for e in edits:
        tid = _edit_track(curated, e)
        if tid is None:
            continue
        edits_per_track[tid] = edits_per_track.get(tid, 0) + 1
        edited.update((tid, t) for t in _edit_frames(e))
    confirmed = {
        (tid, int(t))
        for tid, frames in confirmed_frames(curated, edits, well).items()
        for t in frames
    }
    pairs = list(zip(nodes["track"], nodes["frame"], strict=True))
    nodes["edited"] = [(int(a), int(b)) in edited for a, b in pairs]
    nodes["confirmed"] = [(int(a), int(b)) in confirmed for a, b in pairs]
    nodes = nodes[_NODE_COLUMNS]

    # -- tracks ----------------------------------------------------------------
    span = nodes.groupby("track")["frame"].agg(["min", "max", "size"])
    tracks = pd.DataFrame(
        {
            "start": span["min"].astype(int),
            "end": span["max"].astype(int),
            "n_frames": span["size"].astype(int),
        }
    )
    tracks.index.name = "track"
    present = set(tracks.index)
    tracks["parent"] = [
        p if (p := int(curated.parents.get(int(t), 0))) in present else 0
        for t in tracks.index
    ]
    tracks["n_gaps"] = (
        tracks["end"] - tracks["start"] + 1 - tracks["n_frames"]
    ).astype(int)
    tracks["hours"] = (
        (tracks["end"] - tracks["start"] + 1) * interval_minutes / 60
    )
    tracks["edits"] = [edits_per_track.get(int(t), 0) for t in tracks.index]
    tracks["edits_per_h"] = tracks["edits"] / tracks["hours"]

    # Review ids → tracks (an id names the cell at its anchor). A queue id
    # names the track before an id that only has a verdict; between verdicts
    # on one track a human's beats an agent's, then the latest.
    item_of: dict[int, str] = {}
    rec_of: dict[int, ReviewRecord] = {}
    authors_of: dict[int, set[str]] = {}
    flags_of: dict[int, str] = {}
    queued: set[int] = set()
    auto_outcome: dict[int, str] = {}
    ids = sorted(queue) + sorted(set(records) - set(queue))
    for item_id in ids:
        parsed = _parse_id(item_id)
        if parsed is None or parsed[0] != well:
            continue
        try:
            tid = curated.resolve((parsed[1], parsed[2]))
        except Exception:
            continue
        if tid not in present:
            continue
        item_of.setdefault(tid, item_id)
        if item_id in queue:
            queued.add(tid)
            reason = str(queue[item_id].get("reason", "") or "")
            if reason and reason != "search":
                flags_of.setdefault(tid, reason)
            if queue[item_id].get("outcome"):
                auto_outcome.setdefault(tid, str(queue[item_id]["outcome"]))
        rec = records.get(item_id)
        if rec is None:
            continue
        authors_of.setdefault(tid, set()).update(
            set(rec.reviewed_by.split("+")) - {""}
        )
        old = rec_of.get(tid)
        if old is None or _rank(rec) >= _rank(old):
            rec_of[tid] = ReviewRecord(
                verdict=rec.verdict,
                outcome=rec.outcome,
                reviewed_by=rec.reviewed_by,
                audit=rec.audit or (old.audit if old else ""),
                author=rec.author,
                time=rec.time,
            )
        elif rec.audit and not old.audit:
            rec_of[tid] = replace(old, audit=rec.audit)
    for tid, authors in authors_of.items():
        rec_of[tid] = replace(
            rec_of[tid], reviewed_by="+".join(sorted(authors))
        )

    # Walker outcomes (a cell's outcome on every track it was followed through).
    walker: dict[int, dict[str, Any]] = {}
    if slow.fates is not None and not slow.fates.empty:
        for row in slow.fates.itertuples():
            segs = [int(x) for x in str(getattr(row, "segments", "")).split()]
            for seg in segs:
                walker[seg] = {
                    "outcome": str(row.outcome),
                    "frame": getattr(row, "outcome_frame", np.nan),
                    "excluded": bool(getattr(row, "excluded", False)),
                    "flags": str(getattr(row, "review_flags", "") or ""),
                    # The outcome happens at the end of the last segment.
                    "last": seg == segs[-1],
                }
    children: dict[int, list[int]] = {}
    for t, p in zip(tracks.index, tracks["parent"], strict=True):
        if p:
            children.setdefault(int(p), []).append(int(t))
    n_frames_of = dict(
        zip(
            tracks.index.to_numpy().tolist(),
            tracks["n_frames"].tolist(),
            strict=True,
        )
    )

    def really_divided(tid: int, rec: ReviewRecord, notes: dict) -> bool:
        """The reviewer's outcome, a curated event, else the walker's.

        A track the walker did not follow (a daughter, a cell born after
        the window start) counts as divided if at least two of its
        tracker daughters are long enough to be cells.
        """
        if rec.verdict in ("accept", "correct") and rec.outcome:
            return rec.outcome == "divided"
        kinds = {e["kind"] for e in notes.get("events", [])}
        if "mitosis" in kinds or notes.get("outcome") == "divided":
            return True
        if notes.get("outcome") or "excluded" in notes:
            return False
        w = walker.get(tid)
        if w is not None:
            return w["outcome"] == "divided" and w["last"]
        long_kids = [
            k
            for k in children.get(tid, [])
            if n_frames_of[k] >= MIN_DAUGHTER_FRAMES
        ]
        return len(long_kids) >= 2

    outcome, verdict, by, audit, excluded, flags, item_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    divided: dict[int, bool] = {}
    for tid in tracks.index:
        tid = int(tid)
        rec = rec_of.get(tid, ReviewRecord())
        notes = curated.annotations.get(tid, {})
        w = walker.get(tid, {})
        divided[tid] = really_divided(tid, rec, notes)
        out = ""
        if rec.verdict in ("accept", "correct") and rec.outcome:
            out = rec.outcome
        elif "excluded" in notes:
            out = "excluded"
        elif "outcome" in notes:
            out = str(notes["outcome"])
        elif notes.get("events"):
            kind = sorted(notes["events"], key=lambda e: e["frame"])[0]
            out = {
                "mitosis": "divided",
                "death": "death",
                "slippage": "mitotic_exit_no_division",
            }.get(kind["kind"], "")
        elif w.get("outcome"):
            out = w["outcome"]
        elif tid in auto_outcome:
            out = auto_outcome[tid]
        elif divided[tid]:
            out = "divided"
        outcome.append(out)
        verdict.append(rec.verdict)
        by.append(rec.reviewed_by)
        audit.append(rec.audit)
        excluded.append(
            rec.verdict == "reject"
            or "excluded" in notes
            or bool(w.get("excluded"))
        )
        flags.append(
            " ".join(
                f for f in (flags_of.get(tid, ""), w.get("flags", "")) if f
            )
        )
        item_ids.append(item_of.get(tid, ""))
    tracks["item_id"] = item_ids
    tracks["queued"] = [int(t) in queued for t in tracks.index]
    tracks["flags"] = flags
    tracks["outcome"] = outcome
    tracks["verdict"] = verdict
    tracks["reviewed_by"] = by
    tracks["audit"] = audit
    tracks["excluded"] = excluded
    tracks["divided"] = [divided[int(t)] for t in tracks.index]

    # -- events ------------------------------------------------------------------
    start, stop = window or (
        int(tracks["start"].min()) if len(tracks) else 0,
        int(tracks["end"].max()) if len(tracks) else 0,
    )
    tracks["ev_window"] = np.where(
        (tracks["start"] <= start) & (tracks["end"] >= start), start, np.nan
    )
    bounds: dict[int, tuple[float, float]] = {}
    if slow.boundaries is not None:
        bounds = {
            int(t): (float(s_), float(g2_))
            for t, s_, g2_ in zip(
                slow.boundaries["track"],
                slow.boundaries["s_entry"],
                slow.boundaries["g2_entry"],
                strict=True,
            )
        }
    ids = tracks.index.to_numpy()
    start_of = dict(zip(ids.tolist(), tracks["start"].tolist(), strict=True))
    end_of = dict(zip(ids.tolist(), tracks["end"].tolist(), strict=True))
    by_track: dict[int, pd.DataFrame] = {
        int(np.asarray(t).item()): g
        for t, g in nodes.groupby("track", sort=False)
    }
    ev = {"ev_division": [], "ev_exit": [], "ev_g2": [], "ev_s": []}
    for tid in tracks.index:
        tid = int(tid)
        notes = curated.annotations.get(tid, {})
        events = {e["kind"]: int(e["frame"]) for e in notes.get("events", [])}
        g = by_track[tid]
        div = float("nan")
        if divided[tid]:
            w_frame = walker.get(tid, {}).get("frame", np.nan)
            if "mitosis" in events:
                div = float(events["mitosis"])
            elif tid not in children and np.isfinite(w_frame):
                # The walker dates a division the frame after the mother.
                div = float(min(w_frame - 1, end_of[tid]))
            else:
                div = float(end_of[tid])
        ph = g["phase"].to_numpy()
        fr = g["frame"].to_numpy()
        exit_ = float("nan")
        if "slippage" in events:
            exit_ = float(events["slippage"])
        elif (ph == "E").any():
            exit_ = float(fr[ph == "E"][0])
        elif walker.get(tid, {}).get("outcome") == "mitotic_exit_no_division":
            exit_ = float(walker[tid]["frame"])
        if tid in bounds:
            # An HMM boundary at the track's first frame is "before" it.
            s, g2 = (
                b if start_of[tid] < b <= end_of[tid] else float("nan")
                for b in bounds[tid]
            )
        else:
            g2 = _first_entry(ph, fr, "G2", {"S"})
            s = _first_entry(ph, fr, "S", {"G1", "X"})
        ev["ev_division"].append(div)
        ev["ev_exit"].append(exit_)
        ev["ev_g2"].append(g2)
        ev["ev_s"].append(s)
    for col, vals in ev.items():
        tracks[col] = np.asarray(vals, dtype=float)

    # -- edges -------------------------------------------------------------------
    rows: list[tuple[int, int, int, int, str]] = []
    for tid, g in by_track.items():
        fr = g["frame"].to_numpy()
        for f0, f1 in zip(fr[:-1], fr[1:], strict=True):
            rows.append(
                (tid, tid, int(f0), int(f1), "link" if f1 - f0 == 1 else "gap")
            )
    for parent, kids in children.items():
        for kid in kids:
            real = divided[parent] and n_frames_of[kid] >= MIN_DAUGHTER_FRAMES
            rows.append(
                (
                    kid,
                    parent,
                    end_of[parent],
                    start_of[kid],
                    "division" if real else "split",
                )
            )
    edges = pd.DataFrame(rows, columns=["track", "source", "f0", "f1", "kind"])
    return LineageTable(
        nodes=nodes.reset_index(drop=True),
        edges=edges,
        tracks=tracks,
        well=well,
        interval_minutes=interval_minutes,
        window=(int(start), int(stop)),
        has_hmm=has_hmm,
        pooled=slow.pooled,
    )


# -- layout ----------------------------------------------------------------------


def _passes(row: Any, f: Filters) -> bool:
    if row.n_frames < f.min_length:
        return False
    v = row.verdict
    if f.verdict == "accepted" and v not in ("accept", "correct"):
        return False
    if (
        f.verdict in ("accept", "correct", "reject", "unsure")
        and v != f.verdict
    ):
        return False
    if f.verdict == "any" and not v:
        return False
    if f.verdict == "none" and v:
        return False
    if f.outcome != "all" and row.outcome != f.outcome:
        return False
    by = set(row.reviewed_by.split("+")) - {""}
    want = f.reviewed_by
    if want in ("agent", "human") and by != {want}:
        return False
    if want == "both" and by != {"agent", "human"}:
        return False
    if want == "any" and not by:
        return False
    if want == "none" and by:
        return False
    if f.audited_only and not row.audit:
        return False
    return not (f.flagged_only and not row.flags)


def filter_tracks(tracks: pd.DataFrame, filters: Filters) -> set[int]:
    """Track ids kept by ``filters`` (with descendants if asked)."""
    kept: set[int] = {
        int(t)
        for t, row in zip(
            tracks.index.to_numpy(), tracks.itertuples(), strict=True
        )
        if _passes(row, filters)
    }
    if filters.with_descendants and kept:
        n_frames = dict(
            zip(
                tracks.index.to_numpy().tolist(),
                tracks["n_frames"].tolist(),
                strict=True,
            )
        )
        children: dict[int, list[int]] = {}
        for t, p in zip(tracks.index, tracks["parent"], strict=True):
            if p:
                children.setdefault(int(p), []).append(int(t))
        stack = list(kept)
        while stack:
            for kid in children.get(stack.pop(), []):
                if kid not in kept and n_frames[kid] >= filters.min_length:
                    kept.add(kid)
                    stack.append(kid)
    return kept


def _event_column(align: str) -> str | None:
    return None if align == "frame" else f"ev_{align}"


def layout(
    table: LineageTable,
    align: str = "frame",
    sort: str = "track",
    filters: Filters | None = None,
    park: bool = True,
    tree_gap: float = 0.5,
) -> Layout:
    """Place the kept tracks: a column each, y = frame − the tree's anchor.

    Trees (a founder and its descendants among the kept tracks) are laid out
    side by side. Within a tree the leaves take consecutive columns in
    depth-first order (daughters by track id) and a mother sits centred over
    her daughters, so a division is a fork. A whole tree shares one offset,
    so its edges stay connected: the anchor is the alignment event of the
    first track in depth-first order that has it. Trees without the event are
    hidden, or with ``park`` placed after the others, unaligned (offset = the
    median anchor of the aligned trees).

    Args:
        table: :func:`build_lineage_table` output.
        align: A key of :data:`ALIGNMENTS`.
        sort: One of :data:`SORTS`; trees are ordered by their founder's
            value (``event``: the anchor), ties by founder id.
        filters: Which tracks to keep.
        park: Keep trees without the alignment event, at the end.
        tree_gap: Extra columns between trees.

    Raises:
        ValueError: If ``align`` or ``sort`` is unknown.
    """
    if align not in ALIGNMENTS:
        raise ValueError(f"Unknown alignment {align!r}.")
    if sort not in SORTS:
        raise ValueError(f"Unknown sort {sort!r}.")
    filters = filters or Filters()
    tracks = table.tracks
    kept = filter_tracks(tracks, filters)
    sub = tracks.loc[sorted(kept)]
    children: dict[int, list[int]] = {}
    roots: list[int] = []
    for t, p in zip(sub.index, sub["parent"], strict=True):
        if p and p in kept:
            children.setdefault(int(p), []).append(int(t))
        else:
            roots.append(int(t))

    def preorder(root: int) -> list[int]:
        out, stack = [], [root]
        while stack:
            t = stack.pop()
            out.append(t)
            stack.extend(reversed(children.get(t, [])))
        return out

    col = _event_column(align)
    anchors: dict[int, float] = {}
    for r in roots:
        if col is None:
            anchors[r] = 0.0
            continue
        vals = sub.loc[preorder(r), col].dropna()
        anchors[r] = float(vals.iloc[0]) if len(vals) else float("nan")
    aligned = [r for r in roots if np.isfinite(anchors[r])]
    parked = [r for r in roots if not np.isfinite(anchors[r])]
    if not park:
        parked = []
    fallback = (
        float(np.median([anchors[r] for r in aligned])) if aligned else 0.0
    )

    n_frames = sub["n_frames"].to_dict()
    outcomes = sub["outcome"].to_dict()
    density = sub["edits_per_h"].to_dict()

    def key(r: int) -> tuple[Any, ...]:
        if sort == "length":
            return (-n_frames[r], r)
        if sort == "outcome":
            return (outcomes[r] or "~", r)
        if sort == "event":
            a = anchors[r]
            return (a if np.isfinite(a) else np.inf, r)
        if sort == "edits":
            return (-density[r], r)
        return (r,)

    aligned.sort(key=key)
    parked.sort(key=key)

    x: dict[int, float] = {}
    offset: dict[int, float] = {}
    is_parked: dict[int, bool] = {}
    cursor = 0.0
    parked_from: float | None = None
    for i, r in enumerate(aligned + parked):
        if i == len(aligned) and parked:
            parked_from = cursor
        off = anchors[r] if np.isfinite(anchors[r]) else fallback
        order = preorder(r)
        leaf_x: dict[int, float] = {}
        for t in order:
            if t not in children:
                leaf_x[t] = cursor
                cursor += 1
        for t in reversed(order):
            if t in children:
                xs = [leaf_x[c] for c in children[t]]
                leaf_x[t] = float(np.mean(xs))
        for t in order:
            x[t] = leaf_x[t]
            offset[t] = off
            is_parked[t] = not np.isfinite(anchors[r])
        cursor += tree_gap

    placed = sorted(x)
    lt = sub.loc[placed].copy()
    lt["x"] = [x[t] for t in placed]
    lt["offset"] = [offset[t] for t in placed]
    lt["y0"] = lt["start"] - lt["offset"]
    lt["y1"] = lt["end"] - lt["offset"]
    lt["parked"] = [is_parked[t] for t in placed]

    nodes = table.nodes[table.nodes["track"].isin(placed)].copy()
    nodes["x"] = nodes["track"].map(x).astype(float)
    nodes["y"] = (nodes["frame"] - nodes["track"].map(offset)).astype(float)
    edges = table.edges[
        table.edges["track"].isin(placed) & table.edges["source"].isin(placed)
    ].copy()
    edges["x0"] = edges["source"].map(x).astype(float)
    edges["x1"] = edges["track"].map(x).astype(float)
    edges["y0"] = (edges["f0"] - edges["source"].map(offset)).astype(float)
    edges["y1"] = (edges["f1"] - edges["track"].map(offset)).astype(float)
    return Layout(
        nodes=nodes.reset_index(drop=True),
        edges=edges.reset_index(drop=True),
        tracks=lt,
        n_columns=int(np.ceil(cursor)),
        parked_from=parked_from,
    )


# -- colours ---------------------------------------------------------------------


def edit_density_bin(per_hour: float) -> str:
    """Category of :data:`PALETTES` ``edits`` for an edit density."""
    if per_hour <= 0:
        return "0"
    if per_hour <= 0.5:
        return "≤0.5/h"
    if per_hour <= 1:
        return "≤1/h"
    if per_hour <= 2:
        return "≤2/h"
    return ">2/h"


def node_categories(
    nodes: pd.DataFrame, tracks: pd.DataFrame, mode: str
) -> np.ndarray:
    """Category of every node for a colour mode (keys of :data:`PALETTES` ``[mode]``).

    ``phase`` is per frame; ``verdict``, ``outcome`` and ``edits`` are the
    track's; ``flags`` is per frame, by priority: a reviewer mask, a frame a
    reviewer confirmed (link / swap), any other edited frame, the first frame
    after a gap.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode not in COLOUR_MODES:
        raise ValueError(f"Unknown colour mode {mode!r}.")
    if nodes.empty:
        return np.array([], dtype=object)
    palette = PALETTES[mode]
    if mode == "phase":
        cats = nodes["phase"].fillna("").astype(str).to_numpy()
    elif mode == "flags":
        cats = np.select(
            [
                nodes["is_extra"].to_numpy(bool),
                nodes["confirmed"].to_numpy(bool),
                nodes["edited"].to_numpy(bool),
                nodes["after_gap"].to_numpy(bool),
            ],
            ["mask", "switch", "edited", "after gap"],
            "",
        ).astype(object)
    elif mode == "edits":
        per_track = tracks["edits_per_h"].map(edit_density_bin)
        cats = nodes["track"].map(per_track).to_numpy()
    else:
        cats = nodes["track"].map(tracks[mode]).fillna("").to_numpy()
    cats = np.asarray(cats, dtype=object)
    unknown = ~np.isin(cats, list(palette))
    cats[unknown] = "" if "" in palette else next(iter(palette))
    return cats


# -- gathering from a review session -----------------------------------------------


@dataclass(frozen=True)
class WellInputs:
    """What :func:`build_lineage_table` reads for one well, as a snapshot.

    The objects are never modified after they are read (an edit makes the
    session replay into a new :class:`~omero_screen_track.edit.Curated`), so a
    worker may compute :func:`slow_annotations` from them.

    Attributes:
        plate_id: Plate, for the plate-pooled phase reference (``None``:
            fit the reference on this well).
        db_path: CellView database the plate is read from.
        reference: A phase reference already fitted for the plate.
        review_dir: The review directory the curation comes from, if any.
    """

    det: pd.DataFrame
    curated: Any
    well: str
    edits: list[Any]
    records: dict[str, ReviewRecord]
    queue: dict[str, dict[str, str]]
    window: tuple[int, int]
    interval_minutes: float
    plate_id: int | None = None
    db_path: Path | None = None
    reference: Any = None
    review_dir: Path | None = None


def _spec(queue_path: Path | None) -> dict[str, Any]:
    if queue_path is None or not Path(queue_path).exists():
        return {}
    try:
        return json.loads(Path(queue_path).read_text()).get("spec", {}) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_inputs(
    source: CellSource,
    queue_path: Path | None,
    queue: Queue | None,
    well: str,
    window: tuple[int, int] | None = None,
) -> WellInputs:
    """Read one well's curated lineage and review records.

    Args:
        source: The plate's cached curated lineages (its ``log_path`` is
            the review directory's edit log, or ``None``: CellView's
            lineage as it is).
        queue_path: The queue file; its directory holds the decisions.
            ``None`` for a well without a review.
        queue: The parsed queue (items give flags and automatic outcomes).
        well: The well.
        window: Analysis window if the queue does not name one.
    """
    from omero_screen_track.edit import EditLog

    det, curated = source.well(well)
    spec = _spec(queue_path)
    default = window or (
        int(det["timepoint"].min()),
        int(det["timepoint"].max()),
    )
    start = int(spec.get("start", default[0]))
    stop = int(spec.get("stop", default[1]))
    interval = float(spec.get("interval_minutes", 20.0))
    log = source.log_path
    edits = EditLog(log).entries() if log and log.exists() else []
    items = {
        it.id: {"reason": it.reason, "outcome": it.outcome}
        for it in (queue.items if queue else ())
        if it.well == well
    }
    review_dir = Path(queue_path).parent if queue_path else None
    return WellInputs(
        det=det,
        curated=curated,
        well=well,
        edits=edits,
        records=review_records(review_dir) if review_dir else {},
        queue=items,
        window=(start, stop),
        interval_minutes=interval,
        plate_id=source.plate_id,
        db_path=source.db_path,
        review_dir=review_dir,
    )


def session_inputs(session: ReviewSession, well: str) -> WellInputs:
    """:func:`read_inputs` from a review session (its cached lineage).

    Call ``session.refresh(well)`` first to pick up edits made elsewhere.
    """
    return read_inputs(session.source, session.queue_path, session.queue, well)


def table_from_inputs(
    inputs: WellInputs, slow: SlowAnnotations | None = None
) -> LineageTable:
    """:func:`build_lineage_table` on a :class:`WellInputs` snapshot."""
    return build_lineage_table(
        inputs.det,
        inputs.curated,
        well=inputs.well,
        edits=inputs.edits,
        records=inputs.records,
        queue=inputs.queue,
        window=inputs.window,
        interval_minutes=inputs.interval_minutes,
        slow=slow,
    )


def lineage_from_session(
    session: ReviewSession, well: str, slow: SlowAnnotations | None = None
) -> LineageTable:
    """The :class:`LineageTable` of a well of a review session."""
    return table_from_inputs(session_inputs(session, well), slow)


# -- plate-pooled phase reference ----------------------------------------------------


def cache_dir() -> Path:
    """Where the track view keeps derived data (never in a review directory)."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "omero-screen" / "track_view"


def tracked_wells(plate_id: int, db_path: Path | None = None) -> list[str]:
    """Wells of a plate with tracked detections in CellView, in plate order."""
    import duckdb

    from omero_screen_napari.review.cells import default_db_path

    conn = duckdb.connect(str(db_path or default_db_path()), read_only=True)
    try:
        rows = conn.execute(
            """
            select distinct c.well from measurements m
            join conditions c using (condition_id)
            join repeats r using (repeat_id)
            where r.plate_id = ? and m.track_id_raw is not null
            """,
            [plate_id],
        ).fetchall()
    finally:
        conn.close()
    return sorted((str(r[0]) for r in rows), key=_well_key)


def _well_key(well: str) -> tuple[str, int]:
    row = "".join(c for c in well if c.isalpha())
    col = "".join(c for c in well if c.isdigit())
    return row, int(col or 0)


def plate_reference(
    plate_id: int,
    db_path: Path | None,
    window: tuple[int, int],
    params: Any = None,
) -> Any:
    """The phase reference fitted once on every tracked well of the plate.

    As in the fate build, the bleed-through factor and the PIP / geminin
    scales come from all wells together, so a phase means the same reporter
    state in every well; fitted on a knockdown well alone, few S/G2 cells
    give a low geminin scale that turns noise into spurious S, G2 or exit
    calls. Detections in the analysis window, debris hidden, are pooled.
    Cached in :func:`cache_dir`, keyed by the database file's size and time.

    Returns:
        A :class:`omero_screen_track.phases.WellReference`.
    """
    import duckdb
    from omero_screen_track.cells import load_well
    from omero_screen_track.debris import drop_debris
    from omero_screen_track.phases import (
        PhaseParams,
        WellReference,
        fit_reference,
    )

    from omero_screen_napari.review.cells import default_db_path

    params = params or PhaseParams()
    db = Path(db_path or default_db_path())
    stat = db.stat()
    key = {
        "plate": plate_id,
        "db": str(db.resolve()),
        "db_size": stat.st_size,
        "db_mtime": stat.st_mtime,
        "window": list(window),
        "quantiles": [
            params.bleed_pip_quantile,
            params.bleed_quantile,
            params.pip_ref_quantile,
            params.gem_ref_quantile,
        ],
    }
    path = cache_dir() / f"phase_reference_plate{plate_id}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            if cached.get("key") == key:
                return WellReference(**cached["reference"])
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    wells = tracked_wells(plate_id, db)
    conn = duckdb.connect(str(db), read_only=True)
    try:
        frames = []
        for well in wells:
            det, _ = drop_debris(load_well(conn, plate_id, well))
            if not {"pip", "geminin"} <= set(det.columns):
                continue
            inside = det["timepoint"].between(window[0], window[1])
            frames.append(det.loc[inside, ["pip", "geminin"]])
    finally:
        conn.close()
    if not frames:
        raise ValueError(f"No PIP / geminin data on plate {plate_id}.")
    ref = fit_reference(pd.concat(frames, ignore_index=True), params)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"key": key, "wells": wells, "reference": ref.__dict__},
                indent=1,
            )
        )
    except OSError:
        pass
    return ref


def slow_annotations(
    inputs: WellInputs,
    walker: bool = True,
    hmm: bool = True,
    min_frames: int = MIN_HMM_FRAMES,
) -> SlowAnnotations:
    """Fate-walker outcomes and HMM phases for one well (seconds; run off the UI thread).

    Phases are called as the fate build calls them: with the plate-pooled
    reference (:func:`plate_reference`) and on the analysis window only, so
    S entry, G2 entry and exit match the fate tables. Without a plate (a
    toy well) the reference is fitted on the well itself
    (``SlowAnnotations.pooled`` is then ``False``).

    Args:
        inputs: The well's snapshot (:func:`read_inputs`).
        walker: Run :func:`~omero_screen_track.fate.follow_cells` over the
            window.
        hmm: Call phases with :func:`~omero_screen_track.phases.annotate_phases`.
        min_frames: Shortest track (frames with a mask in the window) given
            HMM phases.
    """
    from omero_screen_track.cells import curated_tracks

    out = SlowAnnotations()
    det, curated = inputs.det, inputs.curated
    try:
        tracks = curated_tracks(curated, det)
    except Exception as err:
        out.errors.append(f"tracks: {err}")
        return out
    have_reporters = {"pip", "geminin"} <= set(tracks.columns)
    if walker and have_reporters:
        from omero_screen_track.edit import confirmed_frames
        from omero_screen_track.fate import apply_annotations, follow_cells

        try:
            if "dna" not in tracks:
                tracks["dna"] = tracks["area"] * 0.0
            shape = (int(det["y"].max()) + 1, int(det["x"].max()) + 1)
            live = live_edits(inputs.edits, inputs.well)
            cells, _ = follow_cells(
                tracks,
                inputs.window[0],
                inputs.window[1],
                None,
                None,
                None,
                shape,
                confirmed_frames(curated, live, inputs.well),
            )
            out.fates = apply_annotations(cells, curated)
        except Exception as err:
            out.errors.append(f"fate walker: {err}")
    if hmm and have_reporters:
        from omero_screen_track.phases import (
            PhaseParams,
            annotate_phases,
            fit_reference,
        )

        params = PhaseParams(n_samples=1)
        try:
            inside = tracks["timepoint"].between(*inputs.window)
            st = tracks.loc[
                inside, ["track_id", "timepoint", "pip", "geminin"]
            ].rename(columns={"track_id": "cell"})
            n = st.groupby("cell")["timepoint"].transform("size")
            st = st[n >= min_frames]
            ref = inputs.reference
            if ref is None and inputs.plate_id is not None:
                try:
                    ref = plate_reference(
                        inputs.plate_id, inputs.db_path, inputs.window, params
                    )
                except Exception as err:
                    out.errors.append(f"plate reference: {err}")
            out.pooled = ref is not None
            if ref is None and not st.empty:
                ref = fit_reference(st, params)
            out.reference = ref
            if not st.empty:
                oc, os_, _ = annotate_phases(
                    st[["cell"]].drop_duplicates(), st, params, reference=ref
                )
                out.phases = os_.rename(
                    columns={"cell": "track", "timepoint": "frame"}
                )[["track", "frame", "phase"]]
                out.boundaries = oc.rename(columns={"cell": "track"})[
                    ["track", "s_entry", "g2_entry", "exit"]
                ]
        except Exception as err:
            out.errors.append(f"phases: {err}")
    elif hmm:
        out.errors.append("phases: no PIP / geminin channels")
    return out


# -- loaders ---------------------------------------------------------------------------


def _reloading(
    source: CellSource, read: Callable[[str], WellInputs]
) -> Callable[[str], WellInputs]:
    """Wrap ``read`` so a well is replayed again after its edit log changed."""
    seen: dict[str, float] = {}

    def load(well: str) -> WellInputs:
        log = source.log_path
        mtime = log.stat().st_mtime if log and log.exists() else 0.0
        if well in seen and seen[well] != mtime:
            source._wells.pop(well, None)
        seen[well] = mtime
        return read(well)

    return load


def session_loader(session: ReviewSession) -> Callable[[str], WellInputs]:
    """A loader for the track view on a review session's wells.

    The session caches each well's replayed lineage; when ``edits.jsonl``
    has changed since the last load (an edit from the review widget, the
    agent or another process), the well is replayed again first.
    """
    return _reloading(
        session.source, lambda well: session_inputs(session, well)
    )


def directory_loader(
    queue_path: Path, db_path: Path | None = None
) -> tuple[list[str], Callable[[str], WellInputs]]:
    """Wells and a loader for a review directory, without a review session.

    Reads only: unlike opening a :class:`ReviewSession`, nothing is written
    to the directory (no proposals are seeded).

    Returns:
        ``(wells in the queue, loader)``.
    """
    from omero_screen_napari.review.cells import CellSource
    from omero_screen_napari.review_queue import read_queue

    queue_path = Path(queue_path)
    queue = read_queue(queue_path)
    source = CellSource(
        queue.plate_id,
        db_path=db_path,
        log_path=queue_path.with_name("edits.jsonl"),
    )
    wells = sorted({it.well for it in queue.items})
    return wells, _reloading(
        source, lambda well: read_inputs(source, queue_path, queue, well)
    )


def watch_paths(session: ReviewSession) -> list[Path]:
    """Files and directories whose change should rebuild the view."""
    return review_watch_paths(session.queue_path.parent)


def review_watch_paths(review_dir: Path | None) -> list[Path]:
    """A review directory, its edit log and its decisions files."""
    if review_dir is None:
        return []
    review_dir = Path(review_dir)
    return [
        review_dir,
        review_dir / "edits.jsonl",
        *sorted(review_dir.glob("*decisions.json")),
    ]


# -- review directories of a plate -----------------------------------------------------


@dataclass(frozen=True)
class ReviewDir:
    """A review directory found beside another.

    Attributes:
        path: The directory.
        plate_id: Its plate.
        wells: Wells its curation is for.
        newest: Time of its newest ``*decisions.json`` (else of its edit
            log): the most recently reviewed directory wins a well.
    """

    path: Path
    plate_id: int
    wells: tuple[str, ...]
    newest: float


def read_review_dir(path: Path) -> ReviewDir | None:
    """Describe ``path`` if it is a review directory (``edits.jsonl`` plus
    ``correction.json`` or ``queue.json``), else ``None``."""
    path = Path(path)
    edits = path / "edits.jsonl"
    correction, queue = path / "correction.json", path / "queue.json"
    if not edits.exists() or not (correction.exists() or queue.exists()):
        return None
    plate: int | None = None
    wells: set[str] = set()
    try:
        if correction.exists():
            data = json.loads(correction.read_text())
            plate = data.get("plate_id")
            wells.update(str(w) for w in data.get("wells", []))
        if queue.exists():
            data = json.loads(queue.read_text())
            plate = plate if plate is not None else data.get("plate_id")
            if not wells:
                wells.update(str(it["well"]) for it in data.get("items", []))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if plate is None or not wells:
        return None
    stamps = [p.stat().st_mtime for p in path.glob("*decisions.json")]
    return ReviewDir(
        path=path,
        plate_id=int(plate),
        wells=tuple(sorted(wells, key=_well_key)),
        newest=max(stamps) if stamps else edits.stat().st_mtime,
    )


def discover_review_dirs(
    review_dir: Path, plate_id: int
) -> dict[str, list[ReviewDir]]:
    """Review directories of a plate beside ``review_dir``, per well.

    ``review_dir`` and its sibling directories are read (one level, so the
    ``_superseded_*`` folders inside a review directory are not). For each
    well the directories are listed newest review first.
    """
    review_dir = Path(review_dir)
    parent = review_dir.parent
    candidates = [review_dir]
    if parent.is_dir():
        candidates += sorted(
            p for p in parent.iterdir() if p.is_dir() and p != review_dir
        )
    found: dict[str, list[ReviewDir]] = {}
    for path in candidates:
        rd = read_review_dir(path)
        if rd is None or rd.plate_id != plate_id:
            continue
        for well in rd.wells:
            found.setdefault(well, []).append(rd)
    for dirs in found.values():
        dirs.sort(key=lambda d: -d.newest)
    return found


class PlateLoader:
    """Loads any well of a plate for the track view.

    The Track Review session's wells, from its own review directory, are
    read through the session (so its edits show at once and clicking a node
    may open the cell). Every other well is read from the review directory
    found for it beside the session's (:func:`discover_review_dirs`; the
    user may choose another, or none), read-only, or from CellView alone.

    Args:
        plate_id: The plate.
        db_path: CellView database.
        session: The open review session, if any.
        review_dir: Where to look for review directories (default: the
            session's).
        wells: Wells to offer (default: every tracked well of the plate).
    """

    def __init__(
        self,
        plate_id: int,
        db_path: Path | None = None,
        session: ReviewSession | None = None,
        review_dir: Path | None = None,
        wells: list[str] | None = None,
    ) -> None:
        self.plate_id = plate_id
        self.db_path = db_path
        self.session = session
        self.home = (
            Path(review_dir)
            if review_dir
            else session.queue_path.parent
            if session
            else None
        )
        self.dirs = (
            discover_review_dirs(self.home, plate_id) if self.home else {}
        )
        session_wells = (
            {it.well for it in session.queue.items} if session else set()
        )
        if wells is None:
            try:
                wells = tracked_wells(plate_id, db_path)
            except Exception:  # no database: what the reviews name
                wells = []
            wells = sorted({*wells, *self.dirs, *session_wells}, key=_well_key)
        self.wells = wells
        self.choice: dict[str, Path | None] = {}
        for well in session_wells:
            if self.home is not None:
                self.choice[well] = self.home
        self.reference: Any = None
        self._sources: dict[Path | None, CellSource] = {}
        self._loaders: dict[tuple[str, Path | None], Callable] = {}
        spec = _spec(self.home / "queue.json") if self.home else {}
        self.window = (
            (int(spec["start"]), int(spec["stop"]))
            if "start" in spec and "stop" in spec
            else None
        )

    def candidates(self, well: str) -> list[Path]:
        """Review directories found for ``well``, newest review first."""
        return [d.path for d in self.dirs.get(well, [])]

    def review_dir(self, well: str) -> Path | None:
        """The review directory ``well`` is read from (``None``: CellView only)."""
        if well in self.choice:
            return self.choice[well]
        found = self.candidates(well)
        return found[0] if found else None

    def set_review_dir(self, well: str, path: Path | None) -> None:
        """Read ``well`` from another review directory (``None``: none)."""
        self.choice[well] = Path(path) if path else None

    def interactive(self, well: str) -> bool:
        """Whether ``well`` is shown from the review session (cells can be opened)."""
        return (
            self.session is not None
            and self.home is not None
            and self.review_dir(well) == self.home
            and any(it.well == well for it in self.session.queue.items)
        )

    def watch_paths(self, well: str) -> list[Path]:
        """What to watch for ``well``."""
        return review_watch_paths(self.review_dir(well))

    def _source(self, review_dir: Path | None) -> CellSource:
        from omero_screen_napari.review.cells import CellSource

        if review_dir not in self._sources:
            log = review_dir / "edits.jsonl" if review_dir else None
            self._sources[review_dir] = CellSource(
                self.plate_id, db_path=self.db_path, log_path=log
            )
        return self._sources[review_dir]

    def __call__(self, well: str) -> WellInputs:
        """Read ``well``."""
        from omero_screen_napari.review_queue import read_queue

        review_dir = self.review_dir(well)
        key = (well, review_dir)
        if key not in self._loaders:
            if self.interactive(well):
                assert self.session is not None
                self._loaders[key] = session_loader(self.session)
            else:
                source = self._source(review_dir)
                qpath = review_dir / "queue.json" if review_dir else None
                queue = read_queue(qpath) if qpath and qpath.exists() else None
                window = self.window

                def read(
                    w: str,
                    source: CellSource = source,
                    qpath: Path | None = qpath,
                    queue: Queue | None = queue,
                ) -> WellInputs:
                    return read_inputs(source, qpath, queue, w, window)

                self._loaders[key] = _reloading(source, read)
        inputs = self._loaders[key](well)
        return replace(inputs, reference=self.reference)

    def remember(self, slow: SlowAnnotations) -> None:
        """Keep the plate's phase reference from a finished run."""
        if slow.pooled and slow.reference is not None:
            self.reference = slow.reference


def run_slow_annotations_subprocess(
    inputs: WellInputs, timeout: float = 900.0
) -> SlowAnnotations:
    """:func:`slow_annotations` in a fresh Python process.

    The HMM's loops hold the GIL, so in a thread of the napari process they
    would stall the viewer. A child process (``python -m`` this module, the
    inputs pickled through stdin and the result back through stdout) leaves
    the calling thread only waiting on a pipe. Unlike a multiprocessing pool,
    the child never re-imports the caller's ``__main__``.

    Raises:
        RuntimeError: If the child fails.
    """
    import pickle
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", __name__],
        input=pickle.dumps(inputs),
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-3:]
        raise RuntimeError("phase worker failed: " + " / ".join(tail))
    return pickle.loads(proc.stdout)


def _worker_main() -> None:
    """Child side of :func:`run_slow_annotations_subprocess`."""
    import pickle
    import sys
    import warnings

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    inputs = pickle.loads(sys.stdin.buffer.read())
    result = slow_annotations(inputs)
    sys.stdout.buffer.write(pickle.dumps(result))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    # Run through the imported module, so the result pickles as
    # ``track_view_model.SlowAnnotations``, not ``__main__.SlowAnnotations``.
    from omero_screen_napari.review import track_view_model

    track_view_model._worker_main()
