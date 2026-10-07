"""Deterministic repair of Trackastra lineages from segmentation flicker.

Trackastra links detections; it cannot fix them. When Cellpose splits one
nucleus into two pieces, the tracker has no choice but to report a division,
and on a 72 h movie most of its divisions are of this kind. This module
rewrites the lineage graph so that those events disappear, and leaves the
original output untouched: it reads ``track_id_raw`` / ``parent_track_id_raw``
and produces the curated ``track_id`` / ``parent_track_id``.

The repair is a relabelling. Every raw track is assigned to a repaired track,
and two raw tracks that are pieces of the same nucleus get the same id. Because
the cached nucleus masks carry the raw track id as their pixel value, applying
the same map to a mask *is* the segmentation fix: the two pieces become one
label. Split pieces share their boundary with no gap between them, so merged
area and intensity are exact (area sum, area-weighted mean) without touching
pixels.

Each two-daughter event is classified, in this order:

``mitosis``
    The daughters lose the mitotic marker. With PIP-FUCCI that is geminin,
    which the APC/C degrades at anaphase: a real division takes the daughters
    below ``marker_drop`` times the parent's late-G2 level, a split nucleus
    keeps it. A track born by mitosis less than ``refractory_hours`` earlier
    cannot divide again, which removes one mitosis called twice.
``fragment``
    The daughters touch and together hold the parent's area: two pieces of one
    nucleus. They are merged back into the parent.
``neighbour``
    Anything else that is not a mitosis: a separate cell was linked in as a
    daughter. The daughter that continues the parent's position is merged into
    it and the other becomes a founder.

Before each pass over the divisions, a short unparented track that touches
one nucleus in every frame of its life (``piece``) is folded into it: a
nucleus cut in two for a frame or two, typically around mitosis, whose piece
started its own track and so was never a daughter. When the piece was handed
the daughters of a nucleus split at metaphase, they pass to the host, and its
division is then judged like any other.

After a pass over the divisions, an unparented track that starts where exactly
one childless track ended in the previous frame is joined to it (``restart``):
Trackastra cannot express a merge, so two pieces fusing back into one nucleus
often end both tracks and open a new id.

Merging can expose a new event — a nucleus that splits at frame 10, is repaired
at 11 and splits again at 12 — so passes repeat until nothing changes. Every
decision is returned as a row of :attr:`RepairResult.events`, which is what a
curator should check first in Mastodon.

Without a marker the mitosis test is unavailable: fragments are still merged,
but no event is called a neighbour, because a real division and a mislinked
neighbour cannot be told apart from geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np
import pandas as pd

#: Columns :func:`repair_lineage` expects in its detections table.
REQUIRED_COLUMNS = (
    "track_id_raw",
    "parent_track_id_raw",
    "timepoint",
    "area",
    "y",
    "x",
)


@dataclass(frozen=True)
class RepairParams:
    """Thresholds for :func:`repair_lineage`.

    Attributes:
        interval_minutes: Time between frames.
        marker_drop: A division is mitotic if the daughters' marker falls below
            this fraction of the parent's.
        parent_window: Frames at the end of the parent over which its marker
            maximum is taken (geminin starts to fall during mitosis itself).
        daughter_window: Frames at the start of each daughter averaged for its
            marker level.
        refractory_hours: Minimum time from a mitotic birth to the next mitosis.
        touch: Daughters touch if their centroid distance is at most this
            multiple of the sum of their equivalent radii.
        area_lo: Lower bound on daughters' summed area over the parent's for a
            fragment.
        area_hi: Upper bound on the same ratio.
        restart_reach: A restart joins a track that ended within this multiple
            of the sum of their equivalent radii.
        piece_frames: An unparented track of at most this many frames that
            touches one nucleus throughout is a piece of it (``piece``).
        max_passes: Safety bound on repair passes.
    """

    interval_minutes: float = 20.0
    marker_drop: float = 0.5
    parent_window: int = 6
    daughter_window: int = 3
    refractory_hours: float = 8.0
    touch: float = 1.1
    area_lo: float = 0.8
    area_hi: float = 1.25
    restart_reach: float = 1.0
    piece_frames: int = 3
    max_passes: int = 50

    @property
    def refractory_frames(self) -> int:
        """The refractory period in frames."""
        return int(round(self.refractory_hours * 60 / self.interval_minutes))


@dataclass
class RepairResult:
    """Outcome of a lineage repair for one well.

    Attributes:
        assignment: Raw track id to repaired track id.
        parents: Repaired track id to its parent's repaired id (0 = founder).
        events: One row per decision, in the order taken.
        passes: Number of passes until no rule fired.
    """

    assignment: dict[int, int]
    parents: dict[int, int]
    events: pd.DataFrame
    passes: int


@dataclass
class _Track:
    """A repaired track: per-frame sums over its member raw tracks."""

    frames: dict[int, np.ndarray]
    parent: int = 0
    children: set[int] = field(default_factory=set)
    resolved: bool = False
    mitotic_birth: int | None = None

    @property
    def begin(self) -> int:
        return min(self.frames)

    @property
    def end(self) -> int:
        return max(self.frames)

    def at(self, t: int) -> tuple[float, float, float, float]:
        """Area, centroid y, centroid x and mean marker at frame ``t``."""
        area, ya, xa, ma = self.frames[t]
        return area, ya / area, xa / area, ma / area

    def marker(self, frames: list[int]) -> list[float]:
        return [self.at(t)[3] for t in frames]


def _radius(area: float) -> float:
    return float(np.sqrt(area / np.pi))


class _Lineage:
    """Mutable lineage graph with union-find over raw track ids."""

    def __init__(self, det: pd.DataFrame, has_marker: bool) -> None:
        self.has_marker = has_marker
        self.rep: dict[int, int] = {}
        self.tracks: dict[int, _Track] = {}
        marker = det["marker"].to_numpy() if has_marker else np.zeros(len(det))
        area = det["area"].to_numpy(dtype=float)
        stacked = np.column_stack(
            [
                area,
                area * det["y"].to_numpy(),
                area * det["x"].to_numpy(),
                area * marker,
            ]
        )
        tids = det["track_id_raw"].to_numpy()
        ts = det["timepoint"].to_numpy()
        for tid, t, row in zip(tids, ts, stacked, strict=True):
            tid, t = int(tid), int(t)
            track = self.tracks.setdefault(tid, _Track(frames={}))
            if t in track.frames:
                track.frames[t] = track.frames[t] + row
            else:
                track.frames[t] = row.copy()
            self.rep[tid] = tid
        parents = det.groupby("track_id_raw")["parent_track_id_raw"].max()
        # Raw track ids are integers; the stubs type the labels Hashable.
        for tid, pid in cast("Iterable[tuple[int, Any]]", parents.items()):
            pid = int(pid) if pd.notna(pid) else 0
            if pid and pid in self.tracks:
                self.tracks[int(tid)].parent = pid
                self.tracks[pid].children.add(int(tid))

    def find(self, tid: int) -> int:
        root = tid
        while self.rep[root] != root:
            root = self.rep[root]
        while self.rep[tid] != root:
            self.rep[tid], tid = root, self.rep[tid]
        return root

    def merge(self, keep: int, absorb: int) -> None:
        """Fold track ``absorb`` into ``keep``; ``keep`` adopts its children."""
        a, k = self.tracks.pop(absorb), self.tracks[keep]
        for t, row in a.frames.items():
            k.frames[t] = k.frames[t] + row if t in k.frames else row
        k.children.discard(absorb)
        for child in a.children:
            self.tracks[child].parent = keep
            k.children.add(child)
        if a.parent and a.parent != keep and a.parent in self.tracks:
            self.tracks[a.parent].children.discard(absorb)
        k.resolved = False
        self.rep[absorb] = keep

    def detach(self, tid: int) -> None:
        """Make ``tid`` a founder."""
        track = self.tracks[tid]
        if track.parent:
            self.tracks[track.parent].children.discard(tid)
        track.parent = 0
        track.mitotic_birth = None


def _division_event(
    lin: _Lineage, pid: int, params: RepairParams
) -> dict[str, float | int | str]:
    """Measure one division and decide its fate."""
    parent = lin.tracks[pid]
    kids = sorted(parent.children, key=lambda k: (lin.tracks[k].begin, k))
    split = min(lin.tracks[k].begin for k in kids)
    p_area, p_y, p_x, _ = parent.at(parent.end)

    firsts = [lin.tracks[k].at(lin.tracks[k].begin) for k in kids]
    area_ratio = sum(f[0] for f in firsts) / p_area

    a, b = lin.tracks[kids[0]], lin.tracks[kids[1]]
    co = max(a.begin, b.begin)
    if co in a.frames and co in b.frames:
        aa, ay, ax, _ = a.at(co)
        ba, by, bx, _ = b.at(co)
        touch = float(np.hypot(ay - by, ax - bx) / (_radius(aa) + _radius(ba)))
    else:
        touch = float("inf")

    marker_ratio = float("nan")
    refractory = (
        parent.mitotic_birth is not None
        and split - parent.mitotic_birth < params.refractory_frames
    )
    if lin.has_marker:
        p_frames = sorted(parent.frames)[-params.parent_window :]
        p_level = max(parent.marker(p_frames))
        d_levels = [
            float(
                np.mean(
                    lin.tracks[k].marker(
                        sorted(lin.tracks[k].frames)[: params.daughter_window]
                    )
                )
            )
            for k in kids
        ]
        marker_ratio = (
            float(np.mean(d_levels) / p_level) if p_level > 0 else float("nan")
        )

    is_fragment = (
        touch <= params.touch
        and params.area_lo <= area_ratio <= params.area_hi
    )
    if lin.has_marker:
        mitotic = marker_ratio < params.marker_drop and not refractory
        rule = (
            "mitosis"
            if mitotic
            else ("fragment" if is_fragment else "neighbour")
        )
    else:
        rule = "fragment" if is_fragment else "kept"

    return {
        "parent": pid,
        "children": " ".join(str(k) for k in kids),
        "frame": split,
        "rule": rule,
        "marker_ratio": marker_ratio,
        "area_ratio": area_ratio,
        "touch": touch,
        "refractory": refractory,
        "y": p_y,
        "x": p_x,
    }


def _continuing_child(lin: _Lineage, pid: int) -> int:
    """The daughter whose first centroid is closest to the parent's last."""
    parent = lin.tracks[pid]
    _, py, px, _ = parent.at(parent.end)

    def dist(k: int) -> tuple[float, int]:
        t = lin.tracks[k]
        _, y, x, _ = t.at(t.begin)
        return (float(np.hypot(y - py, x - px)), k)

    return min(parent.children, key=dist)


def _resolve_divisions(
    lin: _Lineage,
    params: RepairParams,
    pass_no: int,
    log: list[dict[str, Any]],
) -> bool:
    changed = False
    # Earliest division first: a daughter's own division can only be judged
    # once its birth is known, because the refractory test depends on it.
    for pid in sorted(lin.tracks, key=lambda k: (lin.tracks[k].end, k)):
        parent = lin.tracks.get(pid)
        if parent is None or parent.resolved or not parent.children:
            continue
        if len(parent.children) == 1:
            (only,) = parent.children
            log.append(
                {
                    "pass": pass_no,
                    "parent": pid,
                    "children": str(only),
                    "frame": lin.tracks[only].begin,
                    "rule": "continuation",
                }
            )
            lin.merge(pid, only)
            changed = True
            continue

        event = _division_event(lin, pid, params)
        event["pass"] = pass_no
        log.append(event)
        rule = event["rule"]
        if rule in ("mitosis", "kept"):
            parent.resolved = True
            if rule == "mitosis":
                for k in parent.children:
                    kid = lin.tracks[k]
                    if kid.mitotic_birth != int(event["frame"]):
                        kid.mitotic_birth = int(event["frame"])
                        kid.resolved = False
                        changed = True
            continue

        changed = True
        if rule == "fragment":
            for k in sorted(parent.children):
                lin.merge(pid, k)
        else:  # neighbour
            keep = _continuing_child(lin, pid)
            for k in sorted(parent.children - {keep}):
                lin.detach(k)
            lin.merge(pid, keep)
    return changed


def _join_restarts(
    lin: _Lineage,
    params: RepairParams,
    pass_no: int,
    log: list[dict[str, Any]],
) -> bool:
    ends: dict[int, list[int]] = {}
    for tid, t in lin.tracks.items():
        if not t.children:
            ends.setdefault(t.end, []).append(tid)

    changed = False
    for tid in sorted(lin.tracks):
        track = lin.tracks.get(tid)
        if track is None or track.parent or track.begin == 0:
            continue
        qa, qy, qx, _ = track.at(track.begin)
        hits = []
        for cand in ends.get(track.begin - 1, []):
            if cand == tid or cand not in lin.tracks:
                continue
            c = lin.tracks[cand]
            ca, cy, cx, _ = c.at(c.end)
            reach = params.restart_reach * (_radius(qa) + _radius(ca))
            if np.hypot(qy - cy, qx - cx) <= reach:
                hits.append(cand)
        if len(hits) == 1:
            log.append(
                {
                    "pass": pass_no,
                    "parent": hits[0],
                    "children": str(tid),
                    "frame": track.begin,
                    "rule": "restart",
                    "y": qy,
                    "x": qx,
                }
            )
            lin.merge(hits[0], tid)
            ends[track.begin - 1].remove(hits[0])
            changed = True
    return changed


def _host(
    lin: _Lineage,
    tid: int,
    params: RepairParams,
    by_frame: dict[int, list[int]],
) -> int | None:
    """The one track a short piece touches in every frame of its life."""
    piece = lin.tracks[tid]
    host: int | None = None
    for t in sorted(piece.frames):
        pa, py, px, _ = piece.at(t)
        touching = [
            k
            for k in by_frame.get(t, [])
            if k != tid
            and (other := lin.tracks.get(k)) is not None
            and t in other.frames
            and np.hypot(other.at(t)[1] - py, other.at(t)[2] - px)
            <= params.touch * (_radius(pa) + _radius(other.at(t)[0]))
        ]
        if len(touching) != 1 or (host is not None and touching[0] != host):
            return None
        host = touching[0]
    return host


def _absorb_pieces(
    lin: _Lineage,
    params: RepairParams,
    pass_no: int,
    log: list[dict[str, Any]],
) -> bool:
    """Fold short pieces of a nucleus back into it.

    Cellpose often cuts a nucleus in two for a frame or two, most often just
    before or after mitosis. The cut-off piece starts its own track, so it is
    never a daughter and the fragment test never sees it. A piece is an
    unparented track of at most ``piece_frames`` frames touching exactly one
    nucleus (the host) in every frame. It is merged if together they hold
    the host's area from just before or just after the cut, or if the host ends
    with the piece and the piece carries the children: a mitotic nucleus
    split at metaphase whose half was handed the daughters. The geminin test
    then judges that division on the host.
    """
    by_frame: dict[int, list[int]] = {}
    for k, track in lin.tracks.items():
        for t in track.frames:
            by_frame.setdefault(t, []).append(k)
    changed = False
    for tid in sorted(lin.tracks):
        piece = lin.tracks.get(tid)
        if (
            piece is None
            or piece.parent
            or len(piece.frames) > params.piece_frames
        ):
            continue
        host_id = _host(lin, tid, params, by_frame)
        if host_id is None:
            continue
        host = lin.tracks[host_id]
        handover = (
            bool(piece.children)
            and not host.children
            and host.end == piece.end
        )
        # Daughter nuclei grow fast after mitosis, so the area just before
        # the cut can be far below the joint area; either side may vouch.
        joint = np.mean([piece.at(t)[0] + host.at(t)[0] for t in piece.frames])
        ratios = [
            float(joint / host.at(t)[0])
            for t in (piece.begin - 1, piece.end + 1)
            if t in host.frames
        ]
        fits = [r for r in ratios if params.area_lo <= r <= params.area_hi]
        ratio = fits[0] if fits else (ratios[0] if ratios else float("nan"))
        if not (handover or fits):
            continue
        _, y, x, _ = piece.at(piece.begin)
        log.append(
            {
                "pass": pass_no,
                "parent": host_id,
                "children": str(tid),
                "frame": piece.begin,
                "rule": "piece",
                "area_ratio": ratio,
                "y": y,
                "x": x,
            }
        )
        host.resolved = False
        lin.merge(host_id, tid)
        changed = True
    return changed


def repair_lineage(
    det: pd.DataFrame, params: RepairParams | None = None
) -> RepairResult:
    """Repair one well's lineage.

    Args:
        det: One row per detection with :data:`REQUIRED_COLUMNS` and, if a
            mitotic marker is used, a background-subtracted ``marker`` column.
        params: Thresholds; defaults to :class:`RepairParams`.

    Returns:
        The raw-to-repaired assignment, the repaired parent map and the event log.

    Raises:
        ValueError: If a required column is missing.
    """
    params = params or RepairParams()
    missing = [c for c in REQUIRED_COLUMNS if c not in det.columns]
    if missing:
        raise ValueError(
            f"Detections table lacks column(s): {', '.join(missing)}"
        )

    lin = _Lineage(
        det.dropna(subset=["track_id_raw"]), has_marker="marker" in det.columns
    )
    log: list[dict[str, Any]] = []
    passes = 0
    for passes in range(1, params.max_passes + 1):
        changed = _absorb_pieces(lin, params, passes, log)
        changed |= _resolve_divisions(lin, params, passes, log)
        changed |= _join_restarts(lin, params, passes, log)
        if not changed:
            break

    assignment = {tid: lin.find(tid) for tid in lin.rep}
    parents = {tid: t.parent for tid, t in lin.tracks.items()}
    return RepairResult(
        assignment=assignment,
        parents=parents,
        events=pd.DataFrame(log),
        passes=passes,
    )


def apply_repair(
    det: pd.DataFrame, result: RepairResult, value_cols: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Collapse detections onto repaired tracks, one row per track and frame.

    Pieces of one nucleus that share a repaired track in the same frame are
    combined as the nucleus they came from: areas add, centroids and every
    column in ``value_cols`` (mean intensities) are area-weighted.

    Args:
        det: Detections with :data:`REQUIRED_COLUMNS` and ``value_cols``.
        result: Output of :func:`repair_lineage` for the same detections.
        value_cols: Per-detection mean quantities to carry through.

    Returns:
        Columns ``track_id``, ``parent_track_id``, ``timepoint``, ``area``,
        ``y``, ``x``, ``n_pieces`` and ``value_cols``.
    """
    d = det.dropna(subset=["track_id_raw"]).copy()
    d["track_id"] = d["track_id_raw"].astype(int).map(result.assignment)
    weighted = ["y", "x", *value_cols]
    for col in weighted:
        d[col] = d[col] * d["area"]
    out = d.groupby(["track_id", "timepoint"], as_index=False).agg(
        area=("area", "sum"),
        n_pieces=("area", "size"),
        **{col: (col, "sum") for col in weighted},
    )
    for col in weighted:
        out[col] = out[col] / out["area"]
    out["parent_track_id"] = (
        out["track_id"].map(result.parents).fillna(0).astype(int)
    )
    return out
