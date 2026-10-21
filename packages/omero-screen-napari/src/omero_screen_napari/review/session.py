"""A review session: queue, curated lineage, edit log and verdicts, without Qt.

The Track Review widget and the agent's MCP tools both drive the same
:class:`ReviewSession`, so a link made by the reviewer or proposed by the agent
goes through one code path and one edit log.

Files, all next to ``queue.json``:

* ``edits.jsonl``: lineage edits (link, unlink, events, outcome, exclude,
  note); the curated data is rebuilt from it;
* ``decisions.json``: one verdict per reviewed item (accept / correct /
  reject / unsure) with the reviewer's outcome and note; this is the review
  record, not data;
* ``proposals.json``: edits the agent proposed and their status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from omero_screen_napari.review.cells import CellSource, parse_anchor
from omero_screen_napari.review_queue import (
    Decision,
    Queue,
    ReviewItem,
    decisions_path,
    latest_decisions,
    read_queue,
    record_decision,
)

#: The session the open widget is using; agent tools act on it.
_ACTIVE: ReviewSession | None = None
#: The open widget, so agent tools can move the viewer through it.
_ACTIVE_UI: Any = None


def active_session() -> ReviewSession | None:
    """The review session of the open Track Review widget, if any."""
    return _ACTIVE


def set_active_session(session: ReviewSession | None, ui: Any = None) -> None:
    """Register the session (and the widget showing it) agent tools act on."""
    global _ACTIVE, _ACTIVE_UI
    _ACTIVE = session
    _ACTIVE_UI = ui


def active_ui() -> Any:
    """The open Track Review widget, if any."""
    return _ACTIVE_UI


@dataclass
class Proposal:
    """An edit the agent suggests; applied only when the reviewer confirms."""

    id: str
    item: str
    op: str
    args: dict[str, Any]
    reason: str
    confidence: str = ""  # clear | uncertain
    status: str = (
        "pending"  # pending | confirmed | rejected | applied (agent run)
    )
    note: str = ""
    time: str = ""


@dataclass
class ReviewSession:
    """State of one review: the queue and everything needed to act on it."""

    queue_path: Path
    db_path: Path | None = None
    queue: Queue = field(init=False)
    source: CellSource = field(init=False)
    #: Cells opened by search that are not in the queue, by item id.
    adhoc: dict[str, ReviewItem] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.queue_path = Path(self.queue_path)
        self.queue = read_queue(self.queue_path)
        self.source = CellSource(
            self.queue.plate_id, db_path=self.db_path, log_path=self.edits_path
        )

    # -- files -------------------------------------------------------------

    @property
    def edits_path(self) -> Path:
        return self.queue_path.with_name("edits.jsonl")

    @property
    def proposals_path(self) -> Path:
        return self.queue_path.with_name("proposals.json")

    @property
    def mode(self) -> str:
        """``human`` (agent proposes, reviewer confirms) or ``agent`` (agent works alone)."""
        return str(
            json.loads(self.queue_path.read_text()).get("mode", "human")
        )

    def reload_queue(self) -> None:
        """Re-read the queue file (the agent may have rewritten it)."""
        self.queue = read_queue(self.queue_path)

    # -- items -------------------------------------------------------------

    def item(self, item_id: str) -> ReviewItem:
        found = self.queue.get(item_id) or self.adhoc.get(item_id)
        if found is None:
            raise KeyError(f"No item {item_id!r} in the queue.")
        return found

    def find(self, well: str, frame: int, ident: int) -> ReviewItem:
        """The cell that is ``ident`` in ``frame``: a queue item, or a new ad-hoc one.

        ``ident`` is the nucleus label shown in the viewer at ``frame``, or
        else a curated track id present in that frame. A cell already in the
        queue (same curated track) returns its queue item, so a search never
        duplicates one.

        Raises:
            KeyError: If no nucleus in ``frame`` matches ``ident``.
        """
        _, curated = self.source.well(well)
        if (frame, ident) in curated.tracks:
            label = ident
        else:
            hits = [
                lab
                for (t, lab), tid in curated.tracks.items()
                if t == frame and tid == ident
            ]
            if not hits:
                raise KeyError(
                    f"No nucleus or track {ident} in frame {frame} of {well}."
                )
            label = min(hits)
        tid = curated.tracks[(frame, label)]
        for it in self.queue.items:
            if it.well != well:
                continue
            try:
                if curated.resolve(self.anchor(it.id)) == tid:
                    return it
            except Exception:  # an anchor an edit has removed
                continue
        item_id = f"{well}-t{frame}-L{label}"
        if item_id not in self.adhoc:
            self.adhoc[item_id] = ReviewItem(
                id=item_id,
                well=well,
                frame=frame,
                reason="search",
                question="Opened by search; not in the queue.",
            )
        return self.adhoc[item_id]

    def anchor(self, item_id: str) -> tuple[int, int]:
        return parse_anchor(item_id)

    def stop(self) -> int:
        spec = json.loads(self.queue_path.read_text()).get("spec", {})
        return int(spec.get("stop", 10**6))

    def status(self) -> dict[str, Any]:
        done = latest_decisions(decisions_path(self.queue_path))
        per_well: dict[str, list[int]] = {}
        for it in self.queue.items:
            per_well.setdefault(it.well, [0, 0])
            per_well[it.well][0] += 1
            per_well[it.well][1] += it.id in done
        return {
            "plate_id": self.queue.plate_id,
            "items": len(self.queue.items),
            "reviewed": sum(i.id in done for i in self.queue.items),
            "per_well": {
                w: {"queued": q, "reviewed": r}
                for w, (q, r) in per_well.items()
            },
            "pending_proposals": sum(
                p.status == "pending" for p in self.proposals()
            ),
        }

    # -- curated data --------------------------------------------------------

    def refresh(self, well: str) -> None:
        """Forget the cached lineage of ``well`` so the next read replays the log."""
        self.source._wells.pop(well, None)

    def path(self, item_id: str) -> pd.DataFrame:
        """Curated per-frame path of the item's cell from its anchor frame on.

        An ad-hoc (searched) cell is shown over its whole track.
        """
        it = self.item(item_id)
        anchor = self.anchor(item_id)
        start = None if item_id in self.adhoc else anchor[0]
        return self.source.path(it.well, anchor, start=start)

    def lineage(self, item_id: str) -> list[dict[str, Any]]:
        """The cell's track with its mother's and daughters' tracks.

        Returns:
            One dict per track: ``track_id``, ``parent`` (0 for none),
            ``role`` (``cell`` / ``mother`` / ``daughter``) and ``frames``,
            a frame with ``timepoint, y, x, gap`` (gaps interpolated).
        """
        from cellview.tracks.cells import cell_frames

        it = self.item(item_id)
        det, curated = self.source.well(it.well)
        tid = curated.resolve(self.anchor(item_id))
        mother = curated.parents.get(tid, 0)
        tracks = [(tid, "cell")]
        if mother:
            tracks.append((mother, "mother"))
        tracks += [(c, "daughter") for c in curated.children(tid)]
        out = []
        for track, role in tracks:
            keys = curated.detections(track)
            if not keys:
                continue
            frames = cell_frames(curated, det, keys[0])
            out.append(
                {
                    "track_id": track,
                    "parent": curated.parents.get(track, 0),
                    "role": role,
                    "frames": frames[["timepoint", "y", "x", "gap"]].dropna(
                        subset=["y", "x"]
                    ),
                }
            )
        return out

    def preview(
        self, item_id: str, op: str, args: dict[str, Any]
    ) -> tuple[list[str], pd.DataFrame]:
        """What an edit would change, without writing it.

        Returns:
            ``(summary lines, the cell's path after the edit)``.

        Raises:
            cellview.tracks.edit.EditError: If the edit does not apply.
        """
        import copy

        from cellview.tracks.cells import cell_frames
        from cellview.tracks.edit import ANNOTATION_OPS, Edit, apply_edit

        it = self.item(item_id)
        anchor = self.anchor(item_id)
        det, curated = self.source.well(it.well)
        full = {"cell": list(anchor), **args}
        after = copy.deepcopy(curated)
        apply_edit(after, Edit(id="preview", op=op, well=it.well, args=full))
        new_path = cell_frames(after, det, anchor)
        if op in ANNOTATION_OPS:
            detail = ", ".join(f"{k} {v}" for k, v in args.items())
            return [f"{op}: {detail}"], new_path
        old_tid, new_tid = curated.resolve(anchor), after.resolve(anchor)
        before = set(curated.detections(old_tid))
        now = set(after.detections(new_tid))
        lines: list[str] = []
        gained: dict[int, list[int]] = {}
        for t, lab in sorted(now - before):
            gained.setdefault(curated.tracks.get((t, lab), 0), []).append(t)
        for src, frames in gained.items():
            origin = f"from track {src}" if src else "(new nucleus)"
            lines.append(f"gains {_span(frames)} {origin}")
        lost: dict[int, list[int]] = {}
        for t, lab in sorted(before - now):
            lost.setdefault(after.tracks.get((t, lab), 0), []).append(t)
        for dst, frames in lost.items():
            target = (
                "(removed)"
                if not dst
                else f"to track {dst}"
                if dst in curated.parents
                else "to a new track"
            )
            lines.append(f"loses {_span(frames)} {target}")
        kids_before = set(curated.children(old_tid))
        kids_now = set(after.children(new_tid))
        if kids_now != kids_before:
            lines.append(
                f"daughters {sorted(kids_before) or 'none'} → {sorted(kids_now) or 'none'}"
            )
        if curated.parents.get(old_tid, 0) != after.parents.get(new_tid, 0):
            lines.append(
                f"mother {curated.parents.get(old_tid, 0) or 'none'} → "
                f"{after.parents.get(new_tid, 0) or 'none'}"
            )
        return lines or ["no change to this cell's track"], new_path

    def breaks(self, item_id: str) -> list[int]:
        from cellview.tracks.cells import track_breaks

        it = self.item(item_id)
        _, curated = self.source.well(it.well)
        return track_breaks(curated, self.anchor(item_id), self.stop())

    def candidates(self, item_id: str, frame: int) -> pd.DataFrame:
        from cellview.tracks.cells import continuation_candidates

        it = self.item(item_id)
        det, curated = self.source.well(it.well)
        return continuation_candidates(
            curated, det, self.anchor(item_id), frame
        )

    def label_at(
        self, well: str, frame: int, y: float, x: float
    ) -> dict[str, Any]:
        """Nucleus whose centroid is nearest to ``(y, x)`` in ``frame``, with its curated track."""
        det, curated = self.source.well(well)
        rows = det[det["timepoint"] == frame]
        if rows.empty:
            return {}
        d = ((rows["y"] - y) ** 2 + (rows["x"] - x) ** 2) ** 0.5
        row = rows.loc[d.idxmin()]
        out = {
            k: (float(v) if isinstance(v, float) else v)
            for k, v in row.items()
            if k != "measurement_id"
        }
        out["track_id"] = curated.tracks.get(
            (int(frame), int(row["label"])), 0
        )
        out["distance"] = float(d.min())
        return out

    # -- edits ---------------------------------------------------------------

    def edit(
        self,
        item_id: str,
        op: str,
        args: dict[str, Any],
        author: str = "human",
        confirmed_by: str | None = None,
        reason: str = "",
    ) -> Any:
        """Validate and append an edit for the item's cell; replay on next read.

        ``args`` may omit ``cell``; it defaults to the item's anchor.
        """
        from cellview.tracks.edit import EditLog

        it = self.item(item_id)
        full = {"cell": list(self.anchor(item_id)), **args}
        base = self.source.base(it.well)
        entry = EditLog(self.edits_path).append(
            op,
            it.well,
            full,
            author=author,
            confirmed_by=confirmed_by,
            reason=reason,
            base=base,
        )
        self.refresh(it.well)
        return entry

    def edit_many(
        self,
        item_id: str,
        items: list[tuple[str, dict[str, Any], str]],
        author: str = "human",
    ) -> list[Any]:
        """Validate and append several edits for the item's cell at once.

        ``items`` are ``(op, args, reason)``; ``args`` may omit ``cell``
        (the item's anchor) but an explicit ``cell: None`` is kept.
        """
        from cellview.tracks.edit import EditLog

        it = self.item(item_id)
        anchor = list(self.anchor(item_id))
        full = [
            (op, it.well, {"cell": anchor, **args}, reason)
            for op, args, reason in items
        ]
        entries = EditLog(self.edits_path).extend(
            full, author=author, base=self.source.base(it.well)
        )
        self.refresh(it.well)
        return entries

    def undo(self, item_id: str) -> Any:
        from cellview.tracks.edit import EditLog

        it = self.item(item_id)
        entry = EditLog(self.edits_path).undo(
            it.well, base=self.source.base(it.well)
        )
        self.refresh(it.well)
        return entry

    def entries(self, item_id: str) -> list[Any]:
        """Edit-log entries that name this item's cell."""
        from cellview.tracks.edit import EditLog

        anchor = list(self.anchor(item_id))
        it = self.item(item_id)
        return [
            e
            for e in EditLog(self.edits_path)
            if e.well == it.well and e.args.get("cell") == anchor
        ]

    # -- verdicts ------------------------------------------------------------

    def verdict(
        self,
        item_id: str,
        verdict: str,
        outcome: str = "",
        note: str = "",
        frames: list[int] | None = None,
        author: str = "human",
    ) -> Decision:
        """Record a verdict on an item (``author`` is ``agent`` only in an agent run)."""
        return record_decision(
            decisions_path(self.queue_path),
            Decision(
                id=item_id,
                verdict=verdict,
                outcome=outcome,
                note=note,
                frames=frames or [],
                author=author,
            ),
        )

    # -- agent proposals -------------------------------------------------------

    def proposals(self) -> list[Proposal]:
        if not self.proposals_path.exists():
            return []
        return [
            Proposal(**p) for p in json.loads(self.proposals_path.read_text())
        ]

    def _save_proposals(self, props: list[Proposal]) -> None:
        self.proposals_path.write_text(
            json.dumps([p.__dict__ for p in props], indent=1)
        )

    def propose(
        self,
        item_id: str,
        op: str,
        args: dict[str, Any],
        reason: str,
        confidence: str = "",
    ) -> Proposal:
        """Record an agent proposal.

        In a ``human`` run it changes nothing until the reviewer confirms it.
        In an ``agent`` run it is applied at once as an unconfirmed agent edit
        (status ``applied``).
        """
        self.item(item_id)
        props = self.proposals()
        prop = Proposal(
            id=f"p{len(props) + 1:04d}",
            item=item_id,
            op=op,
            args=args,
            reason=reason,
            confidence=confidence,
            time=datetime.now().isoformat(timespec="seconds"),
        )
        if self.mode == "agent":
            self.edit(
                item_id,
                op,
                args,
                author="agent",
                confirmed_by=None,
                reason=reason,
            )
            prop.status = "applied"
        self._save_proposals([*props, prop])
        return prop

    def resolve_proposal(
        self, proposal_id: str, confirm: bool, note: str = ""
    ) -> Proposal:
        """Confirm (apply as an agent edit confirmed by the human) or reject a proposal."""
        props = self.proposals()
        prop = next((p for p in props if p.id == proposal_id), None)
        if prop is None:
            raise KeyError(f"No proposal {proposal_id!r}.")
        if prop.status != "pending":
            raise ValueError(
                f"Proposal {proposal_id} is already {prop.status}."
            )
        if confirm:
            self.edit(
                prop.item,
                prop.op,
                prop.args,
                author="agent",
                confirmed_by="human",
                reason=prop.reason,
            )
        prop.status, prop.note = ("confirmed" if confirm else "rejected"), note
        self._save_proposals(props)
        return prop


def _span(frames: list[int]) -> str:
    """``t85``, ``t85–90`` or ``t85–90, t93`` for a sorted list of frames."""
    runs: list[list[int]] = []
    for t in frames:
        if runs and t == runs[-1][1] + 1:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return ", ".join(f"t{a}" if a == b else f"t{a}–{b}" for a, b in runs)
