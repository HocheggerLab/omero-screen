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


def active_session() -> ReviewSession | None:
    """The review session of the open Track Review widget, if any."""
    return _ACTIVE


def set_active_session(session: ReviewSession | None) -> None:
    """Register the session agent tools should act on."""
    global _ACTIVE
    _ACTIVE = session


@dataclass
class Proposal:
    """An edit the agent suggests; applied only when the reviewer confirms."""

    id: str
    item: str
    op: str
    args: dict[str, Any]
    reason: str
    status: str = "pending"  # pending | confirmed | rejected
    note: str = ""
    time: str = ""


@dataclass
class ReviewSession:
    """State of one review: the queue and everything needed to act on it."""

    queue_path: Path
    db_path: Path | None = None
    queue: Queue = field(init=False)
    source: CellSource = field(init=False)

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

    def reload_queue(self) -> None:
        """Re-read the queue file (the agent may have rewritten it)."""
        self.queue = read_queue(self.queue_path)

    # -- items -------------------------------------------------------------

    def item(self, item_id: str) -> ReviewItem:
        found = self.queue.get(item_id)
        if found is None:
            raise KeyError(f"No item {item_id!r} in the queue.")
        return found

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
        """Curated per-frame path of the item's cell from its anchor frame on."""
        it = self.item(item_id)
        anchor = self.anchor(item_id)
        return self.source.path(it.well, anchor, start=anchor[0])

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
    ) -> Decision:
        return record_decision(
            decisions_path(self.queue_path),
            Decision(
                id=item_id,
                verdict=verdict,
                outcome=outcome,
                note=note,
                frames=frames or [],
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
        self, item_id: str, op: str, args: dict[str, Any], reason: str
    ) -> Proposal:
        """Record an agent proposal; it changes nothing until confirmed."""
        self.item(item_id)
        props = self.proposals()
        prop = Proposal(
            id=f"p{len(props) + 1:04d}",
            item=item_id,
            op=op,
            args=args,
            reason=reason,
            time=datetime.now().isoformat(timespec="seconds"),
        )
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
