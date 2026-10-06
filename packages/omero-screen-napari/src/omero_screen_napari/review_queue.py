"""File-based review queue shared by an analysis agent and a human in napari.

The agent writes a *queue* of cells to inspect; the Track review widget
(:mod:`omero_screen_napari._review_widget`) watches the file, jumps to each
cell and records the reviewer's verdict in a *decisions* file next to it. No
server is involved: either side can run without the other, and the decisions
file is the curation record.

Queue (``queue.json``)::

    {
      "version": 1,
      "plate_id": 5054,
      "focus": "C2-152",              # optional: jump here as soon as it changes
      "outcomes": ["divided", ...],   # optional: choices offered to the reviewer
      "items": [
        {
          "id": "C2-152",
          "well": "C2",
          "frame": 161,               # frame to open at
          "reason": "lost",
          "question": "Where does the nucleus go after t161?",
          "outcome": "lost",          # the automatic call, if any
          "path": [[t, y, x, label], ...]   # pixel coordinates; label 0 = no mask
        }
      ]
    }

Decisions (``decisions.json``, written beside the queue) are append-only; the
latest entry per item wins::

    {"version": 1, "decisions": [
        {"id": "C2-152", "verdict": "reject", "outcome": "death",
         "frames": [163], "note": "...", "time": "2026-10-06T16:02:11",
         "links": [{"frame": 156, "label": 3132}]}

``links`` are continuations the reviewer picked by clicking a nucleus: from
``frame`` on, the cell is the nucleus carrying raw mask ``label``.
    ]}
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

QUEUE_VERSION = 1

#: Outcomes offered when the queue does not name its own.
DEFAULT_OUTCOMES = (
    "divided",
    "mitotic_exit_no_division",
    "mitotic_arrest",
    "death",
    "no_mitosis",
    "lost",
    "not_a_cell",
    "no_reporter",
)

#: Verdicts a reviewer can give.
VERDICTS = ("accept", "correct", "reject", "unsure")


@dataclass(frozen=True)
class PathPoint:
    """Where the cell is (or is expected to be) in one frame, in pixels."""

    t: int
    y: float
    x: float
    label: int = 0


@dataclass(frozen=True)
class ReviewItem:
    """One cell to inspect."""

    id: str
    well: str
    frame: int
    reason: str = ""
    question: str = ""
    outcome: str = ""
    path: tuple[PathPoint, ...] = ()

    def point_at(self, t: int) -> PathPoint | None:
        """The path point for frame ``t``, or the nearest earlier one."""
        best = None
        for p in self.path:
            if p.t <= t and (best is None or p.t > best.t):
                best = p
        return best or (self.path[0] if self.path else None)


@dataclass(frozen=True)
class Queue:
    """A parsed review queue."""

    plate_id: int
    items: tuple[ReviewItem, ...]
    focus: str | None = None
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES

    def get(self, item_id: str) -> ReviewItem | None:
        """Look an item up by id."""
        return next((i for i in self.items if i.id == item_id), None)


@dataclass
class Decision:
    """A reviewer's verdict on one item."""

    id: str
    verdict: str
    outcome: str = ""
    frames: list[int] = field(default_factory=list)
    note: str = ""
    time: str = ""
    links: list[dict[str, int]] = field(default_factory=list)


class QueueError(ValueError):
    """Raised when a queue file is malformed."""


def _item_from_dict(raw: dict[str, Any]) -> ReviewItem:
    try:
        path = tuple(
            PathPoint(
                int(p[0]),
                float(p[1]),
                float(p[2]),
                int(p[3]) if len(p) > 3 else 0,
            )
            for p in raw.get("path", [])
        )
        return ReviewItem(
            id=str(raw["id"]),
            well=str(raw["well"]),
            frame=int(raw["frame"]),
            reason=str(raw.get("reason", "")),
            question=str(raw.get("question", "")),
            outcome=str(raw.get("outcome", "")),
            path=path,
        )
    except (KeyError, TypeError, ValueError, IndexError) as err:
        raise QueueError(f"Malformed queue item {raw!r}: {err}") from err


def read_queue(path: Path) -> Queue:
    """Parse a queue file.

    Raises:
        QueueError: If the file is not a valid queue.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise QueueError(f"Cannot read queue {path}: {err}") from err
    if (
        not isinstance(data, dict)
        or "items" not in data
        or "plate_id" not in data
    ):
        raise QueueError(
            f"{path} is not a review queue (needs 'plate_id' and 'items')."
        )
    items = tuple(_item_from_dict(i) for i in data["items"])
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise QueueError(f"{path} has duplicate item ids.")
    return Queue(
        plate_id=int(data["plate_id"]),
        items=items,
        focus=data.get("focus"),
        outcomes=tuple(data.get("outcomes") or DEFAULT_OUTCOMES),
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON so a watcher never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.chmod(
        tmp, 0o644
    )  # mkstemp creates 0600; keep the file readable/trackable
    os.replace(tmp, path)


def write_queue(
    path: Path,
    plate_id: int,
    items: list[ReviewItem],
    focus: str | None = None,
    outcomes: tuple[str, ...] | None = None,
) -> None:
    """Write a queue file atomically (the agent's side)."""
    payload: dict[str, Any] = {
        "version": QUEUE_VERSION,
        "plate_id": plate_id,
        "items": [
            {
                **{k: v for k, v in asdict(i).items() if k != "path"},
                "path": [[p.t, p.y, p.x, p.label] for p in i.path],
            }
            for i in items
        ],
    }
    if focus:
        payload["focus"] = focus
    if outcomes:
        payload["outcomes"] = list(outcomes)
    _atomic_write(path, payload)


def decisions_path(queue_path: Path) -> Path:
    """Where decisions for ``queue_path`` are stored."""
    return Path(queue_path).with_name("decisions.json")


def read_decisions(path: Path) -> list[Decision]:
    """All recorded decisions, oldest first; empty if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [Decision(**d) for d in data.get("decisions", [])]


def latest_decisions(path: Path) -> dict[str, Decision]:
    """The most recent decision per item id."""
    return {d.id: d for d in read_decisions(path)}


def record_decision(path: Path, decision: Decision) -> Decision:
    """Append a decision (the reviewer's side), stamping the time.

    Raises:
        ValueError: If the verdict is not one of :data:`VERDICTS`.
    """
    if decision.verdict not in VERDICTS:
        raise ValueError(
            f"Unknown verdict {decision.verdict!r}; use one of {VERDICTS}."
        )
    if not decision.time:
        decision.time = datetime.now().isoformat(timespec="seconds")
    existing = read_decisions(path)
    existing.append(decision)
    _atomic_write(
        Path(path),
        {"version": QUEUE_VERSION, "decisions": [asdict(d) for d in existing]},
    )
    return decision


def drafts_path(queue_path: Path) -> Path:
    """Where unfinished work for ``queue_path`` is kept."""
    return Path(queue_path).with_name("drafts.json")


def read_drafts(path: Path) -> dict[str, dict[str, Any]]:
    """Unfinished review state per item id; empty if none."""
    path = Path(path)
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text()).get("drafts", {}))


def save_draft(path: Path, item_id: str, draft: dict[str, Any] | None) -> None:
    """Store (or, with ``None``, drop) the unfinished state for one item.

    Drafts hold marked frames, picked continuations, the note and the chosen
    outcome before a verdict is given, so closing napari loses nothing.
    """
    drafts = read_drafts(path)
    if draft:
        drafts[item_id] = draft
    else:
        drafts.pop(item_id, None)
    _atomic_write(Path(path), {"version": QUEUE_VERSION, "drafts": drafts})
