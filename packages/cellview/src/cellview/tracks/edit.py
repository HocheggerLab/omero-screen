"""Replayable curation of tracked lineages.

Every correction a reviewer or an agent makes is one entry in an append-only
edit log (JSON lines). Curated tracks are never edited in place: they are
rebuilt by applying the log, in order, to the automatically repaired lineage.
Raw data is never touched, the log is a complete audit trail, and replaying it
on the same input always gives the same result.

**Anchors.** Edits name cells by an *anchor*, ``(frame, label)``: the cell
that is raw mask label ``label`` in frame ``frame``. Raw labels are the
tracker's ids, which the cached masks carry as pixel values, so anchors stay
valid when the automatic repair is re-run with different settings. Review
items use the same form (``C2-t72-L524`` is anchor ``(72, 524)`` in well C2).

**Operations** on the lineage:

``link``        from ``frame`` on, the anchored cell is the nucleus with raw
                ``label`` (re-joins a broken track; works across gaps)
``unlink``      cut the anchored cell's track at ``frame``; the rest becomes a
                new founder
``set_parent``  make the anchored cell a daughter of ``parent``
``clear_parent`` make the anchored cell a founder
``swap``        exchange the identities of two cells from ``frame`` on

**Detection operations** (which nuclei belong to the cell in a frame range):

``absorb``      add the nucleus with raw ``label`` to the cell in frames
                ``frames = [t0, t1]`` (a piece the repair missed; measurements
                combine exactly, area-weighted)
``drop``        remove the nucleus with raw ``label`` from whatever track holds
                it in ``frames`` (debris, a neighbour's piece)
``mask_add``    a new nucleus mask drawn or split by the reviewer: ``frame``,
                new ``label``, the ``patch`` file holding its pixels, its
                ``measurements`` (taken from the image when it was committed)
                and the raw labels it ``replaces`` in that frame. It joins the
                anchored cell if ``cell`` is given, else starts a new track.

**Annotations** (no change to the lineage): ``event`` (``mitosis``,
``death``, ``slippage`` at a frame), ``set_outcome``, ``exclude``, ``note``.

**Undo** appends ``revert`` naming an earlier entry; replay skips reverted
entries. Nothing is ever deleted from the log.

Each entry records ``author`` (``human`` / ``agent``), ``confirmed_by``,
``reason`` and ``time``, so agent proposals and human confirmation can be
counted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cellview.tracks.repair import RepairResult

LOG_VERSION = 1
TRACK_OPS = (
    "link",
    "unlink",
    "set_parent",
    "clear_parent",
    "swap",
    "absorb",
    "drop",
    "mask_add",
)
ANNOTATION_OPS = ("event", "set_outcome", "exclude", "note")
EVENT_KINDS = ("mitosis", "death", "slippage")
AUTHORS = ("human", "agent")

Anchor = tuple[int, int]


class EditError(ValueError):
    """Raised when an edit cannot be applied to the current lineage."""


@dataclass
class Edit:
    """One entry of the edit log.

    Attributes:
        id: Sequential id within the log (``e0001`` …).
        op: Operation name.
        well: Well the edit applies to.
        args: Operation arguments (anchors as ``[frame, label]``).
        author: ``human`` or ``agent``.
        confirmed_by: Who approved an agent's proposal (``human``), if anyone.
        reason: Why the edit was made.
        time: ISO timestamp.
    """

    id: str
    op: str
    well: str
    args: dict[str, Any]
    author: str = "human"
    confirmed_by: str | None = None
    reason: str = ""
    time: str = ""


@dataclass
class Curated:
    """A well's lineage as rebuilt from the repair plus the edit log.

    Attributes:
        tracks: ``(frame, raw label) -> curated track id`` for every detection.
        parents: Curated track id to parent id (0 = founder).
        annotations: Curated track id to its events, outcome, exclusion, notes.
        next_id: Next id handed to a track created by an edit.
        extras: Detections added by ``mask_add``: ``(frame, label) -> measurements``.
        removed: Raw detections replaced by a ``mask_add``.
        aliases: A replaced raw detection → the reviewer mask that took its
            place in the cell, so anchors naming it still resolve.
    """

    tracks: dict[Anchor, int]
    parents: dict[int, int]
    annotations: dict[int, dict[str, Any]] = field(default_factory=dict)
    next_id: int = 1
    extras: dict[Anchor, dict[str, Any]] = field(default_factory=dict)
    removed: set[Anchor] = field(default_factory=set)
    aliases: dict[Anchor, Anchor] = field(default_factory=dict)

    # -- lookups ---------------------------------------------------------

    def resolve(self, anchor: Anchor) -> int:
        """Curated track id of the cell at ``anchor``.

        Raises:
            EditError: If no detection carries that raw label in that frame.
        """
        key = (int(anchor[0]), int(anchor[1]))
        while key not in self.tracks and key in self.aliases:
            key = self.aliases[key]
        if key not in self.tracks:
            raise EditError(
                f"No nucleus with raw label {key[1]} in frame {key[0]}."
            )
        return self.tracks[key]

    def detections(self, tid: int) -> list[Anchor]:
        """All ``(frame, label)`` detections of a curated track, by frame."""
        return sorted(k for k, v in self.tracks.items() if v == tid)

    def children(self, tid: int) -> list[int]:
        """Daughters of a curated track."""
        return sorted(c for c, p in self.parents.items() if p == tid)

    # -- primitives ------------------------------------------------------

    def _new_track(self) -> int:
        tid = self.next_id
        self.next_id += 1
        self.parents[tid] = 0
        return tid

    def _reassign(self, keys: Iterable[Anchor], tid: int) -> None:
        for key in keys:
            self.tracks[key] = tid

    def _tail(self, tid: int, frame: int) -> list[Anchor]:
        return [k for k in self.detections(tid) if k[0] >= frame]

    def _drop_if_empty(self, tid: int) -> None:
        if tid in self.parents and not self.detections(tid):
            for child in self.children(tid):
                self.parents[child] = 0
            self.parents.pop(tid, None)
            self.annotations.pop(tid, None)

    def _split_at(self, tid: int, frame: int) -> int | None:
        """Move ``tid``'s detections from ``frame`` on (and its daughters) to a new track."""
        tail = self._tail(tid, frame)
        if not tail:
            return None
        new = self._new_track()
        self._reassign(tail, new)
        for child in self.children(tid):
            self.parents[child] = new
        return new

    # -- operations ------------------------------------------------------

    def link(self, cell: Anchor, frame: int, label: int) -> None:
        """From ``frame`` on, ``cell`` continues as the nucleus with ``label``."""
        tid = self.resolve(cell)
        target = self.resolve((frame, label))
        if target == tid:
            raise EditError(
                f"Label {label} in frame {frame} is already this cell."
            )
        if frame <= cell[0]:
            raise EditError(
                "A link must point to a frame after the anchor's frame."
            )
        # The cell's own detections from `frame` on belong to someone else now.
        self._split_at(tid, frame)
        # The target's earlier part (if any) stays as its own track.
        head_kept = [k for k in self.detections(target) if k[0] < frame]
        moved = self._tail(target, frame)
        self._reassign(moved, tid)
        # Daughters begin after the target's end, so they follow its tail.
        for child in self.children(target):
            self.parents[child] = tid
        if not head_kept:
            self.parents.pop(target, None)
            # Annotations of an absorbed founder follow the cell.
            notes = self.annotations.pop(target, None)
            if notes:
                self.annotations.setdefault(tid, {}).setdefault(
                    "absorbed", []
                ).append(notes)
        self._drop_if_empty(target)

    def unlink(self, cell: Anchor, frame: int) -> None:
        """Cut the cell's track at ``frame``; the tail becomes a founder."""
        tid = self.resolve(cell)
        if frame <= cell[0]:
            raise EditError(
                "Unlink must be at a frame after the anchor's frame."
            )
        if self._split_at(tid, frame) is None:
            raise EditError(
                f"The cell has no detections from frame {frame} on."
            )

    def set_parent(self, cell: Anchor, parent: Anchor) -> None:
        """Make the cell a daughter of ``parent``."""
        tid, pid = self.resolve(cell), self.resolve(parent)
        if tid == pid:
            raise EditError("A cell cannot be its own parent.")
        ancestor = pid
        while ancestor:
            if ancestor == tid:
                raise EditError("This would make a cell its own ancestor.")
            ancestor = self.parents.get(ancestor, 0)
        if self.detections(pid)[-1][0] >= self.detections(tid)[0][0]:
            raise EditError("The parent must end before the daughter begins.")
        self.parents[tid] = pid

    def clear_parent(self, cell: Anchor) -> None:
        """Make the cell a founder."""
        self.parents[self.resolve(cell)] = 0

    def swap(self, cell_a: Anchor, cell_b: Anchor, frame: int) -> None:
        """Exchange the two cells' identities from ``frame`` on."""
        a, b = self.resolve(cell_a), self.resolve(cell_b)
        if a == b:
            raise EditError("Swap needs two different cells.")
        tail_a, tail_b = self._tail(a, frame), self._tail(b, frame)
        if not tail_a or not tail_b:
            raise EditError(
                f"Both cells need detections from frame {frame} on."
            )
        kids_a, kids_b = self.children(a), self.children(b)
        self._reassign(tail_a, b)
        self._reassign(tail_b, a)
        for child in kids_a:
            self.parents[child] = b
        for child in kids_b:
            self.parents[child] = a

    def absorb(self, cell: Anchor, frames: tuple[int, int], label: int) -> int:
        """Add the nucleus ``label`` to the cell in every frame of ``frames`` it exists in."""
        tid = self.resolve(cell)
        keys = [
            (t, label)
            for t in range(frames[0], frames[1] + 1)
            if (t, label) in self.tracks
        ]
        if not keys:
            raise EditError(
                f"No nucleus {label} in frames {frames[0]}–{frames[1]}."
            )
        donors = {self.tracks[k] for k in keys} - {tid}
        self._reassign(keys, tid)
        for donor in donors:
            self._drop_if_empty(donor)
        return len(keys)

    def drop(self, frames: tuple[int, int], label: int) -> int:
        """Remove the nucleus ``label`` from its track in ``frames``."""
        keys = [
            (t, label)
            for t in range(frames[0], frames[1] + 1)
            if (t, label) in self.tracks
        ]
        if not keys:
            raise EditError(
                f"No tracked nucleus {label} in frames {frames[0]}–{frames[1]}."
            )
        owners = {self.tracks[k] for k in keys}
        for key in keys:
            del self.tracks[key]
        for owner in owners:
            self._drop_if_empty(owner)
        return len(keys)

    def mask_add(
        self,
        cell: Anchor | None,
        frame: int,
        label: int,
        measurements: dict[str, Any],
        replaces: list[int],
    ) -> None:
        """Add a reviewer-made nucleus; it replaces raw ``replaces`` labels in ``frame``."""
        key = (int(frame), int(label))
        if key in self.tracks or key in self.extras:
            raise EditError(f"Label {label} already exists in frame {frame}.")
        tid = self.resolve(cell) if cell is not None else None
        for old in replaces:
            old_key = (int(frame), int(old))
            owner = self.tracks.pop(old_key, None)
            self.removed.add(old_key)
            if owner is not None and (tid is None or owner == tid):
                # The new mask takes the replaced nucleus's place in its cell.
                self.aliases.setdefault(old_key, key)
            if owner is not None and owner != tid:
                self._drop_if_empty(owner)
        self.extras[key] = dict(measurements)
        if tid is None:
            tid = self._new_track()
        self.tracks[key] = tid

    def annotate(self, op: str, cell: Anchor, args: dict[str, Any]) -> None:
        """Record an event, outcome, exclusion or note on the cell."""
        notes = self.annotations.setdefault(self.resolve(cell), {})
        if op == "event":
            kind = args.get("kind")
            if kind not in EVENT_KINDS:
                raise EditError(
                    f"Event kind must be one of {EVENT_KINDS}, not {kind!r}."
                )
            notes.setdefault("events", []).append(
                {"kind": kind, "frame": int(args["frame"])}
            )
        elif op == "set_outcome":
            notes["outcome"] = str(args["outcome"])
        elif op == "exclude":
            notes["excluded"] = str(args.get("reason", ""))
        elif op == "note":
            notes.setdefault("notes", []).append(str(args["text"]))

    # -- output ----------------------------------------------------------

    def label_table(self) -> pd.DataFrame:
        """``timepoint, label, track_id, parent_track_id`` for every detection."""
        rows = [
            (t, label, tid, self.parents.get(tid, 0))
            for (t, label), tid in self.tracks.items()
        ]
        return pd.DataFrame(
            rows, columns=["timepoint", "label", "track_id", "parent_track_id"]
        )


def base_lineage(det: pd.DataFrame, result: RepairResult) -> Curated:
    """The automatically repaired lineage as a :class:`Curated` starting point.

    Args:
        det: Detections with ``track_id_raw`` and ``timepoint`` (the raw label
            of a detection is its raw track id).
        result: :func:`~cellview.tracks.repair.repair_lineage` output for ``det``.
    """
    tracks = {
        (int(t), int(raw)): result.assignment[int(raw)]
        for raw, t in zip(det["track_id_raw"], det["timepoint"], strict=True)
    }
    parents = {int(k): int(v) for k, v in result.parents.items()}
    # New tracks get ids above every raw label, so a curated id never equals
    # a raw track that the repair absorbed into another.
    labels = [label for _, label in tracks]
    next_id = max([*labels, *tracks.values(), *parents.keys(), 0]) + 1
    return Curated(tracks=tracks, parents=parents, next_id=next_id)


def _anchor(value: Any) -> Anchor:
    try:
        frame, label = value
        return int(frame), int(label)
    except (TypeError, ValueError) as err:
        raise EditError(
            f"An anchor is [frame, label], not {value!r}."
        ) from err


def _frames(value: Any) -> tuple[int, int]:
    try:
        t0, t1 = value
        return int(t0), int(t1)
    except (TypeError, ValueError) as err:
        raise EditError(f"frames is [first, last], not {value!r}.") from err


def apply_edit(curated: Curated, edit: Edit) -> None:
    """Apply one edit in place.

    Raises:
        EditError: If the edit is malformed or does not fit the lineage.
    """
    a = edit.args
    try:
        if edit.op == "link":
            curated.link(_anchor(a["cell"]), int(a["frame"]), int(a["label"]))
        elif edit.op == "unlink":
            curated.unlink(_anchor(a["cell"]), int(a["frame"]))
        elif edit.op == "set_parent":
            curated.set_parent(_anchor(a["cell"]), _anchor(a["parent"]))
        elif edit.op == "clear_parent":
            curated.clear_parent(_anchor(a["cell"]))
        elif edit.op == "swap":
            curated.swap(
                _anchor(a["cell"]), _anchor(a["other"]), int(a["frame"])
            )
        elif edit.op == "absorb":
            curated.absorb(
                _anchor(a["cell"]), _frames(a["frames"]), int(a["label"])
            )
        elif edit.op == "drop":
            curated.drop(_frames(a["frames"]), int(a["label"]))
        elif edit.op == "mask_add":
            cell = _anchor(a["cell"]) if a.get("cell") is not None else None
            curated.mask_add(
                cell,
                int(a["frame"]),
                int(a["label"]),
                a["measurements"],
                [int(x) for x in a.get("replaces", [])],
            )
        elif edit.op in ANNOTATION_OPS:
            curated.annotate(edit.op, _anchor(a["cell"]), a)
        else:
            raise EditError(f"Unknown operation {edit.op!r}.")
    except KeyError as err:
        raise EditError(f"{edit.op} needs argument {err}.") from err


def replay(base: Curated, edits: Iterable[Edit], well: str) -> Curated:
    """Rebuild a well's curated lineage: ``base`` plus every live edit, in order.

    ``base`` is not modified. Entries for other wells, ``revert`` entries and
    the entries they revert are skipped.
    """
    edits = list(edits)
    reverted = {e.args.get("target") for e in edits if e.op == "revert"}
    cur = Curated(
        tracks=dict(base.tracks),
        parents=dict(base.parents),
        annotations={k: dict(v) for k, v in base.annotations.items()},
        next_id=base.next_id,
        extras=dict(base.extras),
        removed=set(base.removed),
        aliases=dict(base.aliases),
    )
    for edit in edits:
        if edit.well != well or edit.op == "revert" or edit.id in reverted:
            continue
        try:
            apply_edit(cur, edit)
        except EditError as err:
            raise EditError(
                f"Edit {edit.id} ({edit.op}) no longer applies: {err}"
            ) from err
    return cur


class EditLog:
    """An append-only JSON-lines edit log on disk."""

    def __init__(self, path: Path) -> None:
        """Open (or later create) the log at ``path``."""
        self.path = Path(path)

    def __iter__(self) -> Iterator[Edit]:
        """Yield every entry, oldest first."""
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    raw = json.loads(line)
                    raw.pop("v", None)
                    yield Edit(**raw)

    def entries(self) -> list[Edit]:
        """All entries, oldest first."""
        return list(self)

    def append(
        self,
        op: str,
        well: str,
        args: dict[str, Any],
        author: str = "human",
        confirmed_by: str | None = None,
        reason: str = "",
        base: Curated | None = None,
    ) -> Edit:
        """Validate and append an edit.

        Args:
            op: Operation name (or ``revert``).
            well: Well the edit applies to.
            args: Operation arguments.
            author: ``human`` or ``agent``.
            confirmed_by: Who approved an agent's edit.
            reason: Why.
            base: The well's base lineage. If given, the edit is checked by
                replaying the log plus this edit before anything is written.

        Raises:
            EditError: If the edit is invalid; nothing is written.
        """
        if author not in AUTHORS:
            raise EditError(f"author must be one of {AUTHORS}.")
        existing = self.entries()
        if op == "revert":
            ids = {e.id for e in existing}
            if args.get("target") not in ids:
                raise EditError(f"No entry {args.get('target')!r} to revert.")
        elif op not in TRACK_OPS + ANNOTATION_OPS:
            raise EditError(f"Unknown operation {op!r}.")
        edit = Edit(
            id=f"e{len(existing) + 1:04d}",
            op=op,
            well=well,
            args=args,
            author=author,
            confirmed_by=confirmed_by,
            reason=reason,
            time=datetime.now().isoformat(timespec="seconds"),
        )
        if base is not None:
            replay(base, [*existing, edit], well)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps({"v": LOG_VERSION, **asdict(edit)}) + "\n")
        return edit

    def undo(
        self, well: str, author: str = "human", base: Curated | None = None
    ) -> Edit:
        """Revert the most recent live edit for ``well``.

        Raises:
            EditError: If there is nothing to undo.
        """
        entries = self.entries()
        reverted = {e.args.get("target") for e in entries if e.op == "revert"}
        live = [
            e
            for e in entries
            if e.well == well and e.op != "revert" and e.id not in reverted
        ]
        if not live:
            raise EditError(f"Nothing to undo for well {well}.")
        return self.append(
            "revert", well, {"target": live[-1].id}, author=author, base=base
        )
