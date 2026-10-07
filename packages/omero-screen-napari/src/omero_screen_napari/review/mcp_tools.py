"""Track-review tools for agents, contributed to napari-mcp.

napari-mcp discovers this module through the ``napari_mcp.tools`` entry point
(see ``pyproject.toml``) and calls :func:`register` when its server starts in
the napari plugin. The tools act on the review session of the open Track
Review widget:

* navigation and reading: ``review_state``, ``review_list``, ``review_goto``,
  ``cell_trace``, ``filmstrip``, ``continuation_candidates``, ``label_at``,
  ``cell_edits``, ``validate``, ``session_summary``;
* the only way to change anything: ``propose_edit``. A proposal appears in
  the widget's *Agent proposals* list and is applied, as an edit authored by
  the agent and confirmed by the human, only when the reviewer confirms it.
  ``proposal_status`` reports what happened.

Anything that moves the viewer runs on the Qt main thread via
``state.gui_execute``. Reading data does not touch the GUI.
"""

from __future__ import annotations

import io
import math
from typing import Any, cast

#: Operations an agent may propose.
PROPOSABLE = (
    "link",
    "unlink",
    "absorb",
    "drop",
    "event",
    "set_outcome",
    "exclude",
    "note",
)


class NoReviewSession(RuntimeError):
    """Raised when no Track Review widget has a queue open."""


def _session() -> Any:
    from omero_screen_napari.review.session import active_session

    session = active_session()
    if session is None:
        raise NoReviewSession(
            "Open the Track Review widget in napari and load a queue first."
        )
    return session


def _clean(value: Any) -> Any:
    """Make numpy/pandas scalars and NaN JSON-friendly."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    return value


def _cdict(value: dict[str, Any]) -> dict[str, Any]:
    """:func:`_clean` for a dict, typed as one."""
    return cast(dict[str, Any], _clean(value))


def _records(df: Any, cols: list[str] | None = None) -> list[dict[str, Any]]:
    if df is None or len(df) == 0:
        return []
    sub = df[cols] if cols else df
    return [_clean(r) for r in sub.to_dict("records")]


def register(server: Any, state: Any) -> None:
    """Add the track-review tools to a napari-mcp server."""
    from fastmcp.utilities.types import Image

    @server.tool()
    async def review_state() -> dict[str, Any]:
        """Queue progress, the cell on screen, the current frame and pending proposals."""
        from omero_screen_napari.review.session import active_ui

        session = _session()
        out = session.status()
        ui = active_ui()
        if ui is not None and ui.current is not None:
            out["current_item"] = ui.current.id
            out["current_frame"] = state.gui_execute(
                lambda: int(ui.viewer.dims.current_step[0])
            )
        out["queue"] = str(session.queue_path)
        return _cdict(out)

    @server.tool()
    async def review_list(
        well: str | None = None, flag: str | None = None, status: str = "open"
    ) -> list[dict[str, Any]]:
        """Queue items, filtered.

        Args:
            well: Only this well.
            flag: Only items whose reason contains this flag (e.g. "lost").
            status: "open" (no verdict yet), "done", or "all".
        """
        from omero_screen_napari.review_queue import (
            decisions_path,
            latest_decisions,
        )

        session = _session()
        done = latest_decisions(decisions_path(session.queue_path))
        out = []
        for it in session.queue.items:
            if well and it.well != well:
                continue
            if flag and flag not in it.reason.split():
                continue
            if (
                status == "open"
                and it.id in done
                or status == "done"
                and it.id not in done
            ):
                continue
            out.append(
                {
                    "id": it.id,
                    "well": it.well,
                    "frame": it.frame,
                    "reason": it.reason,
                    "automatic_outcome": it.outcome,
                    "verdict": done[it.id].verdict if it.id in done else None,
                }
            )
        return out

    @server.tool()
    async def review_goto(
        item_id: str, frame: int | None = None
    ) -> dict[str, Any]:
        """Show a queued cell in the reviewer's napari: load its well, draw it, jump to a frame.

        Args:
            item_id: Queue item id, e.g. "C2-t72-L524".
            frame: Frame to show (default: the item's frame).
        """
        from omero_screen_napari.review.session import active_ui

        session = _session()
        item = session.item(item_id)
        ui = active_ui()
        if ui is None:
            raise NoReviewSession("The Track Review widget is not open.")

        def _go() -> None:
            ui._select(item_id)
            if frame is not None:
                ui.viewer.dims.set_current_step(0, int(frame))
                ui._centre(int(frame))

        state.gui_execute(_go)
        return _cdict(
            {
                "id": item.id,
                "well": item.well,
                "frame": frame if frame is not None else item.frame,
                "reason": item.reason,
                "question": item.question,
                "automatic_outcome": item.outcome,
                "breaks": session.breaks(item_id),
            }
        )

    @server.tool()
    async def cell_trace(
        item_id: str, start: int | None = None, stop: int | None = None
    ) -> dict[str, Any]:
        """Per-frame record of a cell along its curated track.

        Rows: timepoint, raw label (0 = no mask), gap, area, y, x, every
        nuclear channel (background-subtracted), and the PIP-FUCCI phase.
        Also returns the track's breaks and the cell's edit-log entries.
        """
        session = _session()
        path = session.path(item_id)
        if start is not None:
            path = path[path["timepoint"] >= start]
        if stop is not None:
            path = path[path["timepoint"] <= stop]
        return {
            "item": item_id,
            "breaks": session.breaks(item_id),
            "frames": _records(path),
            "edits": [e.__dict__ for e in session.entries(item_id)],
        }

    @server.tool()
    async def filmstrip(
        item_id: str,
        start: int | None = None,
        stop: int | None = None,
        channels: str | None = None,
        max_tiles: int = 24,
    ) -> Any:
        """Image of the cell followed through time (crops centred on it in every frame) with its
        PIP/geminin/area trace and phase bands. Frames without a mask are crossed.

        Args:
            item_id: Queue item id.
            start: First frame (default: the anchor frame).
            stop: Last frame (default: end of the track).
            channels: Comma-separated channels (default: all but brightfield).
            max_tiles: At most this many frames shown, evenly spread.
        """
        import matplotlib.pyplot as plt

        from omero_screen_napari.review.cells import render_cell

        session = _session()
        item = session.item(item_id)
        anchor = session.anchor(item_id)
        chans = [c.strip() for c in channels.split(",")] if channels else None
        fig = render_cell(
            session.source,
            item.well,
            anchor,
            chans,
            160,
            max_tiles,
            start if start is not None else anchor[0],
            stop,
            title=item_id,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        return Image(data=buf.getvalue(), format="png").to_image_content()

    @server.tool()
    async def continuation_candidates(
        item_id: str, frame: int
    ) -> list[dict[str, Any]]:
        """Nuclei that could continue the cell at or after `frame`, best first.

        Each: rank, timepoint, raw label, distance (nuclear diameters), gap
        (frames skipped), area ratio, reporter ratios and a combined cost. The
        reviewer sees the same numbers on screen (Shift-1…9 links rank n).
        """
        session = _session()
        return _records(session.candidates(item_id, int(frame)))

    @server.tool()
    async def label_at(
        well: str, frame: int, y: float, x: float
    ) -> dict[str, Any]:
        """The nucleus nearest to pixel (y, x) in a frame: raw label, curated track, measurements."""
        return _cdict(
            _session().label_at(well, int(frame), float(y), float(x))
        )

    @server.tool()
    async def cell_edits(item_id: str) -> list[dict[str, Any]]:
        """Edit-log entries that name this cell (who, what, why)."""
        return [_clean(e.__dict__) for e in _session().entries(item_id)]

    @server.tool()
    async def propose_edit(
        item_id: str,
        op: str,
        args: dict[str, Any],
        reason: str,
        confidence: str = "uncertain",
    ) -> dict[str, Any]:
        """Propose a change. In a human run nothing changes until the reviewer confirms;
        in an agent run (queue built with --mode agent) it is applied at once.

        Args:
            item_id: Queue item id.
            op: One of link, unlink, absorb, drop, event, set_outcome, exclude, note.
            args: For link {"frame", "label"}; unlink {"frame"}; absorb / drop
                {"frames": [first, last], "label"}; event
                {"kind": mitosis|death|slippage, "frame"}; set_outcome
                {"outcome"}; exclude {"reason"}; note {"text"}.
            reason: Why, in one sentence the reviewer can check on screen.
            confidence: "clear" if the filmstrip and numbers leave no doubt,
                otherwise "uncertain".
        """
        from omero_screen_napari.review.session import active_ui

        if op not in PROPOSABLE:
            raise ValueError(f"op must be one of {PROPOSABLE}")
        if confidence not in ("clear", "uncertain"):
            raise ValueError("confidence must be 'clear' or 'uncertain'")
        session = _session()
        prop = session.propose(item_id, op, args, reason, confidence)
        ui = active_ui()
        if (
            ui is not None
            and ui.current is not None
            and ui.current.id == item_id
        ):
            state.gui_execute(ui._refresh_proposals)
            if prop.status == "applied":
                state.gui_execute(ui._redraw)
        return _cdict(prop.__dict__)

    @server.tool()
    async def agent_verdict(
        item_id: str, verdict: str, outcome: str = "", note: str = ""
    ) -> dict[str, Any]:
        """Record the agent's own verdict on an item. Only allowed in an agent run
        (queue built with --mode agent); in a human run the reviewer decides.

        Args:
            item_id: Queue item id.
            verdict: accept, correct, reject or unsure.
            outcome: The outcome you conclude, if it differs from the automatic one.
            note: One sentence on why.
        """
        session = _session()
        if session.mode != "agent":
            raise PermissionError(
                "Verdicts are the reviewer's in a human run; use propose_edit and let them decide."
            )
        d = session.verdict(item_id, verdict, outcome, note, author="agent")
        return _cdict(d.__dict__)

    @server.tool()
    async def proposal_status(
        proposal_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Status of one proposal, or of all (pending / confirmed / rejected, with the reviewer's note)."""
        props = _session().proposals()
        if proposal_id:
            props = [p for p in props if p.id == proposal_id]
        return [_clean(p.__dict__) for p in props]

    @server.tool()
    async def validate(item_id: str | None = None) -> dict[str, Any]:
        """Check the curated lineage after replay: per cell (or for every queued cell),
        report breaks before the window end and cells with two masks in one frame."""
        session = _session()
        items = [session.item(item_id)] if item_id else session.queue.items
        report = []
        for it in items:
            path = session.path(it.id)
            entry = {"id": it.id, "breaks": session.breaks(it.id)}
            if "n_pieces" in path:
                entry["multi_piece_frames"] = [
                    int(t) for t in path.loc[path["n_pieces"] > 1, "timepoint"]
                ]
            report.append(entry)
        return {"cells": report}

    @server.tool()
    async def session_summary() -> dict[str, Any]:
        """Review progress and agent–human agreement: verdicts, edits by author, proposals confirmed vs rejected."""
        from cellview.tracks.edit import EditLog

        from omero_screen_napari.review_queue import (
            decisions_path,
            latest_decisions,
        )

        session = _session()
        verdicts: dict[str, int] = {}
        for d in latest_decisions(decisions_path(session.queue_path)).values():
            verdicts[d.verdict] = verdicts.get(d.verdict, 0) + 1
        authors: dict[str, int] = {}
        for e in EditLog(session.edits_path):
            key = e.author + ("/confirmed" if e.confirmed_by else "")
            authors[key] = authors.get(key, 0) + 1
        props: dict[str, int] = {}
        for p in session.proposals():
            props[p.status] = props.get(p.status, 0) + 1
        resolved = props.get("confirmed", 0) + props.get("rejected", 0)
        return _cdict(
            {
                **session.status(),
                "verdicts": verdicts,
                "edits_by_author": authors,
                "proposals": props,
                "agreement": props.get("confirmed", 0) / resolved
                if resolved
                else None,
            }
        )
