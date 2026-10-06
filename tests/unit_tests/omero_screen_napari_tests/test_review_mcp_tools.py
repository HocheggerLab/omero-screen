"""Agent tools registered through napari-mcp's napari_mcp.tools hook."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

fastmcp = pytest.importorskip("fastmcp")

from omero_screen_napari.review import mcp_tools  # noqa: E402
from omero_screen_napari.review.session import (
    ReviewSession,
    set_active_session,
)  # noqa: E402
from omero_screen_napari.review_queue import PathPoint, ReviewItem, write_queue  # noqa: E402


@pytest.fixture
def server(tmp_path: Path):
    """A FastMCP server with our tools, acting on a session over a small queue."""
    q = tmp_path / "queue.json"
    item = ReviewItem(
        id="C2-t72-L5",
        well="C2",
        frame=80,
        reason="lost gap",
        question="?",
        outcome="lost",
        path=(PathPoint(72, 1.0, 1.0, 5), PathPoint(80, 2.0, 2.0, 5)),
    )
    write_queue(q, 5054, [item])
    session = ReviewSession(q)
    set_active_session(session, None)
    srv = fastmcp.FastMCP("test")
    state = MagicMock()
    state.gui_execute = lambda fn: fn()
    mcp_tools.register(srv, state)
    yield srv, session
    set_active_session(None)


def _call(srv, name, args=None):
    async def go():
        async with fastmcp.Client(srv) as c:
            r = await c.call_tool(name, args or {})
            if not r.content:
                return r.structured_content.get("result", r.structured_content)
            return (
                json.loads(r.content[0].text)
                if r.content[0].type == "text"
                else r
            )

    return asyncio.run(go())


def test_tools_are_registered(server) -> None:
    """Every review tool is exposed under its own name."""
    srv, _ = server

    async def names():
        async with fastmcp.Client(srv) as c:
            return {t.name for t in await c.list_tools()}

    assert {
        "review_state",
        "review_list",
        "review_goto",
        "cell_trace",
        "filmstrip",
        "continuation_candidates",
        "label_at",
        "cell_edits",
        "propose_edit",
        "proposal_status",
        "validate",
        "session_summary",
    } <= asyncio.run(names())


def test_state_and_list(server) -> None:
    """State reports progress; list filters by flag."""
    srv, _ = server
    assert _call(srv, "review_state")["items"] == 1
    assert [i["id"] for i in _call(srv, "review_list", {"flag": "gap"})] == [
        "C2-t72-L5"
    ]
    assert _call(srv, "review_list", {"flag": "debris"}) == []


def test_proposals_wait_for_confirmation(server) -> None:
    """propose_edit stores a pending proposal and changes nothing; bad ops are refused."""
    srv, session = server
    with patch.object(ReviewSession, "edit") as edit:
        p = _call(
            srv,
            "propose_edit",
            {
                "item_id": "C2-t72-L5",
                "op": "link",
                "args": {"frame": 81, "label": 9},
                "reason": "close",
            },
        )
        edit.assert_not_called()
    assert p["status"] == "pending"
    assert (
        _call(srv, "proposal_status", {"proposal_id": p["id"]})[0]["status"]
        == "pending"
    )
    with pytest.raises(Exception, match="op must be one of"):
        _call(
            srv,
            "propose_edit",
            {"item_id": "C2-t72-L5", "op": "swap", "args": {}, "reason": "x"},
        )


def test_candidates_are_json_clean(server) -> None:
    """Numeric results come back as plain JSON numbers (NaN as null)."""
    srv, _ = server
    df = pd.DataFrame(
        {
            "rank": [1],
            "timepoint": [81],
            "label": [9],
            "distance": [1.23456],
            "cost": [float("nan")],
        }
    )
    with patch.object(ReviewSession, "candidates", return_value=df):
        out = _call(
            srv,
            "continuation_candidates",
            {"item_id": "C2-t72-L5", "frame": 81},
        )
    assert out == [
        {
            "rank": 1,
            "timepoint": 81,
            "label": 9,
            "distance": 1.2346,
            "cost": None,
        }
    ]


def test_no_session_is_a_clear_error(tmp_path: Path) -> None:
    """Without an open widget the tools say so."""
    set_active_session(None)
    srv = fastmcp.FastMCP("t")
    mcp_tools.register(srv, MagicMock())
    with pytest.raises(Exception, match="Track Review widget"):
        _call(srv, "review_state")
