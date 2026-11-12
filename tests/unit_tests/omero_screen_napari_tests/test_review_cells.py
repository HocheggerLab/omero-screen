"""CellSource caching of curated wells across threads."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from omero_screen_napari.review.cells import CellSource


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> CellSource:
    """A source whose raw table and base lineage need no database."""
    src = CellSource(plate_id=1)
    src._raw["B2"] = pd.DataFrame()
    src._bases["B2"] = "base"
    monkeypatch.setattr(
        "omero_screen_track.cells.apply_extras", lambda raw, curated: curated
    )
    return src


def test_well_is_cached_until_invalidated(
    source: CellSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def replay(base: Any, log: Any, well: str) -> str:
        calls.append(1)
        return f"v{len(calls)}"

    monkeypatch.setattr("omero_screen_track.edit.replay", replay)
    assert source.well("B2")[1] == "v1"
    assert source.well("B2")[1] == "v1"
    source.invalidate("B2")
    assert source.well("B2")[1] == "v2"


def test_replay_overtaken_by_an_edit_is_not_cached(
    source: CellSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit landing mid-replay (another thread) must not be lost."""
    calls: list[int] = []

    def replay(base: Any, log: Any, well: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            # The MCP thread appends an edit and invalidates the well
            # while this replay is still running on the Qt thread.
            source.invalidate("B2")
            return "without the edit"
        return "with the edit"

    monkeypatch.setattr("omero_screen_track.edit.replay", replay)
    assert source.well("B2")[1] == "with the edit"
    assert source.well("B2")[1] == "with the edit"
    assert len(calls) == 2
