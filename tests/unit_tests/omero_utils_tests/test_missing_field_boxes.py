"""Tests for ``missing_field_boxes``.

A field whose acquisition failed leaves a tile-sized hole in the stitched
canvas. Its slot cannot be inferred from the offsets — it has no stage
position, and the acquisition pattern leaves grid cells empty anyway — so
the hole is located geometrically from the area no tile covers.

Verified against production: plate 4217 well F3 (field 6 invalid) yields
exactly one 1066x1066 box, while well A1 on the same plate yields none
despite four unimaged grid corners.
"""

from __future__ import annotations

import numpy as np
from omero_utils.stitching import missing_field_boxes

TILE = 100
STEP = 93  # tile minus a 7px overlap


def _grid(rows: int, cols: int, drop: set[int] | None = None) -> np.ndarray:
    """Offsets for a rows x cols grid, with `drop` field indices invalid."""
    drop = drop or set()
    out = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            out.append((-1, -1) if i in drop else (c * STEP, r * STEP))
    return np.array(out, dtype=np.int_)


def test_no_gap_returns_nothing() -> None:
    """A complete grid has no interior hole."""
    assert missing_field_boxes(_grid(3, 3), TILE, TILE) == []


def test_dropped_interior_field_is_found() -> None:
    """The centre of a 3x3 grid is enclosed, so its gap is reported."""
    boxes = missing_field_boxes(_grid(3, 3, drop={4}), TILE, TILE)
    assert len(boxes) == 1
    y0, x0, y1, x1 = boxes[0]
    # The hole is the tile minus the overlap its neighbours reach into.
    assert (y1 - y0, x1 - x0) == (STEP - (TILE - STEP), STEP - (TILE - STEP))
    # And it sits at the middle cell, inset by the neighbours' overlap.
    assert (y0, x0) == (TILE, TILE)


def test_dropped_edge_field_is_not_reported() -> None:
    """An outer-ring gap runs to the border and reads as pattern, not hole.

    Reporting it would also flag every unimaged grid corner of a normal
    acquisition, which is exactly the false positive to avoid.
    """
    assert missing_field_boxes(_grid(3, 3, drop={0}), TILE, TILE) == []
    assert missing_field_boxes(_grid(3, 3, drop={1}), TILE, TILE) == []


def test_multiple_interior_gaps() -> None:
    """Two enclosed holes give two boxes, ordered top-left first."""
    boxes = missing_field_boxes(_grid(3, 5, drop={6, 8}), TILE, TILE)
    assert len(boxes) == 2
    assert boxes == sorted(boxes)


def test_all_invalid_returns_nothing() -> None:
    """No valid field means no canvas at all — nothing to annotate."""
    offsets = np.full((4, 2), -1, dtype=np.int_)
    assert missing_field_boxes(offsets, TILE, TILE) == []


def test_shear_wedges_are_not_reported() -> None:
    """The rotation between stage and camera frames leaves thin edge wedges.

    They are uncovered but far below half a tile, and touch the border.
    """
    offsets = []
    for r in range(4):
        for c in range(4):
            offsets.append((c * STEP + r * 3, r * STEP - c * 3 + 9))
    assert (
        missing_field_boxes(np.array(offsets, dtype=np.int_), TILE, TILE) == []
    )
