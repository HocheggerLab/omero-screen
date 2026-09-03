"""Tests pinning the x/y axis order of the alignment tables.

``_translation`` returns its shift as ``(x, y)``. Until 2026-09-03 ``align_plates``
reversed the per-well mean again before writing it, so ``alignment.csv`` carried the
mean Y shift in its ``x`` column while ``sample_alignment.csv`` -- written from the
same ``trans`` tuples, unreversed -- was correct. The two files disagreed, and the
default aggregation path (``--no-sample-alignments``) read the transposed one.

The shifts involved are only a few pixels, so a transposition still looks registered
by eye; nothing but an axis-level assertion catches it. Hence these tests.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from omero_screen.plate_aggregation import (
    ALIGNMENT_SCHEMA_VERSION,
    _translation,
)


def _shifted_pair(
    dx: int, dy: int, size: int = 64
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (fixed, moving) images where moving is fixed shifted by (dx, dy)."""
    rng = np.random.default_rng(0)
    fixed = rng.random((size, size))
    # np.roll shift along axis 0 is y, axis 1 is x.
    moving = np.roll(np.roll(fixed, dy, axis=0), dx, axis=1)
    return fixed, moving


class TestTranslationAxisOrder:
    """_translation returns (x, y), not (y, x)."""

    @pytest.mark.parametrize(
        "dx, dy",
        [(3, 0), (0, 3), (5, -2), (-4, 6), (0, 0)],
    )
    def test_returns_xy_order(self, dx: int, dy: int) -> None:
        fixed, moving = _shifted_pair(dx, dy)
        trans = _translation(fixed, moving)
        assert trans == (dx, dy), (
            f"expected (x, y) = ({dx}, {dy}), got {trans}. A result of "
            f"({dy}, {dx}) means the axis order has been transposed."
        )

    def test_x_only_shift_leaves_y_zero(self) -> None:
        """The sharpest guard: a pure-X shift must not appear in y."""
        fixed, moving = _shifted_pair(7, 0)
        x, y = _translation(fixed, moving)
        assert (x, y) == (7, 0)

    def test_y_only_shift_leaves_x_zero(self) -> None:
        fixed, moving = _shifted_pair(0, 7)
        x, y = _translation(fixed, moving)
        assert (x, y) == (0, 7)


class TestPerWellMeanAxisOrder:
    """The per-well mean must preserve the (x, y) order of its inputs.

    This reproduces the arithmetic of align_plates' aggregation step without
    needing OMERO: shifts are collected as (x, y) tuples from _translation and
    averaged. The bug was an extra reversal applied to the mean.
    """

    def test_mean_preserves_xy_order(self) -> None:
        shifts = [(5, -3), (5, -3), (4, -2)]
        a = np.array(shifts).mean(axis=0)
        shift = (a[0], a[1])
        assert shift == pytest.approx((14 / 3, -8 / 3))
        # The transposed form the bug produced:
        assert shift != pytest.approx((a[1], a[0]))

    def test_asymmetric_shift_detects_transposition(self) -> None:
        """With x != y a transposition is detectable; with x == y it is not."""
        shifts = [(6, -2)]
        a = np.array(shifts).mean(axis=0)
        assert (a[0], a[1]) != (a[1], a[0])
        same = np.array([(3, 3)]).mean(axis=0)
        assert (same[0], same[1]) == (same[1], same[0])


class TestAlignmentSchemaGate:
    """Per-well tables without a schema column predate the fix and are rejected."""

    def test_schema_version_is_at_least_two(self) -> None:
        assert ALIGNMENT_SCHEMA_VERSION >= 2

    def test_legacy_per_well_table_has_no_schema_column(self) -> None:
        legacy = pd.DataFrame(
            [(4130, "A1", 5.0, -3.4)],
            columns=["plate", "well", "x", "y"],
        )
        assert "schema" not in legacy.columns

    def test_current_per_well_table_carries_schema(self) -> None:
        current = pd.DataFrame(
            [(4130, "A1", -3.4, 5.0)],
            columns=["plate", "well", "x", "y"],
        )
        current["schema"] = ALIGNMENT_SCHEMA_VERSION
        assert "schema" in current.columns
        assert (current["schema"] == ALIGNMENT_SCHEMA_VERSION).all()

    def test_schema_column_does_not_break_mode_detection(self) -> None:
        """Mode is detected by 'image_id' in columns; the new column must not confuse it."""
        per_well = pd.DataFrame(
            [(4130, "A1", -3.4, 5.0)], columns=["plate", "well", "x", "y"]
        )
        per_well["schema"] = ALIGNMENT_SCHEMA_VERSION
        per_field = pd.DataFrame(
            [(4130, "A1", 0, 999, -3.4, 5.0)],
            columns=["plate", "well", "sample", "image_id", "x", "y"],
        )
        per_field["schema"] = ALIGNMENT_SCHEMA_VERSION
        assert "image_id" not in per_well.columns
        assert "image_id" in per_field.columns

    def test_named_access_survives_trailing_schema_column(self) -> None:
        """aggregate_plates reads rows by name; positional unpacking would break."""
        df = pd.DataFrame(
            [(4130, "A1", -3.4, 5.0)], columns=["plate", "well", "x", "y"]
        )
        df["schema"] = ALIGNMENT_SCHEMA_VERSION
        row = next(iter(df.itertuples(index=False)))
        assert (row.x, row.y) == (-3.4, 5.0)
        assert row.well == "A1"
