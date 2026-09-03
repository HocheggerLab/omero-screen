"""Tests for ``zarr_cache.alignment``.

Two hazards drive most of these tests:

* ``get_file_attachments`` matches attachment names by *suffix*, so a request for
  ``alignment.csv`` would also match ``sample_alignment.csv``. The loader must
  compare names exactly, or a 4i plate silently loads the wrong table.
* Until 2026-09-03 ``alignment.csv`` was written with x and y transposed while
  ``sample_alignment.csv`` was correct. Post-fix files carry a ``schema`` column;
  a per-well table without one must be rejected, never reinterpreted.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from omero.gateway import FileAnnotationWrapper

from omero_screen_napari.zarr_cache.alignment import (
    MIN_PER_WELL_SCHEMA,
    AlignmentError,
    PlateAlignment,
    has_alignment,
    load_alignment,
    validate_table,
)


def _per_field_df(schema: int | None = MIN_PER_WELL_SCHEMA) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            (4130, "A1", 0, 900, 5, -3),
            (4130, "A1", 1, 901, 5, -3),
            (4130, "A1", 2, 902, 4, -2),
            (4130, "B2", 0, 910, 1, 1),
            (4131, "A1", 0, 920, -2, 7),
        ],
        columns=["plate", "well", "sample", "image_id", "x", "y"],
    )
    if schema is not None:
        df["schema"] = schema
    return df


def _per_well_df(schema: int | None = MIN_PER_WELL_SCHEMA) -> pd.DataFrame:
    df = pd.DataFrame(
        [(4130, "A1", 5, -3), (4130, "B2", 1, 1), (4131, "A1", -2, 7)],
        columns=["plate", "well", "x", "y"],
    )
    if schema is not None:
        df["schema"] = schema
    return df


def _annotation(name: str) -> MagicMock:
    ann = MagicMock(spec=FileAnnotationWrapper)
    file_obj = MagicMock()
    file_obj.getName.return_value = name
    ann.getFile.return_value = file_obj
    return ann


def _plate(*names: str) -> MagicMock:
    plate = MagicMock()
    plate.listAnnotations.return_value = [_annotation(n) for n in names]
    return plate


def _conn(plate: MagicMock | None) -> MagicMock:
    conn = MagicMock()
    conn.getObject.return_value = plate
    return conn


class TestHasAlignment:
    def test_true_for_master_plate(self) -> None:
        assert has_alignment(_plate("alignment.csv", "sample_alignment.csv"))

    def test_true_with_only_per_field(self) -> None:
        assert has_alignment(_plate("sample_alignment.csv"))

    def test_false_for_ordinary_plate(self) -> None:
        assert not has_alignment(_plate("canvas.csv", "final_data_cc.csv"))

    def test_case_insensitive(self) -> None:
        assert has_alignment(_plate("Alignment.CSV"))

    def test_does_not_download_the_file(self) -> None:
        """Only annotation metadata is touched; asFileObj must not be called."""
        plate = _plate("alignment.csv")
        assert has_alignment(plate)
        for ann in plate.listAnnotations.return_value:
            ann.getFile.return_value.asFileObj.assert_not_called()


class TestExactNameMatching:
    """The suffix-matching trap: sample_alignment.csv ends with alignment.csv."""

    def test_per_well_request_ignores_per_field_file(self, monkeypatch) -> None:
        plate = _plate("sample_alignment.csv")
        captured: list[str] = []

        def fake_parse(ann):
            captured.append(ann.getFile().getName())
            return _per_field_df()

        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.alignment.parse_csv_data",
            fake_parse,
        )
        # Forcing per-well when only the per-field file exists must not fall
        # through to it by suffix match.
        result = load_alignment(_conn(plate), 4127, prefer_per_field=False)
        assert result.source == "sample_alignment"
        assert captured == ["sample_alignment.csv"]

    def test_plate_with_only_per_well_file(self, monkeypatch) -> None:
        plate = _plate("alignment.csv")
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.alignment.parse_csv_data",
            lambda ann: _per_well_df(),
        )
        result = load_alignment(_conn(plate), 4127)
        assert result.source == "alignment"
        assert not result.per_field


class TestLoadAlignmentPreference:
    def test_prefers_per_field(self, monkeypatch) -> None:
        plate = _plate("alignment.csv", "sample_alignment.csv")
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.alignment.parse_csv_data",
            lambda ann: _per_field_df()
            if ann.getFile().getName() == "sample_alignment.csv"
            else _per_well_df(),
        )
        result = load_alignment(_conn(plate), 4127)
        assert result.source == "sample_alignment"
        assert result.per_field

    def test_falls_back_to_per_well(self, monkeypatch) -> None:
        plate = _plate("alignment.csv")
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.alignment.parse_csv_data",
            lambda ann: _per_well_df(),
        )
        assert load_alignment(_conn(plate), 4127).source == "alignment"

    def test_missing_plate_raises(self) -> None:
        with pytest.raises(AlignmentError, match="not found"):
            load_alignment(_conn(None), 4127)

    def test_no_attachments_raises(self) -> None:
        with pytest.raises(AlignmentError, match="no usable alignment"):
            load_alignment(_conn(_plate("canvas.csv")), 4127)

    def test_legacy_per_well_only_raises(self, monkeypatch) -> None:
        """A pre-fix per-well table is rejected, not silently transposed."""
        plate = _plate("alignment.csv")
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.alignment.parse_csv_data",
            lambda ann: _per_well_df(schema=None),
        )
        with pytest.raises(AlignmentError, match="transposed"):
            load_alignment(_conn(plate), 4127)

    def test_legacy_per_well_falls_through_to_per_field(
        self, monkeypatch
    ) -> None:
        """A legacy per-well file must not block the correct per-field one."""
        plate = _plate("alignment.csv", "sample_alignment.csv")
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.alignment.parse_csv_data",
            lambda ann: _per_field_df(schema=None)
            if ann.getFile().getName() == "sample_alignment.csv"
            else _per_well_df(schema=None),
        )
        # Legacy sample_alignment.csv was never transposed, so it is accepted.
        assert load_alignment(_conn(plate), 4127).source == "sample_alignment"


class TestValidateTable:
    def test_accepts_current_per_well(self) -> None:
        validate_table(_per_well_df(), "alignment", 4127)

    def test_rejects_legacy_per_well(self) -> None:
        with pytest.raises(AlignmentError, match="transposed"):
            validate_table(_per_well_df(schema=None), "alignment", 4127)

    def test_accepts_legacy_per_field(self) -> None:
        """sample_alignment.csv was never affected by the transposition."""
        validate_table(
            _per_field_df(schema=None), "sample_alignment", 4127
        )

    def test_rejects_missing_columns(self) -> None:
        df = pd.DataFrame([(4130, "A1")], columns=["plate", "well"])
        with pytest.raises(AlignmentError, match="missing columns"):
            validate_table(df, "alignment", 4127)

    def test_per_field_requires_sample_and_image_id(self) -> None:
        with pytest.raises(AlignmentError, match="missing columns"):
            validate_table(_per_well_df(), "sample_alignment", 4127)


class TestMemberPlateIds:
    def test_lists_restain_plates_only(self) -> None:
        a = PlateAlignment(4127, "sample_alignment", _per_field_df())
        assert a.member_plate_ids == (4130, 4131)
        assert 4127 not in a.member_plate_ids

    def test_wells_per_plate(self) -> None:
        a = PlateAlignment(4127, "sample_alignment", _per_field_df())
        assert a.wells(4130) == {"A1", "B2"}
        assert a.wells(4131) == {"A1"}


class TestShiftsForWell:
    def test_per_field_shifts_indexed_by_master_field(self) -> None:
        a = PlateAlignment(4127, "sample_alignment", _per_field_df())
        result = a.shifts_for_well(4130, "A1", n_fields=3)
        np.testing.assert_array_equal(
            result.shifts, np.array([[5, -3], [5, -3], [4, -2]])
        )
        assert result.image_ids == [900, 901, 902]
        assert result.imputed == ()

    def test_per_well_shift_broadcast_to_all_fields(self) -> None:
        a = PlateAlignment(4127, "alignment", _per_well_df())
        result = a.shifts_for_well(4130, "A1", n_fields=4)
        np.testing.assert_array_equal(
            result.shifts, np.array([[5, -3]] * 4)
        )
        assert result.image_ids is None

    def test_xy_order_is_preserved(self) -> None:
        """Guards the axis convention: column x -> shifts[:, 0]."""
        df = pd.DataFrame(
            [(4130, "A1", 0, 900, 11, -22)],
            columns=["plate", "well", "sample", "image_id", "x", "y"],
        )
        a = PlateAlignment(4127, "sample_alignment", df)
        result = a.shifts_for_well(4130, "A1", n_fields=1)
        assert tuple(result.shifts[0]) == (11, -22)

    def test_missing_field_imputed_from_well_mean(self) -> None:
        a = PlateAlignment(4127, "sample_alignment", _per_field_df())
        result = a.shifts_for_well(4130, "A1", n_fields=5)
        assert result.imputed == (3, 4)
        # mean of (5,-3), (5,-3), (4,-2) -> (4.67, -2.67) -> rint (5, -3)
        np.testing.assert_array_equal(result.shifts[3], np.array([5, -3]))

    def test_empty_frame_zero_rows_are_imputed_not_trusted(self) -> None:
        """align_plates writes an exact (0, 0) for a frame it could not correlate."""
        df = pd.DataFrame(
            [
                (4130, "A1", 0, 900, 6, -4),
                (4130, "A1", 1, 901, 0, 0),
                (4130, "A1", 2, 902, 6, -4),
            ],
            columns=["plate", "well", "sample", "image_id", "x", "y"],
        )
        a = PlateAlignment(4127, "sample_alignment", df)
        result = a.shifts_for_well(4130, "A1", n_fields=3)
        assert result.imputed == (1,)
        np.testing.assert_array_equal(result.shifts[1], np.array([6, -4]))
        # The image ID is still recovered so the right field is downloaded.
        assert result.image_ids == [900, 901, 902]

    def test_all_zero_rows_fall_back_to_zero_shift(self) -> None:
        df = pd.DataFrame(
            [(4130, "A1", 0, 900, 0, 0)],
            columns=["plate", "well", "sample", "image_id", "x", "y"],
        )
        a = PlateAlignment(4127, "sample_alignment", df)
        result = a.shifts_for_well(4130, "A1", n_fields=1)
        np.testing.assert_array_equal(result.shifts, np.array([[0, 0]]))

    def test_float_means_are_rounded_to_int(self) -> None:
        a = PlateAlignment(4127, "alignment", _per_well_df())
        result = a.shifts_for_well(4130, "A1", n_fields=1)
        assert result.shifts.dtype.kind == "i"

    def test_unknown_well_raises(self) -> None:
        a = PlateAlignment(4127, "sample_alignment", _per_field_df())
        with pytest.raises(AlignmentError, match="no alignment for well"):
            a.shifts_for_well(4130, "H12", n_fields=1)

    def test_shifts_shape_always_matches_n_fields(self) -> None:
        a = PlateAlignment(4127, "sample_alignment", _per_field_df())
        for n in (1, 3, 9):
            assert a.shifts_for_well(4130, "A1", n_fields=n).shifts.shape == (
                n,
                2,
            )
