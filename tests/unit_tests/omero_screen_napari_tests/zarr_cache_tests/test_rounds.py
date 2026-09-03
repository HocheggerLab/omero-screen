"""Tests for ``zarr_cache.rounds`` -- resolving a plate into its 4i group.

The channel plan is the part worth pinning down. Names must be unique across
rounds because ``display._populate_singleton`` builds ``{name: index}`` from
them, and a duplicate would silently collapse that dict rather than fail --
breaking gallery and classifier channel lookups in a way that surfaces far from
the cause.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from omero_screen_napari.zarr_cache.alignment import PlateAlignment
from omero_screen_napari.zarr_cache.rounds import (
    RoundGroup,
    build_channel_plan,
    channel_indices_for_plate,
    resolve_round_group,
)


def _alignment(*plate_ids: int) -> PlateAlignment:
    rows = [(pid, "A1", 0, 900 + pid, 5, -3) for pid in plate_ids]
    df = pd.DataFrame(
        rows,
        columns=["plate", "well", "sample", "image_id", "x", "y"],
    )
    return PlateAlignment(4127, "sample_alignment", df)


def _group(*members: int) -> RoundGroup:
    return RoundGroup(
        master_plate_id=4127,
        member_plate_ids=members,
        alignment=_alignment(*members) if members else None,
    )


class TestRoundGroupProperties:
    def test_plate_ids_put_master_first(self) -> None:
        assert _group(4130, 4131).plate_ids == (4127, 4130, 4131)

    def test_n_rounds_counts_the_master(self) -> None:
        assert _group(4130, 4131).n_rounds == 3
        assert _group().n_rounds == 1

    def test_is_master_requires_members(self) -> None:
        assert _group(4130).is_master
        assert not _group().is_master

    def test_buildable_requires_members_and_no_blockers(self) -> None:
        assert _group(4130).buildable
        assert not _group().buildable
        blocked = RoundGroup(4127, (4130,), None, ("not stitched",))
        assert not blocked.buildable


_TWO_ROUND = {
    4127: {"DAPI": "0", "Tub": "1"},
    4130: {"DAPI": "0", "EdU": "1"},
}


class TestBuildChannelPlan:
    def test_repeated_nuclear_stain_is_dropped(self) -> None:
        """Only the master's DAPI is kept: the later rounds add nothing."""
        names, _, _ = build_channel_plan(_group(4130), _TWO_ROUND)
        assert names == ["DAPI_R1", "Tub_R1", "EdU_R2"]

    def test_dropped_channels_are_not_downloaded(self) -> None:
        """The third return value is what each round actually reads."""
        _, _, load = build_channel_plan(_group(4130), _TWO_ROUND)
        assert load[4127] == {"DAPI": "0", "Tub": "1"}
        assert load[4130] == {"EdU": "1"}

    def test_include_redundant_keeps_them(self) -> None:
        names, attrs, load = build_channel_plan(
            _group(4130), _TWO_ROUND, include_redundant=True
        )
        assert names == ["DAPI_R1", "Tub_R1", "DAPI_R2", "EdU_R2"]
        assert load[4130] == {"DAPI": "0", "EdU": "1"}
        by_name = {
            (e["name"], e["round"]): e["redundant"] for e in attrs["channels"]
        }
        assert by_name[("DAPI", 1)] is False
        assert by_name[("DAPI", 2)] is True

    def test_names_are_unique(self) -> None:
        """The property display._populate_singleton depends on."""
        names, _, _ = build_channel_plan(
            _group(4130, 4131),
            {
                4127: {"DAPI": "0", "Tub": "1"},
                4130: {"DAPI": "0", "EdU": "1"},
                4131: {"DAPI": "0", "H3P": "1"},
            },
        )
        assert len(names) == len(set(names))

    def test_master_channels_are_suffixed_too(self) -> None:
        names, _, _ = build_channel_plan(
            _group(4130), {4127: {"DAPI": "0"}, 4130: {"EdU": "0"}}
        )
        assert names[0] == "DAPI_R1"

    def test_ordered_by_index_not_dict_order(self) -> None:
        names, _, _ = build_channel_plan(
            _group(),
            {4127: {"Tub": "2", "DAPI": "0", "EdU": "1"}},
        )
        assert names == ["DAPI_R1", "EdU_R1", "Tub_R1"]

    def test_position_is_contiguous_after_dropping(self) -> None:
        """Dropping a channel must not leave a gap in the round's positions."""
        _, attrs, _ = build_channel_plan(
            _group(4130),
            {
                4127: {"DAPI": "0", "Tub": "1"},
                4130: {"DAPI": "0", "EdU": "1", "H3P": "2"},
            },
        )
        by_round: dict[int, list[int]] = {}
        for e in attrs["channels"]:
            by_round.setdefault(e["round"], []).append(e["position"])
        assert by_round[1] == [0, 1]
        assert by_round[2] == [0, 1]

    def test_flat_index_is_recorded(self) -> None:
        _, attrs, _ = build_channel_plan(
            _group(4130),
            {4127: {"DAPI": "0", "Tub": "1"}, 4130: {"EdU": "0"}},
        )
        assert [e["index"] for e in attrs["channels"]] == [0, 1, 2]

    def test_attrs_record_the_group_and_convention(self) -> None:
        _, attrs, _ = build_channel_plan(
            _group(4130, 4131),
            {
                4127: {"DAPI": "0"},
                4130: {"EdU": "0"},
                4131: {"H3P": "0"},
            },
        )
        assert attrs["master_plate_id"] == 4127
        assert attrs["member_plate_ids"] == [4130, 4131]
        assert attrs["plate_ids"] == [4127, 4130, 4131]
        assert attrs["alignment_source"] == "sample_alignment"
        # A reader must never have to guess which way the shift goes.
        assert attrs["shift_convention"] == "master = restain - (x, y)"
        assert attrs["include_redundant"] is False

    def test_single_round_group(self) -> None:
        names, attrs, _ = build_channel_plan(
            _group(), {4127: {"DAPI": "0", "Tub": "1"}}
        )
        assert names == ["DAPI_R1", "Tub_R1"]
        assert attrs["member_plate_ids"] == []

    def test_missing_round_channel_data_raises(self) -> None:
        with pytest.raises(KeyError):
            build_channel_plan(_group(4130), {4127: {"DAPI": "0"}})


class TestChannelIndicesForPlate:
    def test_selects_one_round(self) -> None:
        _, attrs, _ = build_channel_plan(_group(4130), _TWO_ROUND)
        assert channel_indices_for_plate(attrs, 4127) == [0, 1]
        assert channel_indices_for_plate(attrs, 4130) == [2]

    def test_unknown_plate_is_empty(self) -> None:
        _, attrs, _ = build_channel_plan(_group(), {4127: {"DAPI": "0"}})
        assert channel_indices_for_plate(attrs, 9999) == []


class TestResolveRoundGroup:
    def _conn(self, plate: MagicMock | None) -> MagicMock:
        conn = MagicMock()
        conn.getObject.return_value = plate
        return conn

    def test_plate_without_alignment_is_not_a_master(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.rounds.has_alignment",
            lambda plate: False,
        )
        group = resolve_round_group(self._conn(MagicMock()), 4200)
        assert not group.is_master
        assert group.blockers == ()

    def test_missing_plate_reports_a_blocker(self) -> None:
        group = resolve_round_group(self._conn(None), 4200)
        assert not group.buildable
        assert "not found" in group.blockers[0]

    def test_unusable_alignment_still_reports_a_blocker(
        self, monkeypatch
    ) -> None:
        """A pre-schema per-well table: still a master, but not buildable."""
        from omero_screen_napari.zarr_cache.alignment import AlignmentError

        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.rounds.has_alignment",
            lambda plate: True,
        )

        def _raise(conn, plate_id):
            raise AlignmentError("x/y transposed")

        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.rounds.load_alignment", _raise
        )
        group = resolve_round_group(self._conn(MagicMock()), 4127)
        assert not group.buildable
        assert "transposed" in group.blockers[0]

    def test_unstitched_round_blocks_the_build(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.rounds.has_alignment",
            lambda plate: True,
        )
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.rounds.load_alignment",
            lambda conn, plate_id: _alignment(4130),
        )
        monkeypatch.setattr(
            "omero_screen_napari.plate_cache.detect_label_stitched_mode",
            lambda conn, pid: pid != 4130,
        )
        group = resolve_round_group(self._conn(MagicMock()), 4127)
        assert group.is_master
        assert not group.buildable
        assert "restain round plate 4130 is not stitched" in group.blockers

    def test_all_stitched_is_buildable(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.rounds.has_alignment",
            lambda plate: True,
        )
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.rounds.load_alignment",
            lambda conn, plate_id: _alignment(4130, 4131),
        )
        monkeypatch.setattr(
            "omero_screen_napari.plate_cache.detect_label_stitched_mode",
            lambda conn, pid: True,
        )
        group = resolve_round_group(self._conn(MagicMock()), 4127)
        assert group.buildable
        assert group.member_plate_ids == (4130, 4131)

    def test_check_stitched_false_skips_the_queries(self, monkeypatch) -> None:
        """The dialog only needs the badge, not the full buildability check."""
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.rounds.has_alignment",
            lambda plate: True,
        )
        monkeypatch.setattr(
            "omero_screen_napari.zarr_cache.rounds.load_alignment",
            lambda conn, plate_id: _alignment(4130),
        )

        def _boom(conn, pid):
            raise AssertionError("should not be called")

        monkeypatch.setattr(
            "omero_screen_napari.plate_cache.detect_label_stitched_mode", _boom
        )
        group = resolve_round_group(
            self._conn(MagicMock()), 4127, check_stitched=False
        )
        assert group.is_master
