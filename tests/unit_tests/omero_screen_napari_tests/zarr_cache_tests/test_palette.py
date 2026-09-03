"""Tests for ``zarr_cache.palette``.

Distinctness is the load-bearing property. Layers are blended additively, so
two channels sharing a colour do not merely look alike -- they *sum*, and the
composite reads as one brighter channel. That presents as an intensity artefact
rather than a colour clash, which is exactly how it was first reported.
"""

from __future__ import annotations

import pytest

from omero_screen_napari.zarr_cache.palette import (
    base_name,
    channel_hex_colors,
)


class TestDistinctness:
    @pytest.mark.parametrize("n", [2, 4, 6, 10, 15, 20, 24])
    def test_all_colours_differ(self, n: int) -> None:
        names = [f"Ch{i}" for i in range(n)]
        colours = channel_hex_colors(names)
        assert len(colours) == n
        assert len(set(colours)) == n, f"collision among {n} channels"

    def test_the_reported_case_no_longer_collides(self) -> None:
        """Tub (round 1) and Y15 (round 2) rendered identically green."""
        colours = channel_hex_colors(
            ["DAPI_R1", "Tub_R1", "Y15_R2", "EdU_R2"]
        )
        assert len(set(colours)) == 4
        assert colours[1] != colours[2]

    def test_repeated_stain_across_rounds_still_differs(self) -> None:
        """Only the first nuclear channel takes blue; a later one must not."""
        colours = channel_hex_colors(["DAPI_R1", "DAPI_R2", "DAPI_R3"])
        assert len(set(colours)) == 3
        assert colours[0] == "0000FF"


class TestRoles:
    def test_nucleus_is_blue(self) -> None:
        assert channel_hex_colors(["DAPI_R1", "EdU_R1"])[0] == "0000FF"

    def test_nucleus_alias_is_blue(self) -> None:
        assert channel_hex_colors(["Hoechst", "EdU"])[0] == "0000FF"

    def test_cell_role_is_green(self) -> None:
        colours = channel_hex_colors(["DAPI_R1", "Tub_R1", "EdU_R1"])
        assert colours[0] == "0000FF"
        assert colours[1] == "00FF00"

    def test_generated_hues_avoid_the_role_colours(self) -> None:
        names = ["DAPI_R1", "Tub_R1"] + [f"Ch{i}" for i in range(18)]
        colours = channel_hex_colors(names)
        assert colours.count("0000FF") == 1
        assert colours.count("00FF00") == 1

    def test_order_is_preserved(self) -> None:
        colours = channel_hex_colors(["A", "DAPI", "B"])
        assert colours[1] == "0000FF"
        assert len(set(colours)) == 3


class TestEdgeCases:
    def test_empty(self) -> None:
        assert channel_hex_colors([]) == []

    def test_single_channel(self) -> None:
        assert len(channel_hex_colors(["DAPI"])) == 1

    def test_no_role_channels(self) -> None:
        colours = channel_hex_colors(["Foo", "Bar", "Baz"])
        assert len(set(colours)) == 3

    def test_colours_are_six_hex_digits(self) -> None:
        for colour in channel_hex_colors([f"Ch{i}" for i in range(12)]):
            assert len(colour) == 6
            int(colour, 16)

    def test_deterministic(self) -> None:
        names = [f"Ch{i}" for i in range(10)]
        assert channel_hex_colors(names) == channel_hex_colors(names)


class TestBaseName:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("DAPI_R1", "dapi"),
            ("DAPI_R12", "dapi"),
            ("DAPI", "dapi"),
            ("H2B_RFP", "h2b_rfp"),
            ("Tub_R2", "tub"),
        ],
    )
    def test_strips_round_qualifier(self, name: str, expected: str) -> None:
        assert base_name(name) == expected
