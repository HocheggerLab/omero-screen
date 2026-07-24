"""Tests for the pure assembly logic of the 4i aligned zarr builder.

Covers ``combine_cycles`` (channel union + single-nucleus + cell sourcing)
and ``crop_or_pad`` (forcing every cycle to the master extent). OMERO I/O
paths are exercised by integration testing on a real 4i plate.
"""

import numpy as np
import pytest
from omero_screen_napari.zarr_cache.aligned_builder import (
    CycleCanvas,
    combine_cycles,
    crop_or_pad,
)


def _canvas(
    channel_values: dict[str, float],
    nucleus: str | None,
    *,
    hw: tuple[int, int] = (4, 4),
    cells: float | None = None,
    tag: str = "",
) -> CycleCanvas:
    """Build a CycleCanvas with one constant-valued frame per channel."""
    h, w = hw
    names = list(channel_values)
    img = np.stack(
        [np.full((1, h, w), channel_values[n], dtype=np.float32) for n in names],
        axis=1,
    )  # (T=1, C, Y, X)
    cells_arr = (
        np.full((1, h, w), cells, dtype=np.uint32) if cells is not None else None
    )
    return CycleCanvas(img, names, nucleus, cells_arr, cycle_tag=tag)


class TestCropOrPad:
    def test_exact_size_is_identity(self) -> None:
        arr = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
        out = crop_or_pad(arr, 4, 4)
        assert out is arr

    def test_crop_keeps_top_left(self) -> None:
        arr = np.arange(6 * 6, dtype=np.float32).reshape(1, 6, 6)
        out = crop_or_pad(arr, 4, 4)
        assert out.shape == (1, 4, 4)
        np.testing.assert_array_equal(out[0], arr[0, :4, :4])

    def test_pad_fills_far_edge_with_zero(self) -> None:
        arr = np.ones((1, 2, 2), dtype=np.float32)
        out = crop_or_pad(arr, 4, 4)
        assert out.shape == (1, 4, 4)
        np.testing.assert_array_equal(out[0, :2, :2], 1.0)
        np.testing.assert_array_equal(out[0, 2:, :], 0.0)
        np.testing.assert_array_equal(out[0, :, 2:], 0.0)

    def test_preserves_leading_axes_and_dtype(self) -> None:
        arr = np.ones((1, 3, 5, 5), dtype=np.uint16)
        out = crop_or_pad(arr, 4, 4)
        assert out.shape == (1, 3, 4, 4)
        assert out.dtype == np.uint16


class TestCombineCycles:
    def test_single_master_keeps_all_channels(self) -> None:
        master = _canvas({"H": 10, "Tub": 20}, nucleus="H", cells=7)
        img, names, cells = combine_cycles([master])
        assert names == ["H", "Tub"]
        assert img.shape == (1, 2, 4, 4)
        np.testing.assert_array_equal(img[:, 0], 10.0)
        np.testing.assert_array_equal(img[:, 1], 20.0)
        assert cells is not None
        np.testing.assert_array_equal(cells, 7)

    def test_repeat_nucleus_dropped_by_role_even_if_named_differently(
        self,
    ) -> None:
        """Master nucleus 'H', repeat nucleus 'DAPI' → only one nucleus kept."""
        master = _canvas({"H": 10, "Tub": 20}, nucleus="H")
        repeat = _canvas({"DAPI": 30, "EdU": 40}, nucleus="DAPI")
        img, names, _ = combine_cycles([master, repeat])
        assert names == ["H", "Tub", "EdU"]  # DAPI (repeat nucleus) dropped
        np.testing.assert_array_equal(img[:, 0], 10.0)  # H
        np.testing.assert_array_equal(img[:, 1], 20.0)  # Tub
        np.testing.assert_array_equal(img[:, 2], 40.0)  # EdU

    def test_duplicate_named_channel_deduped(self) -> None:
        master = _canvas({"H": 10, "Tub": 20}, nucleus="H")
        repeat = _canvas({"DAPI": 30, "Tub": 99}, nucleus="DAPI")
        _, names, _ = combine_cycles([master, repeat])
        assert names == ["H", "Tub"]  # repeat Tub deduped, DAPI dropped

    def test_cells_taken_from_repeat_when_master_has_none(self) -> None:
        master = _canvas({"H": 10, "Tub": 20}, nucleus="H", cells=None)
        repeat = _canvas({"DAPI": 30, "EdU": 40}, nucleus="DAPI", cells=5)
        _, _, cells = combine_cycles([master, repeat])
        assert cells is not None
        np.testing.assert_array_equal(cells, 5)

    def test_master_cells_preferred_over_repeat(self) -> None:
        master = _canvas({"H": 10}, nucleus="H", cells=1)
        repeat = _canvas({"EdU": 40}, nucleus=None, cells=9)
        _, _, cells = combine_cycles([master, repeat])
        assert cells is not None
        np.testing.assert_array_equal(cells, 1)  # master wins

    def test_no_cells_anywhere_returns_none(self) -> None:
        master = _canvas({"H": 10}, nucleus="H", cells=None)
        repeat = _canvas({"EdU": 40}, nucleus=None, cells=None)
        _, _, cells = combine_cycles([master, repeat])
        assert cells is None

    def test_repeat_canvas_cropped_to_master_extent(self) -> None:
        """A larger repeat canvas is cropped to the master extent on combine."""
        master = _canvas({"H": 10}, nucleus="H", hw=(4, 4))
        repeat = _canvas({"EdU": 40}, nucleus=None, hw=(6, 6), cells=3)
        img, names, cells = combine_cycles([master, repeat])
        assert img.shape == (1, 2, 4, 4)
        assert cells is not None and cells.shape == (1, 4, 4)

    def test_empty_cycles_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            combine_cycles([])

    def test_keep_all_nuclei_renames_and_keeps_each(self) -> None:
        """Debug mode keeps every cycle's nucleus, uniquely renamed."""
        master = _canvas({"DAPI": 10, "Tub": 20}, nucleus="DAPI", tag="4127")
        repeat = _canvas({"DAPI": 30, "EdU": 40}, nucleus="DAPI", tag="4130")
        img, names, _ = combine_cycles(
            [master, repeat], keep_all_nuclei=True
        )
        # Both DAPIs survive, renamed per cycle; markers unchanged.
        assert names == ["DAPI_4127", "Tub", "DAPI_4130", "EdU"]
        np.testing.assert_array_equal(img[:, 0], 10.0)  # master DAPI
        np.testing.assert_array_equal(img[:, 2], 30.0)  # repeat DAPI
        np.testing.assert_array_equal(img[:, 3], 40.0)  # repeat EdU

    def test_default_still_drops_repeat_nucleus(self) -> None:
        """Without the debug flag, repeat nuclei are still dropped."""
        master = _canvas({"DAPI": 10, "Tub": 20}, nucleus="DAPI", tag="4127")
        repeat = _canvas({"DAPI": 30, "EdU": 40}, nucleus="DAPI", tag="4130")
        _, names, _ = combine_cycles([master, repeat])
        assert names == ["DAPI", "Tub", "EdU"]


class TestAlignedNamespaceIsolation:
    """The aligned namespace must not collide with the plain plate cache.

    A master plate can have both a plain stitched cache and a 4i aligned
    assembly under the same plate id; they live in separate directories with
    separate registries (the ``root`` param), so neither clobbers the other.
    """

    def test_write_read_isolated_by_root(self, synth_well_data) -> None:
        from omero_screen_napari.zarr_cache import reader
        from omero_screen_napari.zarr_cache.paths import (
            aligned_zarr_root,
            plate_zarr_path,
        )
        from omero_screen_napari.zarr_cache.writer import PlateZarrWriter

        pid = 1234
        aligned = aligned_zarr_root()
        img2, nuc, cell = synth_well_data(c=2)
        img3, _, _ = synth_well_data(c=3)

        with PlateZarrWriter(pid, "Plain", ["a", "b"], 0.65, 1) as w:
            w.ensure_plate(["A1", "A2"])
            w.write_well("A1", img2, nuc, cell)
        with PlateZarrWriter(
            pid, "Aligned", ["a", "b", "c"], 0.65, 1, root=aligned
        ) as w:
            w.ensure_plate(["A1"])
            w.write_well("A1", img3, nuc, cell)

        # Separate directories.
        assert plate_zarr_path(pid) != plate_zarr_path(pid, root=aligned)
        # Each namespace reports its own metadata — no clobber.
        assert reader.plate_info(pid)["plate_name"] == "Plain"
        assert reader.plate_info(pid, root=aligned)["plate_name"] == "Aligned"
        assert reader.plate_info(pid)["channel_names"] == ["a", "b"]
        assert reader.plate_info(pid, root=aligned)["channel_names"] == [
            "a",
            "b",
            "c",
        ]

    def test_registry_isolated_by_root(self) -> None:
        from omero_screen_napari.zarr_cache import registry
        from omero_screen_napari.zarr_cache.paths import aligned_zarr_root

        aligned = aligned_zarr_root()
        registry.upsert(registry.ZarrPlateEntry(plate_id=1, plate_name="P"))
        registry.upsert(
            registry.ZarrPlateEntry(plate_id=1, plate_name="A"), root=aligned
        )
        plain = registry.load_registry()
        algn = registry.load_registry(root=aligned)
        assert plain[1].plate_name == "P"
        assert algn[1].plate_name == "A"
