"""Display: well parsing + viewer wiring against a mock napari Viewer."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from omero_screen_napari.zarr_cache import PlateZarrWriter, plate_zarr_path
from omero_screen_napari.zarr_cache.display import (
    _resolve_well_list,
    load_plate_to_viewer,
)


# ---------------------------------------------------------------------- #
# Well-list parsing                                                      #
# ---------------------------------------------------------------------- #


def test_resolve_all_returns_sorted_available():
    assert _resolve_well_list("All", ["B1", "A1", "C2"]) == ["A1", "B1", "C2"]


def test_resolve_empty_string_returns_all():
    assert _resolve_well_list("", ["A1", "A2"]) == ["A1", "A2"]


def test_resolve_comma_list_filters_to_available():
    assert _resolve_well_list("A1, B2, C3", ["A1", "B2"]) == ["A1", "B2"]


def test_resolve_malformed_skipped():
    assert _resolve_well_list("A1, foo, B2", ["A1", "B2"]) == ["A1", "B2"]


def test_resolve_missing_logged_as_warning(caplog):
    _resolve_well_list("A1, Z9", ["A1"])
    assert any("not present" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------- #
# load_plate_to_viewer                                                   #
# ---------------------------------------------------------------------- #


def _build_two_well_plate(plate_id, synth_well_data):
    image, nuc, cell = synth_well_data(h=256, w=256, c=2)
    w = PlateZarrWriter(
        plate_id=plate_id,
        plate_name="t",
        channel_names=["DAPI", "Tub"],
        pixel_size_um=0.65,
        n_timepoints=1,
    )
    w.ensure_plate(
        all_wells=["A1", "B2"],
        well_metadata={
            "A1": {"cell_line": "U2OS", "condition": "ctrl"},
            "B2": {"cell_line": "U2OS", "condition": "drug"},
        },
    )
    w.write_well("A1", image, nuc, cell)
    w.write_well("B2", image, nuc, cell)
    return plate_zarr_path(plate_id)


def _mock_viewer():
    """Return a MagicMock that fakes napari.Viewer well enough to test
    layer creation, slider hooks, and the text overlay.
    """
    v = MagicMock()
    v.layers = MagicMock()
    v.layers.__len__ = MagicMock(return_value=0)
    v.dims = MagicMock()
    v.dims.current_step = (0, 0, 0)
    v.text_overlay = MagicMock()
    return v


def test_load_single_well_adds_layers(synth_well_data):
    _build_two_well_plate(300, synth_well_data)
    v = _mock_viewer()
    loaded = load_plate_to_viewer(v, 300, well_pos_input="A1")
    assert loaded == ["A1"]
    # 2 image channels + nuclei + cells = 4 layer additions.
    assert v.add_image.call_count == 2
    assert v.add_labels.call_count == 2
    # Single-well: no slider hook on dims.current_step.
    v.dims.events.current_step.connect.assert_not_called()


def test_load_multi_well_stacks_along_well_axis(synth_well_data):
    _build_two_well_plate(301, synth_well_data)
    v = _mock_viewer()
    loaded = load_plate_to_viewer(v, 301, well_pos_input="A1, B2")
    assert loaded == ["A1", "B2"]
    # Each image add_call's first positional arg is the pyramid list.
    first_call = v.add_image.call_args_list[0]
    pyramid = first_call.args[0]
    # Top of the pyramid should have a leading well axis = 2.
    assert pyramid[0].shape[0] == 2
    # Slider hook attached for the well axis.
    v.dims.events.current_step.connect.assert_called_once()


def test_load_all_finds_every_cached_well(synth_well_data):
    _build_two_well_plate(302, synth_well_data)
    v = _mock_viewer()
    loaded = load_plate_to_viewer(v, 302, well_pos_input="All")
    assert sorted(loaded) == ["A1", "B2"]


def test_load_unknown_well_returns_empty(synth_well_data):
    _build_two_well_plate(303, synth_well_data)
    v = _mock_viewer()
    loaded = load_plate_to_viewer(v, 303, well_pos_input="Z9")
    assert loaded == []
    v.add_image.assert_not_called()


def test_overlay_text_includes_metadata(synth_well_data):
    _build_two_well_plate(304, synth_well_data)
    v = _mock_viewer()
    load_plate_to_viewer(v, 304, well_pos_input="A1")
    # Static overlay set for single-well load
    text = v.text_overlay.text
    # Caption built from metadata bits.
    assert "A1" in text or "U2OS" in text
