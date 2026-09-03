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


# ---------------------------------------------------------------------- #
# Per-well canvas sizes                                                  #
# ---------------------------------------------------------------------- #


def _build_mismatched_plate(plate_id, synth_well_data):
    """Two wells whose canvases differ, as a dropped edge field produces.

    A field with no stage position (autofocus failure) drops out of the
    well's offset bounding box; if it held the extreme offset the canvas
    shrinks by one shear step. Here B2 is 3 px narrower and 3 px shorter.
    """
    image_a, nuc_a, cell_a = synth_well_data(h=256, w=256, c=2)
    image_b = image_a[:, :, :253, :253].copy()
    nuc_b = nuc_a[:, :253, :253].copy()
    cell_b = cell_a[:, :253, :253].copy()
    w = PlateZarrWriter(
        plate_id=plate_id,
        plate_name="t",
        channel_names=["DAPI", "Tub"],
        pixel_size_um=0.65,
        n_timepoints=1,
    )
    w.ensure_plate(
        all_wells=["A1", "B2"],
        well_metadata={"A1": {}, "B2": {}},
    )
    w.write_well("A1", image_a, nuc_a, cell_a)
    w.write_well("B2", image_b, nuc_b, cell_b)
    return plate_zarr_path(plate_id)


def test_mismatched_canvases_stack(synth_well_data):
    # Regression: da.stack raised "Stacked arrays must have the same
    # shape" when one well's canvas was a few pixels smaller.
    _build_mismatched_plate(310, synth_well_data)
    v = _mock_viewer()
    loaded = load_plate_to_viewer(v, 310, well_pos_input="A1, B2")

    assert loaded == ["A1", "B2"]
    pyramid = v.add_image.call_args_list[0].args[0]
    # Padded to the larger well on both spatial axes.
    assert pyramid[0].shape[0] == 2
    assert pyramid[0].shape[-2:] == (256, 256)


def test_mismatched_canvases_pad_labels(synth_well_data):
    _build_mismatched_plate(311, synth_well_data)
    v = _mock_viewer()
    load_plate_to_viewer(v, 311, well_pos_input="A1, B2")

    pyramid = v.add_labels.call_args_list[0].args[0]
    assert pyramid[0].shape[0] == 2
    assert pyramid[0].shape[-2:] == (256, 256)


def test_pad_is_zero_filled(synth_well_data):
    _build_mismatched_plate(312, synth_well_data)
    v = _mock_viewer()
    load_plate_to_viewer(v, 312, well_pos_input="A1, B2")

    pyramid = v.add_image.call_args_list[0].args[0]
    padded = np.asarray(pyramid[0][1, 0, :, 253:])
    assert padded.size and not padded.any()


def test_pad_marked_as_region(synth_well_data):
    _build_mismatched_plate(313, synth_well_data)
    v = _mock_viewer()
    load_plate_to_viewer(v, 313, well_pos_input="A1, B2")

    v.add_shapes.assert_called_once()
    captions = v.add_shapes.call_args.kwargs["text"]["string"]
    assert any("B2: padded" in c for c in captions)
    assert not any("A1: padded" in c for c in captions)


def test_matching_canvases_add_no_pad_regions(synth_well_data):
    _build_two_well_plate(314, synth_well_data)
    v = _mock_viewer()
    load_plate_to_viewer(v, 314, well_pos_input="A1, B2")

    v.add_shapes.assert_not_called()


def test_canvas_mismatch_logged(synth_well_data, caplog):
    _build_mismatched_plate(315, synth_well_data)
    v = _mock_viewer()
    load_plate_to_viewer(v, 315, well_pos_input="A1, B2")

    assert any("padding to 256x256" in rec.message for rec in caplog.records)


def test_single_well_is_not_padded(synth_well_data):
    # One well is its own canvas; nothing to reconcile.
    _build_mismatched_plate(316, synth_well_data)
    v = _mock_viewer()
    load_plate_to_viewer(v, 316, well_pos_input="B2")

    pyramid = v.add_image.call_args_list[0].args[0]
    assert pyramid[0].shape[-2:] == (253, 253)


def test_overlay_text_includes_metadata(synth_well_data):
    _build_two_well_plate(304, synth_well_data)
    v = _mock_viewer()
    load_plate_to_viewer(v, 304, well_pos_input="A1")
    # Static overlay set for single-well load
    text = v.text_overlay.text
    # Caption built from metadata bits.
    assert "A1" in text or "U2OS" in text


# ----------------------------------------------------------------------
# Contrast and default visibility
#
# These use a real napari ViewerModel rather than a MagicMock. A mock records
# whatever is assigned to it, so it happily "passes" regardless of what napari
# would actually do with the layer -- it cannot show that the slider ends up
# spanning the full uint16 range, which is the property that matters here.
# ----------------------------------------------------------------------


def _real_viewer():
    from napari.components import ViewerModel

    return ViewerModel()


def _image_layers(viewer):
    return [
        layer
        for layer in viewer.layers
        if hasattr(layer, "contrast_limits_range")
    ]


def test_contrast_range_spans_full_uint16(synth_well_data):
    """The slider must reach the whole dtype range, not just the data range."""
    _build_two_well_plate(320, synth_well_data)
    v = _real_viewer()
    load_plate_to_viewer(v, 320, well_pos_input="A1")
    layers = _image_layers(v)
    assert layers
    for layer in layers:
        assert tuple(layer.contrast_limits_range) == (0, 65535)


def test_contrast_limits_sit_inside_the_range(synth_well_data):
    _build_two_well_plate(321, synth_well_data)
    v = _real_viewer()
    load_plate_to_viewer(v, 321, well_pos_input="A1")
    for layer in _image_layers(v):
        lo, hi = layer.contrast_limits
        assert 0 <= lo < hi <= 65535
        # Percentiles, not the full span: a starting view, not everything.
        assert (lo, hi) != (0, 65535)


def test_channels_get_distinct_colormaps(synth_well_data):
    """Two layers sharing a colormap would sum under additive blending."""
    _build_two_well_plate(323, synth_well_data)
    v = _real_viewer()
    load_plate_to_viewer(v, 323, well_pos_input="A1")
    colors = [
        tuple(map(tuple, layer.colormap.colors))
        for layer in _image_layers(v)
    ]
    assert len(colors) == 2
    assert colors[0] != colors[1]


def test_ordinary_plate_shows_every_channel(synth_well_data):
    """No rounds block: behaviour is unchanged, everything visible."""
    _build_two_well_plate(324, synth_well_data)
    v = _real_viewer()
    load_plate_to_viewer(v, 324, well_pos_input="A1")
    assert all(layer.visible for layer in _image_layers(v))


def test_channel_contrast_uses_percentiles():
    from omero_screen_napari.zarr_cache.display import _channel_contrast

    # Mid-grey with a single hot pixel: min/max would give (1000, 65535) and
    # wash the image out; percentiles must ignore the outlier.
    data = np.full((100, 10), 1000, dtype=np.uint16)
    data[0, 0] = 65535
    lo, hi = _channel_contrast(data)
    assert lo == 1000
    assert hi < 65535


def test_channel_contrast_flat_channel_falls_back_to_full_range():
    from omero_screen_napari.zarr_cache.display import _channel_contrast

    lo, hi = _channel_contrast(np.zeros((50, 50), dtype=np.uint16))
    assert (lo, hi) == (0, 65535)


def test_only_the_master_round_is_visible_on_a_4i_plate(synth_well_data):
    """A dozen additive layers at once renders a saturated white image."""
    from omero_screen_napari.zarr_cache.rounds import (
        RoundGroup,
        build_channel_plan,
    )

    names, attrs, _ = build_channel_plan(
        RoundGroup(330, (331, 332)),
        {
            330: {"DAPI": "0", "Tub": "1"},
            331: {"DAPI": "0", "p21": "1"},
            332: {"DAPI": "0", "TP53": "1"},
        },
    )
    image, nuc, _ = synth_well_data(c=len(names), h=256, w=256)
    w = PlateZarrWriter(330, "4i", names, 0.5, 1, rounds=attrs)
    with w:
        w.ensure_plate(all_wells=["A1"])
        w.write_well("A1", image, nuc)

    v = _real_viewer()
    load_plate_to_viewer(v, 330, well_pos_input="A1")
    visible = {
        layer.name: layer.visible
        for layer in _image_layers(v)
    }
    assert visible == {
        "DAPI_R1": True,
        "Tub_R1": True,
        "p21_R2": False,
        "TP53_R3": False,
    }


def test_4i_layers_still_get_the_full_range(synth_well_data):
    """Hidden layers must still be usable once switched on."""
    from omero_screen_napari.zarr_cache.rounds import (
        RoundGroup,
        build_channel_plan,
    )

    names, attrs, _ = build_channel_plan(
        RoundGroup(340, (341,)),
        {340: {"DAPI": "0", "Tub": "1"}, 341: {"DAPI": "0", "p21": "1"}},
    )
    image, nuc, _ = synth_well_data(c=len(names), h=256, w=256)
    w = PlateZarrWriter(340, "4i", names, 0.5, 1, rounds=attrs)
    with w:
        w.ensure_plate(all_wells=["A1"])
        w.write_well("A1", image, nuc)

    v = _real_viewer()
    load_plate_to_viewer(v, 340, well_pos_input="A1")
    for layer in _image_layers(v):
        assert tuple(layer.contrast_limits_range) == (0, 65535)
