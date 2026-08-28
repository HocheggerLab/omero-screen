"""Tests for label-stitch dispatch in ``_welldata_widget._display_plate``.

The dispatch routes plates through ``stitch_labels_from_offsets``
(merge_labels overlap fusion).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omero_screen_napari import _welldata_widget as wd_widget
from omero_screen_napari.omero_data import OmeroData


@pytest.fixture
def stitched_two_field_omero_data() -> OmeroData:
    """Minimal OmeroData with two fields, one well, valid positions."""
    od = OmeroData()
    od.well_id_list = [1]
    od.image_index = [0, 1]
    od.well_pos_list = ["A1"]
    od.image_positions = [(0.0, 0.0), (50.0, 0.0)]
    # Image stack (N=2, Y=32, X=32, C=2)
    od.images = np.zeros((2, 32, 32, 2), dtype=np.uint16)
    # Label stack matching shape
    od.labels = np.zeros((2, 32, 32, 2), dtype=np.uint16)
    return od


@pytest.fixture
def stitch_params() -> dict:
    return {
        "stitch": True,
        "overlap_x": 14,
        "overlap_y": 14,
        "translate_x": 0,
        "translate_y": 0,
        "edge": 0,
    }


def _run_display_plate(
    omero_data: OmeroData,
    stitch_params: dict,
    *,
    stitched_lbls_canvas: np.ndarray | None = None,
    legacy_lbls_canvas: np.ndarray | None = None,
):
    """Drive ``_display_plate`` with the heavy bits mocked.

    Returns (mock stitch_labels_from_offsets, mock stitch_from_offsets).
    """
    if stitched_lbls_canvas is None:
        stitched_lbls_canvas = np.zeros((50, 50, 2), dtype=np.uint16)
    if legacy_lbls_canvas is None:
        legacy_lbls_canvas = np.zeros((50, 50, 2), dtype=np.uint16)

    viewer = MagicMock()
    with (
        patch.object(wd_widget, "omero_data", omero_data),
        patch.object(wd_widget, "_get_stitch_params", return_value=stitch_params),
        patch.object(
            wd_widget,
            "stitch_from_offsets",
            return_value=np.zeros((50, 50, 2), dtype=np.uint16),
        ) as mock_stitch_img,
        patch.object(
            wd_widget,
            "stitch_labels_from_offsets",
            return_value=legacy_lbls_canvas,
        ) as mock_legacy,
        patch.object(wd_widget, "clear_viewer_layers"),
        patch.object(wd_widget, "_display_stitched"),
    ):
        wd_widget._display_plate(viewer)
        return mock_legacy, mock_stitch_img


class TestDisplayPlateDispatch:
    """Tests all labels go through the same stitch path in the widget.

    This functionality is required to visualise the effect of changing
    the stitch parameters."""
    def test_legacy_mode_uses_stitch_labels(
        self, stitched_two_field_omero_data, stitch_params
    ):
        stitched_two_field_omero_data.label_stitched_mode = False
        stitch_labels, stitch_images = _run_display_plate(
            stitched_two_field_omero_data, stitch_params
        )
        assert stitch_labels.called, "should call stitch_labels_from_offset"
        assert stitch_images.called, "should call stitch_from_offset"

    def test_stitched_mode_uses_stitch_labels(
        self, stitched_two_field_omero_data, stitch_params
    ):
        stitched_two_field_omero_data.label_stitched_mode = True
        stitch_labels, stitch_images = _run_display_plate(
            stitched_two_field_omero_data, stitch_params
        )
        assert stitch_labels.called, "should call stitch_labels_from_offset"
        assert stitch_images.called, "should call stitch_from_offset"
