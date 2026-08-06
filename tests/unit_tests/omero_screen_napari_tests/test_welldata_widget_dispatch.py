"""Tests for label-stitch dispatch in ``_welldata_widget._display_plate``.

The dispatch routes stitched-mode plates through ``recompose_split_labels``
(lossless non-zero copy) and legacy plates through ``stitch_labels_from_positions``
(merge_labels overlap fusion).

Deprecated: The stitched-mode plates use the zarr cache pathway.
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

    Returns (mock_recompose, mock stitch_labels_from_offsets, mock stitch_from_offsets).
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
        # No longer imported by the widget
        # patch.object(
        #     wd_widget,
        #     "recompose_split_labels",
        #     return_value=stitched_lbls_canvas,
        # ) as mock_recompose,
        patch.object(
            wd_widget,
            "stitch_labels_from_offsets",
            return_value=legacy_lbls_canvas,
        ) as mock_legacy,
        patch.object(wd_widget, "clear_viewer_layers"),
        patch.object(wd_widget, "_display_stitched"),
    ):
        wd_widget._display_plate(viewer)
        # Dummy this until this deprecated functionality is removed, or reinstated
        mock_recompose = MagicMock()
        return mock_recompose, mock_legacy, mock_stitch_img


class TestDisplayPlateDispatch:
    @pytest.mark.skip(reason="stitched labels use the zarr cache pathway and images are not loaded")
    def test_stitched_mode_uses_recompose(
        self, stitched_two_field_omero_data, stitch_params
    ):
        stitched_two_field_omero_data.label_stitched_mode = True
        recompose, legacy, _ = _run_display_plate(
            stitched_two_field_omero_data, stitch_params
        )
        assert recompose.called, "stitched mode should call recompose_split_labels"
        assert not legacy.called, (
            "stitched mode must NOT call stitch_labels_from_positions"
        )

    def test_legacy_mode_uses_stitch_labels(
        self, stitched_two_field_omero_data, stitch_params
    ):
        stitched_two_field_omero_data.label_stitched_mode = False
        recompose, stitch_labels, stitch_images = _run_display_plate(
            stitched_two_field_omero_data, stitch_params
        )
        assert stitch_labels.called, "legacy mode should call stitch_labels_from_offset"
        assert stitch_images.called, "legacy mode should call stitch_from_offset"
        assert not recompose.called, (
            "legacy mode must NOT call recompose_split_labels"
        )

    @pytest.mark.skip(reason="stitched labels use the zarr cache pathway and images are not loaded")
    def test_stitched_mode_passes_tile_dimensions(
        self, stitched_two_field_omero_data, stitch_params
    ):
        """Recompose call receives tile_h, tile_w taken from per-field label shape."""
        stitched_two_field_omero_data.label_stitched_mode = True
        recompose, _, _ = _run_display_plate(
            stitched_two_field_omero_data, stitch_params
        )
        kwargs = recompose.call_args.kwargs
        assert kwargs["tile_h"] == 32
        assert kwargs["tile_w"] == 32
        assert kwargs["overlap_x"] == 14
        assert kwargs["overlap_y"] == 14
