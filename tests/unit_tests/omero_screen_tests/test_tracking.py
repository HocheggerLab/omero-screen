"""Unit tests for the Trackastra wrapper in ``omero_screen.tracking``.

Trackastra itself is mocked so these tests are fast and offline: we verify the
no-op gate, input validation, the parent-map construction from the CTC table,
mask-dtype preservation, and the dataframe column derivation. The actual model
inference is exercised by the integration run on a live timelapse plate.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from omero_screen.tracking import (
    TrackingResult,
    add_track_columns,
    track_nucleus_mask,
)


def _ctc_dataframe() -> pd.DataFrame:
    """A CTC track table with one founder track that divides into two.

    Track 1 spans the whole movie (founder). Track 2 is a founder that ends at
    t1 and divides into daughters 3 and 4 at t2 (both parent == 2).
    """
    return pd.DataFrame(
        {
            "label": [1, 2, 3, 4],
            "t1": [0, 0, 2, 2],
            "t2": [2, 1, 2, 2],
            "parent": [0, 0, 2, 2],
        }
    )


class TestTrackNucleusMask:
    def test_single_timepoint_is_noop(self) -> None:
        """T == 1 must return the mask untouched and never call the model."""
        mask = np.array([[[0, 1], [2, 0]]], dtype=np.uint16)  # (1, 2, 2)
        imgs = mask.astype(np.float32)
        model = MagicMock()

        result = track_nucleus_mask(imgs, mask, model)

        assert isinstance(result, TrackingResult)
        assert result.parent_map == {}
        np.testing.assert_array_equal(result.nucleus_mask, mask)
        model.track.assert_not_called()

    def test_invalid_mode_raises(self) -> None:
        mask = np.zeros((2, 4, 4), dtype=np.uint16)
        with pytest.raises(ValueError, match="Unknown tracking mode"):
            track_nucleus_mask(mask.astype(np.float32), mask, MagicMock(), mode="bogus")

    def test_shape_mismatch_raises(self) -> None:
        imgs = np.zeros((2, 4, 4), dtype=np.float32)
        mask = np.zeros((3, 4, 4), dtype=np.uint16)
        with pytest.raises(ValueError, match="same shape"):
            track_nucleus_mask(imgs, mask, MagicMock())

    def test_parent_map_and_relabel(self) -> None:
        """A mocked division yields the right parent map and relabelled mask."""
        imgs = np.zeros((3, 4, 4), dtype=np.float32)
        mask = np.ones((3, 4, 4), dtype=np.uint16)
        model = MagicMock()
        model.track.return_value = ("fake_graph", None)
        relabelled = np.full((3, 4, 4), 3, dtype=np.int32)  # different dtype

        with patch(
            "trackastra.tracking.graph_to_ctc",
            return_value=(_ctc_dataframe(), relabelled),
        ) as mock_ctc:
            result = track_nucleus_mask(imgs, mask, model, mode="greedy")

        model.track.assert_called_once()
        mock_ctc.assert_called_once()
        # Founders map to 0; daughters 3 and 4 point back to parent 2.
        assert result.parent_map == {1: 0, 2: 0, 3: 2, 4: 2}
        # Relabelled mask is cast back to the input mask dtype.
        assert result.nucleus_mask.dtype == mask.dtype
        np.testing.assert_array_equal(result.nucleus_mask, relabelled)


class TestAddTrackColumns:
    def test_columns_derived_from_label(self) -> None:
        """track_id == label; parent columns come from the parent map."""
        df = pd.DataFrame({"label": [1, 2, 3, 4], "area_nucleus": [10, 11, 5, 6]})
        parent_map = {1: 0, 2: 0, 3: 2, 4: 2}

        add_track_columns(df, parent_map)

        assert list(df["track_id"]) == [1, 2, 3, 4]
        assert list(df["track_id_raw"]) == [1, 2, 3, 4]
        assert list(df["parent_track_id"]) == [0, 0, 2, 2]
        assert list(df["parent_track_id_raw"]) == [0, 0, 2, 2]

    def test_missing_track_in_map_defaults_to_zero(self) -> None:
        """A label absent from the parent map becomes a founder (parent 0)."""
        df = pd.DataFrame({"label": [7]})
        add_track_columns(df, parent_map={})
        assert list(df["parent_track_id"]) == [0]
        assert df["parent_track_id"].dtype.kind == "i"
