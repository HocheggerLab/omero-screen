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
    _auto_gpu_window,
    _max_detections_for_window,
    add_track_columns,
    load_tracking_model,
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
            track_nucleus_mask(
                mask.astype(np.float32), mask, MagicMock(), mode="bogus"
            )

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

    def test_batch_size_forwarded_to_model(self) -> None:
        """batch_size is passed through to Trackastra.track (GPU memory knob)."""
        imgs = np.zeros((3, 4, 4), dtype=np.float32)
        mask = np.ones((3, 4, 4), dtype=np.uint16)
        model = MagicMock()
        model.track.return_value = ("fake_graph", None)

        with patch(
            "trackastra.tracking.graph_to_ctc",
            return_value=(_ctc_dataframe(), mask.copy()),
        ):
            track_nucleus_mask(imgs, mask, model, mode="greedy", batch_size=4)

        assert model.track.call_args.kwargs["batch_size"] == 4

    def test_batch_size_defaults_to_none(self) -> None:
        """Without an explicit value, defer to Trackastra's own default."""
        imgs = np.zeros((3, 4, 4), dtype=np.float32)
        mask = np.ones((3, 4, 4), dtype=np.uint16)
        model = MagicMock()
        model.track.return_value = ("fake_graph", None)

        with patch(
            "trackastra.tracking.graph_to_ctc",
            return_value=(_ctc_dataframe(), mask.copy()),
        ):
            track_nucleus_mask(imgs, mask, model, mode="greedy")

        assert model.track.call_args.kwargs["batch_size"] is None

    def test_window_override_sets_model_config(self) -> None:
        """An explicit window shrinks the model's temporal window in place."""
        imgs = np.zeros((3, 4, 4), dtype=np.float32)
        mask = np.ones((3, 4, 4), dtype=np.uint16)
        model = MagicMock()
        model.track.return_value = ("fake_graph", None)
        model.transformer.config = {"window": 10}  # real dict, not a Mock

        with patch(
            "trackastra.tracking.graph_to_ctc",
            return_value=(_ctc_dataframe(), mask.copy()),
        ):
            track_nucleus_mask(imgs, mask, model, mode="greedy", window=2)

        assert model.transformer.config["window"] == 2

    def test_window_none_keeps_model_config(self) -> None:
        """Without an override the model's trained window is untouched."""
        imgs = np.zeros((3, 4, 4), dtype=np.float32)
        mask = np.ones((3, 4, 4), dtype=np.uint16)
        model = MagicMock()
        model.track.return_value = ("fake_graph", None)
        model.transformer.config = {"window": 10}

        with patch(
            "trackastra.tracking.graph_to_ctc",
            return_value=(_ctc_dataframe(), mask.copy()),
        ):
            track_nucleus_mask(imgs, mask, model, mode="greedy")

        assert model.transformer.config["window"] == 10


class TestLoadTrackingModel:
    def test_device_override_is_passed(self) -> None:
        """An explicit device is forwarded to Trackastra.from_pretrained."""
        with patch(
            "trackastra.model.Trackastra.from_pretrained"
        ) as mock_from_pretrained:
            load_tracking_model("general_2d", device="cpu")

        assert mock_from_pretrained.call_args.kwargs["device"] == "cpu"

    def test_device_none_autodetects(self) -> None:
        """device=None falls back to get_device()."""
        with (
            patch(
                "trackastra.model.Trackastra.from_pretrained"
            ) as mock_from_pretrained,
            patch("omero_screen.tracking.get_device", return_value="cuda"),
        ):
            load_tracking_model("general_2d")

        assert mock_from_pretrained.call_args.kwargs["device"] == "cuda"

    def test_mps_falls_back_to_cpu(self) -> None:
        """Trackastra has no MPS kernels — auto-detected mps becomes cpu."""
        with (
            patch(
                "trackastra.model.Trackastra.from_pretrained"
            ) as mock_from_pretrained,
            patch("omero_screen.tracking.get_device", return_value="mps"),
        ):
            load_tracking_model("general_2d")

        assert mock_from_pretrained.call_args.kwargs["device"] == "cpu"


class TestMaxDetectionsForWindow:
    def test_sliding_window_max(self) -> None:
        """N is the max sum over any contiguous window of frames."""
        per_frame = [1, 2, 3, 4]
        assert _max_detections_for_window(per_frame, 2) == 7  # 3+4
        assert _max_detections_for_window(per_frame, 1) == 4
        # Window larger than the movie clamps to the whole movie.
        assert _max_detections_for_window(per_frame, 10) == 10


class TestAutoGpuWindow:
    def test_reduces_window_to_fit_vram(self) -> None:
        """Shrinks the window until the ~200·N² estimate fits free VRAM."""
        per_frame = [5000] * 10  # N(w) = 5000·w
        # window 4 → 80 GB, window 3 → 45 GB, window 2 → 20 GB.
        # free = 30 GB → budget 0.85·30 = 25.5 GB → only window 2 fits.
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache"),
            patch(
                "torch.cuda.mem_get_info",
                return_value=(30 * 10**9, 44 * 10**9),
            ),
        ):
            assert _auto_gpu_window(per_frame, max_window=4) == 2

    def test_keeps_full_window_when_it_fits(self) -> None:
        """Plenty of free VRAM → the model's full window is kept."""
        per_frame = [5000] * 10
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache"),
            patch(
                "torch.cuda.mem_get_info",
                return_value=(200 * 10**9, 200 * 10**9),
            ),
        ):
            assert _auto_gpu_window(per_frame, max_window=4) == 4

    def test_no_cuda_returns_max_window(self) -> None:
        """Without CUDA info, run at the full window rather than guess."""
        with patch("torch.cuda.is_available", return_value=False):
            assert _auto_gpu_window([5000] * 10, max_window=4) == 4


class TestAddTrackColumns:
    def test_columns_derived_from_label(self) -> None:
        """track_id == label; parent columns come from the parent map."""
        df = pd.DataFrame(
            {"label": [1, 2, 3, 4], "area_nucleus": [10, 11, 5, 6]}
        )
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
