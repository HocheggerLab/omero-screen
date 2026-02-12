"""Tests for the plate_cache module."""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from omero_screen_napari.omero_data import OmeroData


# --------------- Fixtures ---------------


@pytest.fixture
def mock_cache():
    """Provide a dict-like mock for the diskcache.Cache."""
    store: dict[str, object] = {}

    class FakeCache:
        size_limit = 20 * 2**30

        def get(self, key: str, default=None):
            return store.get(key, default)

        def __setitem__(self, key: str, value: object) -> None:
            store[key] = value

        def __getitem__(self, key: str) -> object:
            return store[key]

    return FakeCache(), store


@pytest.fixture
def sample_meta() -> dict:
    return {
        "channel_data": {"DAPI": "0", "Tub": "1"},
        "pixel_size": (0.3, 0.3),
        "intensities": {0: (100, 5000), 1: (50, 3000)},
        "plate_name": "TestPlate",
    }


@pytest.fixture
def sample_wells() -> dict:
    return {
        "A1": {
            "well_id": 10,
            "metadata": {"cell_line": "RPE", "condition": "ctrl"},
            "images": [
                {"image_id": 100, "size_t": 1, "index": 0, "pos_x": 0.0, "pos_y": 0.0},
                {"image_id": 101, "size_t": 1, "index": 1, "pos_x": 1.0, "pos_y": 0.0},
            ],
        },
        "A2": {
            "well_id": 11,
            "metadata": {"cell_line": "RPE", "condition": "drug"},
            "images": [
                {"image_id": 200, "size_t": 1, "index": 0, "pos_x": 0.0, "pos_y": 0.0},
            ],
        },
    }


@pytest.fixture
def sample_label_map() -> dict:
    return {
        "A1": [500, 501],
        "A2": [600],
    }


# --------------- is_plate_cached ---------------


class TestIsPlateCached:
    def test_not_cached(self, mock_cache):
        fake_cache, store = mock_cache
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import is_plate_cached

            assert is_plate_cached(999) is False

    def test_only_meta_cached(self, mock_cache, sample_meta):
        fake_cache, store = mock_cache
        store["plate:123:meta"] = sample_meta
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import is_plate_cached

            assert is_plate_cached(123) is False

    def test_fully_cached(self, mock_cache, sample_meta, sample_wells):
        fake_cache, store = mock_cache
        store["plate:123:meta"] = sample_meta
        store["plate:123:wells"] = sample_wells
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import is_plate_cached

            assert is_plate_cached(123) is True


# --------------- get_cached_plate_metadata ---------------


class TestGetCachedMetadata:
    def test_returns_none_when_missing(self, mock_cache):
        fake_cache, store = mock_cache
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_cached_plate_metadata

            assert get_cached_plate_metadata(999) is None

    def test_returns_metadata(self, mock_cache, sample_meta):
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_cached_plate_metadata

            result = get_cached_plate_metadata(42)
            assert result == sample_meta


# --------------- _parse_image_index ---------------


class TestParseImageIndex:
    def test_all(self, sample_wells):
        from omero_screen_napari.plate_cache import _parse_image_index

        result = _parse_image_index("All", sample_wells, ["A1"])
        assert result == [0, 1]

    def test_range(self, sample_wells):
        from omero_screen_napari.plate_cache import _parse_image_index

        result = _parse_image_index("0-2", sample_wells, ["A1"])
        assert result == [0, 1, 2]

    def test_list(self, sample_wells):
        from omero_screen_napari.plate_cache import _parse_image_index

        result = _parse_image_index("0,1", sample_wells, ["A1"])
        assert result == [0, 1]

    def test_single(self, sample_wells):
        from omero_screen_napari.plate_cache import _parse_image_index

        result = _parse_image_index("0", sample_wells, ["A1"])
        assert result == [0]


# --------------- _parse_time_range ---------------


class TestParseTimeRange:
    def test_all(self):
        from omero_screen_napari.plate_cache import _parse_time_range

        assert _parse_time_range("All") == (None, None)

    def test_range(self):
        from omero_screen_napari.plate_cache import _parse_time_range

        assert _parse_time_range("1-3") == (0, 3)

    def test_single(self):
        from omero_screen_napari.plate_cache import _parse_time_range

        assert _parse_time_range("2") == (1, 2)


# --------------- _row_col_to_well_pos ---------------


class TestRowColToWellPos:
    def test_a1(self):
        from omero_screen_napari.plate_cache import _row_col_to_well_pos

        assert _row_col_to_well_pos(0, 0) == "A1"

    def test_b3(self):
        from omero_screen_napari.plate_cache import _row_col_to_well_pos

        assert _row_col_to_well_pos(1, 2) == "B3"

    def test_h12(self):
        from omero_screen_napari.plate_cache import _row_col_to_well_pos

        assert _row_col_to_well_pos(7, 11) == "H12"


# --------------- _partition_round_robin ---------------


class TestPartitionRoundRobin:
    def test_even_split(self):
        from omero_screen_napari.plate_cache import _partition_round_robin

        items = [{"id": i} for i in range(6)]
        batches = _partition_round_robin(items, 3)
        assert len(batches) == 3
        assert all(len(b) == 2 for b in batches)

    def test_uneven_split(self):
        from omero_screen_napari.plate_cache import _partition_round_robin

        items = [{"id": i} for i in range(5)]
        batches = _partition_round_robin(items, 3)
        assert len(batches) == 3
        assert sum(len(b) for b in batches) == 5

    def test_more_workers_than_items(self):
        from omero_screen_napari.plate_cache import _partition_round_robin

        items = [{"id": i} for i in range(2)]
        batches = _partition_round_robin(items, 5)
        assert len(batches) == 2  # Empty batches filtered out


# --------------- load_from_cache ---------------


class TestLoadFromCache:
    def test_load_single_well(self, mock_cache, sample_meta, sample_wells, sample_label_map):
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        store["plate:42:labels"] = sample_label_map

        # Add image data to cache
        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        store["100:0"] = img_array
        store["101:0"] = img_array
        # Add label data
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        store["500:0"] = label_array
        store["501:0"] = label_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            load_from_cache(od, 42, "A1", "All")

            assert od.plate_id == 42
            assert od.channel_data == {"DAPI": "0", "Tub": "1"}
            assert od.pixel_size == (0.3, 0.3)
            assert od.plate_name == "TestPlate"
            assert len(od.image_ids) == 2
            assert od.images.shape[0] == 2  # 2 images
            assert len(od.well_metadata_list) == 1
            assert od.well_metadata_list[0]["cell_line"] == "RPE"

    def test_load_raises_when_not_cached(self, mock_cache):
        fake_cache, store = mock_cache
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            with pytest.raises(ValueError, match="not fully cached"):
                load_from_cache(od, 999, "A1", "All")

    def test_load_with_image_selection(self, mock_cache, sample_meta, sample_wells, sample_label_map):
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        store["plate:42:labels"] = sample_label_map

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        store["100:0"] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        store["500:0"] = label_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            load_from_cache(od, 42, "A1", "0")  # Only first image

            assert len(od.image_ids) == 1
            assert od.image_ids[0] == 100


# --------------- _parse_channel_data ---------------


class TestParseChannelData:
    def test_parses_dapi_channel(self):
        from omero_screen_napari.plate_cache import _parse_channel_data

        mock_plate = MagicMock()
        mock_ann = MagicMock()
        mock_ann.getValue.return_value = [("DAPI", "0"), ("Tub", "1")]
        mock_plate.listAnnotations.return_value = [mock_ann]
        mock_plate.getId.return_value = 42

        # Need to make the annotation look like a MapAnnotationWrapper
        with patch(
            "omero_screen_napari.plate_cache.MapAnnotationWrapper",
            type(mock_ann),
        ):
            result = _parse_channel_data(mock_plate)
            assert "DAPI" in result
            assert result["DAPI"] == "0"

    def test_hoechst_renamed_to_dapi(self):
        from omero_screen_napari.plate_cache import _parse_channel_data

        mock_plate = MagicMock()
        mock_ann = MagicMock()
        mock_ann.getValue.return_value = [("Hoechst", "0"), ("Tub", "1")]
        mock_plate.listAnnotations.return_value = [mock_ann]
        mock_plate.getId.return_value = 42

        with patch(
            "omero_screen_napari.plate_cache.MapAnnotationWrapper",
            type(mock_ann),
        ):
            result = _parse_channel_data(mock_plate)
            assert "DAPI" in result
            assert "Hoechst" not in result


# --------------- _default_intensities ---------------


class TestDefaultIntensities:
    def test_creates_default_range(self):
        from omero_screen_napari.plate_cache import _default_intensities

        channel_data = {"DAPI": "0", "Tub": "1", "EdU": "2"}
        result = _default_intensities(channel_data)
        assert result == {0: (0, 65535), 1: (0, 65535), 2: (0, 65535)}
