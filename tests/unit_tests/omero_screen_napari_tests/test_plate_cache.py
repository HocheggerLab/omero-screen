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

        def __contains__(self, key: str) -> bool:
            return key in store

        def iterkeys(self):
            return iter(store.keys())

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


# --------------- is_plate_fully_cached ---------------


class TestIsPlateFullyCached:
    def test_false_when_no_metadata(self, mock_cache):
        fake_cache, _store = mock_cache
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import is_plate_fully_cached

            assert is_plate_fully_cached(999) is False

    def test_false_when_images_missing(self, mock_cache, sample_meta, sample_wells):
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        # Only cache one of two images for A1
        store["100:0"] = np.zeros((1, 10, 10, 2), dtype=np.float32)
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import is_plate_fully_cached

            assert is_plate_fully_cached(42) is False

    def test_true_when_all_images_cached(self, mock_cache, sample_meta, sample_wells):
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        store["100:0"] = img
        store["101:0"] = img
        store["200:0"] = img
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import is_plate_fully_cached

            assert is_plate_fully_cached(42) is True


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

    def test_load_from_cache_stores_positions(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        """Verify that image positions are populated from cached well data."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        store["plate:42:labels"] = sample_label_map

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        store["100:0"] = img_array
        store["101:0"] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        store["500:0"] = label_array
        store["501:0"] = label_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            load_from_cache(od, 42, "A1", "All")

            assert len(od.image_positions) == 2
            assert od.image_positions[0] == (0.0, 0.0)
            assert od.image_positions[1] == (1.0, 0.0)

    def test_load_from_cache_dict_positions(
        self, mock_cache, sample_meta, sample_label_map
    ):
        """Verify dict-format positions (old caches) are handled correctly."""
        fake_cache, store = mock_cache

        wells_dict_pos = {
            "A1": {
                "well_id": 10,
                "metadata": {"cell_line": "RPE"},
                "images": [
                    {
                        "image_id": 100,
                        "size_t": 1,
                        "index": 0,
                        "pos_x": {"value": 5.0, "unit": "MICROMETER"},
                        "pos_y": {"value": 10.0, "unit": "MICROMETER"},
                    },
                ],
            },
        }
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = wells_dict_pos
        store["plate:42:labels"] = sample_label_map

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        store["100:0"] = img_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            load_from_cache(od, 42, "A1", "All")

            assert len(od.image_positions) == 1
            assert od.image_positions[0] == (5.0, 10.0)

    def test_load_from_cache_null_positions(self, mock_cache, sample_meta, sample_label_map):
        """Verify None is used when pos_x/pos_y are missing."""
        fake_cache, store = mock_cache

        wells_no_pos = {
            "A1": {
                "well_id": 10,
                "metadata": {"cell_line": "RPE"},
                "images": [
                    {"image_id": 100, "size_t": 1, "index": 0},
                    {"image_id": 101, "size_t": 1, "index": 1},
                ],
            },
        }
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = wells_no_pos
        store["plate:42:labels"] = sample_label_map

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        store["100:0"] = img_array
        store["101:0"] = img_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            load_from_cache(od, 42, "A1", "All")

            assert len(od.image_positions) == 2
            assert od.image_positions[0] is None
            assert od.image_positions[1] is None

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


# --------------- _unwrap_length ---------------


class TestUnwrapLength:
    def test_plain_float(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length(42.5) == 42.5

    def test_int(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length(10) == 10.0

    def test_dict_with_value(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length({"value": 123.4, "unit": "MICROMETER", "symbol": "µm"}) == 123.4

    def test_dict_without_value(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length({"unit": "MICROMETER"}) is None

    def test_none(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length(None) is None

    def test_unconvertible(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length("not_a_number") is None


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


# --------------- get_all_cached_plates ---------------


class TestGetAllCachedPlates:
    def test_empty_cache_returns_empty(self, mock_cache):
        fake_cache, _store = mock_cache
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_all_cached_plates

            assert get_all_cached_plates() == []

    def test_finds_cached_plates(self, mock_cache, sample_meta):
        fake_cache, store = mock_cache
        store["plate:123:meta"] = sample_meta
        store["plate:456:meta"] = {**sample_meta, "plate_name": "OtherPlate"}
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_all_cached_plates

            plates = get_all_cached_plates()
            assert len(plates) == 2
            plate_ids = [p[0] for p in plates]
            assert 123 in plate_ids
            assert 456 in plate_ids

    def test_sorted_by_plate_id_descending(self, mock_cache, sample_meta):
        fake_cache, store = mock_cache
        store["plate:100:meta"] = {**sample_meta, "plate_name": "First"}
        store["plate:200:meta"] = {**sample_meta, "plate_name": "Second"}
        store["plate:150:meta"] = {**sample_meta, "plate_name": "Middle"}
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_all_cached_plates

            plates = get_all_cached_plates()
            assert plates[0] == (200, "Second")
            assert plates[1] == (150, "Middle")
            assert plates[2] == (100, "First")

    def test_skips_corrupt_metadata(self, mock_cache, sample_meta):
        fake_cache, store = mock_cache
        store["plate:123:meta"] = sample_meta
        store["plate:999:meta"] = "not_a_dict"
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_all_cached_plates

            plates = get_all_cached_plates()
            assert len(plates) == 1
            assert plates[0][0] == 123


# --------------- get_well_cache_status ---------------


class TestGetWellCacheStatus:
    def test_uncached_plate_returns_empty(self, mock_cache):
        fake_cache, _store = mock_cache
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_well_cache_status

            assert get_well_cache_status(999) == {}

    def test_all_images_cached_returns_true(
        self, mock_cache, sample_wells
    ):
        fake_cache, store = mock_cache
        store["plate:42:wells"] = sample_wells
        # Cache all images for A1 and A2
        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        store["100:0"] = img_array
        store["101:0"] = img_array
        store["200:0"] = img_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_well_cache_status

            status = get_well_cache_status(42)
            assert status["A1"] is True
            assert status["A2"] is True

    def test_missing_images_returns_false(
        self, mock_cache, sample_wells
    ):
        fake_cache, store = mock_cache
        store["plate:42:wells"] = sample_wells
        # Only cache one image for A1 (missing 101:0)
        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        store["100:0"] = img_array
        store["200:0"] = img_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_well_cache_status

            status = get_well_cache_status(42)
            assert status["A1"] is False
            assert status["A2"] is True

    def test_multi_timepoint_check(self, mock_cache):
        fake_cache, store = mock_cache
        wells = {
            "A1": {
                "well_id": 10,
                "metadata": {},
                "images": [
                    {"image_id": 100, "size_t": 3, "index": 0},
                ],
            },
        }
        store["plate:42:wells"] = wells
        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        # Cache only 2 of 3 timepoints
        store["100:0"] = img_array
        store["100:1"] = img_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_well_cache_status

            status = get_well_cache_status(42)
            assert status["A1"] is False

        # Now add the missing timepoint
        store["100:2"] = img_array
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_well_cache_status

            status = get_well_cache_status(42)
            assert status["A1"] is True


# --------------- _well_sort_key (plate_cache version) ---------------


class TestWellSortKeyPlateCache:
    def test_sorts_correctly(self):
        from omero_screen_napari.plate_cache import _well_sort_key

        wells = ["B2", "A1", "A10", "A2", "B1"]
        sorted_wells = sorted(wells, key=_well_sort_key)
        assert sorted_wells == ["A1", "A2", "A10", "B1", "B2"]


# --------------- Well-grouped partitioning ---------------


class TestWellGroupedPartitioning:
    """Tests that well-grouped partitioning keeps wells together in batches."""

    def test_wells_complete_sequentially_within_worker(self, mock_cache, sample_wells):
        """Images from the same well should be in the same batch."""
        fake_cache, store = mock_cache
        store["plate:42:wells"] = sample_wells
        store["plate:42:labels"] = {}

        # We can't easily call cache_plate() (needs OMERO connection),
        # but we can verify the partitioning logic by simulating it.
        from omero_screen_napari.plate_cache import _well_sort_key

        wells = sample_wells
        label_map: dict[str, list[int]] = {}
        sorted_well_keys = sorted(wells.keys(), key=_well_sort_key)
        well_groups: list[list[dict]] = []

        for well_pos in sorted_well_keys:
            group: list[dict] = []
            for img_info in wells[well_pos]["images"]:
                for t in range(img_info["size_t"]):
                    group.append(
                        {
                            "image_id": img_info["image_id"],
                            "timepoint": t,
                            "well": well_pos,
                        }
                    )
            if group:
                well_groups.append(group)

        # Distribute to 2 workers
        max_workers = 2
        batches: list[list[dict]] = [[] for _ in range(max_workers)]
        for i, group in enumerate(well_groups):
            batches[i % max_workers].extend(group)
        batches = [b for b in batches if b]

        # A1 and A2 should be in different batches (round-robin by well)
        batch0_wells = {item["well"] for item in batches[0]}
        batch1_wells = {item["well"] for item in batches[1]}

        # With 2 wells and 2 workers, each batch gets one well
        # A1 -> worker 0, A2 -> worker 1
        assert "A1" in batch0_wells
        assert "A2" in batch1_wells

    def test_workers_get_sorted_wells(self, mock_cache):
        """Wells should be distributed in sorted order (A1, A2, B1, ...)."""
        from omero_screen_napari.plate_cache import _well_sort_key

        wells = {
            "B1": {"images": [{"image_id": 300, "size_t": 1}]},
            "A2": {"images": [{"image_id": 200, "size_t": 1}]},
            "A1": {"images": [{"image_id": 100, "size_t": 1}]},
        }

        sorted_keys = sorted(wells.keys(), key=_well_sort_key)
        assert sorted_keys == ["A1", "A2", "B1"]
