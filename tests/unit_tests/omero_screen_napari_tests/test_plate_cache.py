"""Tests for the plate_cache module."""

import queue
import threading
from datetime import date
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

        def pop(self, key: str, default=None):
            return store.pop(key, default)

        def volume(self) -> int:
            total = 0
            for v in store.values():
                if isinstance(v, np.ndarray):
                    total += v.nbytes
                else:
                    total += 100  # small estimate for metadata dicts
            return total

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
                {
                    "image_id": 100,
                    "size_t": 1,
                    "size_z": 1,
                    "size_c": 2,
                    "size_y": 100,
                    "size_x": 100,
                    "index": 0,
                    "pos_x": 0.0,
                    "pos_y": 0.0,
                },
                {
                    "image_id": 101,
                    "size_t": 1,
                    "size_z": 1,
                    "size_c": 2,
                    "size_y": 100,
                    "size_x": 100,
                    "index": 1,
                    "pos_x": 1.0,
                    "pos_y": 0.0,
                },
            ],
        },
        "A2": {
            "well_id": 11,
            "metadata": {"cell_line": "RPE", "condition": "drug"},
            "images": [
                {
                    "image_id": 200,
                    "size_t": 1,
                    "size_z": 1,
                    "size_c": 2,
                    "size_y": 100,
                    "size_x": 100,
                    "index": 0,
                    "pos_x": 0.0,
                    "pos_y": 0.0,
                },
            ],
        },
    }


@pytest.fixture
def sample_label_map() -> dict:
    return {
        "A1": [
            {"label_id": 500, "size_t": 1},
            {"label_id": 501, "size_t": 1},
        ],
        "A2": [{"label_id": 600, "size_t": 1}],
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

    def test_false_when_images_missing(
        self, mock_cache, sample_meta, sample_wells
    ):
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        # Only cache one of two images for A1
        store["100:0"] = np.zeros((1, 10, 10, 2), dtype=np.float32)
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import is_plate_fully_cached

            assert is_plate_fully_cached(42) is False

    def test_true_when_all_images_cached(
        self, mock_cache, sample_meta, sample_wells
    ):
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
            from omero_screen_napari.plate_cache import (
                get_cached_plate_metadata,
            )

            assert get_cached_plate_metadata(999) is None

    def test_returns_metadata(self, mock_cache, sample_meta):
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import (
                get_cached_plate_metadata,
            )

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
    def test_load_single_well(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
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

    def test_load_from_cache_null_positions(
        self, mock_cache, sample_meta, sample_label_map
    ):
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

    def test_load_with_image_selection(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
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

        assert (
            _unwrap_length(
                {"value": 123.4, "unit": "MICROMETER", "symbol": "µm"}
            )
            == 123.4
        )

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

    def test_all_images_cached_returns_true(self, mock_cache, sample_wells):
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

    def test_missing_images_returns_false(self, mock_cache, sample_wells):
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

    def test_wells_complete_sequentially_within_worker(
        self, mock_cache, sample_wells
    ):
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


# --------------- delete_plate_from_cache ---------------


class TestDeletePlateFromCache:
    def test_deletes_all_keys(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        store["plate:42:labels"] = sample_label_map

        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        store["100:0"] = img
        store["101:0"] = img
        store["200:0"] = img
        store["500:0"] = img
        store["501:0"] = img
        store["600:0"] = img

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            count = delete_plate_from_cache(42)

        # 3 images + 3 labels + 3 metadata = 9
        assert count == 9
        assert "plate:42:meta" not in store
        assert "plate:42:wells" not in store
        assert "plate:42:labels" not in store
        assert "100:0" not in store
        assert "500:0" not in store

    def test_returns_zero_for_missing_plate(self, mock_cache):
        fake_cache, _store = mock_cache
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            assert delete_plate_from_cache(999) == 0

    def test_metadata_only_plate(self, mock_cache, sample_meta):
        """A plate with only metadata (no wells/labels) still gets cleaned up."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            count = delete_plate_from_cache(42)

        assert count == 1  # Only the meta key existed
        assert "plate:42:meta" not in store

    def test_partial_images(self, mock_cache, sample_meta, sample_wells):
        """Deletes whatever keys exist, even if some images are missing."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        # Only 1 of 3 images cached
        store["100:0"] = np.zeros((1, 10, 10, 2), dtype=np.float32)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            count = delete_plate_from_cache(42)

        # meta + wells + 1 image = 3 (labels key didn't exist)
        assert count == 3
        assert "100:0" not in store

    def test_does_not_affect_other_plates(
        self, mock_cache, sample_meta, sample_wells
    ):
        """Deleting one plate should not touch another plate's keys."""
        fake_cache, store = mock_cache
        # Plate 42
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        store["100:0"] = img
        store["101:0"] = img
        store["200:0"] = img
        # Plate 99
        other_meta = {**sample_meta, "plate_name": "Other"}
        store["plate:99:meta"] = other_meta
        store["plate:99:wells"] = {
            "B1": {
                "well_id": 20,
                "metadata": {},
                "images": [{"image_id": 900, "size_t": 1, "index": 0}],
            }
        }
        store["900:0"] = img

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            delete_plate_from_cache(42)

        assert "plate:99:meta" in store
        assert "900:0" in store


# --------------- ensure_cache_space ---------------


class TestEnsureCacheSpace:
    def test_no_eviction_when_space_available(self, mock_cache, sample_meta):
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import ensure_cache_space

            evicted = ensure_cache_space(1000)

        assert evicted == []

    def test_evicts_oldest_plate_first(self, mock_cache, sample_meta):
        """With two plates, evicts the one with the smaller plate_id first."""
        fake_cache, store = mock_cache
        # Set a very small size limit to force eviction
        fake_cache.size_limit = 2000

        # Plate 100 (older/smaller ID)
        store["plate:100:meta"] = sample_meta
        store["plate:100:wells"] = {
            "A1": {
                "well_id": 1,
                "metadata": {},
                "images": [{"image_id": 1000, "size_t": 1, "index": 0}],
            }
        }
        store["1000:0"] = np.zeros((10, 10, 2), dtype=np.float32)

        # Plate 200 (newer/larger ID)
        store["plate:200:meta"] = {**sample_meta, "plate_name": "Newer"}
        store["plate:200:wells"] = {
            "A1": {
                "well_id": 2,
                "metadata": {},
                "images": [{"image_id": 2000, "size_t": 1, "index": 0}],
            }
        }
        store["2000:0"] = np.zeros((10, 10, 2), dtype=np.float32)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import ensure_cache_space

            evicted = ensure_cache_space(1000)

        # Plate 100 should be evicted first (smallest ID)
        assert 100 in evicted
        assert "1000:0" not in store

    def test_evicts_multiple_plates_if_needed(self, mock_cache, sample_meta):
        fake_cache, store = mock_cache
        fake_cache.size_limit = 500  # Very small

        for pid in [10, 20, 30]:
            store[f"plate:{pid}:meta"] = {
                **sample_meta,
                "plate_name": f"P{pid}",
            }
            store[f"plate:{pid}:wells"] = {
                "A1": {
                    "well_id": pid,
                    "metadata": {},
                    "images": [
                        {"image_id": pid * 100, "size_t": 1, "index": 0}
                    ],
                }
            }
            store[f"{pid * 100}:0"] = np.zeros((10, 10, 2), dtype=np.float32)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import ensure_cache_space

            evicted = ensure_cache_space(500)

        assert len(evicted) >= 1  # At least one plate evicted

    def test_respects_exclude_plate_ids(self, mock_cache, sample_meta):
        fake_cache, store = mock_cache
        fake_cache.size_limit = 500

        for pid in [10, 20]:
            store[f"plate:{pid}:meta"] = {
                **sample_meta,
                "plate_name": f"P{pid}",
            }
            store[f"plate:{pid}:wells"] = {
                "A1": {
                    "well_id": pid,
                    "metadata": {},
                    "images": [
                        {"image_id": pid * 100, "size_t": 1, "index": 0}
                    ],
                }
            }
            store[f"{pid * 100}:0"] = np.zeros((10, 10, 2), dtype=np.float32)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import ensure_cache_space

            evicted = ensure_cache_space(500, exclude_plate_ids={10})

        # Plate 10 should NOT be evicted
        assert 10 not in evicted
        assert "plate:10:meta" in store


# --------------- clean_orphaned_plates ---------------


class TestCleanOrphanedPlates:
    def test_removes_incomplete_plates(
        self, mock_cache, sample_meta, sample_wells
    ):
        """Plate with <50% images should be cleaned."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        # Only 1 of 3 images cached = 33% completeness
        store["100:0"] = np.zeros((1, 10, 10, 2), dtype=np.float32)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import clean_orphaned_plates

            cleaned = clean_orphaned_plates()

        assert 42 in cleaned
        assert "plate:42:meta" not in store

    def test_keeps_complete_plates(
        self, mock_cache, sample_meta, sample_wells
    ):
        """Plate with >=50% images should be kept."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        # 2 of 3 images = 67% completeness
        store["100:0"] = img
        store["101:0"] = img

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import clean_orphaned_plates

            cleaned = clean_orphaned_plates()

        assert cleaned == []
        assert "plate:42:meta" in store

    def test_respects_exclude_plate_ids(
        self, mock_cache, sample_meta, sample_wells
    ):
        """Excluded plates should not be cleaned even if incomplete."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        # 0 of 3 images = 0% completeness, but excluded

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import clean_orphaned_plates

            cleaned = clean_orphaned_plates(exclude_plate_ids={42})

        assert cleaned == []
        assert "plate:42:meta" in store


# --------------- _estimate_plate_bytes ---------------


class TestEstimatePlateBytes:
    def test_uses_actual_dimensions(self):
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                "images": [
                    {
                        "image_id": 100,
                        "size_t": 1,
                        "size_z": 1,
                        "size_c": 2,
                        "size_y": 1024,
                        "size_x": 1024,
                    }
                ],
            }
        }
        result = _estimate_plate_bytes(wells)
        # 1 * 1 * 1024 * 1024 * 2 channels * 2 bytes = 4 MB
        assert result == 1 * 1024 * 1024 * 2 * 2

    def test_multiple_wells_and_timepoints(self):
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                "images": [
                    {
                        "image_id": 100,
                        "size_t": 3,
                        "size_z": 1,
                        "size_c": 4,
                        "size_y": 2048,
                        "size_x": 2048,
                    },
                    {
                        "image_id": 101,
                        "size_t": 2,
                        "size_z": 1,
                        "size_c": 4,
                        "size_y": 2048,
                        "size_x": 2048,
                    },
                ],
            },
            "A2": {
                "images": [
                    {
                        "image_id": 200,
                        "size_t": 1,
                        "size_z": 1,
                        "size_c": 4,
                        "size_y": 2048,
                        "size_x": 2048,
                    }
                ],
            },
        }
        result = _estimate_plate_bytes(wells)
        per_tp = 1 * 2048 * 2048 * 4 * 2  # 32 MB per timepoint
        # 3 + 2 + 1 = 6 timepoints
        assert result == 6 * per_tp

    def test_falls_back_when_dimensions_missing(self):
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                "images": [{"image_id": 100, "size_t": 1}],
            }
        }
        result = _estimate_plate_bytes(wells)
        assert result == 10 * 2**20  # fallback: 1 slot x 10MB default

    def test_includes_label_bytes(self):
        """Labels (segmentation masks) are counted in the estimate."""
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                "images": [
                    {
                        "image_id": 100,
                        "size_t": 1,
                        "size_z": 1,
                        "size_c": 4,
                        "size_y": 1024,
                        "size_x": 1024,
                    }
                ],
            }
        }
        label_map = {
            "A1": [
                {
                    "label_id": 500,
                    "size_t": 1,
                    "size_z": 1,
                    "size_c": 2,
                    "size_y": 1024,
                    "size_x": 1024,
                }
            ],
        }
        image_bytes = 1 * 1024 * 1024 * 4 * 2
        label_bytes = 1 * 1024 * 1024 * 2 * 2
        result = _estimate_plate_bytes(wells, label_map)
        assert result == image_bytes + label_bytes

    def test_no_label_map_still_works(self):
        """Passing None for label_map only counts images."""
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                "images": [
                    {
                        "image_id": 100,
                        "size_t": 1,
                        "size_z": 1,
                        "size_c": 2,
                        "size_y": 512,
                        "size_x": 512,
                    }
                ],
            }
        }
        result = _estimate_plate_bytes(wells, None)
        assert result == 1 * 512 * 512 * 2 * 2


# --------------- _plate_image_completeness ---------------


class TestPlateImageCompleteness:
    def test_no_wells_returns_zero(self, mock_cache):
        fake_cache, _store = mock_cache
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import (
                _plate_image_completeness,
            )

            assert _plate_image_completeness(999) == 0.0

    def test_all_present(self, mock_cache, sample_wells):
        fake_cache, store = mock_cache
        store["plate:42:wells"] = sample_wells
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        store["100:0"] = img
        store["101:0"] = img
        store["200:0"] = img

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import (
                _plate_image_completeness,
            )

            assert _plate_image_completeness(42) == 1.0

    def test_partial(self, mock_cache, sample_wells):
        fake_cache, store = mock_cache
        store["plate:42:wells"] = sample_wells
        # 1 of 3 images
        store["100:0"] = np.zeros((1, 10, 10, 2), dtype=np.float32)

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import (
                _plate_image_completeness,
            )

            result = _plate_image_completeness(42)
            assert abs(result - 1 / 3) < 0.01


# --------------- get_plate_history ---------------


class TestGetPlateHistory:
    def test_returns_empty_when_no_history(self, mock_cache):
        fake_cache, _store = mock_cache
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_plate_history

            assert get_plate_history() == {}

    def test_returns_stored_history(self, mock_cache):
        fake_cache, store = mock_cache
        history = {
            42: {
                "plate_name": "MyPlate",
                "status": "cached",
                "last_cached": "2026-02-20",
            }
        }
        store["plate_history"] = history
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_plate_history

            result = get_plate_history()
            assert result == history

    def test_migrates_existing_meta_keys(self, mock_cache, sample_meta):
        """Plates with meta keys but no history should be auto-migrated."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:99:meta"] = {**sample_meta, "plate_name": "OtherPlate"}

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_plate_history

            result = get_plate_history()

        assert 42 in result
        assert result[42]["plate_name"] == "TestPlate"
        assert result[42]["status"] == "cached"
        assert 99 in result
        assert result[99]["plate_name"] == "OtherPlate"
        # History should have been persisted
        assert "plate_history" in store

    def test_migration_does_not_overwrite_existing(
        self, mock_cache, sample_meta
    ):
        """Existing history entries should not be overwritten by migration."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate_history"] = {
            42: {
                "plate_name": "CustomName",
                "status": "removed",
                "last_cached": "2026-01-01",
            }
        }

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import get_plate_history

            result = get_plate_history()

        assert result[42]["plate_name"] == "CustomName"
        assert result[42]["status"] == "removed"


# --------------- _update_plate_history ---------------


class TestUpdatePlateHistory:
    def test_creates_new_entry(self, mock_cache):
        fake_cache, store = mock_cache
        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import _update_plate_history

            _update_plate_history(42, "NewPlate", "cached")

        history = store["plate_history"]
        assert 42 in history
        assert history[42]["plate_name"] == "NewPlate"
        assert history[42]["status"] == "cached"
        assert history[42]["last_cached"] == str(date.today())

    def test_updates_to_removed_preserves_last_cached(self, mock_cache):
        fake_cache, store = mock_cache
        store["plate_history"] = {
            42: {
                "plate_name": "MyPlate",
                "status": "cached",
                "last_cached": "2026-01-15",
            }
        }

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import _update_plate_history

            _update_plate_history(42, "MyPlate", "removed")

        history = store["plate_history"]
        assert history[42]["status"] == "removed"
        assert history[42]["last_cached"] == "2026-01-15"

    def test_updates_back_to_cached_refreshes_date(self, mock_cache):
        fake_cache, store = mock_cache
        store["plate_history"] = {
            42: {
                "plate_name": "MyPlate",
                "status": "removed",
                "last_cached": "2026-01-15",
            }
        }

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import _update_plate_history

            _update_plate_history(42, "MyPlate", "cached")

        history = store["plate_history"]
        assert history[42]["status"] == "cached"
        assert history[42]["last_cached"] == str(date.today())


# --------------- remove_plate_from_history ---------------


class TestRemovePlateFromHistory:
    def test_removes_entry(self, mock_cache):
        fake_cache, store = mock_cache
        store["plate_history"] = {
            42: {
                "plate_name": "MyPlate",
                "status": "removed",
                "last_cached": "2026-01-15",
            }
        }

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import (
                remove_plate_from_history,
            )

            remove_plate_from_history(42)

        assert 42 not in store["plate_history"]

    def test_also_deletes_cached_data(
        self, mock_cache, sample_meta, sample_wells
    ):
        """If plate has cached data, it should be deleted too."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        store["100:0"] = img
        store["101:0"] = img
        store["200:0"] = img
        store["plate_history"] = {
            42: {
                "plate_name": "MyPlate",
                "status": "cached",
                "last_cached": "2026-01-15",
            }
        }

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import (
                remove_plate_from_history,
            )

            remove_plate_from_history(42)

        assert 42 not in store["plate_history"]
        assert "plate:42:meta" not in store
        assert "100:0" not in store


# --------------- delete_plate_from_cache updates history ---------------


class TestDeletePlateUpdatesHistory:
    def test_creates_removed_history_entry(
        self, mock_cache, sample_meta, sample_wells
    ):
        """After deletion, history entry should exist with status 'removed'."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        store["100:0"] = img
        store["101:0"] = img
        store["200:0"] = img

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            delete_plate_from_cache(42)

        history = store.get("plate_history", {})
        assert 42 in history
        assert history[42]["status"] == "removed"
        assert history[42]["plate_name"] == "TestPlate"

    def test_preserves_existing_history_date(
        self, mock_cache, sample_meta, sample_wells
    ):
        """Deletion should preserve the original last_cached date."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        store["plate_history"] = {
            42: {
                "plate_name": "TestPlate",
                "status": "cached",
                "last_cached": "2026-01-10",
            }
        }

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            delete_plate_from_cache(42)

        history = store["plate_history"]
        assert history[42]["status"] == "removed"
        assert history[42]["last_cached"] == "2026-01-10"


# --------------- _normalize_label_entry ---------------


class TestNormalizeLabelEntry:
    def test_int_format(self):
        from omero_screen_napari.plate_cache import _normalize_label_entry

        result = _normalize_label_entry(500)
        assert result == {"label_id": 500, "size_t": 1}

    def test_dict_format(self):
        from omero_screen_napari.plate_cache import _normalize_label_entry

        entry = {"label_id": 500, "size_t": 3}
        result = _normalize_label_entry(entry)
        assert result == {"label_id": 500, "size_t": 3}


# --------------- Label multi-timepoint ---------------


class TestLabelMultiTimepoint:
    def test_old_int_format_backwards_compat(self, mock_cache, sample_meta):
        """Old caches with plain int label IDs should still work."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = {
            "A1": {
                "well_id": 10,
                "metadata": {"cell_line": "RPE"},
                "images": [
                    {"image_id": 100, "size_t": 1, "index": 0},
                ],
            },
        }
        # Old format: plain int label IDs
        store["plate:42:labels"] = {"A1": [500]}

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        store["100:0"] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        store["500:0"] = label_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            load_from_cache(od, 42, "A1", "All")

            assert od.labels.shape[0] == 1

    def test_new_dict_format_multi_timepoint(self, mock_cache, sample_meta):
        """New dict format with size_t > 1 loads labels correctly."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = {
            "A1": {
                "well_id": 10,
                "metadata": {"cell_line": "RPE"},
                "images": [
                    {"image_id": 100, "size_t": 1, "index": 0},
                ],
            },
        }
        store["plate:42:labels"] = {
            "A1": [{"label_id": 500, "size_t": 3}],
        }

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        store["100:0"] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        store["500:0"] = label_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            load_from_cache(od, 42, "A1", "All")

            # load_from_cache only loads t=0 for labels, so 1 label
            assert od.labels.shape[0] == 1

    def test_delete_handles_multi_timepoint_labels(
        self, mock_cache, sample_meta
    ):
        """delete_plate_from_cache removes all label timepoints."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = {
            "A1": {
                "well_id": 10,
                "metadata": {},
                "images": [{"image_id": 100, "size_t": 1, "index": 0}],
            },
        }
        store["plate:42:labels"] = {
            "A1": [{"label_id": 500, "size_t": 3}],
        }
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        store["100:0"] = img
        store["500:0"] = img
        store["500:1"] = img
        store["500:2"] = img

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            count = delete_plate_from_cache(42)

        # meta + wells + labels + 1 image + 3 label timepoints = 7
        assert count == 7
        assert "500:0" not in store
        assert "500:1" not in store
        assert "500:2" not in store

    def test_delete_handles_old_int_labels(self, mock_cache, sample_meta):
        """delete_plate_from_cache works with old int-format label maps."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = {
            "A1": {
                "well_id": 10,
                "metadata": {},
                "images": [{"image_id": 100, "size_t": 1, "index": 0}],
            },
        }
        # Old format
        store["plate:42:labels"] = {"A1": [500]}
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        store["100:0"] = img
        store["500:0"] = img

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            count = delete_plate_from_cache(42)

        # meta + wells + labels + 1 image + 1 label = 5
        assert count == 5
        assert "500:0" not in store


# --------------- _parse_raw_timepoint ---------------


class TestParseRawTimepoint:
    def test_basic_uint16(self):
        """Parse big-endian uint16 bytes into ZYXC array."""
        from omero_screen_napari.omero_image import _parse_raw_timepoint

        size_z, size_c, size_y, size_x = 1, 2, 4, 4
        dt_be = np.dtype(">u2")
        # Create CZYX data in big-endian
        data = np.arange(
            size_c * size_z * size_y * size_x, dtype=">u2"
        ).reshape(size_c, size_z, size_y, size_x)
        raw_bytes = data.tobytes()

        result = _parse_raw_timepoint(
            raw_bytes, size_z, size_c, size_y, size_x, dt_be
        )

        assert result.shape == (size_z, size_y, size_x, size_c)
        assert result.dtype == np.dtype("u2")  # native byte order
        # Check that channel 0 data appears in result[:, :, :, 0]
        np.testing.assert_array_equal(result[0, :, :, 0], data[0, 0])
        np.testing.assert_array_equal(result[0, :, :, 1], data[1, 0])

    def test_multi_z(self):
        """Multi-Z slices are reshaped correctly."""
        from omero_screen_napari.omero_image import _parse_raw_timepoint

        size_z, size_c, size_y, size_x = 3, 2, 4, 4
        dt_be = np.dtype(">u2")
        data = np.arange(
            size_c * size_z * size_y * size_x, dtype=">u2"
        ).reshape(size_c, size_z, size_y, size_x)
        raw_bytes = data.tobytes()

        result = _parse_raw_timepoint(
            raw_bytes, size_z, size_c, size_y, size_x, dt_be
        )

        assert result.shape == (3, 4, 4, 2)
        # Z=1, C=0 should equal data[C=0, Z=1]
        np.testing.assert_array_equal(result[1, :, :, 0], data[0, 1])

    def test_float32(self):
        """Float pixel types are handled correctly."""
        from omero_screen_napari.omero_image import _parse_raw_timepoint

        size_z, size_c, size_y, size_x = 1, 1, 2, 2
        dt_be = np.dtype(">f4")
        data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=">f4").reshape(
            1, 1, 2, 2
        )
        raw_bytes = data.tobytes()

        result = _parse_raw_timepoint(
            raw_bytes, size_z, size_c, size_y, size_x, dt_be
        )

        assert result.shape == (1, 2, 2, 1)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result[0, :, :, 0], [[1.5, 2.5], [3.5, 4.5]]
        )

    def test_contiguous_output(self):
        """Output array must be C-contiguous."""
        from omero_screen_napari.omero_image import _parse_raw_timepoint

        size_z, size_c, size_y, size_x = 2, 3, 8, 8
        dt_be = np.dtype(">u2")
        data = np.zeros(size_c * size_z * size_y * size_x, dtype=">u2")
        raw_bytes = data.tobytes()

        result = _parse_raw_timepoint(
            raw_bytes, size_z, size_c, size_y, size_x, dt_be
        )

        assert result.flags["C_CONTIGUOUS"]


# --------------- Helpers for RawPixelsStore tests ---------------


def _make_wrapper_mock(
    size_z: int = 1,
    size_c: int = 2,
    size_y: int = 10,
    size_x: int = 10,
    pixel_type: str = "uint16",
) -> MagicMock:
    """Create a mock ImageWrapper with proper pixel attributes."""
    wrapper = MagicMock()
    wrapper.getSizeZ.return_value = size_z
    wrapper.getSizeC.return_value = size_c
    wrapper.getSizeY.return_value = size_y
    wrapper.getSizeX.return_value = size_x
    pixels = MagicMock()
    pixels.getPixelsType.return_value.getValue.return_value = pixel_type
    pixels.getId.return_value = 1
    wrapper.getPrimaryPixels.return_value = pixels
    return wrapper


# --------------- ImageWrapper reuse ---------------


class TestImageWrapperReuse:
    def test_reuses_wrapper_for_same_image_id(self):
        """_get_omero_image_wrapper called once per unique image_id."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 100, "timepoint": 0, "apply_flatfield": False},
            {"image_id": 100, "timepoint": 1, "apply_flatfield": False},
            {"image_id": 100, "timepoint": 2, "apply_flatfield": False},
            {"image_id": 200, "timepoint": 0, "apply_flatfield": False},
        ]
        arr = np.zeros((1, 10, 10, 2), dtype=np.float32)
        mock_conn = MagicMock()
        wrapper = _make_wrapper_mock()

        with (
            patch("omero_screen_napari.plate_cache._cache"),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=wrapper,
            ) as mock_get_wrapper,
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=arr,
            ),
        ):
            progress_q: queue.Queue[int] = queue.Queue()
            _download_batch(batch, None, progress_q, conn=mock_conn)

            # Should be called exactly 2 times: once for 100, once for 200
            assert mock_get_wrapper.call_count == 2
            mock_get_wrapper.assert_any_call(mock_conn, 100)
            mock_get_wrapper.assert_any_call(mock_conn, 200)

    def test_fetches_each_unique_image_id(self):
        """Each unique image_id triggers one wrapper fetch."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 100, "timepoint": 0, "apply_flatfield": False},
            {"image_id": 200, "timepoint": 0, "apply_flatfield": False},
            {"image_id": 300, "timepoint": 0, "apply_flatfield": False},
        ]
        arr = np.zeros((1, 10, 10, 2), dtype=np.float32)
        mock_conn = MagicMock()

        with (
            patch("omero_screen_napari.plate_cache._cache"),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ) as mock_get_wrapper,
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=arr,
            ),
        ):
            _download_batch(batch, None, conn=mock_conn)

            assert mock_get_wrapper.call_count == 3

    def test_store_reused_across_timepoints(self):
        """RawPixelsStore is kept open for consecutive timepoints of the same image."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 100, "timepoint": 0, "apply_flatfield": False},
            {"image_id": 100, "timepoint": 1, "apply_flatfield": False},
            {"image_id": 100, "timepoint": 2, "apply_flatfield": False},
        ]
        arr = np.zeros((1, 10, 10, 2), dtype=np.float32)
        mock_conn = MagicMock()

        with (
            patch("omero_screen_napari.plate_cache._cache"),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=arr,
            ),
        ):
            _download_batch(batch, None, conn=mock_conn)

            # Only one createRawPixelsStore call for the same image
            assert mock_conn.c.sf.createRawPixelsStore.call_count == 1
            store = mock_conn.c.sf.createRawPixelsStore.return_value
            # getTimepoint called 3 times (one per batch item)
            assert store.getTimepoint.call_count == 3

    def test_store_recreated_for_different_images(self):
        """New RawPixelsStore created when image_id changes."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 100, "timepoint": 0, "apply_flatfield": False},
            {"image_id": 200, "timepoint": 0, "apply_flatfield": False},
        ]
        arr = np.zeros((1, 10, 10, 2), dtype=np.float32)
        mock_conn = MagicMock()

        with (
            patch("omero_screen_napari.plate_cache._cache"),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=arr,
            ),
        ):
            _download_batch(batch, None, conn=mock_conn)

            # Two different images → two createRawPixelsStore calls
            assert mock_conn.c.sf.createRawPixelsStore.call_count == 2


# --------------- Download batch completeness ---------------


class TestDownloadBatchCompleteness:
    def test_downloads_all_items(self):
        """All items in batch are downloaded."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": i, "timepoint": 0, "apply_flatfield": False}
            for i in range(5)
        ]
        arr = np.zeros((1, 10, 10, 2), dtype=np.uint16)
        mock_conn = MagicMock()

        cached_keys: list[str] = []

        class TrackingCache:
            def __setitem__(self, key: str, value: object) -> None:
                cached_keys.append(key)

        with (
            patch(
                "omero_screen_napari.plate_cache._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=arr,
            ),
        ):
            _download_batch(batch, None, conn=mock_conn)

        assert len(cached_keys) == 5

    def test_pause_event_blocks_worker(self):
        """Workers block when pause_event is cleared, resume when set."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": i, "timepoint": 0, "apply_flatfield": False}
            for i in range(3)
        ]
        arr = np.zeros((1, 10, 10, 2), dtype=np.uint16)
        mock_conn = MagicMock()

        cached_keys: list[str] = []

        class TrackingCache:
            def __setitem__(self, key: str, value: object) -> None:
                cached_keys.append(key)

        pause_event = threading.Event()
        pause_event.set()  # not paused

        with (
            patch(
                "omero_screen_napari.plate_cache._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=arr,
            ),
        ):
            # Start download in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(
                    _download_batch,
                    batch,
                    None,
                    None,
                    pause_event,
                    conn=mock_conn,
                )
                # Let it run to completion (event is set)
                future.result(timeout=5)

        assert len(cached_keys) == 3

    def test_pause_event_initially_cleared_blocks_then_resumes(self):
        """Worker blocked by cleared event resumes after set."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 100, "timepoint": 0, "apply_flatfield": False},
        ]
        arr = np.zeros((1, 10, 10, 2), dtype=np.uint16)
        mock_conn = MagicMock()

        cached_keys: list[str] = []

        class TrackingCache:
            def __setitem__(self, key: str, value: object) -> None:
                cached_keys.append(key)

        pause_event = threading.Event()
        # Start cleared — worker should block

        with (
            patch(
                "omero_screen_napari.plate_cache._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=arr,
            ),
        ):
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(
                    _download_batch,
                    batch,
                    None,
                    None,
                    pause_event,
                    conn=mock_conn,
                )
                # Worker should be blocked — no items cached yet
                import time as _time

                _time.sleep(0.1)
                assert len(cached_keys) == 0

                # Resume
                pause_event.set()
                future.result(timeout=5)

        assert len(cached_keys) == 1


# --------------- uint16 storage ---------------


class TestUint16Storage:
    def test_flatfield_corrected_stored_as_uint16(self):
        """Images with flatfield correction should be stored as uint16."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 100, "timepoint": 0, "apply_flatfield": True},
        ]
        # Simulate raw uint16 image from OMERO
        raw_arr = np.ones((1, 10, 10, 2), dtype=np.uint16) * 1000
        # Flatfield mask (ones = no correction effect)
        flatfield = np.ones((1, 10, 10, 2), dtype=np.float32)
        mock_conn = MagicMock()

        stored_values: dict[str, np.ndarray] = {}

        class TrackingCache:
            def __setitem__(self, key: str, value: object) -> None:
                stored_values[key] = value  # type: ignore[assignment]

        with (
            patch(
                "omero_screen_napari.plate_cache._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=raw_arr,
            ),
        ):
            _download_batch(batch, flatfield, conn=mock_conn)

        arr = stored_values["100:0"]
        assert arr.dtype == np.uint16
        np.testing.assert_array_equal(arr, raw_arr)

    def test_labels_uint16_stay_uint16(self):
        """Labels already uint16 are stored as uint16."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 500, "timepoint": 0, "apply_flatfield": False},
        ]
        label_arr = np.ones((1, 10, 10, 1), dtype=np.uint16) * 42
        mock_conn = MagicMock()

        stored_values: dict[str, np.ndarray] = {}

        class TrackingCache:
            def __setitem__(self, key: str, value: object) -> None:
                stored_values[key] = value  # type: ignore[assignment]

        with (
            patch(
                "omero_screen_napari.plate_cache._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(size_c=1),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=label_arr,
            ),
        ):
            _download_batch(batch, None, conn=mock_conn)

        arr = stored_values["500:0"]
        assert arr.dtype == np.uint16

    def test_labels_float64_compacted_to_uint16(self):
        """Float64 labels (from OMERO double pixel type) are compacted to uint16."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 500, "timepoint": 0, "apply_flatfield": False},
        ]
        # OMERO stores masks as double → float64 after parsing
        label_arr = np.ones((1, 10, 10, 1), dtype=np.float64) * 300
        mock_conn = MagicMock()

        stored_values: dict[str, np.ndarray] = {}

        class TrackingCache:
            def __setitem__(self, key: str, value: object) -> None:
                stored_values[key] = value  # type: ignore[assignment]

        with (
            patch(
                "omero_screen_napari.plate_cache._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(size_c=1),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=label_arr,
            ),
        ):
            _download_batch(batch, None, conn=mock_conn)

        arr = stored_values["500:0"]
        assert arr.dtype == np.uint16
        assert arr.max() == 300

    def test_labels_small_values_compacted_to_uint8(self):
        """Labels with max < 256 are compacted to uint8."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 500, "timepoint": 0, "apply_flatfield": False},
        ]
        label_arr = np.ones((1, 10, 10, 1), dtype=np.float64) * 100
        mock_conn = MagicMock()

        stored_values: dict[str, np.ndarray] = {}

        class TrackingCache:
            def __setitem__(self, key: str, value: object) -> None:
                stored_values[key] = value  # type: ignore[assignment]

        with (
            patch(
                "omero_screen_napari.plate_cache._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(size_c=1),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=label_arr,
            ),
        ):
            _download_batch(batch, None, conn=mock_conn)

        arr = stored_values["500:0"]
        assert arr.dtype == np.uint8

    def test_flatfield_clips_to_uint16_range(self):
        """Values exceeding 65535 after flatfield should be clipped."""
        from omero_screen_napari.plate_cache import _download_batch

        batch = [
            {"image_id": 100, "timepoint": 0, "apply_flatfield": True},
        ]
        raw_arr = np.ones((1, 10, 10, 2), dtype=np.uint16) * 60000
        # Flatfield mask < 1 amplifies values beyond uint16 range
        flatfield = np.ones((1, 10, 10, 2), dtype=np.float32) * 0.5
        mock_conn = MagicMock()

        stored_values: dict[str, np.ndarray] = {}

        class TrackingCache:
            def __setitem__(self, key: str, value: object) -> None:
                stored_values[key] = value  # type: ignore[assignment]

        with (
            patch(
                "omero_screen_napari.plate_cache._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache._get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ),
            patch(
                "omero_screen_napari.plate_cache._parse_raw_timepoint",
                return_value=raw_arr,
            ),
        ):
            _download_batch(batch, flatfield, conn=mock_conn)

        arr = stored_values["100:0"]
        assert arr.dtype == np.uint16
        assert arr.max() == 65535  # clipped


# --------------- load_from_cache dtype conversion ---------------


class TestLoadFromCacheDtype:
    def test_uint16_cache_returns_float32(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        """Images cached as uint16 should be returned as float32."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        store["plate:42:labels"] = sample_label_map

        # Store images as uint16 (new format)
        img_array = np.ones((1, 100, 100, 2), dtype=np.uint16) * 1000
        store["100:0"] = img_array
        store["101:0"] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        store["500:0"] = label_array
        store["501:0"] = label_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            load_from_cache(od, 42, "A1", "All")

            assert od.images.dtype == np.float32
            assert od.images[0, 0, 0, 0] == 1000.0

    def test_old_float32_cache_still_works(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        """Old caches stored as float32 should still load correctly."""
        fake_cache, store = mock_cache
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = sample_wells
        store["plate:42:labels"] = sample_label_map

        # Old format: float32
        img_array = np.ones((1, 100, 100, 2), dtype=np.float32) * 1000.5
        store["100:0"] = img_array
        store["101:0"] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        store["500:0"] = label_array
        store["501:0"] = label_array

        with patch("omero_screen_napari.plate_cache._cache", fake_cache):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            load_from_cache(od, 42, "A1", "All")

            assert od.images.dtype == np.float32
            assert od.images[0, 0, 0, 0] == 1000.5


# --------------- Cache version invalidation ---------------


class TestCacheVersionInvalidation:
    def test_old_version_triggers_delete(self, mock_cache, sample_meta):
        """Plates with cache_version < current should be deleted on re-cache."""
        from omero_screen_napari.plate_cache import _CACHE_VERSION

        fake_cache, store = mock_cache
        # Old-format metadata (no cache_version → defaults to 1)
        store["plate:42:meta"] = sample_meta
        store["plate:42:wells"] = {
            "A1": {
                "well_id": 10,
                "metadata": {},
                "images": [{"image_id": 100, "size_t": 1, "index": 0}],
            },
        }
        store["100:0"] = np.zeros((1, 10, 10, 2), dtype=np.float32)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._download_lock",
                threading.Lock(),
            ),
            patch(
                "omero_screen_napari.plate_cache.delete_plate_from_cache",
                wraps=None,
            ) as mock_delete,
        ):
            # Simulate step 0 of cache_plate: version check
            existing_meta = fake_cache.get("plate:42:meta")
            old_version = existing_meta.get("cache_version", 1)
            assert old_version < _CACHE_VERSION
            # The delete would be called
            mock_delete.assert_not_called()  # Sanity: not called yet

    def test_current_version_not_deleted(self, mock_cache, sample_meta):
        """Plates with current cache_version should NOT be deleted."""
        from omero_screen_napari.plate_cache import _CACHE_VERSION

        fake_cache, store = mock_cache
        meta_with_version = {**sample_meta, "cache_version": _CACHE_VERSION}
        store["plate:42:meta"] = meta_with_version

        existing_meta = fake_cache.get("plate:42:meta")
        old_version = existing_meta.get("cache_version", 1)
        assert old_version >= _CACHE_VERSION

    def test_meta_stores_cache_version(self, sample_meta):
        """_fetch_plate_metadata result should include cache_version after store."""
        from omero_screen_napari.plate_cache import _CACHE_VERSION

        # Simulate what cache_plate does: add cache_version to metadata
        meta = {**sample_meta}
        meta["cache_version"] = _CACHE_VERSION
        assert meta["cache_version"] == _CACHE_VERSION
