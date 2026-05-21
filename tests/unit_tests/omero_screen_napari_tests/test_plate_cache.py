"""Tests for the plate_cache module."""

import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omero_screen_napari.omero_data import OmeroData

# Duplicate cache key methods.
# This avoids importing the module under test allowing mock setup
# before a test executes.
_HISTORY_KEY = b"history"


def _get_bytes(plate_id: int) -> bytes:
    """Get the plate id encode as bytes

    Decode using int.from_bytes().
    """
    # ceil(bit_length / 8)
    size = (plate_id.bit_length() + 7) >> 3
    return plate_id.to_bytes(size)


def _get_meta_key(plate_id: int) -> bytes:
    """Get the key for the plate metadata."""
    return b"m" + _get_bytes(plate_id)


def _get_well_key(plate_id: int) -> bytes:
    """Get the key for the plate well data."""
    return b"w" + _get_bytes(plate_id)


def _get_label_key(plate_id: int) -> bytes:
    """Get the key for the plate label data."""
    return b"l" + _get_bytes(plate_id)


def get_key(image_id: int, t: int) -> str | int | bytes:
    """Get the image key for the cache."""
    return image_id << 16 | t


def exhaust(generator):
     for _ in generator:
         pass


# --------------- Fixtures ---------------


@pytest.fixture
def mock_cache():
    """Provide a dict-like mock for the diskcache.Cache.

    Provides mocks for the plate cache and the image cache.
    """
    class FakeCache:
        def __init__(self, size_limit: int) -> None:
            self.size_limit = size_limit
            self.store: dict[str, tuple[object, object | None]] = {}

        def set(self, key: str, value: object, tag: object | None = None):
            self.store[key] = (value, tag)

        def get(self, key: str, default=None):
            v = self.store.get(key)
            if v is not None:
                return v[0]
            return default

        def __setitem__(self, key: str, value: object) -> None:
            self.set(key, value)

        def __getitem__(self, key: str) -> object:
            v = self.store.get(key)
            if v is not None:
                return v[0]
            raise KeyError(key)

        def __contains__(self, key: str) -> bool:
            return key in self.store

        def __iter__(self):
            return self.store.__iter__()

        def pop(self, key: str, default=None):
            return self.store.pop(key, default)

        def volume(self) -> int:
            total = 0
            print('checking plate volume')
            for v in self.store.values():
                if isinstance(v[0], np.ndarray):
                    total += v[0].nbytes
                else:
                    total += 100  # small estimate for unknown types
            print('checking plate volume', total)
            return total

        def evict(self, tag: object):
            remove = [k for k, v in self.store.items() if v[1] == tag]
            for k in remove:
                del self.store[k]
            return len(remove)

    return FakeCache(2**30), FakeCache(20 * 2**30)


@pytest.fixture
def sample_meta() -> dict:
    from omero_screen_napari.plate_cache import _CACHE_VERSION
    return {
        "channel_data": {"DAPI": "0", "Tub": "1"},
        "pixel_size": (0.3, 0.3),
        "intensities": {0: (100, 5000), 1: (50, 3000)},
        "plate_name": "TestPlate",
        "ff_mask_id": 999,
        "label_stitched_mode": False,
        "cache_version": _CACHE_VERSION,
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
                    "dims": (1, 2, 1, 100, 100),
                    "pos": (0.0, 0.0),
                },
                {
                    "image_id": 101,
                    "dims": (1, 2, 1, 100, 100),
                    "pos": (1.0, 0.0),
                },
            ],
        },
        "A2": {
            "well_id": 11,
            "metadata": {"cell_line": "RPE", "condition": "drug"},
            "images": [
                {
                    "image_id": 200,
                    "dims": (1, 2, 1, 100, 100),
                    "pos": (0.0, 0.0),
                },
            ],
        },
    }


@pytest.fixture
def sample_label_map() -> dict:
    return {
        "A1": [
            {"image_id": 500, "dims": (1, 2, 1, 100, 100)},
            {"image_id": 501, "dims": (1, 2, 1, 100, 100)},
        ],
        "A2": [{"image_id": 600, "dims": (1, 2, 1, 100, 100)}],
    }


# --------------- is_plate_cached ---------------


class TestIsPlateCached:
    def test_not_cached(self, mock_cache):
        fake_cache, fake_image_cache = mock_cache
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import is_plate_cached

            assert is_plate_cached(999) is False

    def test_only_meta_cached(self, mock_cache, sample_meta):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 123
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import is_plate_cached

            assert is_plate_cached(plate_id) is False

    def test_fully_cached(self, mock_cache, sample_meta, sample_wells):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 123
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import is_plate_cached

            assert is_plate_cached(plate_id) is True


# --------------- is_plate_fully_cached ---------------


class TestIsPlateFullyCached:
    def test_false_when_no_metadata(self, mock_cache):
        fake_cache, fake_image_cache = mock_cache
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import is_plate_fully_cached

            assert is_plate_fully_cached(999) is False

    def test_false_when_images_missing(
        self, mock_cache, sample_meta, sample_wells
    ):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        # Only cache one of two images for A1
        fake_image_cache[get_key(100, 0)] = np.zeros((1, 10, 10, 2), dtype=np.float32)
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import is_plate_fully_cached

            assert is_plate_fully_cached(plate_id) is False

    def test_true_when_all_images_cached(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        fake_image_cache[get_key(100, 0)] = img
        fake_image_cache[get_key(101, 0)] = img
        fake_image_cache[get_key(200, 0)] = img
        fake_image_cache[get_key(500, 0)] = img
        fake_image_cache[get_key(501, 0)] = img
        fake_image_cache[get_key(600, 0)] = img
        fake_image_cache[get_key(999, 0)] = img
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import is_plate_fully_cached

            assert is_plate_fully_cached(plate_id) is True

    def test_true_when_no_labels(
        self, mock_cache, sample_meta, sample_wells
    ):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        # Images with missing labels have a corresponding None entry
        fake_cache[_get_label_key(plate_id)] = {
            "A1": [None, None],
            "A2": [None],
        }
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        fake_image_cache[get_key(100, 0)] = img
        fake_image_cache[get_key(101, 0)] = img
        fake_image_cache[get_key(200, 0)] = img
        fake_image_cache[get_key(999, 0)] = img
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import is_plate_fully_cached

            assert is_plate_fully_cached(plate_id) is True

# --------------- get_cached_plate_metadata ---------------


class TestGetCachedMetadata:
    def test_returns_none_when_missing(self, mock_cache):
        fake_cache, fake_image_cache = mock_cache
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import (
                get_cached_plate_metadata,
            )

            assert get_cached_plate_metadata(999) is None

    def test_returns_metadata(self, mock_cache, sample_meta):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import (
                get_cached_plate_metadata,
            )

            result = get_cached_plate_metadata(plate_id)
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


# --------------- load_from_cache ---------------


class TestLoadFromCache:
    def test_load_single_well(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map

        # Add image data to cache
        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(101, 0)] = img_array
        # Add label data
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache[get_key(500, 0)] = label_array
        fake_image_cache[get_key(501, 0)] = label_array
        # Add flat-field correction image
        fake_image_cache[get_key(999, 0)] = label_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            exhaust(load_from_cache(MagicMock(), od, plate_id, "A1", "All", threading.Event()))

            assert od.plate_id == plate_id
            assert od.channel_data == {"DAPI": "0", "Tub": "1"}
            assert od.pixel_size == (0.3, 0.3)
            assert od.plate_name == "TestPlate"
            assert len(od.image_ids) == 2
            assert od.images.shape[0] == 2  # 2 images
            assert len(od.well_metadata_list) == 1
            assert od.well_metadata_list[0]["cell_line"] == "RPE"

    def test_load_restores_stitched_flag_true(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        """When metadata carries label_stitched_mode=True, restore it onto OmeroData."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        meta_with_flag = {**sample_meta, "label_stitched_mode": True}
        fake_cache[_get_meta_key(plate_id)] = meta_with_flag
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(101, 0)] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache[get_key(500, 0)] = label_array
        fake_image_cache[get_key(501, 0)] = label_array
        fake_image_cache[get_key(999, 0)] = label_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            exhaust(load_from_cache(MagicMock(), od, plate_id, "A1", "All", threading.Event()))

            assert od.label_stitched_mode is True

    def test_load_defaults_stitched_flag_false_when_missing(
        self, mock_cache, sample_wells, sample_label_map
    ):
        """Legacy caches without the key restore as False (legacy behaviour)."""
        from omero_screen_napari.plate_cache import _CACHE_VERSION

        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        # Build metadata without the label_stitched_mode key
        legacy_meta = {
            "channel_data": {"DAPI": "0", "Tub": "1"},
            "pixel_size": (0.3, 0.3),
            "intensities": {0: (100, 5000), 1: (50, 3000)},
            "plate_name": "TestPlate",
            "ff_mask_id": 999,
            "cache_version": _CACHE_VERSION,
        }
        fake_cache[_get_meta_key(plate_id)] = legacy_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(101, 0)] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache[get_key(500, 0)] = label_array
        fake_image_cache[get_key(501, 0)] = label_array
        fake_image_cache[get_key(999, 0)] = label_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            # Pre-set to True to confirm load_from_cache resets it
            od.label_stitched_mode = True
            exhaust(load_from_cache(MagicMock(), od, plate_id, "A1", "All", threading.Event()))

            assert od.label_stitched_mode is False

    def test_load_uses_connection_when_not_cached(self, mock_cache):
        fake_cache, fake_image_cache = mock_cache
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            # Connection will be used if not fully cached.
            # Throw an exception when the connection is used to get the plate.
            mock_conn = MagicMock()
            msg = "Boom!"
            mock_conn.get_conn.side_effect = Exception(msg)

            od = OmeroData()
            with pytest.raises(Exception, match=msg):
                exhaust(load_from_cache(mock_conn, od, 999, "A1", "All", threading.Event()))

    def test_load_from_cache_stores_positions(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        """Verify that image positions are populated from cached well data."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(101, 0)] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache[get_key(500, 0)] = label_array
        fake_image_cache[get_key(501, 0)] = label_array
        fake_image_cache[get_key(999, 0)] = label_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            exhaust(load_from_cache(MagicMock(), od, plate_id, "A1", "All", threading.Event()))

            assert len(od.image_positions) == 2
            assert od.image_positions[0] == (0.0, 0.0)
            assert od.image_positions[1] == (1.0, 0.0)

    def test_load_from_cache_null_positions(
        self, mock_cache, sample_meta, sample_label_map
    ):
        """Verify None is used when pos_x/pos_y are missing."""
        fake_cache, fake_image_cache = mock_cache

        wells_no_pos = {
            "A1": {
                "well_id": 10,
                "metadata": {"cell_line": "RPE"},
                "images": [
                    {"image_id": 100, "dims": (1, 0, 0, 0, 0)},
                    {"image_id": 101, "dims": (1, 0, 0, 0, 0)},
                ],
            },
        }
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = wells_no_pos
        fake_cache[_get_label_key(plate_id)] = sample_label_map

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(101, 0)] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache[get_key(500, 0)] = label_array
        fake_image_cache[get_key(501, 0)] = label_array
        fake_image_cache[get_key(999, 0)] = label_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            exhaust(load_from_cache(MagicMock(), od, plate_id, "A1", "All", threading.Event()))

            assert len(od.image_positions) == 2
            assert od.image_positions[0] is None
            assert od.image_positions[1] is None

    def test_load_with_image_selection(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache[get_key(500, 0)] = label_array
        fake_image_cache[get_key(999, 0)] = label_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            exhaust(load_from_cache(MagicMock(), od, plate_id, "A1", "0", threading.Event()))  # Only first image

            assert len(od.image_ids) == 1
            assert od.image_ids[0] == 100

    def test_load_with_stop_flag(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map

        # Require images to avoid using connection
        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(101, 0)] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache[get_key(500, 0)] = label_array
        fake_image_cache[get_key(501, 0)] = label_array
        fake_image_cache[get_key(999, 0)] = label_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            stop_flag = threading.Event()
            stop_flag.set()
            exhaust(load_from_cache(MagicMock(), od, plate_id, "A1", "All", stop_flag))

            assert len(od.image_ids) == 0


# --------------- _unwrap_length ---------------


class TestUnwrapLength:
    def test_plain_float(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length(123.5, None) == 123.5

    def test_int(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length(10, None) == 10.0

    def test_none(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length(None, None) is None

    def test_value_with_unit(self):
        from omero_screen_napari.plate_cache import _unwrap_length

        assert _unwrap_length(123.4, "MICROMETER") == 123.4
        assert _unwrap_length(123.4, "METER") == 123.4 * 1e6


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

    def test_nucleus_channel_keeps_original_name(self):
        """Original channel names (Hoechst, h2b_rfp, etc.) are preserved.

        Previously ``_parse_channel_data`` force-renamed the nucleus
        channel to ``DAPI``, which masked the biological marker. The
        downstream code now uses :func:`resolve_channel_roles` to find
        the nucleus, so the original label can be kept.
        """
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
            assert "Hoechst" in result
            assert result["Hoechst"] == "0"
            assert "DAPI" not in result


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
        fake_cache, fake_image_cache = mock_cache
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import get_all_cached_plates

            assert get_all_cached_plates() == []

    def test_finds_cached_plates(self, mock_cache, sample_meta):
        fake_cache, fake_image_cache = mock_cache
        fake_cache[_get_meta_key(123)] = sample_meta
        fake_cache[_get_meta_key(456)] = {**sample_meta, "plate_name": "OtherPlate"}
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import get_all_cached_plates

            plates = get_all_cached_plates()
            assert len(plates) == 2
            plate_ids = [p[0] for p in plates]
            assert 123 in plate_ids
            assert 456 in plate_ids

    def test_sorted_by_date_cached_descending(self, mock_cache, sample_meta):
        fake_cache, fake_image_cache = mock_cache
        fake_cache[_get_meta_key(100)] = {**sample_meta, "plate_name": "First", "cache_date": "2026-01-03"}
        fake_cache[_get_meta_key(200)] = {**sample_meta, "plate_name": "Second", "cache_date": "2026-01-01"}
        fake_cache[_get_meta_key(150)] = {**sample_meta, "plate_name": "Middle", "cache_date": "2026-01-02"}
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import get_all_cached_plates

            plates = get_all_cached_plates()
            assert plates[0] == (100, "First", "2026-01-03")
            assert plates[1] == (150, "Middle", "2026-01-02")
            assert plates[2] == (200, "Second", "2026-01-01")

    def test_skips_corrupt_metadata(self, mock_cache, sample_meta):
        fake_cache, fake_image_cache = mock_cache
        fake_cache[_get_meta_key(123)] = sample_meta
        fake_cache[_get_meta_key(999)] = "not_a_dict"
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import get_all_cached_plates

            plates = get_all_cached_plates()
            assert len(plates) == 1
            assert plates[0][0] == 123


# --------------- get_well_cache_status ---------------


class TestGetWellCacheStatus:
    def test_uncached_plate_returns_empty(self, mock_cache):
        fake_cache, fake_image_cache = mock_cache
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import get_well_cache_status

            assert get_well_cache_status(999) == {}

    def test_all_images_cached_returns_true(self, mock_cache, sample_meta, sample_wells, sample_label_map):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map
        # Cache all images for A1 and A2
        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(101, 0)] = img_array
        fake_image_cache[get_key(200, 0)] = img_array
        fake_image_cache[get_key(500, 0)] = img_array
        fake_image_cache[get_key(501, 0)] = img_array
        fake_image_cache[get_key(600, 0)] = img_array
        fake_image_cache[get_key(999, 0)] = img_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import get_well_cache_status

            status = get_well_cache_status(plate_id)
            assert status["A1"] is True
            assert status["A2"] is True

    def test_missing_images_returns_false(self, mock_cache, sample_meta, sample_wells, sample_label_map):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map
        # Only cache one image for A1 (missing 101:0)
        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(200, 0)] = img_array
        fake_image_cache[get_key(500, 0)] = img_array
        fake_image_cache[get_key(501, 0)] = img_array
        fake_image_cache[get_key(600, 0)] = img_array
        fake_image_cache[get_key(999, 0)] = img_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import get_well_cache_status

            status = get_well_cache_status(plate_id)
            assert status["A1"] is False
            assert status["A2"] is True

    def test_multi_timepoint_check(self, mock_cache, sample_meta):
        fake_cache, fake_image_cache = mock_cache
        wells = {
            "A1": {
                "well_id": 10,
                "metadata": {},
                "images": [
                    {"image_id": 100, "dims": (3, 0, 0, 0, 0)},
                ],
            },
        }
        labels = {
            "A1": [
                {"image_id": 500, "dims": (3, 0, 0, 0, 0)},
            ],
        }
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = wells
        fake_cache[_get_label_key(plate_id)] = labels
        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        # Cache only 2 of 3 timepoints
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(100,1)] = img_array
        fake_image_cache[get_key(500, 0)] = img_array
        fake_image_cache[get_key(500, 1)] = img_array
        fake_image_cache[get_key(500, 2)] = img_array
        fake_image_cache[get_key(999, 0)] = img_array

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import get_well_cache_status

            status = get_well_cache_status(plate_id)
            assert status["A1"] is False

        # Now add the missing timepoint
        fake_image_cache[get_key(100, 2)] = img_array
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import get_well_cache_status

            status = get_well_cache_status(plate_id)
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
    def test_workers_get_sorted_wells(self, mock_cache):
        """Wells should be distributed in sorted order (A1, A2, B1, ...)."""
        from omero_screen_napari.plate_cache import _well_sort_key

        wells = {
            "B1": {"images": [{"image_id": 300, "dims": (1, 0, 0, 0, 0)}]},
            "A2": {"images": [{"image_id": 200, "dims": (1, 0, 0, 0, 0)}]},
            "A1": {"images": [{"image_id": 100, "dims": (1, 0, 0, 0, 0)}]},
        }

        sorted_keys = sorted(wells.keys(), key=_well_sort_key)
        assert sorted_keys == ["A1", "A2", "B1"]


# --------------- delete_plate_from_cache ---------------


class TestDeletePlateFromCache:
    def test_deletes_all_keys(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache.set(_get_meta_key(plate_id), sample_meta, tag=plate_id)
        fake_cache.set(_get_well_key(plate_id), sample_wells, tag=plate_id)
        fake_cache.set(_get_label_key(plate_id), sample_label_map, tag=plate_id)

        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        fake_image_cache.set(get_key(100, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(101, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(200, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(500, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(501, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(600, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(999, 0), img, tag=plate_id)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            count = delete_plate_from_cache(plate_id)

        # 3 images + 3 labels + 1 flat-field mask 3 metadata = 10
        assert count == 10
        assert _get_meta_key(plate_id) not in fake_cache
        assert _get_well_key(plate_id) not in fake_cache
        assert _get_label_key(plate_id) not in fake_cache
        assert get_key(100, 0) not in fake_image_cache
        assert get_key(101, 0) not in fake_image_cache
        assert get_key(200, 0) not in fake_image_cache
        assert get_key(500, 0) not in fake_image_cache
        assert get_key(501, 0) not in fake_image_cache
        assert get_key(600, 0) not in fake_image_cache
        assert get_key(999, 0) not in fake_image_cache

    def test_returns_zero_for_missing_plate(self, mock_cache):
        fake_cache, fake_image_cache = mock_cache
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            assert delete_plate_from_cache(999) == 0

    def test_metadata_only_plate(self, mock_cache, sample_meta):
        """A plate with only metadata (no wells/labels) still gets cleaned up."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache.set(_get_meta_key(plate_id), sample_meta, tag=plate_id)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            count = delete_plate_from_cache(plate_id)

        assert count == 1  # Only the meta key existed
        assert _get_meta_key(plate_id) not in fake_cache

    def test_partial_images(self, mock_cache, sample_meta, sample_wells):
        """Deletes whatever keys exist, even if some images are missing."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache.set(_get_meta_key(plate_id), sample_meta, tag=plate_id)
        fake_cache.set(_get_well_key(plate_id), sample_wells, tag=plate_id)
        # Only 1 of 3 images cached
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        fake_image_cache.set(get_key(100, 0), img, tag=plate_id)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            count = delete_plate_from_cache(plate_id)

        # meta + wells + 1 image = 3 (labels key didn't exist)
        assert count == 3
        assert get_key(100, 0) not in fake_image_cache

    def test_does_not_affect_other_plates(
        self, mock_cache, sample_meta, sample_wells
    ):
        """Deleting one plate should not touch another plate's keys."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache.set(_get_meta_key(plate_id), sample_meta, tag=plate_id)
        fake_cache.set(_get_well_key(plate_id), sample_wells, tag=plate_id)
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        fake_image_cache.set(get_key(100, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(101, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(200, 0), img, tag=plate_id)
        # Plate 99
        other_meta = {**sample_meta, "plate_name": "Other"}
        fake_cache.set(_get_meta_key(99), other_meta, tag=99)
        fake_cache.set(_get_well_key(99), {
            "B1": {
                "well_id": 20,
                "metadata": {},
                "images": [{"image_id": 900, "dims": (1, 0, 0, 0, 0)}],
            }
        }, tag=99)
        fake_image_cache.set(get_key(900, 0), img, tag=99)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            delete_plate_from_cache(plate_id)

        assert _get_meta_key(99) in fake_cache
        assert _get_well_key(99) in fake_cache
        assert get_key(900, 0) in fake_image_cache


# --------------- ensure_cache_space ---------------


class TestEnsureCacheSpace:
    def test_no_eviction_when_space_available(self, mock_cache, sample_meta):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache.set(_get_meta_key(plate_id), sample_meta, tag=plate_id)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import ensure_cache_space

            evicted, _vol, evicted_flag = ensure_cache_space(1000, 0, 2**30)

        assert evicted == []
        assert evicted_flag == -1

    def test_evicts_oldest_plate_first(self, mock_cache, sample_meta):
        """With two plates, evicts the one with the smaller plate_id first."""
        fake_cache, fake_image_cache = mock_cache

        # Plate 100
        fake_cache.set(_get_meta_key(100), {**sample_meta, "cache_date": "2026-01-01"}, tag=100)
        fake_cache.set(_get_well_key(100), {
            "A1": {
                "well_id": 1,
                "metadata": {},
                "images": [{"image_id": 1000, "dims": (1, 0, 0, 0, 0)}],
            }
        }, tag=100)
        fake_image_cache.set(get_key(1000, 0), np.zeros((10, 10, 2), dtype=np.float32), tag=100)

        # Plate 200 (newer)
        fake_cache.set(_get_meta_key(200), {**sample_meta, "cache_date": "2026-02-01"}, tag=200)
        fake_cache.set(_get_well_key(200), {
            "A1": {
                "well_id": 2,
                "metadata": {},
                "images": [{"image_id": 2000, "dims": (1, 0, 0, 0, 0)}],
            }
        }, tag=200)
        fake_image_cache.set(get_key(2000, 0), np.zeros((10, 10, 2), dtype=np.float32), tag=200)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import ensure_cache_space

            evicted, _vol, evicted_flag = ensure_cache_space(1000, 0, 2000)

        # Plate 100 should be evicted first (smallest ID)
        assert 100 in evicted
        assert evicted_flag == 1
        assert get_key(1000, 0) not in fake_image_cache

    def test_evicts_multiple_plates_if_needed(self, mock_cache, sample_meta):
        fake_cache, fake_image_cache = mock_cache

        for plate_id in [10, 20, 30]:
            fake_cache.set(_get_meta_key(plate_id),  {
                **sample_meta,
                "plate_name": f"P{plate_id}",
            }, tag=plate_id)
            fake_cache.set(_get_well_key(plate_id), {
                "A1": {
                    "well_id": plate_id,
                    "metadata": {},
                    "images": [
                        {"image_id": plate_id * 100, "dims": (1, 0, 0, 0, 0)}
                    ],
                }
            }, tag=plate_id)
            fake_image_cache.set(f"{plate_id * 100}:0", np.zeros((10, 10, 2), dtype=np.float32), tag=plate_id)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import ensure_cache_space

            evicted, _vol, evicted_flag = ensure_cache_space(500, 0, 500)

        assert len(evicted) >= 1  # At least one plate evicted
        assert evicted_flag >= 1

    def test_respects_exclude_plate_ids(self, mock_cache, sample_meta):
        fake_cache, fake_image_cache = mock_cache

        for plate_id in [10, 20]:
            fake_cache.set(_get_meta_key(plate_id), {
                **sample_meta,
                "plate_name": f"P{plate_id}",
            }, tag=plate_id)
            fake_cache.set(_get_well_key(plate_id), {
                "A1": {
                    "well_id": plate_id,
                    "metadata": {},
                    "images": [
                        {"image_id": plate_id * 100, "dims": (1, 0, 0, 0, 0)}
                    ],
                }
            }, tag=plate_id)
            fake_image_cache.set(get_key(plate_id * 100, 0), np.zeros((10, 10, 2), dtype=np.float32), tag=plate_id)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import ensure_cache_space

            evicted = ensure_cache_space(500, 10, 500)

        # Plate 10 should NOT be evicted
        assert 10 not in evicted
        assert _get_meta_key(10) in fake_cache
        assert _get_well_key(10) in fake_cache
        assert get_key(1000, 0) in fake_image_cache


# --------------- clean_orphaned_plates ---------------


class TestCleanOrphanedPlates:
    def test_removes_incomplete_plates(
        self, mock_cache, sample_meta, sample_wells
    ):
        """Plate with <50% images should be cleaned."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache.set(_get_meta_key(plate_id), sample_meta, tag=plate_id)
        fake_cache.set(_get_well_key(plate_id), sample_wells, tag=plate_id)
        # Only 1 of 3 images cached = 33% completeness
        fake_image_cache.set(get_key(100, 0), np.zeros((1, 10, 10, 2), dtype=np.float32), tag=plate_id)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import clean_orphaned_plates

            cleaned = clean_orphaned_plates()

        assert plate_id in cleaned
        assert _get_meta_key(plate_id) not in fake_cache
        assert get_key(100, 0) not in fake_image_cache

    def test_keeps_complete_plates(
        self, mock_cache, sample_meta, sample_wells
    ):
        """Plate with >=50% images should be kept."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache.set(_get_meta_key(plate_id), sample_meta, tag=plate_id)
        fake_cache.set(_get_well_key(plate_id), sample_wells, tag=plate_id)
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        # 2 of 3 images = 67% completeness
        fake_image_cache.set(get_key(100, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(101, 0), img, tag=plate_id)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import clean_orphaned_plates

            cleaned = clean_orphaned_plates()

        assert cleaned == []
        assert _get_meta_key(plate_id) in fake_cache
        assert get_key(100, 0) in fake_image_cache
        assert get_key(101, 0) in fake_image_cache

    def test_respects_exclude_plate_ids(
        self, mock_cache, sample_meta, sample_wells
    ):
        """Excluded plates should not be cleaned even if incomplete."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        # 0 of 3 images = 0% completeness, but excluded

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import clean_orphaned_plates

            cleaned = clean_orphaned_plates(exclude_plate_ids={plate_id})

        assert cleaned == []
        assert _get_meta_key(plate_id) in fake_cache


# --------------- _estimate_plate_bytes ---------------


class TestEstimatePlateBytes:
    def test_uses_actual_dimensions(self):
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                "images": [
                    {
                        "image_id": 100,
                        "dims": (1, 3, 5, 512, 1024),
                    }
                ],
            }
        }
        result = _estimate_plate_bytes(wells)
        # 16-bit image and 32-bit flat-field mask
        im_size = 1 * 3 * 5 * 512 * 1024
        assert result == im_size * (2 + 4)

    def test_multiple_wells_and_timepoints(self):
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                "images": [
                    {
                        "image_id": 100,
                        "dims": (1, 3, 5, 512, 1024),
                    },
                    {
                        "image_id": 101,
                        # ignored
                        "dims": (10, 30, 50, 5120, 10240),
                    },
                ],
            },
            "A2": {
                # ignored
                "images": [
                    {
                        "image_id": 200,
                        # ignored
                        "dims": (100, 300, 500, 51200, 102400),
                    }
                ],
            },
        }
        result = _estimate_plate_bytes(wells)
        im_size = 1 * 3 * 5 * 512 * 1024
        # 16-bit image * (2 wells and 2 images/well) + 32-bit flat-field mask
        assert result == im_size * (2 * 4 + 4)

    def test_falls_back_when_dimensions_missing(self):
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                # no "dims"
                "images": [{"image_id": 100}],
            }
        }
        result = _estimate_plate_bytes(wells)
        # default: 4 channel 1080*1080 image
        im_size = 4 * 1080**2
        # 16-bit image and 32-bit flat-field mask
        assert result == im_size * (2 + 4)

    def test_includes_label_bytes(self):
        """Labels (segmentation masks) are counted in the estimate."""
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                "images": [
                    {
                        "image_id": 100,
                        "dims": (1, 3, 5, 512, 1024),
                    }
                ],
            }
        }
        label_map = {
            "A1": [
                {
                    "image_id": 500,
                    # Labels should be the same size as the images
                    # but the number of channels may differ.
                    # Here we test all label dimensions are used.
                    "dims": (10, 30, 50, 5120, 10240),
                }
            ],
        }
        result = _estimate_plate_bytes(wells, label_map)
        im_size = 1 * 3 * 5 * 512 * 1024
        label_size = 10 * 30 * 50 * 5120 * 10240
        # 16-bit image, 16-bit label and 32-bit flat-field mask
        assert result == im_size * (2 + 4) + label_size * 2

    def test_no_label_map_still_works(self):
        """Passing None for label_map only counts images."""
        from omero_screen_napari.plate_cache import _estimate_plate_bytes

        wells = {
            "A1": {
                "images": [
                    {
                        "image_id": 100,
                        "dims": (1, 3, 5, 512, 1024),
                    }
                ],
            }
        }
        result = _estimate_plate_bytes(wells, None)
        im_size = 1 * 3 * 5 * 512 * 1024
        # 16-bit image, 16-bit label and 32-bit flat-field mask
        assert result == im_size * (2 + 4)


# --------------- _plate_image_completeness ---------------


class TestPlateImageCompleteness:
    def test_no_wells_returns_zero(self, mock_cache):
        fake_cache, fake_image_cache = mock_cache
        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import (
                _plate_image_completeness,
            )

            assert _plate_image_completeness(999) == 0.0

    def test_all_present(self, mock_cache, sample_wells):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_well_key(plate_id)] = sample_wells
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        fake_image_cache[get_key(100, 0)] = img
        fake_image_cache[get_key(101, 0)] = img
        fake_image_cache[get_key(200, 0)] = img

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import (
                _plate_image_completeness,
            )

            assert _plate_image_completeness(plate_id) == 1.0

    def test_partial(self, mock_cache, sample_wells):
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_well_key(plate_id)] = sample_wells
        # 1 of 3 images
        fake_image_cache[get_key(100, 0)] = np.zeros((1, 10, 10, 2), dtype=np.float32)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import (
                _plate_image_completeness,
            )

            result = _plate_image_completeness(plate_id)
            assert abs(result - 1 / 3) < 0.01


# --------------- Label multi-timepoint ---------------


class TestLabelMultiTimepoint:
    def test_multi_timepoint_labels(self, mock_cache, sample_meta):
        """New dict format with size_t > 1 loads labels correctly."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = {
            "A1": {
                "well_id": 10,
                "metadata": {"cell_line": "RPE"},
                "images": [
                    {"image_id": 100, "dims": (3, 0, 0, 0, 0)},
                ],
            },
        }
        fake_cache[_get_label_key(plate_id)] = {
            "A1": [{"image_id": 500, "dims": (3, 0, 0, 0, 0)}],
        }

        img_array = np.random.rand(1, 100, 100, 2).astype(np.float32)
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(100,1)] = img_array
        fake_image_cache[get_key(100, 2)] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache[get_key(500, 0)] = label_array
        fake_image_cache[get_key(500, 1)] = label_array
        fake_image_cache[get_key(500, 2)] = label_array
        fake_image_cache[get_key(999, 0)] = np.ones((1, 100, 100, 2), dtype=np.int32)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            exhaust(load_from_cache(MagicMock(), od, plate_id, "A1", "All", threading.Event()))

            # load_from_cache only loads t=0 for labels, so 1 label
            assert od.labels.shape[0] == 1

    def test_delete_handles_multi_timepoint_labels(
        self, mock_cache, sample_meta
    ):
        """delete_plate_from_cache removes all label timepoints."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache.set(_get_meta_key(plate_id), sample_meta, tag=plate_id)
        fake_cache.set(_get_well_key(plate_id), {
            "A1": {
                "well_id": 10,
                "metadata": {},
                "images": [{"image_id": 100, "dims": (3, 0, 0, 0, 0)}],
            },
        }, tag=plate_id)
        fake_cache.set(_get_label_key(plate_id), {
            "A1": [{"image_id": 500, "dims": (3, 0, 0, 0, 0)}],
        }, tag=plate_id)
        img = np.zeros((1, 10, 10, 2), dtype=np.float32)
        fake_image_cache.set(get_key(100, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(100,1), img, tag=plate_id)
        fake_image_cache.set(get_key(100, 2), img, tag=plate_id)
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache.set(get_key(500, 0), img, tag=plate_id)
        fake_image_cache.set(get_key(500, 1), img, tag=plate_id)
        fake_image_cache.set(get_key(500, 2), img, tag=plate_id)
        fake_image_cache.set(get_key(999, 0), np.ones((1, 100, 100, 2), dtype=np.int32), tag=plate_id)

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import delete_plate_from_cache

            count = delete_plate_from_cache(plate_id)

        # meta + wells + labels + 3 image + 3 label timepoints + ff-mask = 10
        assert count == 10
        assert get_key(100, 0) not in fake_image_cache
        assert get_key(100,1) not in fake_image_cache
        assert get_key(100, 2) not in fake_image_cache
        assert get_key(500, 0) not in fake_image_cache
        assert get_key(500, 1) not in fake_image_cache
        assert get_key(500, 2) not in fake_image_cache
        assert get_key(999, 0) not in fake_image_cache


# --------------- Helpers for RawPixelsStore tests ---------------


def _make_wrapper_mock(
    size_z: int = 1,
    size_c: int = 2,
    size_y: int = 10,
    size_x: int = 10,
    pixel_type: str = "uint16",
) -> MagicMock:
    """Create a mock ImageWrapper with proper pixel attributes."""
    wrapper = MagicMock(name="wrapper")
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
        """get_omero_image_wrapper called once per unique image_id."""
        plate_id = 42
        batch = [(100, 0), (100,1), (200, 0), (200, 1)]
        arr = np.zeros((1, 10, 10, 2), dtype=np.float32)
        mock_conn = MagicMock()
        stop_flag = threading.Event()

        with (
            patch("omero_screen_napari.omero_image._cache"),
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ) as mock_get_wrapper,
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_timepoint",
                return_value=arr,
            ),
        ):
            from omero_screen_napari.plate_cache import _download_batch

            progress_q: queue.Queue[int] = queue.Queue()
            _download_batch(batch, plate_id, stop_flag, mock_conn, progress_q)

            # Should be called exactly 2 times: once for 100, once for 200
            assert mock_get_wrapper.call_count == 2
            conn = mock_conn.create_conn.return_value
            mock_get_wrapper.assert_any_call(conn, 100)
            mock_get_wrapper.assert_any_call(conn, 200)

    def test_fetches_each_unique_image_id(self):
        """Each unique image_id triggers one wrapper fetch."""
        plate_id = 42
        batch = [(100, 0), (200, 0), (300, 0)]
        arr = np.zeros((1, 10, 10, 2), dtype=np.float32)
        mock_conn = MagicMock()
        stop_flag = threading.Event()

        with (
            patch("omero_screen_napari.omero_image._cache"),
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ) as mock_get_wrapper,
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_timepoint",
                return_value=arr,
            ),
        ):
            from omero_screen_napari.plate_cache import _download_batch

            _download_batch(batch, plate_id, stop_flag, mock_conn)

            assert mock_get_wrapper.call_count == 3

    def test_store_reused_across_timepoints(self):
        """RawPixelsStore is kept open for consecutive timepoints of the same image."""
        plate_id = 42
        batch = [(100, 0), (100,1), (100, 2)]
        arr = np.zeros((1, 10, 10, 2), dtype=np.float32)
        mock_conn = MagicMock()
        stop_flag = threading.Event()

        with (
            patch("omero_screen_napari.omero_image._cache"),
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ) as mock_get_wrapper,
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_timepoint",
                return_value=arr,
            ) as mock_get_timepoint,
        ):
            from omero_screen_napari.plate_cache import _download_batch

            _download_batch(batch, plate_id, stop_flag, mock_conn)

            # Only one createRawPixelsStore call for the same image
            assert mock_get_wrapper.call_count == 1
            conn = mock_conn.create_conn.return_value
            assert conn.c.sf.createRawPixelsStore.call_count == 1
            store = conn.c.sf.createRawPixelsStore.return_value
            assert store.setPixelsId.call_count == 1
            # getTimepoint called 3 times (one per batch item)
            assert mock_get_timepoint.call_count == 3

    def test_store_recreated_for_different_images(self):
        """New RawPixelsStore created when image_id changes."""
        plate_id = 42
        batch = [(100, 0), (200, 0)]
        arr = np.zeros((1, 10, 10, 2), dtype=np.float32)
        mock_conn = MagicMock()
        stop_flag = threading.Event()

        with (
            patch("omero_screen_napari.omero_image._cache"),
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ) as mock_get_wrapper,
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_timepoint",
                return_value=arr,
            ),
        ):
            from omero_screen_napari.plate_cache import _download_batch

            _download_batch(batch, plate_id, stop_flag, mock_conn)

            # Two different images → two createRawPixelsStore calls
            conn = mock_conn.create_conn.return_value
            assert conn.c.sf.createRawPixelsStore.call_count == 2


# --------------- Download batch completeness ---------------


class TestDownloadBatchCompleteness:
    def test_downloads_all_items(self):
        """All items in batch are downloaded."""
        plate_id = 42
        batch = [(i, 0) for i in range(5)]
        arr = np.zeros((1, 10, 10, 2), dtype=np.uint16)
        mock_conn = MagicMock()
        stop_flag = threading.Event()

        cached_keys: list[str] = []

        class TrackingCache:
            def set(self, key, value, tag=None) -> None:
                cached_keys.append(key)

        with (
            patch(
                "omero_screen_napari.omero_image._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ) as mock_get_wrapper,
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_timepoint",
                return_value=arr,
            ),
        ):
            from omero_screen_napari.plate_cache import _download_batch

            _download_batch(batch, plate_id, stop_flag, mock_conn)

        expected = [get_key(*x) for x in batch]
        assert expected == cached_keys

    def test_stop_flag_set_prevents_download(self):
        """Stop flag prevents download."""
        plate_id = 42
        batch = [(i, 0) for i in range(5)]
        arr = np.zeros((1, 10, 10, 2), dtype=np.uint16)
        mock_conn = MagicMock()
        stop_flag = threading.Event()
        stop_flag.set()

        cached_keys: list[str] = []

        class TrackingCache:
            def set(self, key, value, tag=None) -> None:
                cached_keys.append(key)

        with (
            patch(
                "omero_screen_napari.omero_image._cache", TrackingCache()
            ),
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_wrapper",
                return_value=_make_wrapper_mock(),
            ) as mock_get_wrapper,
            patch(
                "omero_screen_napari.plate_cache.get_omero_image_timepoint",
                return_value=arr,
            ),
        ):
            from omero_screen_napari.plate_cache import _download_batch

            _download_batch(batch, plate_id, stop_flag, mock_conn)

        expected = []
        assert expected == cached_keys


# --------------- load_from_cache dtype conversion ---------------


class TestLoadFromCacheDtype:
    def test_uint16_cache_returns_float32(
        self, mock_cache, sample_meta, sample_wells, sample_label_map
    ):
        """Images cached as uint16 should be returned as float32."""
        fake_cache, fake_image_cache = mock_cache
        plate_id = 42
        fake_cache[_get_meta_key(plate_id)] = sample_meta
        fake_cache[_get_well_key(plate_id)] = sample_wells
        fake_cache[_get_label_key(plate_id)] = sample_label_map

        # Store images as uint16 (new format)
        img_array = np.ones((1, 100, 100, 2), dtype=np.uint16) * 1000
        fake_image_cache[get_key(100, 0)] = img_array
        fake_image_cache[get_key(101, 0)] = img_array
        label_array = np.ones((1, 100, 100, 1), dtype=np.int32)
        fake_image_cache[get_key(500, 0)] = label_array
        fake_image_cache[get_key(501, 0)] = label_array
        # Flat-field mask
        fake_image_cache[get_key(999, 0)] = np.ones((1, 100, 100, 2), dtype=np.float32) * 2

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch("omero_screen_napari.omero_image._cache", fake_image_cache),
        ):
            from omero_screen_napari.plate_cache import load_from_cache

            od = OmeroData()
            exhaust(load_from_cache(MagicMock(), od, plate_id, "A1", "All", threading.Event()))

            assert od.images.dtype == np.float32
            # image is 1000 / 2 = 500
            assert od.images[0, 0, 0, 0] == 500.0


# --------------- Cache version invalidation ---------------


class TestCacheVersionInvalidation:
    def test_old_version_triggers_delete(self, mock_cache, sample_meta):
        """Plates with cache_version < current should be deleted on re-cache."""
        from omero_screen_napari.plate_cache import _CACHE_VERSION

        plate_id = 42
        fake_cache, fake_image_cache = mock_cache
        # Old-format metadata
        meta_with_version = {**sample_meta, "cache_version": _CACHE_VERSION - 1}
        fake_cache.set(_get_meta_key(plate_id), meta_with_version, tag=plate_id)
        fake_cache.set(_get_well_key(plate_id), {
            "A1": {
                "well_id": 10,
                "metadata": {},
                "images": [{"image_id": 100, "dims": (1, 0, 0, 0, 0)}],
            },
        }, tag=plate_id)
        fake_image_cache.set(get_key(100, 0), np.zeros((1, 10, 10, 2), dtype=np.float32), tag=plate_id)
        mock_conn = MagicMock()

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
            patch(
                "omero_screen_napari.plate_cache._fetch_plate_metadata",
                return_value=sample_meta,
            )
        ):
            from omero_screen_napari.plate_cache import get_plate_metadata

            meta = get_plate_metadata(mock_conn, plate_id)

            # Version check
            existing_meta = fake_cache.get(_get_meta_key(plate_id))
            assert existing_meta.get("cache_version") == _CACHE_VERSION
            # Well metadata would be evicted
            assert fake_cache.get(_get_well_key(plate_id)) is None
            # Images remain
            assert get_key(100, 0) in fake_image_cache

    def test_current_version_not_deleted(self, mock_cache, sample_meta):
        """Plates with current cache_version should NOT be deleted."""
        from omero_screen_napari.plate_cache import _CACHE_VERSION

        plate_id = 42
        fake_cache, fake_image_cache = mock_cache
        # Current-format metadata
        meta_with_version = {**sample_meta, "cache_version": _CACHE_VERSION}
        fake_cache.set(_get_meta_key(plate_id), meta_with_version, tag=plate_id)
        fake_cache.set(_get_well_key(plate_id), {
            "A1": {
                "well_id": 10,
                "metadata": {},
                "images": [{"image_id": 100, "dims": (1, 0, 0, 0, 0)}],
            },
        }, tag=plate_id)
        fake_image_cache.set(get_key(100, 0), np.zeros((1, 10, 10, 2), dtype=np.float32), tag=plate_id)
        mock_conn = MagicMock()

        with (
            patch("omero_screen_napari.plate_cache._cache", fake_cache),
        ):
            from omero_screen_napari.plate_cache import get_plate_metadata

            meta = get_plate_metadata(mock_conn, plate_id)

            # Version check
            existing_meta = fake_cache.get(_get_meta_key(plate_id))
            assert existing_meta.get("cache_version") == _CACHE_VERSION
            # Well metadata would be retained
            assert _get_well_key(plate_id) in fake_cache
            # Images remain
            assert get_key(100, 0) in fake_image_cache


class TestDetectLabelStitchedMode:
    """``_detect_label_stitched_mode`` scans the segmentation dataset names."""

    def test_true_when_stitched_mask_present(self):
        from omero_screen_napari.plate_cache import _detect_label_stitched_mode

        child_a = MagicMock()
        child_a.getName.return_value = "10_segmentation"
        child_b = MagicMock()
        child_b.getName.return_value = "11_stitched_segmentation"

        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = iter([child_a, child_b])

        mock_conn = MagicMock()
        mock_conn.getObject.return_value = mock_dataset

        with patch(
            "omero_screen_napari.plate_cache.get_dataset_id", return_value=999
        ):
            assert _detect_label_stitched_mode(mock_conn, plate_id=42) is True

    def test_false_when_only_legacy_masks(self):
        from omero_screen_napari.plate_cache import _detect_label_stitched_mode

        child_a = MagicMock()
        child_a.getName.return_value = "10_segmentation"
        child_b = MagicMock()
        child_b.getName.return_value = "11_segmentation"

        mock_dataset = MagicMock()
        mock_dataset.listChildren.return_value = iter([child_a, child_b])

        mock_conn = MagicMock()
        mock_conn.getObject.return_value = mock_dataset

        with patch(
            "omero_screen_napari.plate_cache.get_dataset_id", return_value=999
        ):
            assert _detect_label_stitched_mode(mock_conn, plate_id=42) is False

    def test_false_when_dataset_missing(self):
        from omero_screen_napari.plate_cache import _detect_label_stitched_mode

        mock_conn = MagicMock()
        with patch(
            "omero_screen_napari.plate_cache.get_dataset_id", return_value=None
        ):
            assert _detect_label_stitched_mode(mock_conn, plate_id=42) is False
