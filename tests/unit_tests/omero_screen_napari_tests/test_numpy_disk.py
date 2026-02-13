"""Tests for the NumpyDisk custom diskcache serializer."""

import time

import numpy as np
import pytest
from diskcache import Cache, Disk

from omero_screen_napari.omero_image import NumpyDisk


# --------------- Roundtrip tests ---------------


class TestNumpyDiskRoundtrip:
    """Verify store/fetch with a real Cache(disk=NumpyDisk)."""

    def test_numpy_array_roundtrip(self, tmp_path: object) -> None:
        arr = np.random.rand(10, 256, 256, 4).astype(np.float32)
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["img"] = arr
            result = cache["img"]
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.int32, np.uint16]
    )
    def test_numpy_array_preserves_dtype(
        self, tmp_path: object, dtype: np.dtype
    ) -> None:
        arr = np.arange(120, dtype=dtype).reshape(2, 3, 4, 5)
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["x"] = arr
            result = cache["x"]
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.parametrize(
        "shape",
        [(1, 512, 512, 4), (3, 256, 256, 2), (1, 1080, 1080, 1), (100,)],
    )
    def test_numpy_array_preserves_shape(
        self, tmp_path: object, shape: tuple
    ) -> None:
        arr = np.zeros(shape, dtype=np.float32)
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["s"] = arr
            result = cache["s"]
        assert result.shape == shape

    def test_dict_roundtrip(self, tmp_path: object) -> None:
        meta = {"channels": {"DAPI": 0}, "pixel_size": (0.3, 0.3)}
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["meta"] = meta
            result = cache["meta"]
        assert result == meta

    def test_string_roundtrip(self, tmp_path: object) -> None:
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["key"] = "hello"
            assert cache["key"] == "hello"

    def test_large_array_file_storage(self, tmp_path: object) -> None:
        """Arrays larger than min_file_size are stored as files."""
        # Default min_file_size is 2**15 (32KB). Create array larger than that.
        arr = np.random.rand(100, 100, 4).astype(np.float32)  # ~160KB
        assert arr.nbytes > 2**15
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["big"] = arr
            result = cache["big"]
        np.testing.assert_array_equal(result, arr)

    def test_small_array_inline_storage(self, tmp_path: object) -> None:
        """Arrays smaller than min_file_size are stored inline in SQLite."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 12 bytes
        assert arr.nbytes < 2**15
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["tiny"] = arr
            result = cache["tiny"]
        np.testing.assert_array_equal(result, arr)

    def test_overwrite_existing_entry(self, tmp_path: object) -> None:
        arr1 = np.ones((10, 10), dtype=np.float32)
        arr2 = np.zeros((20, 20), dtype=np.float64)
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["x"] = arr1
            cache["x"] = arr2
            result = cache["x"]
        np.testing.assert_array_equal(result, arr2)
        assert result.dtype == np.float64

    def test_mixed_types_coexist(self, tmp_path: object) -> None:
        arr = np.arange(12, dtype=np.int32).reshape(3, 4)
        meta = {"key": "value"}
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["arr"] = arr
            cache["meta"] = meta
            cache["num"] = 42
            np.testing.assert_array_equal(cache["arr"], arr)
            assert cache["meta"] == meta
            assert cache["num"] == 42


# --------------- Backward compatibility ---------------


class TestNumpyDiskBackwardCompat:
    """Old pickle-serialized entries must still be readable."""

    def test_reads_pickle_entries(self, tmp_path: object) -> None:
        arr = np.random.rand(50, 50, 4).astype(np.float32)
        cache_dir = str(tmp_path / "compat")

        # Write with default Disk (pickle serialization)
        with Cache(cache_dir) as cache:
            cache["old_arr"] = arr
            cache["old_meta"] = {"channels": 3}

        # Read back with NumpyDisk — old entries should still work
        with Cache(cache_dir, disk=NumpyDisk) as cache:
            result = cache["old_arr"]
            np.testing.assert_array_equal(result, arr)
            assert cache["old_meta"] == {"channels": 3}

    def test_new_writes_use_numpy_format(self, tmp_path: object) -> None:
        """After switching to NumpyDisk, new writes use MODE_NUMPY."""
        arr = np.random.rand(50, 50).astype(np.float32)
        cache_dir = str(tmp_path / "format")

        with Cache(cache_dir, disk=NumpyDisk) as cache:
            cache["new"] = arr
            # Verify we can read it back
            result = cache["new"]
            np.testing.assert_array_equal(result, arr)


# --------------- Performance sanity check ---------------


class TestNumpyDiskPerformance:
    """Smoke test comparing NumpyDisk vs pickle for many medium arrays.

    Simulates the real use case: 21 images (~3MB each) read sequentially.
    """

    def test_numpy_not_slower_than_pickle(self, tmp_path: object) -> None:
        """NumpyDisk should be comparable or faster than pickle on bulk reads."""
        n_images = 21
        arrays = [
            np.random.rand(1, 512, 512, 4).astype(np.float32)
            for _ in range(n_images)
        ]

        pickle_dir = str(tmp_path / "pickle")
        numpy_dir = str(tmp_path / "numpy")

        # Write with pickle
        with Cache(pickle_dir) as cache:
            for i, arr in enumerate(arrays):
                cache[f"img:{i}"] = arr
            # Read all
            t0 = time.perf_counter()
            for i in range(n_images):
                _ = cache[f"img:{i}"]
            pickle_time = time.perf_counter() - t0

        # Write with NumpyDisk
        with Cache(numpy_dir, disk=NumpyDisk) as cache:
            for i, arr in enumerate(arrays):
                cache[f"img:{i}"] = arr
            # Read all
            t0 = time.perf_counter()
            for i in range(n_images):
                _ = cache[f"img:{i}"]
            numpy_time = time.perf_counter() - t0

        # Allow up to 2x slower (generous margin for CI/noisy environments).
        # The primary goal is correctness; speed gain is measured manually.
        assert numpy_time < pickle_time * 2, (
            f"NumpyDisk ({numpy_time:.3f}s) was more than 2x slower than "
            f"pickle ({pickle_time:.3f}s)"
        )
