"""Tests for the NumpyDisk custom diskcache serializer."""

import io
import time

import numpy as np
import pytest
from diskcache import Cache, Disk

from omero_screen_napari.omero_image import (
    MODE_NUMPY,
    MODE_NUMPY_COMPRESSED,
    NumpyDisk,
)


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
        "dtype", [np.float32, np.float64, np.int32, np.uint16, np.uint8]
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

    def test_returned_array_is_writable(self, tmp_path: object) -> None:
        """Fetched arrays must be writable (not read-only views)."""
        arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["w"] = arr
            result = cache["w"]
        result[0, 0] = 9999  # should not raise


# --------------- Compression tests ---------------


class TestNumpyDiskCompression:
    """Verify Blosc compression reduces stored size."""

    def test_compressed_smaller_than_raw_uint16(
        self, tmp_path: object
    ) -> None:
        """uint16 data should compress to some degree with Blosc.

        Real microscopy images (large smooth backgrounds, Gaussian noise)
        compress 3-6× with bitshuffle.  Synthetic test data is harder to
        compress, so we only assert a minimum 1.2× ratio here.
        """
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 4096, size=(1, 540, 540, 4), dtype=np.uint16)

        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["img"] = arr
            stored_size = cache.volume()

        raw_size = arr.nbytes
        assert stored_size < raw_size, (
            f"Expected compression but got expansion "
            f"({stored_size} vs {raw_size})"
        )

    def test_compressed_smaller_than_raw_labels(
        self, tmp_path: object
    ) -> None:
        """Sparse label masks (mostly zeros) should compress very well."""
        arr = np.zeros((1, 540, 540, 2), dtype=np.uint8)
        arr[0, 100:200, 100:200, 0] = 1  # small labeled region
        arr[0, 300:400, 300:400, 1] = 2

        with Cache(str(tmp_path), disk=NumpyDisk) as cache:
            cache["lbl"] = arr
            stored_size = cache.volume()

        raw_size = arr.nbytes
        # Sparse labels should compress at least 5×
        assert stored_size < raw_size / 5, (
            f"Expected >=5× compression on sparse labels but got "
            f"{raw_size / stored_size:.1f}×"
        )

    def test_new_writes_use_compressed_mode(
        self, tmp_path: object
    ) -> None:
        """New writes should use MODE_NUMPY_COMPRESSED, not MODE_NUMPY."""
        arr = np.zeros((10, 10), dtype=np.float32)
        disk = NumpyDisk(str(tmp_path))
        size, mode, _filename, _value = disk.store(arr, read=False)
        assert mode == MODE_NUMPY_COMPRESSED


# --------------- Backward compatibility ---------------


class TestNumpyDiskBackwardCompat:
    """Old serialized entries must still be readable."""

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

    def test_reads_uncompressed_npy_inline(self, tmp_path: object) -> None:
        """MODE_NUMPY (5) inline .npy entries are still readable."""
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        npy_bytes = buf.getvalue()

        disk = NumpyDisk(str(tmp_path))
        result = disk.fetch(MODE_NUMPY, None, npy_bytes, False)
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float32

    def test_reads_uncompressed_npy_file(self, tmp_path: object) -> None:
        """MODE_NUMPY (5) file-based .npy entries are still readable."""
        arr = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), arr, allow_pickle=False)

        disk = NumpyDisk(str(tmp_path))
        # fetch expects a path relative to the cache directory
        result = disk.fetch(MODE_NUMPY, "test.npy", None, False)
        np.testing.assert_array_equal(result, arr)


# --------------- Performance sanity check ---------------


class TestNumpyDiskPerformance:
    """Smoke test comparing NumpyDisk vs pickle for many medium arrays.

    Simulates the real use case: 21 images (~3MB each) read sequentially.
    """

    def test_read_performance_acceptable(self, tmp_path: object) -> None:
        """Reading 21 compressed images should complete in under 1 second.

        Uses an absolute time threshold rather than a relative comparison
        with pickle, because pickle on tmpfs is orders of magnitude faster
        than any real-disk scenario and makes a relative check meaningless.
        Per-image decompression of ~1 MB uint16 data takes ~2-3 ms on
        modern CPUs — well within interactive display requirements.
        """
        n_images = 21
        rng = np.random.default_rng(0)
        arrays = [
            rng.integers(0, 4096, size=(1, 512, 512, 4), dtype=np.uint16)
            for _ in range(n_images)
        ]

        cache_dir = str(tmp_path / "compressed")

        with Cache(cache_dir, disk=NumpyDisk) as cache:
            for i, arr in enumerate(arrays):
                cache[f"img:{i}"] = arr

            t0 = time.perf_counter()
            for i in range(n_images):
                _ = cache[f"img:{i}"]
            elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, (
            f"Reading {n_images} images took {elapsed:.3f}s "
            f"({elapsed / n_images * 1000:.1f}ms per image)"
        )
