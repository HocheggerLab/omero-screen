Loading a cached well (21 images) and stitching it takes 10-20s. Profiling
 reveals two equally expensive bottlenecks:

 1. Pickle deserialization (~5-8s): 21 × pickle.loads() on ~64 MB float32
 arrays. Pickle walks the object graph, calls __setstate__, etc.
 2. Sequential scipy rotation (~5-8s): 84 independent transform.rotate()
 calls (21 images × 4 channels) running on a single core.

 This phase tackles bottleneck #1 by replacing pickle with numpy's native .npy
 binary format for array serialization. This avoids pickle overhead entirely —
 .npy is just a 128-byte header + raw buffer memcpy.

 Expected speedup: Deserialization from ~5-8s → ~1-2s (3-5x faster).

 ---
 Design: Custom NumpyDisk Subclass

 diskcache.Cache accepts a disk parameter (subclass of diskcache.Disk)
 that controls serialization. We subclass Disk and override store()/fetch()
 to use numpy.save()/numpy.load() for numpy arrays, falling back to the
 default pickle path for everything else (metadata dicts, etc.).

 Mode constant

 Diskcache uses integer modes to tag how each value was serialized:
 - MODE_RAW = 1 (bytes), MODE_BINARY = 2 (file), MODE_TEXT = 3,
 MODE_PICKLE = 4

 We define MODE_NUMPY = 5 for our custom format.

 Backward compatibility

 Existing caches contain pickle-serialized arrays (mode=4). The custom fetch()
 delegates to super().fetch() for any mode that isn't MODE_NUMPY, so old
 entries are read correctly via pickle. New writes use .npy format. Over time,
 as entries are overwritten, the cache naturally migrates.

 No cache wipe required.

 ---
 Step 1: Create NumpyDisk class

 File: packages/omero-screen-napari/src/omero_screen_napari/omero_image.py

 Add NumpyDisk in the same file where _cache is created (keeps cache
 configuration in one place).

 import io
 import sqlite3

 import numpy as np
 from diskcache import Cache, Disk
 from diskcache.core import UNKNOWN

 MODE_NUMPY = 5


 class NumpyDisk(Disk):
     """Diskcache Disk that uses numpy .npy format for ndarray values.

     Falls back to the default pickle serialization for non-array values
     (dicts, strings, etc.) and for reading old pickle-serialized entries.
     """

     def store(self, value, read, key=UNKNOWN):
         if isinstance(value, np.ndarray):
             buf = io.BytesIO()
             np.save(buf, value, allow_pickle=False)
             npy_bytes = buf.getvalue()
             size = len(npy_bytes)
             if size < self.min_file_size:
                 return size, MODE_NUMPY, None, sqlite3.Binary(npy_bytes)
             # Large array → write to file
             filename, full_path = self.filename()
             with open(full_path, "xb") as f:
                 f.write(npy_bytes)
             return size, MODE_NUMPY, filename, None
         return super().store(value, read, key)

     def fetch(self, mode, filename, value, read):
         if mode == MODE_NUMPY:
             if filename is not None:
                 with open(filename, "rb") as f:
                     return np.load(io.BytesIO(f.read()), allow_pickle=False)
             return np.load(io.BytesIO(value), allow_pickle=False)
         return super().fetch(mode, filename, value, read)

 Key details:
 - self.min_file_size and self.filename() are public API from diskcache.Disk
 - filename() returns (filename, full_path) — the relative and absolute paths
 - fetch() receives filename as the full path (already resolved by diskcache)
 - allow_pickle=False prevents numpy from falling back to pickle internally
 - "xb" (exclusive create) matches diskcache's pattern for new files

 Wire into Cache constructor

 Change the _cache initialization:

 _cache = Cache(
     __path,
     disk=NumpyDisk,
     size_limit=getenv_as_int(
         "OMERO_SCREEN_IMAGE_CACHE_SIZE_LIMIT", 20 * 2**30
     ),
 )

 ---
 Step 2: Verify diskcache Disk API details

 Before implementing, verify these assumptions against the installed
 diskcache==5.6.3:

 - Disk.min_file_size is a public attribute (not _min_file_size)
 - Disk.filename() returns (filename, full_path) tuple
 - fetch() receives the full resolved path in the filename parameter
 - Mode constants: check existing mode values don't conflict with 5

 Action: Read diskcache/core.py from the installed package to confirm.

 ---
 Step 3: Tests

 File: tests/unit_tests/omero_screen_napari_tests/test_numpy_disk.py (new)

 Roundtrip tests with real diskcache (not mocks)

 These tests use a temporary Cache(disk=NumpyDisk) to verify actual
 serialization/deserialization:

 class TestNumpyDiskRoundtrip:
     - test_numpy_array_roundtrip: Store float32 array, read back, assert equal
     - test_numpy_array_preserves_dtype: float32, float64, int32, uint16
     - test_numpy_array_preserves_shape: Various shapes (Z,Y,X,C), (Y,X,C), etc.
     - test_dict_roundtrip: Metadata dicts still work (pickle fallback)
     - test_string_roundtrip: String values still work
     - test_large_array_file_storage: Array > min_file_size stored as file
     - test_small_array_inline_storage: Array < min_file_size stored inline

 Backward compatibility test

 class TestNumpyDiskBackwardCompat:
     - test_reads_pickle_entries: Write with default Disk, read with NumpyDisk

 Create a temp cache with default Disk, write a numpy array (pickle mode),
 close it, reopen with NumpyDisk, verify the old entry reads correctly.

 Performance sanity check (not a benchmark, just a smoke test)

 class TestNumpyDiskPerformance:
     - test_numpy_faster_than_pickle: Write/read a 100MB array with both
       Disk and NumpyDisk, assert NumpyDisk is at least 1.5x faster

 Update existing FakeCache mock

 The existing FakeCache in test_plate_cache.py doesn't go through
 serialization (it's a dict), so no changes needed to existing tests.

 ---
 Step 4: Update existing tests if needed

 The FakeCache mock in test_plate_cache.py bypasses serialization entirely
 (stores raw Python objects in a dict). No changes should be needed to existing
 tests since they don't test the serialization layer.

 Run the full test suite to confirm nothing breaks:
 pytest tests/unit_tests/ -v

 ---
 Files Summary
 ┌────────────────────┬────────┬─────────────────────────────────────────────────────┐
 │        File        │ Action │                     Description                     │
 ├────────────────────┼────────┼─────────────────────────────────────────────────────┤
 │ omero_image.py     │ MODIFY │ Add NumpyDisk class, pass disk=NumpyDisk to Cache() │
 ├────────────────────┼────────┼─────────────────────────────────────────────────────┤
 │ test_numpy_disk.py │ CREATE │ Roundtrip, backward-compat, and performance tests   │
 └────────────────────┴────────┴─────────────────────────────────────────────────────┘
 ---
 Verification

 1. pytest tests/unit_tests/ — all existing tests pass
 2. pytest tests/unit_tests/omero_screen_napari_tests/test_numpy_disk.py — new tests pass
 3. Manual: Clear cache (rm -rf ~/.cache/omero_screen/images/), cache a
 plate, load a well — verify images display correctly
 4. Manual: With an OLD cache (pickle entries), load a well — verify backward
 compat (old entries read without error)
 5. Manual: Time a well load before/after — expect ~3-5s improvement on
 the deserialization step

 ---
 Future Work (not in this phase)

 - Parallel rotation: concurrent.futures.ThreadPoolExecutor for the 84
 independent skimage.transform.rotate() calls in compose_tiles(). Skimage
 releases the GIL so threads give real parallelism. Expected: ~5-8s → ~1-2s.
 - Rust stitching kernel: For near-instant experience, rewrite the
 rotate+blend loop in Rust via PyO3. Would combine rotation parallelism with
 SIMD-accelerated interpolation.
