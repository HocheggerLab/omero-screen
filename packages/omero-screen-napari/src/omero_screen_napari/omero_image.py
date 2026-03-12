import io
import os
import sqlite3
import struct
from typing import Any

import numpy as np
import numpy.typing as npt
from diskcache import Cache, Disk
from diskcache.core import UNKNOWN
from numcodecs import Blosc as _BloscCodec
from omero.api import RawPixelsStore
from omero.gateway import BlitzGateway, ImageWrapper
from omero_screen.config import get_logger, getenv_as_int

logger = get_logger(__name__)

MODE_NUMPY = 5
MODE_NUMPY_COMPRESSED = 6

# Blosc compressor for cache storage.  zstd backend with bit-shuffle gives
# excellent compression on uint16 microscopy data (typically 3-6×).
_blosc = _BloscCodec(cname="zstd", clevel=3, shuffle=_BloscCodec.BITSHUFFLE)

# Mapping from OMERO pixel type names to big-endian numpy dtypes.
# OMERO transmits raw pixel bytes in network (big-endian) byte order.
_OMERO_PIXEL_DTYPES: dict[str, np.dtype[Any]] = {
    "uint8": np.dtype(">u1"),
    "uint16": np.dtype(">u2"),
    "int8": np.dtype(">i1"),
    "int16": np.dtype(">i2"),
    "int32": np.dtype(">i4"),
    "float": np.dtype(">f4"),
    "double": np.dtype(">f8"),
}


class NumpyDisk(Disk):  # type: ignore[misc]
    """Diskcache Disk that stores numpy arrays with Blosc compression.

    On-disk format for ``MODE_NUMPY_COMPRESSED`` (mode 6)::

        [4B header_len LE][header: dtype_str\\nshape_csv][blosc compressed data]

    The header encodes dtype (e.g. ``<u2``) and shape (e.g. ``1,1080,1080,4``)
    so the raw array bytes can be reconstructed after decompression.

    Falls back to the default pickle serialization for non-array values
    (dicts, strings, etc.) and for reading old entries serialized with
    pickle (mode 4) or uncompressed .npy (mode 5).
    """

    def store(
        self, value: Any, read: bool, key: Any = UNKNOWN
    ) -> tuple[Any, ...]:
        """Serialize *value* for storage in cache.

        Numpy arrays are Blosc-compressed (zstd + bitshuffle) for ~3-6×
        size reduction on uint16 microscopy data.  Non-array values fall
        back to pickle.

        Args:
            value: Value to store.
            read: True when value is a file-like object.
            key: Cache key (passed through to parent).

        Returns:
            Tuple of (size, mode, filename, value) for the cache table.
        """
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            # Header: dtype descriptor + comma-separated shape
            header = (
                f"{value.dtype.str}\n{','.join(map(str, value.shape))}"
            ).encode()
            compressed = bytes(_blosc.encode(value))
            blob = struct.pack("<I", len(header)) + header + compressed
            size = len(blob)
            if size < self.min_file_size:
                return (
                    size,
                    MODE_NUMPY_COMPRESSED,
                    None,
                    sqlite3.Binary(blob),
                )
            filename, full_path = self.filename(key, value)
            full_dir = os.path.dirname(full_path)
            os.makedirs(full_dir, exist_ok=True)
            with open(full_path, "xb") as f:
                f.write(blob)
            return size, MODE_NUMPY_COMPRESSED, filename, None
        return super().store(value, read, key)  # type: ignore[no-any-return]

    def fetch(
        self, mode: int, filename: str | None, value: Any, read: bool
    ) -> Any:
        """Deserialize value from cache.

        Handles three numpy modes for backward compatibility:
        - Mode 6 (``MODE_NUMPY_COMPRESSED``): Blosc-compressed (current).
        - Mode 5 (``MODE_NUMPY``): Uncompressed ``.npy`` (v2 cache).
        - Mode 4: Pickle (pre-NumpyDisk entries).

        Args:
            mode: Serialization mode tag.
            filename: Relative path to file (joined with cache dir),
                or None for inline values.
            value: Inline bytes when filename is None.
            read: True to return file-like object.

        Returns:
            Deserialized value.
        """
        if mode == MODE_NUMPY_COMPRESSED:
            if filename is not None:
                full_path = os.path.join(self._directory, filename)
                with open(full_path, "rb") as f:
                    blob = f.read()
            else:
                blob = bytes(value)
            header_len = struct.unpack("<I", blob[:4])[0]
            header = blob[4 : 4 + header_len].decode()
            dtype_str, shape_str = header.split("\n", 1)
            shape = (
                tuple(int(x) for x in shape_str.split(",") if x)
                if shape_str
                else ()
            )
            raw = _blosc.decode(blob[4 + header_len :])
            return np.frombuffer(raw, dtype=dtype_str).reshape(shape).copy()
        if mode == MODE_NUMPY:
            if filename is not None:
                full_path = os.path.join(self._directory, filename)
                return np.load(full_path, allow_pickle=False)
            return np.load(io.BytesIO(value), allow_pickle=False)
        return super().fetch(mode, filename, value, read)


def get_cache_path(subdir: str) -> str:
    """Get the path to the cache directory.

    Uses the environment variable OMERO_SCREEN_CACHE_PATH, or defaults
    to HOME/.cache/omero_screen.

    Args:
        subdir: Sub-directory to append to the path. Use empty string to ignore.

    Returns:
        path
    """
    path = os.getenv("OMERO_SCREEN_CACHE_PATH")
    if path is None:
        import pathlib

        path = str(pathlib.Path.home() / ".cache" / "omero_screen")
    return os.path.join(path, subdir)


# Configure cache path and size using environment.
# Note: Cache size of zero will create a cache but never write any images to disk.
# This takes less then 100Kb space for the sqlite db files. The alternative is to
# not create a cache at all and abstract out method calls to the object.
_cache = Cache(
    get_cache_path("images"),
    disk=NumpyDisk,
    tag_index=True,
    size_limit=getenv_as_int(
        "OMERO_SCREEN_IMAGE_CACHE_SIZE_LIMIT", 20 * 2**30
    ),
)
logger.info(
    "Image cache: %s (size limit: %d)", _cache.directory, _cache.size_limit
)


def get_image(
    conn: BlitzGateway,
    image_id: int,
    start: int | None = None,
    end: int | None = None,
    tag: int | float | str | None = None,
) -> npt.NDArray[Any]:
    """Get an image from OMERO.

    Loads the image from the cache, or downloads from OMERO.
    The downloaded image will be added to the cache.

    Args:
        conn: Connection to OMERO
        image_id: Image ID
        start: Start timepoint
        end: End timepoint
        tag: Tag to associate with cached data

    Returns:
        Image (TZYXC)
    """
    image = get_omero_image_wrapper(conn, image_id)
    sizeT = image.getSizeT()
    if start is None or end is None:
        start = 0
        end = int(sizeT)
    elif start < 0 or end > sizeT or start >= end:
        raise RuntimeError(f"Invalid range: [{start}, {end}) for size {sizeT}")

    stack = []
    store = None
    # This checks the cache for each timepoint and retrieves each in turn.
    for t in range(start, end):
        k = get_key(image_id, t)
        a = _cache.get(k)
        if a is None:
            logger.info("Downloading image %s", k)
            if store is None:
                store, shape, dt_be = initialise_download(conn, image)
            a = get_omero_image_timepoint(store, t, shape, dt_be)
            _cache.set(k, a, tag=tag)
        stack.append(a)
    if store is not None:
        store.close()
    return np.stack(stack)


def get_image_timepoint(
    conn: BlitzGateway,
    image_id: int,
    t: int,
    tag: int | float | str | None = None,
) -> npt.NDArray[Any]:
    """Get an image timepoint from OMERO.

    Loads the image from the cache, or downloads from OMERO.
    The downloaded image will be added to the cache.

    Args:
        conn: Connection to OMERO
        image_id: Image ID
        t: Timepoint
        tag: Tag to associate with cached data

    Returns:
        Image (ZYXC)
    """
    k = get_key(image_id, t)
    a = _cache.get(k)
    if a is None:
        logger.info("Downloading image %s:%d", k, t)
        image = get_omero_image_wrapper(conn, image_id)
        sizeT = image.getSizeT()
        if t < 0 or t > sizeT:
            raise RuntimeError(f"Invalid timepoint {t} for size {sizeT}")
        store, shape, dt_be = initialise_download(conn, image)
        a = get_omero_image_timepoint(store, t, shape, dt_be)
        _cache.set(k, a, tag=tag)
        store.close()
    return a  # type: ignore[no-any-return]


def get_bytes_size(conn: BlitzGateway, image_id: int) -> int:
    """Get the size of the image in bytes.

    Args:
        conn: Connection to OMERO
        image: Image

    Returns:
        size in bytes
    """
    image = get_omero_image_wrapper(conn, image_id)
    pixel_type = image.getPrimaryPixels().getPixelsType().getValue()
    dt_be = _OMERO_PIXEL_DTYPES.get(pixel_type)
    if dt_be is None:
        raise ValueError(f"Unsupported OMERO pixel type: {pixel_type}")
    return (
        int(dt_be.itemsize)
        * int(image.getSizeT())
        * int(image.getSizeC())
        * int(image.getSizeZ())
        * int(image.getSizeY())
        * int(image.getSizeX())
    )


def initialise_download(
    conn: BlitzGateway, image: ImageWrapper
) -> tuple[RawPixelsStore, tuple[int, ...], np.dtype[Any]]:
    """Initialise a RawPixelsStore for download of the image.

    Args:
        conn: Connection to OMERO
        image: Image

    Returns:
        store, image shape, pixels type
    """
    store = conn.c.sf.createRawPixelsStore()
    pixels = image.getPrimaryPixels()
    pixel_type = pixels.getPixelsType().getValue()
    dt_be = _OMERO_PIXEL_DTYPES.get(pixel_type)
    if dt_be is None:
        raise ValueError(f"Unsupported OMERO pixel type: {pixel_type}")
    store.setPixelsId(pixels.getId(), False)
    shape = (
        int(image.getSizeC()),
        int(image.getSizeZ()),
        int(image.getSizeY()),
        int(image.getSizeX()),
    )
    return store, shape, dt_be


def get_omero_image_timepoint(
    store: RawPixelsStore,
    t: int,
    shape: tuple[int, ...],
    dt_be: np.dtype[Any],
) -> npt.NDArray[Any]:
    """Get image timepoint from OMERO.

    Args:
        store: RawPixelsStore initialised with the pixels ID for the image.
        t: Timepoint
        shape: Image shape (C, Z, Y, X)
        dt_be: Big-endian numpy dtype matching the OMERO pixel type.

    Returns:
        Image (ZYXC)
    """
    # Single RPC for all Z×C planes instead of a generator of planes
    # using image.getPrimaryPixels().getPlanes(zctList)
    raw_bytes = store.getTimepoint(t)

    # OMERO dimension order is XYZCT (X fastest). For a fixed timepoint the
    # remaining dimensions are XYZC, which in row-major (C-order) memory
    # layout corresponds to shape (C, Z, Y, X).
    arr = np.frombuffer(raw_bytes, dtype=dt_be).reshape(shape)

    # Transpose CZYX → ZYXC and produce a native-endian copy.
    # Preserve the transposed view to allow channel slicing to a contiguous array.
    return np.array(
        arr.transpose(1, 2, 3, 0),
        dtype=dt_be.newbyteorder("="),
        order="K",
    )


def get_omero_image_wrapper(conn: BlitzGateway, image_id: int) -> ImageWrapper:
    """Get an OMERO image wrapper object.

    Args:
        conn: Connection to OMERO
        image_id: Image ID

    Returns:
        Image wrapper
    """
    image = conn.getObject("Image", image_id)
    if image is None:
        raise RuntimeError("Missing", image_id)
    return image


def get_key(image_id: int, t: int) -> str:
    """Get the image key for the cache.

    Args:
        image_id: Image ID
        t: Timepoint

    Returns:
        Key
    """
    return f"{image_id}:{t}"


def add_cached_image(
    key: str,
    array: npt.NDArray[Any],
    tag: int | float | str | None = None,
) -> bool:
    """Add an image to the cache.

    This method should be used with caution as an arbitrary key can be used to
    store any image.

    If using the key from ``get_key`` then the image should be a single timepoint
    in CZYX format.

    Args:
        key: Image key
        array: Image
        tag: Tag to associate with cached data

    Returns:
        True if item was added
    """
    return _cache.set(key, array, tag=tag)  # type: ignore[no-any-return]


def get_cached_image(
    key: str,
) -> npt.NDArray[Any] | None:
    """Get an image from the cache.

    Args:
        key: Image key

    Returns:
        Image (or None)
    """
    return _cache.get(key)  # type: ignore[no-any-return]


def is_cached(key: str) -> bool:
    """Check if the key is the cache.

    Args:
        key: Cache key

    Returns:
        True if the key exists in the cache
    """
    return key in _cache


def evict(tag: int | float | str | None = None) -> int:
    """Evict all tagged items from the cache.

    Args:
        tag: Tag

    Returns:
        Count of evicted items
    """
    return _cache.evict(tag)  # type: ignore[no-any-return]


def cache_volume() -> int:
    """Return the estimated total size in bytes of the image cache on disk.

    Returns:
        Total size in bytes
    """
    return int(_cache.volume())


def cache_size_limit() -> int:
    """Return the size limit in bytes of the image cache on disk.

    Returns:
        Size limit in bytes
    """
    return int(_cache.size_limit)
