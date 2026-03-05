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


def _parse_raw_timepoint(
    raw_bytes: bytes,
    size_z: int,
    size_c: int,
    size_y: int,
    size_x: int,
    dt_be: np.dtype[Any],
) -> npt.NDArray[Any]:
    """Parse raw bytes from RawPixelsStore.getTimepoint() into a ZYXC array.

    OMERO dimension order is XYZCT (X fastest). For a fixed timepoint the
    remaining dimensions are XYZC, which in row-major (C-order) memory
    layout corresponds to shape ``(C, Z, Y, X)``.

    Args:
        raw_bytes: Raw pixel bytes from ``store.getTimepoint(t)``.
        size_z: Number of Z slices.
        size_c: Number of channels.
        size_y: Image height in pixels.
        size_x: Image width in pixels.
        dt_be: Big-endian numpy dtype matching the OMERO pixel type.

    Returns:
        Contiguous array with shape ``(Z, Y, X, C)`` in native byte order.
    """
    arr = np.frombuffer(raw_bytes, dtype=dt_be).reshape(
        size_c, size_z, size_y, size_x
    )
    # Transpose CZYX → ZYXC and produce a contiguous native-endian copy.
    # order="C" is required because np.array defaults to order="K" which
    # would preserve the transposed view's non-contiguous memory layout.
    return np.array(
        arr.transpose(1, 2, 3, 0), dtype=dt_be.newbyteorder("="), order="C"
    )


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


# Configure cache path and size using environment
__path = os.getenv("OMERO_SCREEN_IMAGE_CACHE_PATH")
if __path is None:
    import pathlib

    __path = str(pathlib.Path.home() / ".cache" / "omero_screen" / "images")

# Note: Cache size of zero will create a cache but never write any images to disk.
# This takes less then 100Kb space for the sqlite db files. The alternative is to
# not create a cache at all and abstract out method calls to the object.
_cache = Cache(
    __path,
    disk=NumpyDisk,
    eviction_policy="none",
    size_limit=getenv_as_int(
        "OMERO_SCREEN_IMAGE_CACHE_SIZE_LIMIT", 20 * 2**30
    ),
)
logger.info("Image cache: %s (size limit: %d)", __path, _cache.size_limit)


def get_image(
    conn: BlitzGateway,
    image_id: int,
    start: int | None = None,
    end: int | None = None,
) -> npt.NDArray[Any]:
    """Get an image from OMERO.

    Args:
        conn: Connection to OMERO
        image_id: Image ID
        start: Start timepoint
        end: End timepoint

    Returns:
        Image (TZYXC)
    """
    image = _get_omero_image_wrapper(conn, image_id)
    sizeT = image.getSizeT()
    if start is None or end is None:
        start = 0
        end = int(sizeT)
    elif start < 0 or end > sizeT or start >= end:
        raise RuntimeError(f"Invalid range: [{start}, {end}) for size {sizeT}")

    stack = []
    # TODO: This checks the cache for each timepoint and retrieves each in turn.
    # It would be more efficient to collate missing ranges and download together.
    for t in range(start, end):
        k = _get_key(image_id, t)
        a = _cache.get(k)
        if a is None:
            logger.info("Downloading image %s", k)
            a = _get_omero_image_timepoint(image, t)
            _cache[k] = a
        stack.append(a)
    return np.stack(stack)


def get_image_timepoint(
    conn: BlitzGateway, image_id: int, t: int
) -> npt.NDArray[Any]:
    """Get an image timepoint from OMERO.

    Args:
        conn: Connection to OMERO
        image_id: Image ID
        t: Timepoint

    Returns:
        Image (ZYXC)
    """
    k = _get_key(image_id, t)
    a = _cache.get(k)
    if a is None:
        logger.info("Downloading image %s", k)
        image = _get_omero_image_wrapper(conn, image_id)
        a = _get_omero_image_timepoint(image, t)
        _cache[k] = a
    return a  # type: ignore[no-any-return]


def _get_omero_image_timepoint(
    image: ImageWrapper, t: int
) -> npt.NDArray[Any]:
    """Get image timepoints from OMERO.

    Args:
        image: OMERO image object
        start: Start timepoint
        end: End timepoint

    Returns:
        Image (ZYXC)
    """
    sizeT = image.getSizeT()
    if t < 0 or t > sizeT:
        raise RuntimeError(f"Invalid timepoint {t} for size {sizeT}")

    sizeZ = image.getSizeZ()
    sizeC = image.getSizeC()
    zctList = []
    for z in range(sizeZ):
        for c in range(sizeC):
            zctList.append((z, c, t))
    planes = image.getPrimaryPixels().getPlanes(zctList)
    # create ZCYX
    a = np.array(list(planes))
    a = a.reshape((sizeZ, sizeC, a.shape[-2], a.shape[-1]))
    # return ZYXC
    return np.moveaxis(a, [0, 1, 2, 3], [0, 3, 1, 2])


def _get_omero_image_wrapper(
    conn: BlitzGateway, image_id: int
) -> ImageWrapper:
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


def _get_key(image_id: int, t: int) -> str:
    """Get the image key.

    Args:
        image_id: Image ID
        t: Timepoint

    Returns:
        Key
    """
    return f"{image_id}:{t}"
