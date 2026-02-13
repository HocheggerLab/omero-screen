import io
import os
import sqlite3
import threading
from collections.abc import Generator
from typing import Any

import numpy as np
import numpy.typing as npt
import omero
from diskcache import Cache, Disk
from diskcache.core import UNKNOWN
from omero.gateway import BlitzGateway, ImageWrapper
from omero.rtypes import unwrap
from omero_screen.config import get_logger, getenv_as_int
from omero_screen.plate_dataset import PlateDataset

logger = get_logger(__name__)

MODE_NUMPY = 5


class NumpyDisk(Disk):  # type: ignore[misc]
    """Diskcache Disk that uses numpy .npy format for ndarray values.

    Falls back to the default pickle serialization for non-array values
    (dicts, strings, etc.) and for reading old pickle-serialized entries.
    """

    def store(
        self, value: Any, read: bool, key: Any = UNKNOWN
    ) -> tuple[Any, ...]:
        """Serialize value for storage in cache.

        Args:
            value: Value to store. Numpy arrays use .npy format;
                everything else falls back to pickle.
            read: True when value is a file-like object.
            key: Cache key (passed through to parent).

        Returns:
            Tuple of (size, mode, filename, value) for the cache table.
        """
        if isinstance(value, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, value, allow_pickle=False)
            npy_bytes = buf.getvalue()
            size = len(npy_bytes)
            if size < self.min_file_size:
                return size, MODE_NUMPY, None, sqlite3.Binary(npy_bytes)
            filename, full_path = self.filename(key, value)
            full_dir = os.path.dirname(full_path)
            os.makedirs(full_dir, exist_ok=True)
            with open(full_path, "xb") as f:
                f.write(npy_bytes)
            return size, MODE_NUMPY, filename, None
        return super().store(value, read, key)  # type: ignore[no-any-return]

    def fetch(
        self, mode: int, filename: str | None, value: Any, read: bool
    ) -> Any:
        """Deserialize value from cache.

        Args:
            mode: Serialization mode tag.
            filename: Relative path to file (joined with cache dir),
                or None for inline values.
            value: Inline bytes when filename is None.
            read: True to return file-like object.

        Returns:
            Deserialized value.
        """
        if mode == MODE_NUMPY:
            if filename is not None:
                full_path = os.path.join(self._directory, filename)
                return np.load(full_path, allow_pickle=False)
            return np.load(io.BytesIO(value), allow_pickle=False)
        return super().fetch(mode, filename, value, read)


# Lock to prevent duplicate downloads when multiple generators
# run concurrently (e.g. background caching + foreground image load).
_download_lock = threading.Lock()

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
        with _download_lock:
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
    with _download_lock:
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


def _get_omero_image_timepoints(
    image: ImageWrapper, start: int, end: int
) -> npt.NDArray[Any]:
    """Get image timepoints from OMERO.

    Args:
        image: OMERO image object
        start: Start timepoint
        end: End timepoint

    Returns:
        Image (TZYXC)
    """
    sizeT = image.getSizeT()
    if start < 0 or end > sizeT or start >= end:
        raise RuntimeError(f"Invalid range: [{start}, {end}) for size {sizeT}")

    sizeZ = image.getSizeZ()
    sizeC = image.getSizeC()
    sizeT = end - start
    zctList = []
    for z in range(sizeZ):
        for c in range(sizeC):
            for t in range(start, end + 1):
                zctList.append((z, c, t))
    planes = image.getPrimaryPixels().getPlanes(zctList)
    # create ZCTYX
    a = np.array(list(planes))
    a = a.reshape((sizeZ, sizeC, sizeT, a.shape[-2], a.shape[-1]))
    # return TZYXC
    return np.moveaxis(a, [0, 1, 2, 3, 4], [1, 4, 0, 2, 3])


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


def cache_plate_images(
    conn: BlitzGateway, plate_id: int
) -> Generator[int, None, int]:
    """Cache the images for the OMERO plate.

    This returns a generator that iterates over all images in
    the plate and associated plate dataset fills the cache with missing
    timepoints.

    Yields the number of bytes downloaded for
    each image timepoint and returns the total number of bytes.

    This method stops when the number of bytes downloaded exceed the size
    limit of the cache.

    Args:
        conn: OMERO connection.
        plate_id: OMERO plate ID.

    Yields:
        Bytes downloaded for the plate image timepoint.

    Returns:
        Total bytes downloaded.

    Raises:
        Exception if the plate is not recognised.
    """
    max_bytes = _cache.size_limit
    if max_bytes <= 0:
        # Edge case with no cache
        return 0

    # Plate images
    images = _get_image_ids(conn, plate_id)
    # Plate associated images
    dataset_id = PlateDataset(conn, plate_id).dataset_id
    images.extend(_get_dataset_image_ids(conn, dataset_id))

    total = 0
    for image_id, size_t in images:
        image = None
        for t in range(size_t):
            k = _get_key(image_id, t)
            # Lock prevents duplicate downloads when a foreground
            # get_image_timepoint() call races with this generator.
            with _download_lock:
                a = _cache.get(k)
                if a is None:
                    logger.info("Downloading image %s", k)
                    if image is None:
                        image = _get_omero_image_wrapper(conn, image_id)
                    a = _get_omero_image_timepoint(image, t)
                    _cache[k] = a
                    nbytes = a.nbytes
                    total += nbytes
                else:
                    nbytes = 0

            yield nbytes
            if nbytes > 0 and total > max_bytes:
                logger.info(
                    "Filled image cache with %d > %d bytes",
                    total,
                    max_bytes,
                )
                return total

    return total


def _get_image_ids(conn: BlitzGateway, plate_id: int) -> list[tuple[int, int]]:
    """Get a list of images from the OMERO plate.

    Args:
        conn: OMERO connection.
        plate_id: OMERO plate ID.

    Returns:
        list of image details (image_id, size_t).

    Raises:
        Exception if the plate is not recognised.
    """
    images = []
    # Use the query service for enhanced performance over the OMERO object API
    query_service = conn.getQueryService()
    # https://omero.readthedocs.io/en/stable/developers/Model/EveryObject.html#plate
    # https://omero.readthedocs.io/en/stable/developers/Model/EveryObject.html#image
    params = omero.sys.ParametersI()
    query = f"""select i.id, pi.sizeT from Plate as p
        left join p.wells as w
        left join w.wellSamples as ws
        left join ws.image as i
        left join i.pixels as pi
        where p.id = '{plate_id}'
    """
    results = query_service.projection(query, params, None)
    if not len(results):
        raise Exception(f"OMERO plate does not exist: {plate_id}")
    for image_id, size_t in results:
        images.append((unwrap(image_id), unwrap(size_t)))

    return images


def _get_dataset_image_ids(
    conn: BlitzGateway, dataset_id: int
) -> list[tuple[int, int]]:
    """Get a list of images from the OMERO dataset.

    Args:
        conn: OMERO connection.
        dataset_id: OMERO dataset ID.

    Returns:
        list of image details (image_id, size_t).

    Raises:
        Exception if the dataset is not recognised.
    """
    images = []
    # Use the query service for enhanced performance over the OMERO object API
    query_service = conn.getQueryService()
    # https://omero.readthedocs.io/en/stable/developers/Model/EveryObject.html#plate
    # https://omero.readthedocs.io/en/stable/developers/Model/EveryObject.html#image
    params = omero.sys.ParametersI()
    query = f"""select i.id, pi.sizeT from Dataset as d
        left join d.imageLinks as il
        left join il.child as i
        left join i.pixels as pi
        where d.id = '{dataset_id}'
    """
    results = query_service.projection(query, params, None)
    if not len(results):
        raise Exception(f"OMERO dataset does not exist: {dataset_id}")
    for image_id, size_t in results:
        images.append((unwrap(image_id), unwrap(size_t)))

    return images
