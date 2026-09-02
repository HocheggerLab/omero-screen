"""Assemble a Harmony measurement folder on disk from a :class:`PlateSpec`.

Output layout, which is what ``omero import`` (and ``scripts/load_plates.sh``)
expects::

    <out>/<plate_name>/
        Images/
            Index.idx.xml
            r01c01f01p01-ch1sk1fk1fl1.tiff
            ...
        metadata.xlsx
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import tifffile
from loguru import logger
from omero.gateway import BlitzGateway

from omero_screen.export.harmony_xml import (
    ImageSpec,
    PlateSpec,
    build_index_xml,
)

#: Harmony writes plain uint16 TIFFs; anything wider would not round-trip.
EXPORT_DTYPE = np.uint16


def estimate_size_bytes(plate: PlateSpec) -> int:
    """Rough uncompressed size of the exported TIFFs."""
    return sum(
        spec.size_x * spec.size_y * EXPORT_DTYPE().itemsize
        for spec in plate.images
    )


def _to_uint16(
    plane: npt.NDArray[Any], image_id: int
) -> npt.NDArray[np.uint16]:
    """Coerce a plane to uint16, warning once per image if values are clipped."""
    if plane.dtype == EXPORT_DTYPE:
        return cast("npt.NDArray[np.uint16]", plane)
    # The floating check short-circuits, so ``max()`` is only evaluated on
    # integer planes where it cannot be NaN.
    if np.issubdtype(plane.dtype, np.floating) or int(plane.max()) > 65535:
        logger.warning(
            f"Image {image_id}: {plane.dtype} pixels clipped to uint16 for export"
        )
    clipped: npt.NDArray[np.uint16] = np.clip(plane, 0, 65535).astype(
        EXPORT_DTYPE
    )
    return clipped


def _write_planes(
    conn: BlitzGateway, specs: list[ImageSpec], images_dir: Path
) -> int:
    """Fetch and write every plane, one TIFF per spec.

    Planes are requested per OMERO image in a single ``getPlanes`` call so the
    server streams them rather than answering one round trip per plane.

    Returns:
        Number of TIFFs written.
    """
    by_image: dict[int, list[ImageSpec]] = defaultdict(list)
    for spec in specs:
        by_image[spec.omero_image_id].append(spec)

    written = 0
    for image_id, image_specs in by_image.items():
        image = conn.getObject("Image", image_id)
        if image is None:
            raise ValueError(f"Image {image_id} disappeared during export")
        pixels = image.getPrimaryPixels()
        # (z, c, t) triples, 0-based, in the same order as image_specs.
        zct = [(s.plane - 1, s.channel - 1, s.timepoint) for s in image_specs]
        for spec, plane in zip(
            image_specs, pixels.getPlanes(zct), strict=False
        ):
            tifffile.imwrite(
                images_dir / spec.url, _to_uint16(plane, image_id)
            )
            written += 1
        logger.debug(f"Image {image_id}: wrote {len(image_specs)} plane(s)")

    return written


def write_measurement(
    conn: BlitzGateway, plate: PlateSpec, out_dir: Path
) -> Path:
    """Write the full measurement folder for ``plate``.

    Args:
        conn: OMERO connection used to fetch pixels.
        plate: What to export, from :func:`~.plate_reader.read_plate`.
        out_dir: Parent directory; a ``<plate.name>`` folder is created inside.

    Returns:
        Path to the written ``Index.idx.xml``.
    """
    measurement_dir = out_dir / plate.name
    images_dir = measurement_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    index_path = images_dir / "Index.idx.xml"
    index_path.write_bytes(build_index_xml(plate))

    written = _write_planes(conn, plate.images, images_dir)
    logger.info(
        f"Wrote {written} TIFF(s) and {index_path.name} to {measurement_dir}"
    )
    return index_path
