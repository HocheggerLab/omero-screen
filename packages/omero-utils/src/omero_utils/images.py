"""Module for handling image attachments loaded to the OMERO server.

This module provides functions for uploading masks and maximum intensity projections (MIPs) to OMERO datasets.

Available functions:

- upload_masks(conn, dataset_id, image, n_mask, c_mask): Uploads generated images to OMERO server and links them to the specified dataset.
- delete_masks(conn, dataset_id): Removes all segmentation masks from an OMERO dataset.
- parse_mip(conn, image_id, dataset_id): Get the maximum intensity projection of a z-stack image.
- delete_mip(conn, image_id): Removes a maximum intensity projection of a z-stack image saved in OMERO as an annotation.

"""

from typing import Any

import numpy as np
import numpy.typing as npt
from ezomero import get_image
from omero.gateway import (
    BlitzGateway,
    ImageWrapper,
    MapAnnotationWrapper,
    WellWrapper,
)
from omero_screen.config import get_logger
from omero_screen.constants import OmeroScreenNS
from typing_extensions import Generator

from omero_utils.map_anns import (
    add_map_annotations,
    delete_map_annotation,
    parse_annotations,
)

logger = get_logger(__name__)


def upload_masks(
    conn: BlitzGateway,
    dataset_id: int,
    image: ImageWrapper,
    n_mask: npt.NDArray[Any],
    c_mask: npt.NDArray[Any] | None = None,
    name_suffix: str = "_segmentation",
    annotation_key: str = "Segmentation_Mask",
) -> None:
    """Uploads segmentation masks to OMERO server and links them to the specified dataset.

    The id of the mask is stored as an annotation on the original screen image.
    For stitched-mode masks pass ``name_suffix="_stitched_segmentation"`` and
    ``annotation_key="Stitched_Segmentation_Mask"`` so the per-field and
    per-well masks coexist without overwriting each other.

    Args:
        conn: OMERO connection
        dataset_id: ID of the dataset to link the masks to
        image: Image object
        n_mask: Nuclei segmentation mask (TYX)
        c_mask: Cell segmentation mask (TYX)
        name_suffix: Suffix appended to ``image.getId()`` to form the
            OMERO image name for the uploaded mask.
        annotation_key: Map-annotation key on the source image that
            records the uploaded mask's id.

    """
    image_name = f"{image.getId()}{name_suffix}"
    dataset = conn.getObject("Dataset", dataset_id)

    def plane_gen() -> Generator[npt.NDArray[Any]]:
        """Generator that yields each plane in the n_mask and c_mask arrays.

        Yields T first, then C, then Z. Assumes 2d images so no z iteration.
        """
        for i in range(n_mask.shape[0]):
            yield n_mask[i]
        if c_mask is not None:
            for i in range(n_mask.shape[0]):
                yield c_mask[i]

    # Create the image in the dataset
    num_channels = 2 if c_mask is not None else 1
    mask = conn.createImageFromNumpySeq(
        plane_gen(),
        image_name,
        1,  # Z
        num_channels,  # C
        n_mask.shape[0],  # T
        dataset=dataset,
    )

    # Create a map annotation to store the segmentation mask ID
    delete_map_annotation(
        conn, image, annotation_key, ns=OmeroScreenNS.METADATA
    )
    add_map_annotations(
        conn,
        image,
        {annotation_key: mask.getId()},
        ns=OmeroScreenNS.METADATA,
    )


def upload_masks_tiled(
    conn: BlitzGateway,
    dataset_id: int,
    image: ImageWrapper,
    n_mask: npt.NDArray[Any],
    c_mask: npt.NDArray[Any] | None = None,
    name_suffix: str = "_segmentation",
    annotation_key: str = "Segmentation_Mask",
    tile_size: int = 1024,
) -> None:
    """Tile-aware mask upload for canvases larger than OMERO's pyramid threshold.

    OMERO classifies images above ~3000-4000 px per side as pyramidal and
    rejects whole-plane writes (``setPlane``) — they must use ``setTile``
    instead. ``upload_masks`` uses ``setPlane`` via
    ``createImageFromNumpySeq`` and fails for stitched-well canvases.
    This function creates the image at the right dimensions and writes
    it tile-by-tile using the lower-level OMERO API.

    Args:
        conn: OMERO connection
        dataset_id: ID of the dataset to link the masks to
        image: Source image (annotated with the new mask's id)
        n_mask: Nuclei mask of shape (T, Y, X)
        c_mask: Optional cell mask of shape (T, Y, X)
        name_suffix: Appended to ``image.getId()`` to form the new image name.
        annotation_key: Map-annotation key on the source image recording the
            uploaded mask's id.
        tile_size: Square tile edge in pixels (default 1024).
    """
    if n_mask.ndim != 3:
        raise ValueError(f"n_mask must be (T, Y, X), got shape {n_mask.shape}")

    size_t, size_y, size_x = n_mask.shape
    size_c = 2 if c_mask is not None else 1
    size_z = 1

    # Stack channels along axis 1 for upload: (T, C, Y, X)
    if c_mask is not None:
        if c_mask.shape != n_mask.shape:
            raise ValueError(
                f"c_mask shape {c_mask.shape} != n_mask shape {n_mask.shape}"
            )
        data = np.stack([n_mask, c_mask], axis=1)
    else:
        data = n_mask[:, np.newaxis, :, :]

    image_name = f"{image.getId()}{name_suffix}"
    pixel_dtype = _omero_pixel_type_string(n_mask.dtype)

    pixels_service = conn.c.sf.getPixelsService()
    query_service = conn.c.sf.getQueryService()
    pixels_type = query_service.findByQuery(
        f"from PixelsType as p where p.value='{pixel_dtype}'",
        None,
    )
    if pixels_type is None:
        raise RuntimeError(
            f"OMERO server does not know pixel type '{pixel_dtype}'"
        )

    new_image_id = pixels_service.createImage(
        size_x,
        size_y,
        size_z,
        size_t,
        list(range(size_c)),
        pixels_type,
        image_name,
        f"Tiled upload of {n_mask.dtype} mask",
        conn.SERVICE_OPTS,
    )

    new_image = conn.getObject("Image", new_image_id.getValue())
    pixels_id = new_image.getPixelsId()
    raw = conn.c.sf.createRawPixelsStore()
    try:
        raw.setPixelsId(pixels_id, False, conn.SERVICE_OPTS)
        for t in range(size_t):
            for c in range(size_c):
                plane = data[t, c]
                for y in range(0, size_y, tile_size):
                    h = min(tile_size, size_y - y)
                    for x in range(0, size_x, tile_size):
                        w = min(tile_size, size_x - x)
                        tile = np.ascontiguousarray(
                            plane[y : y + h, x : x + w]
                        )
                        raw.setTile(
                            tile.tobytes(),
                            0,  # z
                            c,
                            t,
                            x,
                            y,
                            w,
                            h,
                            conn.SERVICE_OPTS,
                        )
        raw.save(conn.SERVICE_OPTS)
    finally:
        raw.close(conn.SERVICE_OPTS)

    # Link the new image to the dataset
    dataset = conn.getObject("Dataset", dataset_id)
    link = dataset._obj.linkImage(new_image._obj)  # noqa: SLF001
    conn.getUpdateService().saveObject(link, conn.SERVICE_OPTS)

    # Annotate the source image with the new mask's id
    delete_map_annotation(
        conn, image, annotation_key, ns=OmeroScreenNS.METADATA
    )
    add_map_annotations(
        conn,
        image,
        {annotation_key: new_image.getId()},
        ns=OmeroScreenNS.METADATA,
    )


def _omero_pixel_type_string(dtype: np.dtype[Any]) -> str:
    """Map a numpy dtype to OMERO's PixelType.value string."""
    kind = dtype.kind
    size = dtype.itemsize
    mapping: dict[tuple[str, int], str] = {
        ("u", 1): "uint8",
        ("u", 2): "uint16",
        ("u", 4): "uint32",
        ("i", 1): "int8",
        ("i", 2): "int16",
        ("i", 4): "int32",
        ("f", 4): "float",
        ("f", 8): "double",
    }
    try:
        return mapping[(kind, size)]
    except KeyError as e:
        raise ValueError(
            f"Unsupported numpy dtype for OMERO upload: {dtype}"
        ) from e


def delete_masks(conn: BlitzGateway, dataset_id: int) -> None:
    """Removes all segmentation masks from an OMERO dataset.

    Args:
        conn: OMERO connection
        dataset_id: OMERO dataset ID

    """
    dataset = conn.getObject("Dataset", dataset_id)
    for child in dataset.listChildren():
        if child.getName().endswith("_segmentation"):
            image_id = int(child.getName()[: -len("_segmentation")])
            image = conn.getObject("Image", image_id)
            delete_map_annotation(
                conn, image, "Segmentation_Mask", ns=OmeroScreenNS.METADATA
            )
            conn.deleteObject(child._obj)


def parse_mip(
    conn: BlitzGateway, image_id: int, dataset_id: int
) -> npt.NDArray[Any]:
    """Get the maximum intensity projection of a z-stack image.

    The MIP is created and saved to OMERO as an annotation if absent;
    existing map annotations are loaded.

    Args:
        conn: OMERO connection
        image_id: Image ID
        dataset_id: Dataset ID to save/load the MIP.

    Returns:
        MIP image

    """
    image = conn.getObject("Image", image_id)

    if mip_id := _check_mip_annotation(image):
        _, mip_array = get_image(conn, mip_id)
        if isinstance(mip_array, np.ndarray):
            return mip_array
        logger.warning(
            "The image is linked to a missing MIP; this will be regenerated"
        )
    return _load_mip(conn, image, dataset_id)


def _check_mip_annotation(image: ImageWrapper) -> int:
    """Check if a MIP map annotation exists.

    Args:
        image: OMERO image object
    Returns:
        The annotation MIP image ID; else 0

    """
    annotations = image.listAnnotations(ns=OmeroScreenNS.METADATA)
    if map_anns := [
        ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
    ]:
        for ann in map_anns:
            ann_values = dict(ann.getValue())
            for k, v in ann_values.items():
                if k == "MIP":
                    return (
                        int(v.split(":")[-1]) if v.find(":") >= 0 else int(v)
                    )
    return 0


def _load_mip(
    conn: BlitzGateway, image: ImageWrapper, dataset_id: int
) -> npt.NDArray[Any]:
    """Create a maximum intensity projection of a z-stack image and save to OMERO as an annotation.

    Args:
        conn: OMERO connection
        image: Image object
        dataset_id: Dataset ID to save the MIP.

    Returns:
        MIP image

    """
    dataset = conn.getObject("Dataset", dataset_id)
    mip_array = _process_mip(conn, image.getId())
    channel_num = mip_array.shape[-1]
    mip_name = f"MIP_{image.getId()}"
    img_gen = _image_generator(mip_array)
    new_image = conn.createImageFromNumpySeq(
        # Generator creates size (zct)
        img_gen,
        mip_name,
        1,
        channel_num,
        mip_array.shape[0],
        dataset=dataset,
    )
    delete_map_annotation(conn, image, "MIP", ns=OmeroScreenNS.METADATA)
    add_map_annotations(
        conn, image, {"MIP": new_image.getId()}, ns=OmeroScreenNS.METADATA
    )
    return mip_array


def _process_mip(conn: BlitzGateway, image_id: int) -> npt.NDArray[Any]:
    """Generate maximum intensity projection of an image.

    Args:
        conn: OMERO connection
        image_id: Image ID
    Returns:
        numpy array of maximum intensity projection (t, 1, y, x, c)

    """
    _, array = get_image(conn, image_id)
    return np.max(array, axis=1, keepdims=True)  # type: ignore


def _image_generator(
    image_array: npt.NDArray[Any],
) -> Generator[npt.NDArray[Any]]:
    # Input is TZYXC
    # iterate through T first, then C then Z. Here z=0.
    for c in range(image_array.shape[-1]):
        for t in range(image_array.shape[0]):
            yield image_array[t, 0, ..., c]


def delete_mip(conn: BlitzGateway, image_id: int) -> None:
    """Removes a maximum intensity projection of a z-stack image saved in OMERO as an annotation.

    Args:
        conn: OMERO connection
        image_id: OMERO image ID

    """
    image = conn.getObject("Image", image_id)
    if mip_id := _check_mip_annotation(image):
        delete_map_annotation(conn, image, "MIP", ns=OmeroScreenNS.METADATA)
        mip = conn.getObject("Image", mip_id)
        conn.deleteObject(mip._obj)


STITCHED_MASK_ANNOTATION_KEY = "Stitched_Segmentation_Mask"


def fetch_stitched_field_masks(
    conn: BlitzGateway,
    well: WellWrapper,
) -> tuple[
    list[npt.NDArray[Any]],
    list[npt.NDArray[Any] | None],
    list[int],
]:
    """Fetch per-field stitched-mode segmentation masks for one well.

    Stitched-mode masks are uploaded by the omero-screen pipeline via
    :func:`upload_masks` with ``name_suffix="_stitched_segmentation"`` and
    annotation key ``Stitched_Segmentation_Mask``. The annotation lives on
    the original field image and points to the mask image's id.

    For each field (well sample) in ``well`` this function:

    1. Reads the ``Stitched_Segmentation_Mask`` map annotation on the
       field's source image.
    2. Downloads the corresponding mask image. The mask has shape
       ``(T, Z=1, Y, X, C)`` where ``C=1`` for nucleus-only or ``C=2``
       for nucleus + cell (channel 0 = nuclei, channel 1 = cells).
    3. Squeezes Z and splits channels into separate ``(T, Y, X)`` arrays.

    Args:
        conn: OMERO connection.
        well: Well object whose fields will be queried.

    Returns:
        Tuple ``(nuclei_per_field, cells_per_field, source_image_ids)``:

        * ``nuclei_per_field``: list of ``(T, Y, X)`` ``uint16`` nucleus
          masks, one per field, in well-sample order.
        * ``cells_per_field``: list of ``(T, Y, X)`` cell masks (same
          ordering) or ``None`` for fields that have nucleus-only masks.
        * ``source_image_ids``: list of original field image IDs in the
          same order — useful for downstream operations that need to
          re-link results to the source image.

    Raises:
        KeyError: If any field is missing a ``Stitched_Segmentation_Mask``
            annotation. This indicates the well was not processed in
            stitched mode and the caller should fall back to per-field
            masks.
    """
    n_fields = len(list(well.listChildren()))
    nuclei: list[npt.NDArray[Any]] = []
    cells: list[npt.NDArray[Any] | None] = []
    source_ids: list[int] = []

    for n in range(n_fields):
        ws = well.getWellSample(n)
        field_image = ws.getImage()
        field_id = field_image.getId()
        source_ids.append(field_id)

        anns = parse_annotations(field_image, ns=OmeroScreenNS.METADATA)
        if STITCHED_MASK_ANNOTATION_KEY not in anns:
            raise KeyError(
                f"Field image {field_id} has no "
                f"{STITCHED_MASK_ANNOTATION_KEY!r} annotation. "
                f"Was this well processed in stitched mode?"
            )
        mask_id = int(anns[STITCHED_MASK_ANNOTATION_KEY])

        # ezomero.get_image returns (image_wrapper, (T,Z,Y,X,C) array).
        _, mask_array = get_image(conn, mask_id)
        # Squeeze Z (always 1 for masks) → (T, Y, X, C)
        if mask_array.shape[1] != 1:
            raise ValueError(
                f"Stitched mask image {mask_id} has Z={mask_array.shape[1]}; "
                f"expected Z=1"
            )
        squeezed = np.squeeze(mask_array, axis=1)  # (T, Y, X, C)
        n_channels = squeezed.shape[-1]
        if n_channels not in (1, 2):
            raise ValueError(
                f"Stitched mask image {mask_id} has C={n_channels}; "
                f"expected 1 (nuclei only) or 2 (nuclei + cells)"
            )

        nuclei.append(np.ascontiguousarray(squeezed[..., 0]))
        cells.append(
            np.ascontiguousarray(squeezed[..., 1]) if n_channels == 2 else None
        )

    return nuclei, cells, source_ids
