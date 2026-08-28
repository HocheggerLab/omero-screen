"""Module for handling image attachments loaded to the OMERO server.

This module provides functions for uploading masks and maximum intensity projections (MIPs) to OMERO datasets.

Available functions:

- upload_masks(conn, dataset_id, image, n_mask, c_mask): Uploads generated images to OMERO server and links them to the specified dataset.
- prune_duplicate_masks(conn, dataset_id, image_name): Removes same-named mask images left by earlier runs.
- delete_masks(conn, dataset_id): Removes all segmentation masks from an OMERO dataset.
- parse_mip(conn, image_id, dataset_id): Get the maximum intensity projection of a z-stack image.
- delete_mip(conn, image_id): Removes a maximum intensity projection of a z-stack image saved in OMERO as an annotation.

"""

from typing import Any

import numpy as np
import numpy.typing as npt
from ezomero import get_image
from loguru import logger
from omero.gateway import (
    BlitzGateway,
    ImageWrapper,
    MapAnnotationWrapper,
    WellWrapper,
)
from omero_screen.constants import OmeroScreenNS
from typing_extensions import Generator

from omero_utils.map_anns import (
    add_map_annotations,
    delete_map_annotation,
    parse_annotations,
)


def prune_duplicate_masks(
    conn: BlitzGateway,
    dataset_id: int,
    image_name: str,
    keep_id: int | None = None,
    dry_run: bool = False,
) -> list[int]:
    """Delete mask images in a dataset that duplicate ``image_name``.

    Mask images are named ``{source_image_id}{suffix}`` and are meant to be
    unique within the segmentation dataset. Because :func:`upload_masks`
    historically created a new image on every run without removing the old
    one, re-analysing a plate left a pile of same-named masks behind. Only
    the map annotation on the source image was repointed, so anything that
    resolves masks *by name* — ``plate_aggregation._get_mask_map``, the
    napari well-data loader — would pick an arbitrary one, potentially a
    mask from a previous run with different segmentation settings.

    Args:
        conn: OMERO connection.
        dataset_id: Segmentation dataset to scan.
        image_name: Exact mask image name, e.g. ``"1234_segmentation"``.
            Matched exactly so ``_segmentation`` never catches
            ``_stitched_segmentation``.
        keep_id: Mask image id to preserve — normally the one just
            uploaded. ``None`` keeps the highest id (the newest).
        dry_run: Report what would be deleted without deleting it.

    Returns:
        Ids of the deleted masks (or, under ``dry_run``, of those that
        would be deleted). Empty when there was nothing to prune.
    """
    dataset = conn.getObject("Dataset", dataset_id)
    if dataset is None:
        return []
    matches = [
        int(child.getId())
        for child in dataset.listChildren()
        if child.getName() == image_name
    ]
    if len(matches) < 2 and keep_id is None:
        return []
    if keep_id is None:
        keep_id = max(matches)
    stale = sorted(i for i in matches if i != keep_id)
    if not stale:
        return []
    logger.info(
        f"{'Would delete' if dry_run else 'Deleting'} {len(stale):d} duplicate "
        f"mask(s) named {image_name!r} in dataset {dataset_id:d}: {stale} "
        f"(keeping {keep_id:d})"
    )
    if not dry_run:
        conn.deleteObjects("Image", stale, deleteAnns=True, wait=True)
    return stale


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

    # Only now drop any same-named masks from earlier runs. Ordering is
    # deliberate: create → repoint annotation → delete. The source image
    # therefore always points at a mask that exists, and a crash part-way
    # leaves duplicates (the status quo) rather than no mask at all. The
    # next run prunes whatever was left behind.
    prune_duplicate_masks(
        conn, dataset_id, image_name, keep_id=int(mask.getId())
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
    suffixes = ["_stitched_segmentation", "_segmentation"]
    annotations = ["Stitched_Segmentation_Mask", "Segmentation_Mask"]
    for child in dataset.listChildren():
        for suffix, annotation in zip(suffixes, annotations, strict=True):
            if child.getName().endswith(suffix):
                image_id = int(child.getName()[: -len(suffix)])
                image = conn.getObject("Image", image_id)
                delete_map_annotation(
                    conn, image, annotation, ns=OmeroScreenNS.METADATA
                )
                conn.deleteObject(child._obj)
                break


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


def resolve_stitched_mask_ids(
    well: WellWrapper,
    fields: list[int],
) -> tuple[list[int], list[int]]:
    """Resolve per-field ``(mask_id, source_id)`` for a well's stitched masks.

    The cheap, pixel-free first phase of :func:`fetch_stitched_field_masks_trange`,
    split out so a streaming caller can resolve the ids once and then fetch
    pixels per timepoint-block via :func:`fetch_stitched_field_masks_trange`.

    Args:
        well: Well object whose fields will be queried.
        fields: The indices of the fields.

    Returns:
        ``(mask_ids, source_ids)`` in the provided field order.

    Raises:
        KeyError: If any field lacks a ``Stitched_Segmentation_Mask``
            annotation (well not processed in stitched mode).
    """
    source_ids = []
    mask_ids = []
    for n in fields:
        ws = well.getWellSample(n)
        field_image = ws.getImage()
        field_id = int(field_image.getId())
        source_ids.append(field_id)
        anns = parse_annotations(field_image, ns=OmeroScreenNS.METADATA)
        if STITCHED_MASK_ANNOTATION_KEY not in anns:
            raise KeyError(
                f"Field image {field_id} has no "
                f"{STITCHED_MASK_ANNOTATION_KEY!r} annotation. "
                f"Was this well processed in stitched mode?"
            )
        mask_ids.append(int(anns[STITCHED_MASK_ANNOTATION_KEY]))
    return mask_ids, source_ids


def fetch_stitched_field_masks_trange(
    conn: BlitzGateway,
    mask_ids: list[int],
    *,
    t0: int | None = None,
    t1: int | None = None,
    source_ids: list[int] | None = None,
    conn_factory: Any | None = None,
    max_workers: int = 3,
) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any] | None]]:
    """Download per-field stitched masks for timepoints ``[t0, t1)``.

    ``t0=t1=None`` fetches the full time range (the legacy whole-well
    behaviour); otherwise only the half-open block ``[t0, t1)`` is read via
    an OMERO sub-volume call, so a streaming build never holds more than one
    block of label pixels in memory.

    Args:
        conn: OMERO connection (used when ``conn_factory`` is ``None``).
        mask_ids: Per-field mask image ids from
            :func:`resolve_stitched_mask_ids`.
        t0: Inclusive start timepoint; ``None`` (with ``t1``) means full range.
        t1: Exclusive end timepoint; ``None`` (with ``t0``) means full range.
        source_ids: Optional per-field source ids, for clearer errors.
        conn_factory: Zero-arg callable returning a fresh ``BlitzGateway``
            for parallel, thread-local downloads.
        max_workers: Concurrency for the parallel path.

    Returns:
        ``(nuclei_per_field, cells_per_field)`` — lists of ``(t, Y, X)``
        masks (``t = t1 - t0`` for a block), ``cells`` entries ``None`` for
        nucleus-only fields.
    """
    n_fields = len(mask_ids)
    ids = source_ids if source_ids is not None else mask_ids
    raw_masks: list[npt.NDArray[Any] | None] = [None] * n_fields

    def _download_one(idx: int, mask_id: int) -> tuple[int, npt.NDArray[Any]]:
        worker_conn = conn_factory() if conn_factory is not None else conn
        try:
            if t0 is None or t1 is None:
                _, mask_array = get_image(worker_conn, mask_id)
            else:
                img = worker_conn.getObject("Image", mask_id)
                # ezomero start_coords / axis_lengths are XYZCT.
                _, mask_array = get_image(
                    worker_conn,
                    mask_id,
                    start_coords=(0, 0, 0, 0, t0),
                    axis_lengths=(
                        img.getSizeX(),
                        img.getSizeY(),
                        img.getSizeZ(),
                        img.getSizeC(),
                        t1 - t0,
                    ),
                )
        finally:
            if conn_factory is not None and worker_conn is not conn:
                import contextlib

                with contextlib.suppress(Exception):
                    worker_conn.close()
        return idx, mask_array

    if conn_factory is not None and n_fields > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fut in as_completed(
                ex.submit(_download_one, i, mask_ids[i])
                for i in range(n_fields)
            ):
                idx, arr = fut.result()
                raw_masks[idx] = arr
    else:
        for i in range(n_fields):
            _, raw_masks[i] = _download_one(i, mask_ids[i])

    # Squeeze Z and split channels. CPU-bound, sequential.
    nuclei: list[npt.NDArray[Any]] = []
    cells: list[npt.NDArray[Any] | None] = []
    for n, mask_array in enumerate(raw_masks):
        if mask_array is None:
            raise RuntimeError(
                f"Stitched mask for field {ids[n]} failed to download"
            )
        if mask_array.shape[1] != 1:
            raise ValueError(
                f"Stitched mask image {mask_ids[n]} has Z={mask_array.shape[1]}; "
                f"expected Z=1"
            )
        squeezed = np.squeeze(mask_array, axis=1)  # (T, Y, X, C)
        n_channels = squeezed.shape[-1]
        if n_channels not in (1, 2):
            raise ValueError(
                f"Stitched mask image {mask_ids[n]} has C={n_channels}; "
                f"expected 1 (nuclei only) or 2 (nuclei + cells)"
            )
        nuclei.append(np.ascontiguousarray(squeezed[..., 0]))
        cells.append(
            np.ascontiguousarray(squeezed[..., 1]) if n_channels == 2 else None
        )

    return nuclei, cells
