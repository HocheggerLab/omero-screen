"""Direct OMERO data loader for annotation sessions.

This module provides functionality to load cell crops directly from OMERO
for annotation without requiring welldata_widget pre-loading.
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from omero.gateway import BlitzGateway
from omero_screen.config import get_logger
from omero_utils import omero_connect

from omero_screen_napari.gallery_api import (
    calculate_crop_coordinates,
    draw_contours,
    fill_missing_channels,
    pad_region,
)
from omero_screen_napari.omero_image import get_image_timepoint
from omero_screen_napari.session_utils import apply_masks_to_crops
from omero_screen_napari.zarr_cache import (
    cached_wells,
    fetch_crop,
    fetch_label_crop,
    resolve_to_zarr,
)

if TYPE_CHECKING:
    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroData

logger = get_logger(__name__)


def _load_cellview_well_slice(
    plate_id: int, well_id: str, timepoint: int
) -> Any:
    """Load the CellView slice for one (plate, well, timepoint) via SQL pushdown.

    Uses :func:`cellview.exporters.db_to_polars.export_polars_lf` with the
    ``well`` and ``timepoint`` predicates pushed into the DuckDB ``WHERE``
    clause. For plate 4053 this cuts ~695k rows down to ~1.5k — i.e. the
    DuckDB → in-memory materialisation goes from ~3 s to a fraction of
    one. The returned object is a pandas DataFrame so downstream filter
    code (``df[df["image_id"].isin(...)]`` etc.) stays unchanged.

    Returns ``None`` when CellView is unavailable, the plate isn't in the
    DB, or the query is empty — callers treat that as a "no centroids"
    signal and either skip or fall back.
    """
    try:
        from cellview.db.db import CellViewDB
        from cellview.exporters.db_to_polars import export_polars_lf
    except ImportError:
        logger.warning("CellView not available; cannot load centroids")
        return None

    db = CellViewDB()
    conn = db.connect()
    try:
        lf, _ = export_polars_lf(
            plate_id, conn, well=well_id, timepoint=timepoint
        )
        # `timepoint` is pushed down only if the column exists in this
        # plate's measurements; older non-timelapse plates don't carry it.
        # In that case the predicate is silently dropped — we still get
        # the well slice.
        df = lf.collect().to_pandas()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "CellView export failed for plate %d well %s (%s)",
            plate_id,
            well_id,
            exc,
        )
        return None
    finally:
        conn.close()

    if df.empty:
        logger.warning(
            "CellView has no rows for plate %d well %s at t=%d",
            plate_id,
            well_id,
            timepoint,
        )
        return None
    return df


def _parse_image_input(image_input: str, num_images: int) -> list[int]:
    """Parse an image input string into a list of image indices.

    Supports the same formats as the welldata widget:
    - ``"All"`` (case-insensitive) → all indices ``[0 .. num_images-1]``
    - ``"3-5"`` → ``[3, 4, 5]``
    - ``"0, 1, 2"`` → ``[0, 1, 2]``
    - ``"0"`` → ``[0]``

    Args:
        image_input: User-supplied image selection string.
        num_images: Total number of images in the well.

    Returns:
        Sorted list of 0-based image indices.

    Raises:
        ValueError: If the format is invalid or indices are out of range.
    """
    index = image_input.strip()

    if index.lower() == "all":
        return list(range(num_images))

    if not re.match(r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$", index):
        raise ValueError(
            f"Image input '{index}' doesn't match any of the expected "
            "patterns 'All', '0', '0, 1, 2', or '3-5'."
        )

    if "-" in index and "," not in index:
        start, end = map(int, index.split("-"))
        indices = list(range(start, end + 1))
    elif "," in index:
        indices = [int(x.strip()) for x in index.split(",")]
    else:
        indices = [int(index)]

    # Validate all indices are in range
    for idx in indices:
        if idx < 0 or idx >= num_images:
            raise ValueError(
                f"Image index {idx} out of range (well has {num_images} "
                f"images, use 0-{num_images - 1})."
            )

    return sorted(set(indices))


@omero_connect
def load_crops_from_omero(
    plate_id: int,
    well_id: str,
    image_input: str,
    classifier_name: str,
    omero_data: "OmeroData",
    user_data: "UserData",
    cellcycle: str = "All",
    classifier_column: str = "",
    classifier_class: str = "",
    timepoint: int | None = None,
    conn: BlitzGateway | None = None,
) -> tuple[bool, str]:
    """Load cell crops directly from OMERO for annotation.

    This bypasses welldata_widget by fetching only the needed image(s)
    and generating crops on-demand.

    Args:
        plate_id: OMERO plate ID
        well_id: Well position (e.g., "A1")
        image_input: Image selection string (e.g. "All", "0", "0, 1", "3-5")
        classifier_name: Name of classifier (for metadata)
        omero_data: OmeroData instance to populate
        user_data: UserData instance for settings
        cellcycle: Cell cycle phase filter ("All", "G1", "S", "G2/M", etc.)
        classifier_column: CellView column to filter on (e.g. "classifier_mitosis")
        classifier_class: Class value to keep (e.g. "positive"); empty means no filter
        timepoint: Timepoint to load (overrides the value baked into the
            classifier's metadata.json). When ``None``, falls back to the
            classifier-metadata value.
        conn: OMERO connection (injected by decorator)

    Returns:
        (success: bool, message: str)
    """
    if conn is None:
        return False, "OMERO connection failed"

    try:
        # Step 1: Load classifier metadata
        logger.info(f"Loading metadata for classifier: {classifier_name}")
        metadata_path = (
            Path.home()
            / "omeroscreen_trainingdata"
            / classifier_name
            / "metadata.json"
        )

        if not metadata_path.exists():
            return False, f"Metadata file not found: {metadata_path}"

        with metadata_path.open() as f:
            metadata = json.load(f)

        user_data_dict = metadata["user_data"]

        # Caller-supplied timepoint overrides the value baked into the
        # classifier metadata. Without this, the dialog's Timepoint field
        # is silently ignored and every load is locked to the timepoint
        # at which the classifier was trained.
        if timepoint is not None:
            user_data_dict["timepoint"] = int(timepoint)

        # Step 2: Fetch plate and well from OMERO
        logger.info(f"Fetching plate {plate_id} and well {well_id} from OMERO")
        plate = conn.getObject("Plate", plate_id)
        if not plate:
            return False, f"Plate {plate_id} not found in OMERO"

        well = None
        for well_obj in plate.listChildren():
            if well_obj.getWellPos() == well_id:
                well = well_obj
                break

        if not well:
            return False, f"Well {well_id} not found in plate {plate_id}"

        # Step 3: Parse image input string into indices
        num_images = well.countWellSample()
        try:
            image_indices = _parse_image_input(image_input, num_images)
        except ValueError as exc:
            return False, str(exc)

        logger.info(
            f"Parsed image input '{image_input}' → indices {image_indices}"
        )

        # Get timepoint from metadata
        timepoint = int(user_data_dict.get("timepoint", 0))
        crop_size = user_data_dict["crop_size"]
        segmentation = user_data_dict.get("segmentation", "nucleus")

        # Resolve indices → real OMERO image IDs (needed by both the zarr
        # fast path and the legacy per-image loop below).
        image_id_by_index: dict[int, int] = {}
        for img_idx in image_indices:
            img_obj = well.getImage(img_idx)
            if img_obj is not None:
                image_id_by_index[img_idx] = img_obj.getId()
        wanted_image_ids = list(image_id_by_index.values())

        # Load the CellView slice for this (plate, well, timepoint) ONCE
        # via SQL pushdown, then thread it through both paths. The old
        # code called `cellview_load_data(plate_id)` inside the zarr fast
        # path AND per-image inside the OMERO fallback — each call
        # materialised the full ~695k-row plate dataframe (~3 s). With
        # `well` + `timepoint` predicates pushed into the WHERE clause,
        # the same query returns ~1.5k rows in well under a second, and
        # we only do it once per dialog click.
        cellview_df = _load_cellview_well_slice(plate_id, well_id, timepoint)

        # Zarr fast path: if a stitched zarr cache exists for this well,
        # fetch crops directly from the canvas instead of downloading each
        # field from OMERO + diskcache. Falls through silently to the
        # legacy path when no zarr is available.
        zarr_result = _try_zarr_direct_load(
            plate_id=plate_id,
            well_id=well_id,
            wanted_image_ids=wanted_image_ids,
            segmentation=segmentation,
            crop_size=crop_size,
            timepoint=timepoint,
            cellcycle=cellcycle,
            classifier_column=classifier_column,
            classifier_class=classifier_class,
            cellview_df=cellview_df,
        )
        if zarr_result is not None:
            (
                all_crops,
                all_crop_labels,
                all_image_ids,
                all_cell_meta,
            ) = zarr_result
            logger.info(
                "Zarr fast path produced %d crops for well %s",
                len(all_crops),
                well_id,
            )
        else:
            (
                all_crops,
                all_crop_labels,
                all_image_ids,
                all_cell_meta,
            ) = _load_crops_via_omero(
                conn=conn,
                plate=plate,
                well=well,
                plate_id=plate_id,
                well_id=well_id,
                image_indices=image_indices,
                image_id_by_index=image_id_by_index,
                timepoint=timepoint,
                crop_size=crop_size,
                segmentation=segmentation,
                cellcycle=cellcycle,
                classifier_column=classifier_column,
                classifier_class=classifier_class,
                cellview_df=cellview_df,
            )

        if not all_crops:
            return (
                False,
                "No crops could be generated from any of the selected images",
            )

        logger.info(
            f"Total crops across {len(image_indices)} image(s): {len(all_crops)}"
        )

        # Step 5: Limit to gallery-sized subset
        n_crops = metadata.get("n_crops", 0)
        if n_crops <= 0:
            rows = user_data_dict.get("rows", 0)
            columns = user_data_dict.get("columns", 0)
            if rows > 0 and columns > 0:
                n_crops = rows * columns
        if n_crops > 0 and len(all_crops) > n_crops:
            import random

            total_before = len(all_crops)
            indices = random.sample(range(total_before), n_crops)
            all_crops = [all_crops[i] for i in indices]
            all_crop_labels = [all_crop_labels[i] for i in indices]
            all_cell_meta = [all_cell_meta[i] for i in indices]
            logger.info(
                f"Selected {n_crops} random crops from {total_before} available"
            )

        # Step 6: Populate omero_data
        omero_data.selected_crops = all_crops
        omero_data.selected_labels = all_crop_labels
        omero_data.selected_cell_meta = all_cell_meta
        omero_data.plate_id = plate_id
        omero_data.well_pos_list = [well_id]
        omero_data.image_input = image_input
        omero_data.image_ids = all_image_ids

        # Initialize class assignments as "unassigned"
        omero_data.selected_classes = ["unassigned"] * len(all_crops)

        # Step 7: Apply masks only if no_background is enabled
        no_background = user_data_dict.get("no_background", True)
        if no_background:
            logger.info(
                f"Applying masks to {len(all_crops)} images (no_background=True)"
            )
            masked_images = apply_masks_to_crops(omero_data)
        else:
            logger.info(
                f"Keeping background for {len(all_crops)} images (no_background=False)"
            )
            masked_images = list(omero_data.selected_crops)

        # Convert channel names to indices, matching session_utils approach.
        # Empties are stripped earlier in the gallery widget; here we
        # just resolve names against channel_data and pack into RGB by
        # position (R=ch0, G=ch1, B=ch2 or 0).
        channels = [ch for ch in user_data_dict.get("channels", []) if ch]
        channel_data = metadata.get("channel_data", {})
        channel_indices = []
        for ch in channels:
            try:
                channel_indices.append(int(ch))
                continue
            except (ValueError, TypeError):
                pass
            val = channel_data.get(ch)
            if val is not None:
                try:
                    channel_indices.append(int(float(val)))
                except (ValueError, TypeError):
                    logger.warning(
                        f"Channel index '{val}' for '{ch}' is not a valid number."
                    )
            else:
                logger.warning(
                    f"Channel '{ch}' not found in channel_data map."
                )

        # If channel names could not be resolved, fall back gracefully
        if not channel_indices and masked_images:
            n_requested = len(channels)
            if n_requested <= 1:
                logger.info(
                    "Could not resolve channel names; averaging all "
                    "channels for grayscale display"
                )
            else:
                num_available = (
                    masked_images[0].shape[2]
                    if len(masked_images[0].shape) > 2
                    else 1
                )
                channel_indices = list(range(min(n_requested, num_available)))
                logger.info(
                    f"Could not resolve channel names, using first "
                    f"{len(channel_indices)} of {num_available} channels"
                )

        if channel_indices:
            processed_images = [
                fill_missing_channels(img, channel_indices)
                for img in masked_images
            ]
            # selected_crops stores only the selected channels (no RGB padding) for training data
            omero_data.selected_crops = [
                img[..., channel_indices] for img in masked_images
            ]
        else:
            processed_images = [
                np.mean(img, axis=-1, keepdims=True).astype(img.dtype)
                for img in masked_images
            ]
            omero_data.selected_crops = processed_images
        logger.info(
            f"After channel processing: {len(processed_images)} images"
        )

        # Add contours if requested
        if user_data_dict.get("contour", False):
            processed_images = [
                draw_contours(img, lbl)
                for img, lbl in zip(
                    processed_images, all_crop_labels, strict=False
                )
            ]
            logger.info(
                f"After contour drawing: {len(processed_images)} images"
            )

        omero_data.selected_images = processed_images
        logger.info(
            f"Final: Set {len(omero_data.selected_images)} images in omero_data"
        )

        # Update user_data with loaded metadata
        user_data.populate_from_dict(user_data_dict)
        user_data.well = well_id
        user_data.cellcycle = cellcycle

        logger.info(
            f"Successfully loaded {len(processed_images)} images for annotation"
        )
        return (
            True,
            f"Loaded {len(processed_images)} cells from images '{image_input}'",
        )

    except Exception as e:
        logger.exception(f"Failed to load crops from OMERO: {e}")
        return False, f"Error loading data: {str(e)}"


def _load_segmentation_masks(
    conn: BlitzGateway, plate_id: int, image_id: int, timepoint: int
) -> tuple[bool, Any]:
    """Load segmentation masks from OMERO dataset.

    Args:
        conn: OMERO connection
        plate_id: Plate ID
        image_id: Image ID
        timepoint: Timepoint index

    Returns:
        (success: bool, masks_array or error_message)
    """
    try:
        # Find dataset for this plate
        dataset_name = str(plate_id)

        # Search for dataset by name
        dataset = None
        for ds in conn.getObjects("Dataset"):
            if ds.getName() == dataset_name:
                dataset = ds
                break

        if not dataset:
            return False, f"Dataset for plate {plate_id} not found"

        # Find segmentation image
        seg_image_name = f"{image_id}_segmentation"
        seg_image = None
        for img in dataset.listChildren():
            if img.getName() == seg_image_name:
                seg_image = img
                break

        if not seg_image:
            return False, f"Segmentation masks not found for image {image_id}"

        # Load segmentation pixels (via diskcache)
        cached = get_image_timepoint(
            conn, seg_image.getId(), timepoint, tag=plate_id
        )  # ZYXC
        masks = cached.squeeze(axis=0).astype(np.int32)  # YXC

        logger.info(f"Loaded segmentation masks with shape: {masks.shape}")
        return True, masks

    except Exception as e:
        logger.exception(f"Failed to load segmentation masks: {e}")
        return False, f"Error loading masks: {str(e)}"


def _load_flatfield_correction(
    conn: BlitzGateway, plate_id: int
) -> tuple[bool, Any]:
    """Load flatfield correction masks from OMERO dataset.

    Args:
        conn: OMERO connection
        plate_id: Plate ID

    Returns:
        (success: bool, flatfield_array or None)
    """
    try:
        # Find dataset for this plate
        dataset_name = str(plate_id)
        dataset = None
        for ds in conn.getObjects("Dataset"):
            if ds.getName() == dataset_name:
                dataset = ds
                break

        if not dataset:
            logger.warning(f"Dataset for plate {plate_id} not found")
            return False, None

        # Find flatfield image
        ff_image = None
        for img in dataset.listChildren():
            if "flatfield" in img.getName().lower():
                ff_image = img
                break

        if not ff_image:
            logger.warning("No flatfield correction image found")
            return False, None

        # Load flatfield pixels (via diskcache)
        cached = get_image_timepoint(
            conn, ff_image.getId(), 0, tag=plate_id
        )  # ZYXC
        flatfield = cached.squeeze(axis=0).astype(np.float32)  # YXC

        logger.info(
            f"Loaded flatfield correction with shape: {flatfield.shape}"
        )
        return True, flatfield

    except Exception as e:
        logger.warning(f"Failed to load flatfield correction: {e}")
        return False, None


def _load_centroids_from_cellview(
    cellview_df: Any,
    well_id: str,
    image_id: int,
    segmentation: str,
    cellcycle: str = "All",
    classifier_column: str = "",
    classifier_class: str = "",
) -> tuple[bool, Any]:
    """Extract centroids for one image from a pre-loaded CellView slice.

    Args:
        cellview_df: pandas DataFrame already filtered to the target
            ``(plate, well, timepoint)`` via SQL pushdown by the caller.
            Pass ``None`` to signal "no CellView data available".
        well_id: Well position (used only for diagnostic messages).
        image_id: Image ID (actual OMERO image ID, not index)
        segmentation: "nucleus" or "cell"
        cellcycle: Cell cycle phase filter ("All", "G1", "S", "G2/M", etc.)
        classifier_column: CellView column to filter on (e.g. "classifier_mitosis")
        classifier_class: Class value to keep; empty string means no filter

    Returns:
        (success: bool, centroids_array or error_message)
    """
    if cellview_df is None:
        return False, "No CellView data provided"

    try:
        # SQL pushdown already restricted to (well, timepoint); only the
        # per-image filter remains. This is the loop-body restriction —
        # the well/timepoint reduction happened once at the top level.
        filtered = cellview_df[cellview_df["image_id"] == image_id]

        # Apply cell cycle phase filter
        if cellcycle != "All" and "cell_cycle" in filtered.columns:
            filtered = filtered[filtered["cell_cycle"] == cellcycle]
            logger.info(
                f"Cell cycle filter '{cellcycle}': {len(filtered)} cells remaining"
            )

        # Apply classifier filter
        if (
            classifier_column
            and classifier_class
            and classifier_column in filtered.columns
        ):
            filtered = filtered[
                filtered[classifier_column] == classifier_class
            ]
            logger.info(
                f"Classifier filter '{classifier_column}={classifier_class}': {len(filtered)} cells remaining"
            )

        if filtered.empty:
            return False, f"No data found for well {well_id}, image {image_id}"

        # Extract centroids based on segmentation type
        if segmentation == "nucleus":
            centroid_cols = ["centroid-0-nuc", "centroid-1-nuc"]
        else:
            centroid_cols = ["centroid-0-cell", "centroid-1-cell"]

        # Check if columns exist
        for col in centroid_cols:
            if col not in filtered.columns:
                return (
                    False,
                    f"Centroid column '{col}' not found in CellView data",
                )

        # Extract centroids as numpy array
        centroids = filtered[centroid_cols].to_numpy()

        logger.info(f"Loaded {len(centroids)} centroids from CellView")
        return True, centroids

    except Exception as e:
        logger.exception(f"Failed to load centroids from CellView: {e}")
        return False, f"Error loading centroids: {str(e)}"


def _generate_crops(
    image: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    centroids: np.ndarray[Any, Any],
    crop_size: int,
    segmentation: str,
) -> tuple[
    list[np.ndarray[Any, Any]],
    list[np.ndarray[Any, Any]],
    list[tuple[int, int]],
]:
    """Generate cropped images and label masks around centroids.

    Args:
        image: Full image array (H, W, C)
        labels: Full label masks (H, W, C) with channels [nucleus, cell]
        centroids: Array of centroids (N, 2) with [row, col]
        crop_size: Size of square crop
        segmentation: "nucleus" or "cell"

    Returns:
        Tuple of ``(crops_list, labels_list, kept_centroids)``. The third
        element is the ``(row, col)`` for each crop that survived the
        background / empty-label skips — the caller pairs it with the
        image_id to populate ``omero_data.selected_cell_meta`` so the
        "skip already-annotated cells" feature works for direct-load
        sessions.
    """
    crops = []
    crop_labels = []
    kept_centroids: list[tuple[int, int]] = []

    # Select appropriate label channel
    if labels.ndim == 3 and labels.shape[-1] >= 2:
        channel_idx = 0 if segmentation == "nucleus" else 1
        label_array = labels[..., channel_idx]
    else:
        label_array = np.squeeze(labels)

    for centroid in centroids:
        row, col = int(centroid[0]), int(centroid[1])

        # Calculate crop bounds
        row_start, row_end = calculate_crop_coordinates(
            row, image.shape[0], crop_size
        )
        col_start, col_end = calculate_crop_coordinates(
            col, image.shape[1], crop_size
        )

        # Extract crop
        crop = image[row_start:row_end, col_start:col_end, :]
        label_crop = label_array[row_start:row_end, col_start:col_end]

        # Isolate only the target cell's label (matching the gallery path's
        # erase_masks behaviour so that no_background works correctly).
        # Look up the label ID at the centroid position; if the centroid lands
        # on background (label == 0), skip this crop.
        target_label = label_array[row, col]
        if target_label == 0:
            continue
        label_crop = np.where(label_crop == target_label, label_crop, 0)

        # Pad if needed
        if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
            crop, label_crop = pad_region(crop, label_crop, crop_size)

        # Only add if label is non-empty
        if np.any(label_crop):
            crops.append(crop)
            crop_labels.append(label_crop)
            kept_centroids.append((row, col))

    logger.info(f"Generated {len(crops)} valid crops")
    return crops, crop_labels, kept_centroids


def _load_crops_via_omero(
    *,
    conn: BlitzGateway,
    plate: Any,
    well: Any,
    plate_id: int,
    well_id: str,
    image_indices: list[int],
    image_id_by_index: dict[int, int],
    timepoint: int,
    crop_size: int,
    segmentation: str,
    cellcycle: str,
    classifier_column: str,
    classifier_class: str,
    cellview_df: Any = None,
) -> tuple[
    list[np.ndarray[Any, Any]],
    list[np.ndarray[Any, Any]],
    list[int],
    list[dict[str, Any]],
]:
    """Per-field crop generation via OMERO + diskcache (legacy path).

    Used when no zarr cache is available for the plate. Downloads each
    field's pixels and segmentation masks, applies flatfield correction,
    and crops around CellView centroids.

    ``cellview_df`` is the pre-loaded ``(plate, well, timepoint)`` slice
    (SQL-pushdown). Without the timepoint filter, time-lapse plates would
    centre crops on a mix of timepoints' coordinates while reading pixels
    at a single ``t`` — cells would end up off-frame. This is the same
    fix that's been in the zarr fast path; the legacy path needs it too.

    Returns ``(crops, labels, image_ids, cell_meta)``. ``cell_meta`` is
    one dict per crop (``centroid_row``, ``centroid_col``, ``image_id``)
    matching the gallery path so ``get_used_centroids`` works for
    direct-load sessions too.
    """
    del plate  # unused — kept on signature for symmetry with zarr path
    logger.info(f"Loading flatfield correction for plate {plate_id}")
    ff_success, flatfield = _load_flatfield_correction(conn, plate_id)

    all_crops: list[np.ndarray[Any, Any]] = []
    all_crop_labels: list[np.ndarray[Any, Any]] = []
    all_image_ids: list[int] = []
    all_cell_meta: list[dict[str, Any]] = []

    for img_idx in image_indices:
        image = well.getImage(img_idx)
        if not image:
            logger.warning(f"Image at index {img_idx} not found, skipping")
            continue

        image_id = image_id_by_index.get(img_idx, image.getId())
        logger.info(f"Processing image index {img_idx} (OMERO ID: {image_id})")
        all_image_ids.append(image_id)

        cached = get_image_timepoint(
            conn, image_id, timepoint, tag=plate_id
        )  # ZYXC
        raw = cached.squeeze(axis=0)  # YXC, original dtype
        image_array = raw.astype(np.float32)

        if ff_success and flatfield is not None:
            image_array = image_array / flatfield

        n_channels = image_array.shape[-1]
        for ch in range(n_channels):
            ch_slice = image_array[..., ch]
            p_high = float(np.percentile(ch_slice, 99.9))
            if p_high > 0:
                image_array[..., ch] = np.clip(ch_slice / p_high, 0.0, 1.0)

        success, masks = _load_segmentation_masks(
            conn, plate_id, image_id, timepoint
        )
        if not success:
            logger.warning(
                f"Segmentation masks not found for image {image_id}, "
                f"skipping: {masks}"
            )
            continue

        success, centroids = _load_centroids_from_cellview(
            cellview_df,
            well_id,
            image_id,
            segmentation,
            cellcycle,
            classifier_column,
            classifier_class,
        )
        if not success:
            logger.warning(
                f"Centroids not found for image {image_id}, "
                f"skipping: {centroids}"
            )
            continue

        logger.info(f"Image {image_id}: {len(centroids)} centroids found")

        crops, crop_labels, kept_centroids = _generate_crops(
            image_array, masks, centroids, crop_size, segmentation
        )
        all_crops.extend(crops)
        all_crop_labels.extend(crop_labels)
        all_cell_meta.extend(
            {
                "centroid_row": row,
                "centroid_col": col,
                "image_id": image_id,
            }
            for row, col in kept_centroids
        )

    return all_crops, all_crop_labels, all_image_ids, all_cell_meta


def _try_zarr_direct_load(
    *,
    plate_id: int,
    well_id: str,
    wanted_image_ids: list[int],
    segmentation: str,
    crop_size: int,
    timepoint: int,
    cellcycle: str,
    classifier_column: str,
    classifier_class: str,
    cellview_df: Any = None,
) -> (
    tuple[
        list[np.ndarray[Any, Any]],
        list[np.ndarray[Any, Any]],
        list[int],
        list[dict[str, Any]],
    ]
    | None
):
    """Try to load crops directly from the OME-Zarr stitched plate cache.

    Returns ``(crops, label_crops, image_ids, cell_meta)`` when the zarr
    cache holds this well, otherwise ``None`` so the caller can fall back
    to the OMERO + diskcache path. Crops are normalised per channel to
    ``[0, 1]`` using the 99.9th percentile (matching the OMERO path).
    ``cell_meta`` is one dict per surviving crop with ``centroid_row``,
    ``centroid_col``, ``image_id`` so ``get_used_centroids`` works for
    direct-load sessions.

    ``cellview_df`` should already be filtered to the requested
    ``(plate, well, timepoint)`` slice by the caller (SQL pushdown). The
    timepoint filter is what makes time-lapse crops centre on the right
    cells — without it, ``fetch_crop`` would read pixels at one ``t``
    using centroids averaged across all timepoints.
    """
    if resolve_to_zarr(plate_id) is None:
        return None
    if well_id not in cached_wells(plate_id):
        logger.info(
            "Zarr cache present for plate %d but well %s not built; "
            "falling back to OMERO loader",
            plate_id,
            well_id,
        )
        return None

    if cellview_df is None:
        logger.warning(
            "No CellView slice provided for plate %d well %s; "
            "falling back to OMERO loader",
            plate_id,
            well_id,
        )
        return None

    filtered = cellview_df
    if wanted_image_ids:
        filtered = filtered[filtered["image_id"].isin(wanted_image_ids)]
    if cellcycle != "All" and "cell_cycle" in filtered.columns:
        filtered = filtered[filtered["cell_cycle"] == cellcycle]
    if (
        classifier_column
        and classifier_class
        and classifier_column in filtered.columns
    ):
        filtered = filtered[filtered[classifier_column] == classifier_class]

    if filtered.empty:
        logger.warning(
            "No CellView rows after filtering for plate %d well %s",
            plate_id,
            well_id,
        )
        return None

    if segmentation == "nucleus":
        cy_col, cx_col = "centroid-0-nuc", "centroid-1-nuc"
        mask_name = "nuclei"
    else:
        cy_col, cx_col = "centroid-0-cell", "centroid-1-cell"
        mask_name = "cells"

    for col in (cy_col, cx_col):
        if col not in filtered.columns:
            logger.warning(
                "CellView missing column %s; falling back to OMERO loader",
                col,
            )
            return None

    # Canvas-percentile contrast: prefer the per-channel (lo, hi) window
    # that was set when the plate was loaded into the viewer (matches the
    # napari layer's ``contrast_limits``). Falls back to a per-crop 99.9th
    # percentile when intensities are absent (headless callers).
    from omero_screen_napari.omero_data_singleton import (
        omero_data as _od,
    )

    canvas_intensities: dict[int, tuple[int, int]] = (
        getattr(_od, "intensities", None) or {}
    )

    crops: list[np.ndarray[Any, Any]] = []
    crop_labels: list[np.ndarray[Any, Any]] = []
    seen_image_ids: list[int] = []
    cell_meta: list[dict[str, Any]] = []
    n_skipped = 0

    for _row_idx, row in filtered.iterrows():
        cy = float(row[cy_col])
        cx = float(row[cx_col])
        image_id = int(row["image_id"])
        if image_id not in seen_image_ids:
            seen_image_ids.append(image_id)

        try:
            crop_cyx = fetch_crop(
                plate_id,
                well_id,
                label=0,
                centroid=(cy, cx),
                size=crop_size,
                t=timepoint,
            )
            label_crop = fetch_label_crop(
                plate_id,
                well_id,
                centroid=(cy, cx),
                size=crop_size,
                t=timepoint,
                mask_name=mask_name,
            )
        except (FileNotFoundError, KeyError) as exc:
            logger.warning(
                "Zarr fetch failed mid-iteration (plate %d well %s): %s",
                plate_id,
                well_id,
                exc,
            )
            return None

        # (C, Y, X) → (Y, X, C), then normalise per-channel.
        crop_yxc = np.transpose(crop_cyx, (1, 2, 0)).astype(
            np.float32, copy=False
        )
        n_channels = crop_yxc.shape[-1]
        for ch in range(n_channels):
            ch_slice = crop_yxc[..., ch]
            window = canvas_intensities.get(ch)
            if window is not None:
                lo, hi = float(window[0]), float(window[1])
                rng = max(hi - lo, 1.0)
                crop_yxc[..., ch] = np.clip((ch_slice - lo) / rng, 0.0, 1.0)
            else:
                p_high = float(np.percentile(ch_slice, 99.9))
                if p_high > 0:
                    crop_yxc[..., ch] = np.clip(ch_slice / p_high, 0.0, 1.0)

        # Isolate the target cell's label at the centroid (matches
        # _generate_crops's policy for no_background isolation).
        h, w = label_crop.shape
        cy_local, cx_local = h // 2, w // 2
        target_label = int(label_crop[cy_local, cx_local])
        if target_label == 0:
            n_skipped += 1
            continue
        isolated = np.where(label_crop == target_label, label_crop, 0)
        if not np.any(isolated):
            n_skipped += 1
            continue

        crops.append(crop_yxc)
        crop_labels.append(isolated)
        cell_meta.append(
            {
                "centroid_row": int(cy),
                "centroid_col": int(cx),
                "image_id": image_id,
            }
        )

    if n_skipped:
        logger.info(
            "Zarr fast path skipped %d centroids on background", n_skipped
        )

    if not crops:
        logger.warning(
            "Zarr fast path produced no valid crops; "
            "falling back to OMERO loader"
        )
        return None

    return crops, crop_labels, seen_image_ids, cell_meta
