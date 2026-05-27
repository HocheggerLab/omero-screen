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

from omero_screen_napari.crop_pipeline import (
    CropPipeline,
    CropResult,
    CropSourceError,
    OmeroSource,
    ZarrSource,
)
from omero_screen_napari.gallery_api import (
    draw_contours,
    fill_missing_channels,
)
from omero_screen_napari.session_utils import apply_masks_to_crops

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
        # via SQL pushdown (the `well` + `timepoint` predicates cut the
        # full ~695k-row plate dataframe to ~1.5k rows), then filter to the
        # requested fields / cellcycle / classifier before cropping.
        cellview_df = _load_cellview_well_slice(plate_id, well_id, timepoint)
        if cellview_df is None:
            return False, "No CellView data for this plate/well/timepoint"
        centroids = _filter_cellview_df(
            cellview_df,
            wanted_image_ids=wanted_image_ids,
            cellcycle=cellcycle,
            classifier_column=classifier_column,
            classifier_class=classifier_class,
        )

        # Pick a crop source: the stitched OME-Zarr canvas when this well is
        # built (fast — no per-field downloads), else stream the fields from
        # OMERO. Both feed the same CropPipeline. The viewer's per-channel
        # contrast window (when a plate was loaded via the welldata widget)
        # is passed explicitly so the zarr source reproduces it.
        intensities = getattr(omero_data, "intensities", None) or None
        result = _run_crop_pipeline(
            conn=conn,
            plate_id=plate_id,
            well_id=well_id,
            well=well,
            image_id_by_index=image_id_by_index,
            centroids=centroids,
            segmentation=segmentation,
            crop_size=crop_size,
            timepoint=timepoint,
            intensities=intensities,
        )
        all_crops = result.crops
        all_crop_labels = result.labels
        all_image_ids = result.image_ids
        all_cell_meta = result.cell_meta

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


def _filter_cellview_df(
    cellview_df: Any,
    *,
    wanted_image_ids: list[int],
    cellcycle: str,
    classifier_column: str,
    classifier_class: str,
) -> Any:
    """Filter the well slice to the requested fields / cellcycle / classifier.

    The well + timepoint predicates were already pushed into DuckDB by
    ``_load_cellview_well_slice``; this applies the remaining pandas-side
    filters before the rows reach :class:`CropPipeline` (which expects an
    already-filtered centroid set).
    """
    filtered = cellview_df
    if wanted_image_ids:
        filtered = filtered[filtered["image_id"].isin(wanted_image_ids)]
    if cellcycle != "All" and "cell_cycle" in filtered.columns:
        filtered = filtered[filtered["cell_cycle"] == cellcycle]
        logger.info(
            "Cell cycle filter '%s': %d cells remaining",
            cellcycle,
            len(filtered),
        )
    if (
        classifier_column
        and classifier_class
        and classifier_column in filtered.columns
    ):
        filtered = filtered[filtered[classifier_column] == classifier_class]
        logger.info(
            "Classifier filter '%s=%s': %d cells remaining",
            classifier_column,
            classifier_class,
            len(filtered),
        )
    return filtered


def _run_crop_pipeline(
    *,
    conn: BlitzGateway,
    plate_id: int,
    well_id: str,
    well: Any,
    image_id_by_index: dict[int, int],
    centroids: Any,
    segmentation: str,
    crop_size: int,
    timepoint: int,
    intensities: dict[int, tuple[float, float]] | None,
) -> CropResult:
    """Pick a crop source and run :class:`CropPipeline`.

    Prefers the stitched OME-Zarr canvas (no per-field downloads) when the
    well is built; otherwise streams fields from OMERO. If the zarr store
    can't serve a crop mid-run, falls back to the OMERO source for the whole
    run (matching the legacy per-image fallback at run granularity).
    """
    if centroids is None or centroids.empty:
        logger.warning("No centroids after filtering for well %s", well_id)
        return CropResult()

    def _omero_source() -> OmeroSource:
        return OmeroSource(conn, plate_id, well, image_id_by_index)

    try:
        source: Any = ZarrSource(plate_id, well_id, intensities=intensities)
        logger.info("Direct loader: using zarr fast path for well %s", well_id)
    except CropSourceError as exc:
        logger.info(
            "Zarr unavailable (%s); using OMERO loader for well %s",
            exc,
            well_id,
        )
        source = _omero_source()

    pipeline = CropPipeline(
        source=source,
        centroids_df=centroids,
        segmentation=segmentation,  # type: ignore[arg-type]
        crop_size=crop_size,
        timepoint=timepoint,
    )
    try:
        return pipeline.run()
    except CropSourceError as exc:
        logger.warning(
            "Zarr source failed mid-run (%s); falling back to OMERO loader",
            exc,
        )
        return CropPipeline(
            source=_omero_source(),
            centroids_df=centroids,
            segmentation=segmentation,  # type: ignore[arg-type]
            crop_size=crop_size,
            timepoint=timepoint,
        ).run()
