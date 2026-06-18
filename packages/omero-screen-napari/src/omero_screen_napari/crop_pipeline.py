"""Unified crop-generation core shared by the gallery and training paths.

Three call sites used to each grow their own copy of "turn (well, timepoint,
cell centroids) into a list of normalised crops + isolated label masks":

* the welldata gallery (``gallery_api.CroppedImageParser``),
* the direct-load zarr fast path (``direct_omero_loader._try_zarr_direct_load``),
* the direct-load OMERO fallback (``direct_omero_loader._load_crops_via_omero``).

They diverged subtly — the timepoint filter, ``selected_cell_meta``
population, and label isolation were all fixed in one path but forgotten in
another at various times. This module collapses the shared **core** into one
:class:`CropPipeline` driven by a small :class:`CropSource` strategy, so a
future fix lands once.

Scope is deliberately the *core* only. The "finalize" layer that builds the
two downstream outputs — ``selected_crops`` (channel-subset training data,
saved to NPY) and ``selected_images`` (RGB-packed + contour, on-screen
display) — stays where it is (gallery's ``RandomImageParser`` and the
direct-load post-processing block). The pipeline emits raw normalised
multi-channel crops ``(Y, X, C)`` and isolated label masks ``(Y, X)``; it
produces nothing display- or NPY-shaped.

Centroid filtering (cellcycle / classifier / timepoint) is the caller's job —
the gallery filters a polars LazyFrame and the direct loader pushes predicates
into DuckDB, and those interfaces legitimately differ. Callers pass an
already-filtered pandas DataFrame.
"""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from loguru import logger
from skimage.measure import label as sk_label
from skimage.measure import regionprops

if TYPE_CHECKING:
    import pandas as pd
    from omero.gateway import BlitzGateway

    from omero_screen_napari.omero_data import OmeroData


# ---------------------------------------------------------------------------
# Low-level crop helpers (moved here from gallery_api so both the gallery and
# direct-load paths share one implementation; gallery_api re-exports them for
# backward compatibility).
# ---------------------------------------------------------------------------


def calculate_crop_coordinates(
    centroid: int, max_length: int, crop_size: int
) -> tuple[int, int]:
    """Return ``(start, end)`` pixel bounds for a crop centred on ``centroid``."""
    start = int(max(0, centroid - crop_size // 2))
    end = int(min(max_length, centroid + crop_size // 2))
    return start, end


def crop_region(
    current_data: np.ndarray[Any, Any],
    current_labels: np.ndarray[Any, Any],
    centroid_row: int,
    centroid_col: int,
    crop_size: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Crop an ``(Y, X, C)`` image and its ``(Y, X)`` labels around a centroid."""
    crop_row_start, crop_row_end = calculate_crop_coordinates(
        centroid_row, current_data.shape[-3], crop_size
    )
    crop_col_start, crop_col_end = calculate_crop_coordinates(
        centroid_col, current_data.shape[-2], crop_size
    )
    cropped_region = current_data[
        crop_row_start:crop_row_end, crop_col_start:crop_col_end, :
    ]
    cropped_label = current_labels[
        crop_row_start:crop_row_end, crop_col_start:crop_col_end
    ]
    return cropped_region, cropped_label


def pad_region(
    cropped_region: np.ndarray[Any, Any],
    cropped_label: np.ndarray[Any, Any],
    crop_size: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Centre-pad a crop + label that ran off the image edge to ``crop_size``."""
    pad_row = crop_size - cropped_region.shape[0]
    pad_col = crop_size - cropped_region.shape[1]
    cropped_region = np.pad(
        cropped_region,
        (
            (pad_row // 2, pad_row - pad_row // 2),
            (pad_col // 2, pad_col - pad_col // 2),
            (0, 0),
        ),
        mode="constant",
    )
    cropped_label = np.pad(
        cropped_label,
        (
            (pad_row // 2, pad_row - pad_row // 2),
            (pad_col // 2, pad_col - pad_col // 2),
        ),
        mode="constant",
    )
    return cropped_region, cropped_label


def erase_masks(
    cropped_label: np.ndarray[Any, Any],
    obj_id: int | list[int] | float,
    tol: int = 10,
) -> list[np.ndarray[Any, Any]]:
    """Isolate the target cell's label(s) from neighbours in a label crop.

    Three behaviours, by ``obj_id`` type:

    * scalar integer (or float with ``int(id) == id``) — zero everything that
      isn't that label, return the single mask;
    * iterable of ids (multi-nucleate cells: CellView returns a list) — return
      one isolated mask per id;
    * otherwise (legacy averaged float ids) — fall back to keeping the unique
      label whose centroid lies within ``tol`` px of the crop centre.

    All three current pipelines share this. Direct-load paths historically did
    only a centroid-pixel lookup (no multi-nucleate support); routing them
    through ``erase_masks`` is the agreed capability gain — identical output
    for single-cell crops, correct masks for multi-nucleate cells.
    """
    if isinstance(obj_id, int | float) and int(obj_id) == obj_id:
        cropped_label[cropped_label != obj_id] = 0
        return [cropped_label]

    labels = []
    if isinstance(obj_id, Iterable):
        for i in obj_id:
            label_mask = cropped_label.copy()
            label_mask[cropped_label != i] = 0
            labels.append(label_mask)
        return labels

    center_row, center_col = np.array(cropped_label.shape) // 2
    unique_labels = np.unique(cropped_label)
    for unique_label in unique_labels:
        if unique_label == 0:
            continue
        binary_mask = cropped_label == unique_label
        if np.sum(binary_mask) == 0:
            continue
        label_props = regionprops(sk_label(binary_mask))  # type: ignore[no-untyped-call]
        if len(label_props) == 1:
            cropped_centroid_row, cropped_centroid_col = label_props[
                0
            ].centroid
            if (
                abs(cropped_centroid_row - center_row) <= tol
                and abs(cropped_centroid_col - center_col) <= tol
            ):
                label_mask = cropped_label.copy()
                label_mask[cropped_label != unique_label] = 0
                labels.append(label_mask)
    return labels


# ---------------------------------------------------------------------------
# Pipeline contract
# ---------------------------------------------------------------------------


class CropSourceError(RuntimeError):
    """Raised when a source can't serve a crop mid-run (e.g. evicted zarr).

    The caller catches this to fall back to another source for the whole run
    (matching the gallery's old per-image zarr→in-memory fallback, just at run
    granularity).
    """


@dataclass
class CropResult:
    """Output of :meth:`CropPipeline.run`.

    ``crops`` and ``labels`` align positionally with ``cell_meta``. ``crops``
    are raw normalised ``(Y, X, C)`` float32 arrays in ``[0, 1]`` (all channels
    present — channel selection happens in the finalize layer). ``labels`` are
    ``(Y, X)`` with the target cell's label isolated. ``image_ids`` is the
    distinct set of OMERO image IDs that produced at least one crop.
    """

    crops: list[np.ndarray[Any, Any]] = field(default_factory=list)
    labels: list[np.ndarray[Any, Any]] = field(default_factory=list)
    cell_meta: list[dict[str, Any]] = field(default_factory=list)
    image_ids: list[int] = field(default_factory=list)


class CropSource(Protocol):
    """Where pixels and labels come from for one (plate, well) at one ``t``.

    Implementations own fetch + normalisation, each replicating its origin
    path's current behaviour so display output is unchanged.
    """

    def fetch(
        self,
        image_id: int,
        centroid: tuple[float, float],
        size: int,
        t: int,
        mask_name: Literal["nuclei", "cells"],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return ``(crop, label_crop)``.

        ``crop`` is ``(Y, X, C)`` float32 normalised to ``[0, 1]``;
        ``label_crop`` is ``(Y, X)`` integer labels (not yet isolated).
        Raise :class:`CropSourceError` if the crop can't be served.
        """
        ...


def _centroid_columns(
    segmentation: Literal["nucleus", "cell"],
) -> tuple[str, str, str]:
    """Return ``(row_col, col_col, id_col)`` CellView column names."""
    if segmentation == "nucleus":
        return "centroid-0-nuc", "centroid-1-nuc", "label"
    return "centroid-0-cell", "centroid-1-cell", "Cyto_ID"


def _resolve_centroids(
    image_df: pd.DataFrame, segmentation: Literal["nucleus", "cell"]
) -> tuple[list[Any], list[Any], list[Any]]:
    """Extract ``(rows, cols, ids)`` for one image's centroids.

    Mirrors the gallery's old ``_select_centroids``:

    * nucleus — keep every row; ``label`` ids may be stringified lists
      (multi-nucleate) which are ``literal_eval``-ed back to lists;
    * cell — de-duplicate on the centroid pair (one crop per cell, not per
      nucleus).
    """
    c0, c1, id_col = _centroid_columns(segmentation)
    if segmentation == "nucleus":
        ids = image_df[id_col]
        if ids.dtype == object:
            ids = ids.map(
                lambda v: literal_eval(v) if isinstance(v, str) else v
            )
        return (
            image_df[c0].tolist(),
            image_df[c1].tolist(),
            ids.tolist(),
        )
    unique = image_df.drop_duplicates(subset=[c0, c1])
    return (
        unique[c0].tolist(),
        unique[c1].tolist(),
        unique[id_col].tolist(),
    )


class CropPipeline:
    """Generate crops + isolated label masks for one filtered centroid set.

    The caller supplies an already-filtered ``centroids_df`` (well / timepoint
    / cellcycle / classifier filtering happens upstream) and a
    :class:`CropSource`. ``run`` groups by image, resolves centroids, fetches
    each crop from the source, isolates the target label via
    :func:`erase_masks`, and collects the non-empty results.
    """

    def __init__(
        self,
        *,
        source: CropSource,
        centroids_df: pd.DataFrame,
        segmentation: Literal["nucleus", "cell"],
        crop_size: int,
        timepoint: int,
        excluded_centroids: set[tuple[int, int, int]] | None = None,
    ) -> None:
        self._source = source
        self._df = centroids_df
        self._segmentation = segmentation
        self._crop_size = crop_size
        self._timepoint = timepoint
        self._excluded = excluded_centroids or set()

    def run(self) -> CropResult:
        result = CropResult()
        if self._df is None or self._df.empty:
            logger.warning(
                "CropPipeline: empty centroid set, no crops produced"
            )
            return result

        mask_name: Literal["nuclei", "cells"] = (
            "nuclei" if self._segmentation == "nucleus" else "cells"
        )
        for image_id in sorted(self._df["image_id"].unique()):
            image_id = int(image_id)
            image_df = self._df[self._df["image_id"] == image_id]
            rows, cols, ids = _resolve_centroids(image_df, self._segmentation)
            produced_for_image = False
            for row, col, obj_id in zip(rows, cols, ids, strict=False):
                if (image_id, int(row), int(col)) in self._excluded:
                    continue
                crop, label_crop = self._source.fetch(
                    image_id,
                    (float(row), float(col)),
                    self._crop_size,
                    self._timepoint,
                    mask_name,
                )
                for isolated in erase_masks(label_crop.copy(), obj_id):
                    if np.any(isolated):
                        result.crops.append(crop)
                        result.labels.append(isolated)
                        result.cell_meta.append(
                            {
                                "centroid_row": int(row),
                                "centroid_col": int(col),
                                "image_id": image_id,
                            }
                        )
                        produced_for_image = True
            if produced_for_image:
                result.image_ids.append(image_id)

        logger.info(f"CropPipeline produced {len(result.crops)} crops")
        return result


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _scale_field(
    image: np.ndarray[Any, Any],
    intensities: dict[int, tuple[float, float]] | None,
) -> np.ndarray[Any, Any]:
    """Scale a whole ``(Y, X, C)`` field to float32 ``[0, 1]`` per channel.

    Uses the per-channel ``intensities`` window when available (matching the
    napari viewer's contrast limits), else falls back to the whole-field max.
    Lifted verbatim from the gallery's ``_scale_full_image`` so welldata
    display contrast is unchanged.
    """
    intensities = intensities or {}
    scaled = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[-1]):
        img_slice = image[..., i]
        if i in intensities:
            min_i, max_i = intensities[i]
            range_i = max(max_i - min_i, 1)
            scaled[..., i] = np.clip((img_slice - min_i) / range_i, 0, 1)
        else:
            max_val = np.max(img_slice)
            scaled[..., i] = img_slice / max_val if max_val > 0 else img_slice
    return scaled


class WelldataSource:
    """In-memory per-field source backed by ``omero_data.images`` / ``labels``.

    Replicates the gallery's legacy in-memory path: scale the whole field once
    (cached per image), select the timepoint, then crop. Label channel 0 is
    nucleus, 1 is cell.
    """

    def __init__(self, omero_data: OmeroData) -> None:
        self._od = omero_data
        self._scaled_field_cache: dict[int, np.ndarray[Any, Any]] = {}
        self._label_field_cache: dict[
            tuple[int, str], np.ndarray[Any, Any]
        ] = {}

    def _field(
        self, image_id: int, t: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        try:
            index = self._od.image_ids.index(image_id)
        except ValueError as exc:
            raise CropSourceError(
                f"Image {image_id} not loaded in OmeroData"
            ) from exc
        images = self._od.images[index]
        labels = self._od.labels[index]
        # Collapse a leading time axis when present (5D ZTYXC-ish or 4D TYXC).
        if images.ndim >= 4:
            images = images[t, ...]
            labels = labels[t, ...]
        return images, labels

    def fetch(
        self,
        image_id: int,
        centroid: tuple[float, float],
        size: int,
        t: int,
        mask_name: Literal["nuclei", "cells"],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if image_id not in self._scaled_field_cache:
            images, labels = self._field(image_id, t)
            self._scaled_field_cache[image_id] = _scale_field(
                images, self._od.intensities
            )
            if labels.ndim == 3 and labels.shape[-1] >= 2:
                nuc = labels[..., 0]
                cell = labels[..., 1]
            else:
                nuc = np.squeeze(labels)
                cell = nuc
            self._label_field_cache[(image_id, "nuclei")] = nuc
            self._label_field_cache[(image_id, "cells")] = cell

        scaled = self._scaled_field_cache[image_id]
        label_field = self._label_field_cache[(image_id, mask_name)]
        row, col = int(centroid[0]), int(centroid[1])
        crop, label_crop = crop_region(scaled, label_field, row, col, size)
        if crop.shape[0] != size or crop.shape[1] != size:
            crop, label_crop = pad_region(crop, label_crop, size)
        return crop, label_crop


class ZarrSource:
    """Stitched-canvas source backed by the OME-Zarr plate cache.

    Per-crop normalisation: per-channel window from ``intensities`` when
    present (viewer contrast), else per-crop 99.9th percentile. This union of
    the gallery and direct-load zarr behaviours preserves both — the gallery
    only ever uses this source when ``intensities`` is set, so it always takes
    the window branch; headless direct-load callers take the percentile branch.
    """

    def __init__(
        self,
        plate_id: int,
        well_id: str,
        intensities: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        # Import here so the core module doesn't hard-depend on the zarr stack.
        from omero_screen_napari.zarr_cache import (
            cached_wells,
            fetch_crop,
            fetch_label_crop,
            resolve_to_zarr,
        )

        if resolve_to_zarr(plate_id) is None:
            raise CropSourceError(f"No zarr cache for plate {plate_id}")
        if well_id not in cached_wells(plate_id):
            raise CropSourceError(
                f"Well {well_id} not built in zarr cache for plate {plate_id}"
            )
        self._plate_id = plate_id
        self._well_id = well_id
        self._intensities = intensities or {}
        self._fetch_crop = fetch_crop
        self._fetch_label_crop = fetch_label_crop

    def fetch(
        self,
        image_id: int,
        centroid: tuple[float, float],
        size: int,
        t: int,
        mask_name: Literal["nuclei", "cells"],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        try:
            crop_cyx = self._fetch_crop(
                self._plate_id,
                self._well_id,
                label=0,
                centroid=centroid,
                size=size,
                t=t,
            )
            label_crop = self._fetch_label_crop(
                self._plate_id,
                self._well_id,
                centroid=centroid,
                size=size,
                t=t,
                mask_name=mask_name,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise CropSourceError(
                f"Zarr fetch failed for plate {self._plate_id} "
                f"well {self._well_id}: {exc}"
            ) from exc

        crop_yxc = np.transpose(crop_cyx, (1, 2, 0)).astype(
            np.float32, copy=False
        )
        for ch in range(crop_yxc.shape[-1]):
            ch_slice = crop_yxc[..., ch]
            window = self._intensities.get(ch)
            if window is not None:
                lo, hi = float(window[0]), float(window[1])
                rng = max(hi - lo, 1.0)
                crop_yxc[..., ch] = np.clip((ch_slice - lo) / rng, 0.0, 1.0)
            else:
                p_high = float(np.percentile(ch_slice, 99.9))
                if p_high > 0:
                    crop_yxc[..., ch] = np.clip(ch_slice / p_high, 0.0, 1.0)
        return crop_yxc, label_crop


class OmeroSource:
    """Per-field source that downloads pixels + masks from OMERO on demand.

    Replicates the legacy OMERO fallback: download each field (cached),
    flatfield-correct, per-channel 99.9th-percentile scale the whole field,
    then crop. Segmentation masks come from the plate's ``{image_id}_segmentation``
    image in the OMERO dataset named after the plate.
    """

    def __init__(
        self,
        conn: BlitzGateway,
        plate_id: int,
        well: Any,
        image_id_by_index: dict[int, int],
    ) -> None:
        from omero_screen_napari.omero_image import get_image_timepoint

        self._conn = conn
        self._plate_id = plate_id
        self._well = well
        self._get_image_timepoint = get_image_timepoint
        # image_id → well-sample index, for fetching the right field.
        self._index_by_image_id = {v: k for k, v in image_id_by_index.items()}
        self._scaled_field_cache: dict[int, np.ndarray[Any, Any]] = {}
        self._mask_field_cache: dict[int, np.ndarray[Any, Any]] = {}
        ff_ok, self._flatfield = self._load_flatfield()
        self._flatfield = self._flatfield if ff_ok else None

    def _dataset(self) -> Any:
        for ds in self._conn.getObjects("Dataset"):
            if ds.getName() == str(self._plate_id):
                return ds
        return None

    def _load_flatfield(self) -> tuple[bool, Any]:
        try:
            dataset = self._dataset()
            if not dataset:
                logger.warning(
                    f"Dataset for plate {self._plate_id:d} not found"
                )
                return False, None
            ff_image = next(
                (
                    img
                    for img in dataset.listChildren()
                    if "flatfield" in img.getName().lower()
                ),
                None,
            )
            if not ff_image:
                logger.warning("No flatfield correction image found")
                return False, None
            cached = self._get_image_timepoint(
                self._conn, ff_image.getId(), 0, tag=self._plate_id
            )
            flatfield = cached.squeeze(axis=0).astype(np.float32)
            return True, flatfield
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to load flatfield correction: {exc}")
            return False, None

    def _load_masks(self, image_id: int, t: int) -> np.ndarray[Any, Any]:
        dataset = self._dataset()
        if not dataset:
            raise CropSourceError(
                f"Dataset for plate {self._plate_id} not found"
            )
        seg_name = f"{image_id}_segmentation"
        seg_image = next(
            (
                img
                for img in dataset.listChildren()
                if img.getName() == seg_name
            ),
            None,
        )
        if not seg_image:
            raise CropSourceError(
                f"Segmentation masks not found for image {image_id}"
            )
        cached = self._get_image_timepoint(
            self._conn, seg_image.getId(), t, tag=self._plate_id
        )
        return cached.squeeze(axis=0).astype(np.int32)  # YXC: [nucleus, cell]

    def _ensure_field(self, image_id: int, t: int) -> None:
        if image_id in self._scaled_field_cache:
            return
        cached = self._get_image_timepoint(
            self._conn, image_id, t, tag=self._plate_id
        )
        image_array = cached.squeeze(axis=0).astype(np.float32)  # YXC
        if self._flatfield is not None:
            image_array = image_array / self._flatfield
        for ch in range(image_array.shape[-1]):
            ch_slice = image_array[..., ch]
            p_high = float(np.percentile(ch_slice, 99.9))
            if p_high > 0:
                image_array[..., ch] = np.clip(ch_slice / p_high, 0.0, 1.0)
        self._scaled_field_cache[image_id] = image_array
        self._mask_field_cache[image_id] = self._load_masks(image_id, t)

    def fetch(
        self,
        image_id: int,
        centroid: tuple[float, float],
        size: int,
        t: int,
        mask_name: Literal["nuclei", "cells"],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        self._ensure_field(image_id, t)
        scaled = self._scaled_field_cache[image_id]
        masks = self._mask_field_cache[image_id]
        if masks.ndim == 3 and masks.shape[-1] >= 2:
            channel_idx = 0 if mask_name == "nuclei" else 1
            label_field = masks[..., channel_idx]
        else:
            label_field = np.squeeze(masks)
        row, col = int(centroid[0]), int(centroid[1])
        crop, label_crop = crop_region(scaled, label_field, row, col, size)
        if crop.shape[0] != size or crop.shape[1] != size:
            crop, label_crop = pad_region(crop, label_crop, size)
        return crop, label_crop
