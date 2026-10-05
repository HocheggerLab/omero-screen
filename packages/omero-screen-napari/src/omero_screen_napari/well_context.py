"""Populate an :class:`OmeroData` for a plate's wells without napari or Qt.

The napari widgets fill the ``omero_data`` singleton as a side effect of
loading a plate into the viewer: the zarr path through
``zarr_cache.display.load_plate_to_viewer`` and the per-field path through a
Qt worker around :func:`plate_cache.load_from_cache`. Headless callers (the
``omero-screen-images`` CLI, notebooks) need the same state without a viewer,
so this module runs the same two loaders directly:

* **zarr cache present** — the plate was stitched and cached. Only metadata is
  loaded; crops are read lazily from the stitched canvas on disk
  (``crop_pipeline.ZarrSource``).
* **no zarr cache** — the per-field plate's images and masks for the requested
  wells are loaded into memory via the plate disk cache, downloading from
  OMERO on a miss (``crop_pipeline.WelldataSource``).

Load one well at a time for per-field plates: every field of every requested
well is held in RAM as float32.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from omero_screen_napari.omero_data import OmeroConnection, OmeroData

WellSource = Literal["zarr", "fields"]


class WellContextError(ValueError):
    """The requested plate or wells cannot be loaded headlessly."""


def well_source(plate_id: int) -> WellSource:
    """Return where :func:`load_well_context` reads ``plate_id`` from."""
    from omero_screen_napari.zarr_cache import plate_zarr_path

    return "zarr" if plate_zarr_path(plate_id).exists() else "fields"


def load_well_context(
    plate_id: int,
    wells: list[str],
    *,
    omero_data: OmeroData | None = None,
    timepoint: int | None = None,
    connection: OmeroConnection | None = None,
) -> OmeroData:
    """Populate ``omero_data`` with the plate context for ``wells``.

    Args:
        plate_id: OMERO plate ID.
        wells: Well labels, e.g. ``["E2", "G5"]``.
        omero_data: Instance to populate. A new one is created if omitted;
            pass the same instance to load further wells of the same plate.
        timepoint: 0-based timepoint to load for per-field plates. ``None``
            loads every timepoint. Zarr plates read crops lazily at any
            timepoint and use this one only for the display limits.
        connection: OMERO connection for per-field loads. One is created
            from the environment if omitted. It is closed after the load
            (``load_from_cache`` closes it) and reopens on next use.

    Returns:
        The populated ``omero_data``.

    Raises:
        WellContextError: No wells given, wells missing from the zarr
            cache, or a stitched plate without a zarr cache.
    """
    from omero_screen_napari.omero_data import OmeroConnection, OmeroData

    if not wells:
        raise WellContextError("No wells requested")
    if omero_data is None:
        omero_data = OmeroData()

    if well_source(plate_id) == "zarr":
        _load_from_zarr(omero_data, plate_id, wells, timepoint)
    else:
        _load_from_fields(
            omero_data,
            plate_id,
            wells,
            timepoint,
            connection or OmeroConnection(),
        )
    return omero_data


def _load_from_zarr(
    omero_data: OmeroData,
    plate_id: int,
    wells: list[str],
    timepoint: int | None,
) -> None:
    """Fill metadata from the zarr cache, as the zarr viewer load does."""
    from omero_screen_napari.zarr_cache import (
        cached_wells,
        plate_info,
        read_well,
    )
    from omero_screen_napari.zarr_cache.display import populate_omero_data

    available = cached_wells(plate_id)
    missing = [w for w in wells if w not in available]
    if missing:
        raise WellContextError(
            f"Well(s) {', '.join(missing)} are not built in the zarr cache "
            f"for plate {plate_id}. Cached wells: "
            f"{', '.join(available) or 'none'}. Rebuild the cache with "
            f"those wells first."
        )
    info = plate_info(plate_id)
    populate_omero_data(omero_data, plate_id, ", ".join(wells), wells, info)
    # The viewer takes its contrast from the first loaded well, so a
    # gallery would change with the well list. Pool the requested wells
    # instead: every well of one call is scaled identically, and the same
    # call always gives the same limits.
    pooled = pooled_intensities(
        [read_well(plate_id, w) for w in wells], timepoint or 0
    )
    if pooled:
        omero_data.intensities = pooled


# Pyramid level sampled for display limits: level 1 is 2x downsampled, so
# percentiles match level 0 closely at a quarter of the read.
_LIMITS_LEVEL = 1
_SAMPLES_PER_WELL = 1_000_000


def pooled_intensities(
    wells_data: list[dict[str, Any]], timepoint: int = 0
) -> dict[int, tuple[int, int]]:
    """Per-channel 0.1/99.9-percentile limits pooled over several wells.

    Same percentiles as the napari layers' initial contrast
    (``display._channel_contrast``), over a pixel sample from each well's
    canvas at ``timepoint`` (clamped to the last one). Each well is sampled
    with its own fixed seed and the percentiles are taken over the pooled
    samples, so the result does not depend on the order of the wells.

    Args:
        wells_data: ``zarr_cache.read_well`` results.
        timepoint: 0-based timepoint to sample.

    Returns:
        ``{channel_index: (lo, hi)}``; empty when there is nothing to sample.
    """
    from omero_screen_napari.zarr_cache.display import _UINT16_MAX

    levels = [
        w["image"][min(_LIMITS_LEVEL, len(w["image"]) - 1)] for w in wells_data
    ]
    if not levels:
        return {}
    out: dict[int, tuple[int, int]] = {}
    for c in range(levels[0].shape[1]):
        samples = []
        for level in levels:
            t = min(timepoint, level.shape[0] - 1)
            flat = np.asarray(level[t, c]).ravel()
            if flat.size > _SAMPLES_PER_WELL:
                rng = np.random.default_rng(0)
                flat = flat[
                    rng.choice(flat.size, _SAMPLES_PER_WELL, replace=False)
                ]
            samples.append(flat)
        lo, hi = np.percentile(np.concatenate(samples), [0.1, 99.9])
        # A flat channel: show it against the full span, as the viewer does.
        out[c] = (int(lo), int(hi)) if hi > lo else (0, _UINT16_MAX)
    return out


def _load_from_fields(
    omero_data: OmeroData,
    plate_id: int,
    wells: list[str],
    timepoint: int | None,
    connection: OmeroConnection,
) -> None:
    """Load the wells' fields and masks into memory, as the welldata widget does."""
    from omero_screen_napari.plate_cache import (
        get_plate_metadata,
        load_from_cache,
    )

    meta = get_plate_metadata(connection, plate_id)
    if meta.get("label_stitched_mode"):
        # Same guard as the welldata widget: re-stitching a stitched plate
        # in memory can exhaust RAM, and the gallery needs the zarr canvas.
        raise WellContextError(
            f"Plate {plate_id} was processed in stitched mode but has no "
            f"zarr cache. Build it first (Plate Info → 'Cache Plate')."
        )
    time = "All" if timepoint is None else str(timepoint + 1)
    logger.info(
        f"Loading plate {plate_id:d} wells {wells} from fields (time={time})"
    )
    for _done, _total in load_from_cache(
        connection,
        omero_data,
        plate_id,
        ", ".join(wells),
        "All",
        threading.Event(),
        time=time,
    ):
        pass
