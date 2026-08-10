"""Hand the contents of a plate.zarr cache to a napari viewer.

Load-side counterpart to :mod:`omero_screen_napari.zarr_cache.builder`.
Bypasses the per-field ``plate_cache`` path entirely: data is already
stitched on disk, so we hand zarr arrays to napari layers directly.
Everything is lazy via zarr — only the chunks needed for the current
viewport at the current pyramid level are read.

Multi-well UX: stack all selected wells along a new leading axis. napari
exposes the well axis as a slider, so the user flips through wells one
at a time at full resolution and zooms freely. A HUD overlay shows the
current well's metadata (cell line / condition / timepoint).
"""

from __future__ import annotations

import re
from typing import Any

import dask.array as da
import numpy as np
from loguru import logger
from omero_screen.config import getenv_as_int

from omero_screen_napari.zarr_cache.reader import (
    cached_wells,
    plate_info,
    read_well,
)

# Opportunistic dask cache: caches *decoded* array results (not just raw
# chunk bytes like the reader's LRUStoreCache), so replaying a timelapse,
# scrubbing back, or re-zooming a region already viewed hits RAM instead
# of re-running the from_zarr → slice → stack graph. Registered once per
# process; size override via ``OMERO_SCREEN_DASK_CACHE_MB``.
_dask_cache_registered = False


def _ensure_dask_cache() -> None:
    """Register a process-wide dask opportunistic cache (idempotent)."""
    global _dask_cache_registered
    if _dask_cache_registered:
        return
    try:
        from dask.cache import Cache

        cache_bytes = getenv_as_int("OMERO_SCREEN_DASK_CACHE_MB", 2048) * 2**20
        Cache(cache_bytes).register()  # type: ignore[no-untyped-call]
        _dask_cache_registered = True
        logger.info(
            f"Registered dask opportunistic cache ({cache_bytes >> 20:d} MB)"
        )
    except Exception:  # pragma: no cover - cache is a perf optimisation only
        logger.opt(exception=True).warning(
            "Could not register dask cache; playback may be laggier",
        )


_async_enabled = False


def _ensure_async_slicing() -> None:
    """Turn on napari async slicing (idempotent).

    Slice loading is synchronous on the Qt GUI thread by default, so any
    frame that takes longer than the play interval to read+render — a
    full-res (level 0) tile when zoomed in, or the two uint32 label
    layers — freezes the viewer until it finishes, which reads as the
    timelapse stalling then fast-forwarding. Async moves the read off the
    GUI thread: the viewer keeps the previous frame on screen until the
    new one is ready, so interaction stays responsive. Disable with
    ``OMERO_SCREEN_NAPARI_ASYNC=0``.
    """
    global _async_enabled
    if _async_enabled:
        return
    try:
        from napari.settings import get_settings

        get_settings().experimental.async_ = bool(
            getenv_as_int("OMERO_SCREEN_NAPARI_ASYNC", 1)
        )
        _async_enabled = True
        logger.info(
            f"napari async slicing = {get_settings().experimental.async_}"
        )
    except Exception:  # pragma: no cover - perf optimisation only
        logger.opt(exception=True).warning(
            "Could not toggle napari async slicing"
        )


# Channel colormap rotation — first four match the omero-screen palette.
_CHANNEL_COLORMAPS = ["blue", "green", "red", "magenta", "yellow", "cyan"]


_WELL_RE = re.compile(r"^[A-Za-z]\d+$")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def load_plate_to_viewer(
    viewer: Any,
    plate_id: int,
    well_pos_input: str = "All",
) -> list[str]:
    """Open one or more wells from the zarr cache as napari layers.

    Single well → one layer per channel plus nuclei/cells labels.
    Multiple wells → wells stacked along a new leading axis, slider
    scrolls between them. A HUD overlay shows per-well metadata when
    the slider moves.

    Returns:
        The list of well IDs actually loaded.
    """
    _ensure_dask_cache()
    _ensure_async_slicing()
    info = plate_info(plate_id)
    available = cached_wells(plate_id)
    target_wells = _resolve_well_list(well_pos_input, available)
    if not target_wells:
        logger.warning(
            f"No wells matched '{well_pos_input}' in zarr cache for plate {plate_id:d} (available: {available})"
        )
        return []

    # Drop any layers already in the viewer so reloads don't accumulate.
    while len(viewer.layers) > 0:
        viewer.layers.pop(0)

    wells_data = [read_well(plate_id, w) for w in target_wells]
    channel_names = info["channel_names"]
    px = info["pixel_size_um"] or 1.0
    multi = len(target_wells) > 1

    _add_image_layers(viewer, wells_data, channel_names, px, multi)
    _add_label_layers(viewer, wells_data, "nuclei", px, multi)
    _add_label_layers(viewer, wells_data, "cells", px, multi)
    _add_missing_region_layer(viewer, wells_data, target_wells, px, multi)

    if multi:
        # Axis order is (well, T, Y, X). Naming the slider axis improves
        # napari's dim panel and lets the overlay hook key off step[0].
        viewer.dims.axis_labels = ("well", "t", "y", "x")
        _hook_well_overlay(viewer, target_wells, info, plate_id)
    else:
        viewer.dims.axis_labels = ("t", "y", "x")
        _set_overlay(viewer, target_wells[0], info, plate_id)

    # Populate the OmeroData singleton so downstream widgets (gallery,
    # training, classifier) see the same plate/well context the legacy
    # diskcache path would have produced. We do NOT populate
    # ``omero_data.images`` / ``omero_data.labels`` — those would defeat
    # zarr's lazy chunk loading. Widgets that need pixel arrays use the
    # zarr fetch API directly (gallery does this via _try_zarr_crop_path).
    _populate_singleton(
        plate_id, well_pos_input, target_wells, info, wells_data
    )

    viewer.reset_view()
    logger.info(
        f"Loaded {len(target_wells):d} well(s) from zarr cache for plate {plate_id:d}"
    )
    return target_wells


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _resolve_well_list(well_pos_input: str, available: list[str]) -> list[str]:
    """Parse the user's well input against the wells actually on disk.

    ``"All"`` (case-insensitive) returns every cached well. A comma list
    keeps only the entries that are present on disk; missing wells log
    a warning. Order: sorted for All, user-supplied order for explicit
    lists (sorted to keep the slider monotonic).
    """
    raw = well_pos_input.strip()
    if raw.lower() == "all" or raw == "":
        return sorted(available)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    valid = [item for item in items if _WELL_RE.match(item)]
    if len(valid) != len(items):
        bad = set(items) - set(valid)
        logger.warning(f"Ignoring malformed well labels: {sorted(bad)}")
    available_set = set(available)
    resolved = sorted(w for w in valid if w in available_set)
    missing = [w for w in valid if w not in available_set]
    if missing:
        logger.warning(
            f"Wells not present in zarr cache: {missing} (available: {available})"
        )
    return resolved


def _stack_pyramid_per_channel(
    wells_data: list[dict[str, Any]],
    channel_index: int,
    multi: bool,
) -> list[Any]:
    """Build a multiscale list for one channel, stacked across wells.

    Each pyramid level becomes a dask array of shape
    ``(N_wells, T, Y, X)`` for ``multi=True`` or ``(T, Y, X)`` for a
    single well. Dask wraps the underlying zarr arrays so reads stay
    lazy — napari only pulls the chunks it renders.
    """
    n_levels = len(wells_data[0]["image"])
    pyramid: list[Any] = []
    for lvl in range(n_levels):
        per_well = [
            da.from_zarr(wd["image"][lvl])[:, channel_index]  # type: ignore[no-untyped-call]
            for wd in wells_data
        ]
        pyramid.append(da.stack(per_well, axis=0) if multi else per_well[0])  # type: ignore[no-untyped-call]
    return pyramid


def _stack_pyramid_labels(
    wells_data: list[dict[str, Any]],
    key: str,
    multi: bool,
) -> list[Any] | None:
    """Same as :func:`_stack_pyramid_per_channel` but for label groups.

    Returns ``None`` if no well in the set has the requested label
    group (e.g. ``"cells"`` may be absent on nucleus-only stores).
    """
    if any(wd[key] is None for wd in wells_data):
        return None
    n_levels = len(wells_data[0][key])
    pyramid: list[Any] = []
    for lvl in range(n_levels):
        per_well = [da.from_zarr(wd[key][lvl]) for wd in wells_data]  # type: ignore[no-untyped-call]
        pyramid.append(da.stack(per_well, axis=0) if multi else per_well[0])  # type: ignore[no-untyped-call]
    return pyramid


def _add_image_layers(
    viewer: Any,
    wells_data: list[dict[str, Any]],
    channel_names: list[str],
    pixel_size_um: float,
    multi: bool,
) -> None:
    """Add one layer per channel, with all wells stacked on axis 0 if multi."""
    n_channels = wells_data[0]["image"][0].shape[1]
    # Estimate contrast from the first well's first timepoint per channel.
    for c in range(n_channels):
        ch_name = channel_names[c] if c < len(channel_names) else f"ch{c}"
        pyramid = _stack_pyramid_per_channel(wells_data, c, multi)
        # Scale matches the array axes:
        #   multi: (well, T, Y, X) → (1, 1, px, px)
        #   single: (T, Y, X) → (1, px, px)
        scale = (
            (1.0, 1.0, pixel_size_um, pixel_size_um)
            if multi
            else (1.0, pixel_size_um, pixel_size_um)
        )
        # Sample one Y/X slab of the first well to set sane contrast.
        sample = np.asarray(wells_data[0]["image"][0][0, c])
        viewer.add_image(
            pyramid,
            name=ch_name,
            scale=scale,
            colormap=_CHANNEL_COLORMAPS[c % len(_CHANNEL_COLORMAPS)],
            blending="additive",
            contrast_limits=_channel_contrast(sample),
        )


def _add_label_layers(
    viewer: Any,
    wells_data: list[dict[str, Any]],
    key: str,
    pixel_size_um: float,
    multi: bool,
) -> None:
    """Add a Labels layer for ``key`` (``"nuclei"`` or ``"cells"``)."""
    pyramid = _stack_pyramid_labels(wells_data, key, multi)
    if pyramid is None:
        return
    scale = (
        (1.0, 1.0, pixel_size_um, pixel_size_um)
        if multi
        else (1.0, pixel_size_um, pixel_size_um)
    )
    viewer.add_labels(pyramid, name=key, scale=scale)


def _add_missing_region_layer(
    viewer: Any,
    wells_data: list[dict[str, Any]],
    well_ids: list[str],
    pixel_size_um: float,
    multi: bool,
) -> None:
    """Outline canvas regions where no field was acquired.

    A failed acquisition leaves a tile-sized blank rectangle in the
    stitched canvas with no labels in it. Unmarked, that is
    indistinguishable from a rendering or cache fault, so it gets a
    dashed outline and a caption naming the well.

    No layer is added when every well is complete — which is the common
    case — so this stays invisible unless it has something to say.
    """
    shapes: list[Any] = []
    captions: list[str] = []
    for index, (data, well_id) in enumerate(
        zip(wells_data, well_ids, strict=True)
    ):
        for y0, x0, y1, x1 in data.get("missing_regions", ()):
            corners = [(y0, x0), (y0, x1), (y1, x1), (y1, x0)]
            # Shapes must match the image layers' dimensionality: the
            # leading axes are the well slider (when stacked) and T.
            prefix = (index, 0) if multi else (0,)
            shapes.append([(*prefix, y, x) for y, x in corners])
            captions.append(f"{well_id}: no acquisition")

    if not shapes:
        return

    scale = (
        (1.0, 1.0, pixel_size_um, pixel_size_um)
        if multi
        else (1.0, pixel_size_um, pixel_size_um)
    )
    viewer.add_shapes(
        shapes,
        name="missing fields",
        shape_type="rectangle",
        edge_color="red",
        edge_width=8,
        face_color="transparent",
        opacity=0.9,
        scale=scale,
        text={
            "string": captions,
            "size": 10,
            "color": "red",
            "anchor": "upper_left",
        },
    )
    logger.info(
        f"Marked {len(shapes):d} unimaged region(s) across "
        f"{len({c.split(':')[0] for c in captions}):d} well(s)"
    )


def _intensities_from_canvas(
    wells_data: list[dict[str, Any]],
) -> dict[int, tuple[int, int]] | None:
    """Per-channel (lo, hi) from the first well's first timepoint.

    Uses the same 0.1/99.9-percentile heuristic as the napari image
    layers, so the gallery's per-channel scaling matches the viewer's
    contrast limits. Returns ``None`` on any error so the caller can
    fall back to CellView statistics.
    """
    try:
        level0 = wells_data[0]["image"][0]  # zarr (T, C, Y, X)
        if level0.ndim < 4:
            return None
        n_channels = level0.shape[1]
        out: dict[int, tuple[int, int]] = {}
        for c in range(n_channels):
            sample = np.asarray(level0[0, c])
            lo, hi = _channel_contrast(sample)
            out[c] = (int(lo), int(hi))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"Failed to derive intensities from zarr canvas: {exc} (falling back to CellView)"
        )
        return None


def _channel_contrast(level0_yx: np.ndarray[Any, Any]) -> tuple[int, int]:
    """0.1/99.9-percentile contrast for a sample slice."""
    if level0_yx.size > 1_000_000:
        flat = level0_yx.ravel()
        idx = np.random.default_rng(0).choice(
            flat.size, 1_000_000, replace=False
        )
        flat = flat[idx]
    else:
        flat = level0_yx.ravel()
    lo, hi = np.percentile(flat, [0.1, 99.9])
    return int(lo), int(hi)


# ----------------------------------------------------------------------
# Well metadata HUD (uses napari's text_overlay — no Shapes/Points)
# ----------------------------------------------------------------------


def _format_well_caption(
    well: str, well_meta: dict[str, dict[str, str]]
) -> str:
    meta = well_meta.get(well, {})
    bits = [well]
    for key in ("cell_line", "condition", "timepoint"):
        if val := meta.get(key):
            bits.append(str(val))
    return " | ".join(bits)


def _set_overlay(
    viewer: Any, well: str, info: dict[str, Any], plate_id: int
) -> None:
    """Set the static text overlay for a single-well load."""
    caption = _format_well_caption(well, info.get("well_metadata", {}))
    viewer.text_overlay.text = f"Plate {plate_id} — {caption}"
    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 18
    viewer.text_overlay.color = "yellow"


def _hook_well_overlay(
    viewer: Any,
    wells: list[str],
    info: dict[str, Any],
    plate_id: int,
) -> None:
    """Wire the text overlay to update as the user moves the well slider."""
    well_meta = info.get("well_metadata", {})
    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 18
    viewer.text_overlay.color = "yellow"

    def _update(_event: Any = None) -> None:
        idx = int(viewer.dims.current_step[0])
        idx = max(0, min(idx, len(wells) - 1))
        caption = _format_well_caption(wells[idx], well_meta)
        viewer.text_overlay.text = f"Plate {plate_id} — {caption}"

    viewer.dims.events.current_step.connect(_update)
    _update()


# ----------------------------------------------------------------------
# OmeroData singleton population
# ----------------------------------------------------------------------


def _populate_singleton(
    plate_id: int,
    well_pos_input: str,
    target_wells: list[str],
    info: dict[str, Any],
    wells_data: list[dict[str, Any]] | None = None,
) -> None:
    """Set the minimum fields downstream widgets read from the singleton.

    Mirrors :func:`omero_screen_napari.plate_cache.load_from_cache`'s side
    effects on the ``omero_data`` instance — minus the heavy pixel arrays
    (``images`` / ``labels``), which we keep on disk so zarr reads stay
    lazy.

    Specifically populates:

    * ``plate_id`` — used by every widget for filesystem and DB lookups.
    * ``plate_data`` — Polars ``LazyFrame`` from CellView. Required by
      the gallery widget's ``_get_well_data``.
    * ``well_pos_list`` — wells currently loaded into the viewer.
    * ``image_input`` — the user's well-input string (kept for the
      training widget's NPY filename convention).
    * ``image_ids`` — synthetic per-well IDs from CellView measurements,
      so the gallery's ``_prepare_data_for_cropping`` filter recognises
      "loaded" rows.
    * ``intensities`` — per-channel contrast windows from CellView, so
      :meth:`CroppedImageParser._try_zarr_crop_path` matches the legacy
      gallery's scaling.
    * ``channel_data`` / ``pixel_size`` / ``plate_name`` — referenced by
      classifier metadata and a few UI labels.
    """
    # Import here to avoid a hard dependency / circular import: the
    # singleton lives in omero_data_singleton, which itself imports
    # omero_data which depends on a few of our own modules.
    from omero_screen_napari.omero_data_singleton import omero_data
    from omero_screen_napari.plate_cache import (
        _load_plate_data_from_cellview,
        _parse_intensities_from_cellview,
    )

    # Channel data as the legacy code expects: {name: index_str}.
    channel_data = {
        name: str(i) for i, name in enumerate(info.get("channel_names", []))
    }

    omero_data.plate_id = plate_id
    omero_data.plate_name = info.get("plate_name", "")
    omero_data.well_pos_list = list(target_wells)
    omero_data.image_input = well_pos_input
    omero_data.channel_data = channel_data
    px = info.get("pixel_size_um")
    omero_data.pixel_size = (px, px) if px is not None else None

    # Per-well annotations baked into the zarr at build time (Stage 2).
    # The gallery's _parse_metadata reads well_metadata_list[i] for the
    # currently displayed well, so the order must match well_pos_list.
    well_meta_attrs = info.get("well_metadata", {}) or {}
    omero_data.well_metadata_list = [
        dict(well_meta_attrs.get(w, {})) for w in target_wells
    ]
    # well_id_list is referenced by some downstream widgets; the zarr
    # store doesn't keep OMERO well-object IDs, so we fill with the
    # plate's row/column-derived index (stable per well).
    omero_data.well_id_list = list(range(len(target_wells)))

    plate_data = _load_plate_data_from_cellview(plate_id)
    omero_data.plate_data = plate_data

    # image_ids: pick the synthetic IDs for the loaded wells from
    # CellView measurements. Stitched-mode plates have one synthetic
    # image_id per well. Empty LazyFrame → empty list (gallery will
    # surface a clean "plate not in CellView" error).
    omero_data.image_ids = _resolve_image_ids(plate_data, target_wells)

    # Prefer canvas-percentile intensities matching napari's contrast
    # limits over CellView mean(intensity_max), which is wildly off for
    # live-cell channels (saturates tub-GFP to uniform green, crushes
    # DAPI to near-zero). Falls back to CellView if no canvas sample is
    # available.
    if wells_data:
        canvas_intensities = _intensities_from_canvas(wells_data)
    else:
        canvas_intensities = None
    omero_data.intensities = (
        canvas_intensities
        if canvas_intensities is not None
        else _parse_intensities_from_cellview(plate_id, channel_data)
    )

    logger.info(
        f"Populated omero_data singleton from zarr load: plate={plate_id:d} wells={target_wells} image_ids={omero_data.image_ids} channels={list(channel_data)}"
    )


def _resolve_image_ids(plate_data: Any, wells: list[str]) -> list[int]:
    """Return the distinct image_ids in CellView for the loaded wells.

    Returns an empty list if ``plate_data`` is empty (plate not in
    CellView) or doesn't expose ``well`` / ``image_id`` columns.
    """
    import polars as pl

    if plate_data is None:
        return []
    try:
        schema_names = plate_data.collect_schema().names()
    except Exception:  # noqa: BLE001
        return []
    if (
        not schema_names
        or "well" not in schema_names
        or "image_id" not in schema_names
    ):
        return []
    try:
        df = (
            plate_data.filter(pl.col("well").is_in(wells))
            .select("image_id")
            .unique()
            .collect()
        )
    except Exception:  # noqa: BLE001
        return []
    return sorted(int(x) for x in df["image_id"].to_list())
