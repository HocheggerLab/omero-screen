"""Build a plate.zarr cache from a stitched OMERO plate.

The builder is the heavy module that bridges OMERO data → stitched arrays →
``PlateZarrWriter``. It is the only entry point that talks to OMERO. The
napari widget invokes :func:`build_plate_zarr` as a thread worker; headless
callers (tests, scripts) call it directly.

Build flow per well:

1. Load N fields' raw pixel arrays (T, C, Y, X) and canvas offsets.
2. Apply flatfield per field, then stitch all channels in one call.
3. Fetch per-field stitched-mode segmentation masks via OMERO map
   annotations (``Stitched_Segmentation_Mask``) on the source images.
4. Recompose those per-field tiles into a single canvas via
   :func:`recompose_tiles` (lossless: the masks were produced by a
   canvas-wide segmentation, then split for upload, so label IDs are
   globally unique).
5. Hand the result to :class:`PlateZarrWriter` for one well's worth of
   NGFF output.

Yields each completed well ID so a worker thread can stream progress to
the UI.
"""

from __future__ import annotations

# dask ships only partial type info, so its array/delayed calls read as
# "untyped" under strict mypy. This module is the dask bridge; every internal
# call is typed, so disabling just this one code here is precise enough.
# mypy: disable-error-code="no-untyped-call"
import contextlib
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
from dask.delayed import delayed
from loguru import logger
from omero.gateway import BlitzGateway, WellWrapper
from omero_utils.attachments import get_file_attachments, parse_csv_data
from omero_utils.images import (
    fetch_stitched_field_masks_trange,
    resolve_stitched_mask_ids,
)
from omero_utils.message import PlateDataError
from omero_utils.stitching import (
    get_overlap,
    missing_field_boxes,
    recompose_tiles,
    stitch_from_offsets,
    stitch_into_canvas,
)

from omero_screen_napari.omero_data import OmeroConnection, get_dataset_id
from omero_screen_napari.omero_image import get_image
from omero_screen_napari.plate_cache import (
    get_plate_metadata,
    get_well_data,
    is_empty_well,
)
from omero_screen_napari.zarr_cache.alignment import AlignmentError
from omero_screen_napari.zarr_cache.eviction import (
    enforce_size_cap,
    estimate_plate_size_bytes,
    transient_pin,
)
from omero_screen_napari.zarr_cache.paths import plate_zarr_path
from omero_screen_napari.zarr_cache.registry import ZarrPlateEntry, upsert
from omero_screen_napari.zarr_cache.rounds import (
    RoundGroup,
    build_channel_plan,
)
from omero_screen_napari.zarr_cache.writer import PlateZarrWriter

# Timepoints stitched per lazy dask block. The build never holds more than a
# few blocks at once (bounded by the dask scheduler), so this is the
# memory↔#-OMERO-calls lever. Conservative default keeps long live-cell wells
# within a 16 GB Mac; override with OMERO_SCREEN_CACHE_BLOCK. (Auto-sizing from
# the RAM budget is the planned optimisation.)
_CACHE_BLOCK_T = int(os.getenv("OMERO_SCREEN_CACHE_BLOCK", "4"))
# Concurrent dask blocks during the streamed write. Caps peak memory at
# roughly ``workers × block_t`` frames (plus pyramids), so the two knobs
# together bound RAM. Threads (not processes): the work is blocking Ice I/O,
# which releases the GIL. Override with OMERO_SCREEN_CACHE_WORKERS.
_CACHE_DASK_WORKERS = int(os.getenv("OMERO_SCREEN_CACHE_WORKERS", "2"))


# ----------------------------------------------------------------------
# Dispatch helper
# ----------------------------------------------------------------------


def is_stitched_plate(connection: OmeroConnection, plate_id: int) -> bool:
    """Return True if the plate has any stitched-mode segmentation masks.

    The public entry point used by the widget to choose between the zarr
    builder and the existing diskcache builder.
    """
    return bool(
        get_plate_metadata(connection, plate_id).get("label_stitched_mode")
    )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_flatfield_dict(
    conn: BlitzGateway,
    ff_mask_id: int,
    channel_data: dict[str, str],
    plate_id: int | None = None,
) -> dict[str, npt.NDArray[Any]]:
    """Download the flatfield correction mask and split into per-channel arrays.

    The flatfield image has shape ``(T=1, Z=1, Y, X, C)`` with one C per
    plate channel. Returns ``{channel_name: (Y, X) float array}``.
    """
    ff_array = get_image(conn, ff_mask_id, tag=plate_id)
    # (T, Z, Y, X, C) → squeeze T and Z
    if ff_array.shape[0] != 1 or ff_array.shape[1] != 1:
        raise ValueError(
            f"Flatfield image {ff_mask_id} has unexpected shape {ff_array.shape}"
        )
    ff_yxc = ff_array[0, 0]
    out: dict[str, npt.NDArray[Any]] = {}
    for ch_name, idx_str in channel_data.items():
        ch_idx = int(idx_str)
        out[ch_name] = ff_yxc[..., ch_idx].astype(np.float32, copy=False)
    return out


def _missing_position_fields(well: WellWrapper) -> list[int]:
    """Field indices whose well sample has no recorded stage position.

    A failed acquisition leaves a blank image with no position. The
    pipeline that wrote ``canvas.csv`` could not place such a field, so
    this is the usual reason a well has no attachment.
    """
    return [
        i
        for i, ws in enumerate(well.listChildren())
        if ws.getPosX() is None or ws.getPosY() is None
    ]


def _no_offsets_error(well: WellWrapper, reason: str) -> PlateDataError:
    """Build the failure for a well whose canvas offsets are unusable.

    Deliberately fatal rather than recomputed from stage positions. The
    offsets themselves could be derived — but a well missing its
    attachment is, in practice, a well whose segmentation never
    succeeded, and its masks are present-but-empty. Deriving offsets
    would build a zarr full of blank labels and hide that, whereas every
    other check (mask names, map annotations) already reports the well as
    segmented. Failing loudly is the only signal left.
    """
    bad = _missing_position_fields(well)
    detail = (
        f" Field(s) {bad} have no stage position, so this well most likely "
        "failed segmentation — check its masks are not empty and re-run the "
        "plate."
        if bad
        else " Re-run the plate with --stitch to attach the offsets."
    )
    return PlateDataError(
        f"{reason} for well {well.getId()}: {well.getWellPos()}.{detail}",
        logger,
    )


def _load_canvas_offsets(well: WellWrapper) -> npt.NDArray[np.int_]:
    """Return the canvas offsets attached to the well by a ``--stitch`` run.

    Raises rather than deriving the offsets from stage positions — see
    :func:`_no_offsets_error` for why a missing attachment must stay
    fatal.
    """
    # Get the stitched canvas offsets
    offsets_ann = get_file_attachments(well, "canvas.csv")
    if offsets_ann is None:
        raise _no_offsets_error(well, "Missing stitched canvas offsets")
    offsets_df = parse_csv_data(offsets_ann[0])
    if offsets_df is None:
        raise _no_offsets_error(well, "Failed to load stitched canvas offsets")
    n = well.countWellSample()
    if len(offsets_df) != n:
        raise _no_offsets_error(
            well,
            f"Incorrect size for stitched canvas offsets "
            f"({len(offsets_df)} != {n})",
        )
    # offsets (N, 2)
    return np.column_stack((offsets_df["ox"], offsets_df["oy"])).astype(
        np.int_
    )


# Deprecated: This is only used for testing
def _load_well_fields(
    conn: BlitzGateway,
    well: WellWrapper,
    channel_data: dict[str, str],
    flatfield_dict: dict[str, npt.NDArray[Any]],
    omero_conn: Any | None = None,
    max_workers: int = 3,
    plate_id: int | None = None,
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[np.int_],
]:
    """Load one well's fields, flatfield-correct, and return as a stack.

    When ``omero_conn`` (an ``OmeroConnection``) is provided, field
    downloads run in parallel with one BlitzGateway per worker thread —
    BlitzGateway isn't safe to share across threads, mirroring the
    ``cache_plate`` pattern. Falls back to the sequential path when
    ``omero_conn`` is absent.

    Ignores any fields without a canvas offset.

    Returns:
        images: ``(N_fields, T, Y, X, C)`` float32 array.
        offsets: array of canvas offsets (ox, oy) per field (N_fields, 2).
    """
    image_ids = [int(ws.getImage().getId()) for ws in well.listChildren()]
    offsets = _load_canvas_offsets(well)  # (N, 2)

    valid = offsets[:, 0] >= 0
    valid_offsets = offsets[valid]
    fields = np.arange(len(offsets))[valid].tolist()

    n_fields = len(fields)
    channels = list(channel_data.keys())
    field_arrays: list[npt.NDArray[Any] | None] = [None] * n_fields

    def _download_one(idx: int, image_id: int) -> tuple[int, npt.NDArray[Any]]:
        """Worker: download one field with a thread-local connection.

        Each thread gets its own BlitzGateway (mirroring
        ``plate_cache._download_batch``) since BlitzGateway is not
        thread-safe and concurrent calls on the same conn serialise on
        the Ice transport anyway.
        """
        if omero_conn is not None:
            worker_conn = omero_conn.create_conn()
        else:
            worker_conn = conn
        try:
            # Tag with ``plate_id`` so ``evict_images(plate_id)`` can
            # find these entries when the user deletes the plate.
            # Falling back to ``image_id`` keeps the legacy single-shot
            # caller behaviour for headless tests.
            arr = get_image(
                worker_conn,
                image_id,
                tag=plate_id if plate_id is not None else image_id,
            )
        finally:
            if omero_conn is not None and worker_conn is not conn:
                with contextlib.suppress(Exception):
                    worker_conn.close()
        return idx, arr

    if omero_conn is not None and n_fields > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fut in as_completed(
                ex.submit(_download_one, i, image_ids[fields[i]])
                for i in range(n_fields)
            ):
                idx, arr = fut.result()
                field_arrays[idx] = arr
    else:
        for i in range(n_fields):
            _, field_arrays[i] = _download_one(i, image_ids[fields[i]])

    # Flatfield-correct on the main thread (CPU-bound, fast vs network).
    per_channel_fields: dict[str, list[npt.NDArray[Any]]] = {
        ch: [] for ch in channels
    }
    for n, array in zip(fields, field_arrays, strict=True):
        if array is None:
            raise RuntimeError(f"Field {n} of well failed to download")
        if array.shape[1] != 1:
            raise ValueError(
                f"Field image {image_ids[n]} has Z={array.shape[1]}; "
                f"expected Z=1"
            )
        for ch_name, idx_str in channel_data.items():
            ch_idx = int(idx_str)
            # Flatfield correction needs float arithmetic; immediately cast
            # back to uint16 (the source pixel type). Flatfield masks are
            # normalised to median=1, so values typically stay within the
            # uint16 range; clip handles dark-corner over-correction.
            raw = array[:, 0, :, :, ch_idx].astype(np.float32, copy=False)
            corrected = raw / flatfield_dict[ch_name]
            field = np.clip(corrected, 0, np.iinfo(np.uint16).max).astype(
                np.uint16, copy=False
            )
            per_channel_fields[ch_name].append(field)  # (T, Y, X)

    # Stack into (N, T, Y, X, C). Channel order follows channel_data.
    per_channel_stacks = [
        np.stack(per_channel_fields[ch], axis=0) for ch in channels
    ]  # each (N, T, Y, X)
    stacked = np.stack(per_channel_stacks, axis=-1)  # (N, T, Y, X, C)
    return stacked, valid_offsets


def _stitch_image(
    images_ntyxc: npt.NDArray[Any],
    offsets: npt.NDArray[np.int_],
    edge: int,
    canvas_hw: tuple[int, int] | None = None,
) -> npt.NDArray[Any]:
    """Stitch (N, T, Y, X, C) → (T, C, Y, X).

    Args:
        images_ntyxc: Field stack.
        offsets: Canvas offsets per field. May be negative when ``canvas_hw`` is
            given, which happens for a cyclic-IF restain round whose alignment
            shift pushes tiles left of or above the master's origin.
        edge: Blend width. Always passed explicitly for a 4i build so every
            round blends identically.
        canvas_hw: Force the output canvas to exactly this ``(height, width)``,
            cropping to the master's frame. None keeps the legacy behaviour of
            deriving the canvas from the offsets.
    """
    if canvas_hw is None:
        stitched_tyxc = stitch_from_offsets(
            images_ntyxc, offsets, edge=edge
        )  # (T, Y, X, C)
    else:
        stitched_tyxc = stitch_into_canvas(
            images_ntyxc, offsets, canvas_hw, edge=edge
        )
    # Reorder to writer's expected layout (T, C, Y, X).
    return np.transpose(stitched_tyxc, (0, 3, 1, 2))


def _load_stitch_image_block(
    conn: BlitzGateway,
    omero_conn: Any | None,
    image_ids: list[int],
    offsets: npt.NDArray[np.int_],
    edge: int,
    channel_data: dict[str, str],
    flatfield_dict: dict[str, npt.NDArray[Any]],
    t0: int,
    t1: int,
    plate_id: int,
    canvas_hw: tuple[int, int] | None = None,
) -> npt.NDArray[Any]:
    """Download + flatfield + stitch one timepoint block → ``(bt, C, Y, X)``.

    Runs inside a dask task (on a worker thread), so it uses its own
    thread-local BlitzGateway and reads only timepoints ``[t0, t1)`` of each
    field — the image disk-cache makes the probe re-read of block 0 free.

    ``canvas_hw`` forces the output spatial dims, which a cyclic-IF restain
    round needs so its canvas matches the master's. The result is asserted
    against it: the caller declares this block's dask shape from a probe of
    block 0, so a size mismatch would corrupt the written store rather than
    raise.
    """
    worker_conn = omero_conn.create_conn() if omero_conn is not None else conn
    try:
        channels = list(channel_data)
        per_channel: dict[str, list[npt.NDArray[Any]]] = {
            ch: [] for ch in channels
        }
        for img_id in image_ids:
            arr = get_image(
                worker_conn, img_id, start=t0, end=t1, tag=plate_id
            )  # (bt, Z, Y, X, C)
            for ch_name, idx_str in channel_data.items():
                ch_idx = int(idx_str)
                raw = arr[:, 0, :, :, ch_idx].astype(np.float32, copy=False)
                corrected = raw / flatfield_dict[ch_name]
                field = np.clip(corrected, 0, np.iinfo(np.uint16).max).astype(
                    np.uint16, copy=False
                )
                per_channel[ch_name].append(field)  # (bt, Y, X)
        per_channel_stacks = [
            np.stack(per_channel[ch], axis=0) for ch in channels
        ]  # each (N, bt, Y, X)
        images_ntyxc = np.stack(per_channel_stacks, axis=-1)
    finally:
        if omero_conn is not None and worker_conn is not conn:
            with contextlib.suppress(Exception):
                worker_conn.close()
    stitched = _stitch_image(
        images_ntyxc, offsets, edge, canvas_hw
    )  # (bt, C, Y, X)
    if canvas_hw is not None:
        assert stitched.shape[-2:] == canvas_hw, (
            f"stitched block is {stitched.shape[-2:]}, expected {canvas_hw}"
        )
    return stitched


def _load_recompose_label_block(
    conn: BlitzGateway,
    omero_conn: Any | None,
    mask_ids: list[int],
    source_ids: list[int],
    offsets: npt.NDArray[np.int_],
    t0: int,
    t1: int,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any] | None]:
    """Download + recompose one timepoint block of label masks.

    Returns ``(nuclei (bt, Y, X), cells (bt, Y, X) | None)``. Uses one
    thread-local connection and reads sequentially within the block — dask
    provides the cross-block parallelism.
    """
    worker_conn = omero_conn.create_conn() if omero_conn is not None else conn
    try:
        nuc_fields, cell_fields = fetch_stitched_field_masks_trange(
            worker_conn,
            mask_ids,
            t0=t0,
            t1=t1,
            source_ids=source_ids,
            conn_factory=None,
            max_workers=1,
        )
    finally:
        if omero_conn is not None and worker_conn is not conn:
            with contextlib.suppress(Exception):
                worker_conn.close()
    # uint32 to match the dask array's declared dtype (and the zarr label
    # dtype the writer casts to) — labels can exceed uint16 on big wells.
    nuc = recompose_tiles(nuc_fields, offsets).astype(np.uint32, copy=False)
    if any(c is not None for c in cell_fields):
        if not all(c is not None for c in cell_fields):
            raise ValueError(
                "Well has cell masks for some fields but not all — "
                "refusing to recompose mixed coverage."
            )
        cell = recompose_tiles(
            [c for c in cell_fields if c is not None],
            offsets,
        ).astype(np.uint32, copy=False)
    else:
        cell = None
    return nuc, cell


@dataclass
class RoundSpec:
    """One cyclic-IF restain round, resolved for a single well.

    Attributes:
        plate_id: The restain plate.
        well: That plate's well object, same position as the master's.
        shifts: ``(n_master_fields, 2)`` integer ``(dx, dy)``, indexed by master
            field index. Subtracted from the master's canvas offsets.
        image_ids: Restain image ID per master field index, or None when the
            per-well alignment table was used and the mapping is positional.
        channel_data: ``{channel_name: index_string}`` for this round.
        flatfield_dict: Per-channel flatfield masks for this round.
    """

    plate_id: int
    well: WellWrapper
    shifts: npt.NDArray[np.int_]
    image_ids: list[int] | None
    channel_data: dict[str, str]
    flatfield_dict: dict[str, npt.NDArray[Any]]

    def field_ids(self, fields: list[int]) -> list[int]:
        """Restain image IDs for the given master field indices.

        Prefers the IDs carried in ``sample_alignment.csv``: resolving by ID
        means field correspondence never depends on ``listChildren()`` ordering
        being stable across two separately-imported plates. Falls back to
        position only when the per-well table was used, which carries no IDs.
        """
        if self.image_ids is not None:
            return [int(self.image_ids[i]) for i in fields]
        children = list(self.well.listChildren())
        return [int(children[i].getImage().getId()) for i in fields]


def _build_lazy_well_arrays(
    conn: BlitzGateway,
    omero_conn: Any | None,
    well: WellWrapper,
    channel_data: dict[str, str],
    flatfield_dict: dict[str, npt.NDArray[Any]],
    plate_id: int,
    block_t: int = _CACHE_BLOCK_T,
    round_specs: list[RoundSpec] | None = None,
) -> tuple[
    da.Array,
    da.Array,
    da.Array | None,
]:
    """Build lazy dask ``image (T,C,Y,X)`` + ``nuclei/cells (T,Y,X)`` arrays.

    Each dask chunk is a delayed ``[t0, t1)`` stitch/recompose, so the writer
    (`write_image`/`write_labels`, both dask-streaming) pulls one block at a
    time and never materialises the whole well. Block 0 is loaded eagerly once
    to probe canvas/tile dims and cell-mask presence (cheap; cached on disk).

    Returns:
        ``(image_dask, nuclei_dask, cells_dask_or_None)``.
    """
    offsets = _load_canvas_offsets(well)  # (N, 2)

    valid = (offsets[:, 0] >= 0) & (offsets[:, 1] >= 0)
    valid_offsets = offsets[valid]
    fields = np.arange(len(offsets))[valid].tolist()

    first = well.getWellSample(0).getImage()
    n_t = int(first.getSizeT())
    n_ch = len(channel_data)
    mask_ids, source_ids = resolve_stitched_mask_ids(well, fields)

    # Auto edge
    tile_h, tile_w = int(first.getSizeY()), int(first.getSizeX())
    edge = get_overlap(valid_offsets, tile_h, tile_w)
    logger.debug(f"Stitching {well.getWellPos()} using auto-edge: {edge}")

    # Probe block 0 for canvas dims (image) and cell presence (labels).
    probe_img = _load_stitch_image_block(
        conn,
        omero_conn,
        source_ids,
        valid_offsets,
        edge,
        channel_data,
        flatfield_dict,
        0,
        1,
        plate_id,
    )  # (1, C, Y, X)
    cy, cx = int(probe_img.shape[2]), int(probe_img.shape[3])
    nuc0, cell0 = _load_recompose_label_block(
        conn, omero_conn, mask_ids, source_ids, valid_offsets, 0, 1
    )
    ly, lx = int(nuc0.shape[1]), int(nuc0.shape[2])
    has_cells = cell0 is not None

    blocks = [(t, min(t + block_t, n_t)) for t in range(0, n_t, block_t)]
    img_parts = [
        da.from_delayed(
            delayed(_load_stitch_image_block)(
                conn,
                omero_conn,
                source_ids,
                valid_offsets,
                edge,
                channel_data,
                flatfield_dict,
                t0,
                t1,
                plate_id,
            ),
            shape=(t1 - t0, n_ch, cy, cx),
            dtype=np.uint16,
        )
        for t0, t1 in blocks
    ]
    image_dask = da.concatenate(img_parts, axis=0)

    # Cyclic-IF: append each restain round's channels along C. Every round is
    # stitched onto *this* canvas -- the master's offsets shifted by the
    # alignment, then cropped back to (cy, cx) -- so all rounds share one frame
    # and a crop is a single slice across the whole multiplexed stack. Labels
    # stay master-only; only the master round is segmented.
    for spec in round_specs or []:
        round_ids = spec.field_ids(fields)
        shifted_offsets = valid_offsets - spec.shifts[fields]
        n_round_ch = len(spec.channel_data)
        round_parts = [
            da.from_delayed(
                delayed(_load_stitch_image_block)(
                    conn,
                    omero_conn,
                    round_ids,
                    shifted_offsets,
                    # The master's blend width, not an auto-detected one: the
                    # shifted offsets have a different relative geometry, so
                    # auto-detection would blend this round differently and give
                    # the same stain different pixel values across rounds.
                    edge,
                    spec.channel_data,
                    spec.flatfield_dict,
                    t0,
                    t1,
                    spec.plate_id,
                    (cy, cx),
                ),
                shape=(t1 - t0, n_round_ch, cy, cx),
                dtype=np.uint16,
            )
            for t0, t1 in blocks
        ]
        image_dask = da.concatenate(
            [image_dask, da.concatenate(round_parts, axis=0)], axis=1
        )

    nuc_parts: list[da.Array] = []
    cell_parts: list[da.Array] = []
    for t0, t1 in blocks:
        # One delayed compute per block; nuclei and cells index the same
        # node so dask downloads the block only once.
        lbl = delayed(_load_recompose_label_block)(
            conn,
            omero_conn,
            mask_ids,
            source_ids,
            valid_offsets,
            t0,
            t1,
        )
        nuc_parts.append(
            da.from_delayed(lbl[0], shape=(t1 - t0, ly, lx), dtype=np.uint32)
        )
        if has_cells:
            cell_parts.append(
                da.from_delayed(
                    lbl[1], shape=(t1 - t0, ly, lx), dtype=np.uint32
                )
            )
    nuclei_dask = da.concatenate(nuc_parts, axis=0)
    cells_dask = da.concatenate(cell_parts, axis=0) if has_cells else None
    return image_dask, nuclei_dask, cells_dask


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def resolve_target_wells(
    plate_id: int,
    connection: OmeroConnection,
    *,
    wells: Iterable[str] | None = None,
) -> list[str]:
    """Return the wells that :func:`build_plate_zarr` would build.

    Performs the same empty-well and already-cached filtering as
    :func:`build_plate_zarr` but without any pixel I/O — only the
    OMERO well-map HQL query. Intended for the napari widget so it
    can plan the build (e.g. size a determinate progress bar) on the
    main thread before spawning the worker that does the heavy lifting.

    Args:
        plate_id: OMERO plate ID.
        connection: OMERO connection.
        wells: Optional subset of well labels; default is every well on
            the plate.

    Returns:
        Ordered list of well positions that would be built. Empty when
        nothing is left to do (all wells empty or already cached).
    """
    from omero_screen_napari.zarr_cache.reader import cached_wells

    well_map = get_well_data(connection, plate_id)
    non_empty = {
        pos: info
        for pos, info in well_map.items()
        if not is_empty_well(info.get("metadata", {}))
    }
    if wells is not None:
        candidates = [w for w in sorted(wells) if w in non_empty]
    else:
        candidates = sorted(non_empty.keys())
    already_built = set(cached_wells(plate_id))
    return [w for w in candidates if w not in already_built]


def _round_specs_for_well(
    group: RoundGroup,
    well_pos: str,
    master_well: WellWrapper,
    round_well_objs: dict[int, dict[str, WellWrapper]],
    round_flatfields: dict[int, dict[str, npt.NDArray[Any]]],
    channel_data_by_plate: dict[int, dict[str, str]],
) -> list[RoundSpec] | None:
    """Resolve every restain round for one well, or None if the well must be skipped.

    A well is skipped whole rather than partially: writing a round as zeros
    would be indistinguishable from a dead stain, and the multiplexed feature
    space has no meaning for a cell missing a round.
    """
    assert group.alignment is not None
    n_fields = int(master_well.countWellSample())
    specs: list[RoundSpec] = []
    for member_id in group.member_plate_ids:
        round_well = round_well_objs[member_id].get(well_pos)
        if round_well is None:
            logger.warning(
                f"Well {well_pos} missing from restain plate {member_id}; "
                f"skipping the well for the whole 4i group"
            )
            return None
        round_fields = int(round_well.countWellSample())
        if round_fields != n_fields:
            logger.warning(
                f"Well {well_pos}: master has {n_fields} field(s) but restain "
                f"plate {member_id} has {round_fields}; skipping the well. "
                f"Field correspondence across rounds is positional."
            )
            return None
        try:
            shifts = group.alignment.shifts_for_well(
                member_id, well_pos, n_fields
            )
        except AlignmentError as exc:
            logger.warning(f"Well {well_pos}: {exc}; skipping the well")
            return None
        specs.append(
            RoundSpec(
                plate_id=member_id,
                well=round_well,
                shifts=shifts.shifts,
                image_ids=shifts.image_ids,
                channel_data=channel_data_by_plate[member_id],
                flatfield_dict=round_flatfields[member_id],
            )
        )
    return specs


def build_plate_zarr(
    plate_id: int,
    conn: BlitzGateway,
    *,
    wells: Iterable[str] | None = None,
    progress_cb: Callable[[str], None] | None = None,
    step_cb: Callable[[str, float], None] | None = None,
    omero_conn: Any | None = None,
    max_workers: int = 3,
    round_group: RoundGroup | None = None,
) -> Iterator[str]:
    """Build (or extend) the plate.zarr cache for ``plate_id``.

    Generator yielding each well ID after it has been written. Callers
    (the widget worker thread or a script) can use that to drive a
    progress indicator.

    Args:
        plate_id: OMERO plate ID.
        conn: Live OMERO connection.
        wells: Subset of well labels to build (e.g. ``["A1", "B7"]``).
            If ``None``, every well advertised by the plate is built.
        progress_cb: Optional callback invoked with the well ID after
            each successful write — same value the generator yields.
            Lets non-iterator callers (UI thread) react.
        step_cb: Optional callback invoked with ``(well_id, fraction)`` as
            each well streams to disk (fraction in ``[0, 1]``). Drives a
            sub-well progress bar so long live-cell wells don't read as
            stalled. Called from the build thread — marshal to the UI thread
            (e.g. ``superqt.ensure_main_thread``) before touching widgets.
        omero_conn: Optional ``OmeroConnection`` used to spawn per-thread
            BlitzGateway connections for the parallel download path.
            When ``None``, downloads run sequentially on ``conn``.
        max_workers: Concurrency for the parallel download path.
        round_group: Build a cyclic-IF (4i) store holding every round's
            channels pre-aligned into this plate's frame. ``plate_id`` must be
            the group's master. When None (the default) an ordinary
            single-plate store is written, byte-identical to the pre-4i format.

    Raises:
        ValueError: if ``round_group`` is not buildable, or names a different
            master than ``plate_id``.
    """
    if round_group is not None:
        if round_group.master_plate_id != plate_id:
            raise ValueError(
                f"round_group is for master plate "
                f"{round_group.master_plate_id}, not {plate_id}"
            )
        if not round_group.buildable:
            raise ValueError(
                f"Plate {plate_id} cannot be built as a 4i group: "
                + "; ".join(round_group.blockers)
            )
    cache_conn = omero_conn if omero_conn is not None else OmeroConnection()

    metadata = get_plate_metadata(cache_conn, plate_id)
    channel_data: dict[str, str] = metadata["channel_data"]
    pixel_size = metadata["pixel_size"]
    plate_name = metadata["plate_name"]
    ff_mask_id = metadata["ff_mask_id"]

    well_map = get_well_data(cache_conn, plate_id)
    # Mirror omero-screen's empty-well filter (loops.py): wells with no
    # metadata, no cell_line, or cell_line == "Empty" are excluded from
    # both segmentation and the zarr cache.
    empty_wells = sorted(
        pos
        for pos, info in well_map.items()
        if is_empty_well(info.get("metadata", {}))
    )
    for pos in empty_wells:
        well_map.pop(pos, None)
    if empty_wells:
        logger.info(
            f"Plate {plate_id}: skipping {len(empty_wells):d} empty well(s): {empty_wells}"
        )
    all_wells = sorted(well_map.keys())
    if not all_wells:
        raise ValueError(
            f"Plate {plate_id} has no non-empty wells "
            f"(skipped {len(empty_wells)} empty)"
        )
    if wells is not None:
        # Drop any caller-requested wells that are empty so we don't try
        # to build them. The warning later (well_pos not in well_objs) is
        # not the right channel for this — empty wells exist on the plate.
        requested = sorted(wells)
        target_wells = [w for w in requested if w in well_map]
        dropped = [w for w in requested if w in empty_wells]
        if dropped:
            logger.info(
                f"Plate {plate_id}: requested wells dropped as empty: {dropped}"
            )
    else:
        target_wells = all_wells

    # Resumability: skip wells already on disk so a cancelled build
    # picks up where it left off rather than re-downloading completed
    # wells. ``cached_wells`` reads per-well directory existence under
    # the plate zarr root.
    from omero_screen_napari.zarr_cache.reader import cached_wells

    already_built = set(cached_wells(plate_id))
    if already_built:
        remaining = [w for w in target_wells if w not in already_built]
        skipped = [w for w in target_wells if w in already_built]
        if skipped:
            logger.info(
                f"Plate {plate_id}: {len(skipped):d} well(s) already in zarr cache, skipping: {skipped}"
            )
        target_wells = remaining

    # n_timepoints from the first image of any well. Plates are
    # homogeneous in T.
    first_well = well_map[all_wells[0]]
    first_image = first_well["images"][0]
    n_timepoints = int(first_image["dims"][0])  # dims = (T, C, Z, Y, X)

    pixel_size_um = pixel_size[0] if pixel_size else None

    # Pre-flight: estimate the build's footprint and evict LRU plates if
    # we'd otherwise blow the cap. Raises ZarrCacheTooSmall if a single
    # plate exceeds the cap on its own.
    field_dims = first_image["dims"]  # (T, C, Z, Y, X)
    field_h = int(field_dims[3])
    field_w = int(field_dims[4])
    n_fields_first = len(first_well["images"])
    # Stitched canvas is ~ sqrt(n_fields) × tile. Use a generous upper
    # bound: ceil(sqrt(N)) × tile to account for overlap subtraction not
    # quite reaching the theoretical max.
    import math

    grid_side = math.ceil(math.sqrt(max(n_fields_first, 1)))
    stitched_h = grid_side * field_h
    stitched_w = grid_side * field_w
    # Per-round channel data, and the flat channel list the store will carry.
    channel_data_by_plate: dict[int, dict[str, str]] = {plate_id: channel_data}
    rounds_attrs: dict[str, Any] | None = None
    store_channel_names = list(channel_data.keys())
    if round_group is not None:
        for member_id in round_group.member_plate_ids:
            channel_data_by_plate[member_id] = get_plate_metadata(
                cache_conn, member_id
            )["channel_data"]
        (
            store_channel_names,
            rounds_attrs,
            load_channel_data,
        ) = build_channel_plan(round_group, channel_data_by_plate)
        # Redundant repeats (the per-round nuclear stain) are dropped from the
        # plan, so restrict what each round downloads to what is kept.
        channel_data_by_plate = load_channel_data
        logger.info(
            f"Plate {plate_id}: building 4i store over "
            f"{round_group.n_rounds} round(s) {list(round_group.plate_ids)} "
            f"-> {len(store_channel_names)} channels"
        )

    estimated = estimate_plate_size_bytes(
        n_wells=len(target_wells),
        n_timepoints=n_timepoints,
        # Summed across rounds: passing one round's count would under-reserve
        # by a factor of N and blow the cap mid-build.
        n_channels=len(store_channel_names),
        stitched_h=stitched_h,
        stitched_w=stitched_w,
    )
    evicted = enforce_size_cap(extra_bytes=estimated)
    if evicted:
        logger.info(f"Evicted {len(evicted):d} plates to make room: {evicted}")

    flatfield_dict = _load_flatfield_dict(
        conn, ff_mask_id, channel_data, plate_id=plate_id
    )

    # Need the live Plate object to iterate fields. fetch_well_map only
    # returns ids + positions; per-field pixel download still wants the
    # OMERO wrapper.
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise ValueError(f"Plate {plate_id} not found in OMERO")
    well_objs: dict[str, WellWrapper] = {
        w.getWellPos(): w for w in plate.listChildren()
    }

    # Per-round well objects and flatfield masks. Each round has its own
    # channels and its own flatfield correction.
    round_well_objs: dict[int, dict[str, WellWrapper]] = {}
    round_flatfields: dict[int, dict[str, npt.NDArray[Any]]] = {}
    if round_group is not None:
        for member_id in round_group.member_plate_ids:
            member_plate = conn.getObject("Plate", member_id)
            if member_plate is None:
                raise ValueError(
                    f"Restain plate {member_id} not found in OMERO"
                )
            round_well_objs[member_id] = {
                w.getWellPos(): w for w in member_plate.listChildren()
            }
            round_flatfields[member_id] = _load_flatfield_dict(
                conn,
                get_plate_metadata(cache_conn, member_id)["ff_mask_id"],
                channel_data_by_plate[member_id],
                plate_id=member_id,
            )

    dataset_id = get_dataset_id(conn, plate_id)
    if not dataset_id:
        raise ValueError(
            f"Plate {plate_id} has no segmentation dataset — cannot find "
            f"stitched masks. Was the stitched analysis run?"
        )

    writer = PlateZarrWriter(
        plate_id=plate_id,
        plate_name=plate_name,
        channel_names=store_channel_names,
        pixel_size_um=pixel_size_um,
        n_timepoints=n_timepoints,
        rounds=rounds_attrs,
    )

    # Stash per-well metadata (cell_line, condition, timepoint, ...) so
    # the napari load path can render an overlay without re-hitting OMERO.
    well_meta_map = {
        well_pos: dict(well_map[well_pos].get("metadata", {}))
        for well_pos in all_wells
    }

    # Bound dask to a few worker threads so the streamed writes hold only a
    # few blocks at once — without this the default scheduler would fan out
    # across all cores and undo the memory bound.
    # Pin for the duration of the build: a concurrent build's pre-flight
    # eviction would otherwise be free to reclaim a partially written store.
    with (
        transient_pin(plate_id),
        dask.config.set(scheduler="threads", num_workers=_CACHE_DASK_WORKERS),
        writer,
    ):
        writer.ensure_plate(all_wells=all_wells, well_metadata=well_meta_map)

        for well_pos in target_wells:
            if well_pos not in well_objs:
                logger.warning(
                    f"Well {well_pos} requested but not present on plate {plate_id}; skipping"
                )
                continue
            well_obj = well_objs[well_pos]
            logger.info(
                f"Building zarr for well {well_pos} of plate {plate_id}"
            )

            # Build lazy dask arrays — image + labels stitched per timepoint
            # block on demand, so the writer (dask-streaming) never holds the
            # whole well in RAM. Probing block 0 also surfaces a non-stitched
            # well via the KeyError from the label id resolution.
            round_specs: list[RoundSpec] | None = None
            if round_group is not None:
                round_specs = _round_specs_for_well(
                    round_group,
                    well_pos,
                    well_obj,
                    round_well_objs,
                    round_flatfields,
                    channel_data_by_plate,
                )
                if round_specs is None:
                    continue

            try:
                image_tcyx, nuc_stitched, cell_stitched = (
                    _build_lazy_well_arrays(
                        conn,
                        omero_conn,
                        well_obj,
                        channel_data,
                        flatfield_dict,
                        plate_id,
                        round_specs=round_specs,
                    )
                )
            except KeyError as e:
                # Well wasn't processed in stitched mode (no
                # Stitched_Segmentation_Mask annotation). Skip rather than
                # abort so partially-stitched plates still build for the
                # wells that *do* qualify.
                logger.warning(
                    f"Skipping well {well_pos}: not processed in stitched mode ({e})"
                )
                continue

            # Record where the canvas has tile-sized holes so the viewer can
            # mark them. A field whose acquisition failed is left out of the
            # stitch, and an unlabelled blank rectangle is indistinguishable
            # from a display fault when you are looking at the well.
            first_field = well_obj.getWellSample(0).getImage()
            holes = missing_field_boxes(
                _load_canvas_offsets(well_obj),
                int(first_field.getSizeY()),
                int(first_field.getSizeX()),
            )
            if holes:
                logger.info(
                    f"Well {well_pos}: {len(holes):d} unimaged region(s) on the "
                    f"canvas from failed acquisition; recorded for display"
                )

            # --- Write (streams the dask arrays block-by-block) ---------------
            writer.write_well(
                well_pos,
                image_tcyx,
                nuc_stitched,
                cell_stitched,
                progress_cb=(
                    partial(step_cb, well_pos) if step_cb is not None else None
                ),
                missing_regions=holes,
            )

            if progress_cb is not None:
                progress_cb(well_pos)
            yield well_pos

    # Register the plate (overwrites any prior entry, refreshing size and
    # well count). Doing this once at the end avoids hammering the JSON
    # with per-well writes.
    upsert(
        ZarrPlateEntry(
            plate_id=plate_id,
            plate_name=plate_name,
            size_bytes=_dir_size(plate_zarr_path(plate_id)),
            n_wells_written=len(target_wells),
            # One entry per store. Recording the restain rounds here is what
            # stops ``evict_plate(restain_id)`` deleting this store, and what
            # lets a consumer holding a restain ID find these pixels.
            member_plate_ids=(
                list(round_group.member_plate_ids)
                if round_group is not None
                else []
            ),
        )
    )
    logger.info(
        f"Finished zarr build for plate {plate_id}: {len(target_wells):d} wells"
    )

    # No track export here: the Mastodon CTC bundle (mask TIFFs + res_track.txt)
    # is written on demand, per well, from the Tracks widget
    # (omero_screen_napari.mastodon_export.export_well_ctc) — writing a TIFF
    # stack for every well at build time would be needless duplication.
