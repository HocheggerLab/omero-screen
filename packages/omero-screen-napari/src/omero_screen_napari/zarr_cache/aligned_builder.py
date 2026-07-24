"""Build an aligned cyclic-IF (4i) plate.zarr from multiple cycle plates.

Each 4i imaging cycle is a **separate OMERO plate** of the same physical
wells. :func:`build_aligned_zarr` assembles every cycle of a master plate
into a *single* aligned, stitched OME-Zarr in the isolated ``aligned/``
cache namespace (see :func:`omero_screen_napari.zarr_cache.paths`), so a
downstream viewer / crop sees one multi-channel image in one coordinate
frame — structurally identical to a plain stitched ``plate_<id>.zarr``.

Per well the assembly is:

1. **Master cycle** — stitch the well's fields (no offset) into the master
   canvas; recompose the master's stitched nuclei mask. This canvas defines
   the *master extent* every other cycle is cropped to.
2. **Repeat cycles** — stitch each repeat's fields with a per-field pixel
   ``field_offsets = -(x, y)`` (the per-well alignment from ``alignment.csv``)
   so the repeat lands in the master frame, then crop to the master extent.
   Only the **non-nucleus** channels are appended (a single nucleus channel,
   the master's, is kept).
3. **Cell mask** — the master (first) plate does not necessarily carry a cell
   mask. The cell mask is taken from whichever cycle actually has one
   (master preferred); if it comes from a repeat it is shifted into the
   master frame by ``-(x, y)`` via :func:`plate_aggregation._translate`
   (the per-well shift is uniform across fields, so a whole-canvas translate
   is pixel-identical to a per-field offset). Combined with the master nuclei.

4i acquisitions are fixed-cell (``align_plates`` requires ``T = Z = 1``), so a
well is assembled eagerly one frame at a time — no dask streaming.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger
from omero.gateway import BlitzGateway, WellWrapper
from omero_screen.metadata_parser import MetadataParser
from omero_screen.plate_aggregation import _translate, get_plate_alignments
from omero_utils.images import (
    fetch_stitched_field_masks_trange,
    resolve_stitched_mask_ids,
)
from omero_utils.stitching import (
    OPERETTA_STITCH_DEFAULTS,
    recompose_split_labels,
    stitch_from_positions,
)

from omero_screen_napari.plate_cache import (
    _fetch_plate_metadata,
    _fetch_well_map,
    is_empty_well,
)
from omero_screen_napari.zarr_cache.builder import (
    _LABEL_PLACEMENT_KEYS,
    _load_flatfield_dict,
    _load_well_fields,
)
from omero_screen_napari.zarr_cache.paths import aligned_zarr_root
from omero_screen_napari.zarr_cache.reader import cached_wells
from omero_screen_napari.zarr_cache.registry import ZarrPlateEntry, upsert
from omero_screen_napari.zarr_cache.writer import PlateZarrWriter

# ----------------------------------------------------------------------
# Pure assembly logic (no OMERO) — unit-testable
# ----------------------------------------------------------------------


@dataclass
class CycleCanvas:
    """One imaging cycle's stitched, master-framed arrays for a single well.

    ``image_tcyx`` and ``cells_tyx`` are already in the master coordinate
    frame (offset-shifted + cropped to the master extent). ``nucleus_channel``
    names the channel to drop when this cycle is a repeat (a single nucleus,
    the master's, is kept in the union stack).
    """

    image_tcyx: npt.NDArray[Any]  # (T, C, Y, X), master extent
    channel_names: list[str]
    nucleus_channel: str | None  # name to drop for repeats; None keeps all
    cells_tyx: npt.NDArray[Any] | None  # (T, Y, X) master-framed, or None
    cycle_tag: str = (
        ""  # unique id (plate id) used to name nuclei in debug mode
    )


def crop_or_pad(
    arr: npt.NDArray[Any], height: int, width: int
) -> npt.NDArray[Any]:
    """Crop or zero-pad the trailing ``(Y, X)`` axes of ``arr`` to ``(height, width)``.

    Repeat cycles are stitched from their own stage positions and should land
    on a grid identical to the master's, but a one-pixel grid difference
    between cycle plates must not desync the channel stack — every canvas is
    forced to the master extent. Cropping keeps the top-left origin (the
    stitch anchor); padding fills the far edge with zeros.
    """
    *lead, y, x = arr.shape
    if (y, x) == (height, width):
        return arr
    cy, cx = min(y, height), min(x, width)
    out = np.zeros((*lead, height, width), dtype=arr.dtype)
    out[..., :cy, :cx] = arr[..., :cy, :cx]
    return out


def combine_cycles(
    cycles: list[CycleCanvas],
    *,
    keep_all_nuclei: bool = False,
) -> tuple[npt.NDArray[Any], list[str], npt.NDArray[Any] | None]:
    """Union the cycles' channels into one stack + pick the cell mask.

    The first cycle is the master: all its channels are kept (including its
    nucleus). Each subsequent (repeat) cycle contributes only channels not
    already present by name and not its own nucleus channel — so exactly one
    nucleus channel survives.

    ``keep_all_nuclei`` (debug) keeps *every* cycle's nucleus channel, renamed
    ``{nucleus}_{cycle_tag}`` so each survives de-duplication and shows as its
    own layer. Used to eyeball repeat→master registration: if the per-cycle
    nuclei overlay, the alignment offset is correct.

    The cell mask is the first cycle that carries one (master preferred; the
    repeat canvases are already shifted into the master frame).

    Returns:
        ``(image_tcyx, channel_names, cells_tyx_or_None)``.
    """
    if not cycles:
        raise ValueError("cycles must be non-empty")

    master = cycles[0]
    height, width = master.image_tcyx.shape[2], master.image_tcyx.shape[3]

    channel_arrays: list[npt.NDArray[Any]] = []
    channel_names: list[str] = []
    seen: set[str] = set()

    def _add(cycle: CycleCanvas, *, is_master: bool) -> None:
        for c, name in enumerate(cycle.channel_names):
            is_nucleus = name == cycle.nucleus_channel
            if is_nucleus and not is_master and not keep_all_nuclei:
                continue  # repeat nucleus dropped (normal mode)
            out_name = name
            if is_nucleus and keep_all_nuclei:
                # Unique per-cycle name so every nucleus survives dedup.
                out_name = (
                    f"{name}_{cycle.cycle_tag}" if cycle.cycle_tag else name
                )
            if out_name in seen:
                continue
            seen.add(out_name)
            channel_names.append(out_name)
            channel_arrays.append(
                crop_or_pad(cycle.image_tcyx[:, c], height, width)
            )

    _add(master, is_master=True)
    for cycle in cycles[1:]:
        _add(cycle, is_master=False)

    # (T, Y, X) per channel → (T, C, Y, X).
    image_tcyx = np.stack(channel_arrays, axis=1)

    cells: npt.NDArray[Any] | None = None
    for cycle in cycles:
        if cycle.cells_tyx is not None:
            cells = crop_or_pad(cycle.cells_tyx, height, width)
            break

    return image_tcyx, channel_names, cells


# ----------------------------------------------------------------------
# OMERO I/O helpers
# ----------------------------------------------------------------------


def _nucleus_channel_name(
    conn: BlitzGateway, plate_id: int, channel_data: dict[str, str]
) -> str | None:
    """Resolve a plate's nucleus-role channel name (mirrors ``align_plates``).

    Returns ``None`` if the role can't be resolved, in which case channel
    de-duplication by name is the only guard against a doubled nucleus.
    """
    try:
        meta = MetadataParser(conn, plate_id)
        meta.manage_metadata()
        name = meta.channel_roles.get("nucleus")
        if name is not None and name in channel_data:
            return str(name)
    except Exception as exc:  # noqa: BLE001 — best-effort; dedup still guards
        logger.warning(
            f"Could not resolve nucleus channel for plate {plate_id}: {exc}"
        )
    return None


def _stitch_well_image(
    conn: BlitzGateway,
    well: WellWrapper,
    channel_data: dict[str, str],
    flatfield_dict: dict[str, npt.NDArray[Any]],
    field_offsets: list[tuple[int, int]] | None,
    omero_conn: Any | None,
    plate_id: int,
) -> npt.NDArray[Any]:
    """Load + flatfield + stitch one well's fields → ``(T, C, Y, X)``.

    ``field_offsets`` (per field, ``-(x, y)``) shifts a repeat cycle into the
    master frame; ``None`` for the master.
    """
    images_ntyxc, positions, _, _ = _load_well_fields(
        conn,
        well,
        channel_data,
        flatfield_dict,
        omero_conn=omero_conn,
        plate_id=plate_id,
    )
    stitched_tyxc = stitch_from_positions(
        images_ntyxc,
        positions,
        edge=OPERETTA_STITCH_DEFAULTS["edge"],
        overlap_x=OPERETTA_STITCH_DEFAULTS["overlap_x"],
        overlap_y=OPERETTA_STITCH_DEFAULTS["overlap_y"],
        translate_x=OPERETTA_STITCH_DEFAULTS["translate_x"],
        translate_y=OPERETTA_STITCH_DEFAULTS["translate_y"],
        field_offsets=field_offsets,
    )  # (T, Y, X, C)
    return np.transpose(stitched_tyxc, (0, 3, 1, 2))  # (T, C, Y, X)


def _recompose_well_masks(
    conn: BlitzGateway,
    well: WellWrapper,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any] | None]:
    """Recompose a well's stitched nuclei + cell masks → ``(T, Y, X)`` each.

    ``cells`` is ``None`` when this plate's stitched masks are nucleus-only.
    """
    mask_ids, source_ids = resolve_stitched_mask_ids(well)
    n_fields = len(list(well.listChildren()))
    positions: list[tuple[float, float]] = []
    tile_h = tile_w = 0
    for n in range(n_fields):
        ws = well.getWellSample(n)
        px, py = ws.getPosX(), ws.getPosY()
        positions.append(
            (
                px.getValue() if px is not None else 0.0,
                py.getValue() if py is not None else 0.0,
            )
        )
        img = ws.getImage()
        tile_h, tile_w = int(img.getSizeY()), int(img.getSizeX())

    nuc_fields, cell_fields = fetch_stitched_field_masks_trange(
        conn, mask_ids, t0=0, t1=1, source_ids=source_ids, max_workers=1
    )
    placement = {k: OPERETTA_STITCH_DEFAULTS[k] for k in _LABEL_PLACEMENT_KEYS}
    nuc = recompose_split_labels(
        nuc_fields, positions, tile_h, tile_w, **placement
    ).astype(np.uint32, copy=False)

    if all(c is not None for c in cell_fields):
        cell = recompose_split_labels(
            [c for c in cell_fields if c is not None],
            positions,
            tile_h,
            tile_w,
            **placement,
        ).astype(np.uint32, copy=False)
    elif any(c is not None for c in cell_fields):
        raise ValueError(
            "Well has cell masks for some fields but not all — refusing to "
            "recompose mixed coverage."
        )
    else:
        cell = None
    return nuc, cell


def _shift_cell_canvas(
    cells_tyx: npt.NDArray[Any], shift_xy: tuple[float, float]
) -> npt.NDArray[Any]:
    """Shift a recomposed cell canvas into the master frame by ``-(x, y)``.

    The per-well alignment maps master→repeat, so a repeat's mask is shifted
    by the negated alignment. ``_translate`` takes a ``(y, x)`` translation and
    zero-fills the vacated strip; applied per timepoint (T=1 for 4i).
    """
    x, y = shift_xy
    trans = (round(-y), round(-x))  # YX, matches create_cell_masks
    return np.stack(
        [_translate(t, t, trans, stacked=False) for t in cells_tyx]
    )


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def build_aligned_zarr(
    master_id: int,
    conn: BlitzGateway,
    *,
    wells: Iterable[str] | None = None,
    omero_conn: Any | None = None,
    keep_all_nuclei: bool = False,
    root: Path | None = None,
) -> Iterator[str]:
    """Build (or extend) the aligned 4i zarr for ``master_id``.

    Generator yielding each well ID after it is written. Writes into the
    isolated ``aligned/`` namespace as ``aligned/plate_<master_id>.zarr`` so it
    never collides with a plain stitched cache of the master plate.

    Args:
        master_id: OMERO plate ID of the master (first) cycle.
        conn: Live OMERO connection.
        wells: Optional subset of well labels; default is every non-empty well.
        omero_conn: Optional ``OmeroConnection`` for per-thread parallel field
            downloads (as in the plain builder). ``None`` → sequential.
        keep_all_nuclei: Debug — keep every cycle's DAPI (renamed per cycle)
            instead of a single master nucleus, so repeat→master registration
            can be eyeballed. Produces a different channel set, so build it to a
            separate ``root``.
        root: Cache namespace to write to; defaults to the aligned namespace.
            Pass a distinct root for debug builds so they don't collide with the
            normal aligned cache's plate metadata.

    Yields:
        Each well position after its aligned well group is written.
    """
    root = root if root is not None else aligned_zarr_root()
    alignments = get_plate_alignments(conn, master_id)
    repeat_ids = [int(p) for p in alignments["plate"].unique()]
    logger.info(
        f"Building aligned zarr for master {master_id} + repeats {repeat_ids}"
    )

    plate_ids = [master_id, *repeat_ids]
    metas = {pid: _fetch_plate_metadata(conn, pid) for pid in plate_ids}
    channel_data = {pid: metas[pid]["channel_data"] for pid in plate_ids}
    nucleus = {
        pid: _nucleus_channel_name(conn, pid, channel_data[pid])
        for pid in plate_ids
    }
    flatfields = {
        pid: _load_flatfield_dict(
            conn, metas[pid]["ff_mask_id"], channel_data[pid], plate_id=pid
        )
        for pid in plate_ids
    }

    plates = {pid: conn.getObject("Plate", pid) for pid in plate_ids}
    if plates[master_id] is None:
        raise ValueError(f"Master plate {master_id} not found in OMERO")
    well_objs = {
        pid: {w.getWellPos(): w for w in plates[pid].listChildren()}
        for pid in plate_ids
    }

    well_map = _fetch_well_map(conn, master_id)
    non_empty = [
        pos
        for pos, info in well_map.items()
        if not is_empty_well(info.get("metadata", {}))
    ]
    all_wells = sorted(non_empty)
    if wells is not None:
        target = [w for w in sorted(wells) if w in non_empty]
    else:
        target = list(all_wells)
    already = set(cached_wells(master_id, root=root))
    target = [w for w in target if w not in already]
    if not target:
        logger.info(f"Aligned zarr for {master_id}: nothing to build")
        return

    pixel_size = metas[master_id]["pixel_size"]
    pixel_size_um = pixel_size[0] if pixel_size else None
    well_meta_map = {
        pos: dict(well_map[pos].get("metadata", {})) for pos in all_wells
    }

    # Channel names are known only after the first well's combine; the writer
    # is created lazily so its plate-level channel metadata is correct.
    writer: PlateZarrWriter | None = None
    try:
        for well_pos in target:
            if well_pos not in well_objs[master_id]:
                logger.warning(
                    f"Well {well_pos} not on master plate {master_id}; skipping"
                )
                continue
            logger.info(f"Assembling aligned well {well_pos}")

            master_img = _stitch_well_image(
                conn,
                well_objs[master_id][well_pos],
                channel_data[master_id],
                flatfields[master_id],
                None,
                omero_conn,
                master_id,
            )
            master_nuc, master_cell = _recompose_well_masks(
                conn, well_objs[master_id][well_pos]
            )
            cycles = [
                CycleCanvas(
                    master_img,
                    list(channel_data[master_id].keys()),
                    nucleus[master_id],
                    master_cell,
                    cycle_tag=str(master_id),
                )
            ]

            for rid in repeat_ids:
                if well_pos not in well_objs[rid]:
                    raise ValueError(
                        f"Well {well_pos} missing on repeat plate {rid}"
                    )
                shift = _well_shift(alignments, rid, well_pos)
                n_fields = len(list(well_objs[rid][well_pos].listChildren()))
                offsets = [(round(-shift[0]), round(-shift[1]))] * n_fields
                rep_img = _stitch_well_image(
                    conn,
                    well_objs[rid][well_pos],
                    channel_data[rid],
                    flatfields[rid],
                    offsets,
                    omero_conn,
                    rid,
                )
                _, rep_cell = _recompose_well_masks(
                    conn, well_objs[rid][well_pos]
                )
                if rep_cell is not None:
                    rep_cell = _shift_cell_canvas(rep_cell, shift)
                cycles.append(
                    CycleCanvas(
                        rep_img,
                        list(channel_data[rid].keys()),
                        nucleus[rid],
                        rep_cell,
                        cycle_tag=str(rid),
                    )
                )

            image_tcyx, channel_names, cells = combine_cycles(
                cycles, keep_all_nuclei=keep_all_nuclei
            )

            if writer is None:
                writer = PlateZarrWriter(
                    plate_id=master_id,
                    plate_name=f"{metas[master_id]['plate_name']} (4i aligned)",
                    channel_names=channel_names,
                    pixel_size_um=pixel_size_um,
                    n_timepoints=1,
                    root=root,
                )
                writer.ensure_plate(
                    all_wells=all_wells, well_metadata=well_meta_map
                )

            writer.write_well(well_pos, image_tcyx, master_nuc, cells)
            yield well_pos
    finally:
        if writer is not None:
            writer.close()

    upsert(
        ZarrPlateEntry(
            plate_id=master_id,
            plate_name=f"{metas[master_id]['plate_name']} (4i aligned)",
            size_bytes=_dir_size(root, master_id),
            n_wells_written=len(target),
        ),
        root=root,
    )
    logger.info(
        f"Finished aligned zarr for {master_id}: {len(target)} well(s)"
    )


def _well_shift(
    alignments: Any, plate_id: int, well: str
) -> tuple[float, float]:
    """Return the ``(x, y)`` per-well alignment for a repeat plate."""
    row = alignments[
        (alignments["plate"] == plate_id) & (alignments["well"] == well)
    ]
    if row.empty:
        raise ValueError(f"Alignment missing for plate {plate_id} well {well}")
    return float(row.iloc[0]["x"]), float(row.iloc[0]["y"])


def _dir_size(root: Any, master_id: int) -> int:
    from omero_screen_napari.zarr_cache.paths import plate_zarr_path

    path = plate_zarr_path(master_id, root=root)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
