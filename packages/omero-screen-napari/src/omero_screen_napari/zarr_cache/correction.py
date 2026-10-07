"""Track correction of cached wells, and its storage beside the zarr cache.

The cached nucleus masks are the first tracking pass (pixel value = first-pass
track id) and are never rewritten. A correction
(:func:`omero_screen.track_correction.correct_tracks`) is a table that maps
every first-pass nucleus to its final track; it is stored beside the store::

    zarr/
    ├── plate_5054.zarr/
    └── plate_5054.corrections/
        ├── C2.table.parquet      timepoint, label, track_id (0 = hidden)
        ├── C2.lineage.parquet    track_id, parent_track_id (final)
        ├── C2.events.parquet     repair decisions of both passes
        ├── C2.pass1.parquet      first-pass lineage the correction started from
        └── C2.debris.json        first-pass track ids hidden as debris

Readers apply the table when a frame is read (:func:`corrected_frame`), so the
viewer can show corrected and first-pass nuclei side by side, and
:func:`measure_cached_well` measures the corrected nuclei with the pipeline's
own feature extraction for import into CellView.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from omero_screen.track_correction import (
    CorrectionParams,
    CorrectionResult,
    correct_tracks,
    measure_corrected_well,
    measure_labels,
    relabel_frame,
)

from omero_screen_napari.zarr_cache.paths import plate_zarr_path
from omero_screen_napari.zarr_cache.reader import read_well

if TYPE_CHECKING:
    from omero.gateway import BlitzGateway
    from trackastra.model import Trackastra


def correction_dir(plate_id: int) -> Path:
    """Directory holding the plate's corrections, beside its zarr store."""
    return plate_zarr_path(plate_id).with_suffix(".corrections")


def has_correction(plate_id: int, well: str) -> bool:
    """Whether a correction is stored for the well."""
    return (correction_dir(plate_id) / f"{well}.table.parquet").exists()


def save_correction(
    plate_id: int,
    well: str,
    result: CorrectionResult,
    pass1: dict[int, int],
) -> Path:
    """Store a correction and the first-pass lineage it started from."""
    out = correction_dir(plate_id)
    out.mkdir(parents=True, exist_ok=True)
    result.table.to_parquet(out / f"{well}.table.parquet", index=False)
    pd.DataFrame(
        {
            "track_id": list(result.parents),
            "parent_track_id": list(result.parents.values()),
        }
    ).to_parquet(out / f"{well}.lineage.parquet", index=False)
    result.events.to_parquet(out / f"{well}.events.parquet", index=False)
    _lineage_frame(pass1).to_parquet(
        out / f"{well}.pass1.parquet", index=False
    )
    (out / f"{well}.debris.json").write_text(json.dumps(sorted(result.debris)))
    return out


def load_correction(plate_id: int, well: str) -> CorrectionResult | None:
    """The stored correction of a well, or ``None``."""
    d = correction_dir(plate_id)
    if not has_correction(plate_id, well):
        return None
    lineage = pd.read_parquet(d / f"{well}.lineage.parquet")
    events_path = d / f"{well}.events.parquet"
    debris_path = d / f"{well}.debris.json"
    return CorrectionResult(
        table=pd.read_parquet(d / f"{well}.table.parquet"),
        parents=dict(
            zip(
                lineage["track_id"].astype(int),
                lineage["parent_track_id"].astype(int),
                strict=True,
            )
        ),
        events=pd.read_parquet(events_path)
        if events_path.exists()
        else pd.DataFrame(),
        debris=set(json.loads(debris_path.read_text()))
        if debris_path.exists()
        else set(),
    )


def _lineage_frame(parents: dict[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": list(parents),
            "parent_track_id": list(parents.values()),
        }
    )


def first_pass_lineage(plate_id: int, well: str) -> dict[int, int]:
    """First-pass parent map of a well.

    Read from the corrections directory once stored; otherwise from CellView,
    whose ``track_id_raw`` / ``parent_track_id_raw`` hold the tracker output
    the cached masks carry, as long as the plate was imported before any
    correction.
    """
    stored = correction_dir(plate_id) / f"{well}.pass1.parquet"
    if stored.exists():
        df = pd.read_parquet(stored)
        return dict(
            zip(
                df["track_id"].astype(int),
                df["parent_track_id"].astype(int),
                strict=True,
            )
        )
    from cellview.api import cellview_load_data

    data, _ = cellview_load_data(plate_id)
    data = data[data["well"] == well]
    if "track_id_raw" not in data:
        raise ValueError(f"Plate {plate_id} {well} has no tracks in CellView.")
    parents = (
        data.groupby("track_id_raw")["parent_track_id_raw"]
        .max()
        .fillna(0)
        .astype(int)
    )
    return dict(
        zip(
            parents.index.astype(int), parents.to_numpy(dtype=int), strict=True
        )
    )


@dataclass
class CachedChannel:
    """One channel of a cached well as a ``(T, Y, X)`` stack."""

    image: Any  # zarr array (T, C, Y, X)
    index: int

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(T, Y, X)``."""
        t, _, y, x = self.image.shape
        return (int(t), int(y), int(x))

    def __getitem__(self, t: int) -> npt.NDArray[Any]:
        """Frame ``t``."""
        return np.asarray(self.image[t, self.index])


def cached_channels(plate_id: int, well: str) -> dict[str, CachedChannel]:
    """Every channel of a cached well at full resolution, by name."""
    data = read_well(plate_id, well)
    return {
        name: CachedChannel(data["image"][0], i)
        for i, name in enumerate(data["channel_names"])
    }


def _channel(names: list[str], *keys: str) -> str | None:
    """First channel name containing one of ``keys`` (case-insensitive)."""
    for key in keys:
        for name in names:
            if key in name.lower():
                return name
    return None


def correct_cached_well(
    plate_id: int,
    well: str,
    model: Trackastra,
    params: CorrectionParams | None = None,
    nucleus_channel: str | None = None,
) -> CorrectionResult:
    """Correct and re-track one cached well, and store the result.

    Args:
        plate_id: OMERO plate id.
        well: Well, e.g. ``"C2"``.
        model: Trackastra model (the pipeline's, for a like-for-like second
            pass).
        params: Correction settings.
        nucleus_channel: Nucleus channel name; default: the one whose name
            contains ``nucleus``, ``dapi``, ``hoechst`` or ``dna``.
    """
    channels = cached_channels(plate_id, well)
    names = list(channels)
    nucleus_channel = nucleus_channel or _channel(
        names, "nucleus", "dapi", "hoechst", "dna"
    )
    if nucleus_channel is None:
        raise ValueError(f"No nucleus channel among {names}.")
    signals = {
        key: channels[name]
        for key in ("geminin", "pip")
        if (name := _channel(names, key)) is not None
    }
    masks = read_well(plate_id, well)["nuclei"][0]
    pass1 = first_pass_lineage(plate_id, well)
    logger.info(f"{well}: measuring first-pass nuclei")
    det = measure_labels(masks, signals)
    result = correct_tracks(
        channels[nucleus_channel], masks, pass1, signals, model, params, det
    )
    out = save_correction(plate_id, well, result, pass1)
    logger.info(f"{well}: correction stored in {out}")
    return result


def corrected_frame(
    plate_id: int, well: str, t: int, result: CorrectionResult | None = None
) -> npt.NDArray[np.uint32]:
    """Frame ``t`` of a cached well's nucleus mask with the correction applied."""
    result = result or load_correction(plate_id, well)
    if result is None:
        raise FileNotFoundError(f"No correction stored for {plate_id} {well}.")
    masks = read_well(plate_id, well)["nuclei"][0]
    return relabel_frame(np.asarray(masks[t]), result.frame_map(t))


def _field_geometry(
    well: Any,
) -> tuple[list[int], npt.NDArray[np.int_], int, int]:
    """Field image ids, canvas offsets and tile size, as the pipeline has them."""
    from omero_screen_napari.zarr_cache.builder import _load_canvas_offsets

    image_ids = [int(ws.getImage().getId()) for ws in well.listChildren()]
    offsets = _load_canvas_offsets(well)
    first = well.getWellSample(0).getImage()
    return image_ids, offsets, int(first.getSizeY()), int(first.getSizeX())


def measure_cached_well(
    conn: BlitzGateway,
    metadata: Any,
    plate_id: int,
    well_pos: str,
    result: CorrectionResult | None = None,
) -> pd.DataFrame:
    """Measure a cached well's corrected nuclei as the pipeline would.

    Pixels come from the zarr cache (flatfield-corrected, stored as uint16);
    metadata, field ids and canvas offsets from OMERO.

    Args:
        conn: OMERO connection.
        metadata: The plate's ``MetadataParser`` (after ``manage_metadata``).
        plate_id: OMERO plate id.
        well_pos: Well, e.g. ``"C2"``.
        result: The correction; default: the stored one.

    Returns:
        The well's rows for ``final_data.csv``.
    """
    result = result or load_correction(plate_id, well_pos)
    if result is None:
        raise FileNotFoundError(
            f"No correction stored for {plate_id} {well_pos}."
        )
    plate = conn.getObject("Plate", plate_id)
    well = next(w for w in plate.listChildren() if w.getWellPos() == well_pos)
    cached = cached_channels(plate_id, well_pos)
    order = list(metadata.channel_data)
    missing = [ch for ch in order if ch not in cached]
    if missing:
        raise ValueError(
            f"Channels {missing} of the plate metadata are not in the cache "
            f"({list(cached)})."
        )
    data = read_well(plate_id, well_pos)
    image_ids, offsets, tile_h, tile_w = _field_geometry(well)
    return measure_corrected_well(
        well,
        metadata,
        {ch: cached[ch] for ch in order},
        data["nuclei"][0],
        data["cells"][0] if data["cells"] else None,
        result,
        nucleus_channel=metadata.channel_roles["nucleus"],
        cell_channel=metadata.channel_roles.get("cell"),
        field_image_ids=image_ids,
        field_offsets=offsets,
        tile_h=tile_h,
        tile_w=tile_w,
    )


def correction_lut(
    result: CorrectionResult, n_frames: int
) -> npt.NDArray[np.uint32]:
    """Per-frame lookup table: ``lut[t, first-pass label] = final track id``."""
    table = result.table
    width = int(table["label"].max()) + 1 if len(table) else 1
    lut = np.zeros((n_frames, width), np.uint32)
    lut[
        table["timepoint"].to_numpy(dtype=int),
        table["label"].to_numpy(dtype=int),
    ] = table["track_id"].to_numpy(dtype=np.uint32)
    return lut


def corrected_pyramid(
    levels: list[Any], result: CorrectionResult
) -> list[Any]:
    """Lazy corrected copies of a ``(T, Y, X)`` nucleus label pyramid.

    Every level holds the same label values (downsampled by picking), so one
    per-frame table relabels them all; nothing is computed until napari reads
    a tile.
    """
    import dask.array as da

    lut = correction_lut(result, int(levels[0].shape[0]))
    width = lut.shape[1]

    def relabel(
        block: npt.NDArray[Any], block_info: Any = None
    ) -> npt.NDArray[Any]:
        t0 = block_info[0]["array-location"][0][0]
        out = np.zeros(block.shape, np.uint32)
        for i in range(block.shape[0]):
            frame = block[i]
            inside = frame < width
            out[i][inside] = lut[t0 + i][frame[inside]]
        return out

    pyramid: list[Any] = []
    for lv in levels:
        arr: Any = lv if isinstance(lv, da.Array) else da.from_zarr(lv)  # type: ignore[no-untyped-call]
        pyramid.append(arr.map_blocks(relabel, dtype=np.uint32))
    return pyramid
