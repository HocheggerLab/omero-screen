"""Incremental NGFF writer for stitched plates.

One :class:`PlateZarrWriter` per plate. Plate-level metadata (rows,
columns, full well list) is written once via :meth:`ensure_plate`; each
subsequent :meth:`write_well` adds one well group without touching its
siblings.

Crash safety: each well is written into ``plate_<id>.zarr.tmp/<row>/<col>/``
first, then renamed into the final ``plate_<id>.zarr/<row>/<col>/`` location.
The rename is atomic on POSIX same-filesystem moves.

Axes / chunking decisions are documented in the Stage 2 plan note.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray
from ome_zarr.scale import Scaler
from ome_zarr.writer import (
    write_image,
    write_labels,
    write_plate_metadata,
    write_well_metadata,
)

from omero_screen_napari.zarr_cache.paths import (
    plate_zarr_path,
    plate_zarr_tmp_path,
)

logger = logging.getLogger(__name__)


# Chunking constants. See Stage 2 plan for rationale: 16-frame T-blocks
# keep live-cell scrub-within-block fast while staying small enough that
# single-cell crops only over-read modestly. 256 spatial slab is a balance
# between number of chunks (avoid inode pressure) and over-read on crops.
_T_BLOCK = 16
_SPATIAL_CHUNK = 256


# Default contrast palette for the omero.channels metadata block. Tuned for
# typical 4-channel screens (DAPI, cytoplasm, EdU, H3P) — viewer auto-contrast
# fills in the rest. Keep in OMERO-NGFF "RRGGBB" hex (no #).
_DEFAULT_CHANNEL_COLORS = [
    "0000FF",  # blue
    "00FF00",  # green
    "FF0000",  # red
    "FF00FF",  # magenta
    "FFFF00",  # yellow
    "00FFFF",  # cyan
]


def _channel_window(arr_yx: NDArray[Any]) -> dict[str, float]:
    """Compute a sensible contrast window from one timepoint's channel data.

    Uses 0.1–99.9 percentiles to ignore saturation and dark outliers.
    ``min`` / ``max`` carry the data extrema; ``start`` / ``end`` are the
    suggested display limits.
    """
    flat = arr_yx.ravel()
    # Subsample for speed on a 3232² canvas.
    if flat.size > 1_000_000:
        idx = np.random.default_rng(0).choice(
            flat.size, 1_000_000, replace=False
        )
        flat = flat[idx]
    p_lo, p_hi = np.percentile(flat, [0.1, 99.9])
    return {
        "min": float(arr_yx.min()),
        "max": float(arr_yx.max()),
        "start": float(p_lo),
        "end": float(p_hi),
    }


def _split_well(well: str) -> tuple[str, str]:
    """Split an OMERO well label like 'A1' into ('A', '1'). Pads single-digit
    column to the bare int (matches OME-NGFF convention)."""
    row = well[0]
    col = str(int(well[1:]))
    return row, col


def _image_chunks(t: int) -> tuple[int, int, int, int]:
    return (min(t, _T_BLOCK), 1, _SPATIAL_CHUNK, _SPATIAL_CHUNK)


def _label_chunks(t: int) -> tuple[int, int, int]:
    return (min(t, _T_BLOCK), _SPATIAL_CHUNK, _SPATIAL_CHUNK)


def _coord_transforms(
    pixel_size_um: float | None,
    frame_interval_s: float | None,
    n_levels: int,
    *,
    has_channel: bool,
) -> list[list[dict[str, Any]]]:
    """Build NGFF ``coordinateTransformations`` per pyramid level.

    Y/X scaled by physical pixel size and doubled at each level; T by
    frame interval; C (when present) is identity.
    """
    transforms: list[list[dict[str, Any]]] = []
    px = float(pixel_size_um) if pixel_size_um else 1.0
    dt = float(frame_interval_s) if frame_interval_s else 1.0
    for level in range(n_levels):
        factor = 2**level
        if has_channel:
            scale = [dt, 1.0, px * factor, px * factor]
        else:
            scale = [dt, px * factor, px * factor]
        transforms.append([{"type": "scale", "scale": scale}])
    return transforms


class PlateZarrWriter:
    """Stateful writer for one plate.zarr store.

    Usage::

        writer = PlateZarrWriter(
            plate_id=1234,
            plate_name="MyPlate",
            channel_names=["DAPI", "Tub", "EdU"],
            pixel_size_um=0.65,
            n_timepoints=1,
        )
        writer.ensure_plate(all_wells=["A1", "A2", "B1"])
        writer.write_well("A1", img_tcyx, nuc_tyx, cell_tyx)
        writer.close()
    """

    PYRAMID_LEVELS = 3
    """Number of multiscale levels (0 = full res, 2 = 4x downsampled YX)."""

    def __init__(
        self,
        plate_id: int,
        plate_name: str,
        channel_names: list[str],
        pixel_size_um: float | None,
        n_timepoints: int,
        frame_interval_s: float | None = None,
    ) -> None:
        self.plate_id = plate_id
        self.plate_name = plate_name
        self.channel_names = channel_names
        self.pixel_size_um = pixel_size_um
        self.n_timepoints = n_timepoints
        self.frame_interval_s = frame_interval_s

        self.path = plate_zarr_path(plate_id)
        self.tmp_path = plate_zarr_tmp_path(plate_id)

    # ------------------------------------------------------------------
    # Plate-level setup
    # ------------------------------------------------------------------

    def ensure_plate(
        self,
        all_wells: Iterable[str],
        well_metadata: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Create the plate group and write plate-level NGFF metadata.

        Idempotent: a no-op if the plate has already been initialised.
        Pass the **complete** OMERO well list at first call; later
        :meth:`write_well` calls validate membership.

        Args:
            all_wells: Every well advertised on the OMERO plate, even
                ones we won't write yet. NGFF requires the row/column
                set up front.
            well_metadata: Optional per-well annotations to bake into
                ``omero_screen.well_metadata`` — typically
                ``{"A1": {"cell_line": "RPE", "condition": "ctrl", ...}}``.
                The load path uses this for the on-canvas overlay so it
                does not need an OMERO connection at view time.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(self.path), mode="a")
        if "plate" in root.attrs:
            return
        wells = sorted(all_wells)
        if not wells:
            raise ValueError("all_wells must be non-empty")
        rows = sorted({w[0] for w in wells})
        cols = sorted({int(w[1:]) for w in wells})
        col_strs = [str(c) for c in cols]
        well_dicts = [
            {
                "path": f"{w[0]}/{int(w[1:])}",
                "rowIndex": rows.index(w[0]),
                "columnIndex": cols.index(int(w[1:])),
            }
            for w in wells
        ]
        write_plate_metadata(
            root, rows, col_strs, well_dicts, name=self.plate_name
        )
        # Stash channel + pixel-size hints at plate level for downstream
        # consumers that don't want to walk into every well.
        root.attrs["omero_screen"] = {
            "plate_id": self.plate_id,
            "channel_names": list(self.channel_names),
            "pixel_size_um": self.pixel_size_um,
            "n_timepoints": self.n_timepoints,
            "frame_interval_s": self.frame_interval_s,
            # Per-well annotations baked into the store so the napari
            # load path does not need an OMERO connection.
            "well_metadata": dict(well_metadata or {}),
        }

    # ------------------------------------------------------------------
    # Per-well write
    # ------------------------------------------------------------------

    def write_well(
        self,
        well: str,
        image_tcyx: NDArray[Any],
        label_nuclei_tyx: NDArray[Any],
        label_cells_tyx: NDArray[Any] | None = None,
    ) -> None:
        """Write one well's stitched image and label arrays.

        ``image_tcyx``: shape ``(T, C, Y, X)``, any numeric dtype. For
        fixed-cell assays ``T = 1``.

        ``label_nuclei_tyx`` / ``label_cells_tyx``: shape ``(T, Y, X)``,
        ``uint32`` (or any integer that round-trips through uint32).
        """
        if image_tcyx.ndim != 4:
            raise ValueError(
                f"image_tcyx must be 4-D (T,C,Y,X); got shape {image_tcyx.shape}"
            )
        if label_nuclei_tyx.ndim != 3:
            raise ValueError(
                f"label_nuclei_tyx must be 3-D (T,Y,X); got shape {label_nuclei_tyx.shape}"
            )
        if label_cells_tyx is not None and label_cells_tyx.ndim != 3:
            raise ValueError(
                f"label_cells_tyx must be 3-D (T,Y,X); got shape {label_cells_tyx.shape}"
            )

        # Validate well is advertised in the plate metadata.
        root = zarr.open_group(str(self.path), mode="a")
        advertised = {w["path"] for w in root.attrs["plate"]["wells"]}
        row, col = _split_well(well)
        well_key = f"{row}/{col}"
        if well_key not in advertised:
            raise ValueError(
                f"Well {well} (key {well_key!r}) not advertised in plate "
                f"metadata. Did you call ensure_plate with the full well list?"
            )

        t = image_tcyx.shape[0]
        nuc = label_nuclei_tyx.astype(np.uint32, copy=False)
        cell = (
            label_cells_tyx.astype(np.uint32, copy=False)
            if label_cells_tyx is not None
            else None
        )

        # Stage write to a temp directory, then rename into place. This
        # avoids leaving a half-written well group inside the live store
        # if the process is killed mid-write.
        tmp_well_dir = self.tmp_path / row / col
        final_well_dir = self.path / row / col

        if tmp_well_dir.exists():
            shutil.rmtree(tmp_well_dir)
        tmp_well_dir.mkdir(parents=True, exist_ok=True)

        well_grp = zarr.open_group(str(tmp_well_dir), mode="w")
        write_well_metadata(well_grp, [{"path": "0"}])
        img_grp = well_grp.require_group("0")

        img_scaler = Scaler(
            downscale=2, max_layer=self.PYRAMID_LEVELS - 1, method="nearest"
        )
        label_scaler = Scaler(
            downscale=2,
            max_layer=self.PYRAMID_LEVELS - 1,
            labeled=True,
            method="nearest",
        )

        img_transforms = _coord_transforms(
            self.pixel_size_um,
            self.frame_interval_s,
            self.PYRAMID_LEVELS,
            has_channel=True,
        )
        lbl_transforms = _coord_transforms(
            self.pixel_size_um,
            self.frame_interval_s,
            self.PYRAMID_LEVELS,
            has_channel=False,
        )

        write_image(
            image_tcyx,
            img_grp,
            axes="tcyx",
            chunks=_image_chunks(t),
            scaler=img_scaler,
            coordinate_transformations=img_transforms,
        )

        # NGFF requires the ``omero`` block as a sibling of ``multiscales``
        # at the image-group level, not nested inside it. ``write_image``'s
        # ``**metadata`` puts kwargs inside ``multiscales[0]`` which
        # napari-ome-zarr ignores. Write it directly on the group attrs.
        omero_channels = []
        for c, name in enumerate(self.channel_names):
            window = _channel_window(image_tcyx[0, c])
            omero_channels.append(
                {
                    "label": name,
                    "color": _DEFAULT_CHANNEL_COLORS[
                        c % len(_DEFAULT_CHANNEL_COLORS)
                    ],
                    "active": True,
                    "window": window,
                }
            )
        img_grp.attrs["omero"] = {
            "channels": omero_channels,
            "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color"},
            "version": "0.4",
        }
        write_labels(
            nuc,
            img_grp,
            name="nuclei",
            axes="tyx",
            chunks=_label_chunks(t),
            scaler=label_scaler,
            coordinate_transformations=lbl_transforms,
        )
        if cell is not None:
            write_labels(
                cell,
                img_grp,
                name="cells",
                axes="tyx",
                chunks=_label_chunks(t),
                scaler=label_scaler,
                coordinate_transformations=lbl_transforms,
            )

        # Atomic-ish swap. If the well already exists (re-write case), we
        # move the old one aside, install the new, then clean up.
        final_well_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_well_dir.exists():
            backup = final_well_dir.with_name(final_well_dir.name + ".old")
            if backup.exists():
                shutil.rmtree(backup)
            os.rename(final_well_dir, backup)
            try:
                os.rename(tmp_well_dir, final_well_dir)
            except OSError:
                os.rename(backup, final_well_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.rename(tmp_well_dir, final_well_dir)

        logger.info("Wrote well %s to %s", well, final_well_dir)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Remove the staging directory if it exists."""
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path, ignore_errors=True)

    # Context-manager sugar
    def __enter__(self) -> PlateZarrWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        self.close()
