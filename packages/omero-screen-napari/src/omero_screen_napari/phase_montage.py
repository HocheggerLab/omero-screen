"""Per-well cell-cycle montage figures from the zarr cache.

A single example cell is not representative of a well: cell-cycle phase drives
nuclear size, DNA content and morphology, so which cell you happen to pick
decides what the figure says. This builds a montage instead -- a few cells from
each phase, one row per cell, so a reader sees the range rather than one draw
from it.

Layout, per page (one page per well)::

    G1     [DAPI+Tub composite, mask outline] [ch3 grey] [ch4 grey] ...
           ...4 cells...
    S      ...
    G2/M   ...
    Poly   ...

Two choices in here are about honesty rather than looks:

* **Cells are drawn at random within each phase**, with the seed stamped on the
  page and every cell's well/label printed under its row. Picking the cell
  closest to each phase's median gives prettier panels, but it selects on the
  outcome and understates real variability. Random panels include the
  out-of-focus and awkwardly segmented cells, because those are in the data.
* **Display limits are computed once per channel for the whole well** and
  applied to every crop. Scaling each crop to its own range would make a G1 and
  a polyploid nucleus equally bright, which is the opposite of what the data
  says.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from loguru import logger

from omero_screen_napari.zarr_cache.crop import (
    fetch_crop,
    fetch_label_crop,
    resolve_to_zarr,
)
from omero_screen_napari.zarr_cache.palette import base_name
from omero_screen_napari.zarr_cache.reader import plate_info, read_well

#: Phases shown, in cell-cycle order. Sub-G1 is excluded by default: it is
#: mostly apoptotic bodies and debris, so it reads as a row of noise rather
#: than a population. Pass it explicitly to include it.
DEFAULT_PHASES: tuple[str, ...] = ("G1", "S", "G2/M", "Polyploid")

#: Channel roles rendered as a colour composite in the first column. Everything
#: else is shown greyscale, which is what a reader can actually compare.
DEFAULT_OVERLAY: tuple[str, ...] = ("dapi", "tub")

#: Composite colours, matching the conventions used elsewhere in the plugin.
_OVERLAY_RGB: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.3, 1.0),  # nuclear -> blue
    (0.0, 1.0, 0.2),  # tubulin -> green
)

_OUTLINE_RGB = (1.0, 1.0, 0.25)


class MontageError(Exception):
    """Raised when a montage cannot be built for a well."""


@dataclass(frozen=True)
class MontageConfig:
    """Settings for a well montage.

    Attributes:
        phases: Phase names to show, in row order.
        cells_per_phase: Rows per phase.
        seed: RNG seed for the per-phase draw. Stamped on the figure so a
            montage can be regenerated exactly.
        crop_um: Crop edge in microns. ``None`` sizes it from the largest cell
            selected, so polyploid cells are not clipped by a constant chosen
            for G1 ones.
        overlay: Channel base names composited in the first column.
        mask: Label layer outlined, falling back to ``"nuclei"`` when a plate
            has no cell masks.
        percentiles: Low/high percentiles for the per-channel display limits.
        exclude_edge: Drop cells whose crop would run off the canvas. This is
            about the crop being incomplete, not about the cell, so it does not
            bias which cells are shown.
    """

    phases: tuple[str, ...] = DEFAULT_PHASES
    cells_per_phase: int = 4
    seed: int = 0
    crop_um: float | None = None
    overlay: tuple[str, ...] = DEFAULT_OVERLAY
    mask: str = "cells"
    percentiles: tuple[float, float] = (0.1, 99.9)
    exclude_edge: bool = True


@dataclass(frozen=True)
class CellRef:
    """One selected cell."""

    phase: str
    well: str
    label: int
    centroid: tuple[float, float]
    timepoint: int = 0
    area: float | None = None


@dataclass
class WellMontage:
    """A built montage, ready to render."""

    plate_id: int
    well: str
    channel_names: list[str]
    overlay_indices: list[int]
    grey_indices: list[int]
    cells: dict[str, list[CellRef]]
    limits: dict[int, tuple[float, float]]
    crop_px: int
    pixel_size_um: float | None
    config: MontageConfig
    missing: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def _phase_column(df: pl.DataFrame) -> str:
    for candidate in ("cell_cycle", "cell_cycle_detailed"):
        if candidate in df.columns:
            return candidate
    raise MontageError(
        "No cell-cycle column in the CellView data for this plate. Run the "
        "cell-cycle analysis, or import the plate into CellView first."
    )


def _first_column(df: pl.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in df.columns), None)


def select_cells(
    df: pl.DataFrame,
    well: str,
    config: MontageConfig,
    canvas_hw: tuple[int, int] | None = None,
    crop_px: int | None = None,
) -> tuple[dict[str, list[CellRef]], list[str]]:
    """Draw ``cells_per_phase`` cells at random from each phase.

    The draw is uniform within a phase and seeded, so the same well always
    yields the same cells. Phases with fewer cells than requested contribute
    what they have and are reported.

    Args:
        df: Measurements for one plate, as loaded from CellView.
        well: Well position.
        config: Montage settings.
        canvas_hw: Well canvas size, used with ``crop_px`` to drop cells whose
            crop would run off the edge.
        crop_px: Crop edge in pixels, for the same check.

    Returns:
        ``({phase: [CellRef, ...]}, warnings)``.

    Raises:
        MontageError: if the well has no measurements or no phase column.
    """
    phase_col = _phase_column(df)
    y_col = _first_column(df, "centroid_y", "centroid-0")
    x_col = _first_column(df, "centroid_x", "centroid-1")
    if y_col is None or x_col is None:
        raise MontageError(
            "No centroid columns in the CellView data for this plate"
        )
    area_col = _first_column(df, "area_cell", "area_nucleus", "area")

    well_df = df.filter(pl.col("well") == well)
    if well_df.is_empty():
        raise MontageError(f"Well {well} has no measurements in CellView")

    if config.exclude_edge and canvas_hw is not None and crop_px is not None:
        half = crop_px / 2
        h, w = canvas_hw
        well_df = well_df.filter(
            (pl.col(y_col) >= half)
            & (pl.col(y_col) <= h - half)
            & (pl.col(x_col) >= half)
            & (pl.col(x_col) <= w - half)
        )

    rng = np.random.default_rng(config.seed)
    selected: dict[str, list[CellRef]] = {}
    warnings: list[str] = []
    for phase in config.phases:
        phase_df = well_df.filter(pl.col(phase_col) == phase)
        n_available = phase_df.height
        if n_available == 0:
            warnings.append(f"{well}: no {phase} cells")
            selected[phase] = []
            continue
        take = min(config.cells_per_phase, n_available)
        if take < config.cells_per_phase:
            warnings.append(
                f"{well}: only {n_available} {phase} cell(s), "
                f"wanted {config.cells_per_phase}"
            )
        # Sort first so the draw does not depend on row order from the DB.
        phase_df = phase_df.sort("label")
        picks = rng.choice(n_available, size=take, replace=False)
        rows = phase_df[sorted(int(p) for p in picks)]
        selected[phase] = [
            CellRef(
                phase=phase,
                well=well,
                label=int(row["label"]),
                centroid=(float(row[y_col]), float(row[x_col])),
                timepoint=int(row.get("timepoint", 0) or 0)
                if isinstance(row, dict)
                else 0,
                area=float(row[area_col]) if area_col else None,
            )
            for row in rows.iter_rows(named=True)
        ]
    return selected, warnings


# ----------------------------------------------------------------------
# Display limits
# ----------------------------------------------------------------------


def channel_limits(
    plate_id: int,
    well: str,
    percentiles: tuple[float, float] = (0.1, 99.9),
) -> dict[int, tuple[float, float]]:
    """Per-channel display limits for a whole well.

    Computed once from the well's smallest pyramid level and applied to every
    crop, so brightness is comparable between phases. Per-crop scaling would
    normalise away exactly the differences the montage exists to show.
    """
    data = read_well(plate_id, well)
    smallest = data["image"][-1]
    n_channels = int(smallest.shape[1])
    limits: dict[int, tuple[float, float]] = {}
    for c in range(n_channels):
        plane = np.asarray(smallest[0, c])
        lo, hi = np.percentile(plane, list(percentiles))
        if hi <= lo:
            hi = lo + 1.0
        limits[c] = (float(lo), float(hi))
    return limits


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------


def _resolve_overlay(
    channel_names: list[str], overlay: tuple[str, ...]
) -> tuple[list[int], list[int]]:
    """Split channels into the composite ones and the greyscale rest.

    Matched on the base name so a cyclic-IF store's ``DAPI_R1`` resolves, and
    only the first match for each role is taken -- a repeated stain belongs in
    the greyscale row, not the composite.
    """
    overlay_indices: list[int] = []
    for wanted in overlay:
        for index, name in enumerate(channel_names):
            if index in overlay_indices:
                continue
            if base_name(name).startswith(wanted.lower()):
                overlay_indices.append(index)
                break
    grey = [i for i in range(len(channel_names)) if i not in overlay_indices]
    return overlay_indices, grey


def _crop_pixels(
    config: MontageConfig,
    cells: dict[str, list[CellRef]],
    pixel_size_um: float | None,
) -> int:
    """Crop edge in pixels, sized so the largest selected cell still fits."""
    if config.crop_um is not None and pixel_size_um:
        return max(16, int(round(config.crop_um / pixel_size_um)))
    areas = [
        cell.area
        for refs in cells.values()
        for cell in refs
        if cell.area is not None and cell.area > 0
    ]
    if not areas:
        return 128
    # A cell of area A has a diameter of roughly 2*sqrt(A/pi); 2.5x that leaves
    # visible surroundings and keeps a polyploid cell inside the frame.
    diameter = 2.0 * np.sqrt(max(areas) / np.pi)
    return int(np.clip(round(diameter * 2.5 / 2) * 2, 32, 1024))


def build_montage(
    plate_id: int,
    well: str,
    df: pl.DataFrame,
    config: MontageConfig | None = None,
) -> WellMontage:
    """Select cells and resolve everything needed to render one well.

    Does no drawing, so it can be tested without matplotlib.

    Raises:
        MontageError: if the plate has no zarr store, or the well cannot be
            selected from.
    """
    config = config or MontageConfig()
    if resolve_to_zarr(plate_id) is None:
        raise MontageError(
            f"Plate {plate_id} has no zarr cache. Build one first "
            f"(Cache Plate in the plate-info dialog, or zarr_cache_build.py)."
        )
    info = plate_info(plate_id)
    channel_names: list[str] = list(info["channel_names"])
    pixel_size_um = info.get("pixel_size_um")

    well_data = read_well(plate_id, well)
    canvas = np.asarray(well_data["image"][0].shape[-2:])

    # Two passes: an initial selection to size the crop from real cell areas,
    # then a re-selection with the edge filter applied at that size.
    provisional, _ = select_cells(df, well, config)
    crop_px = _crop_pixels(config, provisional, pixel_size_um)
    cells, warnings = select_cells(
        df,
        well,
        config,
        canvas_hw=(int(canvas[0]), int(canvas[1])),
        crop_px=crop_px,
    )

    overlay_indices, grey_indices = _resolve_overlay(
        channel_names, config.overlay
    )
    return WellMontage(
        plate_id=plate_id,
        well=well,
        channel_names=channel_names,
        overlay_indices=overlay_indices,
        grey_indices=grey_indices,
        cells=cells,
        limits=channel_limits(plate_id, well, config.percentiles),
        crop_px=crop_px,
        pixel_size_um=pixel_size_um,
        config=config,
        missing=warnings,
    )


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _normalise(
    plane: npt.NDArray[Any], limits: tuple[float, float]
) -> npt.NDArray[np.float32]:
    lo, hi = limits
    out = (plane.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    clipped: npt.NDArray[np.float32] = np.clip(out, 0.0, 1.0)
    return clipped


def _composite(
    crop: npt.NDArray[Any],
    indices: list[int],
    limits: dict[int, tuple[float, float]],
) -> npt.NDArray[np.float32]:
    """Additive RGB composite of the overlay channels."""
    h, w = crop.shape[-2:]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for slot, channel in enumerate(indices):
        colour = _OVERLAY_RGB[slot % len(_OVERLAY_RGB)]
        norm = _normalise(crop[channel], limits[channel])
        for band in range(3):
            rgb[..., band] += norm * colour[band]
    return np.clip(rgb, 0.0, 1.0)


def _outline(mask: npt.NDArray[Any], label: int) -> npt.NDArray[np.bool_]:
    """Boundary of the target cell only, so neighbours are not outlined."""
    from skimage.segmentation import find_boundaries

    target = mask == label
    if not target.any():
        # The centroid can land on a neighbouring label after alignment;
        # outlining nothing is better than outlining the wrong cell.
        return np.zeros(mask.shape, dtype=bool)
    boundary: npt.NDArray[np.bool_] = find_boundaries(target, mode="outer")
    return boundary


def render_montage(montage: WellMontage) -> Any:
    """Draw the montage and return the matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cfg = montage.config
    rows = [
        (phase, cell)
        for phase in cfg.phases
        for cell in montage.cells.get(phase, [])
    ]
    if not rows:
        raise MontageError(
            f"Well {montage.well}: no cells in any of {list(cfg.phases)}"
        )
    n_cols = 1 + len(montage.grey_indices)
    fig, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(1.35 * n_cols, 1.35 * len(rows)),
        squeeze=False,
    )

    mask_name = montage.config.mask
    for row_index, (phase, cell) in enumerate(rows):
        crop = fetch_crop(
            montage.plate_id,
            montage.well,
            cell.label,
            centroid=cell.centroid,
            size=montage.crop_px,
            t=cell.timepoint,
        )
        try:
            mask = fetch_label_crop(
                montage.plate_id,
                montage.well,
                centroid=cell.centroid,
                size=montage.crop_px,
                t=cell.timepoint,
                mask_name=mask_name,
            )
        except KeyError:
            mask = fetch_label_crop(
                montage.plate_id,
                montage.well,
                centroid=cell.centroid,
                size=montage.crop_px,
                t=cell.timepoint,
                mask_name="nuclei",
            )
        outline = _outline(mask, cell.label)

        rgb = _composite(crop, montage.overlay_indices, montage.limits)
        rgb[outline] = _OUTLINE_RGB
        ax = axes[row_index][0]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_ylabel(
            f"{phase}\n{montage.well} · {cell.label}",
            fontsize=6,
            rotation=0,
            ha="right",
            va="center",
            labelpad=22,
        )
        if row_index == 0:
            overlay_label = " + ".join(
                montage.channel_names[i] for i in montage.overlay_indices
            )
            ax.set_title(overlay_label, fontsize=6)

        for col, channel in enumerate(montage.grey_indices, start=1):
            gax = axes[row_index][col]
            gax.imshow(
                _normalise(crop[channel], montage.limits[channel]),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            if row_index == 0:
                gax.set_title(montage.channel_names[channel], fontsize=6)

        for ax_ in axes[row_index]:
            ax_.set_xticks([])
            ax_.set_yticks([])
            for spine in ax_.spines.values():
                spine.set_visible(False)

    # Scale bar on the last panel of the last row.
    if montage.pixel_size_um:
        bar_um = 20.0
        bar_px = bar_um / montage.pixel_size_um
        if bar_px < montage.crop_px * 0.9:
            ax = axes[-1][-1]
            y = montage.crop_px * 0.92
            x = montage.crop_px * 0.06
            ax.add_patch(
                Rectangle(
                    (x, y),
                    bar_px,
                    max(montage.crop_px * 0.015, 1.0),
                    color="white",
                )
            )
            ax.text(
                x,
                y - montage.crop_px * 0.03,
                f"{bar_um:.0f} µm",
                color="white",
                fontsize=5,
                va="bottom",
            )

    subtitle = (
        f"plate {montage.plate_id} · well {montage.well} · "
        f"{cfg.cells_per_phase} random cells per phase · seed {cfg.seed} · "
        f"crop {montage.crop_px}px"
    )
    if montage.pixel_size_um:
        subtitle += f" ({montage.crop_px * montage.pixel_size_um:.0f} µm)"
    fig.suptitle(subtitle, fontsize=7)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig


def export_well_pdf(
    plate_id: int,
    well: str,
    df: pl.DataFrame,
    out_dir: Path,
    config: MontageConfig | None = None,
) -> Path:
    """Build and write one well's montage as a vector PDF.

    Returns:
        The path written.
    """
    montage = build_montage(plate_id, well, df, config)
    for warning in montage.missing:
        logger.warning(warning)
    fig = render_montage(montage)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"plate{plate_id}_{well}_phase_montage.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    _close(fig)
    logger.info(f"Wrote {path}")
    return path


def _close(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def load_plate_measurements(plate_id: int) -> pl.DataFrame:
    """Load a plate's CellView measurements as an eager DataFrame.

    Raises:
        MontageError: if the plate is not in CellView.
    """
    from omero_screen_napari.plate_cache import _load_plate_data_from_cellview

    lf = _load_plate_data_from_cellview(plate_id)
    df = lf.collect()
    if df.is_empty():
        raise MontageError(
            f"Plate {plate_id} has no measurements in CellView. Import it "
            f"with 'cellview import plate {plate_id}' first."
        )
    return df
