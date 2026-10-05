"""Render whole-well overviews: a stitched, captioned multichannel composite.

The headless counterpart of the napari well view. For each requested well a
*view* (the whole well, or a 2x, 4x, ... zoom around a centre) is read at a
resolution close to the requested output size, composed additively in the
napari channel colours, optionally overlaid with nuclei / cell mask
outlines, and written with a caption and a scale bar.

Pixels come from the plate's zarr pyramid when it is cached (the coarsest
level that still gives the output size), otherwise from the per-field images
stitched in memory with the plate's stitch calibration, as the welldata
widget does.

Display limits are shared by every well of one call: 0.1/99.9 percentiles
pooled over the views being rendered, unless given explicitly. So wells
rendered together are directly comparable, and the limits used are recorded.

Layers are named: plate channel names plus ``nuclei_masks`` and
``cell_masks``. By default every channel and no mask is drawn.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Mask layer name → label set in the zarr store / column in per-field labels.
MASK_LAYERS: dict[str, str] = {"nuclei_masks": "nuclei", "cell_masks": "cells"}
#: Outline colours, chosen to stay visible over the channel colours.
MASK_HEX: dict[str, str] = {"nuclei": "FFFFFF", "cells": "FFD700"}
#: Per-field label stacks hold nuclei in channel 0 and cells in channel 1.
_FIELD_LABEL_CHANNEL = {"nuclei": 0, "cells": 1}
DEFAULT_SIZE = 2000
_SAMPLES_PER_WELL = 1_000_000
_UINT16_MAX = 65535


class OverviewError(ValueError):
    """An overview cannot be produced for the requested well or layers."""


# ---------------------------------------------------------------------------
# View planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class View:
    """A region of the level-0 canvas and how it is read.

    ``y0:y1, x0:x1`` are level-0 pixel bounds. The region is read from
    pyramid ``level`` (2**level downsampled) taking every ``step``-th pixel,
    so one output pixel covers ``2**level * step`` level-0 pixels.
    """

    y0: int
    y1: int
    x0: int
    x1: int
    level: int
    step: int

    @property
    def downsample(self) -> int:
        """Level-0 pixels per output pixel along each axis."""
        return int(2**self.level * self.step)


def plan_view(
    shape_yx: tuple[int, int],
    n_levels: int,
    *,
    zoom: int = 1,
    center: tuple[float, float] = (0.5, 0.5),
    size: int = DEFAULT_SIZE,
) -> View:
    """Choose the region and read resolution for one overview.

    Args:
        shape_yx: Level-0 canvas shape.
        n_levels: Pyramid levels available (1 for an in-memory canvas).
        zoom: 1 for the whole well; each doubling halves the field of view.
        center: View centre as fractions of the well height and width. The
            view is shifted inwards when it would leave the canvas.
        size: Target output size in pixels along the longer side; the
            output is at most this large.

    Raises:
        OverviewError: ``zoom`` is not a power of two, or ``center`` /
            ``size`` are out of range.
    """
    if zoom < 1 or zoom & (zoom - 1):
        raise OverviewError(f"zoom must be 1, 2, 4, 8, ..., got {zoom}")
    if not all(0.0 <= c <= 1.0 for c in center):
        raise OverviewError(
            f"center must be fractions in [0, 1], got {center}"
        )
    if size < 1:
        raise OverviewError(f"size must be positive, got {size}")

    height, width = shape_yx
    view_h = max(1, math.ceil(height / zoom))
    view_w = max(1, math.ceil(width / zoom))
    y0 = _clamp(round(center[0] * height) - view_h // 2, 0, height - view_h)
    x0 = _clamp(round(center[1] * width) - view_w // 2, 0, width - view_w)

    # Smallest total downsample that fits ``size``, preferring the coarser
    # level on a tie (less to read): a 5378 px well at 2000 px reads level 0
    # every 3rd block (1793 px), not level 1 every 2nd (1345 px).
    needed = max(1, math.ceil(max(view_h, view_w) / size))
    level, step = min(
        ((lvl, math.ceil(needed / 2**lvl)) for lvl in range(max(n_levels, 1))),
        key=lambda ls: (2 ** ls[0] * ls[1], -ls[0]),
    )
    return View(y0, y0 + view_h, x0, x0 + view_w, level, step)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


# ---------------------------------------------------------------------------
# Pixel sources
# ---------------------------------------------------------------------------


class WellPixels(Protocol):
    """Read access to one well's stitched canvas."""

    shape_yx: tuple[int, int]
    n_levels: int

    def read(self, view: View, channels: list[int]) -> np.ndarray[Any, Any]:
        """Return ``(C, y, x)`` raw intensities for ``view``."""
        ...

    def read_mask(self, view: View, name: str) -> np.ndarray[Any, Any] | None:
        """Return ``(y, x)`` labels for mask ``name``, or None if absent."""
        ...


def _level_slice(view: View, *, strided: bool = True) -> tuple[slice, slice]:
    """The view's ``(y, x)`` slices in its pyramid level's coordinates.

    Strided slices take every ``step``-th pixel (used for labels, which must
    not be averaged); unstrided ones the full region for :func:`_block_mean`.
    """
    f = 2**view.level
    step = view.step if strided else 1
    return (
        slice(view.y0 // f, math.ceil(view.y1 / f), step),
        slice(view.x0 // f, math.ceil(view.x1 / f), step),
    )


def _block_mean(
    plane: np.ndarray[Any, Any], step: int
) -> np.ndarray[Any, Any]:
    """Downsample a ``(y, x)`` plane by averaging ``step`` x ``step`` blocks.

    Averaging rather than striding avoids aliasing (sparkle) in the
    overview. Edge blocks are padded by repetition, so the output has the
    same shape as the strided slice of the same region.
    """
    if step == 1:
        return plane
    h, w = plane.shape
    ph, pw = -h % step, -w % step
    padded = np.pad(plane, ((0, ph), (0, pw)), mode="edge")
    return padded.reshape(
        padded.shape[0] // step, step, padded.shape[1] // step, step
    ).mean(axis=(1, 3), dtype=np.float32)


class ZarrWellPixels:
    """A cached well read lazily from its zarr pyramid at one timepoint."""

    def __init__(self, well_data: dict[str, Any], timepoint: int = 0) -> None:
        self._data = well_data
        level0 = well_data["image"][0]  # (T, C, Y, X)
        self._t = min(timepoint, level0.shape[0] - 1)
        self.shape_yx = (int(level0.shape[-2]), int(level0.shape[-1]))
        self.n_levels = len(well_data["image"])

    def read(self, view: View, channels: list[int]) -> np.ndarray[Any, Any]:
        ys, xs = _level_slice(view, strided=False)
        level = self._data["image"][view.level]
        # One channel at a time: a full-resolution region of every channel
        # at once would hold several hundred MB for a whole 10x well.
        return np.stack(
            [
                _block_mean(np.asarray(level[self._t, c, ys, xs]), view.step)
                for c in channels
            ]
        )

    def read_mask(self, view: View, name: str) -> np.ndarray[Any, Any] | None:
        pyramid = self._data.get(name) or []
        if not pyramid:
            return None
        if view.level < len(pyramid):
            level = pyramid[view.level]
        else:
            # Label pyramid shallower than the image's: read level 0 sparsely.
            level = pyramid[0]
            view = View(view.y0, view.y1, view.x0, view.x1, 0, view.downsample)
        ys, xs = _level_slice(view)
        t = min(self._t, level.shape[0] - 1)
        return np.asarray(level[t, ys, xs])


class ArrayWellPixels:
    """An in-memory stitched canvas: ``(C, Y, X)`` image and named masks."""

    n_levels = 1

    def __init__(
        self,
        image_cyx: np.ndarray[Any, Any],
        masks: dict[str, np.ndarray[Any, Any]] | None = None,
    ) -> None:
        self._image = image_cyx
        self._masks = masks or {}
        self.shape_yx = (int(image_cyx.shape[-2]), int(image_cyx.shape[-1]))

    def read(self, view: View, channels: list[int]) -> np.ndarray[Any, Any]:
        ys, xs = _level_slice(view, strided=False)
        return np.stack(
            [_block_mean(self._image[c, ys, xs], view.step) for c in channels]
        )

    def read_mask(self, view: View, name: str) -> np.ndarray[Any, Any] | None:
        mask = self._masks.get(name)
        if mask is None:
            return None
        ys, xs = _level_slice(view)
        return mask[ys, xs]


def stitch_field_well(
    images: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any] | None,
    positions: Sequence[tuple[float, float] | None],
    pixel_size_um: float | None,
) -> ArrayWellPixels:
    """Stitch one well's fields (one timepoint) into an in-memory canvas.

    Uses the stage positions for the grid and the plate's per-objective
    stitch calibration for the spacing, as the welldata widget does.

    Args:
        images: ``(N, Y, X, C)`` flatfield-corrected fields.
        labels: ``(N, Y, X[, 2])`` masks (nuclei, cells), or None.
        positions: Stage position of each field.
        pixel_size_um: Plate pixel size, selecting the stitch calibration.

    Raises:
        OverviewError: The fields cannot be placed from their positions.
    """
    from omero_utils.stitching import (
        positions_to_offsets,
        resolve_stitch_params,
        stitch_from_offsets,
        stitch_labels_from_offsets,
    )

    if images.ndim != 4:
        raise OverviewError(
            f"expected per-field images (N, Y, X, C), got shape {images.shape}"
        )
    size_y, size_x = images.shape[1:3]
    offsets = positions_to_offsets(
        list(positions), size_x, size_y, **resolve_stitch_params(pixel_size_um)
    )
    valid = offsets[:, 0] >= 0
    if not np.any(valid):
        raise OverviewError(
            "the fields cannot be placed from their stage positions"
        )
    canvas = stitch_from_offsets(images[valid], offsets[valid])  # (Y, X, C)
    masks: dict[str, np.ndarray[Any, Any]] = {}
    if labels is not None and labels.size and len(labels) == len(images):
        stack = labels if labels.ndim == 4 else labels[..., np.newaxis]
        stitched = stitch_labels_from_offsets(stack[valid], offsets[valid])
        for name, channel in _FIELD_LABEL_CHANNEL.items():
            if channel < stitched.shape[-1]:
                masks[name] = stitched[..., channel]
    return ArrayWellPixels(np.moveaxis(canvas, -1, 0), masks)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def sample_pixels(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """One well's pixels of one channel, as a fixed-seed sample for limits.

    Exact zeros (canvas no field covers, or masked-out pixels) are dropped:
    they are not dark signal, and left in they pin the low limit at 0 and
    lift the background. At most 1M pixels are kept, with a fresh seed per
    call, so a well's sample does not depend on which wells came before.
    """
    flat = np.asarray(values).ravel()
    flat = flat[flat != 0]
    if flat.size > _SAMPLES_PER_WELL:
        rng = np.random.default_rng(0)
        flat = flat[rng.choice(flat.size, _SAMPLES_PER_WELL, replace=False)]
    return flat


def percentile_limits(
    samples: Sequence[np.ndarray[Any, Any]],
) -> tuple[int, int]:
    """0.1/99.9-percentile limits over several wells' pixels of one channel.

    Each well's pixels go through :func:`sample_pixels` (idempotent, so
    pre-sampled arrays may be passed), and the percentiles are taken over
    the pooled samples: the result does not depend on the order of the
    wells. A flat or empty channel gets the full uint16 range, as in the
    viewer.
    """
    pooled = [sample_pixels(values) for values in samples]
    if not pooled or not sum(p.size for p in pooled):
        return 0, _UINT16_MAX
    lo, hi = np.percentile(np.concatenate(pooled), [0.1, 99.9])
    return (int(lo), int(hi)) if hi > lo else (0, _UINT16_MAX)


def _hex_rgb(hex_colour: str) -> np.ndarray[Any, Any]:
    value = hex_colour.lstrip("#")
    return np.array([int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def compose_rgb(
    planes: np.ndarray[Any, Any],
    colours: Sequence[str],
    limits: Sequence[tuple[float, float]],
) -> np.ndarray[Any, Any]:
    """Additively blend ``(C, y, x)`` planes into an ``(y, x, 3)`` image.

    Each channel is scaled to [0, 1] by its limits and tinted with its
    colour; the tints are summed and clipped, as napari's additive blending
    does.
    """
    rgb = np.zeros((*planes.shape[1:], 3), dtype=np.float32)
    for plane, colour, (lo, hi) in zip(planes, colours, limits, strict=True):
        scaled = np.clip(
            (plane.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1
        )
        rgb += scaled[..., np.newaxis] * _hex_rgb(colour).astype(np.float32)
    return np.clip(rgb, 0, 1)


def draw_outlines(
    rgb: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], colour: str
) -> np.ndarray[Any, Any]:
    """Paint the boundaries of ``labels`` onto ``rgb`` in ``colour``."""
    from skimage.segmentation import find_boundaries

    out = rgb.copy()
    boundary = find_boundaries(labels, mode="inner")  # type: ignore[no-untyped-call]
    out[boundary] = _hex_rgb(colour)
    return out


def scale_bar_um(width_um: float) -> float:
    """A round scale-bar length near a fifth of the view width (1/2/5 x 10^k)."""
    target = width_um / 5
    if target <= 0:
        return 0.0
    magnitude = 10 ** math.floor(math.log10(target))
    return float(
        max(m * magnitude for m in (1, 2, 5) if m * magnitude <= target)
    )


def render_overview(
    rgb: np.ndarray[Any, Any],
    *,
    caption: str | None,
    pixel_size_um: float | None,
    scale_bar: bool = True,
    dpi: int = 300,
) -> Figure:
    """Lay out one overview at one image pixel per output pixel.

    The caption sits top-left inside the image, like the viewer's text
    overlay; the scale bar bottom-right.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    height, width = rgb.shape[:2]
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    # Text size tracks the image so captions read the same at any zoom.
    font_pt = max(height / 40, 6) * 72 / dpi
    if caption:
        # Wrap to the image width (a character is ~0.6 em wide).
        font_px = font_pt * dpi / 72
        per_line = max(int(width * 0.97 / (0.6 * font_px)), 10)
        ax.text(
            0.01,
            0.99,
            textwrap.fill(caption, per_line),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="yellow",
            fontsize=font_pt,
            bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"},
        )
    if scale_bar and pixel_size_um:
        length_um = scale_bar_um(width * pixel_size_um)
        if length_um:
            length_px = length_um / pixel_size_um
            bar_h = max(height / 150, 2)
            x = width * 0.97 - length_px
            y = height * 0.96 - bar_h
            ax.add_patch(Rectangle((x, y), length_px, bar_h, color="white"))
            ax.text(
                x + length_px / 2,
                y - bar_h,
                f"{length_um:g} µm",
                ha="center",
                va="bottom",
                color="white",
                fontsize=font_pt,
            )
    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class WellInput:
    """One well to render: its pixels, caption and level-0 pixel size."""

    well: str
    pixels: WellPixels
    caption: str | None
    pixel_size_um: float | None


@dataclass
class OverviewSettings:
    """The rendering choices shared by every well of one call."""

    channel_names: list[str]
    layers: list[str]
    zoom: int = 1
    center: tuple[float, float] = (0.5, 0.5)
    size: int = DEFAULT_SIZE
    limits: dict[str, tuple[int, int]] = field(default_factory=dict)
    scale_bar: bool = True
    fmt: str = "png"
    dpi: int = 300


def split_layers(
    layers: Sequence[str], channel_names: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Split layer names into (channels, mask names); reject unknown ones."""
    channels, masks = [], []
    for layer in layers:
        if layer in MASK_LAYERS:
            masks.append(MASK_LAYERS[layer])
        elif layer in channel_names:
            channels.append(layer)
        else:
            raise OverviewError(
                f"unknown layer {layer!r}; available: "
                f"{', '.join([*channel_names, *MASK_LAYERS])}"
            )
    if not channels and not masks:
        raise OverviewError("no layers to draw")
    return channels, masks


def render_wells(
    wells: Sequence[str],
    load_well: Callable[[str], WellInput],
    settings: OverviewSettings,
    out_dir: Any,
) -> dict[str, Any]:
    """Render every well's overview with shared display limits.

    Two passes: read each well's view (small: at most ``size`` pixels a
    side), then pool the limits over all views and render. ``load_well`` is
    called once per well, so a per-field plate holds one well's fields in
    memory at a time.

    Returns:
        The manifest entries: ``limits`` (by channel name), ``colours``,
        ``views`` and per-well ``wells`` outcomes. Files are written to
        ``out_dir`` as ``<well>.<fmt>``.
    """
    from pathlib import Path

    from omero_screen_napari.zarr_cache.palette import channel_hex_colors

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    channels, masks = split_layers(settings.layers, settings.channel_names)
    channel_idx = [settings.channel_names.index(c) for c in channels]
    # Colours are assigned over all plate channels, so a channel keeps its
    # napari colour whichever subset is drawn.
    palette = dict(
        zip(
            settings.channel_names,
            channel_hex_colors(settings.channel_names),
            strict=True,
        )
    )

    views: dict[str, dict[str, Any]] = {}
    entries: dict[str, Any] = {}
    for well in wells:
        try:
            item = load_well(well)
            view = plan_view(
                item.pixels.shape_yx,
                item.pixels.n_levels,
                zoom=settings.zoom,
                center=settings.center,
                size=settings.size,
            )
            views[well] = {
                "input": item,
                "view": view,
                "planes": item.pixels.read(view, channel_idx)
                if channels
                else None,
                "masks": {m: item.pixels.read_mask(view, m) for m in masks},
            }
        except Exception as exc:  # noqa: BLE001 — one well must not kill the run
            logger.warning(f"Well {well}: overview failed ({exc})")
            entries[well] = {"exported": False, "reason": str(exc)}

    limits = {
        name: settings.limits.get(name)
        or percentile_limits([v["planes"][i] for v in views.values()])
        for i, name in enumerate(channels)
    }

    for well, v in views.items():
        item, view = v["input"], v["view"]
        planes = v["planes"]
        if planes is not None:
            rgb = compose_rgb(
                planes,
                [palette[c] for c in channels],
                [limits[c] for c in channels],
            )
        else:
            rgb = np.zeros((*_view_out_shape(view), 3), dtype=np.float32)
        missing_masks = []
        for name, labels in v["masks"].items():
            if labels is None:
                missing_masks.append(name)
                continue
            rgb = draw_outlines(rgb, labels, MASK_HEX[name])
        pixel_size = (
            item.pixel_size_um * view.downsample
            if item.pixel_size_um
            else None
        )
        fig = render_overview(
            rgb,
            caption=item.caption,
            pixel_size_um=pixel_size,
            scale_bar=settings.scale_bar,
            dpi=settings.dpi,
        )
        path = out / f"{well}.{settings.fmt}"
        fig.savefig(str(path), format=settings.fmt, dpi=settings.dpi)
        logger.info(f"Well {well}: wrote {path.name}")
        entries[well] = {
            "exported": True,
            "file": path.name,
            "caption": item.caption,
            "region_yx": [view.y0, view.y1, view.x0, view.x1],
            "level": view.level,
            "downsample": view.downsample,
            "shape_yx": list(rgb.shape[:2]),
            "pixel_size_um": pixel_size,
            "missing_masks": missing_masks,
        }

    return {
        "layers": list(settings.layers),
        "zoom": settings.zoom,
        "center": list(settings.center),
        "size": settings.size,
        "limits": {k: list(v) for k, v in limits.items()},
        "limits_source": {
            c: "given" if c in settings.limits else "pooled" for c in channels
        },
        "colours": {c: palette[c] for c in channels},
        "mask_colours": {m: MASK_HEX[m] for m in masks},
        "wells": {w: entries[w] for w in wells if w in entries},
    }


def _view_out_shape(view: View) -> tuple[int, int]:
    """Output ``(y, x)`` shape of a view (matches :func:`_level_slice`)."""
    ys, xs = _level_slice(view)
    return (
        len(range(ys.start, ys.stop, ys.step)),
        len(range(xs.start, xs.stop, xs.step)),
    )


# ---------------------------------------------------------------------------
# Loaders: one well → WellInput
# ---------------------------------------------------------------------------


def _caption(plate_id: int, well: str, meta: dict[str, Any] | None) -> str:
    """``Plate 5108 — G5 | cell_line: RPE-1, ...``: all the well's metadata.

    Fuller than the viewer's overlay (which shows cell line, condition and
    timepoint only) and in the gallery title's ``key: value`` form, so an
    overview identifies the condition on its own.
    """
    bits = [f"{key}: {value}" for key, value in (meta or {}).items() if value]
    return f"Plate {plate_id} — {well}" + (
        " | " + ", ".join(bits) if bits else ""
    )


def zarr_well_input(
    plate_id: int, well: str, info: dict[str, Any], timepoint: int = 0
) -> WellInput:
    """A cached well, read lazily from its zarr pyramid."""
    from omero_screen_napari.zarr_cache import read_well

    meta = (info.get("well_metadata") or {}).get(well)
    return WellInput(
        well=well,
        pixels=ZarrWellPixels(read_well(plate_id, well), timepoint),
        caption=_caption(plate_id, well, meta),
        pixel_size_um=info.get("pixel_size_um"),
    )


def field_well_input(
    plate_id: int,
    well: str,
    omero_data: Any,
    *,
    timepoint: int = 0,
    connection: Any = None,
) -> WellInput:
    """A per-field well: load its fields at ``timepoint`` and stitch them.

    Replaces whatever well ``omero_data`` held, so only one well's fields
    are in memory at a time.
    """
    from omero_screen_napari.well_context import load_well_context

    load_well_context(
        plate_id,
        [well],
        omero_data=omero_data,
        timepoint=timepoint,
        connection=connection,
    )
    meta = (omero_data.well_metadata_list or [None])[0]
    pixel_size = omero_data.pixel_size[0] if omero_data.pixel_size else None
    labels = omero_data.labels if omero_data.labels.size else None
    return WellInput(
        well=well,
        pixels=stitch_field_well(
            omero_data.images, labels, omero_data.image_positions, pixel_size
        ),
        caption=_caption(plate_id, well, meta),
        pixel_size_um=pixel_size,
    )
