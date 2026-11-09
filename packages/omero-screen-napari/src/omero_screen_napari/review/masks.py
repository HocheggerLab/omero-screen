"""Reviewer-made nucleus masks: split a nucleus, draw a missed one, or
separate a cell from a reporter-negative object merged with it.

A mask edit never writes the zarr cache. The new mask is saved as a small
patch (bounding-box origin + boolean array) under ``masks/`` beside the edit
log, and the edit log records a ``mask_add`` naming the patch, the raw labels
it replaces and its measurements. The measurements are taken from the image
when the edit is committed, so replaying the log needs no pixel access.

Measurements match the pipeline's: for each channel, the mean intensity over
the mask minus the frame's background, where the background is recovered
exactly from a nucleus the pipeline measured in the same frame (its raw mean
over its own pixels minus its stored background-subtracted value).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class MaskError(ValueError):
    """Raised when a mask edit cannot be made."""


# -- patch storage ----------------------------------------------------------


def save_patch(
    patch_dir: Path,
    well: str,
    frame: int,
    label: int,
    y0: int,
    x0: int,
    mask: np.ndarray,
) -> str:
    """Store a patch; returns its path relative to ``patch_dir``'s parent."""
    rel = Path("masks") / well / f"t{frame:04d}_L{label}.npz"
    out = patch_dir.parent / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, y0=y0, x0=x0, mask=mask.astype(bool))
    return str(rel)


def load_patch(base_dir: Path, rel: str) -> tuple[int, int, np.ndarray]:
    """``(y0, x0, mask)`` of a stored patch.

    Track Review reads every patch of a well on each redraw; reads are cached
    by file and modification time, so a patch saved again under the same
    name is re-read. The cached mask is read-only.
    """
    path = base_dir / rel
    stat = path.stat()
    return _read_patch(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4096)
def _read_patch(
    path: str, mtime_ns: int, size: int
) -> tuple[int, int, np.ndarray]:
    data = np.load(path)
    mask = data["mask"]
    mask.flags.writeable = False
    return int(data["y0"]), int(data["x0"]), mask


def paint_patches(
    labels: np.ndarray,
    frame: int,
    y0: int,
    x0: int,
    curated: Any,
    base_dir: Path | None,
) -> np.ndarray:
    """Apply reviewer masks to a label crop whose top-left canvas pixel is ``(y0, x0)``.

    Replaced raw labels are cleared; patches of ``frame`` are painted with their label.
    """
    if base_dir is None or (not curated.extras and not curated.removed):
        return labels
    out = labels.copy()
    for t, lab in curated.removed:
        if t == frame:
            out[out == lab] = 0
    h, w = out.shape
    for (t, lab), meas in curated.extras.items():
        if t != frame or "patch" not in meas:
            continue
        py, px, mask = load_patch(base_dir, meas["patch"])
        ys, xs = np.nonzero(mask)
        ys, xs = ys + py - y0, xs + px - x0
        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        out[ys[ok], xs[ok]] = lab
    return out


# -- geometry -----------------------------------------------------------------


def split_mask(
    region: np.ndarray, seeds: list[tuple[int, int]]
) -> list[np.ndarray]:
    """Split a boolean nucleus mask into one part per seed (watershed on distance).

    Args:
        region: Boolean mask of the nucleus (crop coordinates).
        seeds: Two or more ``(y, x)`` points inside the mask.

    Raises:
        MaskError: If a seed lies outside the mask or a part comes out empty.
    """
    from scipy import ndimage
    from skimage.segmentation import watershed

    if len(seeds) < 2:
        raise MaskError("Splitting needs at least two seeds.")
    markers = np.zeros(region.shape, dtype=np.int32)
    for i, (y, x) in enumerate(seeds, start=1):
        if (
            not (0 <= y < region.shape[0] and 0 <= x < region.shape[1])
            or not region[y, x]
        ):
            raise MaskError(f"Seed {i} is not inside the nucleus.")
        markers[y, x] = i
    distance = ndimage.distance_transform_edt(region)
    parts = watershed(-distance, markers, mask=region)
    out = [parts == i for i in range(1, len(seeds) + 1)]
    if any(not p.any() for p in out):
        raise MaskError(
            "A part of the split is empty; place the seeds further apart."
        )
    return out


def cut_mask(
    region: np.ndarray, p1: tuple[float, float], p2: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Divide a boolean nucleus mask by the infinite line through two points.

    The reviewer clicks two points on the boundary between two touching
    nuclei; the line through them is the cut. Unlike :func:`split_mask`
    (a watershed on the distance transform, which cuts wherever the outline
    is thinnest), the cut goes exactly where the reviewer put it, which is
    what is wanted when two nuclei touch broadly and there is no neck.

    Rules:

    * The points need not lie inside the mask, only on the line wanted.
    * Pixels exactly on the line go to the side that is smaller without
      them (to the first side on a tie), so the cut never favours the
      bigger part.
    * A side is kept whole even if the line leaves it in several connected
      pieces (a concave outline): every piece on one side of the line is
      one part. Nothing is discarded, so the two parts always cover the
      mask exactly.

    Args:
        region: Boolean mask of the nucleus (crop coordinates).
        p1, p2: ``(y, x)`` points on the cut line, crop coordinates.

    Returns:
        ``(side_a, side_b)``: ``side_a`` is left of the direction p1 → p2
        (positive cross product in ``(y, x)``), ``side_b`` the other side.

    Raises:
        MaskError: If the points coincide or one side of the cut is empty.
    """
    (y1, x1), (y2, x2) = p1, p2
    dy, dx = float(y2) - float(y1), float(x2) - float(x1)
    if dy == 0 and dx == 0:
        raise MaskError("The two points of the cut line coincide.")
    yy, xx = np.indices(region.shape, dtype=np.float64)
    side = dy * (xx - float(x1)) - dx * (yy - float(y1))
    a = region & (side > 0)
    b = region & (side < 0)
    on = region & (side == 0)
    if a.sum() <= b.sum():
        a = a | on
    else:
        b = b | on
    if not a.any() or not b.any():
        raise MaskError(
            "The cut line does not cross the nucleus; click two points on "
            "the boundary between the nuclei."
        )
    return a, b


def cell_part(
    parts: tuple[np.ndarray, np.ndarray],
    y0: int,
    x0: int,
    reference: tuple[float, float] | None,
) -> int:
    """Index (0 or 1) of the part of a cut that belongs to the cell.

    The part whose centroid (canvas coordinates; crop origin ``y0, x0``) is
    nearest ``reference``, the cell's position in a neighbouring frame. With
    no reference (a cell seen in no other frame) the larger part is the cell.
    """
    if reference is None:
        return 0 if parts[0].sum() >= parts[1].sum() else 1
    d = []
    for part in parts:
        ys, xs = np.nonzero(part)
        d.append(
            np.hypot(
                ys.mean() + y0 - reference[0], xs.mean() + x0 - reference[1]
            )
        )
    return 0 if d[0] <= d[1] else 1


#: Frames before the split whose areas set the cell's normal size.
NORMAL_WINDOW = 10
#: A frame is "normal" when its area is within this factor band of the
#: cell's typical area.
NORMAL_BAND = (0.6, 1.4)


def cell_areas(
    det: pd.DataFrame, curated: Any, tid: int
) -> dict[int, tuple[float, float, float]]:
    """``frame -> (area, y, x)`` of curated track ``tid``.

    Reviewer masks count by their stored measurements, raw detections by
    their row in ``det``. Several detections in one frame sum their areas
    and give their area-weighted centroid.
    """
    rows = det.set_index(["timepoint", "label"])
    by_frame: dict[int, list[tuple[float, float, float]]] = {}
    for t, lab in curated.detections(tid):
        if (t, lab) in curated.extras:
            m = curated.extras[(t, lab)]
            pt = (float(m.get("area", 1.0)), float(m["y"]), float(m["x"]))
        elif (t, lab) in rows.index:
            r = rows.loc[(t, lab)]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            pt = (float(r["area"]), float(r["y"]), float(r["x"]))
        else:
            continue
        by_frame.setdefault(int(t), []).append(pt)
    out = {}
    for t, pts in by_frame.items():
        a = np.array([p[0] for p in pts])
        w = a if a.sum() > 0 else np.ones_like(a)
        out[t] = (
            float(a.sum()),
            float(np.average([p[1] for p in pts], weights=w)),
            float(np.average([p[2] for p in pts], weights=w)),
        )
    return out


def normal_frame(
    areas: dict[int, float],
    frame: int,
    window: int = NORMAL_WINDOW,
    band: tuple[float, float] = NORMAL_BAND,
) -> int | None:
    """The nearest frame where the cell is a single nucleus of normal size.

    Rule (simple, robust to a run of merged frames):

    1. Take the cell's last ``window`` frames before ``frame`` (if it has
       none, its first ``window`` frames after). ``m0`` is the **lower
       quartile** of their areas, not the median: merges (the error a
       split corrects) only ever make the area larger, and a cell about to
       be split is often merged in half the frames before it, so the
       median would be a merged size. The **typical area** is the median
       of the window's areas within ``band`` × ``m0`` (``m0`` itself if
       none is).
    2. The answer is the nearest earlier frame whose area is within
       ``band`` × the typical area; else the nearest such later frame;
       else ``None``.

    On plate 5054 C3 (a ~700 px nucleus merged at t147, t150, t152–156)
    this gives t151 (696 px) for a split at t153–155, where the nearest
    earlier frame, t152, is itself merged (1638 px).

    Args:
        areas: ``frame -> area`` of the cell (``frame`` itself is ignored).
        frame: The frame being split.
        window: How many neighbouring frames set the typical area.
        band: ``(low, high)`` factors of the typical area that count as normal.
    """
    earlier = sorted(t for t in areas if t < frame)
    later = sorted(t for t in areas if t > frame)
    ref = earlier[-window:] or later[:window]
    if not ref:
        return None
    m0 = float(np.percentile([areas[t] for t in ref], 25))
    inside = [
        areas[t] for t in ref if band[0] * m0 <= areas[t] <= band[1] * m0
    ]
    typical = float(np.median(inside)) if inside else m0
    lo, hi = band[0] * typical, band[1] * typical
    for t in [*reversed(earlier), *later]:
        if lo <= areas[t] <= hi:
            return t
    return None


def reference_detection(
    det: pd.DataFrame, curated: Any, tid: int, frame: int
) -> tuple[int, tuple[float, float]] | None:
    """``(frame, (y, x))`` of the cell where it best shows which part it is.

    The cell's position in :func:`normal_frame` (nearest earlier frame where
    it is a single nucleus of normal size, else nearest later); if no frame
    passes that test, the nearest earlier frame it has a detection in, else
    the nearest later one; ``None`` if the track has no other frame.
    """
    found = {
        t: v for t, v in cell_areas(det, curated, tid).items() if t != frame
    }
    if not found:
        return None
    t_ref = normal_frame({t: v[0] for t, v in found.items()}, frame)
    if t_ref is None:
        earlier = [t for t in found if t < frame]
        t_ref = max(earlier) if earlier else min(found)
    _, y, x = found[t_ref]
    return t_ref, (y, x)


def reference_position(
    det: pd.DataFrame, curated: Any, tid: int, frame: int
) -> tuple[float, float] | None:
    """Where curated track ``tid`` is in its reference frame for ``frame``.

    See :func:`reference_detection`; ``None`` if the track has no other frame.
    """
    found = reference_detection(det, curated, tid, frame)
    return None if found is None else found[1]


def label_on_segment(
    labels: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]
) -> int:
    """The nucleus a cut between two clicks is meant for.

    The most frequent non-zero label sampled along the segment p1 → p2
    (the clicks sit on the boundary, so either may miss the mask); 0 if the
    segment crosses no nucleus.
    """
    n = int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))) + 1
    ys = np.rint(np.linspace(p1[0], p2[0], n)).astype(int)
    xs = np.rint(np.linspace(p1[1], p2[1], n)).astype(int)
    ok = (
        (ys >= 0) & (ys < labels.shape[0]) & (xs >= 0) & (xs < labels.shape[1])
    )
    vals = np.asarray(labels[ys[ok], xs[ok]]).astype(np.int64)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0
    found, counts = np.unique(vals, return_counts=True)
    return int(found[np.argmax(counts)])


# -- measurement --------------------------------------------------------------


def channel_map(
    image_channels: list[str], det_columns: list[str]
) -> dict[str, int]:
    """Detection column → image channel index, matched by case-insensitive prefix.

    CellView stores the nuclear stain as ``dapi`` whatever the dye, so
    ``dapi`` maps to the channel with the ``_nucleus`` role (or a DNA dye
    name) when no channel starts with it.
    """
    lower = [c.lower() for c in image_channels]
    out = {}
    for col in det_columns:
        for i, name in enumerate(lower):
            if name == col or name.startswith(col) or col.startswith(name):
                out[col] = i
                break
        if col == "dapi" and col not in out:
            for i, name in enumerate(lower):
                if name.endswith("_nucleus") or any(
                    k in name for k in ("dna", "hoechst", "dapi")
                ):
                    out[col] = i
                    break
    return out


def measure(
    image: Any,
    nuclei: Any,
    det_frame: pd.DataFrame,
    frame: int,
    y0: int,
    x0: int,
    mask: np.ndarray,
    channels: dict[str, int],
) -> dict[str, Any]:
    """Area, centroid and background-subtracted channel means of a new mask.

    Args:
        image: Level-0 image ``(T, C, Y, X)``.
        nuclei: Level-0 raw labels ``(T, Y, X)``.
        det_frame: The pipeline's detections in ``frame`` (background-subtracted).
        frame: Timepoint.
        y0, x0: Canvas origin of ``mask``.
        mask: Boolean mask.
        channels: Detection column → image channel index.

    Raises:
        MaskError: If the mask is empty or no reference nucleus is available.
    """
    if not mask.any():
        raise MaskError("The mask is empty.")
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    out: dict[str, Any] = {
        "area": float(mask.sum()),
        "y": float(ys.mean() + y0),
        "x": float(xs.mean() + x0),
    }
    # Reference nucleus for the frame's background: the measured one nearest the mask.
    if det_frame.empty:
        raise MaskError(
            f"No measured nucleus in frame {frame} to take the background from."
        )
    d = np.hypot(det_frame["y"] - out["y"], det_frame["x"] - out["x"])
    ref = det_frame.loc[d.idxmin()]
    ry, rx, ref_label = (
        int(round(ref["y"])),
        int(round(ref["x"])),
        int(ref["label"]),
    )
    half = int(np.sqrt(ref["area"]) * 2) + 4
    ry0, rx0 = max(ry - half, 0), max(rx - half, 0)
    ref_labels = np.asarray(nuclei[frame, ry0 : ry + half, rx0 : rx + half])
    ref_mask = ref_labels == ref_label
    for col, ch in channels.items():
        crop = np.asarray(
            image[frame, ch, y0 : y0 + h, x0 : x0 + w], dtype=np.float64
        )
        raw_mean = float(crop[mask].mean())
        bg = 0.0
        if ref_mask.any() and pd.notna(ref.get(col)):
            ref_crop = np.asarray(
                image[frame, ch, ry0 : ry + half, rx0 : rx + half],
                dtype=np.float64,
            )
            bg = float(ref_crop[ref_mask].mean()) - float(ref[col])
        out[col] = max(raw_mean - bg, 0.0)
    return out


def next_label(det: pd.DataFrame, curated: Any) -> int:
    """A label above every raw and reviewer-made label of the well."""
    extra = max((lab for _, lab in curated.extras), default=0)
    return int(max(det["label"].max(), extra, curated.next_id)) + 1


# -- edits ----------------------------------------------------------------------


def _arrays(session: Any, well: str) -> tuple[Any, Any, list[str]]:
    from omero_screen_napari.zarr_cache.reader import read_well

    data = read_well(session.queue.plate_id, well)
    if not data["image"] or not data["nuclei"]:
        raise MaskError(f"Well {well} is not in the zarr cache.")
    from omero_screen_napari.zarr_cache.correction import tracked_nuclei

    nuclei = tracked_nuclei(session.queue.plate_id, well, list(data["nuclei"]))
    return data["image"][0], nuclei[0], data["channel_names"]


def commit_drawn(
    session: Any,
    item_id: str,
    frame: int,
    full_mask: np.ndarray,
    overlap: float = 0.5,
) -> Any:
    """Add a nucleus drawn on a full-frame boolean mask to the cell in ``frame``.

    Raw labels covered by more than ``overlap`` of their area are replaced.
    """
    item = session.item(item_id)
    image, nuclei, names = _arrays(session, item.well)
    det, curated = session.source.well(item.well)
    ys, xs = np.nonzero(full_mask)
    if ys.size == 0:
        raise MaskError("Nothing drawn.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    mask = full_mask[y0:y1, x0:x1].astype(bool)
    under = np.asarray(nuclei[frame, y0:y1, x0:x1])
    replaces = []
    for lab in np.unique(under[mask]):
        if (
            lab
            and (under[mask] == lab).sum()
            > overlap * (np.asarray(nuclei[frame]) == lab).sum()
        ):
            replaces.append(int(lab))
    cols = [
        c
        for c in det.columns
        if c
        not in {
            "measurement_id",
            "timepoint",
            "label",
            "track_id_raw",
            "parent_track_id_raw",
            "area",
            "y",
            "x",
        }
    ]
    det_frame = _measured_rows(det, curated, frame, replaces)
    meas = measure(
        image,
        nuclei,
        det_frame,
        frame,
        int(y0),
        int(x0),
        mask,
        channel_map(names, cols),
    )
    new = next_label(det, curated)
    meas["patch"] = save_patch(
        session.edits_path.parent / "masks",
        item.well,
        frame,
        new,
        int(y0),
        int(x0),
        mask,
    )
    return session.edit(
        item_id,
        "mask_add",
        {
            "frame": frame,
            "label": new,
            "measurements": meas,
            "replaces": replaces,
        },
        reason=f"drawn nucleus at t{frame}",
    )


# -- separation by reporter ---------------------------------------------------

#: Detection columns that are not channels.
_NOT_CHANNELS = {
    "measurement_id",
    "timepoint",
    "label",
    "track_id_raw",
    "parent_track_id_raw",
    "area",
    "y",
    "x",
}


@dataclass
class FrameSeparation:
    """How one frame of a merged cell is to be split.

    Attributes:
        frame: Timepoint.
        y0, x0: Canvas origin of the masks.
        cell: Boolean mask of the reporter-positive part (the cell).
        other: Boolean mask of the rest (the merged object).
        replaces: Raw labels of the cell in this frame, replaced by ``cell``.
        score: Median reporter score inside ``cell`` (1 = the cell's own
            level before the merge).
    """

    frame: int
    y0: int
    x0: int
    cell: np.ndarray
    other: np.ndarray
    replaces: list[int]
    score: float


@dataclass
class SeparationPlan:
    """A separation over a range of frames, computed but not yet written."""

    item_id: str
    frames: list[FrameSeparation]
    skipped: dict[int, str]
    reference: dict[str, float]
    threshold: float


def reporter_columns(det: pd.DataFrame) -> list[str]:
    """The PIP-FUCCI reporter columns present (``pip``, ``marker``/``geminin``)."""
    return [c for c in ("pip", "geminin", "marker") if c in det.columns]


def _cell_region(
    nuclei: Any,
    frame: int,
    labels: list[int],
    cy: float,
    cx: float,
    half: int,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Crop around ``(cy, cx)``: ``(y0, x0, all labels, union of labels)``."""
    y0, x0 = max(int(cy) - half, 0), max(int(cx) - half, 0)
    crop = np.asarray(nuclei[frame, y0 : int(cy) + half, x0 : int(cx) + half])
    return y0, x0, crop, np.isin(crop, labels)


def _reporter_score(
    image: Any,
    frame: int,
    y0: int,
    x0: int,
    crop_labels: np.ndarray,
    channels: dict[str, int],
    reference: dict[str, float],
) -> np.ndarray:
    """Per-pixel reporter level relative to the cell's own (max over reporters).

    Each channel is smoothed, has the crop's background (median of pixels in
    no nucleus) subtracted and is divided by the cell's reference level.
    """
    from scipy import ndimage

    h, w = crop_labels.shape
    out = np.zeros((h, w), dtype=np.float64)
    for col, ref in reference.items():
        crop = np.asarray(
            image[frame, channels[col], y0 : y0 + h, x0 : x0 + w],
            dtype=np.float64,
        )
        bg = (
            float(np.median(crop[crop_labels == 0]))
            if (crop_labels == 0).any()
            else 0.0
        )
        out = np.maximum(
            out, ndimage.gaussian_filter(crop - bg, 1.5) / max(ref, 1e-6)
        )
    return out


def plan_separation(
    session: Any,
    item_id: str,
    first: int,
    last: int,
    threshold: float = 0.5,
    reference_frames: int = 3,
) -> SeparationPlan:
    """Split the cell from a reporter-negative object merged with it, frame by frame.

    In every frame from ``first`` to ``last`` where the cell has a mask, the
    pixels whose reporter level (PIP or geminin, relative to the cell's own
    level in the ``reference_frames`` frames before ``first``) is above
    ``threshold`` become the cell (largest connected part, holes filled);
    the rest of the mask is the merged object. Nothing is written.

    Raises:
        MaskError: If the cell has no frames before ``first`` to take its
            reporter level from, or no reporter channel is known.
    """
    from scipy import ndimage

    item = session.item(item_id)
    image, nuclei, names = _arrays(session, item.well)
    det, curated = session.source.well(item.well)
    tid = curated.resolve(session.anchor(item_id))
    keys = curated.detections(tid)
    by_frame: dict[int, list[int]] = {}
    for t, lab in keys:
        by_frame.setdefault(t, []).append(lab)
    reporters = reporter_columns(det)
    cmap = channel_map(names, reporters)
    reporters = [c for c in reporters if c in cmap]
    if not reporters:
        raise MaskError("No reporter channel (pip / geminin) in the image.")
    rows = det.set_index(["timepoint", "label"])

    # The cell's own reporter level, from frames before the merge.
    before = sorted(t for t in by_frame if t < first)[-reference_frames:]
    if not before:
        raise MaskError(
            f"The cell has no frame before t{first} to take its reporter level from."
        )
    samples: dict[str, list[float]] = {c: [] for c in reporters}
    for t in before:
        labs = by_frame[t]
        if any((t, lab) in curated.extras for lab in labs):
            continue
        r = rows.loc[[(t, lab) for lab in labs]]
        half = int(np.sqrt(float(r["area"].sum()))) * 2 + 8
        y0, x0, crop, region = _cell_region(
            nuclei, t, labs, r["y"].mean(), r["x"].mean(), half
        )
        h, w = crop.shape
        for col in reporters:
            px = np.asarray(
                image[t, cmap[col], y0 : y0 + h, x0 : x0 + w],
                dtype=np.float64,
            )
            bg = float(np.median(px[crop == 0])) if (crop == 0).any() else 0
            samples[col].append(float(np.median(px[region])) - bg)
    reference = {
        c: float(np.median(v))
        for c, v in samples.items()
        if v and np.median(v) > 0
    }
    if not reference:
        raise MaskError(
            f"No reporter signal in the cell before t{first}; nothing to separate by."
        )

    plans, skipped = [], {}
    for t in range(first, last + 1):
        labs = by_frame.get(t)
        if not labs:
            skipped[t] = "no mask"
            continue
        if any((t, lab) in curated.extras for lab in labs):
            skipped[t] = "already redrawn"
            continue
        r = rows.loc[[(t, lab) for lab in labs]]
        half = int(np.sqrt(float(r["area"].sum()))) * 2 + 8
        y0, x0, crop, region = _cell_region(
            nuclei, t, labs, r["y"].mean(), r["x"].mean(), half
        )
        score = _reporter_score(image, t, y0, x0, crop, cmap, reference)
        positive, n = ndimage.label((score > threshold) & region)
        if n == 0:
            skipped[t] = "no reporter-positive part"
            continue
        sizes = ndimage.sum_labels(
            np.ones_like(positive), positive, range(1, n + 1)
        )
        cell = ndimage.binary_fill_holes(positive == int(np.argmax(sizes)) + 1)
        cell &= region
        other = region & ~cell
        if other.sum() < 0.05 * region.sum():
            skipped[t] = "nothing merged"
            continue
        plans.append(
            FrameSeparation(
                frame=t,
                y0=y0,
                x0=x0,
                cell=cell,
                other=other,
                replaces=[int(lab) for lab in labs],
                score=float(np.median(score[cell])),
            )
        )
    return SeparationPlan(item_id, plans, skipped, reference, threshold)


def commit_separation(session: Any, plan: SeparationPlan) -> list[Any]:
    """Write a plan: per frame, the cell's new mask and the merged object's.

    The merged object's parts are joined into one new track. All entries
    share a ``batch`` id, so one Undo reverts the whole separation.
    """
    if not plan.frames:
        raise MaskError("Nothing to separate.")
    item = session.item(plan.item_id)
    image, nuclei, names = _arrays(session, item.well)
    det, curated = session.source.well(item.well)
    cols = [c for c in det.columns if c not in _NOT_CHANNELS]
    cmap = channel_map(names, cols)
    new = next_label(det, curated)
    batch = f"separate-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    first = plan.frames[0].frame
    reason = (
        f"separated from a reporter-negative object, "
        f"t{first}–{plan.frames[-1].frame} (threshold {plan.threshold})"
    )
    items: list[tuple[str, dict[str, Any], str]] = []
    for fs in plan.frames:
        t = fs.frame
        det_frame = _measured_rows(det, curated, t, fs.replaces)
        for part, label, extra in (
            (fs.cell, new, {"replaces": fs.replaces}),
            (
                fs.other,
                new + 1,
                {
                    "replaces": [],
                    "cell": None if t == first else [first, new + 1],
                },
            ),
        ):
            meas = measure(
                image, nuclei, det_frame, t, fs.y0, fs.x0, part, cmap
            )
            meas["patch"] = save_patch(
                session.edits_path.parent / "masks",
                item.well,
                t,
                label,
                fs.y0,
                fs.x0,
                part,
            )
            items.append(
                (
                    "mask_add",
                    {
                        "frame": t,
                        "label": label,
                        "measurements": meas,
                        "batch": batch,
                        **extra,
                    },
                    reason,
                )
            )
    return session.edit_many(plan.item_id, items)


# -- split along a line ------------------------------------------------------


@dataclass
class SplitPlan:
    """A nucleus cut in two along a reviewer's line, computed but not written.

    Attributes:
        item_id: Review item whose cell keeps ``cell``.
        frame: Timepoint.
        label: Label of the mask that is cut: a raw label, or a reviewer
            mask (``from_extra``) such as a part of an earlier split.
        y0, x0: Canvas origin of the masks.
        cell: Boolean mask of the part that is the reviewed cell.
        other: Boolean mask of the other part.
        other_cell: Anchor ``[frame, label]`` of the curated track that owned
            ``label`` before the cut when that is not the reviewed cell (the
            other part stays in it), else ``None`` (the other part becomes a
            new track).
        reference: The cell's position in the reference frame that decided
            which part is the cell (``None``: the larger part).
        reference_frame: That frame (see :func:`reference_detection`).
        flipped: The reviewer swapped the automatic choice of the cell's part.
        from_extra: ``label`` is a reviewer mask, cut from its patch.
    """

    item_id: str
    frame: int
    label: int
    y0: int
    x0: int
    cell: np.ndarray
    other: np.ndarray
    other_cell: list[int] | None
    reference: tuple[float, float] | None
    reference_frame: int | None = None
    flipped: bool = False
    from_extra: bool = False

    def flip(self) -> SplitPlan:
        """The same cut with the parts swapped: the other part is the cell.

        Ownership of the cut mask (``other_cell``) does not change: whichever
        part is not the cell stays with that owner.
        """
        return replace(
            self, cell=self.other, other=self.cell, flipped=not self.flipped
        )

    @property
    def frames(self) -> list[FrameSeparation]:
        """The plan as one :class:`FrameSeparation` (for drawing a preview)."""
        return [
            FrameSeparation(
                self.frame,
                self.y0,
                self.x0,
                self.cell,
                self.other,
                [self.label],
                float("nan"),
            )
        ]


def curated_labels(
    nuclei: Any,
    frame: int,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    curated: Any,
    base_dir: Path | None,
) -> np.ndarray:
    """Labels of ``frame`` in the box ``[y0:y1, x0:x1]`` as curated.

    Raw labels replaced by reviewer masks are cleared and the reviewer
    masks painted in (:func:`paint_patches`), so what is under a click is
    what the reviewer sees.
    """
    y0, x0 = max(y0, 0), max(x0, 0)
    crop = np.asarray(nuclei[frame, y0:y1, x0:x1])
    return paint_patches(crop, frame, y0, x0, curated, base_dir)


def _measured_rows(
    det: pd.DataFrame, curated: Any, frame: int, exclude: list[int]
) -> pd.DataFrame:
    """Detections of ``frame`` the pipeline measured (no reviewer masks).

    :func:`measure` recovers the frame's background from one of these, by
    reading its pixels from the raw label array, so a reviewer mask (whose
    label is not in that array) must not be offered as the reference.
    """
    extra = [lab for t, lab in curated.extras if t == frame]
    return det[
        (det["timepoint"] == frame) & ~det["label"].isin([*exclude, *extra])
    ]


def plan_split(
    session: Any,
    item_id: str,
    frame: int,
    p1: tuple[int, int],
    p2: tuple[int, int],
    label: int | None = None,
) -> SplitPlan:
    """Cut the nucleus between two canvas clicks along the line through them.

    The nucleus is ``label`` or, if not given, the one the segment between
    the clicks crosses most, as curated: a reviewer mask (e.g. a part of an
    earlier split) is found and cut from its patch, so a cut can be refined
    by cutting again. Which side is the reviewed cell is decided by
    :func:`reference_detection` (the cell where it was last a single nucleus
    of normal size) and :func:`cell_part`; :meth:`SplitPlan.flip` swaps it.
    Nothing is written.

    Raises:
        MaskError: If no nucleus is under the line or the line does not cut it.
    """
    item = session.item(item_id)
    _, nuclei, _ = _arrays(session, item.well)
    det, curated = session.source.well(item.well)
    base_dir = session.edits_path.parent
    if label is None:
        my, mx = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        r = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) // 2 + 2
        ly0, lx0 = max(my - r, 0), max(mx - r, 0)
        crop = curated_labels(
            nuclei, frame, ly0, lx0, my + r + 1, mx + r + 1, curated, base_dir
        )
        label = label_on_segment(
            crop, (p1[0] - ly0, p1[1] - lx0), (p2[0] - ly0, p2[1] - lx0)
        )
        if not label:
            raise MaskError(
                "No nucleus under the cut line; click two points on the "
                "boundary between the nuclei."
            )
    key = (int(frame), int(label))
    meas = curated.extras.get(key)
    if meas is not None:
        if "patch" not in meas:
            raise MaskError(f"Mask {label} at t{frame} has no stored pixels.")
        y0, x0, region = load_patch(base_dir, meas["patch"])
        region = region.astype(bool)
    else:
        row = det[(det["timepoint"] == frame) & (det["label"] == label)]
        if row.empty:
            raise MaskError(f"No nucleus {label} in frame {frame}.")
        half = int(np.sqrt(float(row["area"].iloc[0]))) * 2 + 8
        cy, cx = (
            int(round(float(row["y"].iloc[0]))),
            int(round(float(row["x"].iloc[0]))),
        )
        y0, x0 = max(cy - half, 0), max(cx - half, 0)
        region = (
            np.asarray(nuclei[frame, y0 : cy + half, x0 : cx + half]) == label
        )
    parts = cut_mask(
        region, (p1[0] - y0, p1[1] - x0), (p2[0] - y0, p2[1] - x0)
    )
    tid = curated.resolve(session.anchor(item_id))
    found = reference_detection(det, curated, tid, frame)
    reference = None if found is None else found[1]
    i = cell_part(parts, y0, x0, reference)
    owner = curated.tracks.get(key)
    return SplitPlan(
        item_id=item_id,
        frame=int(frame),
        label=int(label),
        y0=int(y0),
        x0=int(x0),
        cell=parts[i],
        other=parts[1 - i],
        other_cell=(
            [int(frame), int(label)]
            if owner is not None and owner != tid
            else None
        ),
        reference=reference,
        reference_frame=None if found is None else found[0],
        from_extra=meas is not None,
    )


def commit_split(session: Any, plan: SplitPlan) -> list[Any]:
    """Write a split: the cell's part joins the cell, the other part its owner.

    Two ``mask_add`` entries sharing a ``batch`` id (one Undo reverts both).
    If the cut nucleus belonged to another cell's curated track, the other
    part is written **first**, anchored on the cut nucleus itself
    (``cell = [frame, label]``), so it joins that track before the cell's
    part replaces the cut label: the neighbour's track continues through
    the split frame. If the nucleus was the reviewed cell's own (or no
    track's), the other part becomes a new track (``cell = None``), as
    before — the reviewer then links it where it belongs.

    The cut label may be a reviewer mask (a part of an earlier split); the
    cell's part then replaces it, and replay drops it from the extras.
    Both parts are measured afresh from the image.

    Returns:
        The written entries, the cell's part last.
    """
    item = session.item(plan.item_id)
    image, nuclei, names = _arrays(session, item.well)
    det, curated = session.source.well(item.well)
    cmap = channel_map(
        names, [c for c in det.columns if c not in _NOT_CHANNELS]
    )
    det_frame = _measured_rows(det, curated, plan.frame, [plan.label])
    new = next_label(det, curated)
    batch = (
        f"split-t{plan.frame}-L{plan.label}-"
        f"{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    what = "reviewer mask" if plan.from_extra else "nucleus"
    reason = f"split {what} {plan.label} at t{plan.frame} along a line"
    if plan.flipped:
        reason += " (cell part chosen by the reviewer)"
    parts = [
        (plan.cell, new, {"replaces": [plan.label]}),
        (plan.other, new + 1, {"replaces": [], "cell": plan.other_cell}),
    ]
    if plan.other_cell is not None:
        parts.reverse()
    items: list[tuple[str, dict[str, Any], str]] = []
    for part, label, extra in parts:
        meas = measure(
            image, nuclei, det_frame, plan.frame, plan.y0, plan.x0, part, cmap
        )
        meas["patch"] = save_patch(
            session.edits_path.parent / "masks",
            item.well,
            plan.frame,
            label,
            plan.y0,
            plan.x0,
            part,
        )
        items.append(
            (
                "mask_add",
                {
                    "frame": plan.frame,
                    "label": label,
                    "measurements": meas,
                    "batch": batch,
                    **extra,
                },
                reason,
            )
        )
    return session.edit_many(plan.item_id, items)


def describe(entry: Any) -> str:
    """One-line summary of a mask edit for logs."""
    return json.dumps(
        {k: entry.args.get(k) for k in ("frame", "label", "replaces")}
    )
