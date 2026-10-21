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
from dataclasses import dataclass
from datetime import datetime
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
    """``(y0, x0, mask)`` of a stored patch."""
    data = np.load(base_dir / rel)
    return int(data["y0"]), int(data["x0"]), data["mask"]


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


def commit_split(
    session: Any,
    item_id: str,
    frame: int,
    label: int,
    seeds: list[tuple[int, int]],
) -> list[Any]:
    """Split raw nucleus ``label`` in ``frame`` at canvas ``seeds``; part 1 stays with the cell.

    Returns the two ``mask_add`` edit-log entries.
    """
    item = session.item(item_id)
    image, nuclei, names = _arrays(session, item.well)
    det, curated = session.source.well(item.well)
    row = det[(det["timepoint"] == frame) & (det["label"] == label)]
    if row.empty:
        raise MaskError(f"No nucleus {label} in frame {frame}.")
    half = int(np.sqrt(float(row["area"].iloc[0]))) * 2 + 8
    cy, cx = (
        int(round(float(row["y"].iloc[0]))),
        int(round(float(row["x"].iloc[0]))),
    )
    y0, x0 = max(cy - half, 0), max(cx - half, 0)
    region = np.asarray(nuclei[frame, y0 : cy + half, x0 : cx + half]) == label
    parts = split_mask(region, [(y - y0, x - x0) for y, x in seeds])
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
    cmap = channel_map(names, cols)
    det_frame = det[(det["timepoint"] == frame) & (det["label"] != label)]
    entries, new = [], next_label(det, curated)
    for i, part in enumerate(parts):
        meas = measure(image, nuclei, det_frame, frame, y0, x0, part, cmap)
        meas["patch"] = save_patch(
            session.edits_path.parent / "masks",
            item.well,
            frame,
            new + i,
            y0,
            x0,
            part,
        )
        args = {
            "frame": frame,
            "label": new + i,
            "measurements": meas,
            "replaces": [label] if i == 0 else [],
        }
        if i > 0:
            args["cell"] = None
        entries.append(
            session.edit(
                item_id,
                "mask_add",
                args,
                reason=f"split nucleus {label} at t{frame}",
            )
        )
    return entries


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
    det_frame = det[(det["timepoint"] == frame) & ~det["label"].isin(replaces)]
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
        det_frame = det[
            (det["timepoint"] == t) & ~det["label"].isin(fs.replaces)
        ]
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


def describe(entry: Any) -> str:
    """One-line summary of a mask edit for logs."""
    return json.dumps(
        {k: entry.args.get(k) for k in ("frame", "label", "replaces")}
    )
