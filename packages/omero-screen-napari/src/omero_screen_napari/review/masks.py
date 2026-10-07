"""Reviewer-made nucleus masks: split a nucleus, draw a missed one.

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
    """Detection column → image channel index, matched by case-insensitive prefix."""
    lower = [c.lower() for c in image_channels]
    out = {}
    for col in det_columns:
        for i, name in enumerate(lower):
            if name == col or name.startswith(col) or col.startswith(name):
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


def describe(entry: Any) -> str:
    """One-line summary of a mask edit for logs."""
    return json.dumps(
        {k: entry.args.get(k) for k in ("frame", "label", "replaces")}
    )
