"""Correct a first tracking pass: measure, repair, relabel, track again.

Trackastra links detections; it cannot fix them. A nucleus that Cellpose
splits in two becomes a false division, a dying cell floating over a neighbour
becomes a swap, and debris becomes a stationary track. The FUCCI-gated repair
(:mod:`omero_screen_track.repair`) fixes most of this from the lineage and the
geminin marker alone. This module turns that repair into a segmentation and
tracking correction and tracks the corrected nuclei a second time:

1. Measure every first-pass nucleus: area, centroid and background-subtracted
   mean of each reporter (:func:`measure_labels`).
2. Hide debris (stationary, reporter-negative tracks) and run the repair. The
   result is a label map: debris to 0, and the pieces of one split nucleus to
   one label (:func:`correction_map`).
3. Track the relabelled nuclei again (:func:`correct_tracks` calls the
   ``track`` function it is given; the pipeline passes its Trackastra
   second pass, :func:`omero_screen.track_correction.retrack`). Trackastra
   now sees one detection per nucleus, so its own links improve.
4. Repair the second pass too, so every division in the final lineage has
   passed the geminin test.

The result is a **table**, not a mask: one row per first-pass nucleus
``(timepoint, label)`` with the final ``track_id`` (0 = hidden). The
first-pass masks stay as they were; :func:`relabel_frame` applies the table to
one frame when it is read. Merged pieces share a boundary with no gap, so the
union of their pixels is the corrected nucleus and every intensity measured on
it is exact.

The whole correction runs one frame at a time, so a 217-frame stitched well
fits in laptop memory when it is read from the zarr cache. Moved from
``omero_screen.track_correction`` (#75); Trackastra itself, and the
pipeline's stitched-well steps, stay there.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from omero_screen_track.debris import drop_debris
from omero_screen_track.repair import (
    RepairParams,
    RepairResult,
    repair_lineage,
)


class FrameStack(Protocol):
    """A ``(T, Y, X)`` array read one frame at a time (numpy, zarr, dask)."""

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        ...

    def __getitem__(self, key: Any, /) -> Any: ...  # noqa: D105


@dataclass(frozen=True)
class CorrectionParams:
    """Settings for :func:`correct_tracks`.

    Attributes:
        repair: Thresholds of the FUCCI-gated repair.
        hide_debris: Hide stationary, reporter-negative tracks.
        marker: Signal key of the PIP-FUCCI geminin reporter, used for the
            mitosis test.
        pip: Signal key of the PIP reporter. The mitosis and debris tests
            need both; without them the repair only merges fragments.
        mode: Trackastra linking mode of the second pass.
        batch_size: Trackastra batch size (``None`` = its default).
        window: Trackastra temporal window (``None`` = the model's).
    """

    repair: RepairParams = field(default_factory=RepairParams)
    hide_debris: bool = True
    marker: str = "geminin"
    pip: str = "pip"
    mode: str = "greedy"
    batch_size: int | None = None
    window: int | None = None


@dataclass
class CorrectionResult:
    """Outcome of :func:`correct_tracks` for one well.

    Attributes:
        table: One row per first-pass nucleus: ``timepoint``, ``label``
            (first-pass mask value) and ``track_id`` (final, 0 = hidden).
        parents: Final track id to its parent (0 = founder).
        events: Repair decisions of both passes (column ``pass``: 1 or 2).
        debris: First-pass track ids hidden as debris.
    """

    table: pd.DataFrame
    parents: dict[int, int]
    events: pd.DataFrame
    debris: set[int]

    def frame_map(self, t: int) -> dict[int, int]:
        """``{first-pass label: final track id}`` for frame ``t``."""
        rows = self.table[self.table["timepoint"] == t]
        return dict(
            zip(
                rows["label"].astype(int),
                rows["track_id"].astype(int),
                strict=True,
            )
        )


class Tracker(Protocol):
    """Tracks a label stack: ``(images, masks) -> (table, parents)``.

    ``table`` has one row per linked detection (``timepoint``, ``label``,
    ``track``); ``parents`` maps each track to its parent (0 = founder).
    """

    def __call__(
        self, images: FrameStack, masks: FrameStack, /
    ) -> tuple[pd.DataFrame, dict[int, int]]:
        """Track ``masks`` (with ``images``); return ``(table, parents)``."""
        ...


# --- measurement ---------------------------------------------------------


def measure_labels(
    masks: FrameStack,
    signals: Mapping[str, FrameStack],
    frames: Sequence[int] | None = None,
    cells: FrameStack | None = None,
) -> pd.DataFrame:
    """Area, centroid and mean signals of every labelled object, per frame.

    Each signal's background is the median of the frame's unlabelled pixels;
    the mean is reported with it subtracted and clipped at 0.

    Args:
        masks: Label stack ``(T, Y, X)``.
        signals: Name to intensity stack of the same shape.
        frames: Frames to measure (default: all).
        cells: Cell-mask stack; if given, a ``cell`` column holds the cell
            label covering most of each nucleus (0 if none).

    Returns:
        Columns ``timepoint``, ``label``, ``area``, ``y``, ``x``, one per
        signal and, with ``cells``, ``cell``.
    """
    n_frames, height, width = masks.shape[:3]
    yy, xx = np.indices((height, width), dtype=np.float64)
    yy, xx = yy.ravel(), xx.ravel()
    out: list[pd.DataFrame] = []
    for t in frames if frames is not None else range(n_frames):
        lab = np.asarray(masks[t]).ravel().astype(np.int64)
        n = int(lab.max()) + 1
        area = np.bincount(lab, minlength=n).astype(float)
        present = np.flatnonzero(area[1:]) + 1
        if present.size == 0:
            continue
        cols: dict[str, Any] = {
            "timepoint": t,
            "label": present,
            "area": area[present],
            "y": np.bincount(lab, yy, n)[present] / area[present],
            "x": np.bincount(lab, xx, n)[present] / area[present],
        }
        background = lab == 0
        for name, stack in signals.items():
            img = np.asarray(stack[t], dtype=np.float64).ravel()
            bg = float(np.median(img[background])) if background.any() else 0.0
            mean = np.bincount(lab, img, n)[present] / area[present]
            cols[name] = np.clip(mean - bg, 0, None)
        if cells is not None:
            cols["cell"] = _majority_cell(
                lab,
                np.asarray(cells[t]).ravel().astype(np.int64),
                present,
                area[present],
            )
        out.append(pd.DataFrame(cols))
    if not out:
        return pd.DataFrame(
            columns=["timepoint", "label", "area", "y", "x", *signals]
        )
    return pd.concat(out, ignore_index=True)


def _majority_cell(
    nuclei: npt.NDArray[np.int64],
    cells: npt.NDArray[np.int64],
    present: npt.NDArray[np.int64],
    areas: npt.NDArray[np.float64],
    cover: float = 0.5,
) -> npt.NDArray[np.int64]:
    """For each nucleus in ``present``, the cell covering most of it.

    A nucleus less than ``cover`` covered by its best cell gets 0: the cell
    segmentation missed it, and it is not judged by cell.
    """
    both = (nuclei > 0) & (cells > 0)
    out = np.zeros(present.size, dtype=np.int64)
    if not both.any():
        return out
    pairs, counts = np.unique(
        np.stack([nuclei[both], cells[both]]), axis=1, return_counts=True
    )
    order = np.lexsort((-counts, pairs[0]))
    pairs, counts = pairs[:, order], counts[order]
    first = np.r_[True, pairs[0, 1:] != pairs[0, :-1]]
    best = {
        int(n): (int(c), int(k))
        for n, c, k in zip(
            pairs[0, first], pairs[1, first], counts[first], strict=True
        )
    }
    for i, (n, a) in enumerate(zip(present, areas, strict=True)):
        cell, k = best.get(int(n), (0, 0))
        out[i] = cell if k >= cover * a else 0
    return out


def _contact_groups(
    yx: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
    stay_touch: float,
) -> list[int]:
    """Group index per object, objects in contact sharing a group (union-find)."""
    n = len(radii)
    group = list(range(n))

    def root(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.hypot(*(yx[i] - yx[j])))
            if dist <= stay_touch * (radii[i] + radii[j]):
                group[root(i)] = root(j)
    return [root(i) for i in range(n)]


def split_apart(
    table: pd.DataFrame,
    det: pd.DataFrame,
    stay_touch: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Never let one final label cover nuclei that are apart.

    A last guard over whatever the repair decided: in every frame, the
    first-pass nuclei given the same final track are grouped by contact
    (centroid distance at most ``stay_touch`` times the sum of their
    equivalent radii). The group with the most area keeps the track; every
    other group gets a track of its own, the same one in every frame for the
    same first-pass nucleus, as a founder.

    Args:
        table: ``timepoint, label, track_id`` (first-pass label to final track).
        det: First-pass measurements (``timepoint, label, area, y, x``).
        stay_touch: Contact threshold.

    Returns:
        ``(table, events)``; one ``apart`` event per split-off nucleus and frame.
    """
    d = table.merge(
        det[["timepoint", "label", "area", "y", "x"]],
        on=["timepoint", "label"],
        how="left",
    )
    d = d[d["track_id"] > 0]
    multi = d[d.duplicated(["timepoint", "track_id"], keep=False)]
    if multi.empty:
        return table, pd.DataFrame()
    next_id = int(table["track_id"].max()) + 1
    new_ids: dict[tuple[int, int], int] = {}
    changes: dict[tuple[int, int], int] = {}
    events = []
    for (t, tid), g in multi.groupby(["timepoint", "track_id"]):
        labels = g["label"].to_numpy()
        r = np.sqrt(g["area"].to_numpy() / np.pi)
        yx = g[["y", "x"]].to_numpy()
        n = len(labels)
        roots = _contact_groups(yx, r, stay_touch)
        if len(set(roots)) == 1:
            continue
        area = g["area"].to_numpy()
        keep = max(set(roots), key=lambda k: area[np.array(roots) == k].sum())
        for i, k in enumerate(roots):
            if k == keep:
                continue
            members = [int(labels[m]) for m in range(n) if roots[m] == k]
            key = (int(tid), min(members))
            if key not in new_ids:
                new_ids[key] = next_id
                next_id += 1
            changes[(int(t), int(labels[i]))] = new_ids[key]
            events.append(
                {
                    "pass": 3,
                    "parent": int(tid),
                    "children": str(new_ids[key]),
                    "frame": int(t),
                    "rule": "apart",
                    "y": float(yx[i][0]),
                    "x": float(yx[i][1]),
                }
            )
    if not changes:
        return table, pd.DataFrame()
    keys = list(zip(table["timepoint"], table["label"], strict=True))
    out = table.assign(
        track_id=[
            changes.get((int(t), int(lab)), tid)
            for (t, lab), tid in zip(keys, table["track_id"], strict=True)
        ]
    )
    logger.info(
        f"Guard: {len(changes)} nuclei in {len(new_ids)} groups split off "
        "a track that also held a distant nucleus"
    )
    return out, pd.DataFrame(events)


def merge_detections(
    det: pd.DataFrame, new_label: pd.Series, signals: Sequence[str]
) -> pd.DataFrame:
    """Combine detections that share a new label in a frame.

    Merged pieces of one nucleus are its exact union: areas add, centroids and
    mean signals are area-weighted.

    Args:
        det: Output of :func:`measure_labels`.
        new_label: New label per row of ``det`` (0 drops the row).
        signals: Signal columns to carry.
    """
    d = det.assign(label=new_label.to_numpy())
    d = d[d["label"] != 0]
    weighted = ["y", "x", *signals]
    d = d.assign(**{c: d[c] * d["area"] for c in weighted})
    out = d.groupby(["timepoint", "label"], as_index=False)[
        ["area", *weighted]
    ].sum()
    for c in weighted:
        out[c] = out[c] / out["area"]
    return out


# --- correction ----------------------------------------------------------


def _lineage_table(
    det: pd.DataFrame, track: pd.Series, parents: Mapping[int, int]
) -> pd.DataFrame:
    """The detections in the shape :func:`repair_lineage` reads."""
    track = track.astype(int)
    return det.assign(
        track_id_raw=track.to_numpy(),
        parent_track_id_raw=track.map(lambda t: parents.get(t, 0)).to_numpy(),
    )


def _repair(det: pd.DataFrame, params: CorrectionParams) -> RepairResult:
    """Run the repair, geminin-gated only if the detections are PIP-FUCCI."""
    if {params.marker, params.pip} <= set(det.columns):
        det = det.assign(marker=det[params.marker])
    else:
        logger.info("Not PIP-FUCCI: repair merges fragments only.")
    return repair_lineage(det, params.repair)


def correction_map(
    det: pd.DataFrame,
    parents: Mapping[int, int],
    params: CorrectionParams | None = None,
) -> tuple[dict[int, int], RepairResult, set[int]]:
    """Map first-pass track ids to corrected labels.

    Args:
        det: Measurements of the first-pass masks, whose labels are track ids.
        parents: First-pass lineage, track id to parent (0 = founder).
        params: Settings; defaults to :class:`CorrectionParams`.

    Returns:
        ``(label map, repair result, debris ids)``. The map sends debris to 0
        and every other track to its repaired id.
    """
    params = params or CorrectionParams()
    lin = _lineage_table(det, det["label"], parents)
    debris: set[int] = set()
    if params.hide_debris and {params.marker, params.pip} <= set(det.columns):
        lin, debris = drop_debris(
            lin.rename(columns={params.marker: "geminin", params.pip: "pip"})
        )
        lin = lin.rename(columns={"geminin": params.marker, "pip": params.pip})
    result = _repair(lin, params)
    mapping = {int(t): 0 for t in debris}
    mapping.update({int(k): int(v) for k, v in result.assignment.items()})
    return mapping, result, debris


# --- second pass ---------------------------------------------------------


class LazyFrames:
    """A ``(T, Y, X)`` stack that computes each frame when it is read.

    Trackastra reads its inputs one frame at a time, so wrapping the zarr
    cache (or an in-memory stack) in this keeps one frame in memory.
    """

    def __init__(
        self,
        source: FrameStack,
        fn: Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
        dtype: npt.DTypeLike,
    ) -> None:
        """Wrap ``source``; frame ``t`` is ``fn(source[t])`` cast to ``dtype``."""
        self.source = source
        self.fn = fn
        self.dtype = np.dtype(dtype)
        self.shape = tuple(source.shape[:3])
        self.ndim = 3

    def __len__(self) -> int:
        """Number of frames."""
        return self.shape[0]

    def __getitem__(self, t: int) -> npt.NDArray[Any]:
        """Frame ``t``."""
        return np.asarray(self.fn(np.asarray(self.source[t])), self.dtype)

    def __iter__(self) -> Iterator[npt.NDArray[Any]]:
        """Frames in order."""
        return (self[t] for t in range(len(self)))


def normalise_stack(images: FrameStack, subsample: int = 4) -> LazyFrames:
    """Trackastra's percentile normalisation, computed lazily per frame.

    Same arithmetic as ``trackastra.utils.normalize`` on the whole stack: the
    1st and 99.8th percentiles over every frame subsampled by ``subsample``.
    """
    n_frames = images.shape[0]
    sample = np.stack(
        [
            np.asarray(images[t], dtype=np.float32)[::subsample, ::subsample]
            for t in range(n_frames)
        ]
    )
    lo, hi = np.percentile(sample, (1, 99.8)).astype(np.float32)
    del sample
    scale = np.float32(hi - lo + 1e-8)
    return LazyFrames(
        images, lambda f: (f.astype(np.float32) - lo) / scale, np.float32
    )


def correct_tracks(
    images: FrameStack,
    masks: FrameStack,
    parents: Mapping[int, int],
    signals: Mapping[str, FrameStack],
    track: Tracker,
    params: CorrectionParams | None = None,
    det: pd.DataFrame | None = None,
) -> CorrectionResult:
    """Correct a first tracking pass and track the corrected nuclei again.

    Args:
        images: Nucleus channel, ``(T, Y, X)``.
        masks: First-pass nucleus masks; pixel value = first-pass track id.
        parents: First-pass lineage (track id to parent, 0 = founder).
        signals: Reporter stacks by name (e.g. ``{"geminin": ..., "pip": ...}``).
        track: Tracks the corrected nuclei: called with the nucleus
            images and the relabelled masks, returns ``(table, parents)``
            as :func:`omero_screen.track_correction.retrack` does.
        params: Settings; defaults to :class:`CorrectionParams`.
        det: Precomputed :func:`measure_labels` of ``masks`` and ``signals``.

    Returns:
        The correction table, final lineage, repair events and debris ids.
    """
    params = params or CorrectionParams()
    names = list(signals)
    if det is None:
        logger.info("Measuring first-pass nuclei")
        det = measure_labels(masks, signals)

    mapping, first, debris = correction_map(det, parents, params)
    lut = np.zeros(max([*mapping, int(det["label"].max())]) + 1, np.uint32)
    for raw, new in mapping.items():
        lut[raw] = new
    merged = int((det["label"].map(mapping) != det["label"]).sum())
    logger.info(
        f"Pass 1: {len(debris)} debris tracks hidden, "
        f"{len(first.events)} repair events, {merged} detections relabelled"
    )

    corrected = LazyFrames(masks, lambda f: lut[f], np.uint32)
    logger.info("Pass 2: tracking the corrected nuclei")
    tracked, parents2 = track(images, corrected)

    det1 = det.assign(corrected=lut[det["label"].to_numpy()])
    det2 = merge_detections(det, det1["corrected"], names)
    det2 = det2.merge(tracked, on=["timepoint", "label"], how="left")
    # Trackastra's greedy solver keeps only linked detections; an isolated
    # one is dropped, as graph_to_ctc drops it from the first-pass mask.
    untracked = det2["track"].isna()
    if untracked.any():
        logger.info(
            f"Pass 2: {int(untracked.sum())} unlinked nuclei hidden "
            "(Trackastra drops detections without links)"
        )
        det2 = det2[~untracked]
    second = _repair(_lineage_table(det2, det2["track"], parents2), params)

    final = det2.assign(
        track_id=det2["track"].astype(int).map(second.assignment)
    )[["timepoint", "label", "track_id"]].rename(
        columns={"label": "corrected"}
    )
    table = det1[["timepoint", "label", "corrected"]].merge(
        final, on=["timepoint", "corrected"], how="left"
    )
    table["track_id"] = table["track_id"].fillna(0).astype(np.int64)
    table, apart = split_apart(table, det, params.repair.stay_touch)
    events = pd.concat(
        [
            first.events.assign(**{"pass": 1}),
            second.events.assign(**{"pass": 2}),
            apart,
        ],
        ignore_index=True,
    )
    n_final = table.loc[table["track_id"] > 0, "track_id"].nunique()
    logger.info(
        f"Pass 2: {len(parents2)} Trackastra tracks, {len(second.events)} "
        f"repair events, {n_final} final tracks"
    )
    parents_out = {int(k): int(v) for k, v in second.parents.items()}
    for tid in table["track_id"].unique():
        parents_out.setdefault(int(tid), 0)
    parents_out.pop(0, None)
    return CorrectionResult(
        table=table[["timepoint", "label", "track_id"]],
        parents=parents_out,
        events=events,
        debris=debris,
    )


def relabel_frame(
    frame: npt.NDArray[Any], mapping: Mapping[int, int]
) -> npt.NDArray[np.uint32]:
    """Apply a ``{first-pass label: final track id}`` map to one mask frame.

    Labels absent from the map (hidden or never seen) become 0.
    """
    frame = np.asarray(frame)
    top = max(int(frame.max()), max(mapping, default=0))
    lut = np.zeros(top + 1, np.uint32)
    for raw, new in mapping.items():
        lut[raw] = new
    out: npt.NDArray[np.uint32] = lut[frame]
    return out


def relabel_stack(
    masks: npt.NDArray[Any], result: CorrectionResult
) -> npt.NDArray[Any]:
    """Apply a correction to a whole ``(T, Y, X)`` mask stack, in place.

    Frame by frame, so no second full-size stack is allocated. Labels the
    table does not list (and frames it does not cover) become 0. Only if the
    mask's integer type cannot hold the largest final track id is the stack
    first copied to ``uint32``.

    Returns:
        The stack holding the final track ids (``masks`` itself, unless it
        had to be widened).
    """
    top = int(result.table["track_id"].max()) if len(result.table) else 0
    if top > np.iinfo(masks.dtype).max:
        masks = masks.astype(np.uint32)
    table = result.table
    by_frame: dict[int, dict[int, int]] = {}
    for t, label, tid in zip(
        table["timepoint"].to_numpy(dtype=np.int64),
        table["label"].to_numpy(dtype=np.int64),
        table["track_id"].to_numpy(dtype=np.int64),
        strict=True,
    ):
        by_frame.setdefault(int(t), {})[int(label)] = int(tid)
    for t in range(masks.shape[0]):
        masks[t] = relabel_frame(masks[t], by_frame.get(t, {}))
    return masks


# --- storage as OMERO well attachments -----------------------------------

#: Attachment titles (``<title>.csv``) of a stored correction, by table.
CORRECTION_ATTACHMENTS = {
    "table": "track_correction",
    "lineage": "track_lineage",
    "events": "track_events",
}


def correction_tables(
    result: CorrectionResult, pass1: Mapping[int, int]
) -> dict[str, pd.DataFrame]:
    """A correction as flat tables, for storage beside the first-pass masks.

    Args:
        result: The correction.
        pass1: First-pass lineage the correction started from.

    Returns:
        ``table`` (``timepoint, label, track_id``), ``lineage`` (``pass``
        ``first`` or ``final``, ``track_id``, ``parent_track_id``, ``debris``)
        and ``events`` (the repair decisions), keyed as
        :data:`CORRECTION_ATTACHMENTS`.
    """
    first = pd.DataFrame(
        {
            "pass": "first",
            "track_id": [int(k) for k in pass1],
            "parent_track_id": [int(v) for v in pass1.values()],
        }
    )
    first["debris"] = first["track_id"].isin(sorted(result.debris))
    final = pd.DataFrame(
        {
            "pass": "final",
            "track_id": [int(k) for k in result.parents],
            "parent_track_id": [int(v) for v in result.parents.values()],
            "debris": False,
        }
    )
    return {
        "table": result.table[["timepoint", "label", "track_id"]],
        "lineage": pd.concat([first, final], ignore_index=True),
        "events": result.events,
    }


def correction_from_tables(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[CorrectionResult, dict[int, int]]:
    """Inverse of :func:`correction_tables`.

    Returns:
        ``(correction, first-pass lineage)``.
    """
    lineage = tables["lineage"]

    def parent_map(rows: pd.DataFrame) -> dict[int, int]:
        return dict(
            zip(
                rows["track_id"].astype(int),
                rows["parent_track_id"].astype(int),
                strict=True,
            )
        )

    first = lineage[lineage["pass"] == "first"]
    debris = first.loc[first["debris"].astype(bool), "track_id"]
    result = CorrectionResult(
        table=tables["table"][["timepoint", "label", "track_id"]],
        parents=parent_map(lineage[lineage["pass"] == "final"]),
        events=tables.get("events", pd.DataFrame()),
        debris={int(t) for t in debris},
    )
    return result, parent_map(first)
