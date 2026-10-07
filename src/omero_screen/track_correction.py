"""Correct a first tracking pass and track again.

Trackastra links detections; it cannot fix them. A nucleus that Cellpose
splits in two becomes a false division, a dying cell floating over a neighbour
becomes a swap, and debris becomes a stationary track. The FUCCI-gated repair
(:mod:`cellview.tracks.repair`) fixes most of this from the lineage and the
geminin marker alone. This module turns that repair into a segmentation and
tracking correction and tracks the corrected nuclei a second time:

1. Measure every first-pass nucleus: area, centroid and background-subtracted
   mean of each reporter (:func:`measure_labels`).
2. Hide debris (stationary, reporter-negative tracks) and run the repair. The
   result is a label map: debris to 0, and the pieces of one split nucleus to
   one label (:func:`correction_map`).
3. Track the relabelled nuclei again with the same Trackastra model
   (:func:`retrack`). Trackastra now sees one detection per nucleus, so its
   own links improve.
4. Repair the second pass too, so every division in the final lineage has
   passed the geminin test.

The result is a **table**, not a mask: one row per first-pass nucleus
``(timepoint, label)`` with the final ``track_id`` (0 = hidden). The
first-pass masks stay as they were; :func:`relabel_frame` applies the table to
one frame when it is read. Merged pieces share a boundary with no gap, so the
union of their pixels is the corrected nucleus and every intensity measured on
it is exact.

The whole correction runs one frame at a time, so a 217-frame stitched well
fits in laptop memory when it is read from the zarr cache.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from cellview.tracks.debris import drop_debris
from cellview.tracks.repair import RepairParams, RepairResult, repair_lineage
from loguru import logger

from omero_screen.tracking import VALID_TRACKING_MODES, _configure_window

if TYPE_CHECKING:
    import networkx as nx
    from trackastra.model import Trackastra


class FrameStack(Protocol):
    """A ``(T, Y, X)`` array read one frame at a time (numpy, zarr, dask)."""

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        ...

    def __getitem__(self, key: Any) -> Any: ...  # noqa: D105


@dataclass(frozen=True)
class CorrectionParams:
    """Settings for :func:`correct_tracks`.

    Attributes:
        repair: Thresholds of the FUCCI-gated repair.
        hide_debris: Hide stationary, reporter-negative tracks.
        marker: Signal used for the mitosis test (geminin with PIP-FUCCI).
            Without it the repair only merges fragments.
        pip: Second reporter, used with ``marker`` for the debris test.
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


# --- measurement ---------------------------------------------------------


def measure_labels(
    masks: FrameStack,
    signals: Mapping[str, FrameStack],
    frames: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Area, centroid and mean signals of every labelled object, per frame.

    Each signal's background is the median of the frame's unlabelled pixels;
    the mean is reported with it subtracted and clipped at 0.

    Args:
        masks: Label stack ``(T, Y, X)``.
        signals: Name to intensity stack of the same shape.
        frames: Frames to measure (default: all).

    Returns:
        Columns ``timepoint``, ``label``, ``area``, ``y``, ``x`` and one per
        signal.
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
        out.append(pd.DataFrame(cols))
    if not out:
        return pd.DataFrame(
            columns=["timepoint", "label", "area", "y", "x", *signals]
        )
    return pd.concat(out, ignore_index=True)


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
    """Run the repair, using the marker if the detections have it."""
    if params.marker in det:
        det = det.assign(marker=det[params.marker])
    else:
        logger.info(
            f"No '{params.marker}' signal: repair merges fragments only."
        )
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


def graph_table(
    graph: nx.DiGraph,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Track ids of a Trackastra solution graph, as ``graph_to_ctc`` numbers them.

    Returns the per-detection table (``timepoint``, ``label``, ``track``)
    without building a relabelled mask, and the parent map.
    """
    from trackastra.tracking.utils import ctc_tracklets

    rows: list[tuple[int, int, int]] = []
    node_track: dict[int, int] = {-1: 0}
    parents: dict[int, int] = {}
    for i, tracklet in enumerate(sorted(ctc_tracklets(graph, "time"))):
        track = i + 1
        node_track[tracklet.nodes[-1]] = track
        parents[track] = node_track[tracklet.parent]
        for n in tracklet.nodes:
            node = graph.nodes[n]
            rows.append((int(node["time"]), int(node["label"]), track))
    table = pd.DataFrame(rows, columns=["timepoint", "label", "track"])
    return table, parents


def retrack(
    images: FrameStack,
    masks: FrameStack,
    model: Trackastra,
    mode: str = "greedy",
    batch_size: int | None = None,
    window: int | None = None,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Track a label stack with Trackastra, one frame in memory at a time.

    Same model and arithmetic as :func:`omero_screen.tracking.track_nucleus_mask`,
    but the images are normalised lazily and the solution is returned as a
    table (:func:`graph_table`) rather than a relabelled mask.

    Args:
        images: Nucleus channel, ``(T, Y, X)``.
        masks: Labels, ``(T, Y, X)``.
        model: A model from :func:`omero_screen.tracking.load_tracking_model`.
        mode: Linking mode.
        batch_size: Trackastra batch size.
        window: Temporal window override.

    Returns:
        ``(table, parents)`` as from :func:`graph_table`.
    """
    if mode not in VALID_TRACKING_MODES:
        raise ValueError(f"Unknown tracking mode {mode!r}.")
    per_frame = [
        int(np.unique(np.asarray(masks[t])).size - 1)
        for t in range(masks.shape[0])
    ]
    _configure_window(model, per_frame, window)
    imgs = normalise_stack(images)
    # Trackastra.track would also build a relabelled copy of the full mask;
    # its two steps are called directly to skip that (trackastra pinned 0.5.3).
    predictions = model._predict(
        imgs,  # type: ignore[arg-type]
        masks,  # type: ignore[arg-type]
        normalize_imgs=False,
        batch_size=batch_size or model.batch_size,
    )
    graph = model._track_from_predictions(predictions, mode=mode)
    return graph_table(graph)


def correct_tracks(
    images: FrameStack,
    masks: FrameStack,
    parents: Mapping[int, int],
    signals: Mapping[str, FrameStack],
    model: Trackastra,
    params: CorrectionParams | None = None,
    det: pd.DataFrame | None = None,
) -> CorrectionResult:
    """Correct a first tracking pass and track the corrected nuclei again.

    Args:
        images: Nucleus channel, ``(T, Y, X)``.
        masks: First-pass nucleus masks; pixel value = first-pass track id.
        parents: First-pass lineage (track id to parent, 0 = founder).
        signals: Reporter stacks by name (e.g. ``{"geminin": ..., "pip": ...}``).
        model: Trackastra model for the second pass.
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
    tracked, parents2 = retrack(
        images,
        corrected,
        model,
        params.mode,
        params.batch_size,
        params.window,
    )

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
    events = pd.concat(
        [
            first.events.assign(**{"pass": 1}),
            second.events.assign(**{"pass": 2}),
        ],
        ignore_index=True,
    )
    n_final = table.loc[table["track_id"] > 0, "track_id"].nunique()
    logger.info(
        f"Pass 2: {len(parents2)} Trackastra tracks, {len(second.events)} "
        f"repair events, {n_final} final tracks"
    )
    return CorrectionResult(
        table=table[["timepoint", "label", "track_id"]],
        parents={int(k): int(v) for k, v in second.parents.items()},
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


# --- measurement of the corrected nuclei ---------------------------------


def measure_corrected_well(
    well: Any,
    metadata: Any,
    channels: Mapping[str, FrameStack],
    n_mask: FrameStack,
    c_mask: FrameStack | None,
    result: CorrectionResult,
    nucleus_channel: str,
    cell_channel: str | None,
    field_image_ids: list[int],
    field_offsets: npt.NDArray[np.int_],
    tile_h: int,
    tile_w: int,
) -> pd.DataFrame:
    """Measure the corrected nuclei with the pipeline's own feature extraction.

    Each frame's first-pass mask is relabelled with the correction table, the
    cytoplasm is derived from the corrected nuclei, and the frame goes through
    :class:`~omero_screen.image_analysis.ImageProperties` exactly as a
    stitched, tracked well does in ``omero_screen.loops``. Every operation
    there is per frame, so measuring frame by frame gives the same values as
    one call on the whole stack while holding one frame in memory.

    Args:
        well: OMERO ``WellWrapper``.
        metadata: The plate's ``MetadataParser``.
        channels: Channel name (as in ``metadata.channel_data``, in order) to
            its stitched ``(T, Y, X)`` stack.
        n_mask: First-pass nucleus masks.
        c_mask: Cell masks, or ``None``.
        result: The correction for this well.
        nucleus_channel: Nucleus channel name.
        cell_channel: Cell channel name, or ``None``.
        field_image_ids: OMERO image id of every field, in well-sample order.
        field_offsets: Canvas offset of every field.
        tile_h: Field height in pixels.
        tile_w: Field width in pixels.

    Returns:
        The well's measurement table with the final track columns, as the
        pipeline writes it to ``final_data.csv``.
    """
    from omero_screen.image_analysis import ImageProperties, StitchedWellImage
    from omero_screen.loops import _stitched_cyto
    from omero_screen.tracking import add_track_columns

    names = list(channels)
    by_frame = {
        int(t): result.frame_map(int(t))
        for t in result.table["timepoint"].unique()
    }
    frames: list[pd.DataFrame] = []
    for t in range(n_mask.shape[0]):
        nuclei = relabel_frame(n_mask[t], by_frame.get(t, {}))[np.newaxis]
        cells = None if c_mask is None else np.asarray(c_mask[t])[np.newaxis]
        cyto = None if cells is None else _stitched_cyto(nuclei, cells)
        stack = np.stack([np.asarray(channels[ch][t]) for ch in names], -1)
        image = StitchedWellImage(
            stitched_img=stack[np.newaxis],
            stitched_mask=nuclei,
            channels={ch: i for i, ch in enumerate(names)},
            nucleus_channel=nucleus_channel,
            well_pos=well.getWellPos(),
            synthetic_image_id=field_image_ids[0],
            c_mask=cells,
            cyto_mask=cyto,
            cell_channel=cell_channel,
            field_image_ids=field_image_ids,
            field_offsets=field_offsets,
            tile_h=tile_h,
            tile_w=tile_w,
        )
        props = ImageProperties(
            well,
            image,  # type: ignore[arg-type]  # duck-types Image
            metadata,
            keep_unmatched_nuclei=True,
        )
        frames.append(props.image_df.assign(timepoint=t))
        if t % 20 == 0:
            logger.info(f"{well.getWellPos()}: measured frame {t}")
    df = pd.concat(frames, ignore_index=True)
    add_track_columns(df, result.parents)
    df["stitch_mode"] = True
    return df
