"""The pipeline's track correction: Trackastra second pass and stitched wells.

The correction itself — measuring first-pass nuclei, the FUCCI-gated repair,
the relabel table and its storage — lives in :mod:`omero_screen_track.correction`
(#75). This module supplies what needs the pipeline: the Trackastra second
pass (:func:`retrack`), :func:`correct_tracks` with a Trackastra model, the
in-memory correction of a stitched well (:func:`correct_stitched_well`) and
measuring the corrected nuclei with the pipeline's feature extraction
(:func:`measure_corrected_well`).

The names that moved are re-exported here for one release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from omero_screen_track import correction as _correction

# Re-exported: these moved to omero_screen_track.correction (#75).
from omero_screen_track.correction import (  # noqa: F401
    CORRECTION_ATTACHMENTS,
    CorrectionParams,
    CorrectionResult,
    FrameStack,
    LazyFrames,
    correction_from_tables,
    correction_map,
    correction_tables,
    measure_labels,
    merge_detections,
    normalise_stack,
    relabel_frame,
    relabel_stack,
    split_apart,
)

from omero_screen.tracking import VALID_TRACKING_MODES, _configure_window

if TYPE_CHECKING:
    import networkx as nx
    from trackastra.model import Trackastra


# --- second pass ---------------------------------------------------------


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
        imgs,  # ty: ignore[invalid-argument-type]
        masks,  # ty: ignore[invalid-argument-type]
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
    """Correct a first tracking pass and track again with a Trackastra model.

    :func:`omero_screen_track.correction.correct_tracks` with :func:`retrack`
    as its second pass, run with the mode, batch size and window in
    ``params``.

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

    def track(
        imgs: FrameStack, corrected: FrameStack
    ) -> tuple[pd.DataFrame, dict[int, int]]:
        return retrack(
            imgs,
            corrected,
            model,
            params.mode,
            params.batch_size,
            params.window,
        )

    return _correction.correct_tracks(
        images, masks, parents, signals, track, params, det
    )


# --- the pipeline's stitched route ----------------------------------------


def correct_stitched_well(
    stitched_img: npt.NDArray[Any],
    n_mask: npt.NDArray[Any],
    c_mask: npt.NDArray[Any] | None,
    parents: Mapping[int, int],
    channel_order: Sequence[str],
    nucleus_idx: int,
    model: Trackastra,
    params: CorrectionParams | None = None,
) -> CorrectionResult:
    """Correct a stitched well's first tracking pass in the pipeline.

    The in-memory counterpart of the zarr-cache correction
    (``omero_screen_napari.zarr_cache.correction.correct_cached_well``): the
    same measurement, repair and second Trackastra pass, on the canvas the
    pipeline has just tracked. The geminin-gated repair and the debris test
    run only on PIP-FUCCI plates
    (:func:`~omero_screen.metadata_parser.resolve_pip_fucci`); on any other
    plate the repair merges fragments only.

    Args:
        stitched_img: Canvas ``(T, Y, X, C)``.
        n_mask: First-pass nucleus masks ``(T, Y, X)``; pixel value = track id.
        c_mask: Cell masks ``(T, Y, X)``, or ``None``.
        parents: First-pass lineage (track id to parent, 0 = founder).
        channel_order: Channel names in the canvas' last-axis order.
        nucleus_idx: Index of the nucleus channel in ``channel_order``.
        model: The Trackastra model of the first pass.
        params: Correction settings.

    Returns:
        The correction table, final lineage, repair events and debris ids.
    """
    from omero_utils.channels import resolve_pip_fucci

    names = list(channel_order)
    fucci = resolve_pip_fucci(names, stitched_img.shape[0])
    signals = (
        {
            "geminin": stitched_img[..., names.index(fucci.geminin)],
            "pip": stitched_img[..., names.index(fucci.pip)],
        }
        if fucci
        else {}
    )
    logger.info("Measuring first-pass nuclei for the track correction")
    det = measure_labels(n_mask, signals, cells=c_mask)
    return correct_tracks(
        stitched_img[..., nucleus_idx],
        n_mask,
        parents,
        signals,
        model,
        params,
        det,
    )


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
