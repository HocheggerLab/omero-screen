"""Temporal tracking of segmented nuclei with Trackastra.

This module wraps `Trackastra <https://github.com/weigertlab/trackastra>`_
(Weigert lab, ECCV 2024) so that nuclei segmented by Cellpose carry a stable
``track_id`` across the time axis of a live-cell timelapse.

Trackastra is *track-by-detection*: it links existing per-frame segmentation
masks and needs the full ``(T, Y, X)`` stack of image and mask in memory at
once. The natural insertion point in the pipeline is therefore between
segmentation and feature extraction, while the full stack is resident — see
``omero_screen.loops``.

Only the **nucleus** mask is relabelled. Cell and cytoplasm measurements
inherit the track id automatically because ``ImageProperties`` associates
nuclei to cells *spatially* (``_overlay_mask``), keying the merged dataframe on
the nucleus ``label`` — which, post-relabel, equals the ``track_id``.

Main Functions:
    - load_tracking_model: Load a pretrained Trackastra model once per run.
    - track_nucleus_mask: Relabel a nucleus mask in place with track ids and
      return the lineage (parent) map.
    - add_track_columns: Derive the ``track_id`` / ``parent_track_id`` (+
      ``_raw``) columns on a per-well measurements dataframe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from omero_screen.config import get_logger
from omero_screen.torch import get_device

if TYPE_CHECKING:
    from trackastra.model import Trackastra

logger = get_logger(__name__)

# Linking modes accepted by Trackastra.track. ``ilp`` additionally requires the
# optional ``motile`` + Gurobi/SCIP stack; guard its use at call time.
VALID_TRACKING_MODES = ("greedy", "greedy_nodiv", "ilp")


class TrackingResult(NamedTuple):
    """Result of tracking a nucleus mask.

    Attributes:
        nucleus_mask: Relabelled nucleus mask of shape ``(T, Y, X)`` whose
            pixel values are ``track_id`` (CTC convention — a division spawns
            two new track ids that point back to the parent).
        parent_map: Maps every ``track_id`` to its parent ``track_id``; founder
            tracks map to ``0``. Empty when tracking was a no-op (``T == 1``).
    """

    nucleus_mask: npt.NDArray[Any]
    parent_map: dict[int, int]


def load_tracking_model(
    model_name: str, device: str | None = None
) -> Trackastra:
    """Load a pretrained Trackastra model.

    Call once per run (mirrors the one-time Cellpose/inference model setup) and
    pass the returned model into :func:`track_nucleus_mask`.

    Args:
        model_name: Pretrained model name (e.g. ``"general_2d"``, ``"ctc"``) or
            a path to a fine-tuned checkpoint directory.
        device: Force a torch device — ``"cpu"`` or ``"cuda"``. ``None``
            auto-detects (CUDA if available, else CPU). Trackastra's attention
            builds a dense ``(heads, N, N)`` spatial-bias matrix per window
            (``N`` = detections summed over the window's frames), so a dense
            stitched well can exceed GPU VRAM regardless of batch size; forcing
            ``"cpu"`` runs the *identical* computation in host RAM (slower, but
            no 44 GiB ceiling and no loss of accuracy). See
            ``OMERO_SCREEN_TRACKING_DEVICE``.

    Returns:
        A loaded :class:`trackastra.model.Trackastra` instance.
    """
    from trackastra.model import Trackastra

    if device is None:
        # Trackastra accepts "cuda"/"cpu"; it has no MPS kernels, so fall back
        # to CPU on Apple silicon rather than crashing.
        device = str(get_device())
    if device == "mps":
        logger.info("Trackastra has no MPS backend; using CPU instead.")
        device = "cpu"

    logger.info("Loading Trackastra model '%s' on %s", model_name, device)
    return Trackastra.from_pretrained(model_name, device=device)


def track_nucleus_mask(
    image_stack: npt.NDArray[Any],
    nucleus_mask: npt.NDArray[Any],
    model: Trackastra,
    mode: str = "greedy",
    batch_size: int | None = None,
    window: int | None = None,
) -> TrackingResult:
    """Track nuclei across time and relabel the mask with stable track ids.

    Args:
        image_stack: Nucleus-channel intensity, shape ``(T, Y, X)``.
        nucleus_mask: Per-frame Cellpose nucleus labels, shape ``(T, Y, X)``.
        model: A model from :func:`load_tracking_model`.
        mode: Linking mode — one of :data:`VALID_TRACKING_MODES`.
        batch_size: Number of attention windows the transformer scores per
            forward pass. ``None`` defers to Trackastra's own default (1 on CPU,
            16 on GPU). Note this is only a memory lever when there are *many*
            windows; for a short timelapse the window count collapses to one or
            two (see ``window``) and batch size has no effect. See
            ``OMERO_SCREEN_TRACKING_BATCH_SIZE``.
        window: Override the temporal window (frames concatenated into one
            attention window). The dense ``(heads, N, N)`` spatial-bias matrix
            scales as ``N²`` where ``N ≈ window × detections_per_frame``, so a
            smaller window cuts GPU memory roughly quadratically — the effective
            lever for fitting a dense well on the GPU. Trade-off: less temporal
            context (weaker division / gap-closing inference); minor for
            ``greedy`` frame-to-frame linking. ``None`` keeps the model's
            trained window. See ``OMERO_SCREEN_TRACKING_WINDOW``.

    Returns:
        A :class:`TrackingResult` with the relabelled nucleus mask and the
        parent map.

    Raises:
        ValueError: If ``mode`` is not a recognised Trackastra linking mode or
            the input shapes do not match.
    """
    if mode not in VALID_TRACKING_MODES:
        raise ValueError(
            f"Unknown tracking mode {mode!r}; expected one of "
            f"{VALID_TRACKING_MODES}."
        )
    if image_stack.shape != nucleus_mask.shape:
        raise ValueError(
            "image_stack and nucleus_mask must have the same shape, got "
            f"{image_stack.shape} and {nucleus_mask.shape}."
        )

    # T == 1: nothing to link. Leave the mask untouched so a single-timepoint
    # (fixed-cell) run is byte-identical whether or not tracking is enabled.
    if nucleus_mask.shape[0] == 1:
        logger.debug("Single timepoint — tracking is a no-op.")
        return TrackingResult(nucleus_mask, {})

    from trackastra.tracking import graph_to_ctc

    # Optionally shrink the temporal window before tracking. Trackastra reads
    # the window from the model config at predict time, so overriding it here
    # (a subset of the trained window — never larger) is the effective GPU
    # memory lever for dense wells.
    config = getattr(model.transformer, "config", {})
    if window is not None:
        config["window"] = window
        logger.info(
            "Overriding Trackastra temporal window → %d frames", window
        )

    # Diagnostic: the attention spatial-bias matrix is (heads, N, N) where
    # N = detections summed over the frames in one window — this, not batch
    # size, is what drives GPU memory (~N²). Surface it so the scale is visible
    # rather than guessed.
    n_frames = nucleus_mask.shape[0]
    per_frame = [
        int(np.unique(nucleus_mask[t]).size - 1) for t in range(n_frames)
    ]
    eff_window = min(int(config.get("window", n_frames)), n_frames)
    n_per_window = max(
        sum(per_frame[i : i + eff_window])
        for i in range(n_frames - eff_window + 1)
    )
    logger.info(
        "Tracking %d frames, %d–%d objects/frame; effective window %d → "
        "~%d detections/window (attention memory scales as this squared).",
        n_frames,
        min(per_frame),
        max(per_frame),
        eff_window,
        n_per_window,
    )

    graph, _ = model.track(
        image_stack,
        nucleus_mask,
        mode=mode,
        normalize_imgs=True,
        batch_size=batch_size,
    )
    # graph_to_ctc gives the canonical CTC relabelling (divisions create new
    # ids with explicit parents) plus the lineage table. We use it for both so
    # the relabelled mask and the parent map are guaranteed consistent.
    track_df, relabelled = graph_to_ctc(graph, nucleus_mask, check=False)
    parent_map = {
        int(row.label): int(row.parent)
        for row in track_df.itertuples(index=False)
    }
    relabelled = relabelled.astype(nucleus_mask.dtype, copy=False)

    n_tracks = len(parent_map)
    n_divisions = sum(1 for parent in parent_map.values() if parent != 0) // 2
    logger.info(
        "Tracked %d nuclei into %d tracks (%d divisions).",
        int(nucleus_mask.max()),
        n_tracks,
        n_divisions,
    )
    return TrackingResult(relabelled, parent_map)


def add_track_columns(
    df: pd.DataFrame,
    parent_map: dict[int, int],
) -> None:
    """Add track id columns to a per-well measurements dataframe in place.

    The dataframe's ``label`` column already holds the ``track_id`` because the
    nucleus mask was relabelled before feature extraction. This derives the
    immutable ``_raw`` columns and the lineage columns.

    Args:
        df: Per-well measurements dataframe with a ``label`` column.
        parent_map: ``track_id -> parent_track_id`` from
            :class:`TrackingResult`.
    """
    df["track_id"] = df["label"].astype(int)
    df["track_id_raw"] = df["track_id"]
    df["parent_track_id"] = (
        df["track_id"].map(parent_map).fillna(0).astype(int)
    )
    df["parent_track_id_raw"] = df["parent_track_id"]
