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
from loguru import logger

from omero_screen.torch import get_device

if TYPE_CHECKING:
    from trackastra.model import Trackastra


# Linking modes accepted by Trackastra.track. ``ilp`` additionally requires the
# optional ``motile`` + Gurobi/SCIP stack; guard its use at call time.
VALID_TRACKING_MODES = ("greedy", "greedy_nodiv", "ilp")

# Empirically calibrated peak bytes per N² for Trackastra's attention step,
# where N = detections summed over a window's frames. Anchored on the observed
# A40 OOM: a window-4 well with N≈15,589 needed ≳36 GiB at the failing
# allocation (≳150 bytes/N²). We round up to 200 for headroom — overestimating
# only costs a slightly smaller window, while underestimating costs a lost
# multi-hour run, so we deliberately lean conservative.
_ATTENTION_BYTES_PER_N2 = 200
# Fraction of *free* VRAM the estimated attention peak must fit within.
_VRAM_SAFETY = 0.85


def _max_detections_for_window(per_frame: list[int], window: int) -> int:
    """Largest detections-per-window (N) over all sliding windows of size."""
    n_frames = len(per_frame)
    window = min(window, n_frames)
    return max(
        sum(per_frame[i : i + window]) for i in range(n_frames - window + 1)
    )


def _auto_gpu_window(per_frame: list[int], max_window: int) -> int:
    """Largest temporal window whose attention peak fits free GPU VRAM.

    Trackastra's attention peak scales as ~N² (N = detections summed over a
    window's frames). We estimate that peak with the calibrated factor above
    and pick the largest window from ``max_window`` down whose estimate fits a
    safety fraction of currently free VRAM, having first released the
    segmentation step's cached allocations. Returns ``max_window`` unchanged if
    CUDA memory info is unavailable (run rather than guess) and never goes
    below 2 (the minimum useful temporal window).

    Args:
        per_frame: Object count per timepoint.
        max_window: The model's trained window (the upper bound).

    Returns:
        The chosen window size.
    """
    import torch

    if not torch.cuda.is_available():
        return max_window
    # Reclaim the cached VRAM segmentation left behind so the estimate reflects
    # what tracking can actually use.
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    budget = free * _VRAM_SAFETY
    for window in range(max_window, 1, -1):
        n = _max_detections_for_window(per_frame, window)
        if _ATTENTION_BYTES_PER_N2 * n * n <= budget:
            return window
    return 2


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

    logger.info(f"Loading Trackastra model '{model_name}' on {device}")
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

    # Object count per timepoint — drives both the diagnostic and the window
    # auto-fit (N = detections summed over a window's frames).
    n_frames = nucleus_mask.shape[0]
    per_frame = [
        int(np.unique(nucleus_mask[t]).size - 1) for t in range(n_frames)
    ]

    # Decide the temporal window. Trackastra reads it from the model config at
    # predict time, so we set config["window"] in place (a subset of the
    # trained window — never larger). Precedence:
    #   explicit override  >  auto-fit on GPU  >  full window on CPU
    config = getattr(model.transformer, "config", {})
    device = str(getattr(model, "device", "cpu"))
    model_window = min(int(config.get("window", n_frames)), n_frames)
    if window is not None:
        config["window"] = window
        logger.info(
            f"Overriding Trackastra temporal window → {window:d} frames"
        )
    elif device == "cuda":
        # Shrink the window just enough that the O(N²) attention fits free VRAM,
        # keeping tracking on the GPU (fast) instead of falling back to CPU.
        auto = _auto_gpu_window(per_frame, model_window)
        if auto < model_window:
            config["window"] = auto
            logger.info(
                f"Auto-reduced temporal window {model_window:d} → {auto:d} to fit GPU VRAM (override: --track-window N, or --track-device cpu for the full window at the cost of speed)."
            )

    # Diagnostic: the attention spatial-bias matrix is (heads, N, N) where
    # N = detections summed over the frames in one window — this is what drives
    # GPU memory (~N²). Surface it so the scale is visible, not guessed.
    eff_window = min(int(config.get("window", n_frames)), n_frames)
    n_per_window = _max_detections_for_window(per_frame, eff_window)
    logger.info(
        f"Tracking {n_frames:d} frames, {min(per_frame):d}–{max(per_frame):d} objects/frame; effective window {eff_window:d} → ~{n_per_window:d} detections/window (attention memory scales as this squared)."
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
        f"Tracked {int(nucleus_mask.max()):d} nuclei into {n_tracks:d} tracks ({n_divisions:d} divisions)."
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
