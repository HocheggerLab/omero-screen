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


def load_tracking_model(model_name: str) -> Trackastra:
    """Load a pretrained Trackastra model.

    Call once per run (mirrors the one-time Cellpose/inference model setup) and
    pass the returned model into :func:`track_nucleus_mask`.

    Args:
        model_name: Pretrained model name (e.g. ``"general_2d"``, ``"ctc"``) or
            a path to a fine-tuned checkpoint directory.

    Returns:
        A loaded :class:`trackastra.model.Trackastra` instance.
    """
    from trackastra.model import Trackastra

    # Trackastra accepts "cuda"/"cpu"; it has no MPS kernels, so fall back to
    # CPU on Apple silicon rather than crashing.
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
) -> TrackingResult:
    """Track nuclei across time and relabel the mask with stable track ids.

    Args:
        image_stack: Nucleus-channel intensity, shape ``(T, Y, X)``.
        nucleus_mask: Per-frame Cellpose nucleus labels, shape ``(T, Y, X)``.
        model: A model from :func:`load_tracking_model`.
        mode: Linking mode — one of :data:`VALID_TRACKING_MODES`.

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

    graph, _ = model.track(
        image_stack,
        nucleus_mask,
        mode=mode,
        normalize_imgs=True,
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
