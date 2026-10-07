"""Build a review queue: a random sample of cells per well, flagged ones first.

For each well the automatic repair and the edit log are replayed, the fate
walker follows every cell present at ``start`` up to ``stop``, and curated
annotations override its calls. A fixed-seed random sample of the cells that
are not excluded is drawn per well. Every sampled cell that carries a review
flag is queued, and a random ``audit`` fraction of the unflagged ones is
queued too, so the false-pass rate of automatic acceptance can be measured.
Items from all wells are shuffled together.

Outputs:

* ``queue.json``: the Track Review widget's queue (ids ``<well>-t<start>-L<label>``,
  path with the raw label per frame, reason, question, automatic outcome);
* ``sample.csv``: every sampled cell with its automatic outcome, flags and
  whether it was queued. This is the analysis set: queued cells take the
  reviewer's verdict, the rest keep the automatic one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from cellview.tracks.cells import apply_extras, curated_tracks, load_well
from cellview.tracks.debris import drop_debris
from cellview.tracks.edit import EditLog, base_lineage, replay
from cellview.tracks.fate import FateParams, apply_annotations, follow_cells
from cellview.tracks.repair import repair_lineage

QUESTIONS = {
    "lost": "The track ends without an outcome. Where does the cell go? Pick its continuation, or mark death.",
    "link_ambiguous": "A break was bridged with more than one candidate. Is the continuation the same cell?",
    "gap": "Frames without a mask were bridged. Is it the same cell on both sides?",
    "fragment": "Nucleus pieces were merged. One nucleus, or two cells?",
    "phase_order": "Phases run backwards. Mis-segmented frames, or a wrong link?",
    "outcome_review": "Automatic outcome needs confirmation. Correct, or which outcome?",
    "refractory": "A second division within 8 h was rejected. One mitosis, or two?",
    "edge": "The cell touches the image edge. Still followable?",
    "reporter_unclear": "Both reporters stay low. Early S, or no reporter?",
    "audit": "Random check of an automatically accepted cell. Is the whole track right?",
}


@dataclass(frozen=True)
class QueueSpec:
    """What to sample and queue."""

    start: int = 72
    stop: int = 168
    sample: int = 50
    audit: float = 0.2
    seed: int = 0
    tail: int = 6
    stationary_warn: float = 0.05
    mode: str = "human"  # human: agent proposes, reviewer confirms; agent: agent works alone


def exclusion_summary(
    cells: pd.DataFrame, warn_fraction: float = 0.05
) -> dict[str, Any]:
    """Counts of start cells and of each exclusion, with a stationary-cell warning.

    Objects that do not move are debris, and are excluded. If a sizeable
    fraction of them still express a reporter (outcome ``stationary``), that
    is not debris: a mitotic arrest or another phenotype that stops cells
    moving would look like this. It should not happen in a control well.
    """
    counts = cells["outcome"].where(cells["excluded"]).value_counts()
    n = len(cells)
    stationary = int(counts.get("stationary", 0))
    fraction = stationary / n if n else 0.0
    return {
        "start_cells": n,
        "debris": int(counts.get("debris", 0)),
        "stationary": stationary,
        "no_reporter": int(counts.get("no_reporter", 0)),
        "stationary_fraction": round(fraction, 4),
        "stationary_warning": fraction > warn_fraction,
    }


def _largest_labels(det: pd.DataFrame, curated: Any) -> pd.Series:
    """Raw label of the largest piece per (curated track, frame)."""
    tid = [
        curated.tracks.get((int(t), int(lab)), 0)
        for t, lab in zip(det["timepoint"], det["label"], strict=True)
    ]
    d = det.assign(track_id=tid).sort_values("area")
    return d.groupby(["track_id", "timepoint"])["label"].last()


def _path(
    states: pd.DataFrame,
    labels: pd.Series,
    last_frame: int,
    tail: int,
    stop: int,
) -> list[list[float]]:
    rows = states.sort_values("timepoint")
    pts = [
        [
            int(r.timepoint),
            float(r.y),
            float(r.x),
            int(labels.get((int(r.track_id), int(r.timepoint)), 0)),
        ]
        for r in rows.itertuples()
    ]
    if pts:
        t, y, x, _ = pts[-1]
        pts += [
            [t + k, y, x, 0]
            for k in range(1, tail + 1)
            if t + k <= max(stop, last_frame) + tail
        ]
    return pts


def build_well(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    well: str,
    spec: QueueSpec,
    log: EditLog | None = None,
    marker: str = "Geminin",
    params: FateParams | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Sample and queue one well. Returns ``(sample table, queue items)``."""
    raw = load_well(conn, plate_id, well)
    det, debris = drop_debris(raw)
    key = marker.lower()
    if key not in det:
        raise ValueError(
            f"Plate {plate_id} well {well} has no '{marker}' values."
        )
    result = repair_lineage(det.rename(columns={key: "marker"}))
    curated = replay(
        base_lineage(det, result), log if log is not None else [], well
    )
    det = apply_extras(det, curated)
    tracks = curated_tracks(curated, det)
    shape = (int(det["y"].max()) + 1, int(det["x"].max()) + 1)
    cells, states = follow_cells(
        tracks,
        spec.start,
        spec.stop,
        params,
        result.events,
        result.assignment,
        shape,
    )
    cells = apply_annotations(cells, curated)
    summary = exclusion_summary(cells, spec.stationary_warn)
    # Debris is hidden before the repair and the walker; count what was present at the start.
    at_start = raw[raw["timepoint"] == spec.start]
    summary["debris_hidden"] = int(at_start["track_id_raw"].isin(debris).sum())
    labels = _largest_labels(det, curated)

    rng = np.random.default_rng([spec.seed, sum(map(ord, well))])
    eligible = cells[~cells["excluded"]]
    n = min(spec.sample, len(eligible))
    sampled = eligible.iloc[
        np.sort(rng.choice(len(eligible), size=n, replace=False))
    ].copy()
    sampled["well"] = well
    sampled["label"] = [
        int(labels.get((int(t), spec.start), 0))
        for t in sampled["start_track"]
    ]
    sampled["id"] = [
        f"{well}-t{spec.start}-L{lab}" for lab in sampled["label"]
    ]
    audit_draw = rng.random(len(sampled))
    flagged = sampled["n_flags"] > 0
    sampled["queued"] = flagged | (audit_draw < spec.audit)
    sampled.attrs["exclusions"] = summary
    sampled["reason"] = np.where(
        flagged,
        sampled["review_flags"],
        np.where(sampled["queued"], "audit", ""),
    )

    items = []
    for row in sampled[sampled["queued"]].itertuples():
        first_flag = str(row.reason).split()[0]
        frame = (
            int(row.outcome_frame)
            if pd.notna(row.outcome_frame) and row.outcome != "no_mitosis"
            else spec.start
        )
        path = _path(
            states[states["cell"] == row.cell],
            labels,
            int(row.last_frame),
            spec.tail,
            spec.stop,
        )
        items.append(
            {
                "id": row.id,
                "well": well,
                "frame": frame,
                "reason": row.reason,
                "question": QUESTIONS.get(first_flag, QUESTIONS["audit"]),
                "outcome": row.outcome,
                "path": path,
            }
        )
    return sampled, items


def build_queue(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    wells: list[str],
    spec: QueueSpec,
    out_dir: Path,
    log_path: Path | None = None,
    marker: str = "Geminin",
) -> tuple[Path, Path, pd.DataFrame]:
    """Build and write ``queue.json`` and ``sample.csv`` for several wells."""
    log = EditLog(log_path) if log_path else None
    samples, items, exclusions = [], [], {}
    for well in wells:
        s, i = build_well(conn, plate_id, well, spec, log, marker)
        exclusions[well] = s.attrs.get("exclusions", {})
        samples.append(s)
        items += i
    rng = np.random.default_rng(spec.seed)
    items = [
        items[k] for k in rng.permutation(len(items))
    ]  # blind the well order
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out_dir / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "version": 1,
                "plate_id": plate_id,
                "mode": spec.mode,
                "spec": spec.__dict__,
                "exclusions": exclusions,
                "log": str(log_path) if log_path else None,
                "items": items,
            },
            indent=1,
        )
    )
    sample = pd.concat(samples, ignore_index=True)
    sample.attrs["exclusions"] = exclusions
    sample_path = out_dir / "sample.csv"
    cols = [
        "id",
        "well",
        "label",
        "start_track",
        "start_phase",
        "outcome",
        "outcome_frame",
        "end_phase",
        "last_frame",
        "censored",
        "excluded",
        "curated",
        "review_flags",
        "queued",
        "reason",
        "segments",
    ]
    sample[cols].to_csv(sample_path, index=False)
    return queue_path, sample_path, sample
