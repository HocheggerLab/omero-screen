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
from typing import TYPE_CHECKING, Any, cast

import duckdb
import numpy as np
import pandas as pd

from cellview.tracks.cells import apply_extras, curated_tracks, load_well
from cellview.tracks.corrections import correction_hash, load_stored
from cellview.tracks.debris import drop_debris
from cellview.tracks.edit import EditLog, imported_lineage, replay
from cellview.tracks.fate import FateParams, apply_annotations, follow_cells

if TYPE_CHECKING:
    from collections.abc import Iterable

QUESTIONS = {
    "switch": "The track jumps to a different-looking nucleus while a new label continues the cell. Which nucleus is the cell from here?",
    "jump_suspect": "Area or geminin changes abruptly while a neighbour that already existed looks like the cell. A real change in this cell (e.g. an exit), or a jump onto the neighbour?",
    "daughter_pair": "The track ends with geminin high and two half-size nuclei appear beside it. A division whose daughters were not linked?",
    "lost": "The track ends without an outcome. Where does the cell go? Pick its continuation, or mark death.",
    "link_ambiguous": "A break was bridged with more than one candidate. Is the continuation the same cell?",
    "gap": "Frames without a mask were bridged. Is it the same cell on both sides?",
    "fragment": "Nucleus pieces were merged. One nucleus, or two cells?",
    "geminin_jump": "Geminin falls without a split. A wrong link, or a mis-segmented frame?",
    "outcome_review": "Automatic outcome needs confirmation. Correct, or which outcome?",
    "refractory": "A second division within 8 h was rejected. One mitosis, or two?",
    "edge": "The cell touches the image edge. Still followable?",
    "reporter_unclear": "Both reporters stay low. Early S, or no reporter?",
    "audit": "Random check of an automatically accepted cell. Is the whole track right?",
    "final_audit": "Final check after correction. Follow the cell from the window start to its outcome: is every link, mask and the outcome right?",
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
        # Row attributes are typed Scalar by the stubs; columns are numeric.
        for r in cast("Iterable[Any]", rows.itertuples())
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
    params: FateParams | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Sample and queue one well. Returns ``(sample table, queue items)``."""
    raw = load_well(conn, plate_id, well)
    det, debris = drop_debris(raw)
    # CellView holds the stored correction's final lineage; start from it as
    # it is, so the review, the widget's masks and the fate table agree.
    curated = replay(
        imported_lineage(det), log if log is not None else [], well
    )
    det = apply_extras(det, curated)
    tracks = curated_tracks(curated, det)
    shape = (int(det["y"].max()) + 1, int(det["x"].max()) + 1)
    stored = load_stored(plate_id, well)
    cells, states = follow_cells(
        tracks,
        spec.start,
        spec.stop,
        params,
        stored.events if stored is not None else None,
        {} if stored is not None else None,  # event ids are final track ids
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
    queued = sampled[sampled["queued"]]
    for row in cast("Iterable[Any]", queued.itertuples()):
        first_flag = str(row.reason).split()[0]
        proposals = _proposals(
            json.loads(getattr(row, "proposals", "[]") or "[]"),
            labels,
            spec.start,
            int(row.label),
        )
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
                "proposals": proposals,
            }
        )
    return sampled, items


def _proposals(
    walker: list[dict[str, Any]],
    labels: pd.Series,
    start: int,
    mother_label: int,
) -> list[dict[str, Any]]:
    """The walker's bridges, switches and daughter pairs as edits to propose.

    The walker names curated tracks; an edit names nuclei by raw label, so
    each is translated to the largest raw label of that track in that frame.
    """
    out = []
    for p in walker:
        label = labels.get((int(p["track"]), int(p["frame"])))
        if label is None:
            continue
        if p["op"] == "link":
            args: dict[str, Any] = {
                "frame": int(p["frame"]),
                "label": int(label),
            }
        elif p["op"] == "set_parent":
            args = {
                "cell": [int(p["frame"]), int(label)],
                "parent": [start, mother_label],
            }
        else:
            continue
        out.append(
            {
                "op": p["op"],
                "args": args,
                "reason": p["reason"],
                "confidence": p["confidence"],
            }
        )
    return out


def build_audit(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    wells: list[str],
    n: int,
    out_path: Path,
    log_path: Path | None = None,
    start: int = 72,
    stop: int = 168,
    seed: int = 0,
) -> Path:
    """Write a final-audit queue: ``n`` random analysed cells, end to end.

    Run after the review and corrections are finished. Cells are drawn
    uniformly (fixed ``seed``) from the non-excluded cells of ``wells`` on
    the curated lineage (CellView's tracks plus ``log_path``); excluded cells
    are not drawn because they leave the analysis. Items carry no walker
    proposals: nothing is applied automatically. Write the queue into the
    review directory (e.g. ``<dir>/audit.json``) so fixes go to the same
    edit log, while verdicts go to ``audit_decisions.json``.

    Raises:
        ValueError: If the log was made on another correction.
    """
    log = EditLog(log_path) if log_path else None
    digest = correction_hash(plate_id, wells)
    if log_path is not None:
        check_correction(log_path.parent, digest)
    spec = QueueSpec(
        start=start, stop=stop, sample=10**9, audit=1.0, seed=seed
    )
    pool: list[dict[str, Any]] = []
    for well in wells:
        _, items = build_well(conn, plate_id, well, spec, log)
        pool += items
    rng = np.random.default_rng([seed, n])
    pick = sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))
    items = []
    for k in pick:
        it = dict(pool[k])
        it.update(
            frame=start,
            reason="final_audit",
            question=QUESTIONS["final_audit"],
            proposals=[],
        )
        items.append(it)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "version": 1,
                "plate_id": plate_id,
                "mode": "human",
                "spec": {**spec.__dict__, "audit_n": n, "wells": wells},
                "log": str(log_path) if log_path else None,
                "correction": digest,
                "items": items,
            },
            indent=1,
        )
    )
    return out_path


def correction_path(directory: Path) -> Path:
    """The sidecar that records which correction a directory's edit log was built on."""
    return Path(directory) / "correction.json"


def check_correction(directory: Path, digest: str | None) -> None:
    """Refuse to replay an edit log built on another correction.

    Raises:
        ValueError: If ``directory/correction.json`` names a different hash.
    """
    path = correction_path(directory)
    if digest is None or not path.exists():
        return
    recorded = json.loads(path.read_text()).get("hash")
    if recorded and recorded != digest:
        raise ValueError(
            f"The edit log in {directory} was made on correction {recorded[:12]}…, "
            f"but the stored correction is now {digest[:12]}…. Its anchors name "
            "nuclei of the old correction; start a new review directory."
        )


def record_correction(
    directory: Path, plate_id: int, wells: list[str], digest: str
) -> None:
    """Write (or confirm) the correction sidecar of a review directory."""
    check_correction(directory, digest)
    path = correction_path(directory)
    if not path.exists():
        path.write_text(
            json.dumps(
                {"plate_id": plate_id, "wells": sorted(wells), "hash": digest},
                indent=1,
            )
        )


def build_queue(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    wells: list[str],
    spec: QueueSpec,
    out_dir: Path,
    log_path: Path | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    """Build and write ``queue.json`` and ``sample.csv`` for several wells."""
    log = EditLog(log_path) if log_path else None
    digest = correction_hash(plate_id, wells)
    if log_path is not None:
        check_correction(log_path.parent, digest)
    samples, items, exclusions = [], [], {}
    for well in wells:
        s, i = build_well(conn, plate_id, well, spec, log)
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
                "correction": digest,
                "items": items,
            },
            indent=1,
        )
    )
    if digest is not None:
        record_correction(out_dir, plate_id, wells, digest)
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
