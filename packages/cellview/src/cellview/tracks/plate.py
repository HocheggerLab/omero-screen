"""Run the lineage repair on a plate stored in CellView.

The repair reads the immutable ``track_id_raw`` / ``parent_track_id_raw``
columns and writes the curated ``track_id`` / ``parent_track_id``, so it can be
re-run at any time with the same result, and never loses the tracker's output.
It needs a mitotic marker whose nuclear signal collapses at anaphase; with
PIP-FUCCI that is geminin. Plates without one are refused rather than
repaired on geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from cellview.tracks.edit import Curated, EditLog, base_lineage, replay
from cellview.tracks.repair import RepairParams, repair_lineage


class MarkerNotFoundError(ValueError):
    """Raised when the plate has no nuclear measurement for the marker."""


@dataclass(frozen=True)
class WellSummary:
    """Before/after counts for one well."""

    well: str
    tracks_before: int
    tracks_after: int
    divisions_before: int
    divisions_after: int


def _columns(conn: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in conn.execute("DESCRIBE measurements").fetchall()]


def marker_columns(
    conn: duckdb.DuckDBPyConnection, marker: str
) -> tuple[str, str | None]:
    """The nuclear mean-intensity column for ``marker`` and its background column.

    Matching is case-insensitive (``Geminin`` and ``geminin`` both resolve).

    Raises:
        MarkerNotFoundError: If no ``intensity_mean_<marker>_nucleus`` column exists.
    """
    cols = {c.lower(): c for c in _columns(conn)}
    mean = cols.get(f"intensity_mean_{marker.lower()}_nucleus")
    if mean is None:
        raise MarkerNotFoundError(
            f"No nuclear '{marker}' measurement in this database. Lineage repair "
            f"needs a mitotic marker that is degraded at anaphase (geminin in "
            f"PIP-FUCCI); plates without one are not repaired."
        )
    return mean, cols.get(f"{marker.lower()}_background")


def load_detections(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    marker: str,
    wells: list[str] | None = None,
) -> pd.DataFrame:
    """Tracked detections of a plate in the shape :func:`repair_lineage` expects.

    Raises:
        MarkerNotFoundError: If the marker column is missing, or every value for
            this plate is NULL (the plate was not imaged with the marker).
    """
    mean, bg = marker_columns(conn, marker)
    marker_sql = f'm."{mean}" - coalesce(m."{bg}", 0)' if bg else f'm."{mean}"'
    df = conn.execute(
        f"""
        select m.measurement_id, c.well,
               m.track_id_raw, m.parent_track_id_raw, m.timepoint,
               m.area_nucleus as area,
               m."centroid-0-nuc" as y, m."centroid-1-nuc" as x,
               {marker_sql} as marker
        from measurements m
        join conditions c using (condition_id)
        join repeats r using (repeat_id)
        where r.plate_id = ? and m.track_id_raw is not null
        """,
        [plate_id],
    ).df()
    if wells:
        df = df[df.well.isin(wells)]
    if df.empty:
        raise ValueError(f"No tracked measurements for plate {plate_id}.")
    if df["marker"].isna().all():
        raise MarkerNotFoundError(
            f"Plate {plate_id} has no '{marker}' values; not repaired."
        )
    df["marker"] = df["marker"].clip(lower=0)
    df["parent_track_id_raw"] = df["parent_track_id_raw"].fillna(0)
    return df


def repair_plate(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    marker: str = "Geminin",
    wells: list[str] | None = None,
    dry_run: bool = False,
    events_path: Path | None = None,
    params: RepairParams | None = None,
) -> list[WellSummary]:
    """Repair every well of a plate and (unless ``dry_run``) write the result.

    Args:
        conn: Open CellView connection (read-write unless ``dry_run``).
        plate_id: OMERO plate id.
        marker: Channel whose nuclear signal drops at anaphase.
        wells: Restrict to these wells.
        dry_run: Report only; write nothing.
        events_path: Write every repair decision to this CSV.
        params: Repair thresholds.

    Returns:
        One summary per well.
    """
    det = load_detections(conn, plate_id, marker, wells)
    summaries, updates, events = [], [], []
    for well, sub in det.groupby("well", sort=True):
        result = repair_lineage(sub, params)
        new_tid = sub["track_id_raw"].astype(int).map(result.assignment)
        new_parent = new_tid.map(result.parents).fillna(0).astype(int)
        updates.append(
            pd.DataFrame(
                {
                    "measurement_id": sub["measurement_id"],
                    "track_id": new_tid,
                    "parent_track_id": new_parent,
                }
            )
        )
        raw_parents = sub.groupby("track_id_raw")["parent_track_id_raw"].max()
        summaries.append(
            WellSummary(
                well=str(well),
                tracks_before=int(sub["track_id_raw"].nunique()),
                tracks_after=int(new_tid.nunique()),
                divisions_before=int(raw_parents[raw_parents > 0].nunique()),
                divisions_after=len({p for p in result.parents.values() if p}),
            )
        )
        if not result.events.empty:
            events.append(result.events.assign(well=well))

    if events_path is not None and events:
        pd.concat(events, ignore_index=True).to_csv(events_path, index=False)
    if not dry_run:
        write_tracks(conn, pd.concat(updates, ignore_index=True))
    return summaries


def write_tracks(
    conn: duckdb.DuckDBPyConnection, updates: pd.DataFrame
) -> None:
    """Write ``track_id`` / ``parent_track_id`` by ``measurement_id`` in one transaction."""
    conn.register("_track_updates", updates)
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            update measurements set
                track_id = u.track_id,
                parent_track_id = u.parent_track_id
            from _track_updates u
            where measurements.measurement_id = u.measurement_id
            """
        )
        conn.execute("COMMIT")
    except duckdb.Error:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister("_track_updates")


def well_bases(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    marker: str = "Geminin",
    wells: list[str] | None = None,
    params: RepairParams | None = None,
) -> dict[str, tuple[pd.DataFrame, Curated]]:
    """Detections and automatically repaired base lineage for each well.

    The base is always recomputed from the raw columns, so replaying an edit
    log on it gives the same result however often curated ids were written.
    """
    det = load_detections(conn, plate_id, marker, wells)
    return {
        str(well): (sub, base_lineage(sub, repair_lineage(sub, params)))
        for well, sub in det.groupby("well", sort=True)
    }


def curate_plate(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    log: EditLog,
    marker: str = "Geminin",
    wells: list[str] | None = None,
    write: bool = False,
) -> dict[str, Curated]:
    """Repair, then replay the edit log, for every well; optionally write the ids.

    Returns:
        The curated lineage per well (annotations included).
    """
    curated = {}
    updates = []
    for well, (det, base) in well_bases(conn, plate_id, marker, wells).items():
        cur = replay(base, log, well)
        curated[well] = cur
        keys = zip(
            det["timepoint"].astype(int),
            det["track_id_raw"].astype(int),
            strict=True,
        )
        tid = [cur.tracks[k] for k in keys]
        updates.append(
            pd.DataFrame(
                {
                    "measurement_id": det["measurement_id"].to_numpy(),
                    "track_id": tid,
                    "parent_track_id": [cur.parents.get(t, 0) for t in tid],
                }
            )
        )
    if write and updates:
        write_tracks(conn, pd.concat(updates, ignore_index=True))
    return curated


def curated_well(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    well: str,
    log: EditLog | None = None,
    marker: str = "Geminin",
    params: RepairParams | None = None,
) -> tuple[pd.DataFrame, Curated]:
    """One well's detections (all nuclear channels) and its curated lineage.

    The lineage is the automatic repair of the raw tracks plus every edit in
    ``log`` for this well. This is the starting point for following cells.

    Raises:
        MarkerNotFoundError: If the plate has no values for ``marker``.
    """
    from cellview.tracks.cells import load_well

    det = load_well(conn, plate_id, well)
    key = marker.lower()
    if key not in det.columns:
        raise MarkerNotFoundError(
            f"Plate {plate_id} well {well} has no '{marker}' values; not repaired."
        )
    result = repair_lineage(det.rename(columns={key: "marker"}), params)
    base = base_lineage(det, result)
    return det, replay(base, log if log is not None else [], well)
