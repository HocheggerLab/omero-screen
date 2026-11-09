"""Run the lineage repair on a plate stored in CellView.

The repair reads the immutable ``track_id_raw`` / ``parent_track_id_raw``
columns and writes the curated ``track_id`` / ``parent_track_id``, so it can be
re-run at any time with the same result, and never loses the tracker's output.
It needs a mitotic marker whose nuclear signal collapses at anaphase: the
geminin reporter of PIP-FUCCI. Only PIP-FUCCI plates are repaired — a
timelapse with channels named exactly ``pip`` and ``geminin``
(:func:`omero_utils.channels.resolve_pip_fucci`); every other plate
is refused rather than repaired on geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
from omero_utils.channels import PipFucci, resolve_pip_fucci

from omero_screen_track.edit import Curated, EditLog, base_lineage, replay
from omero_screen_track.repair import RepairParams, repair_lineage


class NotPipFucciError(ValueError):
    """Raised for a plate that is not PIP-FUCCI; its tracks are not repaired."""


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


def _not_fucci(plate_id: int, where: str = "") -> NotPipFucciError:
    return NotPipFucciError(
        f"Plate {plate_id}{where} is not PIP-FUCCI (a timelapse with channels "
        f"named exactly 'pip' and 'geminin'); its tracks are not repaired."
    )


def plate_fucci(conn: duckdb.DuckDBPyConnection, plate_id: int) -> PipFucci:
    """The PIP-FUCCI channels of a plate, as named in CellView.

    Only channels with nuclear values for this plate count, so a column
    another plate created does not make this one FUCCI.

    Raises:
        NotPipFucciError: If the plate is not PIP-FUCCI.
    """
    prefix, suffix = "intensity_mean_", "_nucleus"
    cols = [
        c
        for c in _columns(conn)
        if c.startswith(prefix) and c.endswith(suffix)
    ]
    counts = ", ".join(f'count(m."{c}")' for c in cols)
    row = conn.execute(
        f"""
        select count(distinct m.timepoint){", " + counts if counts else ""}
        from measurements m
        join conditions c using (condition_id)
        join repeats r using (repeat_id)
        where r.plate_id = ?
        """,
        [plate_id],
    ).fetchone()
    assert row is not None
    measured = [
        c[len(prefix) : -len(suffix)]
        for c, n in zip(cols, row[1:], strict=True)
        if n
    ]
    fucci = resolve_pip_fucci(measured, int(row[0]))
    if fucci is None:
        raise _not_fucci(plate_id)
    return fucci


def pip_fucci_input(
    det: pd.DataFrame, plate_id: int, well: str
) -> pd.DataFrame:
    """A well from :func:`omero_screen_track.cells.load_well`, ready for the repair.

    The geminin reporter becomes the repair's ``marker`` column.

    Raises:
        NotPipFucciError: If the well is not PIP-FUCCI.
    """
    n_t = det["timepoint"].nunique() if "timepoint" in det else 0
    fucci = resolve_pip_fucci(det.columns, n_t)
    if fucci is None:
        raise _not_fucci(plate_id, f" well {well}")
    return det.rename(columns={fucci.geminin: "marker"})


def load_detections(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    wells: list[str] | None = None,
) -> pd.DataFrame:
    """Tracked detections of a plate in the shape :func:`repair_lineage` expects.

    Raises:
        NotPipFucciError: If the plate is not PIP-FUCCI.
    """
    gem = plate_fucci(conn, plate_id).geminin
    cols = {c.lower(): c for c in _columns(conn)}
    mean = cols[f"intensity_mean_{gem.lower()}_nucleus"]
    bg = cols.get(f"{gem.lower()}_background")
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
        raise _not_fucci(plate_id)
    df["marker"] = df["marker"].clip(lower=0)
    df["parent_track_id_raw"] = df["parent_track_id_raw"].fillna(0)
    return df


def repair_plate(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    wells: list[str] | None = None,
    dry_run: bool = False,
    events_path: Path | None = None,
    params: RepairParams | None = None,
) -> list[WellSummary]:
    """Repair every well of a plate and (unless ``dry_run``) write the result.

    Args:
        conn: Open CellView connection (read-write unless ``dry_run``).
        plate_id: OMERO plate id.
        wells: Restrict to these wells.
        dry_run: Report only; write nothing.
        events_path: Write every repair decision to this CSV.
        params: Repair thresholds.

    Returns:
        One summary per well.
    """
    det = load_detections(conn, plate_id, wells)
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
    wells: list[str] | None = None,
    params: RepairParams | None = None,
) -> dict[str, tuple[pd.DataFrame, Curated]]:
    """Detections and automatically repaired base lineage for each well.

    The base is always recomputed from the raw columns, so replaying an edit
    log on it gives the same result however often curated ids were written.
    """
    det = load_detections(conn, plate_id, wells)
    return {
        str(well): (sub, base_lineage(sub, repair_lineage(sub, params)))
        for well, sub in det.groupby("well", sort=True)
    }


def curate_plate(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    log: EditLog,
    wells: list[str] | None = None,
    write: bool = False,
) -> dict[str, Curated]:
    """Repair, then replay the edit log, for every well; optionally write the ids.

    Returns:
        The curated lineage per well (annotations included).
    """
    curated = {}
    updates = []
    for well, (det, base) in well_bases(conn, plate_id, wells).items():
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
    params: RepairParams | None = None,
) -> tuple[pd.DataFrame, Curated]:
    """One well's detections (all nuclear channels) and its curated lineage.

    The lineage is the automatic repair of the raw tracks plus every edit in
    ``log`` for this well. This is the starting point for following cells.

    Raises:
        NotPipFucciError: If the plate is not PIP-FUCCI.
    """
    from omero_screen_track.cells import load_well
    from omero_screen_track.debris import drop_debris

    det, _ = drop_debris(load_well(conn, plate_id, well))
    result = repair_lineage(pip_fucci_input(det, plate_id, well), params)
    from omero_screen_track.cells import apply_extras

    base = base_lineage(det, result)
    curated = replay(base, log if log is not None else [], well)
    return apply_extras(det, curated), curated
