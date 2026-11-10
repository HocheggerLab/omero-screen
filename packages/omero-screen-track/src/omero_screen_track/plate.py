"""Run the lineage repair on a plate stored in CellView.

The repair reads the immutable ``track_id_raw`` / ``parent_track_id_raw``
columns and writes the curated ``track_id`` / ``parent_track_id``, so it can be
re-run at any time with the same result, and never loses the tracker's output.
It needs a reporter that tells a mitosis from a mislinked or split nucleus:
the geminin of PIP-FUCCI, whose nuclear signal collapses at anaphase (a
timelapse with channels named exactly ``pip`` and ``geminin``,
:func:`omero_utils.channels.resolve_pip_fucci`), or otherwise an H2B
reporter, whose chromatin condenses (a channel named ``h2b`` or
``h2b_<fluorophore>``, :func:`omero_utils.channels.resolve_h2b`, #85). Every
other plate is refused rather than repaired on geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import duckdb
import pandas as pd
from omero_utils.channels import PipFucci, resolve_h2b, resolve_pip_fucci

from omero_screen_track.edit import Curated, EditLog, base_lineage, replay
from omero_screen_track.repair import RepairParams, repair_lineage

if TYPE_CHECKING:
    from omero_screen_track.fate import Walker


class NotPipFucciError(ValueError):
    """Raised for a plate with neither PIP-FUCCI nor H2B; it is not repaired."""


@dataclass(frozen=True)
class Reporter:
    """The channel the repair's mitosis test reads, as named on the plate."""

    kind: Literal["geminin", "h2b"]
    channel: str


def resolve_reporter(names: list[str], n_timepoints: int) -> Reporter | None:
    """Geminin if the plate is PIP-FUCCI, else H2B if it has one, else None."""
    fucci = resolve_pip_fucci(names, n_timepoints)
    if fucci is not None:
        return Reporter("geminin", fucci.geminin)
    h2b = resolve_h2b(names, n_timepoints)
    return Reporter("h2b", h2b) if h2b is not None else None


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
        f"Plate {plate_id}{where} is neither PIP-FUCCI (a timelapse with "
        f"channels named exactly 'pip' and 'geminin') nor has an H2B channel "
        f"('h2b' or 'h2b_<fluorophore>'); its tracks are not repaired."
    )


def _measured(
    conn: duckdb.DuckDBPyConnection, plate_id: int
) -> tuple[list[str], int]:
    """Channels with nuclear values for this plate, and its timepoint count.

    Only channels measured on this plate count, so a column another plate
    created does not make this one FUCCI or H2B.
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
    return measured, int(row[0])


def plate_fucci(conn: duckdb.DuckDBPyConnection, plate_id: int) -> PipFucci:
    """The PIP-FUCCI channels of a plate, as named in CellView.

    Raises:
        NotPipFucciError: If the plate is not PIP-FUCCI.
    """
    fucci = resolve_pip_fucci(*_measured(conn, plate_id))
    if fucci is None:
        raise _not_fucci(plate_id)
    return fucci


def plate_walker(conn: duckdb.DuckDBPyConnection, plate_id: int) -> Walker:
    """The fate walker for a plate's channels (:func:`~omero_screen_track.fate.resolve_walker`).

    Raises:
        NotPipFucciError: If the plate has neither PIP-FUCCI nor H2B with
            tubulin, so cells cannot be followed.
    """
    from omero_screen_track.fate import resolve_walker

    walker = resolve_walker(*_measured(conn, plate_id))
    if walker is None:
        raise NotPipFucciError(
            f"Plate {plate_id} is neither PIP-FUCCI nor has H2B and tubulin "
            "channels; there is no walker to follow its cells."
        )
    return walker


def plate_reporter(conn: duckdb.DuckDBPyConnection, plate_id: int) -> Reporter:
    """The repair reporter of a plate: geminin, else H2B.

    Raises:
        NotPipFucciError: If the plate has neither.
    """
    reporter = resolve_reporter(*_measured(conn, plate_id))
    if reporter is None:
        raise _not_fucci(plate_id)
    return reporter


def repair_input(det: pd.DataFrame, plate_id: int, well: str) -> pd.DataFrame:
    """A well from :func:`omero_screen_track.cells.load_well`, ready for the repair.

    On a PIP-FUCCI well the geminin reporter becomes the repair's ``marker``
    column. Otherwise the H2B maximum intensity (``<h2b>_max``, loaded with
    ``load_well(..., maxima=[h2b])``) becomes its ``chromatin`` column.

    Raises:
        NotPipFucciError: If the well has neither reporter, or its H2B
            maximum was not loaded.
    """
    n_t = det["timepoint"].nunique() if "timepoint" in det else 0
    names = [c for c in det.columns if not c.endswith("_max")]
    reporter = resolve_reporter(names, n_t)
    if reporter is None:
        raise _not_fucci(plate_id, f" well {well}")
    if reporter.kind == "geminin":
        return det.rename(columns={reporter.channel: "marker"})
    peak = f"{reporter.channel}_max"
    if peak not in det.columns:
        raise _not_fucci(plate_id, f" well {well} (no H2B maximum loaded)")
    return det.assign(chromatin=det[peak])


#: Former name of :func:`repair_input`, from when only PIP-FUCCI was repaired.
pip_fucci_input = repair_input


def load_detections(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    wells: list[str] | None = None,
) -> pd.DataFrame:
    """Tracked detections of a plate in the shape :func:`repair_lineage` expects.

    The reporter column is ``marker`` (background-subtracted geminin mean)
    on a PIP-FUCCI plate, else ``chromatin`` (H2B maximum intensity).

    Raises:
        NotPipFucciError: If the plate has neither reporter.
    """
    reporter = plate_reporter(conn, plate_id)
    ch = reporter.channel.lower()
    cols = {c.lower(): c for c in _columns(conn)}
    if reporter.kind == "geminin":
        mean = cols[f"intensity_mean_{ch}_nucleus"]
        bg = cols.get(f"{ch}_background")
        sql = f'm."{mean}" - coalesce(m."{bg}", 0)' if bg else f'm."{mean}"'
        column = "marker"
    else:
        peak = cols.get(f"intensity_max_{ch}_nucleus")
        if peak is None:
            raise _not_fucci(plate_id, " (no H2B maximum intensity)")
        sql, column = f'm."{peak}"', "chromatin"
    df = conn.execute(
        f"""
        select m.measurement_id, c.well,
               m.track_id_raw, m.parent_track_id_raw, m.timepoint,
               m.area_nucleus as area,
               m."centroid-0-nuc" as y, m."centroid-1-nuc" as x,
               {sql} as {column}
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
    if df[column].isna().all():
        raise _not_fucci(plate_id)
    df[column] = df[column].clip(lower=0)
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

    reporter = plate_reporter(conn, plate_id)
    maxima = [reporter.channel] if reporter.kind == "h2b" else []
    det, _ = drop_debris(load_well(conn, plate_id, well, maxima=maxima))
    result = repair_lineage(repair_input(det, plate_id, well), params)
    from omero_screen_track.cells import apply_extras

    base = base_lineage(det, result)
    curated = replay(base, log if log is not None else [], well)
    return apply_extras(det, curated), curated
