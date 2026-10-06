"""Per-frame records of one tracked cell, for following it through time.

A cell is named by an anchor ``(frame, raw label)`` and followed along its
curated track (see :mod:`cellview.tracks.edit`). :func:`cell_frames` returns
one row per frame from the track's first to its last frame. Frames where the
cell has no mask are kept as *gap* rows with an interpolated position, so a
viewer can still centre on where the cell should be.

Nucleus pieces that the repair or the curation joined in one frame are
combined exactly as one nucleus: areas add, centroids and intensities are
area-weighted. The raw label of the largest piece is reported, which is what a
viewer selects in the cached mask.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from cellview.tracks.edit import Anchor, Curated


def nuclear_channels(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Channels that have a nuclear mean-intensity column, as named in CellView."""
    cols = [row[0] for row in conn.execute("DESCRIBE measurements").fetchall()]
    prefix, suffix = "intensity_mean_", "_nucleus"
    return [
        c[len(prefix) : -len(suffix)]
        for c in cols
        if c.startswith(prefix) and c.endswith(suffix)
    ]


def load_well(
    conn: duckdb.DuckDBPyConnection,
    plate_id: int,
    well: str,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """Tracked detections of one well with background-subtracted channel means.

    Args:
        conn: CellView connection (read-only is enough).
        plate_id: OMERO plate id.
        well: Well, e.g. ``"C2"``.
        channels: Nuclear channels to load (CellView names, any case). Default:
            every channel with data for this plate.

    Returns:
        Columns ``measurement_id, timepoint, label, track_id_raw,
        parent_track_id_raw, area, y, x`` plus one column per channel, named in
        lower case (``pip``, ``geminin``, ``spydna`` …), clipped at 0.
    """
    cols = {
        c.lower(): c
        for c in (
            row[0] for row in conn.execute("DESCRIBE measurements").fetchall()
        )
    }
    wanted = channels or nuclear_channels(conn)
    selects = []
    for ch in wanted:
        mean = cols.get(f"intensity_mean_{ch.lower()}_nucleus")
        if mean is None:
            continue
        bg = cols.get(f"{ch.lower()}_background") or cols.get(
            f"{ch.lower()}_nucleus_background"
        )
        expr = f'm."{mean}" - coalesce(m."{bg}", 0)' if bg else f'm."{mean}"'
        selects.append(f'{expr} as "{ch.lower()}"')
    extra = (", " + ", ".join(selects)) if selects else ""
    df = conn.execute(
        f"""
        select m.measurement_id, m.timepoint,
               m.track_id_raw, m.parent_track_id_raw,
               m.area_nucleus as area,
               m."centroid-0-nuc" as y, m."centroid-1-nuc" as x{extra}
        from measurements m
        join conditions c using (condition_id)
        join repeats r using (repeat_id)
        where r.plate_id = ? and c.well = ? and m.track_id_raw is not null
        """,
        [plate_id, well],
    ).df()
    # Drop channels this plate never measured (all NULL).
    for ch in [s.split(" as ")[1].strip('"') for s in selects]:
        if df[ch].isna().all():
            df = df.drop(columns=ch)
        else:
            df[ch] = df[ch].clip(lower=0)
    df["label"] = df["track_id_raw"].astype(int)
    df["parent_track_id_raw"] = df["parent_track_id_raw"].fillna(0)
    return df


def cell_frames(
    curated: Curated,
    det: pd.DataFrame,
    anchor: Anchor,
    start: int | None = None,
    stop: int | None = None,
) -> pd.DataFrame:
    """One row per frame of the anchored cell's curated track.

    Args:
        curated: The well's curated lineage.
        det: The well's detections (:func:`load_well`).
        anchor: ``(frame, raw label)`` of the cell.
        start: First frame to include (default: the track's first frame).
        stop: Last frame to include (default: the track's last frame).

    Returns:
        Columns ``timepoint, track_id, label, n_pieces, gap, area, y, x`` and
        the channel columns of ``det``. ``gap`` rows have ``label == 0``,
        NaN measurements and an interpolated position.

    Raises:
        cellview.tracks.edit.EditError: If the anchor does not exist.
    """
    tid = curated.resolve(anchor)
    keys = curated.detections(tid)
    if start is not None:
        keys = [k for k in keys if k[0] >= start]
    if stop is not None:
        keys = [k for k in keys if k[0] <= stop]
    if not keys:
        return pd.DataFrame(
            columns=[
                "timepoint",
                "track_id",
                "label",
                "n_pieces",
                "gap",
                "area",
                "y",
                "x",
            ]
        )

    index = pd.MultiIndex.from_tuples(keys, names=["timepoint", "label"])
    rows = det.set_index(["timepoint", "label"]).reindex(index).reset_index()
    value_cols = [
        c
        for c in det.columns
        if c
        not in {
            "measurement_id",
            "timepoint",
            "label",
            "track_id_raw",
            "parent_track_id_raw",
            "area",
        }
    ]
    weighted = rows[value_cols].multiply(rows["area"], axis=0)
    weighted["area"] = rows["area"]
    weighted["timepoint"] = rows["timepoint"]
    agg = weighted.groupby("timepoint").sum(min_count=1)
    for col in value_cols:
        agg[col] = agg[col] / agg["area"]
    largest = rows.sort_values("area").groupby("timepoint")["label"].last()
    agg["label"] = largest
    agg["n_pieces"] = rows.groupby("timepoint").size()
    agg["gap"] = False

    first, last = int(agg.index.min()), int(agg.index.max())
    full = agg.reindex(range(first, last + 1))
    missing = full["label"].isna()
    full.loc[missing, "gap"] = True
    full.loc[missing, "label"] = 0
    full.loc[missing, "n_pieces"] = 0
    for axis in ("y", "x"):
        full[axis] = full[axis].interpolate(limit_area="inside")
    full["track_id"] = tid
    full = full.reset_index().rename(columns={"index": "timepoint"})
    full["label"] = full["label"].astype(int)
    full["n_pieces"] = full["n_pieces"].astype(int)
    full["gap"] = full["gap"].astype(bool)
    ordered = [
        "timepoint",
        "track_id",
        "label",
        "n_pieces",
        "gap",
        "area",
        "y",
        "x",
    ]
    return full[ordered + [c for c in full.columns if c not in ordered]]


def well_diameter(det: pd.DataFrame) -> float:
    """Median nuclear diameter (pixels) of a well, from equivalent-circle areas."""
    return float(2 * np.sqrt(det["area"].median() / np.pi))


def add_phases(
    path: pd.DataFrame,
    det: pd.DataFrame,
    pip: str = "pip",
    geminin: str = "geminin",
) -> pd.DataFrame:
    """Add a PIP-FUCCI ``phase`` column (G1/S/G2; empty on gaps) to a cell path.

    Thresholds come from the whole well (:func:`cellview.tracks.fate.well_thresholds`),
    so the call is the same as the fate walker's.
    """
    from cellview.tracks.fate import FateParams, Thresholds, call_phases

    if pip not in det or geminin not in det or path.empty:
        return path.assign(phase="")
    thr = Thresholds(
        pip_high=float(det[pip].quantile(0.75)),
        gem_low=float(det[geminin].quantile(0.25)),
        dna_median=0.0,
        area_median=float(det["area"].median()),
    )
    measured = path[~path["gap"]].set_index("timepoint")
    calls = call_phases(
        measured.rename(columns={pip: "pip", geminin: "geminin"}),
        thr,
        FateParams(),
    )
    out = path.copy()
    out["phase"] = out["timepoint"].map(calls).fillna("")
    return out


def curated_tracks(
    curated: Curated, det: pd.DataFrame, dna: str | None = None
) -> pd.DataFrame:
    """All detections collapsed onto curated tracks, one row per track and frame.

    The same shape as :func:`cellview.tracks.repair.apply_repair` output, so
    the fate walker runs on curated data unchanged. Channel columns are
    area-weighted; ``dna`` (default: the first channel whose name contains
    ``dna``, ``dapi`` or ``hoechst``) is also exposed as ``dna``.
    """
    key = pd.Series(
        [
            curated.tracks.get((int(t), int(lab)), 0)
            for t, lab in zip(det["timepoint"], det["label"], strict=True)
        ],
        index=det.index,
    )
    d = det.assign(track_id=key)
    d = d[d["track_id"] > 0]
    skip = {
        "measurement_id",
        "timepoint",
        "label",
        "track_id_raw",
        "parent_track_id_raw",
        "area",
        "track_id",
    }
    value_cols = [c for c in d.columns if c not in skip]
    weighted = d[value_cols].multiply(d["area"], axis=0)
    weighted[["track_id", "timepoint", "area"]] = d[
        ["track_id", "timepoint", "area"]
    ]
    out = weighted.groupby(["track_id", "timepoint"], as_index=False).sum(
        min_count=1
    )
    for col in value_cols:
        out[col] = out[col] / out["area"]
    out["n_pieces"] = d.groupby(["track_id", "timepoint"]).size().to_numpy()
    out["parent_track_id"] = (
        out["track_id"].map(curated.parents).fillna(0).astype(int)
    )
    if dna is None:
        dna = next(
            (
                c
                for c in value_cols
                if any(k in c for k in ("dna", "dapi", "hoechst"))
            ),
            None,
        )
    if dna and dna != "dna":
        out["dna"] = out[dna]
    return out
