#!/usr/bin/env python
"""Report spurious Trackastra divisions caused by nucleus segmentation flicker.

Read-only. Opens CellView with ``read_only=True`` and writes nothing anywhere;
the ``--csv`` flag is the only thing that touches the filesystem, and only at a
path you name.

    uv run python scripts/diagnose_track_splits.py 5054 --well C2

Weak nuclear signal lets Cellpose split one nucleus into two masks for a few
frames, then merge them back. Trackastra reads the split as a division and the
merge as the death of one daughter, so a single continuous cell is shattered
into a parent plus two short branches.

The discriminator is **mass conservation**, not proximity — which matters
because two genuinely adjacent nuclei must not be fused:

  * false split  - the survivor's area right after its sibling vanishes returns
                   to the *parent's* area. The mass went apart and came back to
                   one object, so there was only ever one object.
  * real division - each daughter holds ~0.6 of the parent's area and keeps it.
                    Two nuclei that merely pass close by never produce a
                    parent-area-restored signature, because no single parent
                    object existed to conserve.

Division fates reported:

  remerge          one daughter ends, the sibling's area jumps back to the
                   parent's  -> split was invented, splice parent+sibling
  short_leaf       one daughter never divides, is shorter than --min-branch
                   frames and ends before the last frame -> prune, splice
  both_short       both daughters are short leaves -> drop the division and
                   let the parent terminate where it did
  kept             survives every test

Collapsing is iterative: repairing one bubble can expose another nested inside
it, so the rewrite runs to a fixed point before the after-repair numbers are
reported.

Nothing here is applied to the database. Use the printed examples to inspect
the events in napari, and validate the rules against a Mastodon-curated patch
before trusting the collapse.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd
import zarr
from scipy import ndimage
from skimage import measure, morphology

DEFAULT_DB = "~/.cellview/cellview.duckdb"
DEFAULT_CACHE = "~/omero-cache/zarr"

MEASUREMENT_QUERY = """
select
    c.well                as well,
    m.track_id            as tid,
    m.parent_track_id     as pid,
    m.timepoint           as t,
    m.area_nucleus        as area,
    m."centroid-0-nuc"    as y,
    m."centroid-1-nuc"    as x,
    m.intensity_mean_DAPI_nucleus as dna
from measurements m
join conditions c using (condition_id)
join repeats r using (repeat_id)
where r.plate_id = ?
"""


@dataclass
class Track:
    """One track's extent in a lineage, mutated as the repair is simulated."""

    begin: int
    end: int
    n_rows: int
    parent: int = 0
    children: list[int] = field(default_factory=list)


@dataclass
class Division:
    """A parent track and the fate assigned to its two daughters."""

    parent: int
    short: int
    long: int
    frame: int
    fate: str
    ratio_to_parent: float | None = None
    area_jump: float | None = None
    y: float = 0.0
    x: float = 0.0


def load_plate(
    db: Path, plate_id: int, wells: list[str] | None
) -> pd.DataFrame:
    """Read the tracked measurements for ``plate_id`` from CellView.

    Args:
        db: Path to the CellView DuckDB file.
        plate_id: OMERO plate id.
        wells: Restrict to these wells, or ``None`` for every well.

    Returns:
        One row per detection with track, lineage, time, area and centroid.

    Raises:
        SystemExit: If the plate has no rows, or no ``track_id`` values.
    """
    conn = duckdb.connect(str(db), read_only=True)
    try:
        df = conn.execute(MEASUREMENT_QUERY, [plate_id]).df()
    finally:
        conn.close()

    if df.empty:
        raise SystemExit(f"No measurements for plate {plate_id} in {db}")
    if wells:
        df = df[df.well.isin(wells)]
        if df.empty:
            raise SystemExit(f"No rows for wells {wells} in plate {plate_id}")
    if df.tid.isna().all():
        raise SystemExit(
            f"Plate {plate_id} has no track_id values — was it run with --track?"
        )

    df = df.dropna(subset=["tid"]).copy()
    df["tid"] = df.tid.astype(int)
    df["pid"] = df.pid.fillna(0).astype(int)
    df["t"] = df.t.astype(int)
    return df


def build_lineage(df: pd.DataFrame) -> tuple[dict[int, Track], dict[int, int]]:
    """Build the track table and the parent map for one well.

    Args:
        df: Detections for a single well.

    Returns:
        ``(tracks, n_children)`` where ``tracks`` maps track id to its extent
        with children attached, and ``n_children`` counts daughters per parent
        before any repair.
    """
    tracks: dict[int, Track] = {
        cast(int, tid): Track(
            begin=int(sub.t.min()),
            end=int(sub.t.max()),
            n_rows=int(len(sub)),
            parent=int(sub.pid.iloc[0]),
        )
        for tid, sub in df.groupby("tid")
    }
    for tid, track in tracks.items():
        if track.parent and track.parent in tracks:
            tracks[track.parent].children.append(tid)
    n_children = {tid: len(tr.children) for tid, tr in tracks.items()}
    return tracks, n_children


def area_lookup(df: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Map ``(track_id, timepoint)`` to nuclear area for one well."""
    return {
        (int(tid), int(t)): float(a)
        for tid, t, a in zip(df.tid, df.t, df.area, strict=True)
    }


def centroid_lookup(
    df: pd.DataFrame,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Map ``(track_id, timepoint)`` to the nuclear centroid for one well."""
    return {
        (int(tid), int(t)): (float(y), float(x))
        for tid, t, y, x in zip(df.tid, df.t, df.y, df.x, strict=True)
    }


def classify_division(
    parent: int,
    kids: list[int],
    tracks: dict[int, Track],
    areas: dict[tuple[int, int], float],
    centroids: dict[tuple[int, int], tuple[float, float]],
    last_frame: int,
    min_branch: int,
    remerge_lo: float,
    remerge_hi: float,
    area_jump: float,
) -> Division | None:
    """Decide whether a two-daughter division is real or a segmentation artefact.

    Args:
        parent: Track id of the dividing track.
        kids: Exactly two daughter track ids.
        tracks: Lineage table for the well.
        areas: ``(track, frame)`` to nuclear area.
        centroids: ``(track, frame)`` to nuclear centroid.
        last_frame: Final timepoint in the well; branches ending here are
            censored rather than short-lived, so they are never pruned.
        min_branch: A daughter shorter than this many frames cannot be a real
            cell if it also never divides.
        remerge_lo: Lower bound on survivor-area / parent-area for a remerge.
        remerge_hi: Upper bound on the same ratio.
        area_jump: Minimum factor by which the survivor's own area must grow
            when its sibling vanishes.

    Returns:
        The classified division, or ``None`` if ``kids`` is malformed.
    """
    if len(kids) != 2:
        return None
    a, b = kids
    parent_area = areas.get((parent, tracks[parent].end))
    short, long = sorted(kids, key=lambda k: tracks[k].n_rows)
    split_frame = min(tracks[a].begin, tracks[b].begin)
    y, x = centroids.get((short, tracks[short].begin), (0.0, 0.0))

    def make(fate: str, ratio: float | None, jump: float | None) -> Division:
        return Division(
            parent=parent,
            short=short,
            long=long,
            frame=split_frame,
            fate=fate,
            ratio_to_parent=ratio,
            area_jump=jump,
            y=y,
            x=x,
        )

    # Remerge: the shorter branch stops, and the survivor immediately reclaims
    # the parent's full area. Mass returned to a single object.
    if parent_area:
        end_short = tracks[short].end
        survivor_after = areas.get((long, end_short + 1))
        survivor_at_end = areas.get((long, end_short))
        if survivor_after and survivor_at_end and tracks[long].end > end_short:
            ratio = survivor_after / parent_area
            jump = survivor_after / survivor_at_end
            if remerge_lo <= ratio <= remerge_hi and jump >= area_jump:
                return make("remerge", ratio, jump)

    short_leaf = (
        not tracks[short].children
        and tracks[short].n_rows < min_branch
        and tracks[short].end < last_frame
    )
    long_leaf = (
        not tracks[long].children
        and tracks[long].n_rows < min_branch
        and tracks[long].end < last_frame
    )
    if short_leaf and long_leaf:
        return make("both_short", None, None)
    if short_leaf:
        return make("short_leaf", None, None)
    return make("kept", None, None)


def repair_well(
    df: pd.DataFrame,
    min_branch: int,
    remerge_lo: float,
    remerge_hi: float,
    area_jump: float,
) -> tuple[dict[int, Track], dict[int, Track], list[Division], int]:
    """Simulate the lineage repair for one well until it reaches a fixed point.

    Collapsing a bubble can expose another nested inside it, so divisions are
    re-evaluated after every rewrite.

    Args:
        df: Detections for a single well.
        min_branch: Minimum frames a non-dividing daughter must live.
        remerge_lo: Lower bound on survivor-area / parent-area for a remerge.
        remerge_hi: Upper bound on the same ratio.
        area_jump: Minimum area growth factor for the survivor.

    Returns:
        ``(before, after, divisions, rows_dropped)`` — the lineage as imported,
        the lineage after repair, every division with its fate, and the number
        of detections belonging to branches judged spurious.
    """
    before, _ = build_lineage(df)
    areas = area_lookup(df)
    centroids = centroid_lookup(df)
    last_frame = int(df.t.max())

    tracks = {
        tid: Track(tr.begin, tr.end, tr.n_rows, tr.parent, list(tr.children))
        for tid, tr in before.items()
    }
    divisions: list[Division] = []
    rows_dropped = 0
    resolved: set[int] = set()

    changed = True
    while changed:
        changed = False
        for parent in list(tracks):
            track = tracks.get(parent)
            if track is None or len(track.children) != 2:
                continue
            if parent in resolved:
                continue
            verdict = classify_division(
                parent,
                track.children,
                tracks,
                areas,
                centroids,
                last_frame,
                min_branch,
                remerge_lo,
                remerge_hi,
                area_jump,
            )
            if verdict is None:
                continue
            if verdict.fate == "kept":
                resolved.add(parent)
                divisions.append(verdict)
                continue

            divisions.append(verdict)
            changed = True
            if verdict.fate == "both_short":
                for kid in (verdict.short, verdict.long):
                    rows_dropped += tracks[kid].n_rows
                    _detach(tracks, kid)
                track.children = []
                continue

            rows_dropped += tracks[verdict.short].n_rows
            _detach(tracks, verdict.short)
            track.children = [verdict.long]
            _splice(tracks, parent, verdict.long)

    return before, tracks, divisions, rows_dropped


def _detach(tracks: dict[int, Track], tid: int) -> None:
    """Remove ``tid`` and everything descended from it."""
    stack = [tid]
    while stack:
        cur = stack.pop()
        node = tracks.pop(cur, None)
        if node is not None:
            stack.extend(node.children)


def _splice(tracks: dict[int, Track], parent: int, child: int) -> None:
    """Fuse ``child`` into ``parent``, inheriting the child's children."""
    kid = tracks.pop(child)
    node = tracks[parent]
    node.end = kid.end
    node.n_rows += kid.n_rows
    node.children = kid.children
    for grandchild in kid.children:
        tracks[grandchild].parent = parent


def find_orphan_restarts(
    df: pd.DataFrame, tracks: dict[int, Track], radius: float
) -> int:
    """Count unparented tracks that begin where another track just ended.

    Trackastra cannot express a merge, so when two flickering masks fuse it may
    terminate both and open a fresh id. Those are stitch candidates rather than
    division errors, and are reported separately.

    Args:
        df: Detections for a single well.
        tracks: Lineage table for the well.
        radius: Centroid distance in pixels within which a restart counts.

    Returns:
        Number of unparented mid-movie tracks adjacent to a just-ended track.
    """
    centroids = centroid_lookup(df)
    ends: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for tid, track in tracks.items():
        pos = centroids.get((tid, track.end))
        if pos:
            ends[track.end].append(pos)

    count = 0
    for tid, track in tracks.items():
        if track.parent or track.begin == 0:
            continue
        pos = centroids.get((tid, track.begin))
        if pos is None:
            continue
        prior = ends.get(track.begin - 1, [])
        if any(
            (pos[0] - q[0]) ** 2 + (pos[1] - q[1]) ** 2 <= radius**2
            for q in prior
        ):
            count += 1
    return count


def length_summary(tracks: dict[int, Track], interval: float) -> str:
    """Format the track-length distribution as a one-line summary."""
    lengths = np.array([t.n_rows for t in tracks.values()])
    if lengths.size == 0:
        return "no tracks"
    hours = lengths * interval / 60.0
    return (
        f"n={lengths.size:5d}  median={np.median(lengths):5.0f} fr "
        f"({np.median(hours):4.1f} h)  p90={np.percentile(lengths, 90):5.0f} "
        f"max={lengths.max():4d}  >=10h: {(hours >= 10).sum():4d}  "
        f">=24h: {(hours >= 24).sum():4d}"
    )


def report_well(
    well: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Print the before/after report for one well and return its divisions."""
    before, after, divisions, rows_dropped = repair_well(
        df, args.min_branch, args.remerge_lo, args.remerge_hi, args.area_jump
    )
    fates = pd.Series([d.fate for d in divisions]).value_counts()
    n_div_before = sum(1 for t in before.values() if len(t.children) == 2)
    n_div_after = sum(1 for t in after.values() if len(t.children) == 2)

    print(f"\n{'=' * 78}\nWell {well}")
    print(f"{'-' * 78}")
    print(
        f"  frames {int(df.t.min())}-{int(df.t.max())}, {len(df):,} detections"
    )
    print(f"  before  {length_summary(before, args.interval)}")
    print(f"  after   {length_summary(after, args.interval)}")
    print(f"  divisions  {n_div_before}  ->  {n_div_after}")
    print(
        f"  detections in branches judged spurious: {rows_dropped:,} "
        f"({rows_dropped / len(df):.1%} of the well)"
    )
    print("\n  division fates:")
    for fate in ("remerge", "short_leaf", "both_short", "kept"):
        n = int(fates.get(fate, 0))
        share = n / max(len(divisions), 1)
        print(f"    {fate:<12} {n:5d}  ({share:5.1%})")

    orphans = find_orphan_restarts(df, before, args.restart_radius)
    print(
        f"\n  unparented tracks restarting next to a just-ended track "
        f"(merge/stitch candidates, not repaired here): {orphans}"
    )

    rows = pd.DataFrame(
        [
            {
                "well": well,
                "parent_track_id": d.parent,
                "pruned_track_id": d.short,
                "kept_track_id": d.long,
                "split_frame": d.frame,
                "fate": d.fate,
                "survivor_area_over_parent": d.ratio_to_parent,
                "survivor_area_jump": d.area_jump,
                "y": d.y,
                "x": d.x,
            }
            for d in divisions
        ]
    )
    _print_examples(rows, args.examples, args.interval)
    if args.remeasure:
        report_remeasure(well, rows, df, args)
    return rows


def _print_examples(rows: pd.DataFrame, n: int, interval: float) -> None:
    """Print the clearest collapsed bubbles for inspection in napari."""
    if n <= 0 or rows.empty:
        return
    best = rows[rows.fate == "remerge"].copy()
    if best.empty:
        return
    best["off"] = (best.survivor_area_over_parent - 1.0).abs()
    best = best.nsmallest(n, "off")
    print(f"\n  clearest {len(best)} remerges — navigate to these in napari:")
    print(
        f"    {'parent':>8} {'pruned':>8} {'frame':>6} {'time':>7} "
        f"{'surv/par':>9} {'jump':>6}  centroid (y, x)"
    )
    for row in best.to_dict("records"):
        frame = int(cast(int, row["split_frame"]))
        stamp = f"{frame * interval / 60:.1f}h"
        print(
            f"    {int(cast(int, row['parent_track_id'])):>8} "
            f"{int(cast(int, row['pruned_track_id'])):>8} "
            f"{frame:>6} {stamp:>7} "
            f"{float(cast(float, row['survivor_area_over_parent'])):>9.2f} "
            f"{float(cast(float, row['survivor_area_jump'])):>6.2f}"
            f"  ({float(cast(float, row['y'])):7.1f}, "
            f"{float(cast(float, row['x'])):7.1f})"
        )


def pick_nucleus_channel(names: list[str], hint: str | None) -> int:
    """Choose the intensity channel that carries the nuclear stain.

    Args:
        names: Channel labels from the cached OME-NGFF ``omero`` metadata.
        hint: Explicit channel name from the command line, matched case
            insensitively as a substring.

    Returns:
        Index into the channel axis.

    Raises:
        SystemExit: If ``hint`` matches no channel.
    """
    if hint:
        for i, n in enumerate(names):
            if hint.lower() in n.lower():
                return i
        raise SystemExit(f"Channel {hint!r} not in cached channels {names}")
    for i, n in enumerate(names):
        if n.lower().endswith("_nucleus"):
            return i
    return 0


def open_well_arrays(
    cache_root: Path, plate_id: int, well: str, hint: str | None
) -> tuple[Any, Any, int, str]:
    """Open the cached nucleus labels and intensity stack for one well.

    Args:
        cache_root: Directory holding ``plate_<id>.zarr``.
        plate_id: OMERO plate id.
        well: Well name such as ``C2``.
        hint: Optional nucleus-channel name override.

    Returns:
        ``(labels, image, channel_index, channel_name)``. ``labels`` is
        ``(T, Y, X)`` with pixel values equal to ``track_id``; ``image`` is
        ``(T, C, Y, X)``.

    Raises:
        SystemExit: If the cached plate or well is missing.
    """
    path = (cache_root / f"plate_{plate_id}.zarr").expanduser()
    if not path.exists():
        raise SystemExit(
            f"No zarr cache at {path}. Build it from the napari well widget, "
            "or pass --cache-root."
        )
    row, col = well[0], well[1:]
    group = f"{row}/{col}/0"
    try:
        root = zarr.open(str(path), mode="r")
        labels = root[f"{group}/labels/nuclei/0"]
        image = root[f"{group}/0"]
    except KeyError as exc:  # pragma: no cover - depends on cache contents
        raise SystemExit(f"Well {well} not in {path}: {exc}") from exc

    attrs = json.loads((path / row / col / "0" / ".zattrs").read_text())
    names = [
        str(c.get("label", ""))
        for c in attrs.get("omero", {}).get("channels", [])
    ]
    idx = pick_nucleus_channel(names, hint)
    return labels, image, idx, names[idx] if names else f"channel {idx}"


def _crop_bounds(
    centroids: list[tuple[float, float]],
    areas: list[float],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Bounding box around the fragments, with room for the merged object."""
    ys = [c[0] for c in centroids]
    xs = [c[1] for c in centroids]
    margin = 3.0 * float(np.sqrt(max(areas) / np.pi)) + 24.0
    y0 = max(int(min(ys) - margin), 0)
    x0 = max(int(min(xs) - margin), 0)
    y1 = min(int(max(ys) + margin), shape[0])
    x1 = min(int(max(xs) + margin), shape[1])
    return y0, y1, x0, x1


def _measure_mask(
    mask: npt.NDArray[Any], intensity: npt.NDArray[Any]
) -> dict[str, float] | None:
    """Measure area, mean intensity and shape of a boolean region.

    Args:
        mask: Boolean region, treated as a single object.
        intensity: Intensity image of the same shape.

    Returns:
        Feature dict, or ``None`` if the mask is empty.
    """
    if not mask.any():
        return None
    props = measure.regionprops(  # type: ignore[no-untyped-call]
        mask.astype(np.uint8), intensity_image=intensity
    )[0]
    return {
        "area": float(props.area),
        "mean": float(props.intensity_mean),
        "solidity": float(props.solidity),
        "eccentricity": float(props.eccentricity),
    }


def remeasure_event(
    labels: Any,
    image: Any,
    channel: int,
    parent: int,
    pruned: int,
    kept: int,
    frame: int,
    areas: dict[tuple[int, int], float],
    means: dict[tuple[int, int], float],
    centroids: dict[tuple[int, int], tuple[float, float]],
    close_radius: int,
) -> dict[str, Any] | None:
    """Re-measure one flagged remerge from the cached masks.

    Compares four views of the same nucleus at the frame it was split:
    the parent in the frame *before* the split (the reference — it was one
    object then), the two fragments as stored, their arithmetic merge, and a
    fresh pixel measurement of the relabelled union.

    Args:
        labels: ``(T, Y, X)`` label array whose values are track ids.
        image: ``(T, C, Y, X)`` intensity array.
        channel: Index of the nuclear channel.
        parent: Parent track id.
        pruned: Track id judged spurious.
        kept: Surviving sibling track id.
        frame: First frame at which both fragments exist.
        areas: ``(track, frame)`` to nuclear area in pixels.
        means: ``(track, frame)`` to mean nuclear intensity.
        centroids: ``(track, frame)`` to nuclear centroid.
        close_radius: Disk radius used to bridge the split seam.

    Returns:
        Comparison record, or ``None`` if the frames are unusable.
    """
    a1 = areas.get((pruned, frame))
    a2 = areas.get((kept, frame))
    m1 = means.get((pruned, frame))
    m2 = means.get((kept, frame))
    c1 = centroids.get((pruned, frame))
    c2 = centroids.get((kept, frame))
    if None in (a1, a2, m1, m2) or c1 is None or c2 is None:
        return None
    assert (
        a1 is not None and a2 is not None and m1 is not None and m2 is not None
    )

    y0, y1, x0, x1 = _crop_bounds([c1, c2], [a1, a2], labels.shape[1:])
    lab = np.asarray(labels[frame, y0:y1, x0:x1])
    img = np.asarray(image[frame, channel, y0:y1, x0:x1])
    union = (lab == pruned) | (lab == kept)
    closed = ndimage.binary_closing(
        union,
        structure=morphology.disk(close_radius),  # type: ignore[no-untyped-call]
    )

    record: dict[str, Any] = {
        "parent": parent,
        "pruned": pruned,
        "kept": kept,
        "frame": frame,
        "frag_area_1": a1,
        "frag_area_2": a2,
        "frag_mean_1": m1,
        "frag_mean_2": m2,
        "arith_area": a1 + a2,
        "arith_mean": (m1 * a1 + m2 * a2) / (a1 + a2),
    }
    for key, mask in (("union", union), ("closed", closed)):
        measured = _measure_mask(mask, img)
        if measured:
            for name, value in measured.items():
                record[f"{key}_{name}"] = value

    # Reference: the parent one frame earlier, when it was a single object.
    if frame > 0:
        ref_lab = np.asarray(labels[frame - 1, y0:y1, x0:x1])
        ref_img = np.asarray(image[frame - 1, channel, y0:y1, x0:x1])
        ref = _measure_mask(ref_lab == parent, ref_img)
        if ref:
            for name, value in ref.items():
                record[f"ref_{name}"] = value
    return record


def report_remeasure(
    well: str,
    rows: pd.DataFrame,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Preview merged measurements for the flagged remerges of one well.

    Nothing is written; this only shows what the merge would produce.

    Args:
        well: Well name.
        rows: Classified divisions for this well.
        df: Detections for this well.
        args: Parsed command line.
    """
    events = rows[rows.fate == "remerge"]
    if events.empty:
        print("\n  no remerges to re-measure")
        return
    labels, image, channel, channel_name = open_well_arrays(
        args.cache_root, args.plate_id, well, args.channel
    )
    areas = area_lookup(df)
    centroids = centroid_lookup(df)
    means = {
        (int(tid), int(t)): float(d)
        for tid, t, d in zip(df.tid, df.t, df.dna, strict=True)
        if pd.notna(d)
    }

    sample = events.head(args.remeasure)
    print(
        f"\n  re-measure preview — nucleus channel {channel_name!r}, "
        f"{len(sample)} of {len(events)} remerges (nothing written):"
    )
    records = []
    for row in sample.to_dict("records"):
        rec = remeasure_event(
            labels,
            image,
            channel,
            int(cast(int, row["parent_track_id"])),
            int(cast(int, row["pruned_track_id"])),
            int(cast(int, row["kept_track_id"])),
            int(cast(int, row["split_frame"])),
            areas,
            means,
            centroids,
            args.close_radius,
        )
        if rec:
            records.append(rec)
    if not records:
        print("    no event could be re-measured from the cache")
        return

    print(
        f"    {'parent':>7} {'frame':>5} | {'fragments (area/mean)':>25} | "
        f"{'arith':>13} | {'pixels(closed)':>14} | {'ref parent t-1':>14} | "
        f"{'sol':>5} {'ecc':>5}"
    )
    for r in records:
        frag = (
            f"{r['frag_area_1']:5.0f}/{r['frag_mean_1']:6.0f} + "
            f"{r['frag_area_2']:5.0f}/{r['frag_mean_2']:6.0f}"
        )
        arith = f"{r['arith_area']:5.0f}/{r['arith_mean']:6.0f}"
        pix = (
            f"{r.get('closed_area', float('nan')):5.0f}/"
            f"{r.get('closed_mean', float('nan')):6.0f}"
        )
        ref = (
            f"{r.get('ref_area', float('nan')):5.0f}/"
            f"{r.get('ref_mean', float('nan')):6.0f}"
            if "ref_area" in r
            else f"{'-':>12}"
        )
        print(
            f"    {r['parent']:>7} {r['frame']:>5} | {frag:>25} | {arith:>13} | "
            f"{pix:>14} | {ref:>14} | "
            f"{r.get('closed_solidity', float('nan')):5.2f} "
            f"{r.get('closed_eccentricity', float('nan')):5.2f}"
        )

    tab = pd.DataFrame(records)
    print("\n    medians across the sample:")
    if "ref_area" in tab:
        ok = tab.dropna(subset=["ref_area"])
        print(
            f"      merged area / parent area (t-1)  = "
            f"{(ok.closed_area / ok.ref_area).median():.3f}   (1.0 = mass restored)"
        )
        print(
            f"      merged mean / parent mean (t-1)  = "
            f"{(ok.closed_mean / ok.ref_mean).median():.3f}"
        )
    print(
        f"      seam recovered by closing        = "
        f"{(tab.closed_area - tab.union_area).median():.0f} px "
        f"({((tab.closed_area - tab.union_area) / tab.closed_area).median():.1%})"
    )
    print(
        f"      arithmetic vs pixel area         = "
        f"{((tab.arith_area - tab.closed_area) / tab.closed_area).median():+.1%}"
    )
    print(
        f"      arithmetic vs pixel mean         = "
        f"{((tab.arith_mean - tab.closed_mean) / tab.closed_mean).median():+.1%}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("plate_id", type=int, help="OMERO plate id, e.g. 5054")
    p.add_argument(
        "--well",
        action="append",
        dest="wells",
        help="Restrict to this well (repeatable). Default: every well.",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"CellView DuckDB file. Default: $DATABASE_PATH or {DEFAULT_DB}",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=20.0,
        help="Minutes between frames, for the human-readable times (default 20)",
    )
    p.add_argument(
        "--min-branch",
        type=int,
        default=6,
        help="Frames a non-dividing daughter must live to be believable "
        "(default 6 = 2 h at 20 min)",
    )
    p.add_argument(
        "--remerge-lo",
        type=float,
        default=0.75,
        help="Lower bound on survivor-area / parent-area for a remerge",
    )
    p.add_argument(
        "--remerge-hi",
        type=float,
        default=1.35,
        help="Upper bound on survivor-area / parent-area for a remerge",
    )
    p.add_argument(
        "--area-jump",
        type=float,
        default=1.25,
        help="Minimum area growth of the survivor when its sibling vanishes",
    )
    p.add_argument(
        "--restart-radius",
        type=float,
        default=30.0,
        help="Pixel radius for detecting unparented restarts (default 30)",
    )
    p.add_argument(
        "--examples",
        type=int,
        default=15,
        help="How many clearest remerges to list per well (0 to suppress)",
    )
    p.add_argument(
        "--remeasure",
        type=int,
        nargs="?",
        const=10,
        default=0,
        metavar="N",
        help="Preview merged measurements for N flagged remerges from the "
        "cached masks (default 10 when the flag is given). Writes nothing.",
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path(DEFAULT_CACHE),
        help=f"Directory holding plate_<id>.zarr (default {DEFAULT_CACHE})",
    )
    p.add_argument(
        "--channel",
        default=None,
        help="Nucleus channel name for re-measurement. Default: the cached "
        "channel whose label ends in _nucleus.",
    )
    p.add_argument(
        "--close-radius",
        type=int,
        default=2,
        help="Disk radius used to bridge the split seam when re-measuring "
        "(default 2)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write every classified division to. "
        "The only file this script creates.",
    )
    return p.parse_args(argv)


def resolve_db(explicit: Path | None) -> Path:
    """Pick the CellView database, preferring an explicit path.

    Args:
        explicit: Path given on the command line, if any.

    Returns:
        An existing path to the CellView DuckDB file.

    Raises:
        SystemExit: If the resolved path does not exist.
    """
    candidate = explicit or Path(os.environ.get("DATABASE_PATH", DEFAULT_DB))
    db = candidate.expanduser()
    if not db.exists():
        raise SystemExit(f"CellView database not found: {db}")
    return db


def main(argv: list[str] | None = None) -> None:
    """Run the diagnostic and print the report."""
    args = parse_args(argv)
    db = resolve_db(args.db)
    df = load_plate(db, args.plate_id, args.wells)

    print(f"Plate {args.plate_id} — {db}")
    print("Read-only: no database rows are modified by this script.")

    frames = []
    for well, sub in df.groupby("well"):
        frames.append(report_well(str(well), sub, args))

    if args.csv and frames:
        out = pd.concat(frames, ignore_index=True)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.csv, index=False)
        print(f"\nWrote {len(out):,} classified divisions to {args.csv}")


if __name__ == "__main__":
    main()
