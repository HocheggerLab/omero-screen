#!/usr/bin/env python
"""Detect tile-seam misalignment in a stitched plate from its measurements.

Read-only. Opens CellView with ``read_only=True`` and writes nothing; ``--csv``
is the only flag that touches the filesystem, at a path you name.

    uv run python scripts/diagnose_stitch_seams.py 5054

The stitcher places tiles by grid arithmetic (``layout_to_offsets`` in
``omero_utils.stitching``): stage positions give the (col, row) index, and the
canvas spacing is ``tile_w - overlap_x``. The stage *distances* are discarded,
so if ``overlap_x`` is wrong every seam carries a constant shear, and a nucleus
straddling it is segmented twice — once in each tile, displaced by the error.

That is measurable without touching a single pixel. Near a seam, pairs of
detections separated by the shear appear far more often than chance; away from
one, the same statistic is just the nuclear nearest-neighbour distribution. So
the test is a **histogram of pair separations at the seams against an interior
control**, and the excess mode is the misalignment in pixels.

  no peak          -> the calibration is right for this objective
  peak at d px     -> overlap is short by d; the correct value is overlap + d

Wells are **pooled by default**: misalignment is a property of the plate, not of
a well, and a sparse well on its own produces noise rather than a measurement.
Use ``--per-well`` to see the breakdown. A verdict is only issued once there are
``--min-pairs`` on-seam pairs and the peak sits well inside the search window;
otherwise the numbers are printed with the reason no call was made.

Works on any stitched plate, fixed or timelapse, at any magnification — the grid
is inferred from the detection extent and the field count.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import KDTree

DEFAULT_DB = "~/.cellview/cellview.duckdb"

MEASUREMENT_QUERY = """
select
    c.well             as well,
    m.image_id         as image_id,
    m.timepoint        as t,
    m."centroid-0-nuc" as y,
    m."centroid-1-nuc" as x
from measurements m
join conditions c using (condition_id)
join repeats r using (repeat_id)
where r.plate_id = ?
"""

AXES = (
    ("x", "vertical seams (displacement along x)"),
    ("y", "horizontal seams (displacement along y)"),
)


@dataclass
class Separations:
    """Pair separations along one axis, split into on-seam and control sets."""

    on_seam: list[npt.NDArray[Any]] = field(default_factory=list)
    control: list[npt.NDArray[Any]] = field(default_factory=list)

    def extend(
        self, on_seam: npt.NDArray[Any], control: npt.NDArray[Any]
    ) -> None:
        """Accumulate one well's separations."""
        self.on_seam.append(on_seam)
        self.control.append(control)

    def pooled(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Concatenate everything accumulated so far."""
        empty: npt.NDArray[Any] = np.empty(0)
        return (
            np.concatenate(self.on_seam) if self.on_seam else empty,
            np.concatenate(self.control) if self.control else empty,
        )


def load_plate(
    db: Path, plate_id: int, wells: list[str] | None
) -> pd.DataFrame:
    """Read nuclear centroids for a plate from CellView.

    Args:
        db: Path to the CellView DuckDB file.
        plate_id: OMERO plate id.
        wells: Restrict to these wells, or ``None`` for every well.

    Returns:
        One row per detection with well, field, timepoint and centroid.

    Raises:
        SystemExit: If the plate or the requested wells have no rows.
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
    return df.dropna(subset=["y", "x"]).copy()


def infer_grid(
    df: pd.DataFrame, tile: int | None, grid: tuple[int, int] | None
) -> tuple[int, int, list[float], list[float]]:
    """Infer the tile grid and the seam coordinates for one well.

    A plate may not image every field of its grid — plate 3868 has 20-21 of 25 —
    so the square root is rounded **up** rather than required to be exact. Pass
    ``--grid`` for any layout that is not square.

    Args:
        df: Detections for a single well.
        tile: Known tile size in pixels, or ``None`` to infer the pitch.
        grid: Explicit ``(n_cols, n_rows)``, or ``None`` to infer.

    Returns:
        ``(n_cols, n_rows, seams_x, seams_y)``.
    """
    if grid is not None:
        n_cols, n_rows = grid
    else:
        n_fields = int(df.image_id.nunique())
        n_cols = n_rows = int(np.ceil(np.sqrt(n_fields)))

    pitch_x = float(df.x.max()) / n_cols if tile is None else float(tile)
    pitch_y = float(df.y.max()) / n_rows if tile is None else float(tile)
    return (
        n_cols,
        n_rows,
        [pitch_x * k for k in range(1, n_cols)],
        [pitch_y * k for k in range(1, n_rows)],
    )


def pair_separations(df: pd.DataFrame, max_sep: float) -> pd.DataFrame:
    """Find all close detection pairs within each timepoint of one well.

    Args:
        df: Detections for a single well.
        max_sep: Maximum centroid separation in pixels to consider.

    Returns:
        One row per pair with the first point's position and the displacement.
    """
    rows = []
    for _, sub in df.groupby("t"):
        pts = sub[["y", "x"]].to_numpy()
        if len(pts) < 2:
            continue
        for i, j in KDTree(pts).query_pairs(max_sep):
            rows.append(
                (
                    pts[i, 0],
                    pts[i, 1],
                    pts[j, 0] - pts[i, 0],
                    pts[j, 1] - pts[i, 1],
                )
            )
    return pd.DataFrame(rows, columns=["y", "x", "dy", "dx"])


def _distance_to_nearest(
    values: npt.NDArray[Any], seams: list[float]
) -> npt.NDArray[Any]:
    """Distance from each value to the nearest seam (``inf`` when none exist)."""
    if not seams:
        return np.full(values.shape, np.inf)
    return cast(
        npt.NDArray[Any],
        np.min([np.abs(values - s) for s in seams], axis=0),
    )


def split_by_seam(
    pairs: pd.DataFrame,
    seams: list[float],
    axis: str,
    args: argparse.Namespace,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Split one axis' pair separations into on-seam and interior samples.

    Args:
        pairs: Output of :func:`pair_separations`.
        seams: Seam coordinates along ``axis``.
        axis: ``"x"`` for vertical seams, ``"y"`` for horizontal ones.
        args: Parsed command line, for the band and alignment thresholds.

    Returns:
        ``(on_seam, control)`` separations along ``axis``.
    """
    empty: npt.NDArray[Any] = np.empty(0)
    if not seams or pairs.empty:
        return empty, empty
    along, across = ("dx", "dy") if axis == "x" else ("dy", "dx")

    distance = _distance_to_nearest(pairs[axis].to_numpy(), seams)
    # A duplicated object is displaced along the seam normal, so a pair
    # separated mostly perpendicular to that is an unrelated neighbour.
    aligned = np.abs(pairs[across].to_numpy()) < args.perpendicular
    separation = np.abs(pairs[along].to_numpy())
    return (
        separation[(distance < args.band) & aligned],
        separation[(distance > args.interior) & aligned],
    )


def seam_profile(
    on_seam: npt.NDArray[Any],
    control: npt.NDArray[Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Compare on-seam separations against the interior control.

    Args:
        on_seam: Separations sampled near the seams.
        control: Separations sampled far from any seam.
        args: Parsed command line, for the histogram binning.

    Returns:
        Summary of the largest excess over the control, or ``None`` when either
        sample is empty.
    """
    if len(on_seam) == 0 or len(control) == 0:
        return None
    bins = np.arange(0, args.max_sep + args.bin_width, args.bin_width)
    seam_frac = np.histogram(on_seam, bins=bins)[0] / len(on_seam)
    ctrl_frac = np.histogram(control, bins=bins)[0] / len(control)

    excess = seam_frac - ctrl_frac
    peak = int(np.argmax(excess))
    ratio = (
        seam_frac[peak] / ctrl_frac[peak] if ctrl_frac[peak] > 0 else np.inf
    )
    return {
        "n_seam": int(len(on_seam)),
        "n_control": int(len(control)),
        "lo": float(bins[peak]),
        "hi": float(bins[peak + 1]),
        "excess": float(excess[peak]),
        "ratio": float(ratio),
        "seam_frac": seam_frac,
        "ctrl_frac": ctrl_frac,
        "bins": bins,
    }


def verdict(profile: dict[str, Any] | None, args: argparse.Namespace) -> str:
    """Turn a seam profile into a call, or explain why none can be made.

    Two guards stop a sparse sample inventing a shear: a minimum pair count, and
    a requirement that the peak sits well inside the search window. A genuine
    misalignment is a small displacement; a mode at the far edge is the tail of
    the neighbour distribution.
    """
    if profile is None:
        return "no interior seams on this axis"
    if profile["n_seam"] < args.min_pairs:
        return (
            f"INSUFFICIENT SAMPLE - {profile['n_seam']} on-seam pairs "
            f"(need {args.min_pairs}); pool more wells before judging"
        )
    if profile["ratio"] < args.min_ratio:
        return "CLEAN - no duplication peak"
    if profile["lo"] > args.max_shear * args.max_sep:
        return (
            f"NO CALL - excess at {profile['lo']:.0f}-{profile['hi']:.0f} px is "
            "in the tail of the search window, not a duplication peak"
        )
    return (
        f"MISALIGNED by ~{profile['lo']:.0f}-{profile['hi']:.0f} px "
        "-> increase overlap by that much"
    )


def print_profile(
    profile: dict[str, Any] | None,
    label: str,
    args: argparse.Namespace,
    indent: str = "  ",
) -> None:
    """Print one axis' result, optionally with the full histogram."""
    print(f"{indent}{label}")
    if profile is None:
        print(f"{indent}  {verdict(profile, args)}")
        return
    print(
        f"{indent}  pairs: {profile['n_seam']} on-seam / "
        f"{profile['n_control']} interior"
    )
    print(
        f"{indent}  largest excess at "
        f"{profile['lo']:.0f}-{profile['hi']:.0f} px "
        f"({profile['ratio']:.2f}x the interior rate)"
    )
    print(f"{indent}  {verdict(profile, args)}")
    if not args.histogram:
        return
    bins = profile["bins"]
    for i, (s, c) in enumerate(
        zip(profile["seam_frac"], profile["ctrl_frac"], strict=True)
    ):
        if s == 0 and c == 0:
            continue
        print(
            f"{indent}    {bins[i]:5.0f}-{bins[i + 1]:<5.0f} "
            f"seam {'#' * int(300 * s):<32} {s:.3f} | "
            f"interior {'#' * int(300 * c):<20} {c:.3f}"
        )


def _grid_arg(value: str) -> tuple[int, int]:
    """Parse a ``COLSxROWS`` grid argument."""
    try:
        cols, rows = (int(v) for v in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected COLSxROWS, e.g. 5x5; got {value!r}"
        ) from exc
    return cols, rows


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
        "--per-well",
        action="store_true",
        help="Also print a per-well breakdown (noisy on sparse wells)",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"CellView DuckDB file. Default: $DATABASE_PATH or {DEFAULT_DB}",
    )
    p.add_argument(
        "--tile",
        type=int,
        default=None,
        help="Tile size in pixels. Default: infer the pitch from the extent.",
    )
    p.add_argument(
        "--grid",
        type=_grid_arg,
        default=None,
        metavar="COLSxROWS",
        help="Field grid, e.g. 5x5. Default: inferred from the field count, "
        "rounding up to a square.",
    )
    p.add_argument(
        "--max-sep",
        type=float,
        default=70.0,
        help="Largest pair separation to consider, in pixels (default 70)",
    )
    p.add_argument(
        "--band",
        type=float,
        default=70.0,
        help="Half-width of the on-seam band, in pixels (default 70)",
    )
    p.add_argument(
        "--interior",
        type=float,
        default=200.0,
        help="Minimum distance from a seam for the control region (default 200)",
    )
    p.add_argument(
        "--perpendicular",
        type=float,
        default=15.0,
        help="Maximum displacement across the seam for a duplication candidate "
        "(default 15)",
    )
    p.add_argument(
        "--bin-width",
        type=float,
        default=2.0,
        help="Histogram bin width in pixels (default 2)",
    )
    p.add_argument(
        "--min-ratio",
        type=float,
        default=1.5,
        help="Excess over the interior rate above which a peak counts as "
        "misalignment (default 1.5)",
    )
    p.add_argument(
        "--min-pairs",
        type=int,
        default=500,
        help="On-seam pairs required before a verdict is issued (default 500)",
    )
    p.add_argument(
        "--max-shear",
        type=float,
        default=0.6,
        help="A peak beyond this fraction of --max-sep is treated as the tail "
        "of the neighbour distribution, not a shear (default 0.6)",
    )
    p.add_argument(
        "--histogram",
        action="store_true",
        help="Print the full seam-vs-interior histogram",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write the pooled profiles to. "
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
    """Run the seam diagnostic and print the report."""
    args = parse_args(argv)
    db = resolve_db(args.db)
    df = load_plate(db, args.plate_id, args.wells)

    print(f"Plate {args.plate_id} — {db}")
    print("Read-only: no database rows are modified by this script.")

    pooled = {axis: Separations() for axis, _ in AXES}

    for well, sub in df.groupby("well"):
        n_cols, n_rows, seams_x, seams_y = infer_grid(
            sub, args.tile, args.grid
        )
        pairs = pair_separations(sub, args.max_sep)
        if args.per_well:
            print(f"\n{'-' * 78}\nWell {well}")
            print(
                f"  {len(sub):,} detections, {sub.image_id.nunique()} fields "
                f"({n_cols}x{n_rows}), {sub.t.nunique()} timepoint(s), "
                f"{len(pairs):,} pairs"
            )
        for axis, label in AXES:
            seams = seams_x if axis == "x" else seams_y
            on_seam, control = split_by_seam(pairs, seams, axis, args)
            pooled[axis].extend(on_seam, control)
            if args.per_well:
                print_profile(
                    seam_profile(on_seam, control, args), label, args, "    "
                )

    n_wells = int(df.well.nunique())
    print(f"\n{'=' * 78}")
    print(f"POOLED over {n_wells} well(s) — {len(df):,} detections")
    print(f"{'=' * 78}")
    rows: list[dict[str, Any]] = []
    for axis, label in AXES:
        on_seam, control = pooled[axis].pooled()
        profile = seam_profile(on_seam, control, args)
        print_profile(profile, label, args)
        if profile:
            rows.append(
                {
                    "plate_id": args.plate_id,
                    "axis": axis,
                    "n_wells": n_wells,
                    **{
                        k: v
                        for k, v in profile.items()
                        if not isinstance(v, np.ndarray)
                    },
                    "verdict": verdict(profile, args),
                }
            )

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nWrote {len(rows)} profiles to {args.csv}")


if __name__ == "__main__":
    main()
