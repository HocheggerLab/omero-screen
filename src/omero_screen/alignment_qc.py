"""Quality-control metrics for cyclic-IF plate alignment.

This module quantifies the performance of the plate alignment and aggregation
pipeline (:mod:`omero_screen.plate_aggregation`) from its CSV outputs:

- ``alignment.csv`` / ``sample_alignment.csv`` — the per-well and per-field
  translation shifts computed by :func:`~omero_screen.plate_aggregation.align_plates`.
- ``agg_data.csv`` — the aggregated per-cell table written by
  :func:`~omero_screen.plate_aggregation.aggregate_plates`, holding the master
  and (shift-corrected) repeat centroids side by side.

There is no ground-truth registration, so accuracy is assessed on the
*consequences* of the shift: how far matched nuclei land from one another, and
how well nuclear-mask footprints overlap, each compared before and after the
shift is applied. Three families of metric are provided:

1. **Shift magnitude** — how far each field was moved (:func:`shift_summary`).
2. **Per-well agreement** — how consistent the independent field shifts are
   within a well (:func:`per_well_agreement`); this is the pipeline's own
   precision/repeatability measure.
3. **Registration accuracy** — matched-centroid residuals
   (:func:`matched_residuals`), object match rate (:func:`match_rate`) and
   binary mask overlap (:func:`binary_overlap`), each contrasting the aligned
   state against the unaligned baseline.

All functions here are pure (they operate on DataFrames / arrays) so they can
be unit tested without an OMERO connection. The plotting helpers return
Matplotlib figures for the driver script to save or attach.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

# Column-name helpers -------------------------------------------------------

_REPEAT_RE = re.compile(r"^centroid-1\.(\d+)$")
_WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def repeat_indices(agg_df: pd.DataFrame) -> list[int]:
    """Return the repeat-plate indices present in an aggregated data frame.

    Repeat plates contribute suffixed columns ``centroid-1.0``,
    ``centroid-1.1``, ... The integer suffixes are returned in ascending order.

    Args:
        agg_df: Aggregated per-cell table (``agg_data.csv``).

    Returns:
        Sorted list of repeat indices, e.g. ``[0, 1]`` for two repeat plates.
    """
    idx = []
    for col in agg_df.columns:
        m = _REPEAT_RE.match(str(col))
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def well_to_grid(well: str) -> tuple[int, int]:
    """Convert a well label such as ``"B3"`` to zero-based ``(row, column)``.

    Row letters are interpreted as base-26 (``A`` -> 0, ``Z`` -> 25,
    ``AA`` -> 26) to support 96- and 384-well plates and beyond.

    Args:
        well: Well label, e.g. ``"A1"`` or ``"H12"``.

    Returns:
        Zero-based ``(row, column)`` indices.

    Raises:
        ValueError: If the label does not match ``<letters><digits>``.
    """
    m = _WELL_RE.match(str(well))
    if not m:
        raise ValueError(f"Unrecognised well label: {well!r}")
    letters, digits = m.groups()
    row = 0
    for ch in letters.upper():
        row = row * 26 + (ord(ch) - ord("A") + 1)
    return row - 1, int(digits) - 1


# Metric 1: shift magnitude -------------------------------------------------


def shift_summary(
    sample_df: pd.DataFrame,
    pixel_size_um: float | None = None,
    drop_zero: bool = True,
) -> pd.DataFrame:
    """Per-field shift magnitudes from the sample alignment table.

    Args:
        sample_df: ``sample_alignment.csv`` with columns
            ``plate, well, sample, image_id, x, y``.
        pixel_size_um: Physical pixel size in micrometres. When provided a
            ``magnitude_um`` column is added.
        drop_zero: Drop fields with an exactly ``(0, 0)`` shift. These are the
            blank frames that :func:`align_plates` records as zero, and would
            otherwise deflate the magnitude and agreement statistics.

    Returns:
        A copy of the input with ``dx``, ``dy`` and ``magnitude_px`` (and
        optionally ``magnitude_um``) columns.
    """
    df = sample_df.copy()
    df["dx"] = df["x"].astype(float)
    df["dy"] = df["y"].astype(float)
    df["magnitude_px"] = np.hypot(df["dx"], df["dy"])
    if drop_zero:
        df = df[(df["dx"] != 0) | (df["dy"] != 0)].reset_index(drop=True)
    if pixel_size_um:
        df["magnitude_um"] = df["magnitude_px"] * pixel_size_um
    return df


# Metric 2: per-well agreement ---------------------------------------------


def _iqr_keep(
    magnitudes: npt.NDArray[Any], iqr: float
) -> npt.NDArray[np.bool_]:
    """Boolean mask of values kept by the IQR outlier rule used in align_plates."""
    if iqr <= 0 or len(magnitudes) < 2:
        return np.ones(len(magnitudes), dtype=bool)
    q1, q3 = np.quantile(magnitudes, [0.25, 0.75])
    upper = q3 + iqr * (q3 - q1)
    return np.asarray(magnitudes <= upper, dtype=bool)


def per_well_agreement(
    sample_df: pd.DataFrame,
    iqr: float = 1.5,
    drop_zero: bool = True,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Consistency of per-field shifts within each well.

    Mirrors the tolerance check inside
    :func:`~omero_screen.plate_aggregation.align_plates`: outliers are removed
    by the same IQR rule, the retained fields are averaged, and the spread of
    those fields about the mean is reported. A small spread means the
    independent fields agree on the same shift (high registration precision).

    Args:
        sample_df: ``sample_alignment.csv`` table.
        iqr: Inter-quartile-range factor for outlier removal (0 disables it),
            matching the ``--iqr`` alignment option.
        drop_zero: Drop blank ``(0, 0)`` frames before analysis.
        pixel_size_um: Physical pixel size; adds ``rms_residual_um`` when given.

    Returns:
        One row per ``(plate, well)`` with the field count, mean shift,
        per-axis standard deviation, and the root-mean-square residual distance
        of the retained fields to the well mean (``rms_residual_px``).
    """
    df = sample_df.copy()
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    if drop_zero:
        df = df[(df["x"] != 0) | (df["y"] != 0)]

    records = []
    for (plate, well), grp in df.groupby(["plate", "well"], sort=True):
        shifts = grp[["x", "y"]].to_numpy(dtype=float)
        magnitudes = np.hypot(shifts[:, 0], shifts[:, 1])
        keep = _iqr_keep(magnitudes, iqr)
        shifts = shifts[keep]
        if len(shifts) == 0:
            continue
        mean = shifts.mean(axis=0)
        residuals = np.hypot(shifts[:, 0] - mean[0], shifts[:, 1] - mean[1])
        rms = float(np.sqrt(np.mean(residuals**2)))
        rec: dict[str, Any] = {
            "plate": plate,
            "well": well,
            "n_fields": int(len(shifts)),
            "mean_x": float(mean[0]),
            "mean_y": float(mean[1]),
            "std_x": float(shifts[:, 0].std()),
            "std_y": float(shifts[:, 1].std()),
            "rms_residual_px": rms,
            "max_residual_px": float(residuals.max()),
        }
        if pixel_size_um:
            rec["rms_residual_um"] = rms * pixel_size_um
        records.append(rec)
    return pd.DataFrame.from_records(records)


# Metric 3a: matched-centroid residuals ------------------------------------


def matched_residuals(
    agg_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Residual distance between matched master and repeat centroids.

    For every matched cell (a row where both the master and a repeat centroid
    are present) this computes the Euclidean distance between the two
    centroids *after* the shift was applied, and reconstructs the distance that
    would have been seen *before* alignment by adding the stored shift back.

    The aggregation subtracts the shift from repeat centroids
    (``aligned = original - shift``), so with ``after = master - aligned`` the
    unaligned residual is ``before = after - shift``.

    Under the default mask-overlap matching this residual is an *independent*
    accuracy measure (matching does not use centroid distance); under the
    centroid-distance methods it is the matching criterion and is therefore
    somewhat optimistic.

    Args:
        agg_df: Aggregated per-cell table (``agg_data.csv``).
        alignment_df: The alignment table that was used for aggregation —
            per-well (``plate, well, x, y``) or per-sample
            (``plate, well, image_id, x, y``). Per-sample is detected by the
            presence of an ``image_id`` column, matching ``aggregate_plates``.
        pixel_size_um: Physical pixel size; adds micrometre columns when given.

    Returns:
        Long-form frame with one row per matched cell:
        ``repeat, plate, well, residual_before_px, residual_after_px``
        (plus ``*_um`` columns when a pixel size is supplied).
    """
    per_sample = "image_id" in alignment_df.columns
    plate_ids = list(pd.unique(alignment_df["plate"]))

    frames = []
    for i, plate in enumerate(plate_ids):
        cx2, cy2 = f"centroid-1.{i}", f"centroid-0.{i}"
        if cx2 not in agg_df.columns or cy2 not in agg_df.columns:
            continue
        cols = ["well", "centroid-1", "centroid-0", cx2, cy2]
        if per_sample:
            cols.append(f"image_id.{i}")
        sub = agg_df[cols].dropna(
            subset=["centroid-1", "centroid-0", cx2, cy2]
        )
        if sub.empty:
            continue
        sub = sub.copy()
        after_dx = sub["centroid-1"].to_numpy(float) - sub[cx2].to_numpy(float)
        after_dy = sub["centroid-0"].to_numpy(float) - sub[cy2].to_numpy(float)

        align = alignment_df[alignment_df["plate"] == plate]
        if per_sample:
            shift_map = {
                (r.well, r.image_id): (float(r.x), float(r.y))
                for r in align.itertuples()
            }
            keys = list(
                zip(
                    sub["well"].tolist(),
                    sub[f"image_id.{i}"].tolist(),
                    strict=True,
                )
            )
            sx = np.array(
                [shift_map.get(k, (np.nan, np.nan))[0] for k in keys]
            )
            sy = np.array(
                [shift_map.get(k, (np.nan, np.nan))[1] for k in keys]
            )
        else:
            shift_map = {
                r.well: (float(r.x), float(r.y)) for r in align.itertuples()
            }
            sx = np.array(
                [shift_map.get(w, (np.nan, np.nan))[0] for w in sub["well"]]
            )
            sy = np.array(
                [shift_map.get(w, (np.nan, np.nan))[1] for w in sub["well"]]
            )

        before_dx = after_dx - sx
        before_dy = after_dy - sy
        out = pd.DataFrame(
            {
                "repeat": i,
                "plate": plate,
                "well": sub["well"].to_numpy(),
                "residual_after_px": np.hypot(after_dx, after_dy),
                "residual_before_px": np.hypot(before_dx, before_dy),
            }
        )
        frames.append(out)

    if not frames:
        return pd.DataFrame(
            columns=[
                "repeat",
                "plate",
                "well",
                "residual_after_px",
                "residual_before_px",
            ]
        )
    result = pd.concat(frames, ignore_index=True)
    if pixel_size_um:
        result["residual_after_um"] = (
            result["residual_after_px"] * pixel_size_um
        )
        result["residual_before_um"] = (
            result["residual_before_px"] * pixel_size_um
        )
    return result


# Metric 3b: object match rate ---------------------------------------------


def match_rate(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of objects matched between master and each repeat plate.

    Args:
        agg_df: Aggregated per-cell table (``agg_data.csv``).

    Returns:
        One row per repeat: ``repeat, n_master, n_repeat, n_matched,
        match_fraction`` where ``match_fraction`` is matched / min(master,
        repeat) — the fraction of the smaller population that was paired.
    """
    master_present = (
        agg_df["centroid-1"].notna() & agg_df["centroid-0"].notna()
    )
    n_master = int(master_present.sum())
    records = []
    for i in repeat_indices(agg_df):
        cx2, cy2 = f"centroid-1.{i}", f"centroid-0.{i}"
        repeat_present = agg_df[cx2].notna() & agg_df[cy2].notna()
        n_repeat = int(repeat_present.sum())
        n_matched = int((master_present & repeat_present).sum())
        denom = min(n_master, n_repeat)
        records.append(
            {
                "repeat": i,
                "n_master": n_master,
                "n_repeat": n_repeat,
                "n_matched": n_matched,
                "match_fraction": (n_matched / denom) if denom else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


# Metric 3c: binary mask overlap -------------------------------------------


def _shift_binary(
    mask: npt.NDArray[Any], shift_yx: tuple[float, float]
) -> npt.NDArray[Any]:
    """Translate a 2D array by a ``(dy, dx)`` shift (rounded), zero-filling edges."""
    dy, dx = int(round(shift_yx[0])), int(round(shift_yx[1]))
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    sys, sye = max(0, -dy), min(h, h - dy)
    sxs, sxe = max(0, -dx), min(w, w - dx)
    out[ys:ye, xs:xe] = mask[sys:sye, sxs:sxe]
    return out


def binary_overlap(
    mask1: npt.NDArray[Any],
    mask2: npt.NDArray[Any],
    shift_xy: tuple[float, float],
) -> dict[str, float]:
    """Foreground overlap of two nuclear masks, before and after alignment.

    Binarises both masks (foreground = label > 0) and measures how much the
    footprints coincide. The repeat mask (``mask2``) is translated into the
    master frame by the *negative* of the stored shift (the shift maps master
    to repeat), matching the convention in ``create_cell_masks``.

    Args:
        mask1: Master nuclear mask (labelled or binary), 2D.
        mask2: Repeat nuclear mask, same shape as ``mask1``.
        shift_xy: Stored alignment shift ``(x, y)`` for this field.

    Returns:
        ``dice_before``, ``dice_after``, ``iou_before`` and ``iou_after``.
    """
    a = mask1 > 0
    b_raw = mask2 > 0
    x, y = shift_xy
    b_aligned = _shift_binary(b_raw, (-y, -x)) > 0

    def _scores(
        fixed: npt.NDArray[Any], moving: npt.NDArray[Any]
    ) -> tuple[float, float]:
        inter = float(np.logical_and(fixed, moving).sum())
        union = float(np.logical_or(fixed, moving).sum())
        size = float(fixed.sum() + moving.sum())
        dice = (2 * inter / size) if size else np.nan
        iou = (inter / union) if union else np.nan
        return dice, iou

    dice_before, iou_before = _scores(a, b_raw)
    dice_after, iou_after = _scores(a, b_aligned)
    return {
        "dice_before": dice_before,
        "dice_after": dice_after,
        "iou_before": iou_before,
        "iou_after": iou_after,
    }


# Summary -------------------------------------------------------------------


def summarise(
    shift_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    match_df: pd.DataFrame,
    overlap_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collapse the per-object metrics into one row per repeat plate.

    Args:
        shift_df: Output of :func:`shift_summary`.
        agreement_df: Output of :func:`per_well_agreement`.
        residual_df: Output of :func:`matched_residuals`.
        match_df: Output of :func:`match_rate`.
        overlap_df: Optional per-field frame with ``repeat``, ``dice_after``
            and ``dice_before`` columns (built by the driver from
            :func:`binary_overlap`).

    Returns:
        Tidy summary table keyed by ``repeat``/``plate`` with median shift,
        median per-well agreement, median matched residual before/after, match
        fraction and (when available) median mask Dice before/after.
    """
    # plate lookup per repeat from residuals
    plate_by_repeat = (
        residual_df.groupby("repeat")["plate"].first().to_dict()
        if not residual_df.empty
        else {}
    )

    rows = []
    for _, m in match_df.iterrows():
        i = int(m["repeat"])
        plate = plate_by_repeat.get(i)
        res_i = residual_df[residual_df["repeat"] == i]
        # Shift/agreement are per-plate; align by plate id where possible.
        sh = (
            shift_df[shift_df["plate"] == plate]
            if plate is not None and "plate" in shift_df
            else shift_df
        )
        ag = (
            agreement_df[agreement_df["plate"] == plate]
            if plate is not None and "plate" in agreement_df
            else agreement_df
        )
        row: dict[str, Any] = {
            "repeat": i,
            "plate": plate,
            "median_shift_px": float(sh["magnitude_px"].median())
            if not sh.empty
            else np.nan,
            "median_well_rms_px": float(ag["rms_residual_px"].median())
            if not ag.empty
            else np.nan,
            "median_residual_before_px": float(
                res_i["residual_before_px"].median()
            )
            if not res_i.empty
            else np.nan,
            "median_residual_after_px": float(
                res_i["residual_after_px"].median()
            )
            if not res_i.empty
            else np.nan,
            "match_fraction": float(m["match_fraction"]),
        }
        if overlap_df is not None and not overlap_df.empty:
            ov = overlap_df[overlap_df["repeat"] == i]
            if not ov.empty:
                row["median_dice_before"] = float(ov["dice_before"].median())
                row["median_dice_after"] = float(ov["dice_after"].median())
        rows.append(row)
    return pd.DataFrame.from_records(rows)


# Plotting ------------------------------------------------------------------


def use_lab_style() -> bool:
    """Apply the shared lab Matplotlib style (Hoechegger lab palette + fonts).

    Locates ``hhlab_style01.mplstyle`` shipped with the ``omero-screen-plots``
    package and applies it globally, so the QC figures match the rest of the
    lab's publication figures (Arial, clean spines, the standard colour cycle).
    The plotting functions here read colours from the active style, so this is
    the single switch for consistent styling.

    Returns:
        True if the lab style was found and applied, False otherwise (callers
        can fall back to the Matplotlib default).
    """
    import importlib.util
    from pathlib import Path

    import matplotlib.pyplot as plt

    spec = importlib.util.find_spec("omero_screen_plots")
    if spec and spec.origin:
        style = Path(spec.origin).parents[2] / "hhlab_style01.mplstyle"
        if style.exists():
            plt.style.use(str(style))
            return True
    return False


def _cycle_colours(n: int) -> Any:
    """Return ``n`` colours from the active style's cycle (fallback: tab10)."""
    import matplotlib as mpl

    cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(cycle) >= n:
        return list(cycle[:n])
    return mpl.colormaps["tab10"](np.linspace(0, 1, max(n, 1)))


def _new_fig(
    nrows: int = 1, ncols: int = 1, size: tuple[float, float] = (6, 4)
) -> tuple[Figure, Any]:
    fig = Figure(figsize=(size[0] * ncols, size[1] * nrows))
    axs = fig.subplots(nrows, ncols, squeeze=False)
    return fig, axs


def plot_shift_distribution(
    shift_df: pd.DataFrame, pixel_size_um: float | None = None
) -> Figure:
    """Histogram of per-field shift magnitudes, split by repeat plate."""
    fig, axs = _new_fig()
    ax = axs[0][0]
    col = (
        "magnitude_um"
        if pixel_size_um and "magnitude_um" in shift_df
        else "magnitude_px"
    )
    unit = "µm" if col == "magnitude_um" else "px"
    for plate, grp in shift_df.groupby("plate"):
        ax.hist(grp[col], bins=30, alpha=0.5, label=f"plate {plate}")
    ax.set_xlabel(f"Field shift magnitude ({unit})")
    ax.set_ylabel("Fields")
    ax.set_title("Applied alignment shift per field")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_shift_vectorfield(alignment_df: pd.DataFrame) -> Figure:
    """Quiver of per-well mean shift laid out on the plate grid, one panel per plate."""
    plates = list(pd.unique(alignment_df["plate"]))
    fig, axs = _new_fig(1, len(plates), size=(5, 4))
    for ax, plate in zip(axs[0], plates, strict=True):
        grp = alignment_df[alignment_df["plate"] == plate]
        rows, cols, us, vs = [], [], [], []
        for r in grp.itertuples():
            row, col = well_to_grid(str(r.well))
            rows.append(row)
            cols.append(col)
            us.append(float(r.x))
            vs.append(float(r.y))
        # invert y so row A is at the top, like a physical plate
        ax.quiver(
            cols,
            [-r for r in rows],
            us,
            [-v for v in vs],
            angles="xy",
            scale_units="xy",
        )
        ax.set_title(f"plate {plate} shift field")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_agreement(
    agreement_df: pd.DataFrame, pixel_size_um: float | None = None
) -> Figure:
    """Distribution of per-well shift dispersion (RMS residual to the well mean)."""
    fig, axs = _new_fig()
    ax = axs[0][0]
    col = (
        "rms_residual_um"
        if pixel_size_um and "rms_residual_um" in agreement_df
        else "rms_residual_px"
    )
    unit = "µm" if col.endswith("um") else "px"
    for plate, grp in agreement_df.groupby("plate"):
        ax.hist(grp[col], bins=20, alpha=0.5, label=f"plate {plate}")
    ax.set_xlabel(f"Per-well field dispersion, RMS to mean ({unit})")
    ax.set_ylabel("Wells")
    ax.set_title("Alignment agreement within wells")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residual_before_after(
    residual_df: pd.DataFrame, pixel_size_um: float | None = None
) -> Figure:
    """Overlaid histograms of matched-centroid residuals before vs after alignment."""
    fig, axs = _new_fig()
    ax = axs[0][0]
    use_um = bool(pixel_size_um and "residual_after_um" in residual_df)
    before = residual_df[
        "residual_before_um" if use_um else "residual_before_px"
    ]
    after = residual_df["residual_after_um" if use_um else "residual_after_px"]
    unit = "µm" if use_um else "px"
    bins = np.histogram_bin_edges(
        np.concatenate([before.to_numpy(), after.to_numpy()]), bins=40
    )
    ax.hist(before, bins=bins, alpha=0.5, label="before (no shift)")
    ax.hist(after, bins=bins, alpha=0.5, label="after alignment")
    ax.axvline(
        float(after.median()),
        color="k",
        ls="--",
        lw=1,
        label=f"after median {after.median():.1f} {unit}",
    )
    ax.set_xlabel(f"Matched-centroid residual ({unit})")
    ax.set_ylabel("Matched cells")
    ax.set_title("Registration accuracy: matched-nucleus residual")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_match_rate(match_df: pd.DataFrame) -> Figure:
    """Bar chart of the object match fraction per repeat plate."""
    fig, axs = _new_fig()
    ax = axs[0][0]
    labels = [f"plate {int(r)}" for r in match_df["repeat"]]
    ax.bar(labels, match_df["match_fraction"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Matched fraction (of smaller population)")
    ax.set_title("Object match rate")
    fig.tight_layout()
    return fig


def plot_mask_overlap(overlap_df: pd.DataFrame) -> Figure:
    """Overlaid histograms of nuclear-mask Dice overlap before vs after alignment."""
    fig, axs = _new_fig()
    ax = axs[0][0]
    bins = np.linspace(0, 1, 41)
    ax.hist(overlap_df["dice_before"], bins=bins, alpha=0.5, label="before")
    ax.hist(overlap_df["dice_after"], bins=bins, alpha=0.5, label="after")
    ax.set_xlabel("Nuclear-mask Dice overlap")
    ax.set_ylabel("Fields")
    ax.set_title("Registration accuracy: mask footprint overlap")
    ax.legend()
    fig.tight_layout()
    return fig


# Biological replicates -----------------------------------------------------


def across_replicate_stats(
    long_df: pd.DataFrame,
    value_col: str,
    master_col: str = "master",
    by: str | None = "repeat",
) -> pd.DataFrame:
    """Summarise a per-object metric across biological replicates.

    Each parental (master) plate is one biological replicate. To avoid
    pseudoreplication — treating the thousands of cells within a plate as
    independent — the metric is first reduced to a single value per replicate
    (its median), and the descriptive statistics are then computed *across*
    those replicate-level values (``n`` = number of replicates).

    Args:
        long_df: Per-object metric table carrying a replicate column and the
            value column, e.g. the output of :func:`matched_residuals` with a
            ``master`` column added per parental plate.
        value_col: Column to summarise, e.g. ``"residual_after_px"``.
        master_col: Column identifying the biological replicate (parental plate).
        by: Optional grouping column evaluated separately (e.g. ``"repeat"`` for
            the staining round). Use ``None`` to pool all groups together.

    Returns:
        One row per group with ``n_replicates``, ``mean_of_medians``, ``sd``,
        ``sem``, ``cv_percent``, ``min`` and ``max`` of the replicate medians,
        plus a ``replicate_medians`` list for transparency.
    """
    key_cols = ([by] if by else []) + [master_col]
    per_rep = (
        long_df.groupby(key_cols, sort=True)[value_col].median().reset_index()
    )

    records = []
    group_iter = per_rep.groupby(by, sort=True) if by else [(None, per_rep)]
    for group_key, grp in group_iter:
        v = grp[value_col].to_numpy(dtype=float)
        n = len(v)
        sd = float(v.std(ddof=1)) if n > 1 else np.nan
        mean = float(v.mean()) if n else np.nan
        rec: dict[str, Any] = {}
        if by:
            rec[by] = group_key
        rec.update(
            {
                "n_replicates": n,
                "mean_of_medians": mean,
                "sd": sd,
                "sem": (sd / np.sqrt(n)) if n > 1 else np.nan,
                "cv_percent": (100 * sd / mean)
                if (n > 1 and mean)
                else np.nan,
                "min": float(v.min()) if n else np.nan,
                "max": float(v.max()) if n else np.nan,
                "replicate_medians": [round(float(x), 3) for x in v],
            }
        )
        records.append(rec)
    return pd.DataFrame.from_records(records)


def plot_superplot(
    long_df: pd.DataFrame,
    value_col: str,
    group_col: str = "repeat",
    master_col: str = "master",
    ylabel: str | None = None,
    title: str | None = None,
    ax: Any | None = None,
    ymax: float | None = None,
    max_points: int = 2000,
) -> Figure:
    """Superplot of a per-object metric across biological replicates.

    Following Lord et al. (2020), individual objects are drawn as faint,
    horizontally jittered points coloured by replicate; the median of each
    replicate is overlaid as a large point of the same colour; and a black
    error bar shows the mean +/- SD of the replicate medians (n = replicates).
    This makes both the within-replicate spread and the between-replicate
    agreement visible, and keeps the statistics at the replicate level.

    Args:
        long_df: Per-object metric table with ``group_col``, ``master_col`` and
            ``value_col`` columns.
        value_col: Column to plot on the y-axis.
        group_col: Categorical x-axis (e.g. staining round ``"repeat"``).
        master_col: Biological-replicate identifier (parental plate).
        ylabel: Y-axis label (defaults to ``value_col``).
        title: Plot title.
        ax: Existing axis to draw into (for multi-panel figures). A new figure
            is created when omitted.
        ymax: Optional upper y-limit. The replicate medians and error bars are
            computed on the full data (robust to outliers); this only clips the
            *view* so a few extreme points do not compress the axis. The count
            of off-scale points is annotated.
        max_points: Maximum individual points drawn per replicate per group
            (the scatter is subsampled above this; medians/stats use all data).

    Returns:
        The Matplotlib figure containing the plot.
    """
    groups = sorted(long_df[group_col].dropna().unique())
    replicates = sorted(long_df[master_col].dropna().unique())
    colours = _cycle_colours(len(replicates))
    rng = np.random.default_rng(0)

    if ax is None:
        fig, axs = _new_fig(size=(1.6 * max(len(groups), 1) + 3, 4.5))
        ax = axs[0][0]
    else:
        fig = ax.get_figure()

    offsets = (
        np.linspace(-0.22, 0.22, len(replicates)) if replicates else [0.0]
    )
    for gi, g in enumerate(groups):
        rep_medians = []
        for ri, rep in enumerate(replicates):
            vals = long_df[
                (long_df[group_col] == g) & (long_df[master_col] == rep)
            ][value_col].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            med = float(np.median(vals))
            rep_medians.append(med)
            # The median/stats use all cells; the scatter layer is subsampled so
            # a large replicate (millions of cells) does not saturate the cloud.
            shown = vals
            if len(vals) > max_points:
                shown = rng.choice(vals, size=max_points, replace=False)
            x = gi + offsets[ri] + rng.normal(0, 0.02, len(shown))
            ax.scatter(
                x,
                shown,
                s=6,
                color=colours[ri],
                alpha=0.15,
                edgecolors="none",
                label=f"plate {rep}" if gi == 0 else None,
            )
            ax.scatter(
                gi + offsets[ri],
                med,
                s=90,
                color=colours[ri],
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
        if rep_medians:
            m = float(np.mean(rep_medians))
            sd = (
                float(np.std(rep_medians, ddof=1))
                if len(rep_medians) > 1
                else 0.0
            )
            ax.errorbar(
                gi,
                m,
                yerr=sd,
                fmt="_",
                color="black",
                markersize=22,
                capsize=6,
                zorder=4,
            )

    if ymax is not None:
        arr = long_df[value_col].to_numpy(dtype=float)
        n_off = int((arr > ymax).sum())
        # Set both bounds explicitly so the off-scale outliers do not drag the
        # autoscaled bottom far below the data.
        lo = float(np.nanmin(arr))
        ax.set_ylim(lo - 0.03 * (ymax - lo), ymax)
        if n_off:
            ax.text(
                0.99,
                0.98,
                f"{n_off} pts > {ymax:g} off-scale",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="grey",
            )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"round {int(g)}" for g in groups])
    ax.set_ylabel(ylabel or value_col)
    # Explicit title size: the lab style sets axes.titlesize to 3pt (titles are
    # normally added as manual panel labels), which is unreadable as a header.
    ax.set_title(
        title or f"{value_col} across biological replicates",
        fontsize=9,
        loc="center",
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_qc_panel(
    shift_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    pixel_size_um: float | None = None,
    residual_ymax: float = 4.0,
) -> Figure:
    """Assemble the three-panel cross-replicate QC summary figure.

    Panels: (A) applied shift per round; (B) matched-centroid residual after
    alignment (the accuracy readout, y-capped so a few gross mis-pairings do
    not compress the axis); (C) nuclear-mask Dice after alignment. Each panel is
    a superplot across biological replicates (see :func:`plot_superplot`).

    Args:
        shift_df: Pooled :func:`shift_summary` output (with ``master``/``repeat``).
        residual_df: Pooled :func:`matched_residuals` output.
        overlap_df: Pooled per-field mask overlap (``dice_after`` column).
        pixel_size_um: Pixel size; selects micrometre columns and axis labels.
        residual_ymax: Upper y-limit for the residual panel (data units).

    Returns:
        The composed three-panel figure.
    """
    unit = "µm" if pixel_size_um else "px"
    fig = Figure(figsize=(15, 4.6))
    axs = fig.subplots(1, 3)

    plot_superplot(
        shift_df,
        "magnitude_um" if pixel_size_um else "magnitude_px",
        ax=axs[0],
        ylabel=f"Applied shift ({unit})",
        title="A  Stage displacement corrected",
    )
    plot_superplot(
        residual_df,
        "residual_after_um" if pixel_size_um else "residual_after_px",
        ax=axs[1],
        ymax=residual_ymax,
        ylabel=f"Matched-centroid residual ({unit})",
        title="B  Registration accuracy (after)",
    )
    if not overlap_df.empty:
        plot_superplot(
            overlap_df,
            "dice_after",
            ax=axs[2],
            ylabel="Nuclear-mask Dice (after)",
            title="C  Mask footprint overlap (after)",
        )
        axs[2].set_ylim(0, 1)
    # A single legend is enough for the shared replicate colours.
    for a in axs[1:]:
        legend = a.get_legend()
        if legend is not None:
            legend.set_visible(False)
    fig.tight_layout()
    return fig
