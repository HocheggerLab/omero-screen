#!/usr/bin/env python3
"""Quantify and visualise cyclic-IF plate alignment performance.

Reads the alignment and aggregation outputs that
:mod:`omero_screen.plate_aggregation` attaches to the master plate
(``alignment.csv`` / ``sample_alignment.csv`` and ``agg_data.csv``) and
produces quality-control figures and a summary table:

    1. Applied shift magnitude per field (how much each image moved).
    2. Per-well agreement (consistency of the independent field shifts).
    3. Matched-nucleus centroid residual, before vs after alignment.
    4. Object match rate per repeat plate.
    5. (optional) Nuclear-mask footprint overlap, before vs after alignment.

Everything for a run is collected under ``--outdir``: per parental plate a
``plate_<id>/`` folder holding the raw OMERO input CSVs (``raw/``), the derived
metric tables (``tables/``), a summary table and the figures (PDF + PNG); and,
when several plates are given as biological replicates, a ``combined/`` folder
with the pooled tables, replicate-level statistics and superplots. Figures are
also uploaded to the master plate with ``--attach``. There is no ground truth
for registration, so accuracy is shown as the improvement the shift produces on
real objects.

Main Functions:
    - main: Collects arguments, connects to OMERO, and runs the QC report.
"""

import argparse
import os
from typing import Any


def main() -> None:
    """Run the cyclic-IF alignment quality-control report."""
    parser = argparse.ArgumentParser(
        description="Quality-control report for cyclic-IF plate alignment."
    )
    parser.add_argument(
        "ID",
        type=int,
        nargs="+",
        help="OMERO master (parental) plate IDs. Pass several to treat them as "
        "biological replicates and add a combined, replicate-level analysis.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="alignment_qc",
        help="Destination directory collecting everything for the run — the raw "
        "input CSVs, derived metric tables, summary/stats tables and figures "
        "(created if absent; ~ and relative paths are expanded) "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--sample-alignments",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use per-sample alignments for residual reconstruction; must match "
        "how aggregate_plates was run (default: %(default)s)",
    )
    parser.add_argument(
        "--iqr",
        type=float,
        default=1.5,
        help="IQR factor for per-well agreement outlier removal (default: %(default)s)",
    )
    parser.add_argument(
        "--mask-samples",
        type=int,
        default=12,
        help="Number of fields to sample for nuclear-mask overlap (0 to skip; "
        "requires downloading masks) (default: %(default)s)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in micrometres (auto-detected from OMERO if omitted)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for mask-overlap field sampling (default is random)",
    )
    parser.add_argument(
        "--attach",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Attach the figures to the master plate (default: %(default)s)",
    )
    group = parser.add_argument_group("Omero Screen overrides")
    group.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (requires configuration file .env.{name}).",
    )
    args = parser.parse_args()

    if args.env:
        os.environ["ENV"] = args.env

    # Lazy imports so argument errors surface quickly.
    from pathlib import Path

    import pandas as pd
    from loguru import logger
    from omero.gateway import BlitzGateway
    from omero_utils.attachments import (
        attach_figure,
        delete_file_attachment,
        get_file_attachments,
        parse_csv_data,
    )
    from omero_utils.omero_connect import omero_connect

    from omero_screen import alignment_qc as qc
    from omero_screen.plate_aggregation import (
        _get_mask_from_map,
        _get_mask_map,
        _get_well_images,
    )

    def _load_csv(plate: object, filename: str, plate_id: int) -> pd.DataFrame:
        att = get_file_attachments(plate, filename)
        if not att:
            raise SystemExit(
                f"ERROR: master plate {plate_id} is missing '{filename}'. "
                "Run align_plates / aggregate_plates first."
            )
        df = parse_csv_data(att[0])
        if df is None:
            raise SystemExit(f"ERROR: could not parse '{filename}'.")
        return df

    def _pixel_size(conn: BlitzGateway, plate_id: int) -> float | None:
        if args.pixel_size:
            return float(args.pixel_size)
        try:
            from omero_screen.metadata_parser import MetadataParser

            meta = MetadataParser(conn, plate_id)
            meta.manage_metadata()
            return float(meta.pixel_size) or None
        except Exception as exc:  # noqa: BLE001 - QC should degrade gracefully
            logger.warning(f"Could not determine pixel size: {exc}")
            return None

    def _mask_overlap(
        conn: BlitzGateway, plate_id: int, alignment_df: "pd.DataFrame"
    ) -> "pd.DataFrame":
        """Sample fields and compute nuclear-mask overlap before/after alignment."""
        import random

        plate_ids = list(pd.unique(alignment_df["plate"]))
        per_sample = "image_id" in alignment_df.columns
        images1 = _get_well_images(conn, plate_id)
        map1 = _get_mask_map(conn, plate_id)

        if args.seed is None:
            args.seed = int.from_bytes(os.urandom(8))
        random.seed(args.seed)

        records: list[dict[str, Any]] = []
        for i, plate_other in enumerate(plate_ids):
            images2 = _get_well_images(conn, plate_other)
            map2 = _get_mask_map(conn, plate_other)
            idx_pool = list(range(len(images1)))
            random.shuffle(idx_pool)
            taken = 0
            for j in idx_pool:
                if taken >= args.mask_samples:
                    break
                well = images1[j][0]
                id1, id2 = images1[j][1], images2[j][1]
                if id1 not in map1 or id2 not in map2:
                    continue
                if per_sample:
                    a = alignment_df[
                        (alignment_df["plate"] == plate_other)
                        & (alignment_df["well"] == well)
                        & (alignment_df["image_id"] == id2)
                    ]
                else:
                    a = alignment_df[
                        (alignment_df["plate"] == plate_other)
                        & (alignment_df["well"] == well)
                    ]
                if a.empty:
                    continue
                shift = (float(a.iloc[0]["x"]), float(a.iloc[0]["y"]))
                # Masks are retrieved transposed (X/Y swapped) relative to the
                # intensity-image frame in which the alignment shift (x, y) is
                # defined — verified empirically: only the swapped-axis shift
                # restores nuclear overlap on large-shift rounds. Transpose both
                # into the shift frame so binary_overlap's (x, y) applies to the
                # correct axes. (Upstream: _get_mask_from_map orientation.)
                m1 = _get_mask_from_map(conn, id1, map1).T
                m2 = _get_mask_from_map(conn, id2, map2).T
                scores = qc.binary_overlap(m1, m2, shift)
                record: dict[str, Any] = dict(scores)
                record.update(
                    {"repeat": i, "plate": plate_other, "well": well}
                )
                records.append(record)
                taken += 1
            logger.info(
                f"Computed mask overlap for {taken} fields of plate {plate_other}"
            )
        return pd.DataFrame.from_records(records)

    def _write_tables(tables: dict[str, "pd.DataFrame"], dest: "Path") -> None:
        """Write a set of named DataFrames as CSV files into ``dest``."""
        dest.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            if df is not None and not df.empty:
                df.to_csv(dest / f"{name}.csv", index=False)

    def _save_figures(
        figures: dict[str, Any],
        dest: "Path",
        plate: object | None,
        conn: BlitzGateway,
    ) -> None:
        """Write figures as PDF+PNG and optionally attach the PNG to a plate."""
        dest.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            for ext in ("pdf", "png"):
                fig.savefig(
                    dest / f"{name}.{ext}", dpi=200, bbox_inches="tight"
                )
            logger.info(f"Saved figure: {dest.name}/{name}")
            if args.attach and plate is not None:
                att_name = f"alignment_qc_{name}"
                delete_file_attachment(
                    conn, plate, ends_with=att_name + ".png"
                )
                attach_figure(conn, fig, plate, att_name)

    def _process_master(
        conn: BlitzGateway, plate_id: int, outdir: "Path"
    ) -> dict[str, "pd.DataFrame"]:
        """Compute and save single-plate metrics; return them tagged by master."""
        plate = conn.getObject("Plate", plate_id)
        if plate is None:
            raise SystemExit(f"ERROR: plate {plate_id} not found.")
        logger.info(f"Processing parental plate {plate_id}")

        alignment_file = (
            "sample_alignment.csv"
            if args.sample_alignments
            else "alignment.csv"
        )
        sample_df = _load_csv(plate, "sample_alignment.csv", plate_id)
        alignment_df = _load_csv(plate, alignment_file, plate_id)
        agg_df = _load_csv(plate, "agg_data.csv", plate_id)

        pixel_size = _pixel_size(conn, plate_id)
        logger.info(f"Pixel size: {pixel_size} um")

        # Map each repeat plate to its staining-round index so every metric
        # shares a common "repeat" grouping across biological replicates.
        round_of = {
            p: i for i, p in enumerate(pd.unique(alignment_df["plate"]))
        }
        shift_df = qc.shift_summary(sample_df, pixel_size_um=pixel_size)
        shift_df["repeat"] = shift_df["plate"].map(round_of)
        agreement_df = qc.per_well_agreement(
            sample_df, iqr=args.iqr, pixel_size_um=pixel_size
        )
        agreement_df["repeat"] = agreement_df["plate"].map(round_of)
        residual_df = qc.matched_residuals(
            agg_df, alignment_df, pixel_size_um=pixel_size
        )
        match_df = qc.match_rate(agg_df)
        overlap_df = (
            _mask_overlap(conn, plate_id, alignment_df)
            if args.mask_samples > 0
            else pd.DataFrame()
        )

        summary = qc.summarise(
            shift_df,
            agreement_df,
            residual_df,
            match_df,
            overlap_df if not overlap_df.empty else None,
        )
        dest = outdir / f"plate_{plate_id}"
        dest.mkdir(parents=True, exist_ok=True)
        summary.to_csv(dest / "alignment_qc_summary.csv", index=False)
        # Persist the raw OMERO inputs and every derived metric table so the
        # run directory is self-contained.
        _write_tables(
            {
                alignment_file.removesuffix(".csv"): alignment_df,
                "sample_alignment": sample_df,
                "agg_data": agg_df,
            },
            dest / "raw",
        )
        _write_tables(
            {
                "shift": shift_df,
                "well_agreement": agreement_df,
                "matched_residuals": residual_df,
                "match_rate": match_df,
                "mask_overlap": overlap_df,
            },
            dest / "tables",
        )
        print(f"\n=== parental plate {plate_id} ===")
        print(summary.to_string(index=False))

        figures = {
            "shift_magnitude": qc.plot_shift_distribution(
                shift_df, pixel_size
            ),
            "shift_vectorfield": qc.plot_shift_vectorfield(alignment_df),
            "well_agreement": qc.plot_agreement(agreement_df, pixel_size),
            "match_rate": qc.plot_match_rate(match_df),
        }
        if not residual_df.empty:
            figures["centroid_residual"] = qc.plot_residual_before_after(
                residual_df, pixel_size
            )
        if not overlap_df.empty:
            figures["mask_overlap"] = qc.plot_mask_overlap(overlap_df)
        _save_figures(figures, dest, plate, conn)

        # Tag every table with the parental plate (biological replicate).
        out = {
            "shift": shift_df,
            "agreement": agreement_df,
            "residual": residual_df,
            "match": match_df,
            "overlap": overlap_df,
            "summary": summary,
        }
        for df in out.values():
            df["master"] = plate_id
        out["_pixel_size"] = pd.DataFrame({"pixel_size": [pixel_size]})
        return out

    def _combined_analysis(
        conn: BlitzGateway,
        per_master: list[dict[str, "pd.DataFrame"]],
        outdir: "Path",
    ) -> None:
        """Pool metrics across parental plates and add replicate-level stats."""
        dest = outdir / "combined"
        dest.mkdir(parents=True, exist_ok=True)

        def _pool(key: str) -> "pd.DataFrame":
            frames = [m[key] for m in per_master if not m[key].empty]
            return (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame()
            )

        shift = _pool("shift")
        agreement = _pool("agreement")
        residual = _pool("residual")
        overlap = _pool("overlap")
        summary = _pool("summary")

        pixel_sizes = [
            float(m["_pixel_size"]["pixel_size"].iloc[0])
            for m in per_master
            if m["_pixel_size"]["pixel_size"].iloc[0]
        ]
        px = pixel_sizes[0] if pixel_sizes else None

        # Replicate-level stats: median per plate, then mean +/- SD across the
        # n parental plates. Cell counts are NOT treated as independent.
        stats_rows = []
        specs = [
            (shift, "magnitude_um" if px else "magnitude_px", "shift"),
            (
                agreement,
                "rms_residual_um" if px else "rms_residual_px",
                "well_agreement",
            ),
            (
                residual,
                "residual_after_um" if px else "residual_after_px",
                "residual_after",
            ),
            (
                residual,
                "residual_before_um" if px else "residual_before_px",
                "residual_before",
            ),
            (overlap, "dice_after", "mask_dice_after"),
        ]
        for df, col, label in specs:
            if df.empty or col not in df.columns:
                continue
            st = qc.across_replicate_stats(
                df, col, master_col="master", by="repeat"
            )
            st.insert(0, "metric", label)
            stats_rows.append(st)
        if stats_rows:
            stats = pd.concat(stats_rows, ignore_index=True)
            stats.to_csv(dest / "replicate_stats.csv", index=False)
            print("\n=== combined replicate-level statistics ===")
            print(stats.to_string(index=False))

        summary.to_csv(dest / "per_plate_summary.csv", index=False)
        # Pooled per-object tables (tagged by parental plate) for reanalysis.
        _write_tables(
            {
                "shift": shift,
                "well_agreement": agreement,
                "matched_residuals": residual,
                "mask_overlap": overlap,
            },
            dest / "tables",
        )

        # Superplots: individual objects faint, per-plate medians as points,
        # mean +/- SD across plates as the error bar.
        figures: dict[str, Any] = {}
        if not residual.empty:
            figures["superplot_residual_after"] = qc.plot_superplot(
                residual,
                "residual_after_um" if px else "residual_after_px",
                ylabel=f"Matched-centroid residual ({'µm' if px else 'px'})",
                title="Registration accuracy across biological replicates",
            )
        if not shift.empty:
            figures["superplot_shift"] = qc.plot_superplot(
                shift,
                "magnitude_um" if px else "magnitude_px",
                ylabel=f"Applied shift ({'µm' if px else 'px'})",
                title="Applied shift across biological replicates",
            )
        if not overlap.empty:
            figures["superplot_mask_dice"] = qc.plot_superplot(
                overlap,
                "dice_after",
                ylabel="Nuclear-mask Dice (after)",
                title="Mask overlap across biological replicates",
            )
        if not residual.empty:
            figures["qc_panel"] = qc.plot_qc_panel(
                shift, residual, overlap, pixel_size_um=px
            )
        _save_figures(figures, dest, None, conn)

    @omero_connect
    def run(plate_ids: list[int], conn: BlitzGateway | None = None) -> None:
        assert conn is not None
        if qc.use_lab_style():
            logger.info("Applied lab Matplotlib style (hhlab_style01)")
        else:
            logger.warning("Lab style not found; using Matplotlib default")
        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Collecting output under: {outdir}")

        per_master = [_process_master(conn, pid, outdir) for pid in plate_ids]

        if len(per_master) > 1:
            logger.info(
                f"Combining {len(per_master)} parental plates as replicates"
            )
            _combined_analysis(conn, per_master, outdir)

        print(f"\nAll output written to: {outdir}")

    run(args.ID)


if __name__ == "__main__":
    main()
