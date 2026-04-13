#!/usr/bin/env python3
"""Run repeated benchmark analyses to measure timing variability.

Runs the full omero-screen pipeline N times on a small plate,
cleaning up all annotations and the cached dataset before each run
so that every stage is freshly timed from a clean plate state.

Usage::

    python benchmarks/benchmark_repeats.py
    python benchmarks/benchmark_repeats.py --plate-id 370 --repeats 5
    python benchmarks/benchmark_repeats.py --gpu
    python benchmarks/benchmark_repeats.py --no-gpu
    python benchmarks/benchmark_repeats.py --cellpose cp4
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path
from statistics import mean, median, stdev

from omero.gateway import (
    BlitzGateway,
    FileAnnotationWrapper,
    MapAnnotationWrapper,
)
from omero_utils.omero_connect import omero_connect

from omero_screen import default_config
from omero_screen.benchmarking import get_benchmark, init_benchmark
from omero_screen.config import project_root
from omero_screen.loops import plate_loop

# CP4 model overrides: use cpsam for both nuclei and cell segmentation
_CP4_MODELS: dict[str, str] = {
    "nuclei": "cp4:cpsam",
    "RPE": "cp4:cpsam",
    "HELA": "cp4:cpsam",
    "U2OS": "cp4:cpsam",
    "HCC1143": "cp4:cpsam",
    "MM231": "cp4:cpsam",
    "PALB": "cp4:cpsam",
}


def _detect_gpu_label() -> str:
    """Detect the compute device label for output folder naming.

    Returns:
        "cpu", "mps", or the CUDA device name with spaces replaced by underscores.
    """
    try:
        import torch

        if torch.cuda.is_available():
            name: str = torch.cuda.get_device_name(0) or "cuda"
            return name.replace(" ", "_")
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "unknown"


def _build_output_dir(cellpose_version: str, use_gpu: bool | None) -> Path:
    """Build an output directory path based on hostname, GPU, cellpose version, and today's date.

    Args:
        cellpose_version: Cellpose version string, e.g. "cp3" or "cp4".
        use_gpu: True = GPU forced, False = CPU forced, None = auto-detect.

    Returns:
        Path of the form ``<project_root>/benchmarks/{hostname}_{device}_{cellpose}_{DDMMYYYY}/``.
    """
    hostname = platform.node().replace(" ", "_")
    device = "cpu" if use_gpu is False else _detect_gpu_label()
    date = time.strftime("%d%m%Y")
    folder_name = f"{hostname}_{device}_{cellpose_version}_{date}"
    return project_root / "benchmarks" / folder_name


def _attach_excel_file(
    conn: BlitzGateway, plate_id: int, excel_path: Path
) -> None:
    """Attach a local Excel file to the OMERO plate."""
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise ValueError(f"Plate {plate_id} not found on OMERO server")

    file_ann = conn.createFileAnnfromLocalFile(
        str(excel_path),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    plate.linkAnnotation(file_ann)


def _delete_plate_metadata(conn: BlitzGateway, plate_id: int) -> None:
    """Delete all annotations from the plate and every well it contains.

    Removes both MapAnnotations (key-value pairs written by the metadata
    parser) and FileAnnotations (e.g. leftover Excel files from a previous
    run that was interrupted before the parser consumed them).

    Args:
        conn: OMERO connection.
        plate_id: Plate ID whose annotations should be wiped.
    """
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        print(
            f"  WARNING: Plate {plate_id} not found — skipping metadata cleanup"
        )
        return

    def _delete_all_annotations(obj: object, label: str) -> None:
        ann_ids: list[int] = []
        for ann in obj.listAnnotations():  # type: ignore[attr-defined]
            if isinstance(ann, MapAnnotationWrapper | FileAnnotationWrapper):
                ann_id = ann.getId()
                if ann_id is not None:
                    ann_ids.append(ann_id)
        if ann_ids:
            conn.deleteObjects(
                "Annotation", ann_ids, deleteAnns=True, wait=True
            )
            print(f"  Deleted {len(ann_ids)} annotation(s) from {label}")

    _delete_all_annotations(plate, f"plate {plate_id}")

    well_count = 0
    for well in plate.listChildren():
        _delete_all_annotations(well, f"well {well.getWellPos()}")
        well_count += 1

    print(f"  Metadata cleanup complete ({well_count} wells checked)")


def _delete_plate_dataset(conn: BlitzGateway, plate_id: int) -> None:
    """Delete the dataset named after the plate ID from the Screens project.

    This removes all cached flatfield correction masks and segmentation
    masks so the next run generates them fresh.

    Args:
        conn: OMERO connection.
        plate_id: Plate ID (dataset name matches this).
    """
    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        print("  WARNING: PROJECT_ID not set, skipping dataset cleanup")
        return

    dataset_name = str(plate_id)
    datasets = list(
        conn.getObjects(
            "Dataset",
            opts={"project": project_id},
            attributes={"name": dataset_name},
        )
    )

    if not datasets:
        print(f"  No dataset '{dataset_name}' found — nothing to delete")
        return

    for dataset in datasets:
        dataset_id = dataset.getId()
        image_ids = [img.getId() for img in dataset.listChildren()]
        if image_ids:
            print(
                f"  Deleting {len(image_ids)} images from dataset {dataset_id}..."
            )
            conn.deleteObjects("Image", image_ids, deleteAnns=True, wait=True)

        print(f"  Deleting dataset '{dataset_name}' (ID: {dataset_id})...")
        conn.deleteObjects("Dataset", [dataset_id], deleteAnns=True, wait=True)

    print("  Dataset cleanup complete")


def _print_summary(report_paths: list[Path], output_dir: Path) -> None:
    """Load all benchmark JSONs and print a cross-run summary table."""
    reports = []
    for p in report_paths:
        with open(p) as f:
            reports.append(json.load(f))

    n = len(reports)
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK SUMMARY — {n} runs, plate {reports[0]['plate_id']}")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 70}\n")

    # Per-run overview
    print(f"{'Run':>4}  {'Total (s)':>10}  {'Img/min':>8}  {'Images':>7}")
    print(f"{'—' * 4}  {'—' * 10}  {'—' * 8}  {'—' * 7}")
    totals = []
    throughputs = []
    for i, r in enumerate(reports, 1):
        total = r["stages"]["total_s"]
        ipm = r["summary"].get("images_per_minute", 0)
        totals.append(total)
        throughputs.append(ipm)
        print(f"{i:>4}  {total:>10.2f}  {ipm:>8.2f}  {r['n_images']:>7}")

    print(f"\n{'Metric':<30}  {'Mean':>8}  {'Median':>8}  {'Std':>8}")
    print(f"{'—' * 30}  {'—' * 8}  {'—' * 8}  {'—' * 8}")
    _print_stat_row("Total time (s)", totals)
    _print_stat_row("Images/minute", throughputs)

    # Per-stage timing across runs
    stage_names = set()
    for r in reports:
        stage_names.update(r["stages"].keys())
    stage_names.discard("total_s")

    if stage_names:
        print(
            f"\n{'Stage Timings (s)':<30}  {'Mean':>8}  {'Median':>8}  {'Std':>8}"
        )
        print(f"{'—' * 30}  {'—' * 8}  {'—' * 8}  {'—' * 8}")
        for stage in sorted(stage_names):
            values = [
                r["stages"][stage] for r in reports if stage in r["stages"]
            ]
            if values:
                _print_stat_row(stage, values)

    # Per-image stage timing (averaged across runs)
    image_stage_names: set[str] = set()
    for r in reports:
        for img in r["per_image"]:
            image_stage_names.update(
                k.removesuffix("_s")
                for k in img
                if k.endswith("_s") and k != "total_s"
            )

    if image_stage_names:
        print(
            f"\n{'Per-Image Timings (s)':<30}  {'Mean':>8}  {'Median':>8}  {'Std':>8}"
        )
        print(f"{'—' * 30}  {'—' * 8}  {'—' * 8}  {'—' * 8}")
        all_image_totals = [
            img["total_s"] for r in reports for img in r["per_image"]
        ]
        _print_stat_row("image_total", all_image_totals)

        for stage in sorted(image_stage_names):
            key = f"{stage}_s"
            values = [
                img[key]
                for r in reports
                for img in r["per_image"]
                if key in img
            ]
            if values:
                _print_stat_row(stage, values)


def _print_stat_row(label: str, values: list[float]) -> None:
    """Print a single row of mean/median/std statistics."""
    m = mean(values)
    md = median(values)
    s = stdev(values) if len(values) > 1 else 0.0
    print(f"{label:<30}  {m:>8.3f}  {md:>8.3f}  {s:>8.3f}")


@omero_connect
def run_benchmarks(
    plate_id: int,
    repeats: int,
    excel_path: Path,
    output_dir: Path,
    conn: BlitzGateway | None = None,
) -> None:
    """Run repeated benchmark analyses.

    Args:
        plate_id: OMERO plate ID to analyse.
        repeats: Number of repeat runs.
        excel_path: Path to the Excel metadata file.
        output_dir: Directory where JSON reports are written.
        conn: OMERO connection (injected by decorator).
    """
    assert conn is not None

    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []

    for i in range(1, repeats + 1):
        print(f"\n{'=' * 70}")
        print(f"  RUN {i}/{repeats}")
        print(f"{'=' * 70}")

        # Delete cached dataset (flatfield + segmentation masks)
        print("Cleaning up dataset from previous run...")
        _delete_plate_dataset(conn, plate_id)

        # Delete all annotations from plate and wells for a clean slate
        print("Cleaning up plate metadata (annotations + leftover Excel)...")
        _delete_plate_metadata(conn, plate_id)

        # Re-attach Excel metadata so parsing is included in timing
        print("Attaching Excel metadata to plate...")
        _attach_excel_file(conn, plate_id, excel_path)

        # Initialise fresh benchmark timer
        init_benchmark(enabled=True, plate_id=plate_id)
        timer = get_benchmark()

        # Run the full pipeline
        plate_loop(conn, plate_id)

        # Save report into the system/GPU/date subfolder
        report_path = timer.save_report(directory=output_dir)
        report_paths.append(report_path)
        print(f"Report saved: {report_path}")

        # Brief pause between runs
        if i < repeats:
            time.sleep(2)

    _print_summary(report_paths, output_dir)


def main() -> None:
    """Parse arguments and launch benchmark repeats."""
    parser = argparse.ArgumentParser(
        description="Run repeated omero-screen benchmarks to measure timing variability."
    )
    parser.add_argument(
        "--plate-id",
        type=int,
        default=370,
        help="OMERO plate ID (default: %(default)s)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeat runs (default: %(default)s)",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=project_root / "benchmarks" / "benchmark-metadata.xlsx",
        help="Path to Excel metadata file (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Use GPU (Apple Silicon MPS / CUDA) for segmentation. "
        "--no-gpu forces CPU. Default: auto-detect.",
    )
    parser.add_argument(
        "--cellpose",
        choices=["cp3", "cp4"],
        default="cp3",
        help="Cellpose version to use for segmentation (default: %(default)s).",
    )
    args = parser.parse_args()

    if not args.excel.exists():
        raise FileNotFoundError(f"Excel file not found: {args.excel}")

    # Set GPU preference before any torch imports in the pipeline
    if args.gpu is not None:
        os.environ["OMERO_SCREEN_USE_GPU"] = "1" if args.gpu else "0"

    # Override MODEL_DICT for CP4
    if args.cellpose == "cp4":
        default_config.MODEL_DICT = _CP4_MODELS

    output_dir = _build_output_dir(args.cellpose, args.gpu)

    device_label = "auto-detect"
    if args.gpu is True:
        device_label = "GPU (forced)"
    elif args.gpu is False:
        device_label = "CPU (forced)"

    print(f"Plate ID:     {args.plate_id}")
    print(f"Repeats:      {args.repeats}")
    print(f"Cellpose:     {args.cellpose}")
    print(f"Device:       {device_label}")
    print(f"Metadata:     {args.excel}")
    print(f"Output dir:   {output_dir}")

    run_benchmarks(args.plate_id, args.repeats, args.excel, output_dir)


if __name__ == "__main__":
    main()
