#!/usr/bin/env python3
"""Main entry point for the OMERO screen application.

This module provides the OMERO screen application. It supports configuring the environment
and running the analysis.

Main Functions:
    - main: Collects arguments to configure the environment and runs OMERO screen plate analysis.
"""

import argparse
import os

_CP4_MODELS: dict[str, str] = {
    "nuclei": "cp4:cpsam",
    "RPE": "cp4:cpsam",
    "HELA": "cp4:cpsam",
    "U2OS": "cp4:cpsam",
    "HCC1143": "cp4:cpsam",
    "MM231": "cp4:cpsam",
    "PALB": "cp4:cpsam",
}


def main() -> None:
    """Main entry point for the OMERO screen application."""
    parser = argparse.ArgumentParser(
        description="Program to run OMERO screen for the plate ID."
    )
    parser.add_argument("ID", nargs="+", type=int, help="OMERO plate ID")
    group = parser.add_argument_group("Omero Screen overrides")

    group.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (requires configuration file .env.{name}).",
    )
    group.add_argument(
        "--inference",
        nargs="+",
        type=str,
        default=None,
        metavar="MODEL",
        help="Inference model filename(s).",
    )
    group.add_argument(
        "--gallery",
        type=int,
        default=10,
        help="Width N for the inference NxN example gallery (default: %(default)s)",
    )
    group.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Classification batch size (default: %(default)s)",
    )
    group.add_argument(
        "--segmentation",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Only perform image segmentation (default: %(default)s)",
    )
    group.add_argument(
        "--cp4",
        default=False,
        action="store_true",
        help="Use Cellpose 4 (cpsam) for segmentation instead of the default Cellpose 3 models.",
    )
    group.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Override all segmentation models with a single model name (e.g. 'cp4:cpsam', 'cp3:cyto3'). Overrides --cp4.",
    )
    group.add_argument(
        "--benchmark",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Record per-image timing data and write a JSON benchmark report (default: %(default)s)",
    )
    group.add_argument(
        "--stitch",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run stitched-well segmentation: assemble all fields per well into one canvas, segment that canvas, and exclude border objects only at the outer edge (default: %(default)s)",
    )
    group.add_argument(
        "--track",
        type=str,
        nargs="?",
        const="general_2d",
        default=None,
        metavar="MODEL",
        help="Track nuclei across time with Trackastra. Optional MODEL is a pretrained name or checkpoint path (default when flag given: %(const)s). Requires --stitch and a timelapse (T>1); a no-op on single-timepoint plates.",
    )
    group.add_argument(
        "--track-mode",
        type=str,
        default="greedy",
        choices=["greedy", "greedy_nodiv", "ilp"],
        help="Trackastra linking mode (default: %(default)s).",
    )
    group.add_argument(
        "--track-batch-size",
        type=int,
        default=4,
        help="Attention windows Trackastra scores per forward pass (default: %(default)s). "
        "Caps GPU memory during tracking; lower this if tracking hits CUDA OOM, "
        "raise it for faster scoring when VRAM allows (Trackastra's own GPU default is 16).",
    )
    args = parser.parse_args()

    # Note: Lazy import to speed up parsing errors

    # Module initialisation sets the environment variables. Create overrides.

    if args.env:
        os.environ["ENV"] = args.env
    if args.inference:
        os.environ["OMERO_SCREEN_INFERENCE_MODEL"] = ":".join(args.inference)
    if args.gallery:
        os.environ["OMERO_SCREEN_INFERENCE_GALLERY_WIDTH"] = str(args.gallery)
    if args.batch:
        os.environ["OMERO_SCREEN_INFERENCE_BATCH_SIZE"] = str(args.batch)
    if args.track:
        os.environ["OMERO_SCREEN_TRACKING_MODEL"] = args.track
        os.environ["OMERO_SCREEN_TRACKING_MODE"] = args.track_mode
        os.environ["OMERO_SCREEN_TRACKING_BATCH_SIZE"] = str(
            args.track_batch_size
        )

    if args.model or args.cp4:
        from omero_screen import default_config

        model_name = args.model if args.model else "cp4:cpsam"
        default_config.MODEL_DICT = {
            k: model_name for k in default_config.MODEL_DICT
        }

    from omero.gateway import BlitzGateway
    from omero_utils.omero_connect import omero_connect

    from omero_screen.benchmarking import get_benchmark, init_benchmark
    from omero_screen.loops import plate_loop

    @omero_connect
    def run_plate_loop(
        plate_ids: list[int], conn: BlitzGateway | None = None
    ) -> None:
        assert conn is not None
        for plate_id in plate_ids:
            init_benchmark(enabled=args.benchmark, plate_id=plate_id)
            timer = get_benchmark()
            plate_loop(
                conn,
                plate_id,
                segmentation_mode=args.segmentation,
                stitch_mode=args.stitch,
            )
            report_path = timer.save_report()
            if args.benchmark:
                print(f"Benchmark report saved to {report_path}")

    run_plate_loop(args.ID)


if __name__ == "__main__":
    main()
