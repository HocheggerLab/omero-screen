#!/usr/bin/env python3
"""Main entry point for the OMERO screen application.

This module provides the OMERO screen application. It supports configuring the environment
and running the analysis.

Main Functions:
    - main: Collects arguments to configure the environment and runs OMERO screen plate analysis.
"""

import argparse
import os


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
        help="Width N of for the inference galleray NxN (default: %(default)s)",
    )
    group.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Classification batch size (default: %(default)s)",
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

    from omero.gateway import BlitzGateway
    from omero_utils.omero_connect import omero_connect

    from omero_screen.loops import plate_loop

    @omero_connect
    def run_plate_loop(
        plate_ids: list[int], conn: BlitzGateway | None = None
    ) -> None:
        assert conn is not None
        for plate_id in plate_ids:
            plate_loop(conn, plate_id)

    run_plate_loop(args.ID)


if __name__ == "__main__":
    main()
