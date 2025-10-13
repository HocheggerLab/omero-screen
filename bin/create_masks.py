#!/usr/bin/env python3
"""Creates missing cell masks for repeat OMERO screen experiments.

This module provides generation of missing cell masks by translation
of the masks from a master plate. Each repeat plate must have been
previously aligned to the master plate.

Main Functions:
    - main: Collects arguments to configure the environment and runs OMERO screen plate aggregation.
"""

import argparse
import os


def main() -> None:
    """Creates missing cell masks for repeat OMERO screen experiments."""
    parser = argparse.ArgumentParser(
        description="Program to create missing cell masks for repeat OMERO screen experiments."
    )
    parser.add_argument("ID", type=int, help="OMERO plate ID")
    parser.add_argument(
        "--sample-alignments",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use per-sample alignments; else per-well alignments (default: %(default)s)",
    )
    group = parser.add_argument_group("Omero Screen overrides")
    group.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (requires configuration file .env.{name}).",
    )
    args = parser.parse_args()

    # Note: Lazy import to speed up parsing errors

    # Module initialisation sets the environment variables. Create overrides.

    if args.env:
        os.environ["ENV"] = args.env

    from omero.gateway import BlitzGateway
    from omero_utils.omero_connect import omero_connect

    from omero_screen.plate_aggregation import (
        create_cell_masks,
        get_plate_alignments,
    )

    @omero_connect
    def run_plate_loop(
        plate_id: int, conn: BlitzGateway | None = None
    ) -> None:
        assert conn is not None
        alignments = get_plate_alignments(
            conn, plate_id, sample_alignments=args.sample_alignments
        )
        create_cell_masks(
            conn,
            plate_id,
            alignments,
        )

    run_plate_loop(args.ID)


if __name__ == "__main__":
    main()
