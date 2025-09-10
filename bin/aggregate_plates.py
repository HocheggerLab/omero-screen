#!/usr/bin/env python3
"""Combines multiple repeat OMERO screen experiments.

This module provides aggregation of repeated screen experiments. For example a repeat screen
uses the same plate that has been washed and stained with a new dye. This allows staining for many
cell markers.

Main Functions:
    - main: Collects arguments to configure the environment and runs OMERO screen plate aggregation.
"""

import argparse
import os


def main() -> None:
    """Combines multiple repeat OMERO screen experiments."""
    parser = argparse.ArgumentParser(
        description="Program to combine repeat OMERO screen experiments."
    )
    parser.add_argument("ID", type=int, help="OMERO plate ID")
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for samples (default is a random value)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=25,
        help="Distance threshold for alignment mappings (default: %(default)s)",
    )
    parser.add_argument(
        "--method",
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help="""Mapping method:
            0=minimum distance;
            1=KD-Tree minimum distance;
            2=Greedy nearest neighbour;
            3=Mask overlap
            (default: %(default)s)""",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=6,
        help="Number of standard deviations from the mean distance to exclude distance mappings (default: %(default)s)",
    )
    parser.add_argument(
        "--gallery",
        type=int,
        default=4,
        help="Mapping gallery grid size (default: %(default)s)",
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
        aggregate_plates,
        get_plate_alignments,
        mapping_gallery,
    )

    # Same seed for all sampling for convenience
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(8))

    @omero_connect
    def run_plate_loop(
        plate_id: int, conn: BlitzGateway | None = None
    ) -> None:
        assert conn is not None
        alignments = get_plate_alignments(conn, plate_id)
        df, image_map = aggregate_plates(
            conn,
            plate_id,
            alignments,
            threshold=args.threshold,
            method=args.method,
            std_distance=args.std,
        )
        if args.gallery > 0:
            mapping_gallery(
                conn,
                plate_id,
                df,
                image_map,
                grid_size=args.gallery,
                seed=args.seed,
            )

    run_plate_loop(args.ID)


if __name__ == "__main__":
    main()
