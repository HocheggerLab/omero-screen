#!/usr/bin/env python3
"""Aligns multiple repeat OMERO screen plate experiments.

This module provides alignment of repeated screen experiments. For example a repeat screen
uses the same plate that has been washed and stained with a new dye. This allows staining for many
cell markers.

Main Functions:
    - main: Collects arguments to configure the environment and runs OMERO screen plate alignment.
"""

import argparse
import os


def main() -> None:
    """Aligns multiple repeat OMERO screen plate experiments."""
    parser = argparse.ArgumentParser(
        description="Program to align repeat OMERO screen experiments."
    )
    parser.add_argument("ID", nargs="+", type=int, help="OMERO plate IDs")
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for samples (default is a random value)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="DAPI",
        help="Alignment channel (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        dest="number_of_alignments",
        type=int,
        default=5,
        help="Number of alignments used to create the average (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100,
        help="Distance threshold for alignments (default: %(default)s)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10,
        help="Distance tolerance for alignments to their centroids (default: %(default)s)",
    )
    parser.add_argument(
        "--gallery",
        type=int,
        default=4,
        action=argparse.BooleanOptionalAction,
        help="Alignment gallery grid size (default: %(default)s)",
    )
    group = parser.add_argument_group("Omero Screen overrides")
    group.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (requires configuration file .env.{name}).",
    )
    args = parser.parse_args()

    if len(args.ID) < 2:
        print("ERROR: Require multiple plate IDs")
        exit(1)

    # Note: Lazy import to speed up parsing errors

    # Module initialisation sets the environment variables. Create overrides.

    if args.env:
        os.environ["ENV"] = args.env

    from omero.gateway import BlitzGateway
    from omero_utils.omero_connect import omero_connect

    from omero_screen.plate_aggregation import align_plates

    if args.gallery > 0:
        import math
        import random

        from omero_utils.attachments import (
            attach_figure,
            delete_file_attachment,
        )

        from omero_screen.config import get_logger
        from omero_screen.gallery_figure import create_gallery

        logger = get_logger(__name__)

    # Same seed for all sampling for convenience
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(8))

    @omero_connect
    def run_plate_loop(
        plate_ids: list[int], conn: BlitzGateway | None = None
    ) -> None:
        assert conn is not None
        plate_id = plate_ids[0]
        plate_ids = plate_ids[1:]
        alignments, examples = align_plates(
            conn,
            plate_id,
            plate_ids,
            align_ch=args.channel,
            number_of_alignments=args.number_of_alignments,
            seed=args.seed,
            threshold=args.threshold,
            tolerance=args.tolerance,
            output_alignments=args.gallery > 0,
        )
        if examples is not None:
            plate = conn.getObject("Plate", plate_id)
            for plate_other, images in zip(plate_ids, examples, strict=False):
                n = len(images)
                if n > args.gallery**2:
                    n = args.gallery**2
                    random.seed(a=args.seed)
                    images = random.sample(images, n)
                name = f"alignment_{plate_other}"
                logger.info("Saving gallery: %s", name)
                fig = create_gallery(
                    images, math.ceil(math.sqrt(n)), show_contours=False
                )
                delete_file_attachment(conn, plate, ends_with=name + ".png")
                attach_figure(
                    conn,
                    fig,
                    plate,
                    name,
                )

    run_plate_loop(args.ID)


if __name__ == "__main__":
    main()
