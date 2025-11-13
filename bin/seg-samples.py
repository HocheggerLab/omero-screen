#!/usr/bin/env python3
"""Program to export segmentation samples from a plate.

Well sample time points are randomly sampled from the plate and images and
masks exported to a directory. If the image has a z-stack then the
maximum intensity projection (MIP) is exported.
"""

import argparse


def main() -> None:
    """Program to export segmentation samples from a plate."""
    parser = argparse.ArgumentParser(
        description="""Program to export segmentation samples from a plate"""
    )
    _ = parser.add_argument("ID", nargs="+", type=int, help="OMERO plate ID")
    group = parser.add_argument_group("Omero Screen overrides")

    group.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (requires configuration file .env.{name}).",
    )

    group = parser.add_argument_group("Sample Options")
    _ = group.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per plate (default: %(default)s)",
    )
    _ = group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sample seed (default: %(default)s)",
    )

    group = parser.add_argument_group("Export Options")
    _ = group.add_argument(
        "--out",
        help="Output directory (default: $SAMPLING_OUTPUT_DIRECTORY or 'samples')",
    )
    _ = group.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing samples (default: %(default)s)",
    )

    group = parser.add_argument_group("Image Options")
    _ = group.add_argument(
        "--compression",
        default="ZSTD",
        help="TIFF compression (e.g. None, LZW, ZSTD, ZLIB) (default: %(default)s)",
    )

    args = parser.parse_args()

    # Note: Lazy import to speed up parsing errors

    # Module initialisation sets the environment variables. Create overrides.

    import os

    if args.env:
        os.environ["ENV"] = args.env

    from omero.gateway import BlitzGateway
    from omero_utils.omero_connect import omero_connect

    from omero_screen.config import get_logger
    from omero_screen.sampling import segmentation_samples

    logger = get_logger(__name__)

    out = (
        args.out
        if args.out
        else os.getenv("SAMPLING_OUTPUT_DIRECTORY", default="samples")
    )

    logger.info("Exporting to: %s", out)
    os.makedirs(out, exist_ok=True)

    @omero_connect
    def run_sample_loop(
        plate_ids: list[int], conn: BlitzGateway | None = None
    ) -> None:
        assert conn is not None
        for plate_id in plate_ids:
            segmentation_samples(
                conn,
                plate_id,
                out,
                args.samples,
                seed=args.seed,
                overwrite=args.overwrite,
                compression=args.compression,
            )

    run_sample_loop(args.ID)


if __name__ == "__main__":
    main()
