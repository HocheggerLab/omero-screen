#!/usr/bin/env python
r"""Export per-well cell-cycle montage PDFs from the zarr cache.

One page per well: a few randomly drawn cells from each cell-cycle phase, one
row per cell, showing a DAPI+tubulin composite with the cell mask outlined,
then every remaining channel in greyscale.

The plate needs a zarr cache (build it from the plate-info dialog or
``zarr_cache_build.py``) and measurements in CellView.

Examples::

    # Every well of a plate, 4 cells per phase
    python bin/phase_montage.py 4127 --out figures/

    # Two wells, more cells, a different draw
    python bin/phase_montage.py 4127 C3 D4 --cells 6 --seed 12

    # Include Sub-G1, and outline nuclei rather than whole cells
    python bin/phase_montage.py 4127 --phases G1 S G2/M Polyploid Sub-G1 \\
        --mask nuclei
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omero_screen_napari.phase_montage import (
    DEFAULT_PHASES,
    MontageConfig,
    MontageError,
    export_well_pdf,
    load_plate_measurements,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase_montage")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("plate_id", type=int)
    parser.add_argument(
        "wells",
        nargs="*",
        help="Well labels (e.g. C3 D4). Default: every well with measurements.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("phase_montages"),
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=4,
        help="Cells per phase, i.e. rows per phase (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed for the per-phase random draw, stamped on the figure "
            "(default: %(default)s). Change it to redraw."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(DEFAULT_PHASES),
        help=(
            "Phases to show, in row order (default: %(default)s). Sub-G1 is "
            "omitted by default as it is mostly debris."
        ),
    )
    parser.add_argument(
        "--overlay",
        nargs="+",
        default=["dapi", "tub"],
        help=(
            "Channel base names composited in colour; everything else is "
            "shown greyscale (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--mask",
        default="cells",
        choices=["cells", "nuclei"],
        help="Label layer to outline (default: %(default)s).",
    )
    parser.add_argument(
        "--crop-um",
        type=float,
        default=None,
        help=(
            "Crop edge in microns. Default sizes it from the largest selected "
            "cell so polyploid cells are not clipped."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Export one montage PDF per requested well."""
    args = _parse_args()
    try:
        df = load_plate_measurements(args.plate_id)
    except MontageError as exc:
        logger.error("%s", exc)
        return 2

    wells = args.wells or sorted(str(w) for w in df["well"].unique().to_list())
    config = MontageConfig(
        phases=tuple(args.phases),
        cells_per_phase=args.cells,
        seed=args.seed,
        crop_um=args.crop_um,
        overlay=tuple(o.lower() for o in args.overlay),
        mask=args.mask,
    )

    written, failed = 0, 0
    for well in wells:
        try:
            export_well_pdf(args.plate_id, well, df, args.out, config)
            written += 1
        except MontageError as exc:
            # One bad well should not abandon the rest of the plate.
            logger.warning("Skipping well %s: %s", well, exc)
            failed += 1

    logger.info(
        "Wrote %d montage(s) to %s%s",
        written,
        args.out,
        f" ({failed} well(s) skipped)" if failed else "",
    )
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
