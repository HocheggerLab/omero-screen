"""Scratch file to test plate_cache functions against real OMERO server.

Usage:
    python scratch.py

Requires .env.development (or set ENV=development) with valid
OMERO credentials. Update PLATE_ID below.
"""

import os
import time

from omero.gateway import BlitzGateway
from omero_screen_napari.plate_cache import (
    download_flatfield,
    download_plate_metadata,
    download_segmentation_masks,
    download_well_fields,
    download_well_metadata,
)

from omero_screen.config import get_logger

logger = get_logger(__name__)

# ---------- CONFIGURE THESE ----------
HOST = "ome2.hpc.sussex.ac.uk"
PORT = 4064
USERNAME = "helfrid"
PASSWORD = "Omero_21"
PLATE_ID = 1237
# --------------------------------------

# Set env vars so @omero_connect can create per-thread connections
os.environ["HOST"] = HOST
os.environ["USERNAME"] = USERNAME
os.environ["PASSWORD"] = PASSWORD


def main() -> None:
    """Test all plate_cache download functions on real data."""
    conn = BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT)
    try:
        if not conn.connect():
            print("ERROR: Could not connect to OMERO")
            return
        print(f"Connected to {HOST}:{PORT} as {USERNAME}")

        # === 1. Plate metadata ===
        print("\n=== Plate metadata ===")
        meta = download_plate_metadata(conn, PLATE_ID)
        print(f"  Name: {meta.name}")
        print(f"  Channels: {meta.channel_data}")

        # === 2. Flatfield correction ===
        print("\n=== Flatfield correction ===")
        ff = download_flatfield(conn, PLATE_ID)
        if ff is not None:
            print(f"  Shape: {ff.shape}, dtype: {ff.dtype}")
            print(
                f"  Min: {ff.min():.4f}, Max: {ff.max():.4f}, Mean: {ff.mean():.4f}"
            )
        else:
            print("  Not available (None)")

        # === 3. Well metadata ===
        print("\n=== Well metadata ===")
        plate = conn.getObject("Plate", PLATE_ID)
        wells = list(plate.listChildren())
        if not wells:
            print("  ERROR: Plate has no wells")
            return

        well = wells[0]
        row_label = chr(ord("A") + well.row)
        col_label = well.column + 1
        print(f"  Well: {row_label}{col_label}")
        well_meta = download_well_metadata(well)
        for k, v in well_meta.items():
            print(f"    {k}: {v}")
        if not well_meta:
            print("  (no annotations)")

        # === 4. Field download (sequential) ===
        print("\n=== Field download (sequential) ===")
        t0 = time.perf_counter()
        fields = download_well_fields(well, max_workers=6)
        elapsed = time.perf_counter() - t0
        print(f"  Downloaded {len(fields)} fields in {elapsed:.2f}s")
        for f in fields:
            print(
                f"  Field {f.index}: shape={f.image.shape}, "
                f"dtype={f.image.dtype}, "
                f"pos=({f.pos_x:.6f}, {f.pos_y:.6f}) m"
            )

        # === 5. Segmentation masks ===
        print("\n=== Segmentation masks ===")
        # Collect image IDs from the fields we just downloaded
        # We need the actual OMERO image IDs (not field indices)
        well_samples = list(well.listChildren())
        image_ids = [ws.getImage().getId() for ws in well_samples]
        print(f"  Looking up masks for image IDs: {image_ids}")

        masks = download_segmentation_masks(conn, PLATE_ID, image_ids)
        print(f"  Found {len(masks)}/{len(image_ids)} masks")
        for img_id, mask in masks.items():
            print(
                f"  Image {img_id}: shape={mask.shape}, "
                f"dtype={mask.dtype}, "
                f"labels={mask.max()}"
            )
        if not masks:
            print("  (none available — segmentation may not have been run)")

        print("\nDone!")

    finally:
        conn.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
