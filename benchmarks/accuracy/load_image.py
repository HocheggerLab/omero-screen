#!/usr/bin/env python3
"""Load a single OMERO image into napari for ground-truth mask annotation.

Usage::

    python benchmarks/accuracy/load_image.py <image_id>
    python benchmarks/accuracy/load_image.py <image_id> --env production

Each fluorescence channel is loaded as a separate greyscale napari layer,
named using the channel annotations on the parent well (e.g. DAPI, EdU, Tub).
Channel names fall back to 'ch0', 'ch1', ... if annotations are not found.

The viewer is left open for manual annotation.  Use the napari Labels layer
(Add Labels button or keyboard shortcut L) to draw ground-truth nucleus and
cell masks for accuracy benchmarking.
"""

import argparse
import os
import sys


def _get_channel_names(conn: object, image: object) -> list[str]:
    """Resolve channel names from well/plate map annotations.

    Args:
        conn: OMERO BlitzGateway connection
        image: OMERO ImageWrapper

    Returns:
        List of channel name strings, one per channel index.
    """
    from omero_utils.map_anns import parse_annotations

    from omero_screen.constants import OmeroScreenNS

    n_channels = int(image.getSizeC())  # type: ignore[attr-defined]
    defaults = [f"ch{i}" for i in range(n_channels)]

    # image → well → plate; reload each via conn to get fully-loaded objects
    well_id = image.getWellId() if hasattr(image, "getWellId") else None  # type: ignore[attr-defined]
    if well_id is None:
        # Fall back: navigate via listParents
        parents = list(image.listParents())  # type: ignore[attr-defined]
        well_id = parents[0].getId() if parents else None

    if well_id is None:
        return defaults

    well = conn.getObject("Well", well_id)  # type: ignore[attr-defined]
    if well is None:
        return defaults

    plate = conn.getObject("Plate", well.getPlate().getId())  # type: ignore[attr-defined]
    candidates = [obj for obj in [well, plate] if obj is not None]

    channel_map: dict[int, str] = {}
    for obj in candidates:
        anns = parse_annotations(obj, ns=OmeroScreenNS.METADATA)
        # Channel annotations are stored as {"DAPI": "0", "EdU": "1", ...}
        for k, v in anns.items():
            try:
                idx = int(v)
                channel_map[idx] = k
            except ValueError:
                pass
        if channel_map:
            break

    return [channel_map.get(i, defaults[i]) for i in range(n_channels)]


def main() -> None:
    """Entry point: parse args, connect to OMERO, open napari."""
    parser = argparse.ArgumentParser(
        description="Load an OMERO image into napari for ground-truth annotation."
    )
    parser.add_argument("image_id", type=int, help="OMERO image ID")
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (loads .env.{name}). Defaults to 'development'.",
    )
    args = parser.parse_args()

    if args.env:
        os.environ["ENV"] = args.env

    # Trigger env var loading
    from omero.gateway import BlitzGateway
    from omero_screen_napari.omero_image import get_image
    from omero_utils.omero_connect import omero_connect

    import omero_screen  # noqa: F401

    @omero_connect
    def _load(
        image_id: int, conn: BlitzGateway | None = None
    ) -> tuple[object, ...]:  # type: ignore[return]
        assert conn is not None
        image = conn.getObject("Image", image_id)
        if image is None:
            print(f"ERROR: Image {image_id} not found on OMERO server.")
            sys.exit(1)

        channel_names = _get_channel_names(conn, image)
        print(f"Image {image_id}: {image.getName()}")
        print(
            f"  Size: {image.getSizeX()} x {image.getSizeY()}, {image.getSizeC()} channels"
        )
        print(f"  Channels: {channel_names}")

        # Shape: (T, Z, Y, X, C) — take t=0, z=0 for a simple 2D view
        arr = get_image(conn, image_id)  # TZYXC
        frame = arr[0, 0]  # YXC

        return frame, channel_names

    frame, channel_names = _load(args.image_id)

    import napari

    viewer = napari.Viewer(title=f"OMERO image {args.image_id}")

    colormaps = ["gray", "green", "red", "blue", "cyan", "magenta"]
    for i, name in enumerate(channel_names):
        channel_data = frame[..., i]
        cmap = colormaps[i % len(colormaps)]
        viewer.add_image(
            channel_data,
            name=name,
            colormap=cmap,
            blending="additive",
        )

    print(
        "\nNapari open. Add Labels layers (shortcut: L) to draw ground-truth masks."
    )
    print("Suggested layers to create:")
    print("  - 'nuclei_gt'  : nucleus ground-truth masks")
    print("  - 'cells_gt'   : cell ground-truth masks")

    napari.run()


if __name__ == "__main__":
    main()
