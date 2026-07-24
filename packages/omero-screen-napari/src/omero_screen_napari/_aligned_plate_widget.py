"""Widget to load a single well from multiple aligned cyclic-IF (4i) plates.

Two backends, chosen once at the top of import by the master plate's
segmentation mode (mirroring the main welldata dispatch):

* **Stitched** master → build (if needed) and load the combined *aligned*
  OME-Zarr from the isolated ``aligned/`` cache namespace. All cycles are
  baked into one multi-channel canvas in the master frame, so napari sees a
  single well image (see :mod:`omero_screen_napari.zarr_cache.aligned_builder`).
* **Non-stitched** master → the legacy live path: pull each cycle plate from
  OMERO field-by-field and overlay them as napari layers using the per-well
  alignment as a layer ``translate`` offset.

The plugin can be run from napari as Aligned Plate Widget under Plugins.
"""

import re
from typing import Any

import numpy as np
from loguru import logger
from magicgui import magic_factory
from magicgui.widgets import Container
from napari.layers import Image
from napari.qt.threading import create_worker
from napari.utils import notifications
from napari.viewer import Viewer
from omero.gateway import BlitzGateway

from omero_screen_napari._welldata_widget import (
    add_label_layers,
    clear_viewer_layers,
    set_color_maps,
)
from omero_screen_napari.omero_data import OmeroConnection
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.welldata_api import (
    get_plate_alignments,
    parse_omero_data,
)
from omero_screen_napari.zarr_cache import (
    cached_wells,
    is_stitched_plate,
    load_plate_to_viewer,
)
from omero_screen_napari.zarr_cache.aligned_builder import build_aligned_zarr
from omero_screen_napari.zarr_cache.paths import aligned_zarr_root


def aligned_plate_widget_gui() -> Container:  # type: ignore[type-arg]
    """This function combines the widgets into a single widget."""
    from omero_screen_napari._logging import init_plugin_logging

    init_plugin_logging()
    # Call the magic factories to get the widget instances
    aligned_plate_widget_instance = aligned_plate_widget()
    return Container(
        widgets=[
            aligned_plate_widget_instance,
        ]
    )


# Keep references to running build workers so they aren't garbage-collected
# (a dropped QRunnable reference cancels the thread mid-build).
_ALIGNED_WORKERS: list[Any] = []


# Widget to call Omero and load well images
@magic_factory(call_button="Enter")
def aligned_plate_widget(
    viewer: Viewer,
    plate_id: str = "Plate ID",
    well_pos: str = "Well Position",
    image: int = 0,
    sample_alignments: bool = False,
    show_all_nuclei: bool = False,
) -> None:
    """Load one well of a 4i experiment (all cycles) into napari.

    Detects the master plate's segmentation mode once and dispatches:
    stitched → the combined aligned OME-Zarr (built on demand, in the
    background); non-stitched → the legacy live per-field overlay path.

    ``show_all_nuclei`` (debug, stitched path only) keeps every cycle's DAPI as
    its own layer so repeat→master registration can be checked visually. It
    builds a different channel set into a separate ``aligned/debug`` cache.
    """
    # Single well only
    if not re.match("^[A-Z]+[0-9]+$", well_pos):
        raise ValueError("Invalid well position: " + well_pos)

    master_id = int(plate_id)
    stitched = False
    omero_conn = OmeroConnection()
    try:
        conn = omero_conn.get_conn()
        stitched = is_stitched_plate(conn, master_id)
        if not stitched:
            logger.info(
                f"Plate {master_id} is not stitched — using live overlay path"
            )
            _load_aligned_live(
                viewer, conn, master_id, well_pos, image, sample_alignments
            )
    finally:
        omero_conn.close(hard=False)

    # Stitched build runs in a background worker (opens its own connection) so
    # the ~minutes-long per-well assembly doesn't freeze the napari GUI.
    if stitched:
        logger.info(f"Plate {master_id} is stitched — using aligned zarr path")
        _load_aligned_zarr_bg(
            viewer, master_id, well_pos, keep_all_nuclei=show_all_nuclei
        )


def _load_aligned_zarr_bg(
    viewer: Viewer,
    master_id: int,
    well_pos: str,
    *,
    keep_all_nuclei: bool,
) -> None:
    """Build (if absent) the aligned 4i zarr in a worker, then load the well.

    ``keep_all_nuclei`` builds a debug variant (every cycle's DAPI) into a
    separate ``aligned/debug`` namespace so it never collides with the normal
    aligned cache's plate metadata.
    """
    root = (
        aligned_zarr_root() / "debug"
        if keep_all_nuclei
        else aligned_zarr_root()
    )

    if well_pos in cached_wells(master_id, root=root):
        load_plate_to_viewer(viewer, master_id, well_pos, root=root)
        return

    def _build() -> Any:
        build_conn = OmeroConnection()
        try:
            yield from build_aligned_zarr(
                master_id,
                build_conn.get_conn(),
                wells=[well_pos],
                keep_all_nuclei=keep_all_nuclei,
                root=root,
            )
        finally:
            build_conn.close(hard=False)

    def _on_return(_value: Any = None) -> None:
        load_plate_to_viewer(viewer, master_id, well_pos, root=root)
        notifications.show_info(f"Loaded aligned well {well_pos}")

    def _on_error(exc: Exception) -> None:
        logger.error(
            f"Aligned build failed for plate {master_id} well {well_pos}: {exc}"
        )
        notifications.show_error(f"Aligned build failed: {exc}")

    notifications.show_info(
        f"Building aligned well {well_pos} in the background — this can take "
        f"a few minutes."
    )
    worker = create_worker(_build)
    worker.yielded.connect(lambda w: logger.info(f"Built aligned well {w}"))
    worker.returned.connect(_on_return)
    worker.errored.connect(_on_error)
    _ALIGNED_WORKERS.append(worker)
    worker.finished.connect(
        lambda: _ALIGNED_WORKERS.remove(worker)
        if worker in _ALIGNED_WORKERS
        else None
    )
    worker.start()


def _load_aligned_live(
    viewer: Viewer,
    conn: BlitzGateway,
    master_id: int,
    well_pos: str,
    image: int,
    sample_alignments: bool,
) -> None:
    """Legacy live path: overlay each cycle plate's fields via layer translate.

    For aligned plates, the primary plate's agg_data.csv contains all channel
    data from all aligned plates, so we only import cellview data once.
    """
    # Get alignment for the plate
    alignments = get_plate_alignments(
        master_id, sample_alignments=sample_alignments, conn=conn
    )
    plates = alignments["plate"].unique()
    logger.info(f"Loaded alignments for plates: {plates}")

    all_channels: set[str] = set()

    # Load primary plate with cellview data (includes all channel data from aligned plates)
    parse_omero_data(omero_data, str(master_id), well_pos, str(image))
    clear_viewer_layers(viewer)
    _add_image_to_viewer(viewer, all_channels)
    labels = omero_data.labels

    all_channels = set(omero_data.channel_data.keys())

    # Load aligned plates (images only, skip cellview import since data already loaded)
    for plate_other in plates:
        # Get the images (not the labels, and skip cellview since already imported)
        parse_omero_data(
            omero_data,
            str(plate_other),
            well_pos,
            str(image),
            options=["skip_cellview"],
        )
        # Translate
        mask = (alignments["well"] == well_pos) & (
            alignments["plate"] == plate_other
        )
        if sample_alignments:
            mask = mask & (alignments["image_id"] == omero_data.image_ids[0])
        df = alignments[mask]
        if df.empty:
            raise Exception(
                f"Plate {plate_other} is missing alignment for well: {well_pos}"
            )
        # Translation maps plate_id to plate_other so negate
        trans = (-df.iloc[0]["x"], -df.iloc[0]["y"])
        logger.info(f"Plate {plate_other} {well_pos} translation {trans}")

        # Filter channels already added to the viewer (e.g duplicate alignment channel)
        _add_image_to_viewer(viewer, all_channels, trans)

    set_color_maps(viewer)

    add_label_layers(viewer, labels)


def _add_image_to_viewer(
    viewer: Viewer,
    all_channels: set[str],
    trans: tuple[float, float] | None = None,
) -> None:
    num_channels = omero_data.images.shape[-1]
    logger.debug(
        f"The images shape is {omero_data.images.shape} ({omero_data.images.dtype})"
    )
    channel_names: dict[int, str] = {
        int(value): key for key, value in omero_data.channel_data.items()
    }
    # Create translation
    translate = None
    if trans and omero_data.pixel_size is not None:
        # A 1-D array of factors to shift each axis by. X is the last axis.
        # Translation is broadcast to 0 in leading dimensions.
        # Scaling is applied before translation so we must scale the translation.
        translate = [
            trans[1] * omero_data.pixel_size[1],
            trans[0] * omero_data.pixel_size[0],
        ]

    for i in range(num_channels):
        if channel_names[i] in all_channels:
            continue
        all_channels.add(channel_names[i])
        image_data = omero_data.images[..., i]
        layer = viewer.add_image(
            image_data, scale=omero_data.pixel_size, translate=translate
        )
        assert isinstance(layer, Image), (
            "Expected layer to be an instance of Image"
        )
        layer.contrast_limits_range = (0, 65535)
        layer.contrast_limits = (np.min(image_data), np.max(image_data))
        layer.blending = "additive"
        layer.name = channel_names[i]

    # Configure the scale bar
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"
