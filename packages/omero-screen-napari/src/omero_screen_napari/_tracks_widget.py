"""napari widget to overlay Trackastra tracks on a loaded well.

Reads track data straight from the CellView measurements already loaded into
``omero_data.plate_data`` (no GEFF/zarr needed) and adds a napari ``Tracks``
layer. Run it *after* loading a well with the Welldata widget.

The track centroids are in the same (stitched) pixel space as the displayed
well image, so the Tracks layer overlays the segmentation directly.
"""

from __future__ import annotations

import logging

from magicgui import magic_factory
from magicgui.widgets import FunctionGui
from napari.utils import notifications
from napari.viewer import Viewer

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.tracks_loader import has_tracks, load_tracks_for_well

logger = logging.getLogger("omero-screen-napari")

_TRACKS_LAYER_NAME = "tracks"


@magic_factory(
    call_button="Load tracks",
    color_by={"choices": ["track_id", "cell_cycle"]},
    tail_length={"min": 0, "max": 1000},
)
def tracks_widget(
    viewer: Viewer,
    well: str = "Well Position",
    color_by: str = "track_id",
    tail_length: int = 30,
) -> None:
    """Add a napari Tracks layer for one well from the loaded CellView data.

    Args:
        viewer: Active napari viewer (injected by magicgui).
        well: Well position to load tracks for (e.g. ``"C4"``).
        color_by: Track property to colour by — ``track_id`` or ``cell_cycle``
            (the latter only if cell-cycle analysis ran).
        tail_length: Number of past frames drawn behind each track head.
    """
    plate_data = omero_data.plate_data
    if not has_tracks(plate_data):
        notifications.show_warning(
            "No track data in the loaded plate. Re-run the pipeline with "
            "--track (and --stitch) to generate tracks."
        )
        return

    try:
        tracks = load_tracks_for_well(plate_data, well.strip())
    except (KeyError, ValueError) as exc:
        notifications.show_warning(
            f"Could not load tracks for {well!r}: {exc}"
        )
        return
    if tracks is None:  # pragma: no cover - guarded by has_tracks above
        return

    # Fall back to track_id colouring if the requested property is absent.
    prop = color_by if color_by in tracks.properties else "track_id"
    if prop != color_by:
        notifications.show_info(
            f"Property {color_by!r} not available; colouring by track_id."
        )

    # Replace any existing tracks layer so re-running is idempotent.
    if _TRACKS_LAYER_NAME in viewer.layers:
        del viewer.layers[_TRACKS_LAYER_NAME]

    viewer.add_tracks(
        tracks.data,
        graph=tracks.graph,
        properties=tracks.properties,
        color_by=prop,
        tail_length=tail_length,
        name=_TRACKS_LAYER_NAME,
    )
    n_tracks = len({int(t) for t in tracks.data[:, 0]})
    n_div = len(tracks.graph)
    notifications.show_info(
        f"Loaded {n_tracks} tracks ({n_div} divisions) for well {well}."
    )
    logger.info(
        "Added tracks layer for well %s: %d tracks, %d divisions",
        well,
        n_tracks,
        n_div,
    )


def tracks_gui_widget() -> FunctionGui:  # type: ignore[type-arg]
    """Factory used by the napari plugin manifest."""
    return tracks_widget()
