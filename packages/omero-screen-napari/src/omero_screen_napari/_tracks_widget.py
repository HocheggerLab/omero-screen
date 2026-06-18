"""napari widgets for inspecting Trackastra tracks on a loaded well.

Reads track data straight from the CellView measurements already loaded into
``omero_data.plate_data`` (no GEFF/zarr needed) and adds a napari ``Tracks``
layer. Run after loading a well with the Welldata widget.

For the lineage tree view we use the Lowe-lab **napari-arboretum** plugin
(installed as a dependency): with our Tracks layer selected, double-click a
track in napari to render its lineage. Their widget is far more polished than
anything we'd write ourselves — same authors as the napari ``Tracks`` layer.

Two export companions:
- Export Track CSV: type a ``track_id`` → a CSV slice of that one track's
  measurements, for downstream time-course analysis of clean tracks.
- Export well for Mastodon: a self-contained bundle (image + all tracks +
  README) for manual curation in Mastodon.
"""

from pathlib import Path

from loguru import logger
from magicgui import magic_factory
from magicgui.widgets import Container
from napari.layers import Labels
from napari.utils import notifications
from napari.viewer import Viewer
from qtpy.QtWidgets import QFileDialog

from omero_screen_napari.mastodon_export import export_well_for_mastodon
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.tracks_loader import (
    export_track_csv,
    has_tracks,
    load_tracks_for_well,
)
from omero_screen_napari.zarr_cache import (
    pin_plate,
    pinned_plate_ids,
    unpin_plate,
)

_TRACKS_LAYER_NAME = "tracks"


def _disable_double_click_zoom(viewer: Viewer) -> None:
    """Remove napari's default double-click-zoom binding on this viewer.

    napari registers ``double_click_to_zoom`` on every Viewer; in pan/zoom
    mode (the default) every double-click multiplies ``camera.zoom`` by two.
    That makes Arboretum's "double-click a track to draw its lineage"
    workflow unusable — each click also zooms in. Strip the callback so
    Arboretum (and other double-click handlers) see clean events.

    Idempotent: no-op if the callback was already removed (e.g. on a second
    Load tracks click).
    """
    from napari.components._viewer_mouse_bindings import double_click_to_zoom

    callbacks = viewer.mouse_double_click_callbacks
    if double_click_to_zoom in callbacks:
        callbacks.remove(double_click_to_zoom)
        logger.info("Disabled napari double-click-to-zoom (tracks workflow).")


def _reference_scale(viewer: Viewer) -> tuple[float, ...] | None:
    """Return the (T, Y, X) scale of an existing Labels layer, if any.

    OME-Zarr ``coordinateTransformations`` give the spatial layers a physical
    scale (e.g. ``[1.0, 0.5934, 0.5934]``). The Tracks layer must match it or
    the track positions render in raw pixels and drift off the nuclei. Labels
    layers are 3D (T, Y, X) — exactly the dimensions we need.
    """
    for layer in viewer.layers:
        if isinstance(layer, Labels) and layer.scale is not None:
            return tuple(float(s) for s in layer.scale)
    return None


def _resolve_well(typed: str) -> str | None:
    """Trim ``typed``; fall back to the well currently loaded if blank.

    Returns ``None`` and surfaces a warning when no well is available.
    """
    well = typed.strip()
    if well:
        return well
    loaded = list(getattr(omero_data, "well_pos_list", []) or [])
    if not loaded:
        notifications.show_warning(
            "No well loaded — load a well with the Welldata widget first, "
            "or type a well position (e.g. 'B2')."
        )
        return None
    return str(loaded[0])


@magic_factory(
    call_button="Load tracks",
    color_by={"choices": ["track_id", "cell_cycle"]},
    tail_length={"min": 0, "max": 1000},
    show_divisions={"label": "Show divisions (lineage)"},
)
def tracks_widget(
    viewer: Viewer,
    well: str = "",
    color_by: str = "track_id",
    tail_length: int = 10,
    show_divisions: bool = False,
) -> None:
    """Add a napari Tracks layer for one well from the loaded CellView data.

    Args:
        viewer: Active napari viewer (injected by magicgui).
        well: Well position to load tracks for (e.g. ``"C4"``). Leave blank to
            use the well currently displayed in the viewer.
        color_by: Track property to colour by — ``track_id`` or ``cell_cycle``
            (the latter only if cell-cycle analysis ran).
        tail_length: Number of past frames drawn behind each track head. Each
            frame redraws every visible tail segment, so on long timelapses
            with many tracks this dominates playback cost — keep it small
            (~10) for smooth scrubbing, raise only to inspect long histories.
        show_divisions: Pass the division lineage graph to the Tracks layer.
            Off by default: building the graph (thousands of divisions on a
            dense movie) adds a noticeable load freeze plus per-frame draw
            cost, so playback is smoother without it. Turn it on to see
            division links and to drive napari-arboretum's lineage tree.
    """
    plate_data = omero_data.plate_data
    if not has_tracks(plate_data):
        notifications.show_warning(
            "No track data in the loaded plate. Re-run the pipeline with "
            "--track (and --stitch) to generate tracks."
        )
        return

    well_id = _resolve_well(well)
    if well_id is None:
        return
    if not well.strip():
        notifications.show_info(f"Loading tracks for well {well_id}.")

    try:
        tracks = load_tracks_for_well(plate_data, well_id)
    except (KeyError, ValueError) as exc:
        notifications.show_warning(
            f"Could not load tracks for {well_id!r}: {exc}"
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

    scale = _reference_scale(viewer)
    add_kwargs: dict[str, object] = {}
    if scale is not None:
        add_kwargs["scale"] = scale
        logger.info(f"Aligning tracks with labels layer scale {scale}")

    # Only hand napari the lineage graph when the user asks for it: building
    # it for thousands of divisions freezes the load and adds per-frame draw
    # cost. An empty graph keeps the layer to plain tracks (and disables
    # arboretum's lineage view until divisions are turned back on).
    graph = tracks.graph if show_divisions else {}

    viewer.add_tracks(
        tracks.data,
        graph=graph,
        properties=tracks.properties,
        color_by=prop,
        tail_length=tail_length,
        name=_TRACKS_LAYER_NAME,
        **add_kwargs,
    )

    # Arboretum picks a track on double-click; napari's built-in
    # double-click-to-zoom would otherwise also fire and zoom the camera at
    # every click. Strip it so the tracks workflow behaves predictably.
    _disable_double_click_zoom(viewer)

    n_tracks = len({int(t) for t in tracks.data[:, 0]})
    n_div = len(tracks.graph)
    if show_divisions:
        msg = (
            f"Loaded {n_tracks} tracks ({n_div} divisions) for well {well_id}. "
            "Open Plugins → napari-arboretum and double-click a track for its "
            "lineage tree."
        )
    else:
        msg = (
            f"Loaded {n_tracks} tracks for well {well_id} (divisions hidden "
            f"for smoother playback). Tick 'Show divisions' and reload to see "
            f"the {n_div} division links and enable arboretum lineage."
        )
    notifications.show_info(msg)
    logger.info(
        f"Added tracks layer for well {well_id}: {n_tracks:d} tracks, {n_div:d} divisions, show_divisions={show_divisions}, tail_length={tail_length:d}"
    )


@magic_factory(call_button="Export track CSV")
def export_track_widget(
    viewer: Viewer,
    well: str = "",
    track_id: int = 0,
) -> None:
    """Export one track's measurement rows as CSV for downstream analysis.

    Pick a track id by inspecting the Tracks layer or the Arboretum lineage
    tree (each track is labelled with its id), then enter it here.

    Args:
        viewer: Active napari viewer (used only to parent the file dialog).
        well: Well to export from. Leave blank to use the currently-loaded
            well.
        track_id: Track id to export. ``0`` is treated as "no selection".
    """
    if track_id <= 0:
        notifications.show_warning(
            "Enter a track id (a positive integer) before exporting."
        )
        return

    well_id = _resolve_well(well)
    if well_id is None:
        return

    plate_id = getattr(omero_data, "plate_id", None)
    default_name = (
        f"plate_{plate_id}_well_{well_id}_track_{track_id}.csv"
        if plate_id is not None
        else f"well_{well_id}_track_{track_id}.csv"
    )
    parent = getattr(viewer, "window", None)
    parent_widget = getattr(parent, "_qt_window", None) if parent else None
    out_path_str, _ = QFileDialog.getSaveFileName(
        parent_widget,
        "Export track CSV",
        str(Path.home() / default_name),
        "CSV files (*.csv);;All files (*)",
    )
    if not out_path_str:
        return  # user cancelled

    out_path = Path(out_path_str)
    try:
        n_rows = export_track_csv(
            omero_data.plate_data, well_id, int(track_id), out_path
        )
    except (KeyError, ValueError) as exc:
        notifications.show_warning(f"Export failed: {exc}")
        return

    notifications.show_info(
        f"Exported track {track_id} ({n_rows} rows) to {out_path.name}."
    )
    logger.info(
        f"Exported {n_rows:d} rows for well {well_id} track {track_id:d} -> {out_path}"
    )


@magic_factory(call_button="Export well for Mastodon")
def mastodon_export_widget(
    viewer: Viewer,
    well: str = "",
) -> None:
    """Write the Mastodon tracks CSV + README for a well.

    Produces ``~/mastodon_exports/plate_<id>_<well>/README.txt`` and a
    ``tracks.csv`` beside the cached well image (no image copy — Mastodon opens
    the cache in place; the README has the exact paths). This does **not** pin
    the plate; use the Pin button if you'll curate over time.

    Args:
        viewer: Active napari viewer (unused; kept for the magicgui binding).
        well: Well to export. Leave blank to use the currently-loaded well.
    """
    plate_data = omero_data.plate_data
    if not has_tracks(plate_data):
        notifications.show_warning(
            "No track data in the loaded plate. Re-run the pipeline with "
            "--track (and --stitch) to generate tracks."
        )
        return

    well_id = _resolve_well(well)
    if well_id is None:
        return

    plate_id = getattr(omero_data, "plate_id", None)
    if plate_id is None:
        notifications.show_warning("No plate loaded.")
        return

    try:
        paths = export_well_for_mastodon(int(plate_id), well_id, plate_data)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        notifications.show_warning(f"Mastodon export failed: {exc}")
        return

    notifications.show_info(
        f"Exported well {well_id}. README: {paths['readme']}. "
        f"Pin the plate (button below) if curating over time."
    )
    logger.info(f"Mastodon export written to {paths['dir']}")


@magic_factory(call_button="Pin plate (protect from eviction)")
def pin_plate_widget(viewer: Viewer) -> None:
    """Pin the loaded plate so the cache evictor won't delete it.

    Use before curating a well in Mastodon over time (a separate Fiji session
    that can span days). The pin persists across napari restarts.

    Args:
        viewer: Active napari viewer (unused; kept for the magicgui binding).
    """
    plate_id = getattr(omero_data, "plate_id", None)
    if plate_id is None:
        notifications.show_warning("No plate loaded.")
        return
    pin_plate(int(plate_id))
    logger.info(f"Pinned plate {plate_id}")
    notifications.show_info(
        f"Pinned plate {plate_id}. Pinned plates: {sorted(pinned_plate_ids())}."
    )


@magic_factory(call_button="Unpin plate (done curating)")
def unpin_plate_widget(viewer: Viewer) -> None:
    """Release the pin on the loaded plate, so it can be evicted again.

    Press this when you have finished curating in Mastodon. Also reports which
    plates are currently pinned.

    Args:
        viewer: Active napari viewer (unused; kept for the magicgui binding).
    """
    plate_id = getattr(omero_data, "plate_id", None)
    if plate_id is not None:
        unpin_plate(int(plate_id))
        logger.info(f"Unpinned plate {plate_id}")

    still_pinned = sorted(pinned_plate_ids())
    if still_pinned:
        notifications.show_info(
            f"Unpinned plate {plate_id}. Still pinned: {still_pinned}."
        )
    else:
        notifications.show_info(
            f"Unpinned plate {plate_id}. No plates are pinned now."
        )


def tracks_gui_widget() -> Container:  # type: ignore[type-arg]
    """Stack the Tracks, per-track CSV, Mastodon-export and pin/unpin widgets."""
    from omero_screen_napari._logging import init_plugin_logging

    init_plugin_logging()
    return Container(
        widgets=[
            tracks_widget(),
            export_track_widget(),
            mastodon_export_widget(),
            pin_plate_widget(),
            unpin_plate_widget(),
        ]
    )
