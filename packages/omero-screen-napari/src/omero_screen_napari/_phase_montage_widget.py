"""Napari widget wrapping the per-well cell-cycle montage export.

Thin by design: every decision lives in :mod:`omero_screen_napari.phase_montage`
so the widget and ``bin/phase_montage.py`` cannot drift. This exists to check a
single well before committing to a plate-wide batch run.

One caveat worth stating in the UI: re-rolling the seed until the panels look
good is cherry-picking by the back door. The seed is stamped on every figure so
whichever draw you keep is at least reproducible and declared.
"""

from pathlib import Path

from loguru import logger
from magicgui import magic_factory
from magicgui.widgets import Container
from napari.utils import notifications
from napari.utils import progress as napari_progress

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.phase_montage import (
    DEFAULT_PHASES,
    MontageConfig,
    MontageError,
    export_plate_pdfs,
    load_plate_measurements,
    plate_wells,
)

#: Evaluated once at import: magicgui needs a concrete default, and a call
#: in the signature is re-evaluated on every widget construction.
_DEFAULT_OUT = Path.home() / "phase_montages"


def phase_montage_widget_gui() -> Container:  # type: ignore[type-arg]
    """Assemble the montage widget for the napari plugin manifest."""
    from omero_screen_napari._logging import init_plugin_logging

    init_plugin_logging()
    return Container(widgets=[phase_montage_widget()])


@magic_factory(
    call_button="Export montage",
    output_dir={"mode": "d", "label": "Output folder"},
    plate_id={"label": "Plate ID (blank = loaded plate)"},
    well={"label": "Well (All, or C3, or blank = loaded well)"},
    cells_per_phase={"label": "Cells per phase", "min": 1, "max": 12},
    seed={"label": "Random seed", "min": 0, "max": 9999},
    include_subg1={"label": "Include Sub-G1"},
    mask={"label": "Outline", "choices": ["cells", "nuclei"]},
)
def phase_montage_widget(
    output_dir: Path = _DEFAULT_OUT,
    plate_id: str = "",
    well: str = "",
    cells_per_phase: int = 4,
    seed: int = 0,
    include_subg1: bool = False,
    mask: str = "cells",
) -> None:
    """Export a cell-cycle montage PDF for one well.

    Leaving plate and well blank uses whatever is currently loaded in the
    viewer, which is the common case: look at a well, then export it.
    """
    try:
        resolved_plate = (
            int(plate_id) if plate_id.strip() else omero_data.plate_id
        )
    except ValueError:
        notifications.show_error(f"Plate ID '{plate_id}' is not a number")
        return
    if not resolved_plate:
        notifications.show_error("No plate loaded and no plate ID given")
        return

    # "All" mirrors the gallery exporter's well box, so the two batch widgets
    # behave the same way. A comma list works too.
    raw_well = well.strip()
    export_all = raw_well.lower() == "all"
    resolved_wells: list[str] | None
    if export_all:
        resolved_wells = None
    elif raw_well:
        resolved_wells = [w.strip() for w in raw_well.split(",") if w.strip()]
    elif omero_data.well_pos_list:
        resolved_wells = [omero_data.well_pos_list[0]]
    else:
        notifications.show_error("No well loaded and no well given")
        return

    phases = ("Sub-G1", *DEFAULT_PHASES) if include_subg1 else DEFAULT_PHASES
    config = MontageConfig(
        phases=phases,
        cells_per_phase=cells_per_phase,
        seed=seed,
        mask=mask,
    )

    try:
        df = load_plate_measurements(resolved_plate)
        targets = plate_wells(df) if resolved_wells is None else resolved_wells
        # Synchronous, like the gallery exporter: pyplot figure creation is not
        # thread-safe, so a worker thread would be a correctness problem rather
        # than a responsiveness win. ~2 s per well.
        bar = napari_progress(
            total=len(targets), desc=f"Montages to {Path(output_dir).name}"
        )

        def _tick(current: str, index: int, total: int) -> None:
            bar.set_description(f"Montage {current} ({index + 1}/{total})")
            bar.update(1)

        try:
            written, failures = export_plate_pdfs(
                resolved_plate,
                df,
                Path(output_dir),
                config,
                wells=targets,
                on_progress=_tick,
            )
        finally:
            bar.close()
    except MontageError as exc:
        notifications.show_error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 — surface, never crash the viewer
        logger.opt(exception=True).error("Montage export failed")
        notifications.show_error(f"Montage export failed: {exc}")
        return

    if not written:
        notifications.show_error(
            "No montages written. " + ("; ".join(failures) or "")
        )
        return
    message = f"Wrote {len(written)} montage(s) to {Path(output_dir)}"
    if failures:
        message += f" ({len(failures)} well(s) skipped)"
        logger.warning(f"Skipped: {failures}")
    notifications.show_info(message)
