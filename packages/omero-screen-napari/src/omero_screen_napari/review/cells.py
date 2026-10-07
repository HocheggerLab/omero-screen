"""From a cell anchor to its followed crops and filmstrip, in one call.

Shared by ``omero-screen-images track``, batch rendering of review queues and
the agent's ``filmstrip`` tool. Wells are loaded and repaired once per
:class:`CellSource` and reused for every cell of that well.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def default_db_path() -> Path:
    """The CellView database of the active environment (``ENV`` / ``.env.*``).

    Importing omero-screen selects ``development`` unless ``ENV`` is set, which
    points at the test database; CLIs pass ``--env production`` or ``--db``.
    """
    from cellview.db.db import CellViewDB

    return Path(CellViewDB().db_path)


@dataclass
class CellSource:
    """Read access to one plate's curated lineages, cached per well.

    Attributes:
        plate_id: OMERO plate id.
        db_path: CellView DuckDB file (opened read-only).
        log_path: Edit log to replay on the automatic repair, if any.
        marker: Mitotic marker channel for the repair.
    """

    plate_id: int
    db_path: Path | None = None
    log_path: Path | None = None
    marker: str = "Geminin"
    hide_debris: bool = True
    _wells: dict[str, tuple[pd.DataFrame, Any]] = field(
        default_factory=dict, repr=False
    )
    _bases: dict[str, Any] = field(default_factory=dict, repr=False)
    _raw: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _debris: dict[str, set[int]] = field(default_factory=dict, repr=False)

    def raw(self, well: str) -> pd.DataFrame:
        """The well's detections as tracked, before any curation (loaded once)."""
        if well not in self._raw:
            import duckdb
            from cellview.tracks.cells import load_well

            conn = duckdb.connect(
                str(self.db_path or default_db_path()), read_only=True
            )
            try:
                det = load_well(conn, self.plate_id, well)
            finally:
                conn.close()
            if self.hide_debris:
                from cellview.tracks.debris import drop_debris

                det, self._debris[well] = drop_debris(det)
            else:
                self._debris[well] = set()
            self._raw[well] = det
        return self._raw[well]

    def debris(self, well: str) -> set[int]:
        """Raw track ids (= mask labels) hidden as debris in this well."""
        self.raw(well)
        return self._debris.get(well, set())

    def well(self, well: str) -> tuple[pd.DataFrame, Any]:
        """``(curated detections, curated lineage)`` for a well: repair + edit log."""
        if well not in self._wells:
            from cellview.tracks.cells import apply_extras
            from cellview.tracks.edit import EditLog, replay

            log: Iterable[Any] = (
                EditLog(self.log_path) if self.log_path else []
            )
            curated = replay(self.base(well), log, well)
            self._wells[well] = (
                apply_extras(self.raw(well), curated),
                curated,
            )
        return self._wells[well]

    @property
    def patch_dir(self) -> Path | None:
        """Where reviewer-made nucleus masks are stored (beside the edit log)."""
        return self.log_path.parent / "masks" if self.log_path else None

    def base(self, well: str) -> Any:
        """The well's automatically repaired lineage, before any edit."""
        from cellview.tracks.edit import base_lineage
        from cellview.tracks.repair import repair_lineage

        if well not in self._bases:
            det = self.raw(well)
            key = self.marker.lower()
            self._bases[well] = base_lineage(
                det, repair_lineage(det.rename(columns={key: "marker"}))
            )
        return self._bases[well]

    def path(
        self,
        well: str,
        anchor: tuple[int, int],
        start: int | None = None,
        stop: int | None = None,
    ) -> pd.DataFrame:
        """Per-frame path of the anchored cell, with phase calls."""
        from cellview.tracks.cells import add_phases, cell_frames

        det, curated = self.well(well)
        return add_phases(cell_frames(curated, det, anchor, start, stop), det)

    def events(
        self, well: str, anchor: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """Curated events of the anchored cell."""
        _, curated = self.well(well)
        notes = curated.annotations.get(curated.resolve(anchor), {})
        return list(notes.get("events", []))


def parse_anchor(text: str) -> tuple[int, int]:
    """Parse ``FRAME:LABEL`` or a review id ``WELL-tFRAME-LLABEL`` into an anchor."""
    if "-t" in text and "-L" in text:
        _, t, lab = text.split("-")
        return int(t[1:]), int(lab[1:])
    frame, label = text.split(":")
    return int(frame), int(label)


def render_cell(
    source: CellSource,
    well: str,
    anchor: tuple[int, int],
    channels: list[str] | None = None,
    size: int = 160,
    max_tiles: int = 24,
    start: int | None = None,
    stop: int | None = None,
    title: str | None = None,
) -> Any:
    """Filmstrip figure for one cell."""
    from omero_screen_napari.review.filmstrip import render_filmstrip
    from omero_screen_napari.review.follow import follow_from_cache

    path = source.path(well, anchor, start, stop)
    if path.empty:
        raise ValueError(
            f"No frames for cell {anchor} in well {well} in that range."
        )
    from omero_screen_napari.review.masks import paint_patches

    _, curated = source.well(well)
    base_dir = source.patch_dir.parent if source.patch_dir else None
    followed = follow_from_cache(
        source.plate_id,
        well,
        path,
        channels=channels,
        size=size,
        patch_fn=lambda t, y0, x0, crop: paint_patches(
            crop, t, y0, x0, curated, base_dir
        ),
    )
    label = (
        title
        or f"plate {source.plate_id} {well}  cell t{anchor[0]}:L{anchor[1]}  (track {int(path['track_id'].iloc[0])})"
    )
    return render_filmstrip(
        followed,
        title=label,
        max_tiles=max_tiles,
        events=source.events(well, anchor),
    )
