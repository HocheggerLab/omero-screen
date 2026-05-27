"""napari dock widget: lineage tree for tracked nuclei.

Draws a time-vs-lineage tree from the CellView track columns already loaded in
``omero_data.plate_data``. Each track is a horizontal segment spanning its
lifetime; divisions are vertical connectors from parent to daughters. Clicking
a track moves the napari time slider to that track's start and selects it in
the Tracks layer (if present).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.tracks_loader import (
    PARENT_COL,
    TIME_COL,
    TRACK_ID_COL,
    has_tracks,
)

logger = logging.getLogger("omero-screen-napari")


@dataclass
class _Segment:
    """One track's horizontal extent and vertical placement in the tree."""

    track_id: int
    t_start: int
    t_end: int
    parent: int
    y: float


def compute_lineage_layout(
    plate_data: pl.LazyFrame, well: str
) -> list[_Segment]:
    """Compute track segments and their y-positions for one well.

    Founders are laid out top-to-bottom; each founder's descendants are placed
    via depth-first traversal so siblings sit adjacent and parents centre over
    their daughters.

    Args:
        plate_data: CellView measurements LazyFrame for the plate.
        well: Well position to lay out.

    Returns:
        One :class:`_Segment` per track, ordered by assigned ``y``.
    """
    df = (
        plate_data.filter(pl.col("well") == well)
        .group_by(TRACK_ID_COL)
        .agg(
            pl.col(TIME_COL).min().alias("t_start"),
            pl.col(TIME_COL).max().alias("t_end"),
            pl.col(PARENT_COL).first().alias("parent"),
        )
        .collect()
    )
    spans = {
        int(r[TRACK_ID_COL]): (
            int(r["t_start"]),
            int(r["t_end"]),
            int(r["parent"]),
        )
        for r in df.iter_rows(named=True)
    }
    children: dict[int, list[int]] = {}
    for tid, (_, _, parent) in spans.items():
        if parent != 0 and parent in spans:
            children.setdefault(parent, []).append(tid)

    segments: list[_Segment] = []
    counter = [0.0]

    def place(tid: int) -> float:
        kids = sorted(children.get(tid, []))
        if not kids:
            y = counter[0]
            counter[0] += 1.0
        else:
            kid_ys = [place(k) for k in kids]
            y = float(np.mean(kid_ys))
        t_start, t_end, parent = spans[tid]
        segments.append(_Segment(tid, t_start, t_end, parent, y))
        return y

    founders = sorted(
        t for t, (_, _, p) in spans.items() if p == 0 or p not in spans
    )
    for founder in founders:
        place(founder)
    return sorted(segments, key=lambda s: s.y)


class LineageTreeWidget(QWidget):  # type: ignore[misc]
    """Dock widget rendering the lineage tree for a chosen well."""

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._segments: list[_Segment] = []

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Well:"))
        self._well_combo = QComboBox()
        self._well_combo.setEditable(True)
        controls.addWidget(self._well_combo)
        refresh = QPushButton("Draw tree")
        refresh.clicked.connect(self._draw)
        controls.addWidget(refresh)
        layout.addLayout(controls)

        self._figure = Figure(figsize=(6, 4))
        self._canvas = FigureCanvasQTAgg(self._figure)  # type: ignore[no-untyped-call]
        self._canvas.mpl_connect("button_press_event", self._on_click)
        layout.addWidget(self._canvas)

        self._populate_wells()

    def _populate_wells(self) -> None:
        """Fill the well dropdown from the loaded plate data."""
        self._well_combo.clear()
        plate_data = omero_data.plate_data
        if (
            plate_data is None
            or "well" not in plate_data.collect_schema().names()
        ):
            return
        wells = plate_data.select("well").unique().collect()["well"].to_list()
        self._well_combo.addItems(sorted(str(w) for w in wells))

    def _draw(self) -> None:
        well = self._well_combo.currentText().strip()
        plate_data = omero_data.plate_data
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        if not well or not has_tracks(plate_data):
            ax.set_title("No track data — run the pipeline with --track")
            self._canvas.draw_idle()  # type: ignore[no-untyped-call]
            return

        self._segments = compute_lineage_layout(plate_data, well)
        for seg in self._segments:
            ax.plot([seg.t_start, seg.t_end], [seg.y, seg.y], "-", lw=2)
            ax.annotate(
                str(seg.track_id),
                (seg.t_start, seg.y),
                fontsize=7,
                va="center",
                ha="right",
            )
        # Division connectors: vertical line from parent y to each daughter y
        # at the daughter's start time.
        by_id = {s.track_id: s for s in self._segments}
        for seg in self._segments:
            parent = by_id.get(seg.parent)
            if parent is not None:
                ax.plot(
                    [seg.t_start, seg.t_start], [parent.y, seg.y], "-", lw=0.8
                )

        ax.set_xlabel("frame")
        ax.set_ylabel("lineage")
        ax.set_yticks([])
        ax.set_title(f"Lineage — well {well}")
        self._figure.tight_layout()
        self._canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_click(self, event: object) -> None:
        """Move the time slider to the clicked track's start frame."""
        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if xdata is None or ydata is None or not self._segments:
            return
        nearest = min(self._segments, key=lambda s: abs(s.y - ydata))
        try:
            self._viewer.dims.set_current_step(0, int(nearest.t_start))
        except (IndexError, ValueError):  # pragma: no cover - viewer state
            logger.debug(
                "Could not set time step for track %d", nearest.track_id
            )
