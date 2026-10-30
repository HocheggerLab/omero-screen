"""Track view: a TrackScheme-like lineage overview of a well, docked in napari.

Every curated track of the well is a column, time runs down (tick labels in
hours), daughters fork from their mother at the division. One node per
frame with a mask; a frame without one is a dashed edge. The layout and
colours come from :mod:`omero_screen_napari.review.track_view_model`; this
module only draws them with pyqtgraph.

Opened from the Track Review widget (**Track view** button), which passes a
loader for the session's wells, opens the cell of a clicked node and tells the
view which cell is open (highlighted). The view rebuilds itself when the
review directory's ``edits.jsonl`` or a ``*decisions.json`` changes, whoever
wrote it.

Level of detail follows the zoom: zoomed out each track is a line in the
colours of the mode; zoomed in, nodes become markers; further in, tracks get
labels (review id, length, outcome, verdict, who reviewed it, audit).

Outcomes from the fate walker and HMM phases take seconds per well, so they
are computed in a child process (not in a thread of napari: the HMM's loops
would hold the GIL and stall the session being reviewed), a moment after the
last rebuild, so a burst of edits costs one run. Until they arrive (and while they
are recomputed after an edit) the previous ones, or the walker's per-frame
G1/S/G2 calls, are shown.
"""

from __future__ import annotations

import contextlib
import html
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import QFileSystemWatcher, QPointF, Qt, QTimer
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.review import track_view_model as tvm

BACKGROUND = "#262930"  # napari's dark theme
FOREGROUND = "#c8c8c8"
DIVISION_PEN = "#e8e8e8"
GAP_PEN = "#9a9a9a"

#: Pixels per column from which nodes are drawn as markers with an outline.
NODE_PX = 7.0
#: Pixels per column from which tracks get a short / full label.
LABEL_PX = 40.0
FULL_LABEL_PX = 110.0
#: Most labels drawn at once.
MAX_LABELS = 120
#: Pick radius for click and hover, pixels.
PICK_PX = 10.0

HELP = """<b>Track view</b><br>
One column per curated track, time running down; daughters fork from
their mother at a division (reviewer or fate walker); dotted: a tracker
parent link that is not a division; dashed: frames without a mask.<br><br>
<b>Mouse</b><br>
wheel: zoom &nbsp;·&nbsp; drag: pan &nbsp;·&nbsp;
right-drag: zoom x and y separately<br>
click a node: open that cell in Track Review at that frame (the review
session's wells; in other wells the viewer only moves there, if that well
is loaded)<br>
hover a node: frame, label, area, PIP, geminin, phase<br>
<b>Fit</b>: show everything again<br><br>
<b>Zoom</b>: zoomed out, a track is a line in the colours of the mode;
zoomed in, nodes become markers, then tracks get labels.<br><br>
<b>Well / review</b>: any tracked well of the plate, read from the review
directory found for it (or picked here, or CellView only).<br>
<b>Colour</b>: phase (HMM with the plate-pooled reference in the analysis
window, else the walker's G1/S/G2),
the track's verdict or outcome, frame flags (reviewer mask, a link or
swap a reviewer confirmed, another edit, first frame after a gap), or edit
density (edits per hour of track).<br>
<b>Align</b>: each lineage tree is shifted so its first track with the
event has it at 0 h; trees without the event are hidden, or parked on the
right with <i>Unaligned</i> on.<br>
<b>Filters</b>: <i>accepted</i> = verdict accept or correct. A kept cell
brings its daughters if they are long enough."""


class HourAxis(pg.AxisItem):
    """Time axis in hours over a frame coordinate, at round hour steps."""

    def __init__(self, interval_minutes: float = 20.0) -> None:
        super().__init__("left")
        self.interval_minutes = interval_minutes
        self.relative = False

    def _frames(self, hours: float) -> float:
        return hours * 60 / self.interval_minutes

    def tickSpacing(  # noqa: N802 (pyqtgraph API)
        self,
        minVal: float,
        maxVal: float,
        size: float,  # noqa: N803
    ) -> list[tuple[float, float]]:
        """Major and minor ticks at round numbers of hours."""
        span_h = abs(maxVal - minVal) * self.interval_minutes / 60
        if span_h <= 0 or size <= 0:
            return super().tickSpacing(minVal, maxVal, size)
        want = max(2.0, size / 60)  # a major tick every ~60 px
        steps = (0.25, 0.5, 1, 2, 4, 6, 12, 24, 48, 96)
        major = next((s for s in steps if span_h / s <= want), steps[-1])
        minor = major / 4 if major >= 1 else major / 2
        return [(self._frames(major), 0), (self._frames(minor), 0)]

    def tickStrings(  # noqa: N802 (pyqtgraph API)
        self,
        values: list[float],
        scale: float,
        spacing: float,
    ) -> list[str]:
        """Hours, signed when aligned on an event."""
        out = []
        for v in values:
            h = v * self.interval_minutes / 60
            text = f"{h:+g} h" if self.relative and h else f"{h:g} h"
            out.append(text)
        return out


class TrackViewWidget(QWidget):
    """Lineage overview of one well at a time.

    Args:
        wells: Wells offered in the well box.
        loader: Reads a well (:func:`~track_view_model.session_loader`);
            called on the UI thread.
        slow: Computes walker outcomes and HMM phases from a well's inputs
            (default :func:`~track_view_model.run_slow_annotations_subprocess`,
            which runs :func:`~track_view_model.slow_annotations` in a
            child process); ``None`` disables it.
        on_pick: Called with ``(well, frame, label)`` when a node of a well
            the review session edits is clicked: opens the cell there.
        on_navigate: Called with ``(well, frame, y, x)`` when a node of
            any other well is clicked; returns whether the viewer could go
            there (that well's images are loaded). Never opens the cell in
            the review session, whose edit log is another well's.
        watch: Files or directories whose change rebuilds the view, if the
            loader does not say (a
            :class:`~track_view_model.PlateLoader` names them per well).
        well: Well to show first.
        worker: Where ``slow`` is called: ``thread`` (a worker thread, which
            for the default ``slow`` only waits on the child process) or
            ``inline`` (synchronously, for tests).
        slow_delay_ms: Wait this long after the last rebuild before starting
            ``slow``.
    """

    def __init__(
        self,
        wells: Sequence[str],
        loader: Callable[[str], tvm.WellInputs],
        slow: Callable[[tvm.WellInputs], tvm.SlowAnnotations]
        | None = tvm.run_slow_annotations_subprocess,
        on_pick: Callable[[str, int, int], Any] | None = None,
        on_navigate: Callable[[str, int, float, float], bool] | None = None,
        watch: Sequence[Path] = (),
        well: str | None = None,
        worker: str = "thread",
        slow_delay_ms: int = 1500,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.loader = loader
        self.slow_fn = slow
        self.on_pick = on_pick
        self.on_navigate = on_navigate
        self.worker = worker
        self.table: tvm.LineageTable | None = None
        self.lay: tvm.Layout | None = None
        self.inputs: tvm.WellInputs | None = None
        self._slow: dict[str, tvm.SlowAnnotations] = {}
        self._generation = 0
        self._future: Future[tvm.SlowAnnotations] | None = None
        self._future_meta: tuple[int, str] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._highlight: tuple[str, int, int | None] | None = None
        self._labels: list[pg.TextItem] = []
        self._node_xy = np.zeros((0, 2))
        self._lod: tuple[bool, int] | None = None

        # -- controls --------------------------------------------------------
        self.well_box = QComboBox()
        self.well_box.setToolTip("Well (any tracked well of the plate)")
        self.dir_box = QComboBox()
        self.dir_box.setToolTip(
            "Review directory the well's curation and verdicts are read from"
        )
        self.dir_box.setMaximumWidth(190)
        self.colour_box = QComboBox()
        self.colour_box.addItems(list(tvm.COLOUR_MODES))
        self.align_box = QComboBox()
        for key, label in tvm.ALIGNMENTS.items():
            self.align_box.addItem(label, key)
        self.park = QCheckBox("Unaligned")
        self.park.setChecked(True)
        self.park.setToolTip(
            "Keep trees without the alignment event, parked on the right"
        )
        self.sort_box = QComboBox()
        self.sort_box.addItems(list(tvm.SORTS))
        self.verdict_box = QComboBox()
        self.verdict_box.addItems(list(tvm.VERDICT_FILTERS))
        self.outcome_box = QComboBox()
        self.outcome_box.addItem("all")
        self.reviewer_box = QComboBox()
        self.reviewer_box.addItems(list(tvm.REVIEWER_FILTERS))
        self.audited = QCheckBox("Audited")
        self.flagged = QCheckBox("Flagged")
        self.flagged.setToolTip(
            "Only cells the queue builder or the fate walker flagged"
        )
        self.min_len = QSpinBox()
        self.min_len.setRange(1, 10000)
        self.min_len.setValue(1)
        self.min_len.setPrefix("≥ ")
        self.min_len.setSuffix(" fr")
        self.min_len.setToolTip("Minimum frames with a mask")
        fit = QPushButton("Fit")
        fit.clicked.connect(self.fit)
        refresh = QPushButton("↻")
        refresh.setToolTip("Re-read the well (edits, verdicts)")
        refresh.clicked.connect(lambda: self.reload(keep_view=True))
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(28)
        help_btn.clicked.connect(self.show_help)
        self.legend = QLabel("")
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        self.status = QLabel("")
        self.status.setTextFormat(Qt.TextFormat.RichText)
        self.status.setWordWrap(True)

        # -- plot ------------------------------------------------------------
        self.axis = HourAxis()
        self.plot = pg.PlotWidget(
            axisItems={"left": self.axis}, background=BACKGROUND
        )
        self.plot.setMinimumHeight(200)
        item = self.plot.getPlotItem()
        item.hideAxis("bottom")
        item.setMenuEnabled(False)
        item.hideButtons()
        self.axis.setPen(FOREGROUND)
        self.axis.setTextPen(FOREGROUND)
        self.vb = item.getViewBox()
        self.vb.invertY(True)
        self.vb.setDefaultPadding(0.02)
        self.edge_items: dict[str, pg.PlotDataItem] = {}
        self.gap_item = pg.PlotDataItem(
            pen=pg.mkPen(GAP_PEN, width=1, style=Qt.PenStyle.DashLine),
            connect="pairs",
        )
        self.div_item = pg.PlotDataItem(
            pen=pg.mkPen(DIVISION_PEN, width=1), connect="pairs"
        )
        self.split_item = pg.PlotDataItem(
            pen=pg.mkPen(
                (150, 150, 150, 110), width=1, style=Qt.PenStyle.DotLine
            ),
            connect="pairs",
        )
        self.track_hl = pg.PlotDataItem(
            pen=pg.mkPen((255, 230, 0, 90), width=12)
        )
        self.nodes = pg.ScatterPlotItem(pxMode=True)
        self.node_hl = pg.ScatterPlotItem(
            pxMode=True,
            symbol="o",
            size=18,
            brush=None,
            pen=pg.mkPen((255, 230, 0), width=2.5),
        )
        self.zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen((255, 255, 255, 70), width=1),
        )
        self.park_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen((255, 255, 255, 70), style=Qt.PenStyle.DashLine),
            label="no event",
            labelOpts={"position": 0.02, "color": FOREGROUND},
        )
        for it in (
            self.track_hl,
            self.gap_item,
            self.div_item,
            self.split_item,
            self.zero_line,
            self.park_line,
        ):
            self.plot.addItem(it)
        self.plot.addItem(self.nodes)
        self.plot.addItem(self.node_hl)
        self.nodes.setZValue(10)
        self.node_hl.setZValue(20)
        self.track_hl.setZValue(-10)

        scene: Any = self.plot.scene()
        scene.sigMouseClicked.connect(self._on_click)
        scene.sigMouseMoved.connect(self._on_move)
        self._lod_timer = QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.setInterval(40)
        self._lod_timer.timeout.connect(self._update_lod)
        self.vb.sigRangeChanged.connect(lambda *_: self._lod_timer.start())
        self._slow_timer = QTimer(self)
        self._slow_timer.setSingleShot(True)
        self._slow_timer.setInterval(slow_delay_ms)
        self._slow_timer.timeout.connect(self._submit_slow)
        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._check_slow)
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(400)
        self._reload_timer.timeout.connect(lambda: self.reload(keep_view=True))
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._on_watched)
        self.watcher.directoryChanged.connect(self._on_watched)
        self._watch: list[Path] = []
        self._static_watch: list[Path] = []

        # -- layout ----------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        for btn in (fit, refresh):
            btn.setFixedWidth(44)
        self.outcome_box.setMaximumWidth(170)
        rows = [
            [
                self.well_box,
                _labelled("review", self.dir_box),
                self.park,
            ],
            [
                _labelled("colour", self.colour_box),
                _labelled("align", self.align_box),
            ],
            [
                _labelled("sort", self.sort_box),
                _labelled("verdict", self.verdict_box),
                _labelled("outcome", self.outcome_box),
            ],
            [
                _labelled("by", self.reviewer_box),
                self.audited,
                self.flagged,
                self.min_len,
                fit,
                refresh,
                help_btn,
            ],
        ]
        for row in rows:
            h = QHBoxLayout()
            for w in row:
                if isinstance(w, QWidget):
                    h.addWidget(w)
                else:
                    h.addLayout(w)
            h.addStretch(1)
            layout.addLayout(h)
        layout.addWidget(self.legend)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.status)

        self.set_source(wells, loader, watch=watch, well=well)
        self.well_box.currentTextChanged.connect(self._on_well)
        self.dir_box.currentIndexChanged.connect(self._on_dir)
        self.colour_box.currentTextChanged.connect(lambda _: self.redraw())
        for box in (
            self.align_box,
            self.sort_box,
            self.verdict_box,
            self.outcome_box,
            self.reviewer_box,
        ):
            box.currentIndexChanged.connect(
                lambda _: self.relayout(keep_view=False)
            )
        for check in (self.park, self.audited, self.flagged):
            check.toggled.connect(lambda _: self.relayout(keep_view=False))
        self.min_len.valueChanged.connect(
            lambda _: self.relayout(keep_view=False)
        )

    # -- source ------------------------------------------------------------------

    def set_source(
        self,
        wells: Sequence[str],
        loader: Callable[[str], tvm.WellInputs],
        watch: Sequence[Path] = (),
        well: str | None = None,
    ) -> None:
        """Show another session's wells (drops cached annotations)."""
        self.loader = loader
        self._slow.clear()
        self._static_watch = [Path(p) for p in watch]
        self.well_box.blockSignals(True)
        self.well_box.clear()
        self.well_box.addItems(list(wells))
        if well and well in wells:
            self.well_box.setCurrentText(well)
        self.well_box.blockSignals(False)
        self._fill_dirs()
        self._rewatch()
        if wells:
            self.reload(keep_view=False)

    def _on_well(self, _well: str) -> None:
        self._fill_dirs()
        self._rewatch()
        self.reload(keep_view=False)

    def _fill_dirs(self) -> None:
        """Offer the review directories found for the well (and none)."""
        self.dir_box.blockSignals(True)
        self.dir_box.clear()
        candidates = getattr(self.loader, "candidates", None)
        chosen = getattr(self.loader, "review_dir", None)
        if candidates is None or chosen is None:
            self.dir_box.addItem("(as loaded)", None)
            self.dir_box.setEnabled(False)
        else:
            current = chosen(self.well)
            paths = list(candidates(self.well))
            if current is not None and current not in paths:
                paths.insert(0, current)
            for path in paths:
                self.dir_box.addItem(path.name, str(path))
            self.dir_box.addItem("CellView only", None)
            index = paths.index(current) if current in paths else len(paths)
            self.dir_box.setCurrentIndex(index)
            self.dir_box.setEnabled(True)
        self.dir_box.blockSignals(False)

    def _on_dir(self, _index: int) -> None:
        setter = getattr(self.loader, "set_review_dir", None)
        if setter is None:
            return
        data = self.dir_box.currentData()
        setter(self.well, Path(data) if data else None)
        self._slow.pop(self.well, None)
        self._rewatch()
        self.reload(keep_view=False)

    def _rewatch(self) -> None:
        """Watch the current well's review directory."""
        paths = self.watcher.files() + self.watcher.directories()
        if paths:
            self.watcher.removePaths(paths)
        per_well = getattr(self.loader, "watch_paths", None)
        self._watch = (
            list(per_well(self.well)) if per_well else self._static_watch
        )
        self._add_watches()

    def interactive(self, well: str | None = None) -> bool:
        """Whether a click opens the cell in the review session."""
        check = getattr(self.loader, "interactive", None)
        return True if check is None else bool(check(well or self.well))

    @property
    def well(self) -> str:
        """The well shown."""
        return self.well_box.currentText()

    def _add_watches(self) -> None:
        have = set(self.watcher.files()) | set(self.watcher.directories())
        for p in self._watch:
            if p.exists() and str(p) not in have:
                self.watcher.addPath(str(p))
        # A decisions file written later appears in the watched directory.
        for d in [p for p in self._watch if p.is_dir()]:
            for f in d.glob("*decisions.json"):
                if str(f) not in have:
                    self.watcher.addPath(str(f))
            edits = d / "edits.jsonl"
            if edits.exists() and str(edits) not in have:
                self.watcher.addPath(str(edits))

    def _on_watched(self, _path: str) -> None:
        # Atomic writes replace the file, which drops it from the watcher.
        self._add_watches()
        self._reload_timer.start()

    # -- building ----------------------------------------------------------------

    def reload(self, keep_view: bool = True) -> None:
        """Re-read the well and redraw; start the slow annotations."""
        well = self.well
        if not well:
            return
        try:
            self.inputs = self.loader(well)
        except Exception as err:
            self.table = None
            self.status.setText(
                f"<span style='color:#e06c6c'>Could not read {html.escape(well)}: "
                f"{html.escape(str(err))}</span>"
            )
            self._clear()
            return
        self._generation += 1
        self._build_table()
        self._fill_outcomes()
        self.relayout(keep_view=keep_view)
        self._start_slow()

    def _build_table(self) -> None:
        assert self.inputs is not None
        self.table = tvm.table_from_inputs(
            self.inputs, self._slow.get(self.inputs.well)
        )

    def _fill_outcomes(self) -> None:
        if self.table is None:
            return
        current = self.outcome_box.currentText()
        found = sorted(o for o in set(self.table.tracks["outcome"]) if o)
        self.outcome_box.blockSignals(True)
        self.outcome_box.clear()
        self.outcome_box.addItems(["all", *found])
        if current in found:
            self.outcome_box.setCurrentText(current)
        self.outcome_box.blockSignals(False)

    def filters(self) -> tvm.Filters:
        """The filters set in the toolbar."""
        return tvm.Filters(
            verdict=self.verdict_box.currentText(),
            outcome=self.outcome_box.currentText() or "all",
            reviewed_by=self.reviewer_box.currentText(),
            audited_only=self.audited.isChecked(),
            flagged_only=self.flagged.isChecked(),
            min_length=self.min_len.value(),
        )

    @property
    def align(self) -> str:
        """The alignment key."""
        return str(self.align_box.currentData() or "frame")

    def relayout(self, keep_view: bool = True) -> None:
        """Lay the table out again (filters, sort, alignment) and redraw."""
        if self.table is None:
            return
        self.lay = tvm.layout(
            self.table,
            align=self.align,
            sort=self.sort_box.currentText(),
            filters=self.filters(),
            park=self.park.isChecked(),
        )
        self.axis.interval_minutes = self.table.interval_minutes
        self.axis.relative = self.align != "frame"
        self.redraw()
        if not keep_view:
            self.fit()
        self._update_status()

    def _clear(self) -> None:
        for it in self.edge_items.values():
            it.setData([], [])
        for it in (
            self.gap_item,
            self.div_item,
            self.split_item,
            self.track_hl,
        ):
            it.setData([], [])
        self.nodes.clear()
        self.node_hl.clear()
        self._clear_labels()
        self._node_xy = np.zeros((0, 2))

    def redraw(self) -> None:
        """Draw nodes and edges in the current colour mode."""
        if self.lay is None or self.table is None:
            return
        lay, mode = self.lay, self.colour_box.currentText()
        nodes, edges = lay.nodes, lay.edges
        self._clear()
        self._lod = None
        palette = tvm.PALETTES[mode]
        cats = tvm.node_categories(nodes, self.table.tracks, mode)
        self._cats = cats
        self._node_xy = nodes[["x", "y"]].to_numpy(dtype=float)
        brushes = {c: pg.mkBrush(col) for c, col in palette.items()}
        self._brushes = [brushes[c] for c in cats]
        self.nodes.setData(
            x=self._node_xy[:, 0],
            y=self._node_xy[:, 1],
            brush=self._brushes,
            pen=None,
            symbol="s",
            size=3,
        )

        # Links coloured by the later node's category; gaps and divisions apart.
        cat_of = dict(
            zip(
                zip(nodes["track"], nodes["frame"], strict=True),
                cats,
                strict=True,
            )
        )
        links = edges[edges["kind"] == "link"]
        link_cats = np.array(
            [
                cat_of.get((t, f), "")
                for t, f in zip(links["track"], links["f1"], strict=True)
            ],
            dtype=object,
        )
        for cat, colour in palette.items():
            sel = links[link_cats == cat]
            item = self.edge_items.get(cat)
            if item is None:
                item = pg.PlotDataItem(connect="pairs")
                item.setZValue(1)
                self.plot.addItem(item)
                self.edge_items[cat] = item
            item.setPen(pg.mkPen(colour, width=2))
            item.setData(*_pairs(sel))
        for cat in set(self.edge_items) - set(palette):
            self.edge_items[cat].setData([], [])
        self.gap_item.setData(*_pairs(edges[edges["kind"] == "gap"]))
        self.div_item.setData(*_pairs(edges[edges["kind"] == "division"]))
        self.split_item.setData(*_pairs(edges[edges["kind"] == "split"]))

        self.zero_line.setVisible(self.align != "frame")
        if lay.parked_from is not None:
            self.park_line.setPos(lay.parked_from - 0.5)
            self.park_line.setVisible(True)
        else:
            self.park_line.setVisible(False)
        self._draw_legend(mode, cats)
        self._draw_highlight()
        self._update_lod()

    def _draw_legend(self, mode: str, cats: np.ndarray) -> None:
        present = set(cats.tolist())
        parts = []
        for cat, colour in tvm.PALETTES[mode].items():
            if cat not in present:
                continue
            name = cat or "none"
            parts.append(
                f"<span style='color:{colour}'>■</span>&nbsp;{html.escape(name)}"
            )
        note = ""
        if mode == "phase" and self.table is not None:
            t0, t1 = self.table.window
            hmm = (
                "HMM, plate-pooled"
                if self.table.pooled
                else "HMM, this well only"
            )
            note = (
                f" &nbsp;<i>({hmm}, t{t0}–t{t1}; walker G1/S/G2 elsewhere)</i>"
                if self.table.has_hmm
                else " &nbsp;<i>(walker G1/S/G2; HMM pending)</i>"
                if self.slow_fn is not None
                else " &nbsp;<i>(walker G1/S/G2)</i>"
            )
        self.legend.setText(" &nbsp; ".join(parts) + note)

    def fit(self) -> None:
        """Show every track."""
        if self.lay is None or self.lay.nodes.empty:
            return
        x = self.lay.nodes["x"]
        y = self.lay.nodes["y"]
        self.vb.setRange(
            xRange=(float(x.min()) - 1, float(x.max()) + 1),
            yRange=(float(y.min()) - 2, float(y.max()) + 2),
            padding=0.02,
        )
        self._update_lod()

    def zoom_to(self, x0: float, x1: float, y0: float, y1: float) -> None:
        """Show a region (columns ``x0``–``x1``, rows ``y0``–``y1``)."""
        self.vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        self._update_lod()

    # -- level of detail ---------------------------------------------------------

    def _pixels(self) -> tuple[float, float]:
        """Pixels per column and per frame."""
        sx, sy = self.vb.viewPixelSize()
        return (1 / sx if sx else 1.0), (1 / sy if sy else 1.0)

    def _update_lod(self) -> None:
        if self.lay is None or not len(self._node_xy):
            return
        px_col, px_frame = self._pixels()
        markers = px_col >= NODE_PX and px_frame >= 3
        size = int(
            np.clip(
                min(px_col * 0.55, px_frame * 0.85), 3 if markers else 2, 12
            )
        )
        if (markers, size) != self._lod:
            self._lod = (markers, size)
            if markers:
                self.nodes.setData(
                    x=self._node_xy[:, 0],
                    y=self._node_xy[:, 1],
                    brush=self._brushes,
                    pen=pg.mkPen((20, 20, 20), width=0.8),
                    symbol="o",
                    size=size,
                )
            else:
                self.nodes.setData(
                    x=self._node_xy[:, 0],
                    y=self._node_xy[:, 1],
                    brush=self._brushes,
                    pen=None,
                    symbol="s",
                    size=size,
                )
            width = 1.5 if markers else float(np.clip(px_col * 0.5, 1, 4))
            palette = tvm.PALETTES[self.colour_box.currentText()]
            for cat, item in self.edge_items.items():
                if cat in palette:
                    item.setPen(pg.mkPen(palette[cat], width=width))
        self._draw_labels(px_col)

    def _clear_labels(self) -> None:
        for t in self._labels:
            self.plot.removeItem(t)
        self._labels = []

    def _draw_labels(self, px_col: float) -> None:
        self._clear_labels()
        if self.lay is None or px_col < LABEL_PX:
            return
        (x0, x1), (y0, y1) = self.vb.viewRange()
        tr = self.lay.tracks
        visible = tr[
            (tr["x"] >= x0)
            & (tr["x"] <= x1)
            & (tr["y1"] >= y0)
            & (tr["y0"] <= y1)
        ]
        full = px_col >= FULL_LABEL_PX
        assert self.table is not None
        shown = visible.head(MAX_LABELS)
        for tid, (_, row) in zip(
            shown.index.to_numpy().tolist(), shown.iterrows(), strict=True
        ):
            text = track_label(
                tid, row, self.table.interval_minutes, full=full
            )
            # Above the track's first node; inside the top edge if the
            # track starts above the view.
            start = float(row["y0"])
            inside = start - 0.6 < y0
            item = pg.TextItem(
                text,
                color=FOREGROUND,
                anchor=(0.5, 0.0 if inside else 1.0),
                fill=pg.mkBrush(38, 41, 48, 200),
            )
            item.setPos(float(row["x"]), y0 if inside else start - 0.6)
            item.setZValue(30)
            self.plot.addItem(item)
            self._labels.append(item)

    # -- picking -----------------------------------------------------------------

    def node_at(self, x: float, y: float) -> int | None:
        """Index (into the layout's nodes) of the node nearest a view point, if close."""
        if not len(self._node_xy):
            return None
        sx, sy = self.vb.viewPixelSize()
        dx = (self._node_xy[:, 0] - x) / (sx or 1)
        dy = (self._node_xy[:, 1] - y) / (sy or 1)
        d2 = dx * dx + dy * dy
        i = int(np.argmin(d2))
        return i if d2[i] <= PICK_PX**2 else None

    def _on_click(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = self.vb.mapSceneToView(event.scenePos())
        i = self.node_at(pos.x(), pos.y())
        if i is not None:
            self.pick(i)

    def pick(self, i: int) -> None:
        """Select node ``i`` of the layout and tell the review widget.

        In a well the review session edits, the cell opens there. In any
        other well the viewer only moves to the cell (if that well's images
        are loaded); the cell is described here instead.
        """
        assert self.lay is not None
        row = self.lay.nodes.iloc[i]
        track, frame, label = (
            int(row["track"]),
            int(row["frame"]),
            int(row["label"]),
        )
        self.highlight(self.well, track, frame)
        try:
            if self.interactive():
                if self.on_pick is not None:
                    self.on_pick(self.well, frame, label)
                return
            moved = bool(
                self.on_navigate is not None
                and self.on_navigate(
                    self.well, frame, float(row["y"]), float(row["x"])
                )
            )
        except Exception as err:
            self.status.setText(
                f"<span style='color:#e06c6c'>{html.escape(str(err))}</span>"
            )
            return
        where = (
            "viewer moved there"
            if moved
            else f"load well {html.escape(self.well)} in napari to go there"
        )
        self.status.setText(
            f"<b>Read-only well</b> (not the review session's; {where}).<br>"
            + self.describe(i)
        )

    def _on_move(self, pos: QPointF) -> None:
        if self.lay is None:
            return
        p = self.vb.mapSceneToView(pos)
        i = self.node_at(p.x(), p.y())
        if i is None:
            QToolTip.hideText()
            return
        text = self.describe(i)
        QToolTip.showText(
            self.plot.mapToGlobal(self.plot.mapFromScene(pos)),
            text,
            self.plot,
        )

    def describe(self, i: int) -> str:
        """Hover text of node ``i``."""
        assert self.lay is not None and self.table is not None
        n = self.lay.nodes.iloc[i]
        tr = self.table.tracks.loc[int(n["track"])]
        h = self.table.hours(float(n["frame"]))
        parts = [
            f"<b>t{int(n['frame'])}</b> ({h:.1f} h) · label {int(n['label'])}"
            + (" (reviewer mask)" if n["is_extra"] else ""),
            f"track {int(n['track'])}"
            + (f" · {tr['item_id']}" if tr["item_id"] else ""),
            f"area {_num(n['area'])} · PIP {_num(n['pip'])} · "
            f"geminin {_num(n['geminin'])}",
            f"phase {n['phase'] or '–'}"
            + (
                f" (walker {n['phase_walker']})"
                if self.table.has_hmm and n["phase_walker"] != n["phase"]
                else ""
            ),
        ]
        flags = [
            name
            for name, col in (
                ("edited", "edited"),
                ("confirmed switch", "confirmed"),
                ("after gap", "after_gap"),
            )
            if n[col]
        ]
        if flags:
            parts.append(", ".join(flags))
        return "<br>".join(parts)

    # -- highlight ---------------------------------------------------------------

    def highlight(
        self, well: str, track: int | None, frame: int | None = None
    ) -> None:
        """Mark the cell open in the review widget (``track`` None clears)."""
        self._highlight = None if track is None else (well, track, frame)
        self._draw_highlight()

    def set_frame(self, frame: int) -> None:
        """Move the highlighted node to ``frame`` (the viewer's frame)."""
        if self._highlight is not None:
            well, track, _ = self._highlight
            self._highlight = (well, track, frame)
            self._draw_highlight()

    def _draw_highlight(self) -> None:
        self.track_hl.setData([], [])
        self.node_hl.clear()
        if self._highlight is None or self.lay is None:
            return
        well, track, frame = self._highlight
        tr = self.lay.tracks
        if well != self.well or track not in tr.index:
            return
        rec: dict[Any, Any] = tr.loc[track].to_dict()
        x = float(rec["x"])
        self.track_hl.setData([x, x], [float(rec["y0"]), float(rec["y1"])])
        if frame is not None:
            n = self.lay.nodes
            hit = n[(n["track"] == track) & (n["frame"] == frame)]
            if len(hit):
                self.node_hl.setData(
                    x=[float(hit["x"].iloc[0])], y=[float(hit["y"].iloc[0])]
                )

    def centre_on(self, track: int) -> None:
        """Pan (not zoom) so the track's column is in view."""
        if self.lay is None or track not in self.lay.tracks.index:
            return
        row: dict[Any, Any] = self.lay.tracks.loc[track].to_dict()
        (x0, x1), (y0, y1) = self.vb.viewRange()
        x = float(row["x"])
        if not x0 <= x <= x1:
            w = x1 - x0
            self.vb.setXRange(x - w / 2, x + w / 2, padding=0)

    # -- slow annotations --------------------------------------------------------

    def _start_slow(self) -> None:
        if self.slow_fn is None or self.inputs is None:
            return
        if self.worker == "inline":
            self._submit_slow()
        else:
            self._slow_timer.start()
            self._update_status()

    def _submit_slow(self) -> None:
        if self.slow_fn is None or self.inputs is None:
            return
        inputs, gen = self.inputs, self._generation
        if self.worker == "inline":
            self._apply_slow(gen, inputs.well, self.slow_fn(inputs))
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="track-view"
            )
        if self._future is not None:
            self._future.cancel()
        self._future = self._executor.submit(self.slow_fn, inputs)
        self._future_meta = (gen, inputs.well)
        self._poll.start()
        self._update_status()

    def _check_slow(self) -> None:
        fut = self._future
        if fut is None or not fut.done():
            return
        self._poll.stop()
        self._future = None
        meta = self._future_meta
        try:
            result = fut.result()
        except Exception as err:
            result = tvm.SlowAnnotations(errors=[str(err)])
        if meta is not None:
            self._apply_slow(meta[0], meta[1], result)

    def _apply_slow(
        self, generation: int, well: str, result: tvm.SlowAnnotations
    ) -> None:
        self._slow[well] = result
        remember = getattr(self.loader, "remember", None)
        if remember is not None:
            remember(result)
        if generation != self._generation or well != self.well:
            return  # stale: a newer rebuild has started its own
        self._build_table()
        self._fill_outcomes()
        self.relayout(keep_view=True)

    def _update_status(self) -> None:
        if self.table is None or self.lay is None:
            return
        tr = self.lay.tracks
        busy = self._slow_timer.isActive() or (
            self._future is not None and not self._future.done()
        )
        slow = self._slow.get(self.well)
        extra = ""
        if busy:
            extra = " · fitting phases and outcomes…"
        elif slow is not None and slow.errors:
            extra = " · " + html.escape("; ".join(slow.errors))
        self.status.setText(
            f"{self.well}: {len(tr)} of {len(self.table.tracks)} tracks · "
            f"{len(self.lay.nodes)} nodes · "
            f"{int(tr['divided'].sum())} divisions"
            + (
                f" · {int(tr['parked'].sum())} unaligned"
                if tr["parked"].any()
                else ""
            )
            + extra
            + (
                ""
                if self.interactive()
                else " · <i>read-only: not the review session's well</i>"
            )
        )

    # -- misc --------------------------------------------------------------------

    def show_help(self) -> None:
        """Mouse and display help."""
        box = QMessageBox(self)
        box.setWindowTitle("Track view")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(HELP)
        box.show()

    def wait_for_annotations(self, timeout: float = 120.0) -> None:
        """Block until the slow annotations of the current build are in (tests, scripts)."""
        if self._slow_timer.isActive():
            self._slow_timer.stop()
            self._submit_slow()
        while self._future is not None:
            fut = self._future
            with contextlib.suppress(Exception):  # _check_slow reports it
                fut.result(timeout=timeout)
            self._check_slow()
            if self._future is fut:
                break

    def closeEvent(self, a0: QCloseEvent | None) -> None:  # noqa: N802 (Qt API)
        """Stop the worker with the widget."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(a0)


def track_label(
    track: int, row: pd.Series, interval_minutes: float, full: bool = False
) -> str:
    """Label of a track column: id, and with ``full`` length, outcome, review."""
    name = str(row["item_id"] or f"#{track}")
    if not full:
        return name
    hours = row["n_frames"] * interval_minutes / 60
    lines = [name, f"{int(row['n_frames'])} fr / {hours:.1f} h"]
    if row["outcome"]:
        lines.append(str(row["outcome"]).replace("_", " "))
    review = " ".join(
        str(v) for v in (row["verdict"], row["reviewed_by"]) if v
    )
    if review:
        lines.append(review)
    if row["audit"]:
        lines.append(f"audit {row['audit']}")
    return "\n".join(lines)


def _pairs(edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Edges as ``connect='pairs'`` coordinate arrays."""
    if edges.empty:
        return np.zeros(0), np.zeros(0)
    x = np.column_stack([edges["x0"], edges["x1"]]).ravel()
    y = np.column_stack([edges["y0"], edges["y1"]]).ravel()
    return x.astype(float), y.astype(float)


def _num(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(f) else f"{f:.0f}"


def _labelled(text: str, widget: QWidget) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setSpacing(3)
    h.addWidget(QLabel(text))
    h.addWidget(widget)
    return h
