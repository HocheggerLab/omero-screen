"""Track Review widget (v3): check and curate tracked cells, with an agent.

The dock widget holds only what is needed for the cell on screen: a search
field, navigation, the verdict and a note. Two windows open from it:

* **Queue**: the cells to review, with why each was flagged and what the
  agent and the reviewer decided; double-click a row to go to the cell;
* **Repair**: the edits (link, unlink, add / remove / split / draw a nucleus,
  events, exclude), the agent's proposals and the cell's edit log.

Search takes ``85 327`` (frame, then the nucleus label shown in the viewer or
a curated track id), ``327`` (at the current frame) or a review id
``C2-t72-L327``. A cell found by search that is not in the queue opens as an
ad-hoc item; it can be repaired and given a verdict like any other.

Every repair is staged first: the Repair window says what it would change
(frames gained or lost, from or to which track, daughters, mother) and the
viewer draws the cell's path after the change in orange. Nothing is written
until **Confirm**. Edits go to ``edits.jsonl`` next to the queue and
verdicts to ``decisions.json``.

Layers:

* ``review past`` / ``review ahead``: the cell's track with its mother and
  daughters, *N* frames back (cyan) and ahead (magenta) of the current frame;
* ``review marker``: a ring on the cell, a cross where it has no mask;
* ``review cell``: an outline of the cell's own nucleus;
* ``review candidates``: numbered nuclei that could continue the cell;
* ``review proposal``: where a pending agent proposal points;
* ``review preview``: the cell's path after a staged change;
* ``review separation``: a staged separation, the cell (1) and the merged
  object (2) in every frame of the range.

**Separate by reporter** (Repair window) splits the cell from a
reporter-negative object merged into its mask, such as a dead cell it has
engulfed, over a range of frames: pixels with the cell's own PIP or geminin
level stay with the cell, the rest becomes a separate object. The preview
lists every frame; frames where the cell's part is much smaller than the cell
was before the merge are flagged, as the cell may have left the mask there.

Keyboard (napari canvas focused):

===========  ==========================================
Shift-A/C    accept / correct
Shift-R/U    reject / unsure
Shift-N      next break of this cell
Shift-F      toggle follow cell
Shift-L      link: click the nucleus that is this cell
1…9          link to numbered candidate
Shift-X      unlink at current frame
Shift-Enter  confirm the staged change
Shift-Z      undo last edit for this well
===========  ==========================================
"""

import os
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from napari.utils import notifications
from napari.utils.colormaps import Colormap
from napari.viewer import Viewer
from qtpy.QtCore import QFileSystemWatcher, QSettings, Qt
from qtpy.QtGui import QKeySequence, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.layer_sync import hold_until_loaded
from omero_screen_napari.review.session import (
    ReviewSession,
    set_active_session,
)
from omero_screen_napari.review_queue import (
    QueueError,
    ReviewItem,
    decisions_path,
    read_decisions,
)

PAST_LAYER = "review past"
AHEAD_LAYER = "review ahead"
MARKER_LAYER = "review marker"
CELL_LAYER = "review cell"
CAND_LAYER = "review candidates"
PROPOSAL_LAYER = "review proposal"
PREVIEW_LAYER = "review preview"
SEPARATION_LAYER = "review separation"
DRAW_LAYER = "review draw"
REVIEW_LAYERS = (
    PAST_LAYER,
    AHEAD_LAYER,
    MARKER_LAYER,
    CELL_LAYER,
    CAND_LAYER,
    PROPOSAL_LAYER,
    PREVIEW_LAYER,
    SEPARATION_LAYER,
)
NUCLEI_LAYER = "nuclei"
QUEUE_ENV = "OMERO_SCREEN_REVIEW_QUEUE"
DB_ENV = "OMERO_SCREEN_REVIEW_DB"
REVIEW_ID = re.compile(r"^([A-Za-z]+\d+)-t(\d+)-L(\d+)$")
QUEUE_FILTERS = ("all", "open", "unsure", "reviewed", "proposals")


class TrackReviewWidget(QWidget):
    """Dock widget driving a :class:`ReviewSession`."""

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.session: ReviewSession | None = None
        self.current: ReviewItem | None = None
        self.path_df: Any = None
        self.cands: Any = None
        self._loaded: tuple[int, str] | None = None
        self._pixel_size = 1.0
        self._nuclei: Any = None
        self._focus_seen: str | None = None
        self._mode: str | None = None  # absorb | drop | split
        self._seeds: list[tuple[int, int]] = []
        self._staged: dict[str, Any] | None = None
        self._queue_pos = -1  # last queue row visited, for ◀ / ▶

        self.settings = QSettings("omero-screen", "track-review")
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._on_file_changed)

        # -- queue file and database (shown in the Queue window) ------------
        last = str(self.settings.value("queue_path", "") or "")
        self.path_edit = QLineEdit(os.environ.get(QUEUE_ENV, "") or last)
        default_db = Path.home() / ".cellview" / "cellview.duckdb"
        self.db_edit = QLineEdit(
            os.environ.get(DB_ENV, "")
            or str(self.settings.value("db_path", "") or "")
            or (str(default_db) if default_db.exists() else "")
        )
        self.db_edit.setPlaceholderText("CellView database (.duckdb)")

        # -- search ---------------------------------------------------------
        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "frame id — e.g. 85 327, or C2-t72-L327"
        )
        self.search.setToolTip(
            "Frame and nucleus label (as shown in the viewer) or track id;\n"
            "an id alone searches the current frame."
        )
        self.search.returnPressed.connect(self.search_cell)
        go = QPushButton("Go")
        go.clicked.connect(self.search_cell)

        # -- windows ----------------------------------------------------------
        queue_btn = QPushButton("Queue…")
        queue_btn.clicked.connect(self.show_queue)
        repair_btn = QPushButton("Repair…")
        repair_btn.clicked.connect(self.show_repair)
        strip = QPushButton("Filmstrip")
        strip.clicked.connect(self.show_filmstrip)

        self.details = QLabel("No queue loaded.")
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.RichText)

        # -- navigation and display -------------------------------------------
        prev_btn, next_btn = QPushButton("◀ Cell"), QPushButton("Cell ▶")
        prev_btn.clicked.connect(lambda: self._step(-1))
        next_btn.clicked.connect(lambda: self._step(1))
        brk = QPushButton("Next break")
        brk.clicked.connect(self.next_break)
        self.follow = QCheckBox("Follow")
        self.follow.setChecked(True)
        self.hide_debris = QCheckBox("Hide debris")
        self.hide_debris.setChecked(True)
        self.hide_debris.setToolTip(
            "Blank stationary, reporter-negative objects in the nuclei layer (data untouched)"
        )
        self.hide_debris.toggled.connect(lambda _: self._apply_debris())
        self.isolate = QCheckBox("Isolate")
        self.isolate.toggled.connect(lambda _: self._update_isolation())
        self.zoom = QSpinBox()
        self.zoom.setRange(1, 50)
        self.zoom.setValue(6)
        self.zoom.setPrefix("zoom ")
        self.reach = QSpinBox()
        self.reach.setRange(1, 100)
        self.reach.setValue(int(self.settings.value("reach", 5) or 5))
        self.reach.setPrefix("track ±")
        self.reach.setToolTip("Frames of the track drawn back and ahead")
        self.reach.valueChanged.connect(self._on_reach)

        # -- verdict ------------------------------------------------------------
        self.outcome = QComboBox()
        self.note = QLineEdit()
        self.note.setPlaceholderText("note (also the reason for an edit)")
        verdicts = QHBoxLayout()
        for verdict in ("accept", "correct", "reject", "unsure"):
            btn = QPushButton(verdict.capitalize())
            btn.clicked.connect(lambda _=False, v=verdict: self.decide(v))
            verdicts.addWidget(btn)
        self.status = QLabel("")

        # -- repair controls (shown in the Repair window) ----------------------
        self.pick = QPushButton("Link: click the cell")
        self.pick.setCheckable(True)
        self.pick.setToolTip(
            "Then click the nucleus that is this cell in the current frame (Shift-L)."
        )
        self.mode_label = QLabel("")
        self.mode_label.setWordWrap(True)
        self.staged_label = QLabel("Nothing staged.")
        self.staged_label.setWordWrap(True)
        self.staged_label.setTextFormat(Qt.TextFormat.RichText)
        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.setToolTip("Write the staged change (Shift-Enter)")
        self.confirm_btn.clicked.connect(self.confirm_staged)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_staged)
        self.proposal_list = QListWidget()
        self.proposal_list.setMaximumHeight(90)
        self.proposal_list.currentRowChanged.connect(
            lambda _: self._draw_proposal()
        )
        self.log_list = QListWidget()
        self.log_list.setMaximumHeight(110)
        self._set_staged_enabled(False)

        # -- layout ------------------------------------------------------------
        layout = QVBoxLayout(self)
        for row in (
            [self.search, go],
            [queue_btn, repair_btn, strip],
        ):
            layout.addLayout(_row(row))
        layout.addWidget(self.details)
        for row in (
            [prev_btn, next_btn, brk],
            [self.follow, self.hide_debris, self.isolate],
            [self.zoom, self.reach],
        ):
            layout.addLayout(_row(row))
        h = QHBoxLayout()
        h.addWidget(QLabel("outcome"))
        h.addWidget(self.outcome)
        layout.addLayout(h)
        layout.addWidget(self.note)
        layout.addLayout(verdicts)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.queue_window = QueueWindow(self)
        self.repair_window = RepairWindow(self)

        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self.viewer.dims.events.current_step.connect(
            lambda _: self._on_frame()
        )
        self._bind_keys()
        if self.path_edit.text():
            self.load_queue(Path(self.path_edit.text()))

    # -- keys ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        bindings: dict[str, Callable[[], Any]] = {
            "Shift-A": lambda: self.decide("accept"),
            "Shift-C": lambda: self.decide("correct"),
            "Shift-R": lambda: self.decide("reject"),
            "Shift-U": lambda: self.decide("unsure"),
            "Shift-N": self.next_break,
            "Shift-F": lambda: self.follow.setChecked(
                not self.follow.isChecked()
            ),
            "Shift-L": lambda: self.pick.setChecked(True),
            "Shift-X": self.unlink_here,
            "Shift-Z": self.undo,
            "Shift-Enter": self.confirm_staged,
        }
        for n in range(1, 10):
            # Plain digits: Qt reports Shift-digit as the shifted symbol (!, @,
            # …), which napari cannot bind. No layer we keep active uses them.
            bindings[str(n)] = partial(self.link_candidate, n)
        for key, fn in bindings.items():
            try:
                self.viewer.bind_key(
                    key, lambda _v, fn=fn: fn(), overwrite=True
                )
            except Exception as err:  # never let a key clash break the widget
                logger.warning(f"Track review: could not bind {key}: {err}")
        # The same keys in the Repair window, which has focus after a click on
        # one of its buttons (napari's bindings need the canvas focused).
        self.repair_window.add_shortcuts(bindings)

    # -- windows -----------------------------------------------------------------

    def show_queue(self) -> None:
        """Open (or raise) the Queue window."""
        self.queue_window.refresh()
        self.queue_window.show()
        self.queue_window.raise_()

    def show_repair(self) -> None:
        """Open (or raise) the Repair window."""
        self.repair_window.show()
        self.repair_window.raise_()

    # -- queue -----------------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Review queue", "", "JSON (*.json)"
        )
        if path:
            self.path_edit.setText(path)
            self.load_queue(Path(path))

    def load_queue(self, path: Path) -> None:
        """Open (or re-open) a queue and its session."""
        db = Path(self.db_edit.text()) if self.db_edit.text() else None
        keep = self.current.id if self.current else None
        try:
            if self.session is None or self.session.queue_path != Path(path):
                self.session = ReviewSession(Path(path), db_path=db)
            else:
                self.session.reload_queue()
        except QueueError as err:
            notifications.show_warning(str(err))
            return
        set_active_session(self.session, self)
        self.settings.setValue("queue_path", str(Path(path).resolve()))
        if db:
            self.settings.setValue("db_path", str(db))
        for f in (str(path), str(self.session.proposals_path)):
            if Path(f).exists() and f not in self.watcher.files():
                self.watcher.addPath(f)
        self.outcome.clear()
        self.outcome.addItems(["", *self.session.queue.outcomes])
        self._refresh_list(select=keep)
        focus = self.session.queue.focus
        if focus and focus != self._focus_seen:
            self._focus_seen = focus
            self._select(focus)
        elif keep is None:
            self._select_first_open()

    def _on_file_changed(self, path: str) -> None:
        if Path(path).exists() and path not in self.watcher.files():
            self.watcher.addPath(path)
        if self.session is None:
            return
        if Path(path) == self.session.proposals_path:
            self._refresh_proposals()
        elif Path(path).exists():
            self.load_queue(Path(path))

    def _ids(self) -> list[str]:
        return [i.id for i in self.session.queue.items] if self.session else []

    def _refresh_list(
        self, select: str | None = None, silent: bool = False
    ) -> None:
        """Refresh the Queue window and the status line (never moves the viewer)."""
        if self.session is None:
            return
        self.queue_window.refresh(highlight=select)
        st = self.session.status()
        self.status.setText(
            f"{st['reviewed']}/{st['items']} reviewed · {st['pending_proposals']} proposals pending"
        )

    def _select(self, item_id: str) -> None:
        """Go to a queue item (or an ad-hoc cell already opened)."""
        if self.session is None:
            return
        ids = self._ids()
        if item_id in ids:
            self._queue_pos = ids.index(item_id)
        try:
            item = self.session.item(item_id)
        except KeyError:
            return
        self.go_to(item)

    def _select_first_open(self) -> None:
        assert self.session is not None
        done = self.session.decided()
        open_items = [i for i in self._ids() if i not in done]
        if open_items:
            self._select(open_items[0])

    def _step(self, delta: int) -> None:
        ids = self._ids()
        if not ids:
            return
        if self.current is not None and self.current.id in ids:
            self._queue_pos = ids.index(self.current.id)
        row = self._queue_pos + delta
        if 0 <= row < len(ids):
            self._select(ids[row])

    # -- search ------------------------------------------------------------------

    def search_cell(self) -> None:
        """Go to the cell named in the search field."""
        if self.session is None:
            notifications.show_warning("Load a review queue first.")
            return
        text = self.search.text().strip()
        well = (
            self.current.well
            if self.current
            else self._loaded[1]
            if self._loaded
            else self.session.queue.items[0].well
            if self.session.queue.items
            else ""
        )
        match = REVIEW_ID.match(text)
        numbers = [int(n) for n in re.findall(r"\d+", text)]
        if match:
            well = match.group(1).upper()
            frame, ident = int(match.group(2)), int(match.group(3))
        elif len(numbers) == 2:
            frame, ident = numbers
        elif len(numbers) == 1:
            frame, ident = int(self.viewer.dims.current_step[0]), numbers[0]
        else:
            notifications.show_info("Enter 'frame id', 'id' or a review id.")
            return
        try:
            self._ensure_well(self.session.queue.plate_id, well)
            item = self.session.find(well, frame, ident)
        except (KeyError, QueueError) as err:
            notifications.show_warning(str(err).strip("'\""))
            return
        if item.id in self._ids():
            self._queue_pos = self._ids().index(item.id)
        self.go_to(item, frame=frame)

    # -- navigation ---------------------------------------------------------------

    def _ensure_well(self, plate_id: int, well: str) -> None:
        if self._loaded == (plate_id, well):
            return
        from omero_screen_napari.zarr_cache.display import load_plate_to_viewer
        from omero_screen_napari.zarr_cache.reader import plate_info, read_well

        if not load_plate_to_viewer(self.viewer, plate_id, well):
            raise QueueError(
                f"Well {well} of plate {plate_id} is not in the zarr cache."
            )
        self._pixel_size = float(
            plate_info(plate_id).get("pixel_size_um") or 1.0
        )
        from omero_screen_napari.zarr_cache.correction import tracked_nuclei

        # CellView holds the corrected tracks when a correction is stored, so
        # the review works on corrected nuclei; the first-pass layer it
        # replaces is not shown separately here.
        nuclei = tracked_nuclei(
            plate_id, well, list(read_well(plate_id, well)["nuclei"] or [])
        )
        self._nuclei = nuclei[0] if nuclei else None
        self._nuclei_levels = list(nuclei)
        if "nuclei (corrected)" in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers["nuclei (corrected)"])
        if NUCLEI_LAYER in self.viewer.layers:
            self.viewer.layers[NUCLEI_LAYER].visible = True
        self._loaded = (plate_id, well)
        self._apply_debris()

    def go_to(self, item: ReviewItem, frame: int | None = None) -> None:
        """Load the item's well, draw the cell and jump to ``frame`` (default: the item's)."""
        assert self.session is not None
        try:
            self._ensure_well(self.session.queue.plate_id, item.well)
        except QueueError as err:
            notifications.show_warning(str(err))
            return
        self.current = item
        self.pick.setChecked(False)
        self._disarm()
        self._clear_staged()
        self.note.clear()
        self.outcome.setCurrentText(item.outcome)
        self._redraw()
        t = item.frame if frame is None else frame
        self.viewer.dims.set_current_step(0, t)
        self._centre(t)
        self.viewer.scene.camera.zoom = self.zoom.value()
        self._refresh_proposals()

    def _redraw(self) -> None:
        """Recompute the curated path and redraw every review layer."""
        assert self.session is not None and self.current is not None
        try:
            self.path_df = self.session.path(self.current.id)
        except Exception as err:  # the database may be unavailable
            notifications.show_warning(
                f"Could not load the cell's path: {err}"
            )
            self.path_df = None
        for name in REVIEW_LAYERS:
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
        self.cands = None
        self._draw_path()
        self._show_details()
        self._refresh_log()
        self._update_isolation()

    def _points(self) -> list[tuple[int, float, float, int]]:
        if self.path_df is not None and len(self.path_df):
            return [
                (int(r.timepoint), float(r.y), float(r.x), int(r.label))
                for r in self.path_df.dropna(subset=["y", "x"]).itertuples()
            ]
        assert self.current is not None
        return [(p.t, p.y, p.x, p.label) for p in self.current.path]

    def _track_data(self) -> tuple[np.ndarray, dict[int, list[int]]]:
        """Napari tracks ``[id, t, y, x]`` of the cell, its mother and daughters."""
        assert self.session is not None and self.current is not None
        try:
            lineage = self.session.lineage(self.current.id)
        except Exception as err:
            logger.warning(
                f"Track review: no lineage for {self.current.id}: {err}"
            )
            lineage = []
        if not lineage:
            pts = self._points()
            data = np.array([[1, t, y, x] for t, y, x, _ in pts], dtype=float)
            return data.reshape(-1, 4), {}
        rows, ids = [], {tr["track_id"] for tr in lineage}
        for tr in lineage:
            f = tr["frames"]
            rows.append(
                np.column_stack(
                    [
                        np.full(len(f), tr["track_id"]),
                        f["timepoint"].to_numpy(dtype=float),
                        f["y"].to_numpy(dtype=float),
                        f["x"].to_numpy(dtype=float),
                    ]
                )
            )
        graph = {
            tr["track_id"]: [tr["parent"]]
            for tr in lineage
            if tr["parent"] in ids
        }
        return np.vstack(rows), graph

    def _add_tracks(
        self,
        data: np.ndarray,
        graph: dict[int, list[int]],
        name: str,
        colour: str,
        tail: int,
        head: int,
        width: float,
    ) -> Any:
        layer = self.viewer.add_tracks(  # ty: ignore[unresolved-attribute]
            data,
            graph=graph,
            name=name,
            scale=(1.0, self._pixel_size, self._pixel_size),
            tail_length=tail,
            head_length=head,
            tail_width=width,
            properties={"c": np.zeros(len(data))},
            blending="translucent",
        )
        layer.colormaps_dict = {"c": Colormap([colour, colour], name=name)}
        layer.color_by = "c"
        hold_until_loaded(self.viewer, layer)
        return layer

    def _draw_path(self) -> None:
        pts = self._points()
        if not pts:
            return
        scale = (1.0, self._pixel_size, self._pixel_size)
        data, graph = self._track_data()
        if len(data) > 1:
            n = self.reach.value()
            self._add_tracks(data, graph, PAST_LAYER, "cyan", n, 0, 4)
            self._add_tracks(data, graph, AHEAD_LAYER, "magenta", 0, n, 4)
        coords = np.array([[t, y, x] for t, y, x, _ in pts], dtype=float)
        gap = np.array([lab == 0 for *_, lab in pts])
        self.viewer.add_points(  # ty: ignore[unresolved-attribute]
            coords,
            name=MARKER_LAYER,
            scale=scale,
            size=np.where(gap, 18, 40) * self._pixel_size,
            symbol=["cross" if g else "ring" for g in gap],
            face_color=["magenta" if g else "yellow" for g in gap],
            border_width=0,
            projection_mode="none",
        )
        if self._nuclei is not None:
            self.viewer.add_labels(  # ty: ignore[unresolved-attribute]
                _cell_mask(
                    self._nuclei, {t: lab for t, _, _, lab in pts if lab}
                ),
                name=CELL_LAYER,
                scale=scale,
                opacity=1.0,
            ).contour = 2
        self._activate_image()

    def _on_reach(self, n: int) -> None:
        self.settings.setValue("reach", n)
        layers: Any = self.viewer.layers
        if PAST_LAYER in layers:
            layers[PAST_LAYER].tail_length = n
        if AHEAD_LAYER in layers:
            layers[AHEAD_LAYER].head_length = n
        if PREVIEW_LAYER in layers:
            layers[PREVIEW_LAYER].tail_length = n
            layers[PREVIEW_LAYER].head_length = n

    def _patches(
        self, pts: list[tuple[int, float, float, int]]
    ) -> dict[int, tuple[int, int, np.ndarray]]:
        """Reviewer-made masks of the cell, by frame."""
        if self.session is None or self.current is None:
            return {}
        from omero_screen_napari.review.masks import load_patch

        try:
            _, curated = self.session.source.well(self.current.well)
        except Exception:
            return {}
        out = {}
        for t, _, _, lab in pts:
            meas = curated.extras.get((t, lab))
            if meas and "patch" in meas:
                out[t] = load_patch(
                    self.session.edits_path.parent, meas["patch"]
                )
        return out

    def _activate_image(self) -> None:
        # Keep an image layer active so clicks and keys never edit our layers.
        for layer in self.viewer.layers:
            if layer.__class__.__name__ == "Image":
                self.viewer.layers.selection.active = layer
                return

    def _show_details(self) -> None:
        assert self.current is not None and self.session is not None
        it = self.current
        breaks = ""
        try:
            b = self.session.breaks(it.id)
            breaks = (
                ", ".join(map(str, b[:8])) + (" …" if len(b) > 8 else "")
                if b
                else "none"
            )
        except Exception:
            breaks = "?"
        span = ""
        if self.path_df is not None and len(self.path_df):
            tid = int(self.path_df.track_id.iloc[0])
            span = f"track {tid}, frames {int(self.path_df.timepoint.min())}–{int(self.path_df.timepoint.max())}, "
            span += f"{int(self.path_df.gap.sum())} without mask, "
        where = "" if it.id in self._ids() else " · <i>not in the queue</i>"
        self.details.setText(
            f"<b>{it.id}</b>{where} — {it.reason or '–'} · automatic: {it.outcome or '–'}<br>"
            f"{span}breaks at: {breaks}<br><i>{it.question}</i>"
        )

    def _centre(self, t: int) -> None:
        pts = self._points()
        if not pts:
            return
        before = [p for p in pts if p[0] <= t] or pts[:1]
        _, y, x, _ = before[-1]
        self.viewer.scene.camera.center = (
            y * self._pixel_size,
            x * self._pixel_size,
        )

    def _on_frame(self) -> None:
        if self.follow.isChecked() and self.current is not None:
            self._centre(int(self.viewer.dims.current_step[0]))

    def next_break(self) -> None:
        """Jump to the next frame after the current one where the track is interrupted."""
        if self.session is None or self.current is None:
            return
        t = int(self.viewer.dims.current_step[0])
        later = [b for b in self.session.breaks(self.current.id) if b > t]
        if not later:
            notifications.show_info("No further breaks for this cell.")
            return
        self.viewer.dims.set_current_step(0, later[0])
        self._centre(later[0] - 1)
        self.show_candidates()

    def _apply_debris(self) -> None:
        """Show the nuclei layer with debris labels blanked, or the original."""
        if NUCLEI_LAYER not in self.viewer.layers or not getattr(
            self, "_nuclei_levels", None
        ):
            return
        layer: Any = self.viewer.layers[NUCLEI_LAYER]
        ids: set[int] = set()
        if (
            self.hide_debris.isChecked()
            and self.session is not None
            and self._loaded is not None
        ):
            try:
                ids = self.session.source.debris(self._loaded[1])
            except Exception as err:
                notifications.show_warning(f"Could not classify debris: {err}")
        layer.data = (
            _hide_labels(self._nuclei_levels, ids)
            if ids
            else self._nuclei_levels
        )
        self.status.setToolTip(
            f"{len(ids)} debris objects hidden" if ids else ""
        )

    def _update_isolation(self) -> None:
        if NUCLEI_LAYER in self.viewer.layers:
            self.viewer.layers[
                NUCLEI_LAYER
            ].visible = not self.isolate.isChecked()

    def show_filmstrip(self) -> None:
        """Render the cell's filmstrip and show it in a window."""
        if self.session is None or self.current is None:
            return
        from omero_screen_napari.review.cells import render_cell
        from omero_screen_napari.review.filmstrip import save_filmstrip

        anchor = self.session.anchor(self.current.id)
        out = (
            self.session.queue_path.parent
            / "filmstrips"
            / f"{self.current.id}.png"
        )
        try:
            fig = render_cell(
                self.session.source,
                self.current.well,
                anchor,
                start=anchor[0],
                title=self.current.id,
            )
            save_filmstrip(fig, out)
        except Exception as err:
            notifications.show_warning(f"Filmstrip failed: {err}")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(self.current.id)
        area = QScrollArea(dlg)
        lab = QLabel()
        lab.setPixmap(QPixmap(str(out)))
        area.setWidget(lab)
        lay = QVBoxLayout(dlg)
        lay.addWidget(area)
        dlg.resize(1200, 700)
        dlg.show()

    # -- staging: every repair is previewed, then confirmed -----------------------

    def _set_staged_enabled(self, on: bool) -> None:
        self.confirm_btn.setEnabled(on)
        self.cancel_btn.setEnabled(on)

    def _clear_staged(self) -> None:
        self._staged = None
        self.staged_label.setText("Nothing staged.")
        self._set_staged_enabled(False)
        for name in (PREVIEW_LAYER, SEPARATION_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])

    def _stage(
        self, staged: dict[str, Any], title: str, lines: list[str]
    ) -> None:
        self._staged = staged
        body = "".join(f"<br>• {line}" for line in lines)
        self.staged_label.setText(f"<b>{title}</b>{body}")
        self._set_staged_enabled(True)
        self.show_repair()

    def _stage_edit(
        self, op: str, args: dict[str, Any], title: str, reason: str = ""
    ) -> None:
        """Preview an edit: say what it changes and draw the path after it."""
        if self.session is None or self.current is None:
            return
        from cellview.tracks.edit import EditError

        self._clear_staged()
        try:
            lines, after = self.session.preview(self.current.id, op, args)
        except (EditError, KeyError, ValueError) as err:
            notifications.show_warning(f"{title}: {err}")
            return
        self._draw_preview(after)
        self._stage(
            {"kind": "edit", "op": op, "args": args, "reason": reason},
            title,
            lines,
        )

    def _draw_preview(self, after: Any) -> None:
        if PREVIEW_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[PREVIEW_LAYER])
        if after is None or not {"y", "x"} <= set(after.columns):
            return
        pts = after.dropna(subset=["y", "x"])
        if len(pts) < 2:
            return
        data = np.column_stack(
            [
                np.ones(len(pts)),
                pts["timepoint"].to_numpy(dtype=float),
                pts["y"].to_numpy(dtype=float),
                pts["x"].to_numpy(dtype=float),
            ]
        )
        n = self.reach.value()
        self._add_tracks(data, {}, PREVIEW_LAYER, "orange", n, n, 2)
        self._activate_image()

    def confirm_staged(self) -> None:
        """Write the staged change."""
        staged = self._staged
        if staged is None or self.session is None or self.current is None:
            return
        self._clear_staged()
        if staged["kind"] == "edit":
            self._edit(staged["op"], staged["args"], staged["reason"])
            return
        from omero_screen_napari.review.masks import (
            MaskError,
            commit_drawn,
            commit_separation,
            commit_split,
        )

        t = staged["frame"]
        try:
            if staged["kind"] == "separate":
                entries = commit_separation(self.session, staged["plan"])
                notifications.show_info(
                    f"Separated in {len(entries) // 2} frames (one Undo reverts all)."
                )
            elif staged["kind"] == "split":
                commit_split(
                    self.session,
                    self.current.id,
                    t,
                    staged["label"],
                    staged["seeds"],
                )
            else:
                commit_drawn(self.session, self.current.id, t, staged["mask"])
                if DRAW_LAYER in self.viewer.layers:
                    self.viewer.layers.remove(self.viewer.layers[DRAW_LAYER])
        except (MaskError, ValueError) as err:
            notifications.show_warning(str(err))
            return
        self._redraw()
        self.viewer.dims.set_current_step(0, t)

    def cancel_staged(self) -> None:
        """Drop the staged change; nothing was written."""
        self._clear_staged()

    # -- track edits -------------------------------------------------------------

    def _edit(self, op: str, args: dict[str, Any], reason: str = "") -> bool:
        assert self.session is not None and self.current is not None
        from cellview.tracks.edit import EditError

        try:
            entry = self.session.edit(
                self.current.id,
                op,
                args,
                reason=reason or self.note.text().strip(),
            )
        except (EditError, KeyError, ValueError) as err:
            notifications.show_warning(str(err))
            return False
        logger.info(f"Track review {self.current.id}: {entry.id} {op} {args}")
        t = int(self.viewer.dims.current_step[0])
        self._redraw()
        self.viewer.dims.set_current_step(0, t)
        return True

    def _on_click(self, viewer: Any, event: Any) -> None:
        if event.type != "mouse_press":
            return
        if self.pick.isChecked():
            self.pick_at(tuple(event.position))
        elif self._mode is not None:
            self._mask_click(tuple(event.position))

    def pick_at(self, position: tuple[float, ...]) -> int:
        """Stage a link of the cell to the nucleus at a world ``position``."""
        self.pick.setChecked(False)
        if self.current is None or self._nuclei is None:
            return 0
        t, y, x = self._canvas(position)
        if not (
            0 <= y < self._nuclei.shape[-2] and 0 <= x < self._nuclei.shape[-1]
        ):
            return 0
        label = int(np.asarray(self._nuclei[t, y, x]))
        if label == 0:
            notifications.show_warning(
                "No nucleus under the cursor — click inside one."
            )
            return 0
        self._stage_edit(
            "link",
            {"frame": t, "label": label},
            f"Link: from t{t} on, this cell is nucleus {label}",
        )
        return label

    # -- mask edits ------------------------------------------------------------------

    def arm(self, mode: str) -> None:
        """Wait for a click that adds, removes or splits a nucleus."""
        self._mode, self._seeds = mode, []
        self.mode_label.setText(
            {
                "absorb": "Click the nucleus to add to this cell.",
                "drop": "Click the nucleus to remove from its track.",
                "split": "Click seed 1: the part that is this cell.",
            }[mode]
        )
        self.show_repair()

    def _disarm(self) -> None:
        self._mode, self._seeds = None, []
        self.mode_label.setText("")

    def _canvas(self, position: tuple[float, ...]) -> tuple[int, int, int]:
        t = int(self.viewer.dims.current_step[0])
        return (
            t,
            int(round(position[-2] / self._pixel_size)),
            int(round(position[-1] / self._pixel_size)),
        )

    def _mask_click(self, position: tuple[float, ...]) -> None:
        if (
            self.current is None
            or self._nuclei is None
            or self.session is None
        ):
            self._disarm()
            return
        t, y, x = self._canvas(position)
        label = (
            int(np.asarray(self._nuclei[t, y, x]))
            if 0 <= y < self._nuclei.shape[-2]
            and 0 <= x < self._nuclei.shape[-1]
            else 0
        )
        mode = self._mode
        if mode in ("absorb", "drop"):
            self._disarm()
            if not label:
                notifications.show_warning("No nucleus under the cursor.")
                return
            title = (
                f"Add nucleus {label} to this cell at t{t}"
                if mode == "absorb"
                else f"Remove nucleus {label} from its track at t{t}"
            )
            self._stage_edit(mode, {"frames": [t, t], "label": label}, title)
            return
        # split: collect two seeds on the same raw label
        self._seeds.append((y, x))
        if len(self._seeds) == 1:
            self._split_label = label
            self.mode_label.setText("Click seed 2: the other part.")
            return
        seeds, split_label = self._seeds, getattr(self, "_split_label", 0)
        self._disarm()
        if not split_label:
            notifications.show_warning("Seed 1 was not on a nucleus.")
            return
        self._clear_staged()
        self._stage(
            {
                "kind": "split",
                "frame": t,
                "label": split_label,
                "seeds": seeds,
            },
            f"Split nucleus {split_label} at t{t}",
            [
                "the part at seed 1 stays with this cell",
                "the part at seed 2 becomes a new nucleus",
            ],
        )

    def start_drawing(self) -> None:
        """Add an empty paint layer for the current frame; paint the missed nucleus."""
        if self._nuclei is None:
            return
        if DRAW_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[DRAW_LAYER])
        shape = self._nuclei.shape[-2:]
        layer = self.viewer.add_labels(  # ty: ignore[unresolved-attribute]
            np.zeros(shape, dtype=np.uint8),
            name=DRAW_LAYER,
            scale=(self._pixel_size, self._pixel_size),
            opacity=0.6,
        )
        layer.mode = "paint"
        layer.selected_label = 1
        layer.brush_size = 6
        self.viewer.layers.selection.active = layer
        self.mode_label.setText(
            f"Paint the nucleus in frame {int(self.viewer.dims.current_step[0])}, "
            "then Finish drawing."
        )

    def commit_drawing(self) -> None:
        """Stage the painted pixels as a nucleus of this cell in the current frame."""
        if (
            DRAW_LAYER not in self.viewer.layers
            or self.session is None
            or self.current is None
        ):
            notifications.show_info("Start with Draw nucleus.")
            return
        mask = np.asarray(self.viewer.layers[DRAW_LAYER].data) > 0
        if not mask.any():
            notifications.show_info("Nothing painted yet.")
            return
        t = int(self.viewer.dims.current_step[0])
        self.mode_label.setText("")
        self._clear_staged()
        self._stage(
            {"kind": "draw", "frame": t, "mask": mask},
            f"Add the drawn nucleus to this cell at t{t}",
            [f"{int(mask.sum())} pixels"],
        )

    def stage_separation(
        self, first: int, last: int, threshold: float = 0.5
    ) -> None:
        """Preview separating the cell from a merged reporter-negative object."""
        if self.session is None or self.current is None:
            return
        from omero_screen_napari.review.masks import MaskError, plan_separation

        self._clear_staged()
        try:
            plan = plan_separation(
                self.session, self.current.id, first, last, threshold
            )
        except (MaskError, KeyError, ValueError) as err:
            notifications.show_warning(f"Separate: {err}")
            return
        if not plan.frames:
            notifications.show_info(
                f"Nothing to separate in t{first}–{last}: "
                + ", ".join(f"t{t} {why}" for t, why in plan.skipped.items())
            )
            return
        before = (
            self.path_df[self.path_df.timepoint < first]["area"].dropna()
            if self.path_df is not None and len(self.path_df)
            else None
        )
        normal = (
            float(before.tail(5).median())
            if before is not None and len(before)
            else 0.0
        )
        lines, flagged = [], []
        for fs in plan.frames:
            area = int(fs.cell.sum())
            small = bool(normal) and area < 0.5 * normal
            if small:
                flagged.append(fs.frame)
            lines.append(
                f"t{fs.frame}: cell {area} px, other {int(fs.other.sum())} px, "
                f"reporter {fs.score:.2f}"
                + (" ⚠ cell part small" if small else "")
            )
        head = [
            f"{len(plan.frames)} frames; reference "
            + ", ".join(f"{k} {v:.0f}" for k, v in plan.reference.items())
            + (f"; cell ≈ {normal:.0f} px before" if normal else ""),
        ]
        if flagged:
            head.append(
                f"⚠ cell part under half its size in t{flagged[0]}"
                + (f"–{flagged[-1]}" if len(flagged) > 1 else "")
                + ": the cell may have left the mask; end the range before."
            )
        if plan.skipped:
            head.append(
                "unchanged: "
                + ", ".join(f"t{t} ({why})" for t, why in plan.skipped.items())
            )
        self._draw_separation(plan)
        self._stage(
            {"kind": "separate", "frame": first, "plan": plan},
            f"Separate by reporter, t{first}–{last}",
            head + lines,
        )

    def _draw_separation(self, plan: Any) -> None:
        if self._nuclei is None or not plan.frames:
            return
        data, (y0, x0) = _separation_mask(self._nuclei.shape[0], plan)
        ps = self._pixel_size
        layer = self.viewer.add_labels(  # ty: ignore[unresolved-attribute]
            data,
            name=SEPARATION_LAYER,
            scale=(1.0, ps, ps),
            translate=(0.0, y0 * ps, x0 * ps),
            opacity=0.6,
        )
        layer.contour = 2
        self._activate_image()

    def show_candidates(self) -> None:
        """Mark numbered continuation candidates at the current frame."""
        if self.session is None or self.current is None:
            return
        t = int(self.viewer.dims.current_step[0])
        try:
            self.cands = self.session.candidates(self.current.id, t)
        except Exception as err:
            notifications.show_warning(str(err))
            return
        if CAND_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[CAND_LAYER])
        if self.cands is None or self.cands.empty:
            notifications.show_info("No candidates within reach.")
            return
        coords = self.cands[["timepoint", "y", "x"]].to_numpy(dtype=float)
        self.viewer.add_points(  # ty: ignore[unresolved-attribute]
            coords,
            name=CAND_LAYER,
            scale=(1.0, self._pixel_size, self._pixel_size),
            size=34 * self._pixel_size,
            symbol="square",
            face_color="transparent",
            border_color="cyan",
            border_width=0.08,
            text={
                "string": [str(r) for r in self.cands["rank"]],
                "color": "cyan",
                "size": 14,
            },
            projection_mode="all",
        )
        self._activate_image()

    def link_candidate(self, rank: int) -> None:
        """Stage a link of the cell to numbered candidate ``rank``."""
        if self.cands is None or self.cands.empty or rank > len(self.cands):
            notifications.show_info(
                "Show candidates first (Next break or Candidates)."
            )
            return
        row = self.cands.iloc[rank - 1]
        t, label = int(row.timepoint), int(row.label)
        self._stage_edit(
            "link",
            {"frame": t, "label": label},
            f"Link to candidate {rank}: from t{t} on, this cell is nucleus {label}",
            reason=f"candidate {rank}",
        )

    def unlink_here(self) -> None:
        """Stage cutting the track at the current frame."""
        t = int(self.viewer.dims.current_step[0])
        self._stage_edit(
            "unlink",
            {"frame": t},
            f"Unlink at t{t}: the rest of the track becomes a new cell",
        )

    def mark_event(self, kind: str) -> None:
        """Stage marking ``kind`` (mitosis, death, slippage) at the current frame."""
        t = int(self.viewer.dims.current_step[0])
        self._stage_edit(
            "event", {"kind": kind, "frame": t}, f"Mark {kind} at t{t}"
        )

    def exclude(self) -> None:
        """Stage excluding the cell; the note (or outcome) is the reason."""
        reason = (
            self.note.text().strip()
            or self.outcome.currentText()
            or "excluded"
        )
        self._stage_edit(
            "exclude", {"reason": reason}, f"Exclude this cell ({reason})"
        )

    def undo(self) -> None:
        """Revert the last edit in this cell's well."""
        if self.session is None or self.current is None:
            return
        from cellview.tracks.edit import EditError

        try:
            entry = self.session.undo(self.current.id)
        except EditError as err:
            notifications.show_warning(str(err))
            return
        notifications.show_info(f"Reverted {entry.args['target']}")
        self._redraw()

    def _refresh_log(self) -> None:
        self.log_list.clear()
        if self.session is None or self.current is None:
            return
        for e in self.session.entries(self.current.id):
            who = e.author + (f"/{e.confirmed_by}" if e.confirmed_by else "")
            self.log_list.addItem(f"{e.id} {e.op} {e.args} [{who}] {e.reason}")

    # -- agent proposals ------------------------------------------------------------

    def _pending(self) -> list[Any]:
        """Proposals for this cell that still need the reviewer: pending, or applied but unconfirmed."""
        if self.session is None or self.current is None:
            return []
        return [
            p
            for p in self.session.proposals()
            if p.item == self.current.id and p.status in ("pending", "applied")
        ]

    def _refresh_proposals(self) -> None:
        self.proposal_list.clear()
        for p in self._pending():
            tag = (
                f"[{p.status}"
                + (f", {p.confidence}" if p.confidence else "")
                + "] "
            )
            who = f"{p.author}: " if p.author != "agent" else ""
            self.proposal_list.addItem(
                f"{tag}{p.id} {p.op} {p.args} — {who}{p.reason}"
            )
        if self.proposal_list.count():
            self.proposal_list.setCurrentRow(0)
            self.show_repair()
        if self.session is not None:
            self._refresh_list(
                select=self.current.id if self.current else None, silent=True
            )

    def _draw_proposal(self) -> None:
        if PROPOSAL_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[PROPOSAL_LAYER])
        pending = self._pending()
        row = self.proposal_list.currentRow()
        if (
            not pending
            or not 0 <= row < len(pending)
            or self.session is None
            or self.current is None
        ):
            return
        p = pending[row]
        if p.status == "pending":
            try:
                lines, after = self.session.preview(
                    self.current.id, p.op, p.args
                )
                self._draw_preview(after)
                entry = self.proposal_list.item(row)
                if entry is not None:
                    entry.setToolTip("\n".join(lines))
            except Exception:  # a proposal that no longer applies
                pass
        else:
            entry = self.proposal_list.item(row)
            if entry is not None:
                entry.setToolTip(
                    "Applied already; Reject reverts it, Confirm keeps it."
                )
        frame, label = p.args.get("frame"), p.args.get("label")
        if frame is None:
            return
        self.viewer.dims.set_current_step(0, int(frame))
        if label is None:
            return
        det, _ = self.session.source.well(self.current.well)
        hit = det[(det.timepoint == int(frame)) & (det.label == int(label))]
        if hit.empty:
            return
        self.viewer.add_points(  # ty: ignore[unresolved-attribute]
            hit[["timepoint", "y", "x"]].to_numpy(dtype=float),
            name=PROPOSAL_LAYER,
            scale=(1.0, self._pixel_size, self._pixel_size),
            size=46 * self._pixel_size,
            symbol="diamond",
            face_color="transparent",
            border_color="orange",
            border_width=0.1,
            projection_mode="none",
        )
        self._activate_image()

    def resolve_proposal(self, confirm: bool) -> None:
        pending = self._pending()
        row = self.proposal_list.currentRow()
        if self.session is None or not 0 <= row < len(pending):
            return
        from cellview.tracks.edit import EditError

        try:
            self.session.resolve_proposal(
                pending[row].id, confirm, note=self.note.text().strip()
            )
        except (EditError, ValueError, KeyError) as err:
            notifications.show_warning(str(err))
            return
        self._redraw()
        self._refresh_proposals()

    # -- verdicts -------------------------------------------------------------------

    def decide(self, verdict: str) -> None:
        """Record a verdict on the current cell; in the queue, move to the next one."""
        if self.session is None or self.current is None:
            notifications.show_warning("Select a cell first.")
            return
        # Accepting the cell as shown confirms whatever was applied to it.
        self.session.confirm_applied(self.current.id)
        self.session.verdict(
            self.current.id,
            verdict,
            self.outcome.currentText(),
            self.note.text().strip(),
        )
        self._refresh_proposals()
        self._refresh_list()
        if self.current.id in self._ids():
            self._step(1)
        else:
            notifications.show_info(f"{verdict}: {self.current.id}")


class QueueWindow(QDialog):
    """The review queue: one row per cell, with flags and verdicts."""

    COLUMNS = ("cell", "frame", "flagged", "automatic", "agent", "you")

    def __init__(self, ui: TrackReviewWidget) -> None:
        super().__init__(ui)
        self.ui = ui
        self.setWindowTitle("Review queue")
        self.setModal(False)
        browse = QPushButton("Browse")
        browse.clicked.connect(ui._browse)
        load = QPushButton("Load")
        load.clicked.connect(lambda: ui.load_queue(Path(ui.path_edit.text())))
        self.filter = QComboBox()
        self.filter.addItems(QUEUE_FILTERS)
        self.filter.currentTextChanged.connect(lambda _: self.refresh())
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        rows_header = self.table.verticalHeader()
        if rows_header is not None:
            rows_header.setVisible(False)
        columns_header = self.table.horizontalHeader()
        if columns_header is not None:
            columns_header.setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._open)
        self.table.currentCellChanged.connect(
            lambda row, *_: self._show_note(row)
        )
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.TextFormat.RichText)
        self.note.setMinimumHeight(70)
        self.note.setAlignment(Qt.AlignmentFlag.AlignTop)
        go = QPushButton("Go to cell")
        go.clicked.connect(lambda: self._open(self.table.currentRow(), 0))
        self._rows: list[str] = []
        self._notes: dict[str, str] = {}

        lay = QVBoxLayout(self)
        lay.addLayout(_row([ui.path_edit, browse, load]))
        lay.addWidget(ui.db_edit)
        h = QHBoxLayout()
        h.addWidget(QLabel("show"))
        h.addWidget(self.filter)
        h.addStretch(1)
        h.addWidget(go)
        lay.addLayout(h)
        lay.addWidget(self.table)
        lay.addWidget(self.note)
        self.resize(720, 640)

    def refresh(self, highlight: str | None = None) -> None:
        """Rebuild the table from the queue, decisions and proposals."""
        session = self.ui.session
        if session is None:
            return
        path = decisions_path(session.queue_path)
        agent: dict[str, Any] = {}
        human: dict[str, Any] = {}
        for d in read_decisions(path):
            (agent if d.author == "agent" else human)[d.id] = d
        pending = {
            p.item
            for p in session.proposals()
            if p.status in ("pending", "applied")
        }
        show = self.filter.currentText()
        self._rows, self._notes = [], {}
        rows = []
        for it in session.queue.items:
            a, h = agent.get(it.id), human.get(it.id)
            done = h or a
            keep = (
                show == "all"
                or (show == "open" and h is None)
                or (
                    show == "unsure"
                    and done is not None
                    and done.verdict == "unsure"
                )
                or (show == "reviewed" and h is not None)
                or (show == "proposals" and it.id in pending)
            )
            if not keep:
                continue
            star = " *" if it.id in pending else ""
            rows.append(
                (
                    it.id + star,
                    it.frame,
                    it.reason,
                    it.outcome,
                    _verdict_text(a),
                    _verdict_text(h),
                )
            )
            self._rows.append(it.id)
            self._notes[it.id] = (
                f"<b>{it.id}</b> — <i>{it.question}</i>"
                + (f"<br><b>agent:</b> {a.note}" if a and a.note else "")
                + (f"<br><b>you:</b> {h.note}" if h and h.note else "")
            )
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(str(value or "")))
        self.table.resizeColumnsToContents()
        target = highlight or (self.ui.current.id if self.ui.current else None)
        if target in self._rows:
            self.table.setCurrentCell(self._rows.index(target), 0)
            self._show_note(self._rows.index(target))
        self.table.blockSignals(False)
        self.setWindowTitle(f"Review queue — {len(rows)} shown")

    def _show_note(self, row: int) -> None:
        if 0 <= row < len(self._rows):
            self.note.setText(self._notes.get(self._rows[row], ""))

    def _open(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._rows):
            self.ui._select(self._rows[row])


class RepairWindow(QDialog):
    """Repair actions for the current cell; each is staged, then confirmed."""

    def __init__(self, ui: TrackReviewWidget) -> None:
        super().__init__(ui)
        self.ui = ui
        self.setWindowTitle("Repair")
        self.setModal(False)

        def button(text: str, fn: Callable[[], Any], tip: str = "") -> Any:
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False: fn())
            if tip:
                btn.setToolTip(tip)
            return btn

        link = QGroupBox("Track")
        lay = QVBoxLayout(link)
        lay.addLayout(
            _row(
                [
                    ui.pick,
                    button(
                        "Candidates",
                        ui.show_candidates,
                        "Number the nuclei that could continue the cell (then press 1…9)",
                    ),
                    button(
                        "Unlink here",
                        ui.unlink_here,
                        "Cut the track at this frame (Shift-X)",
                    ),
                ]
            )
        )
        nucleus = QGroupBox("Nucleus in this frame")
        lay = QVBoxLayout(nucleus)
        lay.addLayout(
            _row(
                [
                    button(
                        "Add",
                        lambda: ui.arm("absorb"),
                        "Click a nucleus to add it to this cell",
                    ),
                    button(
                        "Remove",
                        lambda: ui.arm("drop"),
                        "Click a nucleus to remove it from its track",
                    ),
                    button(
                        "Split",
                        lambda: ui.arm("split"),
                        "Click two seeds: this cell's part, then the other",
                    ),
                ]
            )
        )
        lay.addLayout(
            _row(
                [
                    button("Draw", ui.start_drawing, "Paint a missed nucleus"),
                    button("Finish drawing", ui.commit_drawing),
                ]
            )
        )
        separate = QGroupBox("Separate by reporter (frames)")
        separate.setToolTip(
            "Split the cell from a reporter-negative object merged into its mask "
            "(e.g. an engulfed dead cell), in every frame of the range"
        )
        lay = QVBoxLayout(separate)
        self.sep_from, self.sep_to = QSpinBox(), QSpinBox()
        for spin, prefix in ((self.sep_from, "from t"), (self.sep_to, "to t")):
            spin.setRange(0, 100000)
            spin.setPrefix(prefix)
        self.sep_threshold = QDoubleSpinBox()
        self.sep_threshold.setRange(0.05, 0.95)
        self.sep_threshold.setSingleStep(0.05)
        self.sep_threshold.setValue(0.5)
        self.sep_threshold.setPrefix("reporter > ")
        self.sep_threshold.setToolTip(
            "Fraction of the cell's own reporter level before the range"
        )
        lay.addLayout(
            _row(
                [
                    button(
                        "From here",
                        self._from_here,
                        "Start at the current frame",
                    ),
                    self.sep_from,
                    self.sep_to,
                ]
            )
        )
        lay.addLayout(
            _row(
                [
                    self.sep_threshold,
                    button(
                        "Preview",
                        lambda: ui.stage_separation(
                            self.sep_from.value(),
                            self.sep_to.value(),
                            self.sep_threshold.value(),
                        ),
                    ),
                ]
            )
        )
        mark = QGroupBox("Mark")
        lay = QVBoxLayout(mark)
        lay.addLayout(
            _row(
                [
                    button(
                        k.capitalize(),
                        partial(ui.mark_event, k),
                        f"{k} at this frame",
                    )
                    for k in ("mitosis", "death", "slippage")
                ]
                + [
                    button(
                        "Exclude",
                        ui.exclude,
                        "The note (or outcome) is the reason",
                    )
                ]
            )
        )
        staged = QGroupBox("Change")
        lay = QVBoxLayout(staged)
        lay.addWidget(ui.mode_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(ui.staged_label)
        scroll.setMinimumHeight(70)
        scroll.setMaximumHeight(220)
        lay.addWidget(scroll)
        lay.addLayout(_row([ui.confirm_btn, ui.cancel_btn]))
        agent = QGroupBox("Agent proposals")
        lay = QVBoxLayout(agent)
        lay.addWidget(ui.proposal_list)
        lay.addLayout(
            _row(
                [
                    button(
                        "Confirm proposal",
                        lambda: ui.resolve_proposal(True),
                        "Write a pending proposal, or keep an applied one",
                    ),
                    button(
                        "Reject proposal",
                        lambda: ui.resolve_proposal(False),
                        "Discard a pending proposal, or revert an applied one",
                    ),
                ]
            )
        )
        log = QGroupBox("Edits for this cell")
        lay = QVBoxLayout(log)
        lay.addWidget(ui.log_list)
        lay.addWidget(
            button(
                "Undo last edit",
                ui.undo,
                "Revert the last edit in this well (Shift-Z)",
            )
        )

        main = QVBoxLayout(self)
        for box in (link, nucleus, separate, mark, staged, agent, log):
            main.addWidget(box)
        self.resize(460, 680)

    def _from_here(self) -> None:
        """Range from the current frame to the end of the cell's track."""
        t = int(self.ui.viewer.dims.current_step[0])
        self.sep_from.setValue(t)
        df = self.ui.path_df
        end = int(df.timepoint.max()) if df is not None and len(df) else t
        self.sep_to.setValue(max(end, t))

    def add_shortcuts(self, bindings: dict[str, Callable[[], Any]]) -> None:
        """Mirror the viewer's review keys (napari ``Shift-A`` form) here."""
        for key, fn in bindings.items():
            qt_key = key.replace("-", "+").replace("Enter", "Return")
            QShortcut(QKeySequence(qt_key), self).activated.connect(fn)
            if qt_key.endswith("Return"):
                QShortcut(
                    QKeySequence(qt_key.replace("Return", "Enter")), self
                ).activated.connect(fn)


def _row(widgets: list[Any]) -> QHBoxLayout:
    h = QHBoxLayout()
    for w in widgets:
        h.addWidget(w)
    return h


def _verdict_text(decision: Any) -> str:
    if decision is None:
        return ""
    return decision.verdict + (
        f" ({decision.outcome})" if decision.outcome else ""
    )


def _separation_mask(
    n_frames: int, plan: Any
) -> tuple[np.ndarray, tuple[int, int]]:
    """Labels of a separation plan (1 the cell, 2 the rest) and their canvas origin.

    Only the box the plan touches is allocated: a full-canvas layer made
    napari compute every frame of the well and hang.
    """
    y0 = min(fs.y0 for fs in plan.frames)
    x0 = min(fs.x0 for fs in plan.frames)
    y1 = max(fs.y0 + fs.cell.shape[0] for fs in plan.frames)
    x1 = max(fs.x0 + fs.cell.shape[1] for fs in plan.frames)
    out = np.zeros((n_frames, y1 - y0, x1 - x0), dtype=np.uint8)
    for fs in plan.frames:
        h, w = fs.cell.shape
        view = out[
            fs.frame, fs.y0 - y0 : fs.y0 - y0 + h, fs.x0 - x0 : fs.x0 - x0 + w
        ]
        view[fs.other] = 2
        view[fs.cell] = 1
    return out, (y0, x0)


def _hide_labels(levels: list[Any], ids: set[int]) -> list[Any]:
    """Lazy copies of a label pyramid with ``ids`` set to background."""
    import dask.array as da

    lut = np.array(sorted(ids), dtype=np.int64)

    def blank(block: np.ndarray) -> np.ndarray:
        out = block.copy()
        out[np.isin(block, lut)] = 0
        return out

    return [
        (
            da.from_zarr(lv)
            if not isinstance(lv, np.ndarray | da.Array)
            else da.asarray(lv)
        ).map_blocks(blank, dtype=lv.dtype)
        for lv in levels
    ]


def _cell_mask(
    nuclei: Any,
    labels: dict[int, int],
    patches: dict[int, tuple[int, int, np.ndarray]] | None = None,
) -> Any:
    """Lazy ``(T, Y, X)`` uint8 mask of only the cell's nucleus in each frame.

    Frames where the cell is a reviewer-made mask are drawn from its patch.
    """
    patches = patches or {}
    import dask.array as da

    if isinstance(nuclei, da.Array):  # corrected nuclei are lazy already
        arr = nuclei
    elif isinstance(nuclei, np.ndarray):
        arr = da.from_array(nuclei, chunks=(1, *nuclei.shape[1:]))
    else:
        arr = da.from_zarr(nuclei)

    def select(block: np.ndarray, block_info: Any = None) -> np.ndarray:
        t0 = block_info[0]["array-location"][0][0]
        out = np.zeros(block.shape, dtype=np.uint8)
        y_off = block_info[0]["array-location"][1][0]
        x_off = block_info[0]["array-location"][2][0]
        for i in range(block.shape[0]):
            t = t0 + i
            if t in patches:
                py, px, mask = patches[t]
                ys, xs = np.nonzero(mask)
                ys, xs = ys + py - y_off, xs + px - x_off
                ok = (
                    (ys >= 0)
                    & (ys < block.shape[1])
                    & (xs >= 0)
                    & (xs < block.shape[2])
                )
                out[i, ys[ok], xs[ok]] = 1
                continue
            label = labels.get(t)
            if label:
                out[i] = block[i] == label
        return out

    return arr.map_blocks(select, dtype=np.uint8)
