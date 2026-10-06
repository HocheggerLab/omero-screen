"""Track Review widget (v2): check and curate tracked cells, with an agent.

Load a review queue (``cellview review-queue … --out DIR`` writes
``DIR/queue.json``). Selecting an item loads the well from the zarr cache,
jumps to the cell and draws it:

* ``review path``: the cell's curated track;
* ``review marker``: a ring on the cell in every frame, a cross where it has
  no mask;
* ``review cell``: an outline of the cell's own nucleus (lazy, per frame);
* ``review candidates``: numbered nuclei that could continue the cell;
* ``review proposal``: where a pending agent proposal points.

Every change to the data is an edit in ``edits.jsonl`` next to the queue
(link, unlink, events, exclude), applied immediately and replayable. The
verdict on an item (accept / correct / reject / unsure) goes to
``decisions.json``. Agent proposals (written through the MCP tools) appear in
the *Agent proposals* list and change nothing until confirmed here.

Keyboard (napari canvas focused):

=========  ==========================================
Shift-A/C  accept / correct
Shift-R/U  reject / unsure
Shift-N    next break of this cell
Shift-F    toggle follow cell
Shift-L    pick continuation (then click a nucleus)
Shift-1…9  link to numbered candidate
Shift-X    unlink at current frame
Shift-Z    undo last edit for this well
=========  ==========================================
"""

import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from napari.utils import notifications
from napari.viewer import Viewer
from qtpy.QtCore import QFileSystemWatcher, QSettings, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.review.session import (
    ReviewSession,
    set_active_session,
)
from omero_screen_napari.review_queue import (
    QueueError,
    ReviewItem,
    decisions_path,
    latest_decisions,
)

PATH_LAYER = "review path"
MARKER_LAYER = "review marker"
CELL_LAYER = "review cell"
CAND_LAYER = "review candidates"
PROPOSAL_LAYER = "review proposal"
DRAW_LAYER = "review draw"
REVIEW_LAYERS = (
    PATH_LAYER,
    MARKER_LAYER,
    CELL_LAYER,
    CAND_LAYER,
    PROPOSAL_LAYER,
)
NUCLEI_LAYER = "nuclei"
QUEUE_ENV = "OMERO_SCREEN_REVIEW_QUEUE"
DB_ENV = "OMERO_SCREEN_REVIEW_DB"


class TrackReviewWidget(QWidget):  # type: ignore[misc]
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

        self.settings = QSettings("omero-screen", "track-review")
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._on_file_changed)

        # -- queue and database ------------------------------------------------
        last = str(self.settings.value("queue_path", "") or "")
        self.path_edit = QLineEdit(os.environ.get(QUEUE_ENV, "") or last)
        default_db = Path.home() / ".cellview" / "cellview.duckdb"
        self.db_edit = QLineEdit(
            os.environ.get(DB_ENV, "")
            or str(self.settings.value("db_path", "") or "")
            or (str(default_db) if default_db.exists() else "")
        )
        self.db_edit.setPlaceholderText("CellView database (.duckdb)")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        load = QPushButton("Load")
        load.clicked.connect(
            lambda: self.load_queue(Path(self.path_edit.text()))
        )

        # -- item list and cell panel ------------------------------------------
        self.items = QListWidget()
        self.items.currentRowChanged.connect(self._on_row)
        self.details = QLabel("No queue loaded.")
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.RichText)

        # -- navigation ------------------------------------------------------
        prev_btn, next_btn = QPushButton("◀ Cell"), QPushButton("Cell ▶")
        prev_btn.clicked.connect(lambda: self._step(-1))
        next_btn.clicked.connect(lambda: self._step(1))
        brk = QPushButton("Next break")
        brk.clicked.connect(self.next_break)
        self.follow = QCheckBox("Follow cell")
        self.follow.setChecked(True)
        self.isolate = QCheckBox("Isolate")
        self.isolate.toggled.connect(lambda _: self._update_isolation())
        self.zoom = QSpinBox()
        self.zoom.setRange(1, 50)
        self.zoom.setValue(6)
        self.zoom.setPrefix("zoom ")
        strip = QPushButton("Filmstrip")
        strip.clicked.connect(self.show_filmstrip)

        # -- track edits -----------------------------------------------------
        self.pick = QPushButton("Pick continuation")
        self.pick.setCheckable(True)
        self.pick.setToolTip(
            "Then click the nucleus that is this cell in the current frame."
        )
        cand = QPushButton("Candidates")
        cand.clicked.connect(self.show_candidates)
        unlink = QPushButton("Unlink here")
        unlink.clicked.connect(self.unlink_here)
        undo = QPushButton("Undo")
        undo.clicked.connect(self.undo)

        # -- mask edits (current frame) ----------------------------------------------
        mask_buttons = []
        for label, mode, tip in (
            (
                "Add nucleus",
                "absorb",
                "Click a nucleus to add it to this cell in the current frame",
            ),
            (
                "Remove nucleus",
                "drop",
                "Click a nucleus to remove it from its track in the current frame",
            ),
            (
                "Split nucleus",
                "split",
                "Click two seeds: first on this cell's part, then on the other",
            ),
        ):
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, m=mode: self._arm(m))
            mask_buttons.append(btn)
        draw = QPushButton("Draw nucleus")
        draw.setToolTip(
            "Paint a missed nucleus in the 'review draw' layer, then Commit drawing"
        )
        draw.clicked.connect(self.start_drawing)
        commit = QPushButton("Commit drawing")
        commit.clicked.connect(self.commit_drawing)
        self.mode_label = QLabel("")

        # -- events / outcome ------------------------------------------------------
        events = QHBoxLayout()
        for kind in ("mitosis", "death", "slippage"):
            btn = QPushButton(kind.capitalize())
            btn.setToolTip(f"Mark {kind} at the current frame")
            btn.clicked.connect(lambda _=False, k=kind: self.mark_event(k))
            events.addWidget(btn)
        excl = QPushButton("Exclude")
        excl.setToolTip(
            "Exclude the cell; the note (or outcome) is the reason"
        )
        excl.clicked.connect(self.exclude)
        events.addWidget(excl)
        self.outcome = QComboBox()
        self.note = QLineEdit()
        self.note.setPlaceholderText("note")

        # -- agent proposals and edit log ----------------------------------------
        self.proposal_list = QListWidget()
        self.proposal_list.setMaximumHeight(90)
        self.proposal_list.currentRowChanged.connect(
            lambda _: self._draw_proposal()
        )
        confirm, reject = (
            QPushButton("Confirm proposal"),
            QPushButton("Reject proposal"),
        )
        confirm.clicked.connect(lambda: self.resolve_proposal(True))
        reject.clicked.connect(lambda: self.resolve_proposal(False))
        self.log_list = QListWidget()
        self.log_list.setMaximumHeight(90)

        # -- verdicts ----------------------------------------------------------
        verdicts = QHBoxLayout()
        for verdict in ("accept", "correct", "reject", "unsure"):
            btn = QPushButton(verdict.capitalize())
            btn.clicked.connect(lambda _=False, v=verdict: self.decide(v))
            verdicts.addWidget(btn)
        self.status = QLabel("")

        # -- layout ------------------------------------------------------------
        layout = QVBoxLayout(self)
        for row in (
            [self.path_edit, browse, load],
            [self.db_edit],
        ):
            h = QHBoxLayout()
            for w in row:
                h.addWidget(w)
            layout.addLayout(h)
        layout.addWidget(self.items)
        layout.addWidget(self.details)
        for row in (
            [prev_btn, next_btn, brk, strip],
            [self.follow, self.isolate, self.zoom],
            [self.pick, cand, unlink, undo],
            mask_buttons,
            [draw, commit, self.mode_label],
        ):
            h = QHBoxLayout()
            for w in row:
                h.addWidget(w)
            layout.addLayout(h)
        layout.addLayout(events)
        h = QHBoxLayout()
        h.addWidget(QLabel("outcome"))
        h.addWidget(self.outcome)
        layout.addLayout(h)
        layout.addWidget(self.note)
        layout.addWidget(QLabel("Agent proposals"))
        layout.addWidget(self.proposal_list)
        h = QHBoxLayout()
        h.addWidget(confirm)
        h.addWidget(reject)
        layout.addLayout(h)
        layout.addWidget(QLabel("Edits for this cell"))
        layout.addWidget(self.log_list)
        layout.addLayout(verdicts)
        layout.addWidget(self.status)

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
        }
        for n in range(1, 10):
            bindings[f"Shift-{n}"] = partial(self.link_candidate, n)
        for key, fn in bindings.items():
            try:
                self.viewer.bind_key(
                    key, lambda _v, fn=fn: fn(), overwrite=True
                )
            except Exception as err:  # never let a key clash break the widget
                logger.warning(f"Track review: could not bind {key}: {err}")

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

    def _refresh_list(
        self, select: str | None = None, silent: bool = False
    ) -> None:
        assert self.session is not None
        done = latest_decisions(decisions_path(self.session.queue_path))
        pending = {
            p.item for p in self.session.proposals() if p.status == "pending"
        }
        self.items.blockSignals(True)
        self.items.clear()
        for it in self.session.queue.items:
            tag = done[it.id].verdict if it.id in done else "·"
            star = " *" if it.id in pending else ""
            self.items.addItem(f"[{tag}] {it.id}  {it.reason}{star}")
        self.items.blockSignals(False)
        st = self.session.status()
        self.status.setText(
            f"{st['reviewed']}/{st['items']} reviewed · {st['pending_proposals']} proposals pending"
        )
        if select:
            # A silent refresh only restores the highlight; it must not reload
            # the cell (which refreshes this list again).
            self.items.blockSignals(silent)
            self._select(select)
            self.items.blockSignals(False)

    def _select(self, item_id: str) -> None:
        assert self.session is not None
        ids = [i.id for i in self.session.queue.items]
        if item_id in ids:
            self.items.setCurrentRow(ids.index(item_id))

    def _select_first_open(self) -> None:
        assert self.session is not None
        done = latest_decisions(decisions_path(self.session.queue_path))
        open_items = [
            i.id for i in self.session.queue.items if i.id not in done
        ]
        if open_items:
            self._select(open_items[0])

    def _step(self, delta: int) -> None:
        row = self.items.currentRow() + delta
        if 0 <= row < self.items.count():
            self.items.setCurrentRow(row)

    def _on_row(self, row: int) -> None:
        if self.session is None or not 0 <= row < len(
            self.session.queue.items
        ):
            return
        self.go_to(self.session.queue.items[row])

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
        nuclei = read_well(plate_id, well)["nuclei"]
        self._nuclei = nuclei[0] if nuclei else None
        self._loaded = (plate_id, well)

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
        self.note.clear()
        self.outcome.setCurrentText(item.outcome)
        self._redraw()
        t = item.frame if frame is None else frame
        self.viewer.dims.set_current_step(0, t)
        self._centre(t)
        self.viewer.camera.zoom = self.zoom.value()
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
                self.viewer.layers.remove(name)
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

    def _draw_path(self) -> None:
        pts = self._points()
        if not pts:
            return
        scale = (1.0, self._pixel_size, self._pixel_size)
        coords = np.array([[t, y, x] for t, y, x, _ in pts], dtype=float)
        if len(coords) > 1:
            self.viewer.add_tracks(
                np.column_stack([np.zeros(len(coords)), coords]),
                name=PATH_LAYER,
                scale=scale,
                tail_length=30,
                head_length=0,
                colormap="hsv",
                blending="translucent",
            )
        gap = np.array([lab == 0 for *_, lab in pts])
        self.viewer.add_points(
            coords,
            name=MARKER_LAYER,
            scale=scale,
            size=np.where(gap, 18, 40) * self._pixel_size,
            symbol=["cross" if g else "ring" for g in gap],
            face_color=["magenta" if g else "yellow" for g in gap],
            border_width=0,
            out_of_slice_display=False,
        )
        if self._nuclei is not None:
            self.viewer.add_labels(
                _cell_mask(
                    self._nuclei, {t: lab for t, _, _, lab in pts if lab}
                ),
                name=CELL_LAYER,
                scale=scale,
                opacity=1.0,
            ).contour = 2
        self._activate_image()

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
            span = f"frames {int(self.path_df.timepoint.min())}–{int(self.path_df.timepoint.max())}, "
            span += f"{int(self.path_df.gap.sum())} without mask, "
        self.details.setText(
            f"<b>{it.id}</b> — {it.reason or '–'} · automatic: {it.outcome or '–'}<br>"
            f"{span}breaks at: {breaks}<br><i>{it.question}</i>"
        )

    def _centre(self, t: int) -> None:
        pts = self._points()
        if not pts:
            return
        before = [p for p in pts if p[0] <= t] or pts[:1]
        _, y, x, _ = before[-1]
        self.viewer.camera.center = (
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
        """Link the cell to the nucleus at a world ``position`` in the current frame."""
        self.pick.setChecked(False)
        if self.current is None or self._nuclei is None:
            return 0
        t = int(self.viewer.dims.current_step[0])
        y = int(round(position[-2] / self._pixel_size))
        x = int(round(position[-1] / self._pixel_size))
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
        return label if self._edit("link", {"frame": t, "label": label}) else 0

    # -- mask edits ------------------------------------------------------------------

    def _arm(self, mode: str) -> None:
        self._mode, self._seeds = mode, []
        self.mode_label.setText(
            {
                "absorb": "click a nucleus to add",
                "drop": "click a nucleus to remove",
                "split": "click seed 1 (this cell)",
            }[mode]
        )

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
            self._edit(mode, {"frames": [t, t], "label": label})
            return
        # split: collect two seeds on the same raw label
        self._seeds.append((y, x))
        if len(self._seeds) == 1:
            self._split_label = label
            self.mode_label.setText("click seed 2 (the other part)")
            return
        seeds, split_label = self._seeds, getattr(self, "_split_label", 0)
        self._disarm()
        if not split_label:
            notifications.show_warning("Seed 1 was not on a nucleus.")
            return
        from omero_screen_napari.review.masks import MaskError, commit_split

        try:
            commit_split(self.session, self.current.id, t, split_label, seeds)
        except (MaskError, ValueError) as err:
            notifications.show_warning(str(err))
            return
        self._redraw()
        self.viewer.dims.set_current_step(0, t)

    def start_drawing(self) -> None:
        """Add an empty paint layer for the current frame; paint the missed nucleus."""
        if self._nuclei is None:
            return
        if DRAW_LAYER in self.viewer.layers:
            self.viewer.layers.remove(DRAW_LAYER)
        shape = self._nuclei.shape[-2:]
        layer = self.viewer.add_labels(
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
            f"painting frame {int(self.viewer.dims.current_step[0])}"
        )

    def commit_drawing(self) -> None:
        """Turn the painted pixels into a nucleus of this cell in the current frame."""
        if (
            DRAW_LAYER not in self.viewer.layers
            or self.session is None
            or self.current is None
        ):
            notifications.show_info("Start with Draw nucleus.")
            return
        from omero_screen_napari.review.masks import MaskError, commit_drawn

        mask = np.asarray(self.viewer.layers[DRAW_LAYER].data) > 0
        t = int(self.viewer.dims.current_step[0])
        try:
            commit_drawn(self.session, self.current.id, t, mask)
        except (MaskError, ValueError) as err:
            notifications.show_warning(str(err))
            return
        self.viewer.layers.remove(DRAW_LAYER)
        self.mode_label.setText("")
        self._redraw()
        self.viewer.dims.set_current_step(0, t)

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
            self.viewer.layers.remove(CAND_LAYER)
        if self.cands is None or self.cands.empty:
            notifications.show_info("No candidates within reach.")
            return
        coords = self.cands[["timepoint", "y", "x"]].to_numpy(dtype=float)
        self.viewer.add_points(
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
            out_of_slice_display=True,
        )
        self._activate_image()

    def link_candidate(self, rank: int) -> None:
        """Link the cell to numbered candidate ``rank``."""
        if self.cands is None or self.cands.empty or rank > len(self.cands):
            notifications.show_info(
                "Show candidates first (Next break or Candidates)."
            )
            return
        row = self.cands.iloc[rank - 1]
        self._edit(
            "link",
            {"frame": int(row.timepoint), "label": int(row.label)},
            reason=f"candidate {rank}",
        )

    def unlink_here(self) -> None:
        self._edit("unlink", {"frame": int(self.viewer.dims.current_step[0])})

    def mark_event(self, kind: str) -> None:
        self._edit(
            "event",
            {"kind": kind, "frame": int(self.viewer.dims.current_step[0])},
        )

    def exclude(self) -> None:
        reason = (
            self.note.text().strip()
            or self.outcome.currentText()
            or "excluded"
        )
        self._edit("exclude", {"reason": reason})

    def undo(self) -> None:
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
        if self.session is None or self.current is None:
            return []
        return [
            p
            for p in self.session.proposals()
            if p.item == self.current.id and p.status == "pending"
        ]

    def _refresh_proposals(self) -> None:
        self.proposal_list.clear()
        for p in self._pending():
            self.proposal_list.addItem(f"{p.id} {p.op} {p.args} — {p.reason}")
        if self.proposal_list.count():
            self.proposal_list.setCurrentRow(0)
        if self.session is not None:
            self._refresh_list(
                select=self.current.id if self.current else None, silent=True
            )

    def _draw_proposal(self) -> None:
        if PROPOSAL_LAYER in self.viewer.layers:
            self.viewer.layers.remove(PROPOSAL_LAYER)
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
        self.viewer.add_points(
            hit[["timepoint", "y", "x"]].to_numpy(dtype=float),
            name=PROPOSAL_LAYER,
            scale=(1.0, self._pixel_size, self._pixel_size),
            size=46 * self._pixel_size,
            symbol="diamond",
            face_color="transparent",
            border_color="orange",
            border_width=0.1,
            out_of_slice_display=False,
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
        """Record a verdict on the current item and move to the next one."""
        if self.session is None or self.current is None:
            notifications.show_warning("Select a queue item first.")
            return
        self.session.verdict(
            self.current.id,
            verdict,
            self.outcome.currentText(),
            self.note.text().strip(),
        )
        row = self.items.currentRow()
        self._refresh_list()
        self.items.setCurrentRow(min(row + 1, self.items.count() - 1))


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

    arr = (
        da.from_zarr(nuclei)
        if not isinstance(nuclei, np.ndarray)
        else da.from_array(nuclei, chunks=(1, *nuclei.shape[1:]))
    )

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
