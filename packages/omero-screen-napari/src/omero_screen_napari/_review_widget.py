"""Track review widget: step through cells an agent has queued for inspection.

An analysis agent writes a queue file (:mod:`omero_screen_napari.review_queue`)
listing cells to check: a well, a frame, the cell's path and a question. This
widget watches the file. Selecting an item loads the well from the zarr cache
if needed, jumps to the frame, centres on the cell and marks it:

* ``review path``: the cell's track as a napari Tracks layer;
* ``review marker``: a ring on the cell in every frame, and a cross where the
  cell has no mask (a gap the agent bridged or a frame segmentation missed);
* ``review cell``: an outline of the cell's own nucleus mask, computed lazily
  frame by frame from the cached labels. *Isolate* hides every other nucleus.

The reviewer records a verdict, optionally a corrected outcome, frames of
interest and a note; each verdict is appended to ``decisions.json`` beside the
queue, which the agent reads back. When the agent sets ``focus`` in the queue,
the widget jumps there as soon as the file changes, which makes a quick "look
at this cell" in conversation a one-line edit.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from napari.utils import notifications
from napari.viewer import Viewer
from qtpy.QtCore import QFileSystemWatcher
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.review_queue import (
    Decision,
    Queue,
    QueueError,
    ReviewItem,
    decisions_path,
    latest_decisions,
    read_queue,
    record_decision,
)

PATH_LAYER = "review path"
MARKER_LAYER = "review marker"
CELL_LAYER = "review cell"
NUCLEI_LAYER = "nuclei"
QUEUE_ENV = "OMERO_SCREEN_REVIEW_QUEUE"


class TrackReviewWidget(QWidget):  # type: ignore[misc]
    """Dock widget driving a review session from a queue file."""

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.queue: Queue | None = None
        self.queue_path: Path | None = None
        self.current: ReviewItem | None = None
        self.marked: list[int] = []
        self._loaded: tuple[int, str] | None = None
        self._pixel_size = 1.0
        self._nuclei: Any = None
        self._focus_seen: str | None = None

        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._on_file_changed)

        self.path_edit = QLineEdit(os.environ.get(QUEUE_ENV, ""))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        load = QPushButton("Load")
        load.clicked.connect(
            lambda: self.load_queue(Path(self.path_edit.text()))
        )

        self.items = QListWidget()
        self.items.currentRowChanged.connect(self._on_row)
        self.details = QLabel("No queue loaded.")
        self.details.setWordWrap(True)

        prev_btn, next_btn = QPushButton("◀ Prev"), QPushButton("Next ▶")
        prev_btn.clicked.connect(lambda: self._step(-1))
        next_btn.clicked.connect(lambda: self._step(1))
        self.isolate = QCheckBox("Isolate cell")
        self.isolate.toggled.connect(lambda _: self._update_isolation())
        self.zoom = QSpinBox()
        self.zoom.setRange(1, 50)
        self.zoom.setValue(6)
        self.zoom.setPrefix("zoom ")

        self.outcome = QComboBox()
        mark = QPushButton("Mark frame")
        mark.clicked.connect(self._mark_frame)
        self.marked_label = QLabel("marked: –")
        self.note = QLineEdit()
        self.note.setPlaceholderText("note")

        verdicts = QHBoxLayout()
        for verdict in ("accept", "correct", "reject", "unsure"):
            btn = QPushButton(verdict.capitalize())
            btn.clicked.connect(lambda _=False, v=verdict: self.decide(v))
            verdicts.addWidget(btn)

        self.status = QLabel("")

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(self.path_edit)
        row.addWidget(browse)
        row.addWidget(load)
        layout.addLayout(row)
        layout.addWidget(self.items)
        layout.addWidget(self.details)
        nav = QHBoxLayout()
        for w in (prev_btn, next_btn, self.isolate, self.zoom):
            nav.addWidget(w)
        layout.addLayout(nav)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("outcome"))
        out_row.addWidget(self.outcome)
        out_row.addWidget(mark)
        layout.addLayout(out_row)
        layout.addWidget(self.marked_label)
        layout.addWidget(self.note)
        layout.addLayout(verdicts)
        layout.addWidget(self.status)

        if self.path_edit.text():
            self.load_queue(Path(self.path_edit.text()))

    # -- queue -----------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Review queue", "", "JSON (*.json)"
        )
        if path:
            self.path_edit.setText(path)
            self.load_queue(Path(path))

    def load_queue(self, path: Path) -> None:
        """Read (or re-read) the queue and refresh the item list."""
        try:
            queue = read_queue(path)
        except QueueError as err:
            notifications.show_warning(str(err))
            return
        keep = self.current.id if self.current else None
        self.queue, self.queue_path = queue, Path(path)
        if str(path) not in self.watcher.files():
            self.watcher.addPath(str(path))
        self.outcome.clear()
        self.outcome.addItems(["", *queue.outcomes])
        self._refresh_list(select=keep)
        if queue.focus and queue.focus != self._focus_seen:
            self._focus_seen = queue.focus
            self._select(queue.focus)

    def _on_file_changed(self, path: str) -> None:
        # Atomic replacement drops the watch; re-add before re-reading.
        if Path(path).exists() and path not in self.watcher.files():
            self.watcher.addPath(path)
        if Path(path).exists():
            self.load_queue(Path(path))

    def _refresh_list(self, select: str | None = None) -> None:
        assert self.queue is not None and self.queue_path is not None
        done = latest_decisions(decisions_path(self.queue_path))
        self.items.blockSignals(True)
        self.items.clear()
        for item in self.queue.items:
            tag = done[item.id].verdict if item.id in done else "·"
            self.items.addItem(f"[{tag}] {item.id}  {item.reason}")
        self.items.blockSignals(False)
        n_done = sum(i.id in done for i in self.queue.items)
        self.status.setText(f"{n_done}/{len(self.queue.items)} reviewed")
        if select:
            self._select(select)

    def _select(self, item_id: str) -> None:
        assert self.queue is not None
        ids = [i.id for i in self.queue.items]
        if item_id in ids:
            self.items.setCurrentRow(ids.index(item_id))

    def _step(self, delta: int) -> None:
        row = self.items.currentRow() + delta
        if 0 <= row < self.items.count():
            self.items.setCurrentRow(row)

    def _on_row(self, row: int) -> None:
        if self.queue is None or not 0 <= row < len(self.queue.items):
            return
        self.go_to(self.queue.items[row])

    # -- navigation ------------------------------------------------------

    def _ensure_well(self, plate_id: int, well: str) -> None:
        if self._loaded == (plate_id, well):
            return
        from omero_screen_napari.zarr_cache.display import load_plate_to_viewer
        from omero_screen_napari.zarr_cache.reader import plate_info, read_well

        loaded = load_plate_to_viewer(self.viewer, plate_id, well)
        if not loaded:
            raise QueueError(
                f"Well {well} of plate {plate_id} is not in the zarr cache."
            )
        self._pixel_size = float(
            plate_info(plate_id).get("pixel_size_um") or 1.0
        )
        nuclei = read_well(plate_id, well)["nuclei"]
        self._nuclei = nuclei[0] if nuclei else None
        self._loaded = (plate_id, well)

    def go_to(self, item: ReviewItem) -> None:
        """Load the item's well, jump to its frame and mark the cell."""
        assert self.queue is not None
        try:
            self._ensure_well(self.queue.plate_id, item.well)
        except QueueError as err:
            notifications.show_warning(str(err))
            return
        self.current = item
        self.marked = []
        self.marked_label.setText("marked: –")
        self.note.clear()
        self.outcome.setCurrentText(item.outcome)
        self.details.setText(
            f"<b>{item.id}</b> — well {item.well}, frame {item.frame}<br>"
            f"reason: {item.reason or '–'} · automatic call: {item.outcome or '–'}<br>"
            f"{item.question}"
        )
        self._draw(item)
        self.viewer.dims.set_current_step(0, item.frame)
        point = item.point_at(item.frame)
        if point is not None:
            px = self._pixel_size
            self.viewer.camera.center = (point.y * px, point.x * px)
            self.viewer.camera.zoom = self.zoom.value()
        self._update_isolation()

    def _draw(self, item: ReviewItem) -> None:
        for name in (PATH_LAYER, MARKER_LAYER, CELL_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        if not item.path:
            return
        scale = (1.0, self._pixel_size, self._pixel_size)
        coords = np.array([[p.t, p.y, p.x] for p in item.path], dtype=float)
        tracks = np.column_stack([np.zeros(len(coords)), coords])
        self.viewer.add_tracks(
            tracks,
            name=PATH_LAYER,
            scale=scale,
            tail_length=30,
            head_length=0,
            colormap="hsv",
            blending="translucent",
        )
        gap = np.array([p.label == 0 for p in item.path])
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
                _cell_mask(self._nuclei, item),
                name=CELL_LAYER,
                scale=scale,
                opacity=1.0,
            ).contour = 2
        # Keep the image the active layer so clicks do not edit the markers.
        self.viewer.layers.selection.active = self.viewer.layers[0]

    def _update_isolation(self) -> None:
        """*Isolate* hides every nucleus label except the reviewed cell's outline."""
        if NUCLEI_LAYER in self.viewer.layers:
            self.viewer.layers[
                NUCLEI_LAYER
            ].visible = not self.isolate.isChecked()

    # -- decisions -------------------------------------------------------

    def _mark_frame(self) -> None:
        t = int(self.viewer.dims.current_step[0])
        if t not in self.marked:
            self.marked.append(t)
        self.marked_label.setText(
            f"marked: {', '.join(map(str, sorted(self.marked)))}"
        )

    def decide(self, verdict: str) -> None:
        """Record a verdict on the current item and move to the next one."""
        if self.current is None or self.queue_path is None:
            notifications.show_warning("Select a queue item first.")
            return
        decision = Decision(
            id=self.current.id,
            verdict=verdict,
            outcome=self.outcome.currentText(),
            frames=sorted(self.marked),
            note=self.note.text().strip(),
        )
        record_decision(decisions_path(self.queue_path), decision)
        logger.info(
            f"Review {decision.id}: {verdict} ({decision.outcome or 'no outcome'})"
        )
        row = self.items.currentRow()
        self._refresh_list()
        self.items.setCurrentRow(min(row + 1, self.items.count() - 1))


def _cell_mask(nuclei: Any, item: ReviewItem) -> Any:
    """Lazy ``(T, Y, X)`` uint8 mask of only the item's nucleus in each frame.

    The cached nucleus labels carry raw track ids, and the item's path names
    the label the cell has in each frame, so a frame is one comparison. Built
    with dask so only frames the viewer actually shows are read.
    """
    import dask.array as da

    labels = {p.t: p.label for p in item.path if p.label}
    arr = da.from_zarr(nuclei)

    def select(block: np.ndarray, block_info: Any = None) -> np.ndarray:
        t0 = block_info[0]["array-location"][0][0]
        out = np.zeros(block.shape, dtype=np.uint8)
        for i in range(block.shape[0]):
            label = labels.get(t0 + i)
            if label:
                out[i] = block[i] == label
        return out

    return arr.map_blocks(select, dtype=np.uint8)
