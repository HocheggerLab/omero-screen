"""Track view widget: builds headless, draws, picks, and syncs with Track Review.

``QT_QPA_PLATFORM=offscreen`` is enough; pyqtgraph needs no display.

Two optional smoke runs write PNGs for a human to look at:

* ``TRACK_VIEW_PNG_DIR=<dir>``: the synthetic well, zoomed out and in;
* plus ``TRACK_VIEW_QUEUE=<review dir>/queue.json`` (and ``ENV=production``
  or ``TRACK_VIEW_DB``): a real well read (read-only) from CellView and the
  review directory, e.g. plate 5054 C3.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from cellview.tracks.edit import Curated
from omero_screen_napari.review import track_view_model as tvm

WELL = "C2"


@pytest.fixture(scope="module")
def qapp():
    """One offscreen QApplication for the module."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def synthetic_well(
    n_founders: int = 30, frames: int = 60, seed: int = 0
) -> tuple[pd.DataFrame, Curated]:
    """Founders (label = track id) of which a third divide; a few gaps.

    Reporters follow a G1 → S → G2 cycle ending at the division.
    """
    rng = np.random.default_rng(seed)
    rows, tracks, parents = [], {}, {}
    next_id = n_founders + 1

    def add(tid: int, t0: int, t1: int, div_at: int | None) -> None:
        for t in range(t0, t1 + 1):
            if t not in (t0, t1) and rng.random() < 0.03:
                continue  # no mask: a gap
            to_div = (div_at if div_at is not None else 10**6) - t
            pip = 10.0 if to_div > 25 or to_div < 8 else 1.0
            gem = 20.0 if to_div <= 18 else 1.0
            rows.append((t, tid, 100.0, pip, gem))
            tracks[(t, tid)] = tid

    for tid in range(1, n_founders + 1):
        t0 = int(rng.integers(0, 10))
        divides = tid % 3 == 0
        t1 = int(rng.integers(30, 50)) if divides else frames - 1
        add(tid, t0, t1, t1 + 1 if divides else None)
        parents[tid] = 0
        if divides:
            for _ in range(2):
                parents[next_id] = tid
                add(next_id, t1 + 1, frames - 1, None)
                next_id += 1
    det = pd.DataFrame(
        rows, columns=["timepoint", "label", "area", "pip", "geminin"]
    )
    det["y"], det["x"] = 10.0, 10.0
    return det, Curated(tracks=tracks, parents=parents, next_id=next_id)


def _inputs(review_dir: Path | None = None) -> tvm.WellInputs:
    det, curated = synthetic_well()
    records = tvm.review_records(review_dir) if review_dir else {}
    return tvm.WellInputs(
        det=det,
        curated=curated,
        well=WELL,
        edits=[],
        records=records,
        queue={f"{WELL}-t5-L3": {"reason": "lost", "outcome": "lost"}},
        window=(5, 59),
        interval_minutes=20.0,
    )


def _view(qapp, **kwargs):
    from omero_screen_napari.review.track_view import TrackViewWidget

    kwargs.setdefault("slow", None)
    kwargs.setdefault("worker", "inline")
    w = TrackViewWidget([WELL], lambda well: _inputs(), well=WELL, **kwargs)
    w.resize(480, 800)
    w.show()
    qapp.processEvents()
    return w


def test_builds_and_draws(qapp) -> None:
    """Every node is drawn; each colour mode and alignment redraws."""
    w = _view(qapp)
    assert w.table is not None and w.lay is not None
    assert len(w._node_xy) == len(w.table.nodes)
    for mode in tvm.COLOUR_MODES:
        w.colour_box.setCurrentText(mode)
        assert set(w._cats) <= set(tvm.PALETTES[mode])
    for i in range(w.align_box.count()):
        w.align_box.setCurrentIndex(i)
        assert w.lay.tracks["x"].notna().all()
    w.close()


def test_division_alignment_puts_divisions_on_row_zero(qapp) -> None:
    """Every division node lands on y = 0; unaligned trees can be hidden."""
    w = _view(qapp)
    w.align_box.setCurrentIndex(list(tvm.ALIGNMENTS).index("division"))
    lay = w.lay
    assert lay is not None
    mothers = lay.tracks[lay.tracks["ev_division"].notna()]
    assert len(mothers) == 10
    assert (mothers["end"] - mothers["offset"] == 0).all()
    w.park.setChecked(False)
    assert not w.lay.tracks["parked"].any()
    w.close()


def test_filters_from_the_toolbar(qapp) -> None:
    """Toolbar filters reach the layout."""
    w = _view(qapp)
    w.min_len.setValue(40)
    assert (w.lay.tracks["n_frames"] >= 40).all()
    w.flagged.setChecked(True)
    assert set(w.lay.tracks.index) == {3}  # its daughters are too short
    w.min_len.setValue(5)
    assert set(w.lay.tracks.index) == {3, 31, 32}  # with its daughters
    w.close()


def test_zoom_shows_markers_then_labels(qapp) -> None:
    """Level of detail: squares when zoomed out, markers and labels when in."""
    w = _view(qapp)
    w.zoom_to(-0.5, 200, 0, 60)
    qapp.processEvents()
    assert w._lod is not None and not w._lod[0]
    assert not w._labels
    w.zoom_to(-0.5, 3.5, 0, 20)
    qapp.processEvents()
    assert w._lod[0]
    assert w._labels
    w.close()


def test_pick_calls_back_and_highlights(qapp) -> None:
    """A picked node opens the cell and is highlighted."""
    picked = []
    w = _view(qapp, on_pick=lambda *a: picked.append(a))
    n = w.lay.nodes
    i = int(np.flatnonzero((n["track"] == 3).to_numpy())[2])
    w.pick(i)
    frame = int(n.iloc[i]["frame"])
    assert picked == [(WELL, frame, 3)]
    assert w._highlight == (WELL, 3, frame)
    xs, _ = w.track_hl.getData()
    assert xs is not None and len(xs) == 2
    w.set_frame(frame + 1)
    assert w._highlight == (WELL, 3, frame + 1)
    assert "track 3" in w.describe(i)
    w.close()


def test_click_on_canvas_finds_the_node(qapp) -> None:
    """A click within a few pixels of a node selects it."""
    picked = []
    w = _view(qapp, on_pick=lambda *a: picked.append(a))
    w.zoom_to(-0.5, 3.5, 0, 20)
    qapp.processEvents()
    x, y = w._node_xy[5]
    assert w.node_at(x + 0.01, y) == 5
    assert w.node_at(x + 0.5, y + 5) is None
    w.close()


def test_reload_picks_up_new_verdicts(qapp, tmp_path: Path) -> None:
    """A decisions file written later shows after a reload (as the watcher does)."""
    from omero_screen_napari.review.track_view import TrackViewWidget

    w = TrackViewWidget(
        [WELL],
        lambda well: _inputs(tmp_path),
        slow=None,
        watch=[tmp_path],
        well=WELL,
        worker="inline",
    )
    assert w.table is not None and (w.table.tracks["verdict"] == "").all()
    (tmp_path / "decisions.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {"id": f"{WELL}-t5-L3", "verdict": "accept"},
                ]
            }
        )
    )
    w._on_watched(str(tmp_path))
    assert str(tmp_path / "decisions.json") in w.watcher.files()
    w.reload()
    assert w.table.tracks.loc[3, "verdict"] == "accept"
    w.close()


def test_slow_annotations_arrive_from_the_worker(qapp) -> None:
    """Phases computed in the background replace the walker's when done."""

    def slow(inputs: tvm.WellInputs) -> tvm.SlowAnnotations:
        return tvm.SlowAnnotations(
            phases=pd.DataFrame({"track": [1], "frame": [5], "phase": ["E"]})
        )

    w = _view(qapp, slow=slow, worker="thread")
    w.wait_for_annotations()
    assert w.table is not None and w.table.has_hmm
    w.close()


def test_slow_annotations_in_a_child_process(qapp) -> None:
    """The default: walker and HMM run in a child Python process."""
    w = _view(
        qapp, slow=tvm.run_slow_annotations_subprocess, worker="thread"
    )
    w.wait_for_annotations(timeout=300)
    assert not w._slow[WELL].errors
    assert w.table is not None and w.table.has_hmm
    assert w.table.tracks["outcome"].ne("").any()
    w.close()


def test_png_smoke(qapp, tmp_path: Path) -> None:
    """Render the synthetic well zoomed out and in (kept if TRACK_VIEW_PNG_DIR is set)."""
    out = Path(os.environ.get("TRACK_VIEW_PNG_DIR", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    w = _view(qapp)
    w.fit()
    qapp.processEvents()
    assert w.grab().save(str(out / "synthetic_overview.png"))
    w.align_box.setCurrentIndex(list(tvm.ALIGNMENTS).index("division"))
    w.zoom_to(-0.5, 5.5, -25, 12)
    qapp.processEvents()
    assert w.grab().save(str(out / "synthetic_zoom.png"))
    w.close()


@pytest.mark.skipif(
    not os.environ.get("TRACK_VIEW_QUEUE"),
    reason="set TRACK_VIEW_QUEUE to a review queue to smoke-test real data",
)
def test_real_well_smoke(qapp, tmp_path: Path) -> None:
    """A real review directory, read-only: build, annotate, render."""
    from omero_screen_napari.review.track_view import TrackViewWidget

    queue = Path(os.environ["TRACK_VIEW_QUEUE"])
    db = os.environ.get("TRACK_VIEW_DB")
    wells, load = tvm.directory_loader(queue, Path(db) if db else None)
    out = Path(os.environ.get("TRACK_VIEW_PNG_DIR", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    w = TrackViewWidget(wells, load, watch=[queue.parent])
    w.resize(480, 900)
    w.show()
    w.wait_for_annotations(timeout=300)
    qapp.processEvents()
    assert w.table is not None and len(w.table.tracks)
    tag = w.well
    w.fit()
    qapp.processEvents()
    w.grab().save(str(out / f"{tag}_overview.png"))
    w.verdict_box.setCurrentText("accepted")
    w.align_box.setCurrentIndex(list(tvm.ALIGNMENTS).index("division"))
    lay = w.lay
    assert lay is not None
    aligned = lay.tracks[~lay.tracks["parked"]]
    # Every aligned founder that divides has its division on row 0.
    roots = aligned[aligned["parent"] == 0]
    divs = roots[roots["ev_division"].notna()]
    assert ((divs["ev_division"] - divs["offset"]) == 0).all()
    w.zoom_to(-0.5, 5.5, -30, 15)
    qapp.processEvents()
    w.grab().save(str(out / f"{tag}_accepted_division_zoom.png"))
    w.close()


# -- wiring in the Track Review widget ------------------------------------------


def test_review_widget_docks_view_and_opens_picked_cell(
    qapp, tmp_path: Path
) -> None:
    """Track view button docks the view; a picked node opens that cell."""
    from omero_screen_napari._review_widget import TrackReviewWidget
    from omero_screen_napari.review.session import ReviewSession
    from omero_screen_napari.review.track_view import TrackViewWidget
    from omero_screen_napari.review_queue import ReviewItem, write_queue

    det, curated = synthetic_well()
    q = tmp_path / "queue.json"
    write_queue(q, 5054, [ReviewItem(id=f"{WELL}-t5-L3", well=WELL, frame=5)])
    viewer = MagicMock()
    viewer.dims.current_step = (5, 0, 0)
    viewer.layers.__contains__.return_value = False
    viewer.layers.__iter__.return_value = iter([])
    settings = MagicMock()
    settings.value.return_value = ""
    patches = [
        patch("omero_screen_napari._review_widget.QSettings", return_value=settings),
        patch.object(TrackReviewWidget, "_ensure_well"),
        patch.object(TrackReviewWidget, "_redraw"),
        patch("omero_screen_napari._review_widget.notifications"),
        patch(
            "omero_screen_napari.review.cells.CellSource.well",
            return_value=(det, curated),
        ),
        patch(
            "omero_screen_napari.review.track_view_model.slow_annotations",
            return_value=tvm.SlowAnnotations(),
        ),
    ]
    for p in patches:
        p.start()
    try:
        widget = TrackReviewWidget(napari_viewer=viewer)
        widget.load_queue(q)
        assert isinstance(widget.session, ReviewSession)
        widget.show_track_view()
        view = widget.track_view
        assert isinstance(view, TrackViewWidget)
        viewer.window.add_dock_widget.assert_called_once()
        view.slow_fn = None
        # Pick a node of founder 6 (not in the queue): an ad-hoc item opens.
        n = view.lay.nodes
        i = int(np.flatnonzero((n["track"] == 6).to_numpy())[3])
        frame = int(n.iloc[i]["frame"])
        view.pick(i)
        assert widget.current.id == f"{WELL}-t{frame}-L6"
        viewer.dims.set_current_step.assert_called_with(0, frame)
        # A node of the queued cell opens the queue item.
        i = int(np.flatnonzero((n["track"] == 3).to_numpy())[4])
        view.pick(i)
        assert widget.current.id == f"{WELL}-t5-L3"
        # Showing again reuses the dock.
        widget.show_track_view()
        viewer.window.add_dock_widget.assert_called_once()
        view.close()
    finally:
        for p in patches:
            p.stop()


class _ReadOnlyLoader:
    """A loader for a well the review session does not edit."""

    def __call__(self, well: str) -> tvm.WellInputs:
        return _inputs()

    def interactive(self, well: str) -> bool:
        return False


def test_read_only_well_navigates_but_never_opens(qapp) -> None:
    """A node of another well moves the viewer at most; the cell is described."""
    from omero_screen_napari.review.track_view import TrackViewWidget

    picked, moved = [], []
    w = TrackViewWidget(
        [WELL],
        _ReadOnlyLoader(),
        slow=None,
        worker="inline",
        on_pick=lambda *a: picked.append(a),
        on_navigate=lambda *a: moved.append(a) or False,
    )
    w.pick(3)
    assert not picked
    assert len(moved) == 1 and moved[0][0] == WELL
    assert "Read-only well" in w.status.text()
    assert "load well" in w.status.text()
    w.close()
