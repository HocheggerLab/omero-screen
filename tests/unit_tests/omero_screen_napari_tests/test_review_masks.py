"""Reviewer masks: watershed split, background-exact measurement, painting patches."""

import numpy as np
import pandas as pd
import pytest

from omero_screen_napari.review.masks import (
    MaskError,
    channel_map,
    load_patch,
    measure,
    paint_patches,
    save_patch,
    split_mask,
)


def test_split_mask_follows_the_seeds() -> None:
    """Two touching discs split along the waist, one part per seed."""
    yy, xx = np.mgrid[0:40, 0:60]
    region = ((yy - 20) ** 2 + (xx - 18) ** 2 < 100) | ((yy - 20) ** 2 + (xx - 40) ** 2 < 100)
    a, b = split_mask(region, [(20, 18), (20, 40)])
    assert a[20, 18] and b[20, 40] and not (a & b).any() and ((a | b) == region).all()
    with pytest.raises(MaskError):
        split_mask(region, [(0, 0), (20, 40)])


def test_measure_recovers_the_pipeline_background() -> None:
    """A new mask's mean equals raw mean minus the frame background implied by a measured nucleus."""
    image = np.full((1, 1, 50, 50), 100.0)  # background 100
    image[0, 0, 5:15, 5:15] = 600.0  # reference nucleus, raw 600 → measured 500
    image[0, 0, 30:40, 30:40] = 400.0  # new nucleus, raw 400 → expect 300
    nuclei = np.zeros((1, 50, 50), np.uint32)
    nuclei[0, 5:15, 5:15] = 3
    det_frame = pd.DataFrame({"label": [3], "area": [100.0], "y": [9.5], "x": [9.5], "pip": [500.0]})
    mask = np.ones((10, 10), bool)
    out = measure(image, nuclei, det_frame, 0, 30, 30, mask, {"pip": 0})
    assert out["area"] == 100 and out["y"] == 34.5 and out["pip"] == pytest.approx(300.0)


def test_patches_round_trip_and_paint(tmp_path) -> None:
    """Saved patches load back and are painted (replaced labels cleared) into a crop."""
    from cellview.tracks.edit import Curated

    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    rel = save_patch(tmp_path / "masks", "C2", 5, 900, 10, 20, mask)
    y0, x0, back = load_patch(tmp_path, rel)
    assert (y0, x0) == (10, 20) and (back == mask).all()
    cur = Curated(tracks={}, parents={}, extras={(5, 900): {"patch": rel}}, removed={(5, 7)})
    crop = np.full((8, 8), 7, np.uint32)
    painted = paint_patches(crop, 5, 8, 18, cur, tmp_path)
    assert painted[3, 3] == 900 and painted[0, 0] == 0
    assert paint_patches(crop, 6, 8, 18, cur, tmp_path)[0, 0] == 7  # other frames untouched


def test_channel_map_prefix() -> None:
    """Detection columns map to image channels by prefix, either way round."""
    assert channel_map(["geminin", "pip", "spyDNA_nucleus", "bf_cell"], ["pip", "geminin", "spydna"]) == {
        "pip": 1, "geminin": 0, "spydna": 2}


def test_channel_map_dapi_is_the_nuclear_stain() -> None:
    """CellView's ``dapi`` column is whatever channel carries the nucleus role."""
    names = ["geminin", "pip", "spyDNA_nucleus", "bf_cell"]
    assert channel_map(names, ["dapi", "bf"]) == {"dapi": 2, "bf": 3}


def _merged_well():
    """A PIP-bright cell (label 1) whose mask swallows a reporter-negative
    object from t3; label 9 is a bystander nucleus for the background."""
    import numpy as np
    import pandas as pd

    from cellview.tracks.edit import Curated

    T, H, W = 6, 60, 80
    yy, xx = np.mgrid[:H, :W]
    cell = (yy - 30) ** 2 + (xx - 20) ** 2 <= 36
    corpse = (yy - 30) ** 2 + (xx - 33) ** 2 <= 36
    bystander = (yy - 10) ** 2 + (xx - 70) ** 2 <= 16
    nuclei = np.zeros((T, H, W), dtype=np.uint32)
    image = np.full((T, 3, H, W), 100.0)
    for t in range(T):
        nuclei[t][cell | (corpse if t >= 3 else False)] = 1
        nuclei[t][bystander] = 9
        image[t, 0][cell] += 1000  # pip
        image[t, 0][bystander] += 500
        image[t, 2][cell | corpse] += 800  # dna
    rows = []
    for t in range(T):
        m = nuclei[t] == 1
        ys, xs = np.nonzero(m)
        rows.append((t, 1, ys.mean(), xs.mean(), m.sum(), image[t, 0][m].mean() - 100, 0.0, 0.0))
        rows.append((t, 9, 10.0, 70.0, bystander.sum(), 500.0, 0.0, 0.0))
    det = pd.DataFrame(rows, columns=["timepoint", "label", "y", "x", "area", "pip", "geminin", "dapi"])
    curated = Curated(
        tracks={**{(t, 1): 1 for t in range(T)}, **{(t, 9): 9 for t in range(T)}},
        parents={1: 0, 9: 0},
        next_id=10,
    )
    return image, nuclei, det, curated, cell


def test_separation_keeps_the_reporter_positive_part(tmp_path, monkeypatch) -> None:
    """From the merge on, the cell keeps its own pixels; the rest is split off
    and written as one new track, undone in one step."""
    from omero_screen_napari.review import masks
    from omero_screen_napari.review.session import ReviewSession
    from omero_screen_napari.review_queue import ReviewItem, write_queue

    image, nuclei, det, curated, cell = _merged_well()
    q = tmp_path / "queue.json"
    write_queue(q, 1, [ReviewItem(id="C2-t0-L1", well="C2", frame=0)])
    session = ReviewSession(q)
    monkeypatch.setattr(masks, "_arrays", lambda s, w: (image, nuclei, ["pip", "geminin", "dna"]))
    monkeypatch.setattr(session.source, "base", lambda well: curated)
    monkeypatch.setattr(session.source, "well", lambda well: (det, curated))

    plan = masks.plan_separation(session, "C2-t0-L1", 2, 5)
    assert plan.skipped == {2: "nothing merged"}
    assert [f.frame for f in plan.frames] == [3, 4, 5]
    for fs in plan.frames:
        canvas = np.zeros(cell.shape, bool)
        h, w = fs.cell.shape
        canvas[fs.y0 : fs.y0 + h, fs.x0 : fs.x0 + w] = fs.cell
        overlap = (canvas & cell).sum() / (canvas | cell).sum()
        assert overlap > 0.8
        assert fs.other.sum() > 50

    entries = masks.commit_separation(session, plan)
    assert len(entries) == 6
    assert {e.args["batch"] for e in entries} == {entries[0].args["batch"]}
    cell_adds = [e for e in entries if e.args.get("replaces")]
    assert all(e.args["replaces"] == [1] for e in cell_adds)
    assert all(e.args["measurements"]["pip"] > 800 for e in cell_adds)
    others = [e for e in entries if not e.args.get("replaces")]
    assert others[0].args["cell"] is None
    assert all(e.args["cell"] == [3, others[0].args["label"]] for e in others[1:])

    session.undo("C2-t0-L1")
    from cellview.tracks.edit import EditLog, replay

    after = replay(curated, EditLog(session.edits_path), "C2")
    assert not after.extras
