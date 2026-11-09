"""Reviewer masks: watershed split, background-exact measurement, painting patches."""

import numpy as np
import pandas as pd
import pytest

from omero_screen_napari.review.masks import (
    MaskError,
    cell_part,
    channel_map,
    cut_mask,
    label_on_segment,
    load_patch,
    measure,
    paint_patches,
    reference_position,
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
    from omero_screen_track.edit import Curated

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

    from omero_screen_track.edit import Curated

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
    from omero_screen_track.edit import EditLog, replay

    after = replay(curated, EditLog(session.edits_path), "C2")
    assert not after.extras


# -- split along a line ---------------------------------------------------------


def _two_touching_nuclei() -> np.ndarray:
    """Two equal discs overlapping broadly (no neck): centres 14 px apart, r 10."""
    yy, xx = np.mgrid[0:40, 0:60]
    return ((yy - 20) ** 2 + (xx - 23) ** 2 < 100) | (
        (yy - 20) ** 2 + (xx - 37) ** 2 < 100
    )


def test_cut_mask_follows_the_line_not_the_outline() -> None:
    """A vertical line through the contact cuts two broad-touching discs in half,
    wherever the outline's waist is; the parts cover the mask exactly.
    """
    region = _two_touching_nuclei()
    a, b = cut_mask(region, (5, 30), (35, 30))
    assert not (a & b).any() and ((a | b) == region).all()
    assert (
        abs(int(a.sum()) - int(b.sum())) <= region[:, 30].sum()
    )  # line pixels only
    left = a if a[20, 20] else b
    assert left[20, 20] and not left[20, 40]
    # A slanted line: pixels go to the side they are on.
    a, b = cut_mask(region, (0, 25), (40, 35))
    assert a[20, 20] != b[20, 20] and ((a | b) == region).all()


def test_cut_mask_line_pixels_go_to_the_smaller_side() -> None:
    """Pixels exactly on the line join the part that is smaller without them."""
    region = np.zeros((10, 10), bool)
    region[2:8, 2:8] = True
    a, b = cut_mask(region, (0, 3), (9, 3))  # x = 3: columns 2 | 3 | 4..7
    small = a if a[5, 2] else b
    assert small[:, 3].sum() == 6 and small.sum() == 12


def test_cut_mask_keeps_every_piece_of_a_side() -> None:
    """A concave mask cut by a line keeps both pieces on one side in one part."""
    region = np.zeros((20, 20), bool)
    region[2:18, 2:6] = True  # U shape: two arms joined at the bottom
    region[2:18, 14:18] = True
    region[14:18, 2:18] = True
    a, b = cut_mask(region, (10, 0), (10, 19))  # horizontal line y = 10
    top = a if a[3, 3] else b
    assert top[3, 3] and top[3, 15] and not top[16, 10]


def test_cut_mask_errors() -> None:
    """Coincident points or a line missing the nucleus are refused."""
    region = _two_touching_nuclei()
    with pytest.raises(MaskError, match="coincide"):
        cut_mask(region, (5, 5), (5, 5))
    with pytest.raises(MaskError, match="does not cross"):
        cut_mask(region, (0, 59), (39, 59))


def test_cell_part_nearest_reference_else_larger() -> None:
    """The cell's part is the one nearest its previous position; else the larger."""
    region = _two_touching_nuclei()
    parts = cut_mask(region, (5, 30), (35, 30))
    i = cell_part(parts, 100, 200, (120.0, 240.0))  # canvas: right disc
    assert parts[i][20, 40]
    i = cell_part(parts, 100, 200, (120.0, 220.0))
    assert parts[i][20, 20]
    small = cut_mask(region, (5, 15), (35, 15))
    assert small[cell_part(small, 0, 0, None)].sum() > region.sum() / 2


def test_reference_position_nearest_earlier_then_later() -> None:
    """The track's position in its nearest earlier frame, else nearest later."""
    from omero_screen_track.edit import Curated

    det = pd.DataFrame(
        {
            "timepoint": [3, 5, 5, 9],
            "label": [1, 1, 2, 1],
            "y": [0.0, 10.0, 20.0, 99.0],
            "x": [0.0, 10.0, 10.0, 99.0],
            "area": [10.0, 10.0, 30.0, 10.0],
        }
    )
    cur = Curated(
        tracks={(3, 1): 1, (5, 1): 1, (5, 2): 1, (9, 1): 1}, parents={1: 0}
    )
    assert reference_position(det, cur, 1, 7) == (
        17.5,
        10.0,
    )  # t5, area-weighted
    assert reference_position(det, cur, 1, 2) == (
        0.0,
        0.0,
    )  # nothing earlier: t3
    only = Curated(tracks={(9, 1): 1}, parents={1: 0})
    assert reference_position(det, only, 1, 9) is None


def test_label_on_segment_takes_the_nucleus_under_the_line() -> None:
    """Clicks on the outline may miss the mask; the label under the segment counts."""
    labels = np.zeros((20, 20), np.uint32)
    labels[5:15, 5:15] = 7
    assert label_on_segment(labels, (0, 10), (19, 10)) == 7
    assert label_on_segment(labels, (0, 0), (0, 19)) == 0


def _touching_well():
    """Cell (track 1, label 1) at the left and neighbour (track 2, label 2) at the
    right in t0–1; at t2 they are one mask with label 2, owned by the neighbour.
    """
    from omero_screen_track.edit import Curated

    T, H, W = 3, 60, 80
    yy, xx = np.mgrid[:H, :W]
    left = (yy - 30) ** 2 + (xx - 30) ** 2 < 100
    right = (yy - 30) ** 2 + (xx - 44) ** 2 < 100
    bystander = (yy - 8) ** 2 + (xx - 70) ** 2 <= 16
    nuclei = np.zeros((T, H, W), np.uint32)
    image = np.full((T, 1, H, W), 100.0)
    rows = []
    for t in range(T):
        if t < 2:
            nuclei[t][right] = 2
            nuclei[t][left & ~right] = 1
        else:
            nuclei[t][left | right] = 2
        nuclei[t][bystander] = 9
        image[t, 0][left | right] += 500
        image[t, 0][bystander] += 500
        for lab in np.unique(nuclei[t]):
            if lab:
                m = nuclei[t] == lab
                ys, xs = np.nonzero(m)
                rows.append(
                    (t, int(lab), ys.mean(), xs.mean(), float(m.sum()), 500.0)
                )
    det = pd.DataFrame(
        rows, columns=["timepoint", "label", "y", "x", "area", "dna"]
    )
    curated = Curated(
        tracks={
            **{(t, 1): 1 for t in range(2)},
            **{(t, 2): 2 for t in range(T)},
            **{(t, 9): 9 for t in range(T)},
        },
        parents={1: 0, 2: 0, 9: 0},
        next_id=10,
    )
    return image, nuclei, det, curated


def _split_session(tmp_path, monkeypatch, item_id="C2-t0-L1"):
    from omero_screen_napari.review import masks
    from omero_screen_napari.review.session import ReviewSession
    from omero_screen_napari.review_queue import ReviewItem, write_queue

    image, nuclei, det, curated = _touching_well()
    q = tmp_path / "queue.json"
    write_queue(q, 1, [ReviewItem(id=item_id, well="C2", frame=0)])
    session = ReviewSession(q)
    monkeypatch.setattr(
        masks, "_arrays", lambda s, w: (image, nuclei, ["dna"])
    )
    monkeypatch.setattr(session.source, "base", lambda well: curated)

    def well(_):
        from omero_screen_track.cells import apply_extras
        from omero_screen_track.edit import EditLog, replay

        cur = replay(curated, EditLog(session.edits_path), "C2")
        return apply_extras(det, cur), cur

    monkeypatch.setattr(session.source, "well", well)
    return session, curated


def test_split_along_a_line_keeps_the_neighbour_track(
    tmp_path, monkeypatch
) -> None:
    """Cutting the neighbour's merged mask: the left part joins the cell (nearest
    its t1 position), the right part stays in the neighbour's track, both written
    as one batch that one cell-scoped undo reverts.
    """
    from omero_screen_track.edit import EditLog, replay
    from omero_screen_napari.review import masks

    session, curated = _split_session(tmp_path, monkeypatch)
    plan = masks.plan_split(session, "C2-t0-L1", 2, (15, 37), (45, 37))
    assert plan.label == 2 and plan.other_cell == [2, 2]
    h, w = plan.cell.shape
    full = np.zeros((60, 80), bool)
    full[plan.y0 : plan.y0 + h, plan.x0 : plan.x0 + w] = plan.cell
    assert full[30, 30] and not full[30, 44]
    assert (
        plan.cell.sum() + plan.other.sum()
        == (np.asarray(masks._arrays(None, "")[1][2]) == 2).sum()
    )

    entries = masks.commit_split(session, plan)
    assert [e.args["cell"] for e in entries] == [
        [2, 2],
        [0, 1],
    ]  # other part first
    assert entries[0].args["replaces"] == [] and entries[1].args[
        "replaces"
    ] == [2]
    assert entries[0].args["batch"] == entries[1].args["batch"]
    after = replay(curated, EditLog(session.edits_path), "C2")
    cell_lab, other_lab = entries[1].args["label"], entries[0].args["label"]
    assert after.tracks[(2, cell_lab)] == 1
    assert after.tracks[(2, other_lab)] == 2  # the neighbour continues
    assert [t for t, _ in after.detections(2)] == [0, 1, 2]

    session.undo("C2-t0-L1")
    assert not replay(curated, EditLog(session.edits_path), "C2").extras


def test_split_of_the_cells_own_nucleus_starts_a_new_track(
    tmp_path, monkeypatch
) -> None:
    """When the cut nucleus is the cell's own, the other part becomes a new track."""
    from omero_screen_track.edit import EditLog, replay
    from omero_screen_napari.review import masks

    session, curated = _split_session(
        tmp_path, monkeypatch, item_id="C2-t0-L2"
    )
    plan = masks.plan_split(session, "C2-t0-L2", 2, (15, 37), (45, 37))
    assert plan.other_cell is None
    h, w = plan.cell.shape
    full = np.zeros((60, 80), bool)
    full[plan.y0 : plan.y0 + h, plan.x0 : plan.x0 + w] = plan.cell
    assert full[30, 44]  # the right part: where track 2 was at t1
    entries = masks.commit_split(session, plan)
    assert (
        entries[0].args["cell"] == [0, 2] and entries[1].args["cell"] is None
    )
    after = replay(curated, EditLog(session.edits_path), "C2")
    assert after.tracks[(2, entries[0].args["label"])] == 2
    assert after.tracks[(2, entries[1].args["label"])] not in (1, 2, 9)


def test_split_line_missing_the_nucleus_is_refused(
    tmp_path, monkeypatch
) -> None:
    """A line beside every nucleus gives an error, nothing is written."""
    from omero_screen_napari.review import masks

    session, _ = _split_session(tmp_path, monkeypatch)
    with pytest.raises(MaskError):
        masks.plan_split(session, "C2-t0-L1", 2, (50, 0), (50, 10))
    assert not session.edits_path.exists()


def test_normal_frame_skips_merged_frames() -> None:
    """Plate 5054 C3: the cell is ~700 px alone but merged (1600 px) at t150
    and t152; splitting at t153 takes t151 as the reference, not t152.
    """
    from omero_screen_napari.review.masks import normal_frame

    areas = {t: 700.0 + (t % 3) * 20 for t in range(140, 152)}
    areas.update({150: 1600.0, 152: 1638.0, 154: 1700.0, 157: 690.0})
    assert normal_frame(areas, 153) == 151
    # Merged in most frames before (6 of 10): the band still comes from the
    # normal ones, since merges only make the area larger.
    merged = {t: 700.0 for t in range(143, 147)} | {
        t: 1650.0 for t in range(147, 153)
    }
    assert normal_frame(merged, 153) == 146
    # No frame before: the nearest normal frame after.
    later = {155: 720.0, 156: 3000.0, 157: 700.0}
    assert normal_frame(later, 153) == 155
    assert normal_frame({153: 700.0}, 153) is None


def test_reference_detection_uses_reviewer_masks_and_falls_back() -> None:
    """Areas of reviewer masks count; with no normal frame, the nearest earlier."""
    from omero_screen_track.edit import Curated
    from omero_screen_napari.review.masks import reference_detection

    det = pd.DataFrame(
        {
            "timepoint": [0, 1, 2, 3],
            "label": [1, 1, 1, 1],
            "y": [0.0, 10.0, 50.0, 90.0],
            "x": [0.0, 10.0, 50.0, 90.0],
            "area": [100.0, 100.0, 400.0, 100.0],
        }
    )
    # t2 is a merged raw mask, but at t3 a reviewer mask (area 100) is the cell.
    cur = Curated(
        tracks={(0, 1): 1, (1, 1): 1, (2, 1): 1, (3, 900): 1},
        parents={1: 0},
        extras={(3, 900): {"area": 100.0, "y": 70.0, "x": 70.0}},
    )
    assert reference_detection(det, cur, 1, 4) == (3, (70.0, 70.0))
    assert reference_detection(det, cur, 1, 3) == (1, (10.0, 10.0))
    odd = Curated(tracks={(1, 1): 1, (2, 1): 1}, parents={1: 0})
    det2 = det.assign(area=[10.0, 10.0, 1000.0, 100.0])
    # t1 (10 px) and t2 (1000 px): neither is within the band of their median.
    assert reference_detection(det2, odd, 1, 3) == (2, (50.0, 50.0))


def test_flipped_split_gives_the_cell_the_other_part(
    tmp_path, monkeypatch
) -> None:
    """Flip swaps the parts (twice is the identity); commit writes the flip."""
    from omero_screen_track.edit import EditLog, replay
    from omero_screen_napari.review import masks

    session, curated = _split_session(tmp_path, monkeypatch)
    plan = masks.plan_split(session, "C2-t0-L1", 2, (15, 37), (45, 37))
    flipped = plan.flip()
    assert flipped.flipped and not flipped.flip().flipped
    assert (flipped.cell == plan.other).all() and (
        flipped.other == plan.cell
    ).all()
    assert flipped.other_cell == plan.other_cell == [2, 2]
    entries = masks.commit_split(session, flipped)
    cell_entry = entries[-1]
    assert cell_entry.args["cell"] == [0, 1]
    assert cell_entry.args["measurements"]["area"] == float(plan.other.sum())
    assert "chosen by the reviewer" in cell_entry.reason
    after = replay(curated, EditLog(session.edits_path), "C2")
    py, px, mask = load_patch(
        session.edits_path.parent, cell_entry.args["measurements"]["patch"]
    )
    full = np.zeros((60, 80), bool)
    full[py : py + mask.shape[0], px : px + mask.shape[1]] = mask
    assert full[30, 44] and not full[30, 30]  # the right part, as flipped
    assert after.tracks[(2, cell_entry.args["label"])] == 1


def test_a_split_part_can_be_split_again(tmp_path, monkeypatch) -> None:
    """Cut the merged mask, then cut the cell's part again: the second cut is
    found under the line on the curated labels, read from the patch, replaces
    the part; replay paints only the new parts; undo restores the first cut.
    """
    from omero_screen_track.edit import EditLog, replay
    from omero_screen_napari.review import masks

    session, curated = _split_session(tmp_path, monkeypatch)
    first = masks.commit_split(
        session, masks.plan_split(session, "C2-t0-L1", 2, (15, 37), (45, 37))
    )
    part = first[-1].args["label"]  # the cell's (left) part
    plan = masks.plan_split(session, "C2-t0-L1", 2, (15, 30), (45, 30))
    assert plan.label == part and plan.from_extra
    assert plan.other_cell is None  # the cell's own mask: rest is a new track
    assert (
        plan.cell.sum() + plan.other.sum()
        == first[-1].args["measurements"]["area"]
    )
    second = masks.commit_split(session, plan)
    assert second[0].args["replaces"] == [part]
    assert "reviewer mask" in second[0].reason
    after = replay(curated, EditLog(session.edits_path), "C2")
    new_cell, new_other = second[0].args["label"], second[1].args["label"]
    assert (2, part) not in after.extras
    assert after.tracks[(2, new_cell)] == 1
    assert after.tracks[(2, new_other)] not in (1, 2, 9)
    assert after.tracks[(2, first[0].args["label"])] == 2  # neighbour's part
    labels = masks.paint_patches(
        np.asarray(masks._arrays(None, "")[1][2]),
        2,
        0,
        0,
        after,
        session.edits_path.parent,
    )
    assert part not in labels and 2 not in labels
    assert {new_cell, new_other} <= set(np.unique(labels))
    # The measurement background came from a raw nucleus, not a reviewer mask.
    assert second[0].args["measurements"]["dna"] == pytest.approx(500.0)

    session.undo("C2-t0-L1")
    back = replay(curated, EditLog(session.edits_path), "C2")
    assert set(back.extras) == {(2, part), (2, first[0].args["label"])}
    assert back.tracks[(2, part)] == 1


def test_nuclei_layer_shows_reviewer_masks_at_every_level(tmp_path) -> None:
    """The nuclei layer is repainted as curated: replaced raw labels cleared,
    live reviewer masks painted (scaled per pyramid level), debris blanked; a
    mask replaced by a later split is not painted.
    """
    from omero_screen_track.edit import Curated
    from omero_screen_napari._review_widget import (
        _curated_overrides,
        _overlay_labels,
    )

    full = np.zeros((2, 8, 8), np.uint32)
    full[1, 0:4, 0:8] = 5  # merged raw nucleus at t1
    full[:, 6:8, 6:8] = 7  # debris
    left = np.zeros((4, 4), bool)
    left[:, :] = True
    right = left.copy()
    rel_a = save_patch(tmp_path / "masks", "C2", 1, 10, 0, 0, left)
    rel_b = save_patch(tmp_path / "masks", "C2", 1, 11, 0, 4, right)
    cur = Curated(
        tracks={(1, 10): 1, (1, 11): 2},
        parents={1: 0, 2: 0},
        extras={(1, 10): {"patch": rel_a}, (1, 11): {"patch": rel_b}},
        removed={(1, 5)},
    )
    removed, patches = _curated_overrides(cur, tmp_path)
    levels = [full, full[:, ::2, ::2]]
    out = [np.asarray(lv) for lv in _overlay_labels(levels, {7}, removed, patches)]
    assert out[0][1, 0, 0] == 10 and out[0][1, 0, 7] == 11
    assert 5 not in out[0] and 7 not in out[0] and 7 not in out[1]
    assert out[1][1, 0, 0] == 10 and out[1][1, 1, 3] == 11
    assert (out[0][0] == 0).all()  # t0: only debris, blanked
