"""Unit tests for ``omero_screen.track_correction``.

Trackastra's second pass is replaced by an identity tracker where a test needs
one; :func:`graph_table` is checked against Trackastra's own ``graph_to_ctc``.
"""

from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from omero_screen.track_correction import (
    CorrectionParams,
    LazyFrames,
    correct_tracks,
    correction_map,
    graph_table,
    measure_labels,
    merge_detections,
    normalise_stack,
    relabel_frame,
)


def _disc(
    shape: tuple[int, int], cy: float, cx: float, r: float
) -> np.ndarray:
    yy, xx = np.indices(shape)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


class TestMeasureLabels:
    def test_area_centroid_and_background(self) -> None:
        mask = np.zeros((1, 10, 10), np.uint32)
        mask[0, 1:3, 1:3] = 5  # 4 px, centroid (1.5, 1.5)
        mask[0, 6:9, 6:8] = 9  # 6 px, centroid (7, 6.5)
        img = np.full((1, 10, 10), 10.0)
        img[0][mask[0] == 5] = 30.0
        img[0][mask[0] == 9] = 5.0  # below background: clipped to 0

        det = measure_labels(mask, {"g": img})

        assert det["label"].tolist() == [5, 9]
        assert det["area"].tolist() == [4, 6]
        np.testing.assert_allclose(det[["y", "x"]], [[1.5, 1.5], [7, 6.5]])
        np.testing.assert_allclose(det["g"], [20.0, 0.0])

    def test_frames_subset(self) -> None:
        mask = np.zeros((3, 4, 4), np.uint32)
        mask[:, 0, 0] = 1
        det = measure_labels(mask, {}, frames=[2])
        assert det["timepoint"].tolist() == [2]


def test_merge_detections_is_area_weighted() -> None:
    det = pd.DataFrame(
        {
            "timepoint": [0, 0, 0],
            "label": [1, 2, 3],
            "area": [10.0, 30.0, 5.0],
            "y": [0.0, 4.0, 9.0],
            "x": [0.0, 0.0, 9.0],
            "g": [100.0, 20.0, 1.0],
        }
    )
    out = merge_detections(det, pd.Series([7, 7, 0]), ["g"])
    assert out["label"].tolist() == [7]
    assert out["area"].tolist() == [40.0]
    np.testing.assert_allclose(out[["y", "g"]].iloc[0], [3.0, 40.0])


def test_relabel_frame_drops_unmapped() -> None:
    frame = np.array([[0, 1, 2], [3, 3, 9]], np.uint32)
    out = relabel_frame(frame, {1: 5, 2: 5, 3: 0})
    np.testing.assert_array_equal(out, [[0, 5, 5], [0, 0, 0]])


def test_lazy_frames_and_normalisation() -> None:
    stack = np.arange(2 * 300 * 300, dtype=np.uint16).reshape(2, 300, 300)
    lazy = normalise_stack(stack)
    from trackastra.utils import normalize

    np.testing.assert_allclose(
        np.stack(list(lazy)), normalize(stack), rtol=1e-5, atol=1e-6
    )
    doubled = LazyFrames(stack, lambda f: f.astype(np.int64) * 2, np.int64)
    assert len(doubled) == 2 and doubled[1][0, 0] == 2 * int(stack[1, 0, 0])


def test_graph_table_matches_graph_to_ctc() -> None:
    """Same ids as Trackastra's CTC export: division, gap and founders."""
    from trackastra.tracking.utils import graph_to_ctc

    masks = np.zeros((4, 20, 20), np.uint16)
    # t0: A(1); t1: A(1), B(2); t2: daughters of A (1, 3), B gone;
    # t3: daughter 1 continues, B reappears after a gap (label 2).
    masks[0, 2:5, 2:5] = 1
    masks[1, 2:5, 2:5] = 1
    masks[1, 12:15, 12:15] = 2
    masks[2, 2:5, 2:4] = 1
    masks[2, 6:9, 2:5] = 3
    masks[3, 2:5, 2:4] = 1
    masks[3, 12:15, 12:15] = 2
    g = nx.DiGraph()
    nodes = {
        "a0": (0, 1),
        "a1": (1, 1),
        "b1": (1, 2),
        "d1": (2, 1),
        "d2": (2, 3),
        "d1b": (3, 1),
        "b3": (3, 2),
    }
    for n, (t, lab) in nodes.items():
        g.add_node(n, time=t, label=lab)
    g.add_edges_from(
        [("a0", "a1"), ("a1", "d1"), ("a1", "d2"), ("d1", "d1b"), ("b1", "b3")]
    )

    table, parents = graph_table(g)
    ctc_df, relabelled = graph_to_ctc(g, masks, check=False)

    for row in table.itertuples():
        region = masks[row.timepoint] == row.label
        assert np.all(relabelled[row.timepoint][region] == row.track)
    assert parents == dict(zip(ctc_df["label"], ctc_df["parent"], strict=True))


def _fragment_well() -> tuple[pd.DataFrame, dict[int, int]]:
    """Track 1 moves 5 px/frame and splits at t10 into touching 2 and 3
    that keep its geminin (a fragment); track 9 is still and dark (debris);
    track 4 is a bright founder so the well has an expressing population.
    """
    rows = []
    for t in range(20):
        if t < 10:
            rows.append((t, 1, 100.0, 50.0, 10.0 + 5 * t, 80.0, 80.0))
        else:
            x = 10.0 + 5 * t
            rows.append((t, 2, 50.0, 46.0, x, 80.0, 80.0))
            rows.append((t, 3, 50.0, 54.0, x, 80.0, 80.0))
        rows.append((t, 4, 100.0, 150.0, 10.0 + 4 * t, 80.0, 80.0))
        rows.append((t, 9, 60.0, 300.0, 300.0, 0.0, 0.0))
    det = pd.DataFrame(
        rows,
        columns=["timepoint", "label", "area", "y", "x", "geminin", "pip"],
    )
    return det, {1: 0, 2: 1, 3: 1, 4: 0, 9: 0}


def test_correction_map_merges_fragments_and_hides_debris() -> None:
    det, parents = _fragment_well()
    mapping, result, debris = correction_map(det, parents)
    assert debris == {9}
    assert mapping[9] == 0
    assert mapping[2] == mapping[3] == mapping[1] == 1
    assert mapping[4] == 4
    assert "fragment" in set(result.events["rule"])


def test_correct_tracks_end_to_end_with_identity_tracker() -> None:
    det, parents = _fragment_well()

    def identity(images, masks, model, *args):  # noqa: ANN001, ANN202
        tracked = merge_detections(
            det, pd.Series(masks.fn(det["label"].to_numpy())), []
        )[["timepoint", "label"]]
        tracked["track"] = tracked["label"]
        return tracked, {int(t): 0 for t in tracked["track"].unique()}

    masks = np.zeros((20, 4, 4), np.uint32)
    with patch("omero_screen.track_correction.retrack", identity):
        res = correct_tracks(
            masks,
            masks,
            parents,
            {},
            model=None,
            params=CorrectionParams(),
            det=det,
        )

    final = res.table.set_index(["timepoint", "label"])["track_id"]
    assert final[(12, 2)] == final[(12, 3)] == final[(5, 1)]
    assert final[(5, 9)] == 0
    assert res.debris == {9}
    assert set(res.events["pass"]) <= {1, 2}
    assert res.frame_map(12) == {2: 1, 3: 1, 4: 4, 9: 0}
