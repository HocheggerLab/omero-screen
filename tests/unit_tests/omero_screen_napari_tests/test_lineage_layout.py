"""Tests for the pure lineage-tree layout in _lineage_tree_widget.

Only ``compute_lineage_layout`` is exercised here — it is pure data and needs no
Qt/napari viewer.
"""

import polars as pl

from omero_screen_napari._lineage_tree_widget import compute_lineage_layout


def _frame() -> pl.LazyFrame:
    """Founder 1 spans 0-2; founder 2 divides into 3 and 4 at frame 2."""
    return pl.LazyFrame(
        {
            "well": ["C4"] * 7,
            "track_id": [1, 1, 1, 2, 2, 3, 4],
            "parent_track_id": [0, 0, 0, 0, 0, 2, 2],
            "timepoint": [0, 1, 2, 0, 1, 2, 2],
        }
    )


def test_one_segment_per_track() -> None:
    segments = compute_lineage_layout(_frame(), "C4")
    assert {s.track_id for s in segments} == {1, 2, 3, 4}


def test_spans_and_parents() -> None:
    by_id = {s.track_id: s for s in compute_lineage_layout(_frame(), "C4")}
    assert (by_id[1].t_start, by_id[1].t_end) == (0, 2)
    assert (by_id[2].t_start, by_id[2].t_end) == (0, 1)
    assert by_id[3].parent == 2 and by_id[4].parent == 2
    assert by_id[1].parent == 0


def test_parent_centred_over_daughters() -> None:
    by_id = {s.track_id: s for s in compute_lineage_layout(_frame(), "C4")}
    # Parent 2 should sit at the mean y of its daughters 3 and 4.
    assert by_id[2].y == (by_id[3].y + by_id[4].y) / 2
