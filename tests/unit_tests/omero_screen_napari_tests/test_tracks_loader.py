"""Tests for tracks_loader — CellView LazyFrame to napari track data.

These are pure-data tests using synthetic polars frames; no napari viewer or
OMERO connection is involved.
"""

import numpy as np
import polars as pl
import pytest

from omero_screen_napari.tracks_loader import (
    TracksData,
    has_tracks,
    load_tracks_for_well,
)


def _tracked_frame() -> pl.LazyFrame:
    """Two wells. Well C4 has a founder track that divides into two daughters.

    track 1: founder, frames 0-2 (parent 0)
    track 2: founder, frames 0-1 (parent 0), divides into 3 and 4 at frame 2
    track 3, 4: daughters at frame 2 (parent 2)
    Well C5 has a single untracked-into founder to prove well filtering.
    """
    return pl.LazyFrame(
        {
            "well": ["C4"] * 7 + ["C5"],
            "track_id": [1, 1, 1, 2, 2, 3, 4, 9],
            "parent_track_id": [0, 0, 0, 0, 0, 2, 2, 0],
            "timepoint": [0, 1, 2, 0, 1, 2, 2, 0],
            "centroid-0-nuc": [10.0, 11, 12, 40, 41, 38, 48, 5],
            "centroid-1-nuc": [10.0, 12, 14, 40, 40, 40, 40, 5],
            "cell_cycle": ["G1", "S", "G2", "G1", "G1", "G1", "G1", "G2"],
        }
    )


class TestHasTracks:
    def test_present(self) -> None:
        assert has_tracks(_tracked_frame()) is True

    def test_absent(self) -> None:
        lf = pl.LazyFrame({"well": ["C4"], "area_nucleus": [10.0]})
        assert has_tracks(lf) is False


class TestLoadTracksForWell:
    def test_returns_none_without_track_column(self) -> None:
        lf = pl.LazyFrame({"well": ["C4"], "area_nucleus": [10.0]})
        assert load_tracks_for_well(lf, "C4") is None

    def test_data_array_shape_and_order(self) -> None:
        result = load_tracks_for_well(_tracked_frame(), "C4")
        assert isinstance(result, TracksData)
        # 7 rows in well C4, four columns [track_id, t, y, x]
        assert result.data.shape == (7, 4)
        # Sorted by track_id then timepoint
        assert list(result.data[:, 0]) == [1, 1, 1, 2, 2, 3, 4]
        assert list(result.data[:, 1]) == [0, 1, 2, 0, 1, 2, 2]
        # First row centroid (y, x) carried through
        np.testing.assert_array_equal(result.data[0, 2:], [10.0, 10.0])

    def test_lineage_graph(self) -> None:
        result = load_tracks_for_well(_tracked_frame(), "C4")
        assert result is not None
        # Daughters 3 and 4 point to parent 2; founders absent.
        assert result.graph == {3: [2], 4: [2]}

    def test_properties_include_track_id_and_cell_cycle(self) -> None:
        result = load_tracks_for_well(_tracked_frame(), "C4")
        assert result is not None
        assert "track_id" in result.properties
        assert "cell_cycle" in result.properties
        assert len(result.properties["track_id"]) == 7

    def test_well_filtering(self) -> None:
        result = load_tracks_for_well(_tracked_frame(), "C5")
        assert result is not None
        assert result.data.shape == (1, 4)
        assert result.graph == {}

    def test_missing_centroid_raises(self) -> None:
        lf = pl.LazyFrame(
            {
                "well": ["C4"],
                "track_id": [1],
                "parent_track_id": [0],
                "timepoint": [0],
            }
        )
        with pytest.raises(KeyError, match="required columns missing"):
            load_tracks_for_well(lf, "C4")

    def test_empty_well_raises(self) -> None:
        with pytest.raises(ValueError, match="No tracked rows"):
            load_tracks_for_well(_tracked_frame(), "Z9")
