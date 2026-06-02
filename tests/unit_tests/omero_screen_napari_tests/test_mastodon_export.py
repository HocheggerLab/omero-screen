"""Tests for the Mastodon export: pure CSV translation + cache-side writers.

``build_mastodon_csv`` is pure data. ``write_well_tracks_csv`` /
``write_plate_tracks_csvs`` touch a (synthetic) cache on disk; the full
Mastodon round-trip is covered by the manual integration run.
"""

import json
from pathlib import Path

import polars as pl
import pytest

from omero_screen_napari.mastodon_export import (
    build_mastodon_csv,
    write_plate_tracks_csvs,
    write_well_tracks_csv,
)


def _frame() -> pl.LazyFrame:
    """One founder that divides, plus a gapped track.

    track 10: founder, frames 0-2 → divides into 11 and 12 at frame 3.
    track 20: founder present at frames 0, 2 (missing frame 1 — a gap).
    """
    return pl.LazyFrame(
        {
            "well": ["B2"] * 9,
            "track_id": [10, 10, 10, 11, 11, 12, 12, 20, 20],
            "parent_track_id": [0, 0, 0, 10, 10, 10, 10, 0, 0],
            "timepoint": [0, 1, 2, 3, 4, 3, 4, 0, 2],
            "centroid-0-nuc": [10.0, 11, 12, 13, 14, 20, 21, 50, 52],
            "centroid-1-nuc": [10.0, 10, 10, 11, 12, 30, 31, 60, 61],
            "area_nucleus": [100.0] * 9,
        }
    )


def _by_label(df: pl.DataFrame) -> dict[int, pl.DataFrame]:
    return {
        int(lbl): df.filter(pl.col("label") == lbl).sort("frame")
        for lbl in df["label"].unique()
    }


def test_columns_and_count() -> None:
    df = build_mastodon_csv(_frame(), "B2", pixel_size=1.0)
    assert df.columns == [
        "id",
        "parent_id",
        "x",
        "y",
        "z",
        "frame",
        "radius",
        "label",
    ]
    assert df.height == 9  # one spot per (track, timepoint) row


def test_unique_spot_ids() -> None:
    df = build_mastodon_csv(_frame(), "B2", pixel_size=1.0)
    assert df["id"].n_unique() == df.height


def test_founder_first_spot_has_no_parent() -> None:
    df = build_mastodon_csv(_frame(), "B2", pixel_size=1.0)
    by = _by_label(df)
    # Track 10 founder: first spot (frame 0) → parent -1.
    assert by[10]["parent_id"][0] == -1
    # Its second spot links back to the first.
    assert by[10]["parent_id"][1] == by[10]["id"][0]


def test_division_links_daughters_to_parent_last_spot() -> None:
    df = build_mastodon_csv(_frame(), "B2", pixel_size=1.0)
    by = _by_label(df)
    parent_last = by[10].filter(pl.col("frame") == 2)["id"][0]
    # Both daughters' first spots (frame 3) link to track 10's last spot.
    assert by[11].filter(pl.col("frame") == 3)["parent_id"][0] == parent_last
    assert by[12].filter(pl.col("frame") == 3)["parent_id"][0] == parent_last


def test_gap_is_bridged_not_fragmented() -> None:
    """Track 20 missing frame 1: its frame-2 spot links to frame-0, not -1."""
    df = build_mastodon_csv(_frame(), "B2", pixel_size=1.0)
    by = _by_label(df)
    first = by[20].filter(pl.col("frame") == 0)["id"][0]
    second = by[20].filter(pl.col("frame") == 2)
    assert second["parent_id"][0] == first  # bridged, not a new track start


def test_pixel_size_scales_coordinates() -> None:
    df1 = build_mastodon_csv(_frame(), "B2", pixel_size=1.0)
    df2 = build_mastodon_csv(_frame(), "B2", pixel_size=0.5)
    # x/y/radius scale linearly; frame/id/label unchanged.
    assert df2["x"][0] == pytest.approx(df1["x"][0] * 0.5)
    assert df2["radius"][0] == pytest.approx(df1["radius"][0] * 0.5)


def test_missing_well_raises() -> None:
    with pytest.raises(ValueError, match="No tracked rows"):
        build_mastodon_csv(_frame(), "Z9", pixel_size=1.0)


def test_no_track_column_raises() -> None:
    lf = pl.LazyFrame({"well": ["B2"], "area_nucleus": [1.0]})
    with pytest.raises(KeyError, match="no track_id column"):
        build_mastodon_csv(lf, "B2", pixel_size=1.0)


def _fake_cached_well(cache_root: Path, plate_id: int, well: str) -> Path:
    """Create a minimal cached well image group with a scale in .zattrs."""
    row, col = well[0], well[1:]
    grp = cache_root / "zarr" / f"plate_{plate_id}.zarr" / row / col / "0"
    grp.mkdir(parents=True)
    (grp / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "datasets": [
                            {
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.0, 1.0, 0.5, 0.5]}
                                ]
                            }
                        ]
                    }
                ]
            }
        )
    )
    return grp


def test_write_well_tracks_csv_lands_beside_image(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OMERO_SCREEN_CACHE_PATH", str(tmp_path))
    grp = _fake_cached_well(tmp_path, 4155, "B2")
    out = write_well_tracks_csv(4155, "B2", _frame(), pixel_size=1.0)
    # CSV sits next to the "0" image group, in the well dir.
    assert out == grp.parent / "tracks.csv"
    assert out.exists()
    assert pl.read_csv(out).height == 9


def test_write_well_tracks_csv_reads_pixel_size_from_zattrs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("OMERO_SCREEN_CACHE_PATH", str(tmp_path))
    _fake_cached_well(tmp_path, 4155, "B2")
    out = write_well_tracks_csv(4155, "B2", _frame())  # no pixel_size override
    # .zattrs scale x = 0.5 → first spot x = centroid-1 (10.0) * 0.5 = 5.0.
    assert pl.read_csv(out)["x"][0] == pytest.approx(5.0)


def test_write_plate_tracks_csvs_is_best_effort(tmp_path, monkeypatch) -> None:
    """No CellView / no cached wells → returns [] rather than raising."""
    monkeypatch.setenv("OMERO_SCREEN_CACHE_PATH", str(tmp_path))
    # CellView connect raises in this isolated env → graceful empty result.
    monkeypatch.setattr(
        "cellview.db.db.CellViewDB.connect",
        lambda self: (_ for _ in ()).throw(RuntimeError("no db")),
    )
    assert write_plate_tracks_csvs(4155) == []
