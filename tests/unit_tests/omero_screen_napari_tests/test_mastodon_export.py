"""Tests for the CTC bundle export.

``build_ctc_export`` and ``relabel_mask`` are pure data. ``export_well_ctc``
touches a (synthetic) zarr cache on disk; the full Mastodon round-trip is
covered by the manual integration run.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile
import zarr
from omero_screen_napari.mastodon_export import (
    build_ctc_export,
    export_well_ctc,
    relabel_mask,
    write_unit_scale_view,
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


# --- build_ctc_export: res_track.txt table -----------------------------------


def test_table_columns_and_one_row_per_track() -> None:
    export = build_ctc_export(_frame(), "B2")
    assert export.track_table.columns == ["L", "B", "E", "P"]
    # 4 distinct tracks (10, 11, 12, 20) → 4 rows, not one per spot.
    assert export.track_table.height == 4


def test_relabels_consecutively_in_begin_frame_order() -> None:
    export = build_ctc_export(_frame(), "B2")
    # Labels 1..N, sorted; column B non-decreasing (CTC convention).
    assert export.track_table["L"].to_list() == [1, 2, 3, 4]
    assert export.track_table["B"].to_list() == [0, 0, 3, 3]


def test_lifetimes_and_parent_links() -> None:
    """Track 10 (founder, 0-2) divides into 11 and 12 (3-4).

    After relabel by begin frame: 10→1, 20→2, 11→3, 12→4.
    """
    export = build_ctc_export(_frame(), "B2")
    rows = {int(r["L"]): r for r in export.track_table.to_dicts()}
    assert (rows[1]["B"], rows[1]["E"], rows[1]["P"]) == (
        0,
        2,
        0,
    )  # founder 10
    assert (rows[2]["B"], rows[2]["E"], rows[2]["P"]) == (
        0,
        2,
        0,
    )  # founder 20
    # Both daughters point at the relabelled parent (10 → 1).
    assert (rows[3]["B"], rows[3]["E"], rows[3]["P"]) == (3, 4, 1)
    assert (rows[4]["B"], rows[4]["E"], rows[4]["P"]) == (3, 4, 1)


def test_every_parent_label_precedes_its_daughters() -> None:
    export = build_ctc_export(_frame(), "B2")
    for r in export.track_table.to_dicts():
        if r["P"] != 0:
            assert r["P"] < r["L"]  # CTC invariant: parent defined earlier


def test_relabel_map_round_trips_track_ids() -> None:
    export = build_ctc_export(_frame(), "B2")
    assert export.relabel == {0: 0, 10: 1, 20: 2, 11: 3, 12: 4}


def test_no_track_column_raises() -> None:
    lf = pl.LazyFrame({"well": ["B2"], "area_nucleus": [1.0]})
    with pytest.raises(KeyError, match="no track_id column"):
        build_ctc_export(lf, "B2")


def test_missing_well_raises() -> None:
    with pytest.raises(ValueError, match="No tracked rows"):
        build_ctc_export(_frame(), "Z9")


# --- build_ctc_export: round-trip manifest -----------------------------------


def test_manifest_maps_labels_to_original_track_ids() -> None:
    export = build_ctc_export(_frame(), "B2")
    m = export.manifest
    assert m["label_scheme"] == "relabel_1_to_n_by_begin_frame"
    assert m["n_tracks"] == 4
    # CTC label 1 ← original track_id 10; label 3 ← daughter 11 (parent 10).
    assert m["tracks"]["1"]["track_id"] == 10
    assert m["tracks"]["1"]["parent_track_id"] == 0
    assert m["tracks"]["3"]["track_id"] == 11
    assert m["tracks"]["3"]["parent_track_id"] == 10


def test_manifest_carries_per_frame_centroids_for_reconciliation() -> None:
    export = build_ctc_export(_frame(), "B2")
    # Track 20 (label 2) appears at frames 0 and 2 with its centroids.
    frames = export.manifest["tracks"]["2"]["frames"]
    assert [f["t"] for f in frames] == [0, 2]
    assert (frames[0]["y"], frames[0]["x"]) == (50.0, 60.0)


# --- relabel_mask ------------------------------------------------------------


def test_relabel_mask_maps_values_and_drops_unmapped() -> None:
    mask = np.array([[0, 10, 20], [11, 12, 99]], dtype=np.uint32)
    relabel = {0: 0, 10: 1, 20: 2, 11: 3, 12: 4}
    out = relabel_mask(mask, relabel, np.uint16)
    # 0→0, mapped labels remapped, unmapped 99 → 0 (no spurious spot).
    assert out.tolist() == [[0, 1, 2], [3, 4, 0]]
    assert out.dtype == np.uint16


def test_relabel_mask_handles_empty_relabel() -> None:
    mask = np.zeros((2, 2), dtype=np.uint32)
    out = relabel_mask(mask, {0: 0}, np.uint16)
    assert out.tolist() == [[0, 0], [0, 0]]


# --- export_well_ctc: on-disk bundle -----------------------------------------


def _fake_cached_well_with_labels(
    cache_root, plate_id: int, well: str, nuclei_tyx: np.ndarray
) -> None:
    """Create a minimal cached well zarr carrying a nuclei label pyramid.

    Mirrors what the reader expects: ``plate_<id>.zarr/<row>/<col>/0`` with a
    ``labels/nuclei/0`` array of shape ``(T, Y, X)``.
    """
    import os

    row, col = well[0], str(int(well[1:]))
    path = os.path.join(cache_root, "zarr", f"plate_{plate_id}.zarr")
    root = zarr.open_group(path, mode="w")
    nuc = (
        root.require_group(f"{row}/{col}/0")
        .require_group("labels")
        .require_group("nuclei")
    )
    nuc.create_dataset("0", data=nuclei_tyx.astype(np.uint32))


def _tracks_3frames() -> pl.LazyFrame:
    """Two founders over 3 frames; track 7 divides into 8 at frame 2."""
    return pl.LazyFrame(
        {
            "well": ["B2"] * 6,
            "track_id": [7, 7, 7, 8, 5, 5],
            "parent_track_id": [0, 0, 0, 7, 0, 0],
            "timepoint": [0, 1, 2, 2, 0, 1],
            "centroid-0-nuc": [1.0, 1, 1, 2, 5, 5],
            "centroid-1-nuc": [1.0, 1, 1, 2, 5, 5],
            "area_nucleus": [4.0] * 6,
        }
    )


def _nuclei_3frames() -> np.ndarray:
    """3 frames, 4x4; pixel value == track_id (as the tracked cache stores)."""
    nuclei = np.zeros((3, 4, 4), dtype=np.uint32)
    nuclei[0, 0, 0] = 7
    nuclei[0, 3, 3] = 5
    nuclei[1, 0, 0] = 7
    nuclei[1, 3, 3] = 5
    nuclei[2, 0, 0] = 7
    nuclei[2, 2, 2] = 8
    return nuclei


def test_export_well_ctc_writes_full_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OMERO_SCREEN_CACHE_PATH", str(tmp_path))
    _fake_cached_well_with_labels(tmp_path, 4155, "B2", _nuclei_3frames())

    paths = export_well_ctc(4155, "B2", _tracks_3frames(), out_base=tmp_path)

    out_dir = paths["dir"]
    masks = sorted(out_dir.glob("mask*.tif"))
    assert [p.name for p in masks] == [
        "mask000.tif",
        "mask001.tif",
        "mask002.tif",
    ]
    assert paths["res_track"].exists()
    assert paths["manifest"].exists()
    assert paths["readme"].exists()


def test_export_well_ctc_mask_values_match_res_track(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("OMERO_SCREEN_CACHE_PATH", str(tmp_path))
    _fake_cached_well_with_labels(tmp_path, 4155, "B2", _nuclei_3frames())

    paths = export_well_ctc(4155, "B2", _tracks_3frames(), out_base=tmp_path)

    # Relabel by begin frame: 5→1, 7→2, 8→3. Masks must carry the NEW labels.
    m0 = tifffile.imread(paths["dir"] / "mask000.tif")
    assert m0[0, 0] == 2  # track 7 → label 2
    assert m0[3, 3] == 1  # track 5 → label 1
    m2 = tifffile.imread(paths["dir"] / "mask002.tif")
    assert m2[2, 2] == 3  # daughter track 8 → label 3

    # res_track.txt labels are exactly those mask values: L B E P per line.
    lines = paths["res_track"].read_text().splitlines()
    rows = {int(ln.split()[0]): [int(x) for x in ln.split()] for ln in lines}
    assert rows[2] == [2, 0, 2, 0]  # track 7: founder, frames 0-2
    assert rows[3] == [3, 2, 2, 2]  # track 8: daughter of label 2, frame 2

    # manifest maps the labels back to raw track_ids for the round-trip.
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["tracks"]["2"]["track_id"] == 7
    assert manifest["tracks"]["3"]["track_id"] == 8


# --- write_unit_scale_view ---------------------------------------------------


def _make_image_group(root: Path, pixel_size: float) -> Path:
    """Write a minimal OME-NGFF image multiscale group with a µm pixel scale."""
    group = root / "C" / "4" / "0"
    group.mkdir(parents=True)
    (group / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    attrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1.0, 1.0, pixel_size, pixel_size],
                            }
                        ],
                    },
                    {
                        "path": "1",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [
                                    1.0,
                                    1.0,
                                    pixel_size * 2,
                                    pixel_size * 2,
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
    }
    (group / ".zattrs").write_text(json.dumps(attrs))
    # Per-level chunk dirs the view should symlink (content is irrelevant here).
    for lvl in ("0", "1"):
        (group / lvl).mkdir()
        (group / lvl / ".zarray").write_text("{}")
    return group


def test_write_unit_scale_view_rescales_and_symlinks(tmp_path) -> None:
    src = _make_image_group(tmp_path / "cache", 0.5)
    dst = tmp_path / "bundle" / "mastodon_image"

    out = write_unit_scale_view(src, dst)

    assert out == dst
    attrs = json.loads((dst / ".zattrs").read_text())
    scales = [
        d["coordinateTransformations"][0]["scale"]
        for d in attrs["multiscales"][0]["datasets"]
    ]
    # 0.5 µm/px -> level 0 unit, pyramid kept as 1 / 2.
    assert scales == [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]
    # Heavy level dirs are symlinks back to the cache, not copies.
    assert (dst / "0").is_symlink()
    assert (dst / "0").resolve() == (src / "0").resolve()


def test_write_unit_scale_view_falls_back_without_metadata(tmp_path) -> None:
    # A group with no .zattrs multiscales must not crash the export.
    src = tmp_path / "C" / "4" / "0"
    src.mkdir(parents=True)
    (src / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    dst = tmp_path / "bundle" / "mastodon_image"

    out = write_unit_scale_view(src, dst)

    assert out == src  # fell back to the raw group
    assert not dst.exists()


def test_export_well_ctc_raises_without_cached_labels(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("OMERO_SCREEN_CACHE_PATH", str(tmp_path))
    # Cache a well with an image group but no nuclei labels.
    path = tmp_path / "zarr" / "plate_4155.zarr"
    zarr.open_group(str(path), mode="w").require_group("B/2/0")
    with pytest.raises(FileNotFoundError):
        export_well_ctc(4155, "B2", _tracks_3frames(), out_base=tmp_path)
