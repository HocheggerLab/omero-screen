"""Tests for the per-well cell-cycle montage.

Selection and scaling are what decide what the figure *claims*, so they carry
the weight here:

* the draw must be reproducible from the seed, or a figure cannot be regenerated
  and the panels are effectively hand-picked;
* display limits must be shared across phases, since per-crop scaling would
  normalise away the size and intensity differences the montage exists to show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from omero_screen_napari.phase_montage import (
    DEFAULT_PHASES,
    _close,
    MontageConfig,
    MontageError,
    _crop_pixels,
    _outline,
    _resolve_overlay,
    build_montage,
    channel_limits,
    export_plate_pdfs,
    export_well_pdf,
    plate_wells,
    render_montage,
    resolve_mask_label,
    select_cells,
)
from omero_screen_napari.zarr_cache import PlateZarrWriter
from omero_screen_napari.zarr_cache.registry import ZarrPlateEntry, upsert


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OMERO_SCREEN_CACHE_PATH", str(tmp_path))
    return tmp_path


DEFAULT_PER_PHASE = {"G1": 10, "S": 8, "G2/M": 6, "Polyploid": 5}


def _cell_positions(
    per_phase: dict[str, int] | None = None,
) -> dict[int, tuple[int, int]]:
    """``{label: (cy, cx)}`` — the single source of truth for cell placement."""
    per_phase = per_phase or DEFAULT_PER_PHASE
    rng = np.random.default_rng(0)
    positions: dict[int, tuple[int, int]] = {}
    label = 0
    for count in per_phase.values():
        for _ in range(count):
            label += 1
            positions[label] = (
                int(rng.integers(80, 430)),
                int(rng.integers(80, 430)),
            )
    return positions


def _measurements(
    well: str = "C3", per_phase: dict[str, int] | None = None
) -> pl.DataFrame:
    per_phase = per_phase or DEFAULT_PER_PHASE
    positions = _cell_positions(per_phase)
    rows = []
    label = 0
    for phase, count in per_phase.items():
        for _ in range(count):
            label += 1
            cy, cx = positions[label]
            rows.append(
                {
                    "plate_id": 99,
                    "well": well,
                    "label": label,
                    "cell_cycle": phase,
                    "centroid_y": float(cy),
                    "centroid_x": float(cx),
                    "equivalent_diameter_area_cell": (
                        34.0 if phase != "Polyploid" else 56.0
                    ),
                }
            )
    return pl.DataFrame(rows)


CHANNELS = ["DAPI_R1", "Tub_R1", "Y15_R1", "p21_R2", "TP53_R3"]


#: Realistic for these plates: plate 4127 is 1.187 um/px. The value matters —
#: at 0.3 um/px a 20 um scale bar is wider than half the crop and is correctly
#: suppressed, which silently disables the scale-bar assertions.
PIXEL_SIZE_UM = 1.2


def _build_plate(plate_id: int = 99, well: str = "C3", size: int = 512):
    """Write a synthetic plate whose masks match ``_measurements()``.

    The measurement centroids and the drawn masks are generated from the same
    RNG stream, so a crop centred on a measurement actually contains that cell.
    An earlier version drew masks at unrelated positions; every render test then
    passed against panels with no cell in them, which is precisely the class of
    fixture unrealism that hid the real-data bugs.

    The cell mask is deliberately labelled **differently** from the nuclei mask
    (offset by 1000), mirroring plate 4127 where nucleus 2184 sits in cell 2152.
    """
    rng = np.random.default_rng(1)
    image = rng.integers(
        200, 3000, (1, len(CHANNELS), size, size), dtype=np.uint16
    )
    nuc = np.zeros((1, size, size), dtype=np.uint32)
    cells = np.zeros((1, size, size), dtype=np.uint32)
    yy, xx = np.ogrid[:size, :size]
    for label, (cy, cx) in _cell_positions().items():
        nuc[0][(yy - cy) ** 2 + (xx - cx) ** 2 <= 10**2] = label
        cells[0][(yy - cy) ** 2 + (xx - cx) ** 2 <= 18**2] = label + 1000
    writer = PlateZarrWriter(plate_id, "demo", CHANNELS, PIXEL_SIZE_UM, 1)
    with writer:
        writer.ensure_plate(all_wells=[well])
        writer.write_well(well, image, nuc, cells)
    upsert(ZarrPlateEntry(plate_id=plate_id))


class TestSelection:
    def test_takes_the_requested_number_per_phase(self) -> None:
        cells, warnings = select_cells(
            _measurements(), "C3", MontageConfig(cells_per_phase=3)
        )
        assert {p: len(c) for p, c in cells.items()} == {
            "G1": 3,
            "S": 3,
            "G2/M": 3,
            "Polyploid": 3,
        }
        assert warnings == []

    def test_is_reproducible_from_the_seed(self) -> None:
        """Without this a published figure cannot be regenerated."""
        config = MontageConfig(seed=7)
        first, _ = select_cells(_measurements(), "C3", config)
        second, _ = select_cells(_measurements(), "C3", config)
        assert {p: [c.label for c in v] for p, v in first.items()} == {
            p: [c.label for c in v] for p, v in second.items()
        }

    def test_a_different_seed_draws_different_cells(self) -> None:
        a, _ = select_cells(_measurements(), "C3", MontageConfig(seed=1))
        b, _ = select_cells(_measurements(), "C3", MontageConfig(seed=2))
        assert [c.label for c in a["G1"]] != [c.label for c in b["G1"]]

    def test_draw_does_not_depend_on_row_order(self) -> None:
        """The DB may return rows in any order; the draw must not follow it."""
        df = _measurements()
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
        config = MontageConfig(seed=3)
        a, _ = select_cells(df, "C3", config)
        b, _ = select_cells(shuffled, "C3", config)
        assert [c.label for c in a["G1"]] == [c.label for c in b["G1"]]

    def test_no_duplicate_cells_within_a_phase(self) -> None:
        cells, _ = select_cells(
            _measurements(), "C3", MontageConfig(cells_per_phase=5)
        )
        for refs in cells.values():
            labels = [c.label for c in refs]
            assert len(labels) == len(set(labels))

    def test_cells_come_from_the_phase_they_are_filed_under(self) -> None:
        df = _measurements()
        cells, _ = select_cells(df, "C3", MontageConfig())
        by_label = dict(
            zip(df["label"].to_list(), df["cell_cycle"].to_list(), strict=True)
        )
        for phase, refs in cells.items():
            for ref in refs:
                assert by_label[ref.label] == phase

    def test_sparse_phase_reports_rather_than_fails(self) -> None:
        cells, warnings = select_cells(
            _measurements(per_phase={"G1": 10, "S": 1, "G2/M": 0}),
            "C3",
            MontageConfig(cells_per_phase=4, phases=("G1", "S", "G2/M")),
        )
        assert len(cells["S"]) == 1
        assert cells["G2/M"] == []
        assert any("only 1 S" in w for w in warnings)
        assert any("no G2/M" in w for w in warnings)

    def test_edge_filter_drops_cells_whose_crop_overruns(self) -> None:
        df = pl.DataFrame(
            [
                {
                    "plate_id": 99,
                    "well": "C3",
                    "label": 1,
                    "cell_cycle": "G1",
                    "centroid_y": 5.0,
                    "centroid_x": 5.0,
                    "equivalent_diameter_area_cell": 34.0,
                },
                {
                    "plate_id": 99,
                    "well": "C3",
                    "label": 2,
                    "cell_cycle": "G1",
                    "centroid_y": 250.0,
                    "centroid_x": 250.0,
                    "equivalent_diameter_area_cell": 34.0,
                },
            ]
        )
        cells, _ = select_cells(
            df,
            "C3",
            MontageConfig(cells_per_phase=2, phases=("G1",)),
            canvas_hw=(512, 512),
            crop_px=128,
        )
        assert [c.label for c in cells["G1"]] == [2]

    def test_unknown_well_raises(self) -> None:
        with pytest.raises(MontageError, match="no measurements"):
            select_cells(_measurements(), "Z9", MontageConfig())

    def test_missing_phase_column_raises(self) -> None:
        df = _measurements().drop("cell_cycle")
        with pytest.raises(MontageError, match="cell-cycle column"):
            select_cells(df, "C3", MontageConfig())


class TestCropSizing:
    def test_large_cells_give_a_larger_crop(self) -> None:
        """A size tuned for G1 would clip exactly the polyploid phenotype."""
        small = _crop_pixels(
            MontageConfig(phases=("G1",)),
            _measurements(per_phase={"G1": 20}),
            "C3",
            0.3,
        )
        big = _crop_pixels(
            MontageConfig(phases=("Polyploid",)),
            _measurements(per_phase={"Polyploid": 20}),
            "C3",
            0.3,
        )
        assert big > small

    def test_does_not_depend_on_the_draw(self) -> None:
        """Sized from the plate, so the crop is stable across seeds and wells."""
        df = _measurements()
        a = _crop_pixels(MontageConfig(seed=1), df, "C3", 0.3)
        b = _crop_pixels(MontageConfig(seed=99), df, "C3", 0.3)
        assert a == b

    def test_only_shown_phases_count(self) -> None:
        """Excluding Polyploid must not leave the crop sized for it."""
        df = _measurements()
        with_poly = _crop_pixels(
            MontageConfig(phases=("G1", "Polyploid")), df, "C3", 0.3
        )
        without = _crop_pixels(MontageConfig(phases=("G1",)), df, "C3", 0.3)
        assert with_poly > without

    def test_explicit_microns_win(self) -> None:
        assert (
            _crop_pixels(
                MontageConfig(crop_um=30.0), _measurements(), "C3", 0.5
            )
            == 60
        )

    def test_always_even_and_bounded(self) -> None:
        px = _crop_pixels(MontageConfig(), _measurements(), "C3", 0.3)
        assert px % 2 == 0
        assert 32 <= px <= 1024


class TestOverlayResolution:
    def test_splits_composite_from_greyscale(self) -> None:
        overlay, grey = _resolve_overlay(CHANNELS, ("dapi", "tub"))
        assert [CHANNELS[i] for i in overlay] == ["DAPI_R1", "Tub_R1"]
        assert [CHANNELS[i] for i in grey] == ["Y15_R1", "p21_R2", "TP53_R3"]

    def test_matches_through_the_round_suffix(self) -> None:
        overlay, _ = _resolve_overlay(["DAPI_R1", "Tub_R1"], ("dapi",))
        assert overlay == [0]

    def test_repeated_stain_stays_greyscale(self) -> None:
        """Only the first match per role joins the composite."""
        names = ["DAPI_R1", "Tub_R1", "DAPI_R2"]
        overlay, grey = _resolve_overlay(names, ("dapi", "tub"))
        assert overlay == [0, 1]
        assert grey == [2]

    def test_absent_overlay_channel_is_skipped(self) -> None:
        overlay, grey = _resolve_overlay(["p21_R1", "TP53_R1"], ("dapi",))
        assert overlay == []
        assert grey == [0, 1]


class TestOutline:
    def test_outlines_only_the_target_cell(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint32)
        mask[5:15, 5:15] = 1
        mask[25:35, 25:35] = 2
        outline = _outline(mask, 1)
        assert outline[:20, :20].any()
        assert not outline[20:, 20:].any()

    def test_absent_label_outlines_nothing(self) -> None:
        """The centroid can land on a neighbour; outline nothing, not the wrong cell."""
        mask = np.zeros((20, 20), dtype=np.uint32)
        mask[5:10, 5:10] = 3
        assert not _outline(mask, 99).any()


class TestLimitsAndRender:
    def test_limits_are_per_channel_and_shared(self) -> None:
        _build_plate()
        limits = channel_limits(99, "C3")
        assert set(limits) == set(range(len(CHANNELS)))
        for lo, hi in limits.values():
            assert lo < hi

    def test_build_montage_resolves_everything(self) -> None:
        _build_plate()
        montage = build_montage(99, "C3", _measurements(), MontageConfig())
        assert montage.channel_names == CHANNELS
        assert [montage.channel_names[i] for i in montage.overlay_indices] == [
            "DAPI_R1",
            "Tub_R1",
        ]
        assert len(montage.grey_indices) == 3
        assert set(montage.cells) == set(DEFAULT_PHASES)
        assert montage.pixel_size_um == pytest.approx(PIXEL_SIZE_UM)

    def test_missing_zarr_store_raises(self) -> None:
        with pytest.raises(MontageError, match="no zarr cache"):
            build_montage(12345, "C3", _measurements(), MontageConfig())

    def test_exports_a_pdf(self, tmp_path: Path) -> None:
        _build_plate()
        out = tmp_path / "figs"
        path = export_well_pdf(
            99, "C3", _measurements(), out, MontageConfig(cells_per_phase=2)
        )
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.read_bytes()[:4] == b"%PDF"

    def test_export_is_reproducible(self, tmp_path: Path) -> None:
        """Same seed, same plate: the same cells are drawn."""
        _build_plate()
        config = MontageConfig(seed=5, cells_per_phase=2)
        a = build_montage(99, "C3", _measurements(), config)
        b = build_montage(99, "C3", _measurements(), config)
        assert {p: [c.label for c in v] for p, v in a.cells.items()} == {
            p: [c.label for c in v] for p, v in b.cells.items()
        }


class TestRealSchemaColumns:
    """Column names as omero-screen actually writes them.

    Plate 4127 exports `centroid-0-nuc` / `centroid-1-nuc`, not `centroid_y` /
    `centroid_x`, so the first run against real data failed with "No centroid
    columns". regionprops names are per compartment and the montage has to
    accept all of them.
    """

    def _regionprops_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "plate_id": 4127,
                    "well": "C3",
                    "label": 2184,
                    "cell_cycle": "G1",
                    "centroid-0-nuc": 900.0,
                    "centroid-1-nuc": 800.0,
                    "centroid-0-cell": 905.0,
                    "centroid-1-cell": 812.0,
                    "equivalent_diameter_area_cell": 41.0,
                    "equivalent_diameter_area_nucleus": 14.0,
                }
            ]
        )

    def test_regionprops_centroid_columns_resolve(self) -> None:
        cells, _ = select_cells(
            self._regionprops_df(), "C3", MontageConfig(phases=("G1",))
        )
        assert len(cells["G1"]) == 1

    def test_nucleus_centroid_is_preferred_over_cell(self) -> None:
        """The nucleus is the segmentation anchor the phase call is made on."""
        cells, _ = select_cells(
            self._regionprops_df(), "C3", MontageConfig(phases=("G1",))
        )
        assert cells["G1"][0].centroid == (900.0, 800.0)

    def test_cell_extent_preferred_over_nuclear(self) -> None:
        """The crop must hold the cell; in polyploid cells the two diverge."""
        cells, _ = select_cells(
            self._regionprops_df(), "C3", MontageConfig(phases=("G1",))
        )
        assert cells["G1"][0].diameter == 41.0

    def test_error_names_the_columns_it_found(self) -> None:
        df = (
            self._regionprops_df()
            .rename(
                {
                    "centroid-0-nuc": "y_position",
                    "centroid-1-nuc": "x_position",
                }
            )
            .drop(["centroid-0-cell", "centroid-1-cell"])
        )
        with pytest.raises(MontageError, match="no centroid columns at all"):
            select_cells(df, "C3", MontageConfig())

    def test_area_columns_still_work(self) -> None:
        """Plates without an equivalent-diameter column fall back to area."""
        df = (
            self._regionprops_df()
            .drop(
                [
                    "equivalent_diameter_area_cell",
                    "equivalent_diameter_area_nucleus",
                ]
            )
            .with_columns(pl.lit(1256.0).alias("area_cell"))
        )
        cells, _ = select_cells(df, "C3", MontageConfig(phases=("G1",)))
        # area 1256 -> equivalent diameter 2*sqrt(1256/pi) ~= 40
        assert cells["G1"][0].diameter == pytest.approx(40.0, abs=0.5)


class TestMaskLabelResolution:
    """CellView's `label` is the nucleus label; the cell mask is labelled independently.

    On plate 4127, nucleus 2184 sits inside cell 2152. Matching on the ID alone
    found nothing and every panel was drawn with no outline at all — silently,
    because an empty boundary is a valid array.
    """

    def _mask(self, label: int) -> np.ndarray:
        mask = np.zeros((40, 40), dtype=np.uint32)
        mask[10:30, 10:30] = label
        return mask

    def test_matching_label_is_used_directly(self) -> None:
        assert resolve_mask_label(self._mask(2184), 2184) == 2184

    def test_falls_back_to_the_label_under_the_centre(self) -> None:
        """The nucleus centroid is the crop centre by construction."""
        assert resolve_mask_label(self._mask(2152), 2184) == 2152

    def test_background_centre_resolves_to_nothing(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint32)
        mask[0:5, 0:5] = 7
        assert resolve_mask_label(mask, 2184) is None

    def test_outline_is_drawn_for_the_resolved_label(self) -> None:
        mask = self._mask(2152)
        outline = _outline(mask, resolve_mask_label(mask, 2184))
        assert outline.any(), "the cell must be outlined via the centre lookup"

    def test_outline_of_none_is_empty_not_an_error(self) -> None:
        assert not _outline(self._mask(5), None).any()

    def test_outline_does_not_cover_the_cell_interior(self) -> None:
        """It is an outline, not a filled overlay — the pixels must stay visible."""
        mask = self._mask(3)
        outline = _outline(mask, 3, width=3)
        interior = np.zeros_like(mask, dtype=bool)
        interior[15:25, 15:25] = True
        assert not (outline & interior).any()

    def test_outline_width_is_thicker_than_one_pixel(self) -> None:
        """A 1px boundary is invisible once scaled to a montage panel."""
        mask = self._mask(3)
        assert (
            _outline(mask, 3, width=3).sum() > _outline(mask, 3, width=1).sum()
        )


class TestPanelAnnotation:
    """The contour and scale bar go on *every* panel, not just the composite.

    A montage is read panel by panel — a reader looking at one marker in
    isolation has no other cue to which cell is the subject, or to scale.
    """

    def _figure(self):  # type: ignore[no-untyped-def]
        _build_plate()
        montage = build_montage(
            99, "C3", _measurements(), MontageConfig(cells_per_phase=1)
        )
        return render_montage(montage), montage

    def test_every_panel_has_a_scale_bar(self) -> None:
        fig, montage = self._figure()
        try:
            for ax in fig.axes:
                assert ax.patches, "panel has no scale bar"
        finally:
            _close(fig)

    def test_every_panel_carries_an_image(self) -> None:
        fig, montage = self._figure()
        try:
            n_cols = 1 + len(montage.grey_indices)
            assert len(fig.axes) == n_cols * sum(
                len(v) for v in montage.cells.values()
            )
            for ax in fig.axes:
                assert ax.images
        finally:
            _close(fig)

    def test_greyscale_panels_are_rgb_so_the_contour_can_be_coloured(
        self,
    ) -> None:
        fig, montage = self._figure()
        try:
            # Column 0 is the composite; the rest are the greyscale markers.
            for ax in fig.axes:
                data = ax.images[0].get_array()
                assert data.ndim == 3 and data.shape[2] == 3
        finally:
            _close(fig)

    def test_contour_colour_present_on_a_grey_panel(self) -> None:
        """The grey panel must actually carry the outline pixels."""
        _build_plate()
        montage = build_montage(
            99, "C3", _measurements(), MontageConfig(cells_per_phase=1)
        )
        fig = render_montage(montage)
        try:
            n_cols = 1 + len(montage.grey_indices)
            grey_ax = fig.axes[1]  # first row, first greyscale marker
            data = np.asarray(grey_ax.images[0].get_array())
            # Outline pixels are the only ones where the three bands differ.
            assert (data[..., 0] != data[..., 2]).any(), (
                "greyscale panel carries no coloured contour"
            )
            assert n_cols > 1
        finally:
            _close(fig)

    def test_no_scale_bar_without_a_pixel_size(self) -> None:
        """A bar with no known scale would be a lie."""
        from omero_screen_napari.phase_montage import _add_scale_bar

        fig = None
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            _add_scale_bar(ax, 128, None)
            assert not ax.patches
        finally:
            plt.close(fig)

    def test_no_scale_bar_when_it_would_span_the_panel(self) -> None:
        from omero_screen_napari.phase_montage import _add_scale_bar

        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            # 20 um at 1 um/px is 20px, against a 24px crop.
            _add_scale_bar(ax, 24, 1.0)
            assert not ax.patches
        finally:
            plt.close(fig)


class TestBatchExport:
    """Whole-plate export. One bad well must not abandon the other twenty."""

    def test_all_wells_by_default(self, tmp_path: Path) -> None:
        _build_plate(well="C3")
        df = _measurements(well="C3")
        paths, failures = export_plate_pdfs(
            99, df, tmp_path, MontageConfig(cells_per_phase=1)
        )
        assert [p.name for p in paths] == ["plate99_C3_phase_montage.pdf"]
        assert failures == []

    def test_plate_wells_lists_every_well(self) -> None:
        df = pl.concat([_measurements(well="C3"), _measurements(well="A1")])
        assert plate_wells(df) == ["A1", "C3"]

    def test_explicit_subset_is_honoured(self, tmp_path: Path) -> None:
        _build_plate(well="C3")
        df = pl.concat([_measurements(well="C3"), _measurements(well="A1")])
        paths, _ = export_plate_pdfs(
            99,
            df,
            tmp_path,
            MontageConfig(cells_per_phase=1),
            wells=["C3"],
        )
        assert len(paths) == 1

    def test_a_bad_well_is_reported_not_fatal(self, tmp_path: Path) -> None:
        """The whole point of the batch: 20 good montages survive 1 bad well."""
        _build_plate(well="C3")
        df = pl.concat([_measurements(well="C3"), _measurements(well="Z9")])
        paths, failures = export_plate_pdfs(
            99, df, tmp_path, MontageConfig(cells_per_phase=1)
        )
        assert len(paths) == 1
        assert len(failures) == 1
        assert failures[0].startswith("Z9:")

    def test_progress_callback_reports_each_well(self, tmp_path: Path) -> None:
        _build_plate(well="C3")
        df = pl.concat([_measurements(well="C3"), _measurements(well="A1")])
        seen: list[tuple[str, int, int]] = []
        export_plate_pdfs(
            99,
            df,
            tmp_path,
            MontageConfig(cells_per_phase=1),
            on_progress=lambda w, i, n: seen.append((w, i, n)),
        )
        assert seen == [("A1", 0, 2), ("C3", 1, 2)]
