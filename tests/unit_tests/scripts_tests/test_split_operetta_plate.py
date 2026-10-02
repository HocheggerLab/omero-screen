"""Tests for ``scripts/split_operetta_plate.py``.

The script's job is to prune a Harmony index without disturbing anything else,
so the tests build a miniature measurement on disk and assert both halves of
that contract: the right wells survive, and everything that is not a well list
comes through unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest

NS = "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"


def _load_script() -> ModuleType:
    """Import the script by path -- ``scripts/`` is not an installed package."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "split_operetta_plate.py"
    spec = importlib.util.spec_from_file_location("split_operetta_plate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


split = _load_script()


def _tiff_name(row: int, col: int, field: int, channel: int) -> str:
    """Harmony's file name for one plane."""
    return f"r{row:02d}c{col:02d}f{field:02d}p01-ch{channel}sk1fk1fl1.tiff"


@pytest.fixture
def measurement(tmp_path: Path) -> Path:
    """A two-well, two-field, two-channel measurement with a Maps block."""
    wells = [(1, 1), (2, 3)]
    fields, channels = (1, 2), (1, 2)

    root = ET.Element(f"{{{NS}}}EvaluationInputData", {"Version": "2"})
    ET.SubElement(root, "User").text = "tester"
    plate = ET.SubElement(ET.SubElement(root, "Plates"), "Plate")
    ET.SubElement(plate, "PlateID").text = "source_plate"
    ET.SubElement(plate, "Name").text = "source_plate"
    ET.SubElement(plate, "PlateRows").text = "8"
    ET.SubElement(plate, "PlateColumns").text = "12"
    for row, col in wells:
        ET.SubElement(plate, "Well", {"id": f"{row:02d}{col:02d}"})

    wells_el = ET.SubElement(root, "Wells")
    images_el = ET.SubElement(root, "Images")
    for row, col in wells:
        well_el = ET.SubElement(wells_el, "Well")
        ET.SubElement(well_el, "id").text = f"{row:02d}{col:02d}"
        ET.SubElement(well_el, "Row").text = str(row)
        ET.SubElement(well_el, "Col").text = str(col)
        for fld in fields:
            for ch in channels:
                image_id = f"{row:02d}{col:02d}K1F{fld}P1R{ch}"
                ET.SubElement(well_el, "Image", {"id": image_id})
                img = ET.SubElement(images_el, "Image", {"Version": "1"})
                ET.SubElement(img, "id").text = image_id
                ET.SubElement(img, "URL").text = _tiff_name(row, col, fld, ch)
                ET.SubElement(img, "Row").text = str(row)
                ET.SubElement(img, "Col").text = str(col)

    maps = ET.SubElement(root, "Maps")
    ET.SubElement(ET.SubElement(maps, "Map"), "Entry", {"ChannelID": "1"}).text = (
        "flatfield-blob"
    )

    source = tmp_path / "source__Measurement 1"
    (source / "Images").mkdir(parents=True)
    (source / "Assaylayout").mkdir()
    (source / "Assaylayout" / "Unnamed.xml").write_text("<AssayLayout />")
    (source / ".ExportDone_IndexAndImages").write_text("done")
    for row, col in wells:
        for fld in fields:
            for ch in channels:
                name = _tiff_name(row, col, fld, ch)
                (source / "Images" / name).write_bytes(name.encode())

    ET.indent(root, space="  ")
    ET.register_namespace("", NS)
    (source / "Images" / "Index.xml").write_bytes(
        b"\xef\xbb\xbf" + ET.tostring(root, encoding="utf-8")
    )
    return source


class TestParseWell:
    """Well-label parsing."""

    @pytest.mark.parametrize(
        ("text", "row", "col"),
        [("A1", 1, 1), ("b2", 2, 2), ("H12", 8, 12), ("B02", 2, 2), ("AA1", 27, 1)],
    )
    def test_accepts_valid_labels(self, text: str, row: int, col: int) -> None:
        """Letters map to 1-based rows and digits to 1-based columns."""
        assert split.parse_well(text) == split.Well(row, col)

    @pytest.mark.parametrize("text", ["", "1A", "A", "A0", "A1-B2", "??"])
    def test_rejects_junk(self, text: str) -> None:
        """Anything that is not a bare well label raises."""
        with pytest.raises(split.SplitError):
            split.parse_well(text)


class TestExpandRange:
    """``--group`` argument expansion."""

    def test_single_well(self) -> None:
        """A bare label yields exactly one well."""
        assert split.expand_range("B2", 12) == (split.Well(2, 2),)

    def test_range_wraps_rows_in_reading_order(self) -> None:
        """A range runs across the end of a row onto the next."""
        wells = split.expand_range("A11-B2", 12)
        assert [w.label for w in wells] == ["A11", "A12", "B1", "B2"]

    def test_range_respects_plate_width(self) -> None:
        """Wrapping uses the plate's own column count, not a fixed 12."""
        wells = split.expand_range("A5-B2", 6)
        assert [w.label for w in wells] == ["A5", "A6", "B1", "B2"]

    def test_backwards_range_raises(self) -> None:
        """The stop well must not precede the start well."""
        with pytest.raises(split.SplitError, match="runs backwards"):
            split.expand_range("C3-A1", 12)

    def test_column_beyond_plate_raises(self) -> None:
        """A column the plate does not have is caught early."""
        with pytest.raises(split.SplitError, match="columns"):
            split.expand_range("A1-B13", 12)

    def test_malformed_spec_raises(self) -> None:
        """More than one dash is not a range."""
        with pytest.raises(split.SplitError):
            split.expand_range("A1-B2-C3", 12)


class TestSplitMeasurement:
    """End-to-end splitting of a miniature measurement."""

    @pytest.fixture
    def outputs(self, measurement: Path, tmp_path: Path) -> tuple[Path, Path]:
        """Split the fixture into ``left`` (A1) and ``right`` (B3)."""
        out = tmp_path / "out"
        groups = [
            split.Group("left", (split.Well(1, 1),)),
            split.Group("right", (split.Well(2, 3),)),
        ]
        split.split_measurement(
            measurement, groups, out, mode="copy", dry_run=False, force=False
        )
        return out / "left", out / "right"

    def test_each_plate_keeps_only_its_own_tiffs(
        self, outputs: tuple[Path, Path]
    ) -> None:
        """TIFFs are partitioned by well, with no leakage either way."""
        left, right = outputs
        assert sorted(p.name for p in (left / "Images").glob("*.tiff")) == [
            _tiff_name(1, 1, f, c) for f in (1, 2) for c in (1, 2)
        ]
        assert sorted(p.name for p in (right / "Images").glob("*.tiff")) == [
            _tiff_name(2, 3, f, c) for f in (1, 2) for c in (1, 2)
        ]

    def test_index_lists_only_the_kept_well(
        self, outputs: tuple[Path, Path]
    ) -> None:
        """``Plates``, ``Wells`` and ``Images`` are pruned consistently."""
        index = split.Index.from_path(outputs[0] / "Images" / "Index.xml")
        assert [w.label for w in index.wells()] == ["A1"]
        assert [
            el.get("id") for el in index.findall(index.plate, "Well")
        ] == ["0101"]
        images = index.findall(index.root, "Images/Image")
        assert len(images) == 4
        assert {index.text(el, "Row") for el in images} == {"1"}

    def test_plate_is_renamed(self, outputs: tuple[Path, Path]) -> None:
        """``PlateID`` and ``Name`` both become the group name."""
        index = split.Index.from_path(outputs[1] / "Images" / "Index.xml")
        assert index.plate_name == "right"
        assert index.text(index.plate, "Name") == "right"

    def test_maps_and_header_survive_untouched(
        self, outputs: tuple[Path, Path]
    ) -> None:
        """Flatfield profiles and plate geometry are carried over verbatim."""
        index = split.Index.from_path(outputs[0] / "Images" / "Index.xml")
        entry = index.find(index.root, "Maps/Map/Entry")
        assert entry is not None and entry.text == "flatfield-blob"
        assert index.columns == 12
        assert index.text(index.plate, "PlateRows") == "8"

    def test_output_is_bom_prefixed_utf8(
        self, outputs: tuple[Path, Path]
    ) -> None:
        """Harmony writes a BOM; Bio-Formats readers expect the same shape."""
        raw = (outputs[0] / "Images" / "Index.xml").read_bytes()
        assert raw.startswith(b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?>')

    def test_sidecars_are_copied(self, outputs: tuple[Path, Path]) -> None:
        """The assay layout and export marker follow each split plate."""
        for plate in outputs:
            assert (plate / "Assaylayout" / "Unnamed.xml").is_file()
            assert (plate / ".ExportDone_IndexAndImages").is_file()

    def test_dry_run_writes_nothing(
        self, measurement: Path, tmp_path: Path
    ) -> None:
        """A dry run reports the plan without creating output."""
        out = tmp_path / "dry"
        split.split_measurement(
            measurement,
            [split.Group("left", (split.Well(1, 1),))],
            out,
            mode="copy",
            dry_run=True,
            force=False,
        )
        assert not (out / "left").exists()

    def test_existing_output_needs_force(
        self, measurement: Path, tmp_path: Path
    ) -> None:
        """Refuse to clobber an existing measurement unless told to."""
        out = tmp_path / "out"
        (out / "left").mkdir(parents=True)
        groups = [split.Group("left", (split.Well(1, 1),))]
        with pytest.raises(split.SplitError, match="--force"):
            split.split_measurement(
                measurement, groups, out, mode="copy", dry_run=False, force=False
            )
        split.split_measurement(
            measurement, groups, out, mode="copy", dry_run=False, force=True
        )
        assert (out / "left" / "Images" / "Index.xml").is_file()

    def test_hardlinks_share_inodes_with_the_source(
        self, measurement: Path, tmp_path: Path
    ) -> None:
        """The default mode costs no extra disk."""
        out = tmp_path / "linked"
        split.split_measurement(
            measurement,
            [split.Group("left", (split.Well(1, 1),))],
            out,
            mode="hardlink",
            dry_run=False,
            force=False,
        )
        name = _tiff_name(1, 1, 1, 1)
        assert (
            (out / "left" / "Images" / name).stat().st_ino
            == (measurement / "Images" / name).stat().st_ino
        )

    def test_move_consumes_the_source(
        self, measurement: Path, tmp_path: Path
    ) -> None:
        """``--mode move`` relocates the TIFFs instead of duplicating them."""
        out = tmp_path / "moved"
        split.split_measurement(
            measurement,
            [split.Group("left", (split.Well(1, 1),))],
            out,
            mode="move",
            dry_run=False,
            force=False,
        )
        name = _tiff_name(1, 1, 1, 1)
        assert (out / "left" / "Images" / name).is_file()
        assert not (measurement / "Images" / name).exists()

    def test_index_is_written_before_the_tiffs(
        self, measurement: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An interrupted run must not leave a measurement with no index."""
        out = tmp_path / "interrupted"
        original = split.copy_tiffs

        def fail_after_index(*args: object, **kwargs: object) -> object:
            raise KeyboardInterrupt

        monkeypatch.setattr(split, "copy_tiffs", fail_after_index)
        with pytest.raises(KeyboardInterrupt):
            split.split_measurement(
                measurement,
                [split.Group("left", (split.Well(1, 1),))],
                out,
                mode="copy",
                dry_run=False,
                force=False,
            )
        assert original is not None
        assert (out / "left" / "Images" / "Index.xml").is_file()

    def test_group_with_no_present_wells_raises(self, measurement: Path) -> None:
        """Selecting only absent wells is an error, not an empty plate."""
        index = split.Index.from_path(split.find_index(measurement))
        with pytest.raises(split.SplitError, match="no wells present"):
            split.build_split_index(
                index, split.Group("empty", (split.Well(8, 12),))
            )

    def test_source_is_left_untouched(
        self, measurement: Path, outputs: tuple[Path, Path]
    ) -> None:
        """Splitting never mutates the original index."""
        index = split.Index.from_path(split.find_index(measurement))
        assert index.plate_name == "source_plate"
        assert [w.label for w in index.wells()] == ["A1", "B3"]


class TestResolveMode:
    """Choosing how TIFFs are placed, before any are touched."""

    def test_explicit_mode_is_never_overridden(self, tmp_path: Path) -> None:
        """Only ``hardlink`` is subject to probing."""
        assert split.resolve_mode("copy", tmp_path, tmp_path) == "copy"
        assert split.resolve_mode("move", tmp_path, tmp_path) == "move"

    def test_hardlink_survives_when_supported(self, tmp_path: Path) -> None:
        """On a normal filesystem the default is kept."""
        assert split.resolve_mode("hardlink", tmp_path, tmp_path) == "hardlink"

    def test_falls_back_to_copy_without_hardlink_support(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """exFAT has no hardlinks, so the fallback must be announced."""
        monkeypatch.setattr(split, "supports_hardlinks", lambda _: False)
        assert split.resolve_mode("hardlink", tmp_path, tmp_path) == "copy"
        assert "does not support hardlinks" in capsys.readouterr().out

    def test_falls_back_across_filesystems(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A destination on another device cannot be hardlinked to."""
        real_stat = Path.stat
        devices = {str(tmp_path / "a"): 1, str(tmp_path / "b"): 2}

        def fake_stat(self: Path, **kwargs: object) -> object:
            result = real_stat(self, **kwargs)  # type: ignore[arg-type]
            if str(self) in devices:
                return type(
                    "S", (), {"st_dev": devices[str(self)], "st_size": result.st_size}
                )()
            return result

        for name in ("a", "b"):
            (tmp_path / name).mkdir()
        monkeypatch.setattr(Path, "stat", fake_stat)
        mode = split.resolve_mode("hardlink", tmp_path / "a", tmp_path / "b")
        assert mode == "copy"
        assert "different filesystems" in capsys.readouterr().out


class TestSupportsHardlinks:
    """The hardlink probe."""

    def test_true_on_a_normal_filesystem(self, tmp_path: Path) -> None:
        """tmp_path is APFS/ext4, which supports links."""
        assert split.supports_hardlinks(tmp_path) is True

    def test_leaves_no_probe_files_behind(self, tmp_path: Path) -> None:
        """The probe cleans up after itself, successfully or not."""
        split.supports_hardlinks(tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_false_when_link_is_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``OSError`` from ``os.link`` means no hardlink support."""

        def refuse(*_: object) -> None:
            raise OSError("Operation not supported")

        monkeypatch.setattr(split.os, "link", refuse)
        assert split.supports_hardlinks(tmp_path) is False
        assert list(tmp_path.iterdir()) == []


class TestHumanBytes:
    """Byte-count formatting."""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(0, "0.0 B"), (1536, "1.5 KB"), (1024**3, "1.0 GB"), (1024**5, "1024.0 TB")],
    )
    def test_scales_to_the_right_unit(self, count: int, expected: str) -> None:
        """Each threshold picks the largest unit that stays above 1."""
        assert split.human_bytes(count) == expected


class TestResolveGroups:
    """Turning CLI arguments into named groups."""

    @pytest.fixture
    def index(self, measurement: Path) -> object:
        """The fixture measurement's parsed index."""
        return split.Index.from_path(split.find_index(measurement))

    def test_default_names_are_numbered_parts(self, index: object) -> None:
        """Without ``--names`` each plate is ``<plate>_partN``."""
        groups = split.resolve_groups(["A1", "B3"], None, index)  # type: ignore[arg-type]
        assert [g.name for g in groups] == ["source_plate_part1", "source_plate_part2"]

    def test_explicit_names_are_used(self, index: object) -> None:
        """``--names`` pairs positionally with ``--group``."""
        groups = split.resolve_groups(["A1", "B3"], ["one", "two"], index)  # type: ignore[arg-type]
        assert [g.name for g in groups] == ["one", "two"]

    def test_name_count_mismatch_raises(self, index: object) -> None:
        """A short ``--names`` list is caught before anything is written."""
        with pytest.raises(split.SplitError, match="--names"):
            split.resolve_groups(["A1", "B3"], ["only-one"], index)  # type: ignore[arg-type]

    def test_overlapping_groups_raise(self, index: object) -> None:
        """A well may belong to at most one output plate."""
        with pytest.raises(split.SplitError, match="claimed by both"):
            split.resolve_groups(["A1-A3", "A3-B1"], None, index)  # type: ignore[arg-type]


class TestFindIndex:
    """Locating the index inside a measurement."""

    def test_prefers_idx_name(self, measurement: Path) -> None:
        """``Index.idx.xml`` wins when both spellings are present."""
        idx = measurement / "Images" / "Index.idx.xml"
        idx.write_bytes((measurement / "Images" / "Index.xml").read_bytes())
        assert split.find_index(measurement) == idx

    def test_missing_index_raises(self, tmp_path: Path) -> None:
        """A directory that is not a measurement is rejected with a hint."""
        with pytest.raises(split.SplitError, match="Operetta measurement"):
            split.find_index(tmp_path)
