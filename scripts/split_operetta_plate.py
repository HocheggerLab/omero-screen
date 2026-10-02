#!/usr/bin/env python3
r"""Split a PerkinElmer Operetta/Harmony measurement into several plates.

A Harmony export is a directory holding ``Images/Index.xml`` (or
``Index.idx.xml``) plus one TIFF per plane, and optionally an ``Assaylayout``.
Sometimes one physical plate carries two unrelated experiments -- different
labels, different cell lines -- and they should reach OMERO as two separate
plates.

This script rewrites the index rather than regenerating it: every element of
the original is preserved (``Maps`` with its flatfield profiles, the real
``OrientationMatrix``, per-image timestamps and stage positions), and only the
well/image lists are pruned. The TIFFs are hardlinked by default, so splitting
a 45 GB measurement costs no extra disk and takes seconds.

Example:
    Split the first two wells of a demo plate into one plate each::

        python scripts/split_operetta_plate.py \\
            "/Volumes/Helfrid/20260911_20x_demo__...-Measurement 1" \\
            --group B2 --group C2 \\
            --names demo_mCherry,demo_EGFP

Each group is either a single well (``A1``) or an inclusive range in plate
reading order (``A1-B5`` covers A1..A12, B1..B5 on a 12-column plate).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

#: ``B2`` / ``b2`` / ``B02`` -- a row letter (A-P) plus a 1-based column.
WELL_RE = re.compile(r"^([A-Za-z]{1,2})(\d{1,2})$")

#: Index files Harmony is known to write, in the order we look for them.
INDEX_NAMES = ("Index.idx.xml", "Index.xml")

#: How a plane's TIFF is linked into the split measurement.
LINK_MODES = ("hardlink", "copy", "symlink", "move")

#: Report progress no more often than this, in seconds.
PROGRESS_INTERVAL = 2.0


class SplitError(RuntimeError):
    """A problem with the source measurement or the requested split."""


@dataclass(frozen=True)
class Well:
    """A plate well addressed by 1-based row and column."""

    row: int
    col: int

    @property
    def label(self) -> str:
        """Human-facing name, e.g. ``B2``."""
        return f"{chr(ord('A') + self.row - 1)}{self.col}"

    @property
    def harmony_id(self) -> str:
        """Harmony well id, e.g. ``0202``."""
        return f"{self.row:02d}{self.col:02d}"


@dataclass(frozen=True)
class Group:
    """One output plate: a name and the wells that go into it."""

    name: str
    wells: tuple[Well, ...]


def parse_well(text: str) -> Well:
    """Parse a well label such as ``A1`` or ``B02``.

    Args:
        text: Well label, case-insensitive.

    Returns:
        The parsed well.

    Raises:
        SplitError: If ``text`` is not a well label.
    """
    match = WELL_RE.match(text.strip())
    if match is None:
        raise SplitError(f"{text!r} is not a well label (expected e.g. 'A1')")
    letters, digits = match.groups()
    row = 0
    for char in letters.upper():
        row = row * 26 + (ord(char) - ord("A") + 1)
    col = int(digits)
    if col < 1:
        raise SplitError(f"{text!r} has a column below 1")
    return Well(row, col)


def expand_range(spec: str, columns: int) -> tuple[Well, ...]:
    """Expand a ``--group`` argument into wells in plate reading order.

    Args:
        spec: Either a single well (``A1``) or an inclusive range (``A1-B5``).
        columns: Number of columns on the plate, used to wrap a range at the
            end of a row.

    Returns:
        The wells covered by ``spec``.

    Raises:
        SplitError: If the spec is malformed or the range runs backwards.
    """
    parts = spec.split("-")
    if len(parts) == 1:
        return (parse_well(parts[0]),)
    if len(parts) != 2:
        raise SplitError(
            f"{spec!r} is not a well or a range (expected 'A1' or 'A1-B5')"
        )
    start, stop = parse_well(parts[0]), parse_well(parts[1])
    for well in (start, stop):
        if well.col > columns:
            raise SplitError(
                f"{spec!r} refers to column {well.col} "
                f"but the plate has {columns} columns"
            )
    first = (start.row - 1) * columns + start.col
    last = (stop.row - 1) * columns + stop.col
    if last < first:
        raise SplitError(
            f"{spec!r} runs backwards: {stop.label} comes before {start.label}"
        )
    return tuple(
        Well((index - 1) // columns + 1, (index - 1) % columns + 1)
        for index in range(first, last + 1)
    )


def find_index(source: Path) -> Path:
    """Locate the Harmony index inside a measurement directory.

    Args:
        source: The measurement directory (the one holding ``Images/``).

    Returns:
        Path to the index XML.

    Raises:
        SplitError: If no index file is found.
    """
    for name in INDEX_NAMES:
        candidate = source / "Images" / name
        if candidate.is_file():
            return candidate
    raise SplitError(
        f"no {' or '.join(INDEX_NAMES)} under {source / 'Images'} -- "
        "is this an Operetta measurement directory?"
    )


def namespace_of(root: ET.Element) -> str:
    """Return the default XML namespace of ``root``, or ``''`` if unqualified."""
    return root.tag[1:].split("}")[0] if root.tag.startswith("{") else ""


class Index:
    """A parsed Harmony index, with the helpers needed to prune it."""

    def __init__(self, root: ET.Element, path: Path) -> None:
        """Wrap an already-parsed index root.

        Args:
            root: The ``EvaluationInputData`` element.
            path: Where the index came from; used in error messages and to
                name the file written back out.

        Raises:
            SplitError: If ``root`` is not a Harmony index.
        """
        self.path = path
        self.root = root
        self.ns = namespace_of(root)
        if not root.tag.endswith("EvaluationInputData"):
            raise SplitError(
                f"{path} is not a Harmony index (root element is {root.tag!r})"
            )
        plate = self.find(root, "Plates/Plate")
        if plate is None:
            raise SplitError(f"{path} has no <Plates>/<Plate> element")
        self.plate = plate

    @classmethod
    def from_path(cls, path: Path) -> Index:
        """Parse the index file at ``path``."""
        return cls(ET.parse(path).getroot(), path)

    def copy(self) -> Index:
        """Return an independent deep copy of this index."""
        return Index(
            ET.fromstring(ET.tostring(self.root, encoding="unicode")),
            self.path,
        )

    def tag(self, name: str) -> str:
        """Qualify ``name`` with the document namespace."""
        return f"{{{self.ns}}}{name}" if self.ns else name

    def qualify(self, path: str) -> str:
        """Qualify every step of an ElementTree path."""
        return "/".join(self.tag(step) for step in path.split("/"))

    def find(self, parent: ET.Element, path: str) -> ET.Element | None:
        """Find a single descendant by namespace-free path."""
        return parent.find(self.qualify(path))

    def findall(self, parent: ET.Element, path: str) -> list[ET.Element]:
        """Find all descendants matching a namespace-free path."""
        return parent.findall(self.qualify(path))

    def text(self, parent: ET.Element, name: str) -> str:
        """Return the text of a child element, or ``''`` if absent/empty."""
        child = parent.find(self.tag(name))
        return "" if child is None or child.text is None else child.text

    @property
    def plate_name(self) -> str:
        """``PlateID`` of the source plate, falling back to ``Name``."""
        return self.text(self.plate, "PlateID") or self.text(
            self.plate, "Name"
        )

    @property
    def columns(self) -> int:
        """Number of plate columns declared in the index."""
        raw = self.text(self.plate, "PlateColumns")
        if not raw.isdigit():
            raise SplitError(
                f"{self.path} declares no usable <PlateColumns> (got {raw!r})"
            )
        return int(raw)

    def wells(self) -> list[Well]:
        """Wells present in the measurement, in index order."""
        found: list[Well] = []
        for element in self.findall(self.root, "Wells/Well"):
            row, col = self.text(element, "Row"), self.text(element, "Col")
            if row.isdigit() and col.isdigit():
                found.append(Well(int(row), int(col)))
        return found


def _element_well(index: Index, element: ET.Element) -> Well | None:
    """Read the ``Row``/``Col`` pair off an element, if it has one."""
    row, col = index.text(element, "Row"), index.text(element, "Col")
    if row.isdigit() and col.isdigit():
        return Well(int(row), int(col))
    return None


def _prune_children(
    parent: ET.Element, keep: Iterable[ET.Element]
) -> list[ET.Element]:
    """Remove from ``parent`` every child not in ``keep``.

    In a pretty-printed document the indentation preceding a closing tag lives
    in the *last* child's ``tail``. Dropping that child would leave the closing
    tag mis-indented, so the original tail is carried over to whichever child
    ends up last.

    Args:
        parent: Element to prune in place.
        keep: Children to retain.

    Returns:
        The retained children, in their original order.
    """
    closing_tail = parent[-1].tail if len(parent) else None
    kept = set(map(id, keep))
    retained = [child for child in parent if id(child) in kept]
    for child in list(parent):
        if id(child) not in kept:
            parent.remove(child)
    if retained:
        retained[-1].tail = closing_tail
    return retained


def build_split_index(source: Index, group: Group) -> tuple[Index, list[str]]:
    """Produce a pruned copy of the index for one output plate.

    Everything the source declares is preserved -- including ``<Maps>`` and
    every per-image field -- and only the well and image lists are filtered.

    Args:
        source: The parsed source index.
        group: The output plate and its wells.

    Returns:
        The pruned index and the TIFF file names it references.

    Raises:
        SplitError: If none of the group's wells exist in the measurement.
    """
    index = source.copy()
    plate = index.plate

    wanted = {well.harmony_id for well in group.wells}

    wells_el = index.find(index.root, "Wells")
    kept_wells = (
        [
            element
            for element in wells_el
            if (found := _element_well(index, element)) is not None
            and found.harmony_id in wanted
        ]
        if wells_el is not None
        else []
    )
    if not kept_wells:
        labels = ", ".join(well.label for well in group.wells)
        raise SplitError(
            f"group {group.name!r} selects no wells present in the "
            f"measurement (asked for {labels})"
        )
    if wells_el is not None:
        _prune_children(wells_el, kept_wells)

    kept_ids = {index.text(element, "id") for element in kept_wells}
    _prune_children(
        plate,
        [
            child
            for child in plate
            if not child.tag.endswith("Well") or child.get("id") in kept_ids
        ],
    )

    images_el = index.find(index.root, "Images")
    urls: list[str] = []
    if images_el is not None:
        kept_images = []
        for element in images_el:
            well = _element_well(index, element)
            if well is not None and well.harmony_id in wanted:
                kept_images.append(element)
                url = index.text(element, "URL")
                if url:
                    urls.append(url)
        _prune_children(images_el, kept_images)

    for name in ("PlateID", "Name"):
        child = plate.find(index.tag(name))
        if child is not None:
            child.text = group.name

    return index, urls


def write_index(index: Index, destination: Path) -> None:
    """Write a pruned index the way Harmony writes one.

    Harmony emits UTF-8 with a byte-order mark and a double-quoted XML
    declaration; Bio-Formats is happy either way, but matching keeps the output
    diffable against the source.

    Args:
        index: The pruned index to serialise. Its default namespace is
            re-registered so tags stay unprefixed.
        destination: File to write.
    """
    if index.ns:
        ET.register_namespace("", index.ns)
    body = ET.tostring(index.root, encoding="utf-8", xml_declaration=False)
    declaration = b'<?xml version="1.0" encoding="utf-8"?>\n'
    destination.write_bytes(b"\xef\xbb\xbf" + declaration + body)


def link_file(source: Path, destination: Path, mode: str) -> None:
    """Materialise ``source`` at ``destination`` using ``mode``.

    Args:
        source: Existing TIFF.
        destination: Path to create.
        mode: One of :data:`LINK_MODES`, already resolved by
            :func:`resolve_mode` -- no silent fallback happens here.

    Raises:
        SplitError: If the operation fails.
    """
    try:
        if mode == "copy":
            shutil.copy2(source, destination)
        elif mode == "symlink":
            destination.symlink_to(source.resolve())
        elif mode == "move":
            shutil.move(str(source), str(destination))
        else:
            os.link(source, destination)
    except OSError as error:
        raise SplitError(
            f"could not {mode} {source} -> {destination}: {error}"
        ) from error


def supports_hardlinks(directory: Path) -> bool:
    """Probe whether ``directory`` can hold a hardlink.

    exFAT -- the usual format for the USB disks these measurements travel on --
    has no hardlinks at all, so this has to be tested rather than assumed.

    Args:
        directory: An existing, writable directory to probe in.

    Returns:
        ``True`` if a hardlink could be created and removed.
    """
    probe = directory / ".split_operetta_linkprobe"
    link = directory / ".split_operetta_linkprobe.link"
    try:
        probe.write_bytes(b"")
        link.unlink(missing_ok=True)
        os.link(probe, link)
        return True
    except OSError:
        return False
    finally:
        link.unlink(missing_ok=True)
        probe.unlink(missing_ok=True)


def resolve_mode(requested: str, source_images: Path, output_dir: Path) -> str:
    """Decide how TIFFs will be placed, before any of them are touched.

    A hardlink request that the destination filesystem cannot honour becomes a
    copy -- but the caller is told, so a 45 GB copy is never a surprise.

    Args:
        requested: The ``--mode`` argument.
        source_images: Source ``Images/`` directory.
        output_dir: Directory the split measurements will be created in.

    Returns:
        The mode that will actually be used.
    """
    if requested != "hardlink":
        return requested
    if source_images.stat().st_dev != output_dir.stat().st_dev:
        print(
            "note: source and output are on different filesystems, "
            "so TIFFs will be copied rather than hardlinked"
        )
        return "copy"
    if not supports_hardlinks(output_dir):
        print(
            "note: this filesystem does not support hardlinks (exFAT and FAT "
            "never do), so TIFFs will be copied rather than hardlinked"
        )
        return "copy"
    return "hardlink"


def human_bytes(count: float) -> str:
    """Render a byte count in the largest unit that keeps it above 1."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if count < 1024 or unit == "TB":
            return f"{count:.1f} {unit}"
        count /= 1024
    raise AssertionError("unreachable")


def copy_tiffs(
    source_images: Path,
    destination_images: Path,
    urls: Sequence[str],
    mode: str,
    dry_run: bool,
    label: str = "",
) -> tuple[int, int]:
    """Place every referenced TIFF into the split measurement.

    Copying a large measurement takes minutes, so progress is reported to
    stderr as it goes: a silent process moving tens of gigabytes is
    indistinguishable from a hung one.

    Args:
        source_images: The source ``Images/`` directory.
        destination_images: The destination ``Images/`` directory.
        urls: TIFF file names referenced by the pruned index.
        mode: Resolved link mode.
        dry_run: Report only, touch nothing.
        label: Plate name to prefix progress lines with.

    Returns:
        The number of files and the total bytes they occupy.

    Raises:
        SplitError: If the index references a TIFF that is not on disk.
    """
    names = list(dict.fromkeys(urls))
    total_bytes = 0
    done_bytes = 0
    sizes: list[int] = []
    for name in names:
        origin = source_images / name
        if not origin.is_file():
            raise SplitError(f"index references a missing TIFF: {origin}")
        size = origin.stat().st_size
        sizes.append(size)
        total_bytes += size
    if dry_run:
        return len(names), total_bytes

    verb = {"copy": "copying", "move": "moving"}.get(mode, f"{mode}ing")
    last = 0.0
    for position, (name, size) in enumerate(
        zip(names, sizes, strict=True), start=1
    ):
        target = destination_images / name
        if target.exists() or target.is_symlink():
            target.unlink()
        link_file(source_images / name, target, mode)
        done_bytes += size
        now = time.monotonic()
        if now - last >= PROGRESS_INTERVAL or position == len(names):
            print(
                f"\r  {label}: {verb} {position}/{len(names)} files "
                f"({human_bytes(done_bytes)} of {human_bytes(total_bytes)})",
                end="" if position < len(names) else "\n",
                file=sys.stderr,
                flush=True,
            )
            last = now
    return len(names), total_bytes


def copy_sidecars(source: Path, destination: Path, dry_run: bool) -> list[str]:
    """Copy the non-image parts of a measurement verbatim.

    The assay layout and the ``.ExportDone`` marker describe the whole physical
    plate. They are copied unchanged: the layout carries no per-well data that
    Bio-Formats reads on import, and dropping it would lose the record of how
    the original plate was laid out.

    Args:
        source: Source measurement directory.
        destination: Destination measurement directory.
        dry_run: Report only, touch nothing.

    Returns:
        Names of the entries copied.
    """
    copied: list[str] = []
    for entry in sorted(source.iterdir()):
        if entry.name == "Images":
            continue
        if not dry_run:
            target = destination / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)
        copied.append(entry.name)
    return copied


def resolve_groups(
    specs: Sequence[str], names: Sequence[str] | None, index: Index
) -> list[Group]:
    """Turn ``--group``/``--names`` arguments into named well groups.

    Args:
        specs: Raw ``--group`` arguments.
        names: Optional output plate names, one per group.
        index: The parsed source index, for plate geometry and naming.

    Returns:
        The resolved groups.

    Raises:
        SplitError: On malformed specs, a name/group count mismatch, or a well
            claimed by more than one group.
    """
    if names is not None and len(names) != len(specs):
        raise SplitError(
            f"got {len(names)} --names for {len(specs)} --group arguments"
        )
    plate = index.plate_name or index.path.parent.parent.name
    groups: list[Group] = []
    seen: dict[str, str] = {}
    for position, spec in enumerate(specs):
        wells = expand_range(spec, index.columns)
        name = (
            names[position]
            if names is not None
            else f"{plate}_part{position + 1}"
        )
        for well in wells:
            owner = seen.get(well.harmony_id)
            if owner is not None:
                raise SplitError(
                    f"well {well.label} is claimed by both {owner!r} and {name!r}"
                )
            seen[well.harmony_id] = name
        groups.append(Group(name, wells))
    return groups


def iter_report(groups: Sequence[Group], index: Index) -> Iterator[str]:
    """Yield one summary line per group, plus a line for dropped wells."""
    present = {well.harmony_id: well for well in index.wells()}
    claimed: set[str] = set()
    for group in groups:
        hits = [w for w in group.wells if w.harmony_id in present]
        misses = [w for w in group.wells if w.harmony_id not in present]
        claimed.update(w.harmony_id for w in hits)
        line = f"{group.name}: {', '.join(w.label for w in hits) or '(none)'}"
        if misses:
            line += (
                f"  [not in measurement: {', '.join(w.label for w in misses)}]"
            )
        yield line
    dropped = [w.label for wid, w in present.items() if wid not in claimed]
    if dropped:
        yield f"dropped (in no group): {', '.join(dropped)}"


def split_measurement(
    source: Path,
    groups: Sequence[Group],
    output_dir: Path,
    mode: str,
    dry_run: bool,
    force: bool,
) -> None:
    """Write one measurement directory per group.

    The index and sidecars are written *before* the TIFFs are placed, so an
    interrupted run leaves a directory that is obviously incomplete in its
    image count rather than one missing its index entirely.

    Args:
        source: Source measurement directory.
        groups: Output plates and their wells.
        output_dir: Directory the new measurements are created in.
        mode: Resolved link mode.
        dry_run: Report only, touch nothing.
        force: Overwrite an existing output directory.

    Raises:
        SplitError: If an output directory exists and ``force`` is not set.
    """
    index = Index.from_path(find_index(source))
    source_images = index.path.parent
    for group in groups:
        split, urls = build_split_index(index, group)
        destination = output_dir / group.name
        if destination.exists():
            if not force:
                raise SplitError(
                    f"{destination} already exists (pass --force to overwrite)"
                )
            if not dry_run:
                shutil.rmtree(destination)
        if not dry_run:
            (destination / "Images").mkdir(parents=True)
            write_index(split, destination / "Images" / index.path.name)
        sidecars = copy_sidecars(source, destination, dry_run)
        count, size = copy_tiffs(
            source_images,
            destination / "Images",
            urls,
            mode,
            dry_run,
            group.name,
        )
        print(
            f"  {destination}: {count} TIFFs, {human_bytes(size)} ({mode}), "
            f"index {index.path.name}, sidecars {', '.join(sidecars) or 'none'}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build and run the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Split an Operetta/Harmony measurement into several measurements, "
            "one per group of wells, each importable into OMERO as its own plate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Each --group is a single well (A1) or an inclusive range in plate\n"
            "reading order (A1-B5 covers A1..A12, B1..B5 on a 12-column plate).\n\n"
            "Example:\n"
            "  split_operetta_plate.py '.../Measurement 1' \\\n"
            "      --group B2 --group C2 --names demo_mCherry,demo_EGFP"
        ),
    )
    parser.add_argument(
        "source", type=Path, help="measurement directory (contains Images/)"
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="WELLS",
        help="wells for one output plate; repeat once per plate",
    )
    parser.add_argument(
        "--names",
        metavar="A,B",
        help="comma-separated output plate names (default: <plate>_partN)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="where to create the new measurements (default: alongside source)",
    )
    parser.add_argument(
        "--mode",
        choices=LINK_MODES,
        default="hardlink",
        help=(
            "how TIFFs are placed: hardlink (default; instant and free, but "
            "needs a filesystem that supports them -- exFAT and FAT do not, "
            "and the script falls back to copy with a warning), copy, symlink, "
            "or move (instant on one filesystem, but consumes the source)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without creating anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing output directories",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Returns:
        ``0`` on success, ``1`` if the split could not be performed.
    """
    args = parse_args(argv)
    try:
        source = args.source.expanduser()
        if not source.is_dir():
            raise SplitError(f"{source} is not a directory")
        index = Index.from_path(find_index(source))
        names = args.names.split(",") if args.names else None
        groups = resolve_groups(args.group, names, index)
        output_dir = (args.output_dir or source.parent).expanduser()

        print(f"source: {source}")
        print(f"plate:  {index.plate_name} ({len(index.wells())} wells)")
        for line in iter_report(groups, index):
            print(f"  {line}")
        if args.dry_run:
            print("dry run -- nothing written")
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = resolve_mode(args.mode, find_index(source).parent, output_dir)
        if mode == "move" and not args.dry_run:
            print(
                "warning: --mode move consumes the source measurement; "
                f"{source} will no longer be importable"
            )
        split_measurement(
            source, groups, output_dir, mode, args.dry_run, args.force
        )
    except SplitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
