"""Reading cyclic-IF (4i) plate alignments for the zarr cache.

A 4i experiment is one dataset spread over several OMERO plates: a **master**
plate carrying the segmentation, plus N **restain** plates imaged in later rounds.
``omero_screen.plate_aggregation.align_plates`` computes the rigid pixel shift
between master and each round and attaches two tables to the *master* plate:

``alignment.csv``
    One ``(x, y)`` per ``(plate, well)`` -- the mean over sampled fields.
``sample_alignment.csv``
    One ``(x, y)`` per ``(plate, well, sample, image_id)`` -- per field.

In both, the ``plate`` column holds only the *restain* plate IDs; the master is
the plate the attachment hangs on.

Sign convention: a restain-plate coordinate maps into master frame by
**subtracting** ``(x, y)``. So a restain tile belongs on the master canvas at
``master_offset - (x, y)``.

Two traps this module exists to avoid:

* ``omero_utils.attachments.get_file_attachments`` matches by *suffix*, so asking
  for ``"alignment.csv"`` also returns ``"sample_alignment.csv"``. Names are
  compared exactly here.
* Until 2026-09-03 ``alignment.csv`` was written with x and y transposed, while
  ``sample_alignment.csv`` was always correct. Post-fix files carry a ``schema``
  column; a per-well table without one is rejected rather than reinterpreted.

This module never opens its own connection -- callers pass one in, because the
cache builder already holds a connection and ``@omero_connect`` would force a
fresh connect/disconnect per plate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from omero.gateway import BlitzGateway, FileAnnotationWrapper
from omero_utils.attachments import parse_csv_data

#: Minimum ``schema`` value accepted in a per-well ``alignment.csv``. Mirrors
#: ``omero_screen.plate_aggregation.ALIGNMENT_SCHEMA_VERSION``; duplicated rather
#: than imported because ``plate_aggregation`` pulls in ``matplotlib.pyplot``,
#: which must not be imported inside the napari Qt process.
MIN_PER_WELL_SCHEMA = 2

PER_FIELD_FILENAME = "sample_alignment.csv"
PER_WELL_FILENAME = "alignment.csv"

AlignmentSource = Literal["sample_alignment", "alignment"]


class AlignmentError(Exception):
    """Raised when a plate's alignment data is missing or unusable."""


@dataclass(frozen=True)
class WellShifts:
    """Per-field shifts for one well of one restain round.

    Attributes:
        shifts: ``(n_fields, 2)`` integer ``(dx, dy)``, indexed by *master* field
            index. Subtract from the master canvas offset to place the tile.
        image_ids: Restain image ID per master field index, or ``None`` when the
            per-well table was used and the mapping is unknown. Resolving the
            restain field by ID avoids depending on ``listChildren()`` ordering
            being stable across two separately-imported plates.
        imputed: Master field indices whose shift was filled in from the well
            mean rather than read directly -- either absent from the table or
            recorded as an exact ``(0, 0)`` empty-frame placeholder.
    """

    shifts: npt.NDArray[np.int_]
    image_ids: list[int] | None
    imputed: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlateAlignment:
    """The alignment table of one master plate."""

    master_plate_id: int
    source: AlignmentSource
    table: pd.DataFrame

    @property
    def member_plate_ids(self) -> tuple[int, ...]:
        """Restain plate IDs, in ascending order. Excludes the master."""
        return tuple(sorted(int(p) for p in self.table["plate"].unique()))

    @property
    def per_field(self) -> bool:
        """True when the per-field table is in use."""
        return self.source == "sample_alignment"

    def wells(self, plate_id: int) -> set[str]:
        """Well positions covered for one restain plate."""
        rows = self.table[self.table["plate"] == plate_id]
        return set(rows["well"].astype(str))

    def shifts_for_well(
        self, plate_id: int, well: str, n_fields: int
    ) -> WellShifts:
        """Resolve per-field shifts for one restain round of one well.

        Args:
            plate_id: The restain plate ID.
            well: Well position, e.g. ``"B8"``.
            n_fields: Number of fields on the *master* well.

        Returns:
            The shifts, indexed by master field index.

        Raises:
            AlignmentError: if the well has no rows for this plate.
        """
        rows = self.table[
            (self.table["plate"] == plate_id)
            & (self.table["well"].astype(str) == well)
        ]
        if rows.empty:
            raise AlignmentError(
                f"Plate {plate_id} has no alignment for well {well}"
            )

        # Rows recorded as exactly (0, 0) are empty-frame placeholders written
        # when a field could not be correlated, not measured zero shifts. They
        # are excluded from the mean and imputed from it below. A genuine zero
        # shift is indistinguishable, but imputing the well mean over it is
        # harmless -- the mean is that same near-zero value.
        measured = rows[(rows["x"] != 0) | (rows["y"] != 0)]
        basis = measured if not measured.empty else rows
        mean_shift = (
            int(np.rint(basis["x"].mean())),
            int(np.rint(basis["y"].mean())),
        )

        if not self.per_field:
            shifts = np.tile(mean_shift, (n_fields, 1)).astype(int)
            return WellShifts(shifts=shifts, image_ids=None)

        # Extract via numpy rather than itertuples: the column dtypes are known
        # here, where an itertuples attribute is untyped.
        by_sample = {
            int(s): (int(x), int(y), int(i))
            for s, x, y, i in zip(
                measured["sample"].to_numpy(dtype=int),
                np.rint(measured["x"].to_numpy(dtype=float)).astype(int),
                np.rint(measured["y"].to_numpy(dtype=float)).astype(int),
                measured["image_id"].to_numpy(dtype=int),
                strict=True,
            )
        }
        ids_by_sample = {
            int(s): int(i)
            for s, i in zip(
                rows["sample"].to_numpy(dtype=int),
                rows["image_id"].to_numpy(dtype=int),
                strict=True,
            )
        }

        shifts = np.zeros((n_fields, 2), dtype=int)
        image_ids: list[int] = []
        imputed: list[int] = []
        for i in range(n_fields):
            if i in by_sample:
                dx, dy, image_id = by_sample[i]
            else:
                dx, dy = mean_shift
                image_id = ids_by_sample.get(i, -1)
                imputed.append(i)
            shifts[i] = (dx, dy)
            image_ids.append(image_id)

        if imputed:
            logger.debug(
                f"Plate {plate_id} well {well}: imputed the well-mean shift "
                f"{mean_shift} for {len(imputed)} field(s) {imputed}"
            )
        return WellShifts(
            shifts=shifts, image_ids=image_ids, imputed=tuple(imputed)
        )


def _exact_attachment(plate: Any, filename: str) -> Any | None:
    """Return the file annotation whose name matches ``filename`` exactly.

    ``get_file_attachments`` matches by suffix, which would let a request for
    ``alignment.csv`` return ``sample_alignment.csv``.
    """
    for ann in plate.listAnnotations():
        if isinstance(ann, FileAnnotationWrapper):
            name = ann.getFile().getName()
            if name and name.lower() == filename:
                return ann
    return None


def has_alignment(plate: Any) -> bool:
    """Cheaply test whether a plate is a 4i master.

    Only annotation metadata is touched -- ``getName()`` does not download the
    file body -- so this is a single ``listAnnotations()`` round trip.
    """
    return _exact_attachment(plate, PER_WELL_FILENAME) is not None or (
        _exact_attachment(plate, PER_FIELD_FILENAME) is not None
    )


def validate_table(
    df: pd.DataFrame, source: AlignmentSource, plate_id: int
) -> None:
    """Check an alignment table has the columns and schema the cache needs.

    Raises:
        AlignmentError: if columns are missing, or a per-well table predates the
            transposition fix.
    """
    required = {"plate", "well", "x", "y"}
    if source == "sample_alignment":
        required |= {"sample", "image_id"}
    missing = required - set(df.columns)
    if missing:
        raise AlignmentError(
            f"Plate {plate_id} {source}.csv is missing columns: "
            f"{sorted(missing)}"
        )
    if source == "alignment":
        schema = df["schema"].min() if "schema" in df.columns else 0
        if schema < MIN_PER_WELL_SCHEMA:
            raise AlignmentError(
                f"Plate {plate_id} alignment.csv is schema {schema}, which "
                "predates the 2026-09-03 fix and has x/y transposed. Re-run "
                "align_plates on this plate, or use sample_alignment.csv."
            )


def load_alignment(
    conn: BlitzGateway,
    master_plate_id: int,
    prefer_per_field: bool = True,
) -> PlateAlignment:
    """Load the alignment table of a 4i master plate.

    Prefers ``sample_alignment.csv``: it is per field (so a round's tiles can be
    placed individually) and it was never affected by the transposition bug.
    Falls back to ``alignment.csv``, which must carry a post-fix ``schema``.

    Args:
        conn: A live connection. Not opened or closed here.
        master_plate_id: The plate carrying the alignment attachments.
        prefer_per_field: Set False to force the per-well table.

    Returns:
        The loaded alignment.

    Raises:
        AlignmentError: if the plate does not exist, carries no usable alignment
            table, or the only table available is a pre-fix per-well one.
    """
    plate = conn.getObject("Plate", master_plate_id)
    if plate is None:
        raise AlignmentError(f"Plate {master_plate_id} not found")

    order: list[tuple[AlignmentSource, str]] = [
        ("sample_alignment", PER_FIELD_FILENAME),
        ("alignment", PER_WELL_FILENAME),
    ]
    if not prefer_per_field:
        order.reverse()

    errors: list[str] = []
    for source, filename in order:
        ann = _exact_attachment(plate, filename)
        if ann is None:
            errors.append(f"{filename}: not attached")
            continue
        df = parse_csv_data(ann)
        if df is None or df.empty:
            errors.append(f"{filename}: empty or unparseable")
            continue
        try:
            validate_table(df, source, master_plate_id)
        except AlignmentError as exc:
            errors.append(f"{filename}: {exc}")
            continue
        logger.info(
            f"Plate {master_plate_id}: using {filename} "
            f"({len(df)} rows, {df['plate'].nunique()} restain plate(s))"
        )
        return PlateAlignment(
            master_plate_id=master_plate_id, source=source, table=df
        )

    raise AlignmentError(
        f"Plate {master_plate_id} has no usable alignment table -- "
        + "; ".join(errors)
    )
