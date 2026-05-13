"""Nucleus-channel resolution helpers for CellView import.

CellView's measurement schema historically named all nucleus-segmentation
columns with the literal ``DAPI`` token (``intensity_mean_DAPI_nucleus``,
``integrated_int_DAPI_norm``, etc.). After the omero-screen channel-roles
refactor, upstream feature DataFrames carry the actual fluorophore name
in the column (e.g. ``intensity_mean_Hoechst_nucleus``).

To keep the DB schema stable while remaining biologically truthful, CellView:

1. **On import**, identifies which channel plays the nucleus role and
   records its name on ``repeats.nucleus_channel``. Any ``*_{channel}_*``
   measurement columns are then renamed into the canonical ``*_DAPI_*`` slot
   before INSERT.
2. **On export**, reads ``nucleus_channel`` back and rehydrates the column
   names so downstream notebooks see the actual fluorophore.

This module provides the pure helpers; orchestration lives in
:mod:`cellview.utils.state`.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

import pandas as pd
from rich.prompt import Prompt

from cellview.utils.error_classes import DataError, StateError
from cellview.utils.ui import CellViewUI

if TYPE_CHECKING:
    from omero.gateway import PlateWrapper

_NUCLEUS_COL_RE = re.compile(
    r"^intensity_(?:max|min|mean)_([A-Za-z0-9_]+)_nucleus$"
)


def detect_nucleus_candidates(df: pd.DataFrame) -> list[str]:
    """Return the list of channel names that appear in ``*_nucleus`` columns.

    Order is preserved by first appearance in ``df.columns``. Duplicates are
    removed. Channels are matched against the pattern
    ``intensity_{min|mean|max}_{channel}_nucleus``.

    Args:
        df: Single-cell DataFrame, typically loaded from a CellView CSV.

    Returns:
        Channel names appearing in nucleus-segmentation columns.
    """
    seen: list[str] = []
    for col in df.columns:
        if match := _NUCLEUS_COL_RE.match(col):
            channel = match.group(1)
            if channel not in seen:
                seen.append(channel)
    return seen


def resolve_nucleus_channel_from_plate(plate: PlateWrapper) -> str:
    """Read OMERO map annotations from ``plate`` and resolve the nucleus channel.

    Looks up the plate's channel map annotation, runs
    :func:`omero_screen.metadata_parser.resolve_channel_roles` over the keys,
    and returns the channel name that resolves to the ``nucleus`` role.

    Args:
        plate: OMERO :class:`PlateWrapper` for the plate being imported.

    Returns:
        The actual fluorophore name (e.g. ``"DAPI"``, ``"Hoechst"``,
        ``"H2B_RFP"``) annotated as the nucleus channel for this plate.

    Raises:
        DataError: If the plate has no channel annotation, or no channel
            resolves to a nucleus role. The error message points the user at
            re-annotating the plate or passing ``--nucleus-channel``.
    """
    # Imported here to avoid pulling omero-screen into module import time for
    # the CSV-only code path.
    from omero_utils.map_anns import parse_annotations
    from omero_utils.message import ChannelAnnotationError

    from omero_screen.constants import OmeroScreenNS
    from omero_screen.metadata_parser import (
        resolve_channel_roles,
        strip_role_suffix,
    )

    plate_id = plate.getId()
    channel_data = parse_annotations(plate, ns=OmeroScreenNS.METADATA)
    if not channel_data:
        # Fall back to any map annotation if no metadata-namespaced one exists,
        # to accommodate plates annotated outside the omero-screen pipeline.
        channel_data = parse_annotations(plate)
    if not channel_data:
        raise DataError(
            "Plate has no channel map annotations — cannot determine the "
            "nucleus channel. Re-annotate the plate (key-value pairs of "
            "channel name → index) or pass --nucleus-channel CH.",
            context={"plate_id": plate_id},
        )

    try:
        roles = resolve_channel_roles(dict.fromkeys(channel_data, 0))
    except ChannelAnnotationError as exc:
        raise DataError(
            f"Plate channel annotation {dict(channel_data)} does not resolve "
            "to a nucleus role. Re-annotate the plate or pass "
            "--nucleus-channel CH.",
            context={"plate_id": plate_id, "channels": list(channel_data)},
        ) from exc

    # Strip the optional ``_nucleus`` suffix from the channel name — the
    # upstream pipeline (omero_screen.image_analysis) strips the same suffix
    # when building feature column tokens, so the column name in the CSV is
    # ``intensity_mean_SirDNA_nucleus`` rather than
    # ``intensity_mean_SirDNA_nucleus_nucleus``. We mirror that here so the
    # downstream lookup matches the actual column.
    return strip_role_suffix(roles["nucleus"])


def prompt_nucleus_channel(
    candidates: list[str],
    *,
    cli_flag: str | None = None,
    ui: CellViewUI | None = None,
) -> str:
    """Resolve the nucleus channel from CLI flag, interactive prompt, or fail.

    Resolution order:

    1. If ``cli_flag`` is given, validate it against ``candidates`` and return.
    2. If running interactively (``sys.stdin.isatty()``), prompt the user with
       ``candidates`` as choices. A single candidate is used as the default.
    3. Otherwise raise :class:`StateError` — the caller needs to pass the flag.

    Args:
        candidates: Channel names detected in the CSV's ``*_nucleus`` columns.
            Must be non-empty.
        cli_flag: Value of ``--nucleus-channel`` if the user passed it.
        ui: Optional UI for emitting notifications. Created if ``None``.

    Returns:
        The chosen nucleus channel name. Guaranteed to be a member of
        ``candidates``.

    Raises:
        StateError: If ``candidates`` is empty, the CLI flag does not match
            a discovered candidate, or no flag was given in a non-interactive
            environment.
    """
    ui = ui or CellViewUI()

    if not candidates:
        raise StateError(
            "CSV contains no nucleus-segmentation columns "
            "(`intensity_*_<channel>_nucleus`). Cannot determine the nucleus "
            "channel — is this the right file?"
        )

    if cli_flag is not None:
        if cli_flag not in candidates:
            raise StateError(
                f"--nucleus-channel={cli_flag!r} not found in CSV. "
                f"Discovered nucleus channels: {candidates}.",
                context={"cli_flag": cli_flag, "candidates": candidates},
            )
        return cli_flag

    if not sys.stdin.isatty():
        raise StateError(
            "Nucleus channel could not be determined automatically and the "
            "session is non-interactive. Pass --nucleus-channel CH to specify "
            "(one of: " + ", ".join(candidates) + ").",
            context={"candidates": candidates},
        )

    if len(candidates) == 1:
        return str(
            Prompt.ask(
                "Which channel is the nucleus channel?",
                choices=candidates,
                default=candidates[0],
            )
        )
    return str(
        Prompt.ask(
            "Which channel is the nucleus channel?",
            choices=candidates,
        )
    )


def validate_nucleus_channel_in_df(channel: str, df: pd.DataFrame) -> None:
    """Verify that ``intensity_mean_{channel}_nucleus`` exists in ``df``.

    Args:
        channel: Resolved nucleus channel name.
        df: DataFrame whose columns are being validated.

    Raises:
        DataError: If the canonical nucleus-intensity column for ``channel``
            is absent from the DataFrame. The message includes the discovered
            nucleus channels so the user can spot a mismatch between annotation
            and CSV.
    """
    expected = f"intensity_mean_{channel}_nucleus"
    if expected not in df.columns:
        discovered = detect_nucleus_candidates(df)
        raise DataError(
            f"Expected column {expected!r} not found in the CSV. The plate "
            f"reports nucleus channel {channel!r} but the CSV's nucleus "
            f"channels are {discovered}. Re-run the pipeline or pass "
            "--nucleus-channel CH to override.",
            context={"channel": channel, "discovered": discovered},
        )


def rehydrate_dapi_to_nucleus(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Rename canonical ``*_DAPI_*`` columns back to ``*_{channel}_*`` on export.

    Inverse of :func:`rename_nucleus_to_dapi`. The DB stores nucleus-channel
    measurements in the canonical DAPI slot regardless of fluorophore; this
    helper restores the actual fluorophore name for downstream notebooks.

    No-op when ``channel == "DAPI"`` (legacy plates already match).

    Args:
        df: DataFrame as returned by the exporter, with ``*_DAPI_*`` columns.
        channel: Actual fluorophore name to rehydrate (``repeats.nucleus_channel``).

    Returns:
        A copy of ``df`` with the nucleus columns renamed to use ``channel``.
        If a target column name already exists in ``df`` the rename is skipped
        for that pair (defensive — should not happen with healthy data).
    """
    if channel == "DAPI":
        return df

    rename_map: dict[str, str] = {}
    for col in df.columns:
        new_col: str | None = None
        for stat in ("min", "mean", "max"):
            for segment in ("nucleus", "cell", "cyto"):
                src = f"intensity_{stat}_DAPI_{segment}"
                if col == src:
                    new_col = f"intensity_{stat}_{channel}_{segment}"
                    break
            if new_col is not None:
                break
        if new_col is None and col == "integrated_int_DAPI":
            new_col = f"integrated_int_{channel}"
        if new_col is None and col == "integrated_int_DAPI_norm":
            new_col = f"integrated_int_{channel}_norm"
        if new_col is not None and new_col not in df.columns:
            rename_map[col] = new_col

    return df.rename(columns=rename_map)


def rename_nucleus_to_dapi(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Rename ``*_{channel}_*`` measurement columns into the ``*_DAPI_*`` slot.

    No-op when ``channel == "DAPI"`` (legacy plates).

    Renames columns matching the pattern ``intensity_{min|mean|max}_{channel}_{segment}``
    and ``integrated_int_{channel}_norm`` only, leaving any other channel
    columns intact.

    Args:
        df: DataFrame whose columns will be renamed (a copy is returned).
        channel: The actual nucleus channel name.

    Returns:
        A copy of ``df`` with nucleus columns renamed into the canonical DAPI
        slot.

    Raises:
        DataError: If renaming would clobber a pre-existing ``*_DAPI_*``
            column — i.e. the CSV contains both ``DAPI`` and ``{channel}``
            nucleus columns, which is ambiguous.
    """
    if channel == "DAPI":
        return df

    rename_map: dict[str, str] = {}
    for col in df.columns:
        new_col: str | None = None
        for stat in ("min", "mean", "max"):
            for segment in ("nucleus", "cell", "cyto"):
                src = f"intensity_{stat}_{channel}_{segment}"
                if col == src:
                    new_col = f"intensity_{stat}_DAPI_{segment}"
                    break
            if new_col is not None:
                break
        if new_col is None and col == f"integrated_int_{channel}":
            new_col = "integrated_int_DAPI"
        if new_col is None and col == f"integrated_int_{channel}_norm":
            new_col = "integrated_int_DAPI_norm"
        if new_col is not None:
            rename_map[col] = new_col

    clobbers = [tgt for tgt in rename_map.values() if tgt in df.columns]
    if clobbers:
        raise DataError(
            f"Cannot rename nucleus channel {channel!r} to canonical DAPI "
            f"slot: the CSV already contains columns {clobbers}. The file "
            "appears to mix DAPI and non-DAPI nucleus data.",
            context={"channel": channel, "clobbers": clobbers},
        )

    return df.rename(columns=rename_map)
