"""Channel-name conventions shared by the pipeline and the tracking tools.

A channel's role is carried in its name: a trailing ``_nucleus`` or ``_cell``
marks the channel used to segment that compartment, and a PIP-FUCCI plate is
recognised by channels named exactly ``pip`` and ``geminin``. These rules live
here, below both the pipeline (``omero_screen``) and the tracking code, so the
tracking code can apply them without importing the pipeline (#75).
``omero_screen.metadata_parser`` re-exports them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from loguru import logger

from omero_utils.message import ChannelAnnotationError

ROLE_SUFFIXES: dict[str, str] = {"_nucleus": "nucleus", "_cell": "cell"}
# PIP-FUCCI (Grant et al. 2018) is recognised by these exact channel names
# only — whole name, case-insensitive, after any role suffix — and only on a
# timelapse. Anything else (``Geminin_488``, ``Cdt1``, a fixed IF plate) is
# not FUCCI, so geminin-gated repair and phase calls never touch it.
PIP_FUCCI_PIP = "pip"
PIP_FUCCI_GEMININ = "geminin"


def strip_role_suffix(channel_name: str) -> str:
    """Return the channel name with any trailing `_nucleus` / `_cell` role suffix removed.

    Matching is case-insensitive; the rest of the name is preserved verbatim.
    Used to build feature column names without introducing duplicated suffix tokens
    (e.g. avoids ``intensity_mean_CellMask_cell_cell``).
    """
    lowered = channel_name.lower()
    for suffix in ROLE_SUFFIXES:
        if lowered.endswith(suffix):
            return channel_name[: -len(suffix)]
    return channel_name


@dataclass(frozen=True)
class PipFucci:
    """The two PIP-FUCCI reporter channels of a plate, named as on the plate."""

    pip: str
    geminin: str


def resolve_pip_fucci(
    channel_names: Iterable[str], n_timepoints: int
) -> PipFucci | None:
    """The plate's PIP-FUCCI channels, or ``None`` if it is not PIP-FUCCI.

    A plate is PIP-FUCCI if and only if it is a timelapse and has one channel
    named ``pip`` and one named ``geminin`` (see :data:`PIP_FUCCI_PIP`). The
    decision is logged either way.

    Args:
        channel_names: Channel names as on the plate (or CellView columns).
        n_timepoints: Number of timepoints; a single timepoint is never FUCCI.

    Raises:
        ChannelAnnotationError: If two channels resolve to the same reporter.
    """
    found: dict[str, list[str]] = {PIP_FUCCI_PIP: [], PIP_FUCCI_GEMININ: []}
    for name in channel_names:
        key = strip_role_suffix(name).lower()
        if key in found:
            found[key].append(name)
    for key, names in found.items():
        if len(names) > 1:
            raise ChannelAnnotationError(
                f"Channels {names} all resolve to the PIP-FUCCI '{key}' reporter.",
                logger,
            )
    pip, gem = found[PIP_FUCCI_PIP], found[PIP_FUCCI_GEMININ]
    if not (pip and gem):
        logger.info(
            "Not PIP-FUCCI: needs channels named exactly 'pip' and 'geminin'."
        )
        return None
    if n_timepoints < 2:
        logger.info("Not PIP-FUCCI: a single timepoint.")
        return None
    logger.info(f"PIP-FUCCI: pip={pip[0]!r}, geminin={gem[0]!r}.")
    return PipFucci(pip=pip[0], geminin=gem[0])
