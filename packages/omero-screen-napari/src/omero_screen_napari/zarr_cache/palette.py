"""Distinct display colours for many channels.

A cyclic-IF store can carry 10-20 channels. Assigning colours from a short
repeating list makes two channels identical, and because layers are blended
additively two identically-coloured layers *sum* -- the composite reads as a
single brighter channel, which looks like an intensity artefact rather than a
colour clash. So every channel gets its own hue.

Two roles keep their conventional colours because readers expect them: the
nuclear stain is blue and the cytoplasmic/tubulin stain green. Every other
channel is placed on an evenly spaced hue circle with those two regions
excluded, so nothing collides with them either.

The hex form is written into the store's NGFF ``omero`` block; the napari form
is used at display time. Both come from the same assignment, so a store's
recorded colours and what the viewer shows agree.
"""

from __future__ import annotations

import colorsys
import re
from typing import Any

#: Channel-name aliases resolving to the nuclear role. Mirrors
#: ``_welldata_widget._NUCLEUS_ALIASES``.
NUCLEUS_ALIASES = frozenset({"dapi", "hoechst", "dna", "h2b_rfp"})
#: Channel-name aliases resolving to the cytoplasmic/cell role.
CELL_ALIASES = frozenset({"tub", "gapdh"})

#: Trailing cyclic-IF round qualifier, e.g. the ``_r2`` of ``"DAPI_R2"``.
ROUND_SUFFIX_RE = re.compile(r"_r\d+$")

_NUCLEUS_HEX = "0000FF"
_CELL_HEX = "00FF00"

#: Hue bands (fractions of the circle) reserved for the two role colours, so a
#: generated hue never lands close enough to be confused with them.
#: 2/3 is blue, 1/3 is green.
_RESERVED_HUES = ((2 / 3, 0.08), (1 / 3, 0.08))


def base_name(channel_name: str) -> str:
    """Strip a cyclic-IF round qualifier: ``"DAPI_R2"`` -> ``"dapi"``."""
    return ROUND_SUFFIX_RE.sub("", channel_name.lower())


def _hue_is_reserved(hue: float) -> bool:
    return any(
        abs(((hue - centre) + 0.5) % 1.0 - 0.5) < width
        for centre, width in _RESERVED_HUES
    )


def _spread_hues(count: int) -> list[float]:
    """``count`` hues spread over the circle, avoiding the reserved bands.

    Oversamples the circle and drops reserved hues, so the returned hues stay
    evenly spaced among themselves rather than bunching at the band edges.
    """
    if count <= 0:
        return []
    hues: list[float] = []
    # Oversample so enough survive the reserved-band filter.
    steps = max(count * 3, 24)
    for i in range(steps):
        hue = i / steps
        if not _hue_is_reserved(hue):
            hues.append(hue)
    if not hues:  # pragma: no cover - only if the bands cover the circle
        return [i / count for i in range(count)]
    # Take `count` of them, evenly spaced through the surviving list.
    stride = len(hues) / count
    return [hues[min(int(i * stride), len(hues) - 1)] for i in range(count)]


def _hsv_to_hex(hue: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return f"{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def channel_hex_colors(channel_names: list[str]) -> list[str]:
    """One ``"RRGGBB"`` colour per channel, all visually distinct.

    The first nuclear-role channel is blue and the first cytoplasmic-role
    channel green; the rest are spread over the remaining hue circle. Later
    repeats of a role (a nuclear stain re-imaged in a further round) are treated
    as ordinary channels so they stay distinguishable from the first.

    Args:
        channel_names: Channel names, round-qualified or not.

    Returns:
        Hex colours in the same order, without a leading ``#``.
    """
    if not channel_names:
        return []

    assigned: dict[int, str] = {}
    used_nucleus = used_cell = False
    for index, name in enumerate(channel_names):
        base = base_name(name)
        tokens = {tok for tok in re.split(r"[_-]", base) if tok}
        if not used_nucleus and (
            base in NUCLEUS_ALIASES or base.endswith("_nucleus")
        ):
            assigned[index] = _NUCLEUS_HEX
            used_nucleus = True
        elif not used_cell and (
            tokens & CELL_ALIASES or base.endswith("_cell")
        ):
            assigned[index] = _CELL_HEX
            used_cell = True

    remaining = [i for i in range(len(channel_names)) if i not in assigned]
    for hue, index in zip(
        _spread_hues(len(remaining)), remaining, strict=True
    ):
        assigned[index] = _hsv_to_hex(hue)
    return [assigned[i] for i in range(len(channel_names))]


def channel_colormaps(channel_names: list[str]) -> list[Any]:
    """Napari colormaps matching :func:`channel_hex_colors`.

    Each is a black-to-colour ramp, which is what additive blending needs: the
    zero level must be black or channels brighten each other's background.

    A single channel renders grey, matching the rest of the plugin.
    """
    from napari.utils.colormaps import Colormap

    if len(channel_names) == 1:
        return ["gray"]
    return [
        Colormap(["black", f"#{hex_color}"], name=name)
        for hex_color, name in zip(
            channel_hex_colors(channel_names), channel_names, strict=True
        )
    ]
