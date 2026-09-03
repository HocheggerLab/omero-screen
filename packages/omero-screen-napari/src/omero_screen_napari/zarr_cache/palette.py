"""Display colours for many channels.

A cyclic-IF store can carry 10-20 channels. Assigning colours from a short
repeating list makes two channels identical, and because layers are blended
additively two identically-coloured layers *sum* -- the composite reads as a
single brighter channel, which looks like an intensity artefact rather than a
colour clash. So every channel gets its own colour.

The sequence is the curated one from ``_welldata_widget._generate_color_map``
(the direct-from-OMERO path used by the aligned-plate widget), kept here as the
single source of truth so the zarr path and the direct path look the same.
Roles come first -- nuclear blue, cytoplasmic green, EdU grey -- then the
curated sequence, then generated hues if a plate has more channels than the
sequence holds.

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

NUCLEUS_HEX = "0000FF"
CELL_HEX = "00FF00"
EDU_HEX = "808080"

#: Curated colour sequence, in assignment order. This is
#: ``_welldata_widget._generate_color_map``'s ``remaining_colors`` list read in
#: the order it pops from (it pops off the end), with the named napari
#: colormaps resolved to their hex equivalents. Ordered so the most
#: distinguishable colours are used first and the subtler ones only appear on
#: plates with many channels.
CURATED_HEX: tuple[str, ...] = (
    "FF0000",  # red
    "FFFF00",  # yellow
    "FF00FF",  # magenta
    "00FFFF",  # cyan
    "FFA500",  # orange
    "9D00FF",  # bop purple
    "0080FF",  # bop blue
    "FF8000",  # bop orange
    "FFC0CB",  # pink
    "AA7942",  # brown
    "009193",  # teal
    "8EFA00",  # lime
    "FF2F92",  # strawberry
)

#: Hue bands (fractions of the circle) reserved for the role colours, so an
#: overflow hue never lands close enough to be confused with them. 2/3 is blue,
#: 1/3 is green.
_RESERVED_HUES = ((2 / 3, 0.06), (1 / 3, 0.06))


def base_name(channel_name: str) -> str:
    """Strip a cyclic-IF round qualifier: ``"DAPI_R2"`` -> ``"dapi"``."""
    return ROUND_SUFFIX_RE.sub("", channel_name.lower())


def _hue_is_reserved(hue: float) -> bool:
    return any(
        abs(((hue - centre) + 0.5) % 1.0 - 0.5) < width
        for centre, width in _RESERVED_HUES
    )


def _hsv_to_hex(hue: float, value: float = 1.0) -> str:
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, value)
    return f"{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def _overflow_hex(count: int, used: set[str]) -> list[str]:
    """Extra distinct colours for plates with more channels than the sequence.

    Walks the hue circle at a golden-angle step, which spreads successive
    colours far apart without needing to know the total in advance, skipping the
    reserved role bands and anything already assigned.
    """
    out: list[str] = []
    hue = 0.0
    guard = 0
    while len(out) < count and guard < 10_000:
        guard += 1
        hue = (hue + 0.61803398875) % 1.0
        if _hue_is_reserved(hue):
            continue
        # Alternate full and dimmed value so a second pass round the circle
        # still differs from the first.
        candidate = _hsv_to_hex(hue, 1.0 if len(out) % 2 == 0 else 0.65)
        if candidate in used or candidate in out:
            continue
        out.append(candidate)
    return out


def _role_hex(channel_name: str, used_roles: set[str]) -> str | None:
    """Conventional colour for a channel's role, if it is the first of that role."""
    base = base_name(channel_name)
    tokens = {tok for tok in re.split(r"[_-]", base) if tok}
    if "nucleus" not in used_roles and (
        base in NUCLEUS_ALIASES or base.endswith("_nucleus")
    ):
        used_roles.add("nucleus")
        return NUCLEUS_HEX
    if "cell" not in used_roles and (
        tokens & CELL_ALIASES or base.endswith("_cell")
    ):
        used_roles.add("cell")
        return CELL_HEX
    if "edu" not in used_roles and base == "edu":
        used_roles.add("edu")
        return EDU_HEX
    return None


def channel_hex_colors(channel_names: list[str]) -> list[str]:
    """One ``"RRGGBB"`` colour per channel, all distinct.

    The first nuclear-role channel is blue, the first cytoplasmic-role channel
    green and the first EdU channel grey; the rest take :data:`CURATED_HEX` in
    order, then generated hues. A later repeat of a role is treated as an
    ordinary channel so it stays distinguishable from the first.

    Args:
        channel_names: Channel names, round-qualified or not.

    Returns:
        Hex colours in the same order, without a leading ``#``.
    """
    if not channel_names:
        return []

    assigned: dict[int, str] = {}
    used_roles: set[str] = set()
    for index, name in enumerate(channel_names):
        if (role_hex := _role_hex(name, used_roles)) is not None:
            assigned[index] = role_hex

    remaining = [i for i in range(len(channel_names)) if i not in assigned]
    used = set(assigned.values())
    sequence = [hex_ for hex_ in CURATED_HEX if hex_ not in used]
    if len(sequence) < len(remaining):
        sequence += _overflow_hex(
            len(remaining) - len(sequence), used | set(sequence)
        )
    for index, hex_color in zip(remaining, sequence, strict=False):
        assigned[index] = hex_color
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
