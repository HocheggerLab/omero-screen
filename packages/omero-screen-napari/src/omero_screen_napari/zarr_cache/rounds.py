"""Resolving a plate into its cyclic-IF (4i) round group.

A 4i experiment is one dataset spread over several OMERO plates. Given any plate
ID this module answers: is it a master, which restain rounds belong to it, and
can the group actually be cached as one pre-aligned store?

Shared by the plate-info dialog (which shows the badge and offers the choice) and
the cache builder (which does the work), so both agree on what a group is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from omero.gateway import BlitzGateway

from omero_screen_napari.zarr_cache.alignment import (
    AlignmentError,
    PlateAlignment,
    has_alignment,
    load_alignment,
)


@dataclass(frozen=True)
class RoundGroup:
    """A master plate and the restain rounds aligned to it.

    Attributes:
        master_plate_id: The plate carrying the alignment tables.
        member_plate_ids: Restain plate IDs, ascending. Empty when the plate is
            not a 4i master.
        alignment: The loaded alignment, or None when not a master.
        blockers: Human-readable reasons the group cannot be built as one
            pre-aligned store. Empty means buildable.
    """

    master_plate_id: int
    member_plate_ids: tuple[int, ...] = ()
    alignment: PlateAlignment | None = None
    blockers: tuple[str, ...] = field(default=())

    @property
    def is_master(self) -> bool:
        """True when this plate has restain rounds aligned to it."""
        return bool(self.member_plate_ids)

    @property
    def buildable(self) -> bool:
        """True when the whole group can be cached as one store."""
        return self.is_master and not self.blockers

    @property
    def plate_ids(self) -> tuple[int, ...]:
        """Master first, then restain rounds -- the round order of channels."""
        return (self.master_plate_id, *self.member_plate_ids)

    @property
    def n_rounds(self) -> int:
        """Number of rounds, counting the master as round 1."""
        return len(self.plate_ids)


def resolve_round_group(
    conn: BlitzGateway, plate_id: int, *, check_stitched: bool = True
) -> RoundGroup:
    """Resolve a plate into its 4i round group.

    Never raises for an ordinary plate -- a plate with no alignment attachment
    simply comes back with no members. Reasons a *master* cannot be built are
    collected in ``blockers`` rather than raised, so the dialog can show the
    badge and explain why the option is unavailable.

    Args:
        conn: A live connection. Not opened or closed here.
        plate_id: Any plate ID; only a master yields members.
        check_stitched: Verify every round is stitched. The builder composes
            rounds from per-field canvas offsets, which only exist for a
            ``--stitch`` run. Pass False to skip the extra OMERO queries when
            the caller only needs the badge.

    Returns:
        The resolved group.
    """
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        return RoundGroup(plate_id, blockers=("plate not found",))
    if not has_alignment(plate):
        return RoundGroup(plate_id)

    try:
        alignment = load_alignment(conn, plate_id)
    except AlignmentError as exc:
        # A master whose only table is a pre-schema per-well file lands here.
        # It is still a master -- the badge should show -- but not buildable.
        logger.info(f"Plate {plate_id} has unusable alignment data: {exc}")
        return RoundGroup(plate_id, blockers=(str(exc),))

    members = alignment.member_plate_ids
    blockers: list[str] = []
    if not members:
        blockers.append("alignment table lists no restain plates")

    if check_stitched:
        blockers.extend(_stitching_blockers(conn, plate_id, members))

    return RoundGroup(
        master_plate_id=plate_id,
        member_plate_ids=members,
        alignment=alignment,
        blockers=tuple(blockers),
    )


def _stitching_blockers(
    conn: BlitzGateway, master_plate_id: int, members: tuple[int, ...]
) -> list[str]:
    """Report rounds that are not stitched.

    The combined store is built by placing each round's fields onto the master's
    stitched canvas, which needs the per-well ``canvas.csv`` a ``--stitch`` run
    writes. An unstitched round cannot take part.
    """
    # Imported here rather than at module scope: this module is imported from
    # the zarr_cache package __init__, which also pulls in the builder.
    #
    # ``detect_label_stitched_mode`` takes a raw gateway, unlike
    # ``builder.is_stitched_plate`` which wants an OmeroConnection it can open
    # sub-connections from. Callers here already hold a live gateway and must
    # not open a second one.
    from omero_screen_napari.plate_cache import detect_label_stitched_mode

    blockers: list[str] = []
    for pid in (master_plate_id, *members):
        try:
            if not detect_label_stitched_mode(conn, pid):
                role = "master" if pid == master_plate_id else "restain round"
                blockers.append(f"{role} plate {pid} is not stitched")
        except (ValueError, KeyError) as exc:
            blockers.append(f"plate {pid} could not be checked: {exc}")
    return blockers


def build_channel_plan(
    group: RoundGroup,
    channel_data_by_plate: dict[int, dict[str, str]],
) -> tuple[list[str], dict[str, Any]]:
    """Flatten every round's channels into one list plus a ``rounds`` descriptor.

    Channels are concatenated in round order (master first) and every name is
    suffixed ``_R{n}`` from round 1. Suffixing the master too keeps
    channel-to-round derivable from the name alone; more importantly the names
    must be unique, because ``display._populate_singleton`` builds
    ``{name: index}`` and duplicates would silently collapse it, breaking
    gallery and classifier channel lookups.

    A channel whose base name already appeared in an earlier round is marked
    redundant -- in practice the nuclear stain, which is re-imaged every round
    because it is the registration channel. It is kept rather than dropped: it
    is the only in-store way to re-verify the alignment, and dropping it would
    make channel index to round non-uniform. Consumers default redundant layers
    to hidden.

    Args:
        group: The resolved round group.
        channel_data_by_plate: ``{plate_id: {channel_name: index_string}}`` as
            returned by ``plate_cache.get_plate_metadata``.

    Returns:
        ``(channel_names, rounds_attrs)`` ready for ``PlateZarrWriter``.

    Raises:
        KeyError: if a round has no channel data.
    """
    channel_names: list[str] = []
    entries: list[dict[str, Any]] = []
    seen_base: set[str] = set()

    for round_index, plate_id in enumerate(group.plate_ids, start=1):
        channel_data = channel_data_by_plate[plate_id]
        # Order by the channel's index on its own plate, not dict order.
        ordered = sorted(channel_data.items(), key=lambda kv: int(kv[1]))
        for position, (base_name, _) in enumerate(ordered):
            qualified = f"{base_name}_R{round_index}"
            entries.append(
                {
                    "index": len(channel_names),
                    "plate_id": plate_id,
                    "round": round_index,
                    "name": base_name,
                    "position": position,
                    "redundant": base_name in seen_base,
                }
            )
            channel_names.append(qualified)
        seen_base.update(name for name, _ in ordered)

    alignment = group.alignment
    rounds_attrs: dict[str, Any] = {
        "master_plate_id": group.master_plate_id,
        "member_plate_ids": list(group.member_plate_ids),
        "plate_ids": list(group.plate_ids),
        "alignment_source": alignment.source if alignment else None,
        # Recorded so a reader never has to guess which way the shift goes.
        "shift_convention": "master = restain - (x, y)",
        "channels": entries,
    }
    return channel_names, rounds_attrs


def channel_indices_for_plate(
    rounds_attrs: dict[str, Any], plate_id: int
) -> list[int]:
    """Flat channel indices contributed by one plate of a 4i store.

    Lets a consumer holding a restain plate ID pull just that round's channels
    out of the combined array.
    """
    return [
        int(entry["index"])
        for entry in rounds_attrs.get("channels", [])
        if int(entry["plate_id"]) == plate_id
    ]
