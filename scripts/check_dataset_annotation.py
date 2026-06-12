#!/usr/bin/env python
"""Inspect (and optionally delete) the 'Dataset' map annotation on a plate.

Background
----------
The core pipeline links a plate to its analysis :class:`Dataset` by writing a
map annotation on the plate under the namespace ``omero-screen/dataset`` with
the key ``Dataset`` -> dataset id (see
:class:`omero_screen.plate_dataset.PlateDataset`). If a user deletes that
dataset to re-run the screen, the stale annotation is left behind. On the next
run ``PlateDataset`` reads the annotation, returns the now-dangling id, and the
pipeline fails downstream when ``conn.getObject("Dataset", id)`` returns
``None``.

The intended fix is to validate the dataset on lookup and, if it is gone, drop
the stale annotation before recreating the dataset. That only works if we can
actually delete the map annotation via the OMERO API -- which is uncertain when
the annotation was written by a *different* user than the one re-running the
screen (OMERO permissions / group permission levels).

This script answers that question empirically. By default it only inspects;
pass ``--delete`` to actually attempt the deletion and observe whether the
server permits it.

Usage
-----
    uv run python scripts/check_dataset_annotation.py <plate_id>
    uv run python scripts/check_dataset_annotation.py <plate_id> --delete

The OMERO connection uses the standard env credentials (USERNAME / PASSWORD /
HOST), selected by the ENV variable as elsewhere in the pipeline.
"""

import argparse

from omero.gateway import BlitzGateway, MapAnnotationWrapper
from omero_utils.omero_connect import omero_connect

# Importing the package triggers set_env_vars() which loads .env.{ENV}.
import omero_screen  # noqa: F401
from omero_screen.constants import OmeroScreenNS


def _describe_owner(obj: object) -> str:
    """Return a 'name (id=N)' string for an OMERO object's owner, if available."""
    try:
        details = obj.getDetails()  # type: ignore[attr-defined]
        owner = details.getOwner()
        return f"{owner.getOmeName()} (id={owner.getId()})"
    except Exception:  # pragma: no cover - best-effort diagnostics
        return "<unknown>"


@omero_connect
def inspect_dataset_annotation(
    plate_id: int,
    delete: bool = False,
    conn: BlitzGateway | None = None,
) -> None:
    """Report the Dataset map annotation(s) on a plate and optionally delete them.

    Args:
        plate_id: OMERO plate ID to inspect.
        delete: If True, attempt to delete each Dataset map annotation found.
        conn: OMERO connection (injected by the decorator).
    """
    assert conn is not None

    me = conn.getUser()
    print(f"Connected as: {me.getOmeName()} (id={me.getId()})")
    print(
        f"Group: {conn.getGroupFromContext().getName()} "
        f"(perms={conn.getGroupFromContext().getDetails().getPermissions()})"
    )
    print("-" * 70)

    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        print(f"Plate {plate_id} not found.")
        return
    print(
        f"Plate {plate_id}: {plate.getName()}  owner={_describe_owner(plate)}"
    )

    # Collect the Dataset-namespace map annotations directly so we have the
    # wrapper objects (and their ids) rather than just the parsed key/values.
    map_anns = [
        ann
        for ann in plate.listAnnotations(ns=OmeroScreenNS.DATASET)
        if isinstance(ann, MapAnnotationWrapper)
    ]

    if not map_anns:
        print(
            f"\nNo map annotations under namespace '{OmeroScreenNS.DATASET}'. "
            "Nothing to do."
        )
        return

    print(f"\nFound {len(map_anns)} Dataset map annotation(s):")
    for ann in map_anns:
        values = dict(ann.getValue())
        print(f"\n  Annotation id={ann.getId()}  owner={_describe_owner(ann)}")
        print(f"    namespace: {ann.getNs()}")
        print(f"    values:    {values}")
        print(f"    canEdit={ann.canEdit()}  canDelete={ann.canDelete()}")

        dataset_id = values.get("Dataset")
        if dataset_id:
            dataset = conn.getObject("Dataset", int(dataset_id))
            if dataset is None:
                print(
                    f"    -> Dataset {dataset_id} DOES NOT EXIST "
                    "(stale annotation -- this is the bug case)"
                )
            else:
                n_children = len(list(dataset.listChildren()))
                print(
                    f"    -> Dataset {dataset_id} exists: "
                    f"'{dataset.getName()}' owner={_describe_owner(dataset)} "
                    f"({n_children} images)"
                )

    if not delete:
        print("\n(Inspection only. Re-run with --delete to attempt deletion.)")
        return

    print("\n" + "-" * 70)
    print("Attempting deletion via conn.deleteObjects(...) ...")
    ann_ids = [ann.getId() for ann in map_anns]
    try:
        # wait=True so the server-side delete completes and surfaces any
        # permission error here rather than silently in the background.
        conn.deleteObjects("Annotation", ann_ids, wait=True, deleteAnns=True)
        # Verify by re-reading.
        remaining = [
            ann
            for ann in conn.getObject("Plate", plate_id).listAnnotations(
                ns=OmeroScreenNS.DATASET
            )
            if isinstance(ann, MapAnnotationWrapper)
        ]
        if remaining:
            print(
                f"DELETE reported success but {len(remaining)} annotation(s) "
                "remain -- check permissions."
            )
        else:
            print(f"SUCCESS: deleted annotation(s) {ann_ids}.")
    except Exception as exc:
        print(f"FAILED to delete annotation(s) {ann_ids}: {exc!r}")


def main() -> None:
    """Parse arguments and run the inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plate_id", type=int, help="OMERO plate ID to inspect."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Attempt to delete the Dataset map annotation(s) found.",
    )
    args = parser.parse_args()
    inspect_dataset_annotation(args.plate_id, delete=args.delete)


if __name__ == "__main__":
    main()
