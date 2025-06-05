#!/usr/bin/env python3

"""Get information about an OMERO object."""

import argparse


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Get information about an OMERO object."
    )
    parser.add_argument(
        "ID",
        nargs="+",
        type=str,
        help="OMERO object ID (e.g. Image:123, Plate:456",
    )
    parser.add_argument(
        "--parents",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Show parent ancestry",
    )
    group = parser.add_argument_group("Omero Screen overrides")
    group.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (requires configuration file .env.{name}).",
    )
    args = parser.parse_args()

    import os

    if args.env:
        os.environ["ENV"] = args.env

    import json

    from omero.gateway import BlitzGateway
    from omero_utils.omero_connect import omero_connect

    @omero_connect
    def get_info(
        ids: list[str], parents: bool, conn: BlitzGateway | None = None
    ) -> None:
        assert conn is not None
        for object_id in ids:
            obj_type, oid = object_id.split(":")
            obj = conn.getObject(obj_type, oid)
            if obj is None:
                print("ERROR: Missing", object_id)
                continue
            print(
                f"""{object_id}: owner: {obj.getOwnerOmeName()} - {obj.getOwnerFullName()}"""
            )
            try:
                print(json.dumps(obj.simpleMarshal(parents=parents), indent=4))
            except Exception:  # noqa: BLE001
                # Something is missing. Do a simple version with members of BlitzObjectWrapper.
                ancestry = obj.getAncestry() if parents else []
                ancestry.insert(0, obj)
                for a in ancestry:
                    print(f"""    type: {a.OMERO_CLASS}
    id: {a.getId()}
    name: {a.getName()},
    description: {a.getDescription()}
""")

    get_info(args.ID, args.parents)


if __name__ == "__main__":
    _main()
