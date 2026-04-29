"""Module for handling map annotations in OMERO.

This module provides functions for parsing, deleting, and adding map annotations to OMERO objects.

Available functions:

- parse_annotations(omero_object): Parse the key value pair annotations from any OMERO object.
- delete_map_annotations(conn, omero_object): Delete all map annotations from an OMERO object.
- delete_map_annotation(conn, omero_object, key): Remove the map annotation from an OMERO object.
- add_map_annotations(conn, omero_object, map_annotations): Add map annotations to an OMERO object.

"""

from typing import Any

from omero.gateway import (
    BlitzGateway,
    BlitzObjectWrapper,
    MapAnnotationWrapper,
)


def parse_annotations(
    omero_object: BlitzObjectWrapper, ns: str | None = None
) -> dict[str, str]:
    """Parse the key value pair annotations from any OMERO object.

    Args:
        omero_object: Any OMERO object (Plate, Well, Image, etc.)
        ns: Optional namespace to filter annotations by. If None, all map annotations are returned.

    Returns:
        Dictionary of key-value pairs from map annotations
    """
    annotations = omero_object.listAnnotations(ns=ns)
    map_anns = [
        ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
    ]
    return {k: v for ann in map_anns for k, v in ann.getValue()}


def delete_map_annotations(
    conn: BlitzGateway, omero_object: BlitzObjectWrapper, ns: str | None = None
) -> None:
    """Delete all map annotations from an OMERO object.

    Args:
        conn: OMERO connection
        omero_object: Any OMERO object (Plate, Well, Image, etc.)
        ns: Optional namespace to filter annotations by. If None, all map annotations are deleted.

    """
    annotations = omero_object.listAnnotations(ns=ns)
    for ann in annotations:
        if isinstance(ann, MapAnnotationWrapper):
            conn.deleteObject(ann._obj)


def delete_map_annotation(
    conn: BlitzGateway,
    omero_object: BlitzObjectWrapper,
    key: str,
    ns: str | None = None,
) -> None:
    """Remove the map annotation from an OMERO object.

    Args:
        conn: OMERO connection
        omero_object: Any OMERO object (Plate, Well, Image, etc.)
        key: Key to identify annotation
        ns: Optional namespace to filter annotations by. If None, all map annotations are searched.

    """
    # Get the existing map annotations of the image
    annotations = omero_object.listAnnotations(ns=ns)
    if map_anns := [
        ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
    ]:
        for ann in map_anns:
            if key in dict(ann.getValue()):
                conn.deleteObject(ann._obj)  # Delete the annotation


def add_map_annotations(
    conn: BlitzGateway,
    omero_object: BlitzObjectWrapper,
    map_annotations: dict[str, Any],
    ns: str | None = None,
) -> None:
    """Add map annotations to an OMERO object.

    Args:
        conn: OMERO connection
        omero_object: Any OMERO object (Plate, Well, Image, etc.)
        map_annotations: Dictionary of key-value pairs
        ns: Optional namespace.

    """
    key_value_data = [[str(k), str(v)] for k, v in map_annotations.items()]
    ann = MapAnnotationWrapper(conn)
    ann.setValue(key_value_data)
    if ns is not None:
        ann.setNs(ns)
    ann.save()
    omero_object.linkAnnotation(ann)
