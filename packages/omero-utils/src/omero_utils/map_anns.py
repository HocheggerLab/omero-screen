from typing import Any

from omero.gateway import (
    BlitzGateway,
    BlitzObjectWrapper,
    MapAnnotationWrapper,
)


def parse_annotations(omero_object: BlitzObjectWrapper) -> dict[str, str]:
    """Parse the key value pair annotations from any OMERO object.

    Args:
        omero_object: Any OMERO object (Plate, Well, Image, etc.)

    Returns:
        Dictionary of key-value pairs from map annotations
    """
    annotations = omero_object.listAnnotations()
    map_annotations = [
        ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
    ]
    return {k: v for ann in map_annotations for k, v in ann.getValue()}


def delete_map_annotations(
    conn: BlitzGateway, omero_object: BlitzObjectWrapper
) -> None:
    """Delete all map annotations from an OMERO object.

    Args:
        omero_object: Any OMERO object (Plate, Well, Image, etc.)
    """
    annotations = omero_object.listAnnotations()
    for ann in annotations:
        if isinstance(ann, MapAnnotationWrapper):
            conn.deleteObject(ann._obj)


def add_map_annotations(
    conn: BlitzGateway,
    omero_object: BlitzObjectWrapper,
    map_annotations: dict[str, Any],
) -> None:
    """Add map annotations to an OMERO object.

    Args:
        omero_object: Any OMERO object (Plate, Well, Image, etc.)
        map_annotations: Dictionary of key-value pairs
    """
    for key, value in map_annotations.items():
        ann = MapAnnotationWrapper(conn)
        ann.setValue([(key, str(value))])
        omero_object.linkAnnotation(ann)
