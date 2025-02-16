from typing import Any

from omero.gateway import BlitzObjectWrapper, MapAnnotationWrapper


def parse_annotations(omero_object: BlitzObjectWrapper) -> dict[str, Any]:
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
