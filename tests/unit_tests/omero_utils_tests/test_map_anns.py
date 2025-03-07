import pytest
from omero.gateway import MapAnnotationWrapper

from omero_utils.map_anns import (
    add_map_annotations,
    delete_map_annotations,
    parse_annotations,
)


@pytest.mark.parametrize(
    "obj_type,obj_id_index", [("Screen", 0), ("Plate", 1), ("Image", 2)]
)
def test_parse_annotations_different_objects(
    test_screen_plate_image, obj_type, obj_id_index, omero_conn
):
    # Get IDs from fixture
    ids = test_screen_plate_image
    obj_id = ids[obj_id_index]

    # Get the object
    obj = omero_conn.getObject(obj_type, obj_id)

    # Create and link map annotation
    test_map = {"key1": "value1", "key2": "value2"}
    map_ann = MapAnnotationWrapper(conn=omero_conn)
    map_ann.setValue(list(test_map.items()))
    map_ann.save()
    obj.linkAnnotation(map_ann)

    # Test the parser
    result = parse_annotations(obj)
    assert result == test_map


@pytest.mark.parametrize(
    "obj_type,obj_id_index", [("Screen", 0), ("Plate", 1), ("Image", 2)]
)
def test_delete_map_annotations(
    test_screen_plate_image, omero_conn, obj_type, obj_id_index
):
    # Get IDs from fixture
    ids = test_screen_plate_image
    obj_id = ids[obj_id_index]

    # Get the object
    obj = omero_conn.getObject(obj_type, obj_id)

    # Create and link map annotation
    test_map = {"key1": "value1", "key2": "value2"}
    map_ann = MapAnnotationWrapper(conn=omero_conn)
    map_ann.setValue(list(test_map.items()))
    map_ann.save()
    obj.linkAnnotation(map_ann)

    # Delete map annotation
    delete_map_annotations(omero_conn, obj)
    obj = omero_conn.getObject(obj_type, obj_id)
    # Verify deletion
    assert map_ann not in obj.listAnnotations()


@pytest.mark.parametrize(
    "obj_type,obj_id_index", [("Screen", 0), ("Plate", 1), ("Image", 2)]
)
def test_add_map_annotations(
    test_screen_plate_image, omero_conn, obj_type, obj_id_index
):
    # Get IDs from fixture
    ids = test_screen_plate_image
    obj_id = ids[obj_id_index]

    # Get the object
    obj = omero_conn.getObject(obj_type, obj_id)

    # Add map annotations
    test_map = {"key1": "value1", "key2": "value2"}
    add_map_annotations(omero_conn, obj, test_map)

    # Get all map annotations
    map_anns = [
        ann
        for ann in obj.listAnnotations()
        if isinstance(ann, MapAnnotationWrapper)
    ]

    # Verify each annotation contains one key-value pair from test_map
    assert len(map_anns) == len(test_map)
    found_pairs = []
    for ann in map_anns:
        value_pairs = ann.getValue()
        assert (
            len(value_pairs) == 1
        )  # Each annotation should have one key-value pair
        key, value = value_pairs[0]
        assert test_map[key] == value
        found_pairs.append((key, value))

    # Verify all key-value pairs were found
    assert sorted(found_pairs) == sorted(test_map.items())
