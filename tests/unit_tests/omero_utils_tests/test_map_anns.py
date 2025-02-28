# import pytest
# from omero.gateway import MapAnnotationWrapper
# from omero_utils.map_anns import parse_annotations


# @pytest.mark.parametrize(
#     "obj_type,obj_id_index", [("Screen", 0), ("Plate", 1), ("Image", 2)]
# )
# def test_parse_annotations_different_objects(
#     test_screen_plate_image, obj_type, obj_id_index, omero_conn
# ):
#     # Get IDs from fixture
#     ids = test_screen_plate_image
#     obj_id = ids[obj_id_index]

#     # Get the object
#     obj = omero_conn.getObject(obj_type, obj_id)

#     # Create and link map annotation
#     test_map = {"key1": "value1", "key2": "value2"}
#     map_ann = MapAnnotationWrapper(conn=omero_conn)
#     map_ann.setValue(list(test_map.items()))
#     map_ann.save()
#     obj.linkAnnotation(map_ann)

#     # Test the parser
#     result = parse_annotations(obj)
#     assert result == test_map
