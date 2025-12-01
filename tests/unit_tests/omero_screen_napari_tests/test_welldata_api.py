import os
# Clear existing OMERO variables
for key in ["HOST", "USERNAME", "PASSWORD", "PROJECT_ID"]:
    if key in os.environ:
        del os.environ[key]
# Set environment to production
os.environ["ENV"] = "production"

from omero_screen.config import set_env_vars

set_env_vars()

print(os.environ["ENV"])
print(os.environ["HOST"])
print(os.environ["USERNAME"])
print(os.environ["PASSWORD"])

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.welldata_api import parse_omero_data

# Define the plate ID, well position, and image index


def test_welldata_api_basic():
      """Basic e2e test for welldata_api functionality"""
      plate_id = "1237"
      well_pos_list = "B7"
      images = "0"

      parse_omero_data(omero_data, plate_id, well_pos_list, images)

      print(omero_data.well_list)
      print(omero_data.well_id_list)
      print(omero_data.well_metadata_list)
      print(omero_data.well_ifdata.head())
      print(omero_data.image_index)


      assert omero_data.well_list is not None
      assert omero_data.well_id_list is not None
      assert omero_data.well_metadata_list is not None
