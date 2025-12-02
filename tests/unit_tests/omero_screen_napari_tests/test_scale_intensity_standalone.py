import sys
from unittest.mock import MagicMock, patch

# Mock Ice and omero modules before they are imported
sys.modules["Ice"] = MagicMock()
sys.modules["omero"] = MagicMock()
sys.modules["omero.gateway"] = MagicMock()
sys.modules["ezomero"] = MagicMock()
sys.modules["omero_utils"] = MagicMock()
sys.modules["qtpy"] = MagicMock()
sys.modules["qtpy.QtWidgets"] = MagicMock()
sys.modules["skimage"] = MagicMock()
sys.modules["skimage.transform"] = MagicMock()
sys.modules["skimage.util"] = MagicMock()

# Mock omero_screen_napari.omero_data and omero_data_singleton
sys.modules["omero_screen_napari.omero_data"] = MagicMock()
sys.modules["omero_screen_napari.omero_data_singleton"] = MagicMock()
sys.modules["omero_screen_napari.utils"] = MagicMock()

mock_omero_data_module = MagicMock()
class MockOmeroData:
    def __init__(self):
        self.plate_id = 123
        self.plate_data = None
        self.channel_data = {"DAPI": 0, "Tub": 1}
        self.intensities = {}

mock_omero_data_module.OmeroData = MockOmeroData
sys.modules["omero_screen_napari.omero_data"] = mock_omero_data_module

mock_singleton_module = MagicMock()
omero_data_instance = MockOmeroData()
mock_singleton_module.omero_data = omero_data_instance
sys.modules["omero_screen_napari.omero_data_singleton"] = mock_singleton_module

# Mock cellview.api
sys.modules["cellview"] = MagicMock()
sys.modules["cellview.api"] = MagicMock()

import os
sys.path.insert(0, os.path.abspath("packages/omero-screen-napari/src"))

from omero_screen_napari.welldata_api import ScaleIntensityParser
import polars as pl

def test_scale_intensity_parser():
    print("Testing ScaleIntensityParser...")
    omero_data = MockOmeroData()

    # Create a dummy dataframe with necessary columns
    # We need _cell or _nucleus columns, and intensity_max/min columns
    data = {
        "intensity_max_DAPI_cell": [100, 200],
        "intensity_min_DAPI_cell": [10, 20],
        "intensity_max_Tub_cell": [150, 250],
        "intensity_min_Tub_cell": [15, 25],
        "other_col": [1, 2]
    }
    df = pl.DataFrame(data).lazy()
    omero_data.plate_data = df

    parser = ScaleIntensityParser(omero_data)
    parser.parse_intensities()

    print(f"Intensities: {omero_data.intensities}")

    # Verify results
    # DAPI (0): min of [10, 20] -> 10, mean of [100, 200] -> 150
    # Tub (1): min of [15, 25] -> 15, mean of [150, 250] -> 200

    assert 0 in omero_data.intensities
    assert 1 in omero_data.intensities

    assert omero_data.intensities[0] == (10, 150)
    assert omero_data.intensities[1] == (15, 200)

    print("Success: ScaleIntensityParser works correctly")

if __name__ == "__main__":
    try:
        test_scale_intensity_parser()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
