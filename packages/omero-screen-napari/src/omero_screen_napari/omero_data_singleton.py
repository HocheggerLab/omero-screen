from omero_screen_napari.omero_data import OmeroData

omero_data = OmeroData()  # Instantiate your data class here


def reset_omero_data() -> None:
    global omero_data
    omero_data.reset()  # Reset the data class
