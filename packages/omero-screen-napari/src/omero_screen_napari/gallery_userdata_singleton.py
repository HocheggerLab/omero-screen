from omero_screen_napari.gallery_userdata import UserData

userdata = UserData()


def reset_userdata() -> None:
    global userdata
    userdata.reset()  # Reset the data class
