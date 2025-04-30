import os

from omero.gateway import BlitzGateway

from omero_screen.plate_dataset import PlateDataset


def run_plate_dataset_test(
    conn: BlitzGateway, teardown: bool = False, plate_id: int = 1
) -> PlateDataset:
    """Get the plate dataset for the plate."""
    return PlateDataset(conn, plate_id)


def run_plate_dataset_missing_project_test(
    conn: BlitzGateway, teardown: bool = False, plate_id: int = 1
) -> PlateDataset:
    """Get the plate dataset when the plate project does not exist."""
    project_id = os.environ["PROJECT_ID"]
    try:
        os.environ["PROJECT_ID"] = "-123"
        return PlateDataset(conn, plate_id)
    finally:
        os.environ["PROJECT_ID"] = project_id
