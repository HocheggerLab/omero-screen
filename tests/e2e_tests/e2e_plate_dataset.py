import os

from omero.gateway import BlitzGateway

from omero_screen.plate_dataset import PlateDataset


def run_plate_dataset_test(
    conn: BlitzGateway, teardown: bool = False, plate_id: int = 1
) -> PlateDataset:
    """Get the plate dataset for the plate."""
    return PlateDataset(conn, plate_id)
