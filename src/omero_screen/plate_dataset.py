"""This module provides the PlateDataset class for managing OMERO datasets associated with screening plates.

It enables the creation and retrieval of datasets linked to a specific plate within a designated OMERO project (typically named 'Screens').

Features:
- Ensures a dataset exists for a given plate, creating one if necessary.
- Links the dataset to the specified OMERO project.
- Handles error cases such as missing projects or duplicate datasets.
- Logs key actions and errors for traceability.

Typical usage:
    from omero_screen.plate_dataset import PlateDataset
    dataset = PlateDataset(conn, plate_id)
    dataset_id = dataset.dataset_id

"""

import os

import omero
from omero.gateway import BlitzGateway
from omero_utils.message import PlateDataError, log_success

from omero_screen.config import get_logger

logger = get_logger(__name__)

SUCCESS_STYLE = "bold cyan"


class PlateDataset:
    """Manages the creation and retrieval of OMERO datasets associated with screening plates.

    This class ensures that a dataset corresponding to a given plate ID exists within the OMERO 'Screens' project. If the dataset does not exist, it will be created and linked to the project. The class also provides access to the dataset's ID for further operations.

    Args:
        conn (BlitzGateway): An active OMERO connection.
        plate_id (int): The unique identifier of the plate.

    Attributes:
        conn (BlitzGateway): The OMERO connection used for operations.
        plate_id (int): The plate identifier.
        dataset_id (int): The OMERO dataset ID associated with the plate.

    Raises:
        PlateDataError: If the project is missing, the project name is incorrect, or multiple datasets are found with the same name.
    """

    def __init__(self, conn: BlitzGateway, plate_id: int):
        """Initialize the PlateDataset instance.

        Args:
            conn (BlitzGateway): The OMERO connection.
            plate_id (int): The ID of the plate.
        """
        self.conn = conn
        self.plate_id = plate_id
        self.dataset_id = self._create_dataset()

    def _create_dataset(self) -> int:
        """Create a new dataset or return the ID of an existing one.

        This method checks if a dataset exists for the given plate ID within the 'Screens' project.
        If the dataset does not exist, it creates a new one and links it to the project.
        If multiple datasets are found with the same name, it raises an error.

        Returns:
            int: The ID of the dataset.
        """
        dataset_name = str(self.plate_id)
        project_id = os.getenv("PROJECT_ID")
        try:
            project = self.conn.getObject("Project", project_id)
            assert project is not None, "Project is missing"
            assert project.getName() == "Screens", (
                "Project name does not match 'Screens'"
            )
        except Exception as e:
            raise PlateDataError(
                f"Screens project with ID {project_id} not found", logger
            ) from e

        datasets = list(
            self.conn.getObjects(
                "Dataset",
                opts={"project": project_id},
                attributes={"name": dataset_name},
            )
        )

        if len(datasets) > 1:
            raise PlateDataError(
                f"Multiple plate datasets found with the same name: '{dataset_name}'",
                logger,
            )
        elif len(datasets) == 1:
            dataset_id = datasets[0].getId()
            log_success(
                SUCCESS_STYLE,
                f"Plate dataset exists with ID: {dataset_id}",
                logger,
            )
            return int(dataset_id)
        else:
            obj = omero.model.DatasetI()
            obj.setName(omero.rtypes.rstring(self.plate_id))
            obj = self.conn.getUpdateService().saveAndReturnObject(obj)
            new_dataset_id = obj.getId().val
            link = omero.model.ProjectDatasetLinkI()
            link.setChild(obj)
            link.setParent(omero.model.ProjectI(project_id, False))
            self.conn.getUpdateService().saveObject(link)
            log_success(
                SUCCESS_STYLE,
                f"Plate dataset created with ID {new_dataset_id} and linked to Screens project",
                logger,
            )
            return int(new_dataset_id)
