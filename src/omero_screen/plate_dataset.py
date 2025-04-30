"""Module for managing a dataset associated with a plate."""

import os

import omero
from omero.gateway import BlitzGateway
from omero_utils.message import (
    PlateDataError,
    log_success,
)

from omero_screen.config import get_logger

logger = get_logger(__name__)

SUCCESS_STYLE = "bold cyan"


class PlateDataset:
    """Class to create a dataset for the Omero-Screen plate."""

    def __init__(self, conn: BlitzGateway, plate_id: int):
        self.conn = conn
        self.plate_id = plate_id
        self.dataset_id = self._create_dataset()

    def _create_dataset(self) -> int:
        """Create a new dataset or return the ID of an existing one."""
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
