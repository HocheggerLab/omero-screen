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

import omero
from loguru import logger
from omero.gateway import (
    BlitzGateway,
    BlitzObjectWrapper,
    MapAnnotationWrapper,
)
from omero_utils.map_anns import add_map_annotations, parse_annotations
from omero_utils.message import PlateDataError, log_success

from omero_screen.constants import OmeroScreenNS

SUCCESS_STYLE = "bold cyan"


class PlateDataset:
    """Manages the creation and retrieval of OMERO datasets associated with screening plates.

    This class ensures that a dataset corresponding to a given plate ID exists within the OMERO 'Screens' project.
    If the project does not exist, it will be created.
    If the dataset does not exist, it will be created and linked to the project.
    The class also provides access to the dataset's ID for further operations.

    Args:
        conn (BlitzGateway): An active OMERO connection.
        plate_id (int): The unique identifier of the plate.

    Attributes:
        conn (BlitzGateway): The OMERO connection used for operations.
        plate_id (int): The plate identifier.
        dataset_id (int): The OMERO dataset ID associated with the plate.

    Raises:
        PlateDataError: If the plate is missing, or multiple datasets are found with the same name.
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

        The plate's ``Dataset`` map annotation is the cached link to the
        analysis dataset, but it can go stale: if the dataset was deleted to
        re-run the screen, the annotation still points at a now-missing ID.
        We therefore *validate* the annotation before trusting it. If it is
        stale (or absent), we fall through to find/create the dataset in the
        'Screens' project and rewrite the annotation in place, keeping a single
        ``Dataset`` annotation as the source of truth.

        Returns:
            int: The ID of the dataset.

        Raises:
            PlateDataError: If the plate is missing, or multiple datasets are
                found with the same name.
        """
        plate = self.conn.getObject("Plate", self.plate_id)
        if plate is None:
            raise PlateDataError(
                f"Plate missing: '{self.plate_id}'",
                logger,
            )

        # Trust the cached annotation only if it still resolves to a real
        # dataset; a dangling ID is treated as "not annotated".
        dataset_id = self._annotated_dataset_id(plate)
        if dataset_id is not None:
            logger.debug(
                f"Found dataset annotation for plate {self.plate_id}: {dataset_id}"
            )
            return dataset_id

        project_id = self._ensure_screens_project()
        dataset_id = self._find_or_create_dataset(project_id)
        self._write_dataset_annotation(dataset_id)
        return dataset_id

    def _annotated_dataset_id(self, plate: BlitzObjectWrapper) -> int | None:
        """Return the annotated dataset ID if it still resolves to a dataset.

        Args:
            plate: The plate object to read the annotation from.

        Returns:
            The dataset ID if the plate carries a ``Dataset`` annotation that
            points at an existing dataset, otherwise ``None``. A stale
            annotation (dataset deleted) is logged and treated as ``None`` so
            the caller recreates the dataset.
        """
        anns = parse_annotations(plate, ns=OmeroScreenNS.DATASET)
        dataset_id = int(anns.get("Dataset", 0)) if anns else 0
        if not dataset_id:
            return None
        if self.conn.getObject("Dataset", dataset_id) is not None:
            return dataset_id
        logger.warning(
            f"Plate {self.plate_id} is annotated with dataset {dataset_id} which no longer exists; recreating the dataset."
        )
        return None

    def _ensure_screens_project(self) -> int:
        """Return the 'Screens' project ID for the current user, creating it if needed."""
        owner_id = self.conn.getUser().getId()
        projects = list(
            self.conn.getObjects(
                "Project",
                opts={"owner": owner_id},
                attributes={"name": "Screens"},
            )
        )
        if len(projects) == 0:
            logger.debug("Creating Screens project")
            obj = omero.model.ProjectI()
            obj.setName(omero.rtypes.rstring("Screens"))
            project_id = (
                self.conn.getUpdateService()
                .saveAndReturnObject(obj)
                .getId()
                .val
            )
        else:
            project_id = projects[0].getId()
        logger.debug(f"Using Screens project {project_id}")
        return int(project_id)

    def _find_or_create_dataset(self, project_id: int) -> int:
        """Find the plate's dataset in the project, or create and link a new one.

        Args:
            project_id: The 'Screens' project the dataset lives in.

        Returns:
            The dataset ID.

        Raises:
            PlateDataError: If multiple datasets share the plate's name.
        """
        dataset_name = str(self.plate_id)
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

    def _write_dataset_annotation(self, dataset_id: int) -> None:
        """Record ``dataset_id`` on the plate as the single ``Dataset`` annotation.

        If a (stale) ``Dataset`` annotation already exists, it is updated in
        place rather than duplicated -- updating needs only ``canEdit``, which
        is weaker than the ``canDelete`` required to remove another user's
        annotation. Any extra annotations are collapsed so downstream code that
        assumes a single ``Dataset`` annotation stays correct. Permission
        failures are logged but not fatal: the dataset ID itself is valid for
        this run, and the next run will re-validate.

        Args:
            dataset_id: The dataset ID to record on the plate.
        """
        plate = self.conn.getObject("Plate", self.plate_id)
        existing = [
            ann
            for ann in plate.listAnnotations(ns=OmeroScreenNS.DATASET)
            if isinstance(ann, MapAnnotationWrapper)
        ]
        if not existing:
            add_map_annotations(
                self.conn,
                plate,
                {"Dataset": dataset_id},
                ns=OmeroScreenNS.DATASET,
            )
            return
        # Update the first annotation in place; drop any duplicates.
        primary, *extras = existing
        try:
            primary.setValue([["Dataset", str(dataset_id)]])
            primary.save()
            for extra in extras:
                self.conn.deleteObject(extra._obj)
        except Exception as exc:  # noqa: BLE001 - non-fatal cache update
            logger.warning(
                f"Could not update stale Dataset annotation on plate {self.plate_id} (dataset {dataset_id} is valid for this run): {exc}"
            )
