"""Integration tests for PlateDataset dataset/annotation lifecycle.

These run against the local test OMERO server (see ``scripts/manage_test_server.sh``)
via the session-scoped ``omero_conn`` fixture.
"""

from omero.gateway import BlitzGateway, MapAnnotationWrapper
from omero_utils.map_anns import parse_annotations
from omero_utils.omero_plate import base_plate, cleanup_plate

from omero_screen.constants import OmeroScreenNS
from omero_screen.plate_dataset import PlateDataset


def _dataset_annotations(conn: BlitzGateway, plate_id: int) -> list:
    """Return the plate's map annotations in the Dataset namespace."""
    plate = conn.getObject("Plate", plate_id)
    return [
        ann
        for ann in plate.listAnnotations(ns=OmeroScreenNS.DATASET)
        if isinstance(ann, MapAnnotationWrapper)
    ]


def test_stale_dataset_annotation_self_heals(omero_conn):
    """A deleted dataset should be recreated, not crash, with a single annotation.

    Reproduces the re-run bug: the plate caches its analysis dataset id in a
    ``Dataset`` map annotation; deleting that dataset (to re-run the screen)
    leaves the annotation dangling. PlateDataset must detect the stale id,
    recreate the dataset, and keep exactly one annotation pointing at it.
    """
    plate = None
    try:
        plate = base_plate(omero_conn, ["A1"])
        plate_id = plate.getId()

        # First run: creates a dataset and annotates the plate.
        ds1 = PlateDataset(omero_conn, plate_id).dataset_id
        assert omero_conn.getObject("Dataset", ds1) is not None
        anns = _dataset_annotations(omero_conn, plate_id)
        assert len(anns) == 1
        assert (
            int(
                parse_annotations(
                    omero_conn.getObject("Plate", plate_id),
                    ns=OmeroScreenNS.DATASET,
                )["Dataset"]
            )
            == ds1
        )

        # Simulate a re-run: delete the dataset. deleteAnns removes annotations
        # *on the dataset*, not the plate's annotation that references it, so
        # the plate is left with a dangling Dataset annotation.
        omero_conn.deleteObjects("Dataset", [ds1], deleteAnns=True, wait=True)
        assert omero_conn.getObject("Dataset", ds1) is None
        assert (
            len(_dataset_annotations(omero_conn, plate_id)) == 1
        )  # still stale

        # Second run: must recreate (new id) rather than return the dead id,
        # and must not leave a duplicate annotation behind.
        ds2 = PlateDataset(omero_conn, plate_id).dataset_id
        assert ds2 != ds1
        assert omero_conn.getObject("Dataset", ds2) is not None
        anns = _dataset_annotations(omero_conn, plate_id)
        assert len(anns) == 1, (
            "stale annotation must be rewritten, not duplicated"
        )
        assert (
            int(
                parse_annotations(
                    omero_conn.getObject("Plate", plate_id),
                    ns=OmeroScreenNS.DATASET,
                )["Dataset"]
            )
            == ds2
        )

        # Third run: annotation is valid, so the same id is returned (idempotent).
        ds3 = PlateDataset(omero_conn, plate_id).dataset_id
        assert ds3 == ds2
        assert len(_dataset_annotations(omero_conn, plate_id)) == 1

        # Clean up the dataset we created last.
        omero_conn.deleteObjects("Dataset", [ds2], deleteAnns=True, wait=True)
    finally:
        if plate:
            cleanup_plate(omero_conn, plate)
