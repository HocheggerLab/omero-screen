"""Module for handling file attachments loaded to the Omeroserver."""

import os
import tempfile
from typing import Optional

import pandas as pd
from omero.gateway import (
    BlitzGateway,
    BlitzObjectWrapper,
    FileAnnotationWrapper,
)
from omero.model import OriginalFileI, PlateI
from omero_screen.config import get_logger
from pandas import DataFrame

logger = get_logger(__name__)


def get_file_attachments(
    obj: BlitzObjectWrapper,
    extension: str,
) -> Optional[list[FileAnnotationWrapper]]:
    """
    Retrieve FileAnnotationWrappers for files with a specific extension from an OMERO object.

    Args:
        obj: The OMERO object to search for attachments
        extension: File extension to match (e.g., '.xlsx', '.pdf'). Case-insensitive.
            Should include the dot.

    Returns:
        List of matching FileAnnotationWrappers or None if no matches found
    """
    if not extension.startswith("."):
        extension = f".{extension}"
    extension = extension.lower()

    matching_files = []
    for ann in obj.listAnnotations():
        if isinstance(ann, FileAnnotationWrapper):
            original_file = ann.getFile()
            file_name = original_file.getName()
            if file_name and file_name.lower().endswith(extension):
                matching_files.append(ann)

    return matching_files if matching_files else None


def parse_excel_data(
    file_ann: FileAnnotationWrapper,
) -> dict[str, DataFrame] | None:
    """
    Parse Excel data from a file attachment.

    Args:
        file_ann: FileAnnotationWrapper containing an Excel file

    Returns:
        dict[str, DataFrame]: Dictionary mapping sheet names to pandas DataFrames
        or None if no Excel file is found
    """
    original_file: OriginalFileI = file_ann.getFile()
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in original_file.asFileObj():
                tmp.write(chunk)
            tmp.flush()
        logger.info("Parsing Excel Metadata File")
        return pd.read_excel(tmp_path, sheet_name=None)  # type: ignore[no-any-return]
    finally:
        if tmp_path:
            os.unlink(tmp_path)  # Delete the temporary file


def attach_excel_to_plate(
    conn: BlitzGateway,
    plate: PlateI,
    dataframes: dict[str, pd.DataFrame],
    filename: str = "metadata.xlsx",
) -> None:
    """Attach an Excel file with given dataframes to a plate.

    Args:
        conn: OMERO gateway connection
        plate: The plate to attach the file to
        dataframes: Dictionary of sheet_name -> dataframe to write to Excel
        filename: Name of the Excel file to create

    Returns:
        The created file annotation object
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, filename)

        # Write all dataframes to Excel file
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Create and attach file annotation
        file_ann = conn.createFileAnnfromLocalFile(
            temp_path,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        plate.linkAnnotation(file_ann)


def delete_excel_attachment(
    conn: BlitzGateway, omero_obj: BlitzObjectWrapper
) -> None:
    """Delete the excel file attachments from the object."""
    delete_file_attachment(conn, omero_obj, ".xlsx")


def delete_file_attachment(
    conn: BlitzGateway,
    omero_obj: BlitzObjectWrapper,
    ends_with: str | None = None,
) -> None:
    """Delete the file attachment from the object."""
    file_annotations = [
        ann
        for ann in omero_obj.listAnnotations()
        if isinstance(ann, FileAnnotationWrapper)
    ]

    for file_ann in file_annotations:
        delete = True
        # optionally only delete using a filename suffix
        if ends_with is not None:
            name = file_ann.getFile().getName()
            delete = name is not None and name.endswith(ends_with)
        if delete:
            # Get the link first
            links = list(file_ann.getParentLinks(omero_obj.OMERO_CLASS))
            for link in links:
                conn.deleteObject(link._obj)  # Delete the link
            conn.deleteObject(file_ann._obj)  # Then delete the annotation
