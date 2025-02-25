"""Module for handling file attachments loaded to the Omeroserver."""

import os
import tempfile
from typing import Optional

import pandas as pd
from omero.gateway import BlitzObjectWrapper, FileAnnotationWrapper
from omero.model import OriginalFileI
from omero_screen.config import setup_logging
from pandas import DataFrame

logger = setup_logging("omero_utils")


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
