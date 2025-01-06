"""Module for handling file attachments loaded to the Omeroserver."""

import os
import tempfile
from typing import Optional

import pandas as pd
from omero.gateway import BlitzObjectWrapper, FileAnnotationWrapper
from omero.model import OriginalFileI
from pandas import DataFrame


def get_named_file_attachment(
    obj: BlitzObjectWrapper,
    file_name: str,
) -> Optional[FileAnnotationWrapper]:
    """
    Retrieve FileAnnotationWrappers for files with a specific name from an OMERO object.
    """

    matching_files = []
    for ann in obj.listAnnotations():
        if isinstance(ann, FileAnnotationWrapper):
            original_file = ann.getFile()

            # Skip if filename doesn't match (when specified)
            if original_file.getName() != file_name:
                continue
            matching_files.append(ann)

    if len(matching_files) > 1:
        raise ValueError(f"Multiple files found with name='{file_name}'")

    return matching_files[0] if matching_files else None


def parse_excel_data(
    file_ann: FileAnnotationWrapper,
) -> dict[str, DataFrame]:
    """
    Parse Excel data from a file attachment.

    Args:
        file_ann: FileAnnotationWrapper containing an Excel file

    Returns:
        dict[str, DataFrame]: Dictionary mapping sheet names to pandas DataFrames
    """
    original_file: OriginalFileI = file_ann.getFile()
    if original_file is None or not original_file.getName().endswith(".xlsx"):
        raise ValueError("File is not an Excel file")
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in original_file.asFileObj():
                tmp.write(chunk)
            tmp.flush()
        return pd.read_excel(tmp_path, sheet_name=None)  # type: ignore[no-any-return]
    finally:
        if tmp_path:
            os.unlink(tmp_path)  # Delete the temporary file
