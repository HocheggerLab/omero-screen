"""Module for handling file attachments loaded to the Omeroserver."""

import os
import tempfile
from typing import Optional

import pandas as pd
from matplotlib.figure import Figure
from omero.gateway import (
    BlitzGateway,
    BlitzObjectWrapper,
    FileAnnotationWrapper,
    OriginalFileWrapper,
)
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
    original_file: OriginalFileWrapper = file_ann.getFile()
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


def parse_csv_data(
    file_ann: FileAnnotationWrapper,
) -> pd.DataFrame | None:
    """
    Parse CSV data from a file attachment.

    Args:
        file_ann: FileAnnotationWrapper containing a CSV file

    Returns:
        pd.DataFrame: DataFrame containing the CSV data
        or None if no CSV file is found
    """
    original_file: OriginalFileWrapper = file_ann.getFile()
    tmp_path = None
    try:
        tmp_path = tempfile.mktemp(suffix=".csv")
        with open(tmp_path, "wb") as file_on_disk:
            for chunk in original_file.asFileObj():
                file_on_disk.write(chunk)

        logger.info("Parsing CSV Metadata File")
        return pd.read_csv(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)  # Delete the temporary file


def attach_excel(
    conn: BlitzGateway,
    obj: BlitzObjectWrapper,
    dataframes: dict[str, pd.DataFrame],
    filename: str = "metadata.xlsx",
) -> None:
    """Attach an Excel file with given dataframes to a plate.

    Args:
        conn: OMERO gateway connection
        obj: The OMERO object to attach the file to
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
        obj.linkAnnotation(file_ann)


def delete_excel_attachment(
    conn: BlitzGateway, obj: BlitzObjectWrapper
) -> None:
    """Delete the excel file attachments from the object.
    The object should be refreshed before listing the updated annotations.
    Args:
        conn: OMERO gateway connection
        obj: The OMERO object
    """
    delete_file_attachment(conn, obj, ".xlsx")


def delete_file_attachment(
    conn: BlitzGateway,
    obj: BlitzObjectWrapper,
    ends_with: str | None = None,
) -> None:
    """Delete the file attachment from the object.
    The object should be refreshed before listing the updated annotations.
    Args:
        conn: OMERO gateway connection
        obj: The OMERO object
        ends_with: Optional suffix to filter attachments to delete
    """
    file_annotations = [
        ann
        for ann in obj.listAnnotations()
        if isinstance(ann, FileAnnotationWrapper)
    ]

    for ann in file_annotations:
        delete = True
        # optionally only delete using a filename suffix
        if ends_with is not None:
            name = ann.getFile().getName()
            delete = name is not None and name.endswith(ends_with)
        if delete:
            # Get the link first
            links = list(ann.getParentLinks(obj.OMERO_CLASS))
            for link in links:
                conn.deleteObject(link._obj)  # Delete the link
            conn.deleteObject(ann._obj)  # Then delete the annotation


def attach_figure(
    conn: BlitzGateway, fig: Figure, obj: BlitzObjectWrapper, title: str
) -> None:
    """Load a matplotlib figure to OMERO.
    Args:
        conn: OMERO gateway connection
        fig: matplotlib figure
        obj: The OMERO object to attach the figure to
        title: Name of the figure
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, f"{title}.png")
        _save_figure(
            fig,
            temp_path,
        )

        # Create and attach file annotation
        file_ann = conn.createFileAnnfromLocalFile(
            temp_path,
            mimetype="image/png",
        )
        obj.linkAnnotation(file_ann)


def _save_figure(
    fig: Figure,
    path: str,
    tight_layout: bool = True,
    resolution: float = 300,
    transparent: bool = False,
) -> None:
    """Coherent saving of matplotlib figures.
    Args:
        fig: Figure
        path: Path for saving (file extension defines the file type)
        tight_layout: option, default True
        resolution: option, default 300dpi
        transparent: option, default False
    """
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=resolution, transparent=transparent)


def attach_data(
    conn: BlitzGateway,
    df: pd.DataFrame,
    obj: BlitzObjectWrapper,
    title: str,
    cols: list[str] | None = None,
) -> None:
    """Load a table to OMERO.
    Args:
        conn: OMERO gateway connection
        df: Data table
        obj: The OMERO object to attach the table to
        title: Name of the table
        cols: Columns to use
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, f"{title}.csv")
        df.to_csv(temp_path, columns=cols)

        # Create and attach file annotation
        file_ann = conn.createFileAnnfromLocalFile(
            temp_path,
            mimetype="text/csv",
        )
        obj.linkAnnotation(file_ann)
