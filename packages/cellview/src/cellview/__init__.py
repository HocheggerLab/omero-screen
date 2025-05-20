"""CellView: A DuckDB-backed database for OMERO Screen CSV data.

This package provides tools to import, organize, and query single-cell measurement data
exported as CSV files from OMERO Screen high-content imaging experiments. It leverages
DuckDB for efficient local storage and querying, and supports project, experiment, plate,
condition, and measurement management. CellView enables streamlined data integration,
cleanup, and export for downstream analysis.

Access Points:
- Text User Interface (TUI):
    Use the command-line interface to import CSVs, manage projects/experiments, clean up the database,
    and display or export data. Run `python -m cellview.main --help` for available commands and options.
- Python API:
    Import the `cellview.api` module to programmatically load and query data as pandas DataFrames.
    For example, use `cellview.api.cellview_load_data(plate_id)` to load data for a given plate.
"""

__version__ = "0.1.1"
