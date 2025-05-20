"""Module for importing data into CellView.

This module provides parsers for importing data into CellView from
csv files produced by omero screen.

Individual tables are parsed by different parser classes,
and then combined in the import-functions module.
"""

from .import_functions import import_data

__all__ = ["import_data"]
