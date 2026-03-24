CellView Documentation
======================

CellView is a local DuckDB-backed database for managing single-cell
measurement data from high-content microscopy screens. It sits between
the OMERO-Screen analysis pipeline (which produces per-plate CSV results)
and downstream analysis in Python notebooks or plotting libraries.

Key capabilities:

* **Import** data from CSV files, individual OMERO plates, or entire screens
  into a structured, queryable database.
* **Browse** projects, experiments, plates, and conditions via a rich
  terminal interface.
* **Edit** and **delete** metadata and plate records interactively.
* **Export** plate data as pandas or polars DataFrames for analysis.
* **Explore** data interactively by launching a pre-populated Jupyter
  notebook (with optional Napari viewer) from customisable templates.
* **Clean** orphaned records to keep the database consistent.
* **Python API** -- load data directly into DataFrames from your own scripts
  or notebooks with a single function call.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide
   data_structure
   api_reference
   dev_guide
