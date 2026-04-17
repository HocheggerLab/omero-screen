CellView Documentation
======================

CellView is a local DuckDB-backed database that sits between the
OMERO-Screen analysis pipeline and your downstream data analysis. It
stores single-cell measurements from high-content immunofluorescence
screens in a structured, queryable form so you can browse, filter, and
export your data without touching the OMERO server again.

Workflow Overview
-----------------

.. code-block:: text

   OMERO-Screen
   (per-plate CSV results)
         |
         v
      [ import ]
   cellview import csv / plate / screen
         |
         v
     [ explore ]
   cellview projects / experiment / plate
         |
         v
    [ Python API ]
   cellview_load_data(plate_id, ...)
         |
         v
      [ plots ]
   omero-screen-plots / your own notebooks

Key Capabilities
----------------

* **Import** single-cell measurements from CSV files, individual OMERO
  plates, or entire OMERO screens into a structured DuckDB database.
* **Browse** projects, experiments, plates, and conditions via a
  rich terminal interface.
* **Edit** project and experiment metadata interactively.
* **Delete** plate records and all associated measurements safely.
* **Export** plate data as pandas or polars DataFrames for analysis.
* **Explore** data interactively by launching a pre-populated Jupyter
  notebook from customisable templates, with an optional Napari viewer.
* **Templates** — ship your own analysis notebooks and launch them with
  a single command.
* **Python API** — load data directly into DataFrames from scripts or
  notebooks with a single function call.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide
   explore_guide
   api_reference
   data_structure
   dev_guide
