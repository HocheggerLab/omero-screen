API Reference
=============

CellView provides a Python API for loading plate data directly into pandas
DataFrames. This is the recommended way to access data from notebooks and
scripts.

.. contents:: On this page
   :local:
   :depth: 2


Loading Data
------------

.. autofunction:: cellview.api.cellview_load_data

Usage examples:

.. code-block:: python

   from cellview.api import cellview_load_data

   # Load a single plate
   df, variable_names = cellview_load_data(12345)

   # Load multiple plates
   df, variable_names = cellview_load_data(12345, 12346, 12347)

   # Load all plates from an experiment by name
   df, variable_names = cellview_load_data(experiment="palb_washout_recovery")

   # Load all plates from an experiment by ID
   df, variable_names = cellview_load_data(experiment=6)

**Return values**:

- ``df`` -- A pandas DataFrame containing all single-cell measurements with
  condition metadata (cell line, well, condition variables) joined in.
- ``variable_names`` -- A list of the condition variable names (e.g.
  ``["Drug", "Concentration", "Timepoint"]``), useful for grouping and
  faceting in downstream plots.

**Auto-import**: If a requested plate is not yet in the local database,
``cellview_load_data`` will attempt to import it from OMERO automatically.


Advanced: Dependency Injection API
----------------------------------

.. autofunction:: cellview.api.cellview_load_data_with_injection

This is the underlying implementation used by ``cellview_load_data``. It
accepts the same arguments and is exposed for cases where explicit control
over the dependency injection pattern is needed (e.g. testing).


Legacy API
----------

.. autofunction:: cellview.api.cellview_load_data_legacy

Singleton-based version kept for backward compatibility. Prefer
``cellview_load_data`` for new code.
