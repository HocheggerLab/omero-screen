Data Structure and Hierarchy
============================

CellView organises high-content screening data using a hierarchical structure
that maps to OMERO's data model but is optimised for analysis queries.

.. contents:: On this page
   :local:
   :depth: 2

Hierarchy
---------

Data is organised in five levels:

1. **Project** -- The top-level container. A project groups related
   experiments (e.g. "DNA Damage Response").
2. **Experiment** -- A logical grouping of plates that belong to the same
   experimental series (e.g. "PALB2 washout recovery").
3. **Plate (Repeat)** -- A physical plate (e.g. a 96-well plate). In the
   database schema this level is called ``repeats`` because each plate is
   typically a biological repeat of the same experiment.
4. **Well (Condition)** -- A specific well on a plate. This is where
   experimental conditions are defined: cell line, antibodies, drug
   concentrations, and any additional variables.
5. **Measurement** -- Individual single-cell data points: intensities, areas,
   cell-cycle phase assignments, and more.

.. code-block:: text

   Project
   └── Experiment
       └── Plate (Repeat)
           └── Well (Condition)
               ├── condition_variables (key-value pairs)
               └── Measurements (one row per cell)


Inferring Hierarchy from OMERO
------------------------------

When importing data from OMERO, CellView can automatically infer the Project
and Experiment names from **Tags** attached to the Plate object:

-  **Project**: A tag with the format ``Project: <Name>`` sets the project.
-  **Experiment**: A tag with the format ``Experiment: <Name>`` sets the
   experiment.

If these tags are not present, you will be prompted to select or create the
project and experiment interactively (or you can force this with the
``--interactive`` flag).


Database Schema
---------------

CellView uses DuckDB as its storage engine. The schema consists of six
tables:

``projects``
~~~~~~~~~~~~

.. code-block:: sql

   project_id   INTEGER PRIMARY KEY  -- auto-increment
   project_name TEXT    UNIQUE
   description  TEXT

``experiments``
~~~~~~~~~~~~~~~

.. code-block:: sql

   experiment_id   INTEGER PRIMARY KEY  -- auto-increment
   project_id      INTEGER REFERENCES projects
   experiment_name TEXT
   description     TEXT

``repeats``
~~~~~~~~~~~

Stores per-plate metadata including channel assignments.

.. code-block:: sql

   repeat_id     INTEGER PRIMARY KEY  -- auto-increment
   experiment_id INTEGER REFERENCES experiments
   plate_id      INTEGER
   date          DATE
   lab_member    TEXT
   channel_0     TEXT NOT NULL
   channel_1     TEXT
   channel_2     TEXT
   channel_3     TEXT
   classifier    TEXT

``conditions``
~~~~~~~~~~~~~~

One row per well, linked to a plate (repeat).

.. code-block:: sql

   condition_id INTEGER PRIMARY KEY  -- auto-increment
   repeat_id    INTEGER REFERENCES repeats
   well         TEXT
   well_id      TEXT
   cell_line    TEXT
   antibody     TEXT
   antibody_1   TEXT
   antibody_2   TEXT
   antibody_3   TEXT
   UNIQUE(repeat_id, well)

``condition_variables``
~~~~~~~~~~~~~~~~~~~~~~~

Flexible key-value pairs for additional per-well metadata (e.g. drug
concentrations, timepoints, siRNA targets). These are pivoted into separate
DataFrame columns when data is exported.

.. code-block:: sql

   variable_id    INTEGER PRIMARY KEY  -- auto-increment
   condition_id   INTEGER REFERENCES conditions
   variable_name  TEXT
   variable_value TEXT

``measurements``
~~~~~~~~~~~~~~~~

Single-cell data. Each row corresponds to one segmented cell or nucleus.

.. code-block:: sql

   measurement_id               INTEGER PRIMARY KEY  -- auto-increment
   condition_id                 INTEGER REFERENCES conditions
   image_id                     INTEGER
   timepoint                    INTEGER
   cell_cycle                   TEXT
   cell_cycle_detailed          TEXT
   label                        FLOAT
   area_nucleus                 FLOAT
   centroid-0-nuc               FLOAT
   centroid-1-nuc               FLOAT
   intensity_min_DAPI_nucleus   FLOAT
   intensity_mean_DAPI_nucleus  FLOAT
   intensity_max_DAPI_nucleus   FLOAT
   integrated_int_DAPI_norm     FLOAT
   Cyto_ID                      FLOAT
   area_cell                    FLOAT
   area_cyto                    FLOAT
   ...                          (additional per-channel intensity columns)

The measurement columns are created dynamically based on the channels present
in the imported data. Columns for each channel are generated per compartment
(nucleus, cell, cytoplasm) with ``intensity_min``, ``intensity_mean``, and
``intensity_max`` prefixes.


Data Flow
---------

**CSV import path**:

1. CSV is loaded and cleaned (column normalisation, duplicate removal).
2. Project and experiment are selected or created.
3. A ``repeats`` record is created with plate metadata and channel mapping.
4. ``conditions`` rows are created -- one per unique well.
5. Extra metadata columns are pivoted into ``condition_variables``.
6. Single-cell rows are bulk-inserted into ``measurements``.

**OMERO import path**:

1. CSV attachment is downloaded from the OMERO plate object.
2. Metadata is read from OMERO annotations or tags.
3. The remaining steps follow the CSV import path above.

**Export path**:

1. ``repeats`` + ``conditions`` + ``measurements`` are joined.
2. ``condition_variables`` are pivoted back into separate columns.
3. A pandas (or polars) DataFrame is returned along with the list of
   variable names.
