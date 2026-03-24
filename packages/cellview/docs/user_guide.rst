User Guide
==========

CellView is a local DuckDB-based database for organising, querying, and
exploring single-cell measurement data produced by the OMERO-Screen analysis
pipeline. It provides both a CLI and a Python API.

.. contents:: On this page
   :local:
   :depth: 2


Getting Started
---------------

After installing the ``omero-screen`` workspace the ``cellview`` command is
available in your shell:

.. code-block:: bash

   cellview --help

All commands accept an optional ``--db`` flag to point to a custom database
file. When omitted, the default location ``~/.cellview/cellview.duckdb`` is
used:

.. code-block:: bash

   cellview --db /path/to/my.duckdb projects


Browsing Data
-------------

CellView provides four read-only display commands for inspecting the contents
of the database.

``cellview projects``
~~~~~~~~~~~~~~~~~~~~~

List every project with its experiment count.

.. code-block:: bash

   cellview projects

``cellview project <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~

Show experiments and plates that belong to a project.

.. code-block:: bash

   cellview project 1

``cellview experiment <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Show plates, channels, and condition variables for an experiment.

.. code-block:: bash

   cellview experiment 3

``cellview plate <id>``
~~~~~~~~~~~~~~~~~~~~~~~

Show a full plate summary: conditions per well, channel layout, and
measurement statistics.

.. code-block:: bash

   cellview plate 12345


Importing Data
--------------

Data can be imported from a local CSV file, from one or more OMERO plates, or
from an entire OMERO screen.

``cellview import csv <path>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import single-cell measurements from a CSV file. The file must contain at
least ``plate_id``, ``cell_line``, ``condition``, and measurement columns.

.. code-block:: bash

   cellview import csv /data/final_data_cc.csv

``cellview import plate <ids...>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import one or more plates directly from OMERO by their plate IDs. When
multiple IDs are given they must belong to the same screen.

.. code-block:: bash

   # Single plate
   cellview import plate 12345

   # Multiple plates from the same screen
   cellview import plate 12345 12346 12347

.. option:: --interactive

   Force interactive project/experiment selection, even when OMERO tags would
   normally provide the metadata automatically.

   .. code-block:: bash

      cellview import plate 12345 --interactive

``cellview import screen <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import every plate in an OMERO screen in one go.

.. code-block:: bash

   cellview import screen 456

.. option:: --interactive

   Force interactive project/experiment selection.


Editing Metadata
----------------

``cellview edit project <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactively edit a project's name and description.

.. code-block:: bash

   cellview edit project 1

``cellview edit experiment <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactively edit an experiment's name and description.

.. code-block:: bash

   cellview edit experiment 3


Exporting Data
--------------

``cellview export <id>``
~~~~~~~~~~~~~~~~~~~~~~~~

Export all measurements for a plate as a pandas DataFrame (prints a summary to
stdout). Useful for quick checks; for programmatic access see the
:doc:`api_reference`.

.. code-block:: bash

   cellview export 12345


Deleting Data
-------------

``cellview delete plate <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete a plate and **all** of its associated data (conditions, measurements,
condition variables). A database clean-up pass runs automatically afterwards
to remove any resulting orphan records.

.. code-block:: bash

   cellview delete plate 12345


Database Maintenance
--------------------

``cellview clean``
~~~~~~~~~~~~~~~~~~

Run an iterative orphan-removal pass over the entire database. Records that
no longer have a valid parent (e.g. conditions without a repeat, experiments
without plates) are deleted bottom-up until the hierarchy is consistent.

.. code-block:: bash

   cellview clean


Interactive Data Exploration
----------------------------

The ``explore`` command is a one-step launcher that creates a pre-populated
Jupyter notebook from a template, optionally opens a Napari viewer alongside
it, and opens the notebook in your preferred editor.

``cellview explore``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Explore specific plates
   cellview explore 12345 12346

   # Explore all plates from an experiment (by name or ID)
   cellview explore --experiment palb_washout
   cellview explore --experiment 6

Options
^^^^^^^

.. option:: --experiment EXPERIMENT

   Load all plates belonging to the given experiment. Accepts an experiment
   name (string) or numeric ID.

.. option:: --template NAME

   Template notebook to use. Defaults to ``cellcycle``. See
   :ref:`custom-templates` below for details on creating your own templates.

   .. code-block:: bash

      cellview explore 12345 --template myanalysis

.. option:: --fresh

   Regenerate the notebook from the template even if a notebook for the same
   plates already exists.

   .. code-block:: bash

      cellview explore 12345 --fresh

.. option:: --no-napari

   Skip launching the Napari viewer.

   .. code-block:: bash

      cellview explore 12345 --no-napari

.. option:: --list-templates

   Print the available templates and exit.

   .. code-block:: bash

      cellview explore --list-templates


How ``explore`` Works
^^^^^^^^^^^^^^^^^^^^^

1. **Notebook creation** -- CellView copies the selected template to
   ``~/.cellview/explore/`` and injects the plate IDs into a
   ``PLATE_IDS = [...]`` cell. If a notebook for the same set of plates (or
   experiment) already exists it is reused unless ``--fresh`` is passed.

2. **Napari launch** -- Unless ``--no-napari`` is given, a Napari viewer
   window is opened in a background process.

3. **Editor launch** -- The notebook is opened in **JupyterLab** by default.
   Set the ``CELLVIEW_EDITOR`` environment variable to ``vscode`` to open the
   notebook in VS Code instead:

   .. code-block:: bash

      export CELLVIEW_EDITOR=vscode
      cellview explore 12345

   Only two values are recognised:

   * ``vscode`` -- opens the notebook with ``code <path>``
   * anything else (or unset) -- opens the notebook with ``jupyter lab``

Notebook naming conventions:

* Single plate: ``explore_plate_12345.ipynb``
* Multiple plates: ``explore_plates_12345_12346_12347.ipynb``
* By experiment: ``explore_exp_6.ipynb``


.. _custom-templates:

Custom Analysis Templates
^^^^^^^^^^^^^^^^^^^^^^^^^

CellView ships with a built-in ``cellcycle`` template. You can add your own
templates for different analysis workflows.

**Where to put them**

Place ``.ipynb`` files in:

.. code-block:: text

   ~/.cellview/templates/

User templates with the same name as a built-in template take priority.

**Template conventions**

A template is a regular Jupyter notebook that follows one convention: it must
contain a code cell with a ``PLATE_IDS`` assignment that CellView can patch:

.. code-block:: python

   # This line gets replaced with the actual plate IDs at launch time
   PLATE_IDS: list[int] = []

You can then use the IDs to load data via the Python API:

.. code-block:: python

   from cellview.api import cellview_load_data

   df, variable_names = cellview_load_data(*PLATE_IDS)

**Template description**

The first Markdown cell in your notebook is used as the template description
when running ``cellview explore --list-templates``.

**Example workflow**

.. code-block:: bash

   # 1. Create the templates directory (first time only)
   mkdir -p ~/.cellview/templates

   # 2. Copy an existing notebook or create a new one
   cp my_analysis.ipynb ~/.cellview/templates/dose_response.ipynb

   # 3. Make sure it has a PLATE_IDS = [] cell

   # 4. Use it
   cellview explore 12345 --template dose_response


Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``CELLVIEW_EDITOR``
     - Editor for ``cellview explore``. Set to ``vscode`` to open notebooks
       in VS Code; otherwise JupyterLab is used.
   * - ``DATABASE_PATH``
     - Default DuckDB file path (can also be overridden per-command with
       ``--db``).
   * - ``TEST_DATABASE``
     - Set to ``true`` to use the test database instead of production.


CLI Reference (auto-generated)
------------------------------

The full argument parser is documented below for reference.

.. argparse::
   :module: cellview.cli
   :func: get_parser
   :prog: cellview
