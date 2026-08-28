User Guide
==========

CellView is a local database for storing, browsing, and exporting the
single-cell measurements produced by the OMERO-Screen analysis pipeline.
This guide walks through every command you will use day-to-day, with
concrete examples. No programming experience is required — just a
terminal and data to analyse.

.. contents:: On this page
   :local:
   :depth: 2


Getting Started
---------------

CellView is part of the ``omero-screen`` workspace. If the workspace is
already installed you do not need to install anything separately. To
confirm that CellView is available, run:

.. code-block:: bash

   cellview --help

You should see a list of subcommands. If you see a "command not found"
error, make sure the virtual environment is activated:

.. code-block:: bash

   source .venv/bin/activate

**Default database location**

CellView stores everything in a single DuckDB file. By default that file
lives at:

.. code-block:: text

   ~/.cellview/cellview.duckdb

The directory is created automatically the first time you run a command
that writes to the database.

**Using a custom database path**

Every CellView command accepts a ``--db`` flag so you can point to a
different file — useful when you want to keep separate databases for
different projects or when working on a shared server:

.. code-block:: bash

   cellview --db /scratch/myproject/myproject.duckdb projects


Browsing Your Data
------------------

CellView has four read-only commands for inspecting what is in the
database. They do not change anything, so you can run them as often as
you like.

``cellview projects``
~~~~~~~~~~~~~~~~~~~~~

List every project in the database together with the number of
experiments it contains.

.. code-block:: bash

   cellview projects

Example output (formatted as a rich table in your terminal):

.. code-block:: text

   ┌────┬──────────────────────┬────────────────┐
   │ ID │ Name                 │ Experiments    │
   ├────┼──────────────────────┼────────────────┤
   │  1 │ PALB2 washout        │ 3              │
   │  2 │ CDK inhibitor screen │ 1              │
   └────┴──────────────────────┴────────────────┘

Use the IDs shown here as arguments to the commands below.

``cellview project <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~

Show the experiments and plates that belong to one project.

.. code-block:: bash

   cellview project 1

This prints each experiment within the project along with the plates it
contains, so you can quickly find the plate IDs you need for export or
further analysis.

``cellview experiment <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Show the plates in an experiment together with the measurement channels
and any condition variables (for example drug, timepoint, concentration)
that were recorded.

.. code-block:: bash

   cellview experiment 3

This is useful for checking whether an import completed correctly: you
can confirm that all the expected plates are listed and that the channel
and condition information looks right.

``cellview plate <id>``
~~~~~~~~~~~~~~~~~~~~~~~

Show a full summary for one plate: every condition and the number of
cells measured per condition, the channel layout, and basic measurement
statistics.

.. code-block:: bash

   cellview plate 12345

Use this command after importing a plate to verify the data landed
correctly before running any analysis.


Importing Data
--------------

CellView supports three import routes. All of them walk you through an
interactive prompt to assign the plate to a project and experiment. If
the OMERO plate already has tags called ``Project: <name>`` and
``Experiment: <name>``, CellView reads them automatically and skips the
interactive step.

**Naming the target up front**

Every import route accepts ``--project`` and ``--experiment``, which take
existing IDs and skip the prompt altogether:

.. code-block:: bash

   cellview import plate 12345 --experiment 7
   cellview import plate 12345 12346 12347 --experiment 7

This is worth doing whenever you already know where the data belongs —
and especially when importing several plates, since otherwise you answer
the same prompt once per plate.

An experiment already implies its project, so ``--project`` is optional;
give it on its own to pick the project but still resolve the experiment
from the plate's own metadata. Both IDs are checked before any plate is
touched, so a typo fails immediately rather than half-way through a
multi-plate import, and a mismatched pair is rejected.

From a CSV file
~~~~~~~~~~~~~~~

Use this route when you have already run the OMERO-Screen pipeline and
have a CSV file of single-cell measurements on disk.

.. code-block:: bash

   cellview import csv /data/final_data_cc.csv

**Required columns**

The CSV must contain at least the following columns. Any additional
numeric columns are treated as measurements and stored in the database.

* ``plate_id`` — the OMERO plate ID (integer)
* ``cell_line`` — cell line name (e.g. ``RPE``, ``HeLa``)
* ``condition`` — treatment condition (e.g. ``control``, ``drug_10uM``)

.. note::

   Column names are case-sensitive. Check that your CSV uses exactly
   these names before importing. The pipeline output file is usually
   called ``final_data_cc.csv`` and already has the correct column names.

**Interactive prompts**

After reading the CSV, CellView will ask you to select or create a
project and experiment for the plate:

.. code-block:: text

   Select a project:
     1. PALB2 washout
     2. CDK inhibitor screen
     3. [Create new project]
   >

Follow the prompts to assign the plate. If you make a mistake you can
always edit the metadata afterwards (see `Editing Metadata`_) or delete
the plate and re-import it (see `Deleting Data`_).

**Skipping the prompts with OMERO tags**

If the plate on the OMERO server has map annotations (key-value pairs)
with keys ``Project`` and ``Experiment``, CellView reads them
automatically:

.. code-block:: text

   Project: PALB2 washout
   Experiment: siRNA timecourse

When these tags are present the interactive step is skipped entirely.

From OMERO plates
~~~~~~~~~~~~~~~~~

Use this route to import plates directly from the OMERO server without
first exporting a CSV. You will need network access to the OMERO server.

.. code-block:: bash

   # Import a single plate
   cellview import plate 12345

   # Import several plates that belong to the same screen
   cellview import plate 12345 12346 12347

.. note::

   When importing multiple plates they must all come from the same OMERO
   screen. CellView uses the screen to determine which experiment the
   plates belong to.

**Forcing interactive selection**

If you want to override the automatic project/experiment detection (for
example to move plates into a different experiment), add the
``--interactive`` flag:

.. code-block:: bash

   cellview import plate 12345 --interactive

From an entire OMERO screen
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import every plate in a screen at once. This is the most convenient
route when you have just finished a full screen and want to load all the
data in one step.

.. code-block:: bash

   cellview import screen 456

Add ``--interactive`` to override automatic project/experiment detection:

.. code-block:: bash

   cellview import screen 456 --interactive


Editing Metadata
----------------

You can rename projects and experiments or update their descriptions
at any time. These changes are safe — they only update text fields and
do not touch any measurement data.

``cellview edit project <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cellview edit project 1

CellView will show the current name and description and prompt you to
enter new values. Press Enter to keep the existing value for any field.

``cellview edit experiment <id>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cellview edit experiment 3

Works the same way as editing a project.

.. note::

   To find the numeric IDs for projects and experiments, run
   ``cellview projects`` first, then ``cellview project <id>`` to list
   experiments within it.


Exporting Data
--------------

``cellview export <plate_id>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Print a summary of a plate's measurements to the terminal.

.. code-block:: bash

   cellview export 12345

This is useful for a quick sanity-check. For full programmatic access to
the data — for example to load it into a pandas DataFrame for plotting —
use the Python API instead (see :doc:`api_reference`).

**Condition variables in exports**

When you import a plate, CellView stores extra per-condition columns such
as drug name, timepoint, and concentration. In the exported DataFrame
these appear as individual columns alongside the measurement columns, so
you can filter and group by them directly in Python without any extra
wrangling.


Deleting Data
-------------

``cellview delete plate <id> [<id> ...]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cellview delete plate 12345
   cellview delete plate 12345 12346 12347

This removes the plate record and **all** associated data: conditions,
condition variables, and every single-cell measurement row. Several
plates can be given at once; they are deleted in the order listed. A
single clean-up pass runs automatically afterwards to remove any
orphaned records.

.. warning::

   Deletion is permanent. There is no undo. Before deleting a plate,
   make sure you still have the original CSV file or the data is still
   accessible on the OMERO server, in case you need to re-import it.

If you only want to fix the metadata (wrong project or experiment name),
use ``cellview edit`` instead — that is non-destructive.


Database Maintenance
--------------------

``cellview clean``
~~~~~~~~~~~~~~~~~~

Run an orphan-removal pass over the database. Orphaned records are rows
that no longer have a valid parent — for example an experiment that
contains no plates, or conditions that point to a plate that was deleted.

.. code-block:: bash

   cellview clean

You do not normally need to run this manually because plate deletion
triggers a clean-up automatically. Run it if you have edited the
database directly (for example with a DuckDB client) or if something
looks inconsistent when browsing.


Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``DATABASE_PATH``
     - Path to the default DuckDB file. When set, this overrides the
       built-in default of ``~/.cellview/cellview.duckdb``. You can
       still override this per-command with ``--db``.
   * - ``TEST_DATABASE``
     - Set to ``true`` to use a separate test database at
       ``~/.cellview/cellview-test.duckdb``. Useful when trying out
       imports without affecting your production data.
   * - ``CELLVIEW_EDITOR``
     - Controls which editor opens notebooks launched by
       ``cellview explore``. Set to ``vscode`` to open in VS Code;
       leave unset to use JupyterLab. See :doc:`explore_guide` for
       details.


CLI Quick Reference
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Command
     - What it does
   * - ``cellview projects``
     - List all projects with experiment counts
   * - ``cellview project <id>``
     - Show experiments and plates in a project
   * - ``cellview experiment <id>``
     - Show plates, channels, and condition variables in an experiment
   * - ``cellview plate <id>``
     - Show full plate summary: conditions, channels, measurement stats
   * - ``cellview import csv <path>``
     - Import measurements from a CSV file
   * - ``cellview import plate <ids...>``
     - Import one or more plates from OMERO
   * - ``cellview import plate <ids...> --interactive``
     - Import from OMERO, forcing manual project/experiment selection
   * - ``cellview import ... --project <id> --experiment <id>``
     - Import into a known project/experiment, skipping the prompts
   * - ``cellview import screen <id>``
     - Import every plate in an OMERO screen
   * - ``cellview edit project <id>``
     - Rename a project or update its description
   * - ``cellview edit experiment <id>``
     - Rename an experiment or update its description
   * - ``cellview export <plate_id>``
     - Print a plate measurement summary to the terminal
   * - ``cellview delete plate <id> [<id> ...]``
     - Permanently delete one or more plates and all their measurements
   * - ``cellview clean``
     - Remove orphaned records from the database
   * - ``cellview explore <plate_ids...>``
     - Launch a pre-populated Jupyter notebook for interactive analysis
   * - ``cellview explore --experiment <name or id>``
     - Launch a notebook for all plates in an experiment
   * - ``cellview --db <path> <command>``
     - Run any command against a custom database file
