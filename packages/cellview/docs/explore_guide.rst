Exploring Data with ``cellview explore``
=========================================

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

``cellview explore`` is the quickest way to go from a plate ID to a working
analysis notebook. It copies a pre-built template notebook into your personal
explore library (``~/.cellview/explore/``), injects the plate IDs you care
about, and opens the notebook in JupyterLab or VS Code — ready to run.

You do not need to know where the data lives, how to connect to the database,
or how to write loading code. CellView handles all of that; you start with
data already in a ``DataFrame``.

Launching the Explorer
----------------------

By plate ID
~~~~~~~~~~~

If you know which plate (or plates) you want to look at, pass the IDs directly:

.. code-block:: bash

   cellview explore 12345
   cellview explore 12345 12346 12347

When you provide multiple plate IDs, CellView creates a single combined
notebook that loads all of them together. The notebook is saved at:

.. code-block:: text

   ~/.cellview/explore/plates/12345/explore_plate_12345.ipynb

(For multiple plates the directory is named after the first ID in the list.)

JupyterLab opens automatically at that notebook.

By experiment
~~~~~~~~~~~~~

If you want to load all plates belonging to an experiment, use
``--experiment``. You can identify the experiment by name or by its numeric ID:

.. code-block:: bash

   cellview explore --experiment palb_washout
   cellview explore --experiment 6

The notebook is placed in a folder that mirrors the project/experiment
hierarchy in the database:

.. code-block:: text

   ~/.cellview/explore/MyProject/palb_washout/explore_exp_6.ipynb

.. note::

   Use ``cellview display experiments`` to see the names and IDs of all
   experiments currently in your database.

Opening in VS Code instead of JupyterLab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``--code`` to open the notebook in VS Code rather than JupyterLab:

.. code-block:: bash

   cellview explore 12345 --code

VS Code opens with the entire ``~/.cellview/explore/`` library as the workspace
(so you can see all your notebooks in the sidebar) and jumps directly to the
new notebook. This works particularly well with AI coding assistants such as
Claude Code or GitHub Copilot, because a ``CLAUDE.md`` file is placed in the
explore library automatically — it documents the full plotting API and is read
by the assistant the moment it opens the workspace.

To make VS Code your default editor without typing ``--code`` every time, set
the environment variable in your shell profile:

.. code-block:: bash

   export CELLVIEW_EDITOR=vscode

Available options
~~~~~~~~~~~~~~~~~

.. option:: --experiment NAME_OR_ID

   Load all plates from a named experiment or numeric experiment ID instead of
   providing individual plate IDs.

.. option:: --template NAME

   Choose which template to use for the notebook. Defaults to ``cellcycle``.
   See `Managing Templates`_ below for how to list and add templates.

.. option:: --fresh

   Regenerate the notebook even if one already exists for this plate or
   experiment. The old notebook is overwritten.

.. option:: --no-napari

   Skip launching the Napari image viewer in the background. By default,
   Napari is launched alongside the notebook so you can inspect raw images
   while you analyse data.

.. option:: --code

   Open the explore library in VS Code instead of JupyterLab.

What Happens When You Run ``explore``
--------------------------------------

Here is what CellView does behind the scenes each time you run the command:

1. **Looks up plates.** If you used ``--experiment``, CellView queries the
   database for all plate IDs that belong to that experiment. If you gave plate
   IDs directly, it uses those.

2. **Checks for an existing notebook.** If a notebook for this exact set of
   plates or this experiment already exists in the explore library, CellView
   reuses it. This means you can close the editor, come back later, and run
   the same command to reopen exactly where you left off. Use ``--fresh`` to
   override this behaviour.

3. **Copies and patches the template.** The chosen template is copied to the
   correct subfolder in ``~/.cellview/explore/`` and the ``PLATE_IDS`` line is
   replaced with the actual IDs.

4. **Migrates legacy notebooks (if needed).** Older flat-layout notebooks
   (created before the project/experiment subfolder structure was introduced)
   are moved into the new layout automatically.

5. **Launches the editor.** JupyterLab or VS Code opens with the notebook
   ready to run.

6. **Optionally launches Napari.** Unless you passed ``--no-napari``, Napari
   starts in the background so you can view raw microscopy images alongside
   your analysis.

The Explore Library and ``CLAUDE.md``
--------------------------------------

``~/.cellview/explore/`` is your personal analysis library. Every notebook you
create with ``cellview explore`` is stored here, organised into subfolders by
project and experiment (or by plate ID). Nothing is deleted automatically —
old notebooks accumulate over time and you can return to any of them at any
point.

A ``CLAUDE.md`` file is placed in this directory when it is first created. It
contains complete documentation of:

- ``cellview_load_data`` and its arguments
- All plot functions from ``omero_screen_plots`` with their parameters
- Common patterns for filtering and grouping data

When you open the explore library in VS Code with ``--code``, any AI coding
assistant that supports project context files (Claude Code, GitHub Copilot,
Cursor, etc.) reads this file automatically. The assistant then knows the
entire CellView API without you needing to explain it — you can ask questions
like "plot the cell cycle distribution for RPE cells only" and get working
code immediately.

.. note::

   The ``CLAUDE.md`` file is updated when you upgrade CellView. If you
   customise it, your changes may be overwritten on upgrade. Keep personal
   notes in a separate file in the same directory.

Notebook Structure
------------------

A freshly created notebook from the default ``cellcycle`` template contains
the following cells, in order:

.. code-block:: text

   [Markdown]  # Explore — Plate 12345
   [Code]      PLATE_IDS = [12345]
   [Code]      from cellview.api import cellview_load_data
   [Code]      df, variable_names = cellview_load_data(*PLATE_IDS)
   [Code]      ... template-specific analysis cells ...

The ``PLATE_IDS`` cell is the only part of the template that CellView modifies.
Everything else comes from the template as-is, so what you see next depends on
which template you chose.

The default ``cellcycle`` template adds cells that produce:

- A summary table of cell counts per condition
- A combined cell cycle plot (stacked bar + phase quantification)
- A feature comparison plot for nucleus area

You can edit, delete, or add cells freely — the notebook is yours.

The ``--fresh`` Flag
--------------------

By default, running ``cellview explore 12345`` a second time simply reopens
the existing notebook. This is intentional: it lets you accumulate your own
analysis in the notebook without losing it.

Use ``--fresh`` when you want to start again from a clean template:

.. code-block:: bash

   cellview explore 12345 --fresh

This is also useful after a CellView upgrade that ships an improved template —
``--fresh`` gives you the new version.

.. note::

   ``--fresh`` overwrites the existing notebook. If you want to keep your
   previous work, rename or copy the file before running ``--fresh``.

Using a Marimo Notebook
-----------------------

CellView templates can be Marimo reactive notebooks (``*.py`` files) as well
as standard Jupyter notebooks. Marimo notebooks re-run cells automatically
when their inputs change, which can be useful for interactive exploration.

If the selected template is a Marimo file, CellView opens it with:

.. code-block:: bash

   marimo edit explore_plate_12345.py

Everything else works the same way — the ``PLATE_IDS`` line is patched, the
file is placed in the correct subfolder, and the editor is launched for you.

.. note::

   Marimo must be installed separately. Run ``uv add marimo`` (or
   ``pip install marimo``) if you see a "command not found" error.

Managing Templates
------------------

Listing templates
~~~~~~~~~~~~~~~~~

To see all templates that CellView knows about, including their format and a
short description:

.. code-block:: bash

   cellview template list

Using a different template
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``--template`` with the template name:

.. code-block:: bash

   cellview explore 12345 --template drug_screen

Adding your own template
~~~~~~~~~~~~~~~~~~~~~~~~

You can register a notebook you have already created as a named template:

.. code-block:: bash

   cellview template add ~/my_analysis.ipynb --name drug_screen \
       --description "Dose-response analysis"

Alternatively, drop the file directly into ``~/.cellview/templates/``. CellView
checks this directory in addition to its built-in templates, and user templates
take priority when names collide. After dropping a file there, register it with:

.. code-block:: bash

   cellview template sync

Template conventions
~~~~~~~~~~~~~~~~~~~~

For CellView to inject plate IDs correctly, your template must contain exactly
this line somewhere near the top:

.. code-block:: python

   PLATE_IDS: list[int] = []

CellView replaces the right-hand side with the actual list of plate IDs at
launch time. Everything else in the file is left untouched.

The first markdown cell in the notebook is used as the short description shown
by ``cellview template list``.

Syncing templates to the database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you add template files to ``~/.cellview/templates/`` manually (without using
``cellview template add``), run:

.. code-block:: bash

   cellview template sync

This scans both the built-in package templates and your personal template
directory, and registers any new files in the database.

Removing a template record
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cellview template remove drug_screen

This removes the database record for the named template. The file itself is
**not** deleted — only the registration entry is removed.

Showing template details
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cellview template show drug_screen

Displays the full path, format, description, and registration date for a
specific template.

Example: Creating Your Own Template
------------------------------------

The following walkthrough shows how to create a reusable template from
scratch for a dose-response experiment.

**Step 1 — Create the templates directory** (if it does not already exist):

.. code-block:: bash

   mkdir -p ~/.cellview/templates

**Step 2 — Start from an existing notebook or create a new one.**

If you already have a working notebook that you want to reuse, copy it:

.. code-block:: bash

   cp ~/notebooks/dose_response_analysis.ipynb \
       ~/.cellview/templates/dose_response.ipynb

Or open JupyterLab and create a new notebook in ``~/.cellview/templates/``.

**Step 3 — Add the required** ``PLATE_IDS`` **line** near the top of the
notebook, in its own code cell:

.. code-block:: python

   PLATE_IDS: list[int] = []

Make sure this line appears exactly as shown — CellView uses it as a target
for injection.

**Step 4 — Add a markdown description** as the first cell of the notebook.
This text is shown by ``cellview template list``:

.. code-block:: text

   # Dose-Response Analysis
   Analyse viability and cell cycle distribution across drug concentrations.

**Step 5 — Save the file** as ``~/.cellview/templates/dose_response.ipynb``.

**Step 6 — Test it:**

.. code-block:: bash

   cellview explore 12345 --template dose_response

CellView will copy the template, inject the plate IDs, and open the notebook.
If everything looks correct, register it properly so it appears in the list:

.. code-block:: bash

   cellview template add ~/.cellview/templates/dose_response.ipynb \
       --name dose_response \
       --description "Dose-response viability and cell cycle analysis"
