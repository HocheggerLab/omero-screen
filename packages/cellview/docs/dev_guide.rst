Developer Guide
===============

This guide is for anyone who wants to understand, extend, or contribute to the
CellView package. It explains the overall architecture, the key design patterns,
and how the main subsystems fit together.

.. contents:: On this page
   :local:
   :depth: 2


Architecture Overview
----------------------

CellView is organised into a small set of focused modules. The diagram below
shows the top-level layout and what each file is responsible for.

.. code-block:: text

   cellview/
   ├── cli.py            argparse parser — subcommand definitions
   ├── main.py           dispatcher — routes subcommands to handlers
   ├── api.py            public Python API (cellview_load_data)
   ├── db/
   │   ├── db.py         CellViewDB — connection, schema creation, migrations
   │   ├── display.py    Rich table display for all browse commands
   │   ├── edit.py       edit_project / edit_experiment
   │   ├── clean_up.py   orphan removal, plate deletion
   │   └── templates.py  CRUD for the templates registry table
   ├── importers/        CSV and OMERO import pipeline
   ├── exporters/        pandas / polars export
   ├── explore/          notebook launch, template registry, JSON context
   └── utils/            state, UI, error classes

The entry point for the CLI is ``main.py``. The public Python API (used from
notebooks and scripts) lives in ``api.py`` and calls into the same exporters
that the CLI uses. There is deliberately no business logic in ``cli.py`` — it
only defines argument parsers; all logic lives in ``main.py`` and the modules
below it.


Database Layer (db/)
--------------------

**CellViewDB**

``CellViewDB`` (``db/db.py``) wraps a DuckDB connection. It is opened once per
CLI invocation and passed by reference to every handler function. This means
there is a single connection per process — no connection pooling is needed
because DuckDB is an embedded database.

On first run, ``CellViewDB`` creates the full schema. On subsequent runs it
calls ``ensure_templates_table()`` to apply any schema migrations idempotently.
This approach means existing databases are never broken by upgrades that add new
tables.

**Database hierarchy**

Data is stored in a five-level hierarchy:

.. code-block:: text

   projects
     └── experiments
           └── repeats  (one per imported plate)
                 └── conditions  (one per well)
                       ├── condition_variables  (key-value metadata rows)
                       └── measurements  (one row per segmented cell)

``condition_variables`` stores flexible experimental metadata (e.g.
``Drug = "DMSO"``, ``Concentration = "10uM"``) as key-value rows rather than
fixed columns. On export, ``PlateParser`` pivots these back into regular
DataFrame columns and returns their names as ``variable_names``.

**Other db/ modules**

- ``display.py`` — ``Rich``-formatted table output for ``cellview display``
  commands. Each display function takes a ``CellViewDB`` and returns nothing
  (it prints directly).
- ``edit.py`` — interactive project and experiment renaming via ``Rich``
  prompts.
- ``clean_up.py`` — orphan record removal (conditions with no measurements,
  repeats with no conditions, etc.) and full plate deletion.
- ``templates.py`` — CRUD operations for the ``templates`` table used by the
  explore system.


Dependency Injection Pattern
-----------------------------

CellView uses dependency injection throughout. ``CellViewDB`` is created in
``main()`` and passed down to every handler function rather than being accessed
via a global singleton. This keeps each function testable in isolation — a test
can pass a real in-memory DuckDB connection and the function under test behaves
identically to production.

.. code-block:: python

   # main.py — simplified
   def main_with_dependency_injection(args, conn: CellViewDB) -> None:
       match args.command:
           case "display":
               handle_display(args, conn)
           case "import":
               handle_import(args, conn)
           # ...

   def main() -> None:
       args = get_parser().parse_args()
       db_path = args.db or default_db_path()
       with CellViewDB(db_path) as conn:
           main_with_dependency_injection(args, conn)

The ``api.py`` function ``cellview_load_data_with_injection`` follows the same
pattern — it accepts an optional ``conn`` argument so tests can pass a
pre-configured in-memory database without touching the filesystem.


Import Pipeline (importers/)
-----------------------------

When a user runs ``cellview import csv <file>`` or ``cellview import plate
<plate_id>``, the following sequence runs:

1. **ProjectManager.select_or_create_project()** — presents a Rich prompt
   listing existing projects; the user either picks one or creates a new one.
2. **ExperimentManager.select_or_create_experiment()** — same pattern within
   the chosen project.
3. **RepeatsManager** — creates the plate record (a *repeat* in database
   terms) and stores channel metadata.
4. **import_conditions()** — iterates over wells in the CSV or OMERO plate,
   creates one ``conditions`` row per well, and extracts flexible
   ``condition_variables`` key-value pairs.
5. **import_measurements()** — bulk-inserts single-cell measurement rows into
   the ``measurements`` table.

The CSV importer expects specific columns (``plate_id``, ``cell_line``,
``condition``, and all measurement columns). The OMERO importer downloads the
final-data CSV attached to the OMERO plate object and then feeds it through the
same CSV path.


Export Pipeline (exporters/)
-----------------------------

``PlateParser`` (``exporters/plate_queries.py``) is the shared query engine
used by both the pandas exporter (``db_to_pandas.py``) and the polars exporter
(``db_to_polars.py``). It runs a single join across:

.. code-block:: text

   repeats → conditions → condition_variables → measurements

It pivots ``condition_variables`` key-value rows back into columns using
DuckDB's ``PIVOT`` statement, then returns a tuple of ``(DataFrame,
variable_names)``. The pandas and polars exporters are thin wrappers that call
``PlateParser`` and return the result in the requested format.


Explore System (explore/)
--------------------------

The explore system launches pre-configured Jupyter notebooks for interactive
data analysis. It consists of four modules:

- ``_registry.py`` — computes the notebook output directory (``EXPLORE_DIR``)
  and subdirectory logic based on the plate or experiment being explored.
- ``_template_registry.py`` — discovers ``.ipynb`` and ``.py`` templates,
  syncs them to the ``templates`` database table, and injects ``PLATE_IDS``
  into notebooks before launching.
- ``_cli.py`` — the ``launch_explore()`` orchestrator: resolves the template,
  injects plate IDs, opens the notebook in JupyterLab (or another editor via
  ``_open_editor()``), and calls ``_ensure_claude_md()`` to copy the AI
  context file on first run.
- ``_explore_json.py`` — ``explore_json()`` produces a JSON snapshot of the
  current database state for use by agentic workflows (e.g. AI assistants that
  need to know which plates and experiments exist).

**Templates**

Built-in templates live in ``explore/templates/`` and are packaged with
CellView. User-defined templates can be placed in ``~/.cellview/templates/``.
On first run of ``cellview explore``, a ``CLAUDE.md`` file is copied from
``explore/templates/`` into ``EXPLORE_DIR``. This file gives AI coding
assistants the plotting API context automatically, so they can write correct
``omero-screen-plots`` calls without needing to search the documentation.


Templates Table (db/templates.py)
----------------------------------

The ``templates`` table was added after the initial schema, so it is created by
``ensure_templates_table()`` in ``CellViewDB`` rather than in the main schema
creation block. This function is safe to call on every connection — it uses
``CREATE TABLE IF NOT EXISTS`` and is therefore idempotent.

The public interface consists of four functions:

- ``upsert_template(conn, record)`` — insert or update a template record.
- ``get_template_record(conn, name)`` — retrieve a single ``TemplateRecord``
  by name.
- ``list_template_records(conn)`` — return all registered templates.
- ``delete_template(conn, name)`` — remove a template by name.

``TemplateRecord`` is a plain Python dataclass with fields ``name``, ``path``,
``description``, and ``last_synced``.


Error Classes (utils/error_classes.py)
---------------------------------------

CellView defines a small exception hierarchy so callers can catch errors at the
right level of specificity:

.. code-block:: text

   CellViewError          (base — catch this to handle any CellView error)
     ├── DBError          (database connection or query failure)
     ├── DataError        (invalid or missing data in an import or export)
     └── StateError       (invalid application state, e.g. missing config)

Always raise one of these rather than a bare ``Exception`` or ``ValueError``.
This makes it straightforward for callers (including tests) to assert on the
specific error type.


Testing
--------

Unit tests live in ``tests/unit_tests/cellview_tests/``. Each subdirectory
must contain an ``__init__.py`` to avoid pytest module name collisions when
multiple test files share the same filename.

Run the tests with:

.. code-block:: text

   pytest tests/unit_tests/cellview_tests/ -v

**Key testing patterns**

*In-memory DuckDB*: Pass ``duckdb.connect()`` (no path argument) to get a
fresh in-memory database. This is fast and leaves no files on disk.

*FK constraint fixture*: DuckDB enforces foreign key constraints strictly. When
testing orphan-removal functions you need to insert rows that would normally
violate FK constraints. Use a ``db_no_fk`` fixture that creates the schema
without ``REFERENCES`` clauses and opens the connection via raw
``duckdb.connect()``.

*Mocking DuckDB connections*: DuckDB connection objects are C extension
objects and are read-only — ``patch.object(conn, "execute", ...)`` will raise
an ``AttributeError``. The workaround is a thin ``ConnWrapper`` class that
delegates all attribute access to the real connection via ``__getattr__`` but
overrides ``execute()`` so you can intercept calls in tests.

.. code-block:: python

   class ConnWrapper:
       def __init__(self, real_conn):
           self._conn = real_conn
           self.execute_calls = []

       def __getattr__(self, name):
           return getattr(self._conn, name)

       def execute(self, sql, params=None):
           self.execute_calls.append((sql, params))
           return self._conn.execute(sql, params)


Adding a New CLI Subcommand
----------------------------

Follow these four steps to add a new subcommand (e.g. ``cellview mycommand``):

1. **Define the argument parser** in ``cli.py`` inside ``get_parser()``.
   Add a subparser entry and any arguments the command needs.

   .. code-block:: python

      mycommand_parser = subparsers.add_parser(
          "mycommand", help="Brief description of what mycommand does."
      )
      mycommand_parser.add_argument("--option", type=str, help="An option.")

2. **Write a handler function** in ``main.py``. The handler receives
   ``args`` and ``conn: CellViewDB`` and contains all the business logic.

   .. code-block:: python

      def handle_mycommand(args: argparse.Namespace, conn: CellViewDB) -> None:
          result = some_db_query(conn, args.option)
          print_result(result)

3. **Add a dispatch case** in ``main_with_dependency_injection()``:

   .. code-block:: python

      case "mycommand":
          handle_mycommand(args, conn)

4. **Write tests** in ``tests/unit_tests/cellview_tests/test_main.py``.
   Test the parser separately (no mocking needed) and the handler with a
   mocked or in-memory ``CellViewDB``.

.. note::

   Keep ``cli.py`` free of business logic. Its only job is to define parsers
   and return the ``argparse.Namespace``. All logic belongs in ``main.py`` or
   the module the command delegates to (``db/``, ``importers/``, etc.). This
   separation makes the parser trivially testable without any mocking.
