"""Module for the CellView database.

This module provides a class for managing the CellView database.
"""

import os
from pathlib import Path

import duckdb

from cellview.utils.error_classes import DBError
from cellview.utils.ui import CellViewUI
from omero_screen.config import get_logger

logger = get_logger(__name__)


class CellViewDB:
    """Class for managing the CellView database.

    Attributes:
        db_path: The path to the database file
        conn: The database connection
        logger: The logger
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the CellViewDB instance.

        Args:
            db_path: The path to the database file
        """
        self.ui = CellViewUI()
        if db_path is None:
            if os.getenv("TEST_DATABASE") == "true":
                self.db_path = (
                    Path.home() / ".cellview" / "cellview-test.duckdb"
                )
            else:
                self.db_path = Path.home() / ".cellview" / "cellview.duckdb"
        else:
            self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: duckdb.DuckDBPyConnection | None = None
        self.logger = logger

    def _is_initialized(self) -> bool:
        """Check if the database has been initialized with tables.

        Returns:
            True if the database has been initialized, False otherwise
        """
        if not self.conn:
            return False

        conn = self.conn
        try:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            return len(result) > 0
        except duckdb.Error:
            return False

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Connect to the database and initialize schema if needed.

        Returns:
            The database connection
        """
        if self.conn:
            return self.conn
        try:
            is_new_db = not self.db_path.exists()
            self.conn = duckdb.connect(str(self.db_path))

            # Report connection status based on whether database is new or existing
            if is_new_db:
                self.ui.progress(f"Creating new database at {self.db_path}")
            else:
                self.ui.info(
                    f"Connected to existing database at {self.db_path}"
                )

            # Initialize schema if this is a new database
            if is_new_db or not self._is_initialized():
                self.create_tables()
                self.ui.success("Database schema initialized successfully")
            else:
                self.ensure_templates_table()
                self.ensure_nucleus_channel_column()
                self.ensure_stitch_mode_column()

            return self.conn
        except duckdb.Error as err:
            # Use error notification for connection failures
            self.ui.error(f"Failed to connect to database at {self.db_path}")
            raise DBError(
                "Failed to connect to database",
                context={
                    "db_path": str(self.db_path),
                    "error": str(err),
                },
            ) from err

    def ensure_templates_table(self) -> None:
        """Add the templates table to an existing database if absent.

        Safe to call on any database version — does nothing if the table
        already exists.
        """
        conn = self.conn
        if conn is None:
            return
        try:
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='templates'"
            )
            if conn.fetchone():
                return  # already present

            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS template_id_seq START 1"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    template_id   INTEGER PRIMARY KEY DEFAULT nextval('template_id_seq'),
                    name          TEXT NOT NULL UNIQUE,
                    description   TEXT,
                    format        TEXT NOT NULL DEFAULT 'jupyter',
                    path          TEXT NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    parent_template_id INTEGER REFERENCES templates(template_id)
                )
            """)
            self.logger.info(
                "Migrated: added templates table to existing database"
            )
        except duckdb.Error as err:
            self.logger.warning(f"Could not ensure templates table: {err}")

    def ensure_stitch_mode_column(self) -> None:
        """Add ``stitch_mode`` to ``repeats`` if absent (legacy DBs).

        Marks whether a repeat was analysed in stitched-well mode
        (one stitched canvas per well, segmented as a unit) vs the
        legacy per-field mode. Legacy rows default to ``false``.
        Safe to call repeatedly.
        """
        conn = self.conn
        if conn is None:
            return
        try:
            cols = conn.execute("PRAGMA table_info(repeats)").fetchall()
            if any(row[1] == "stitch_mode" for row in cols):
                return  # already migrated
            conn.execute(
                "ALTER TABLE repeats ADD COLUMN stitch_mode BOOLEAN "
                "DEFAULT false"
            )
            self.logger.info(
                "Migrated: added stitch_mode column to repeats "
                "(legacy rows default to false)"
            )
        except duckdb.Error as err:
            self.logger.warning(f"Could not ensure stitch_mode column: {err}")

    def ensure_nucleus_channel_column(self) -> None:
        """Add ``nucleus_channel`` to ``repeats`` if absent (legacy DBs).

        Pre-refactor DBs had no concept of nucleus_channel — every plate was
        assumed to be DAPI because the upstream pipeline renamed the nucleus
        channel before export. Defaulting legacy rows to ``'DAPI'`` therefore
        preserves correct semantics for existing data. Safe to call repeatedly.
        """
        conn = self.conn
        if conn is None:
            return
        try:
            cols = conn.execute("PRAGMA table_info(repeats)").fetchall()
            if any(row[1] == "nucleus_channel" for row in cols):
                return  # already migrated
            # DuckDB does not support adding NOT NULL columns via ALTER —
            # use the DEFAULT to backfill every existing row with 'DAPI',
            # then enforce non-null at the application layer (the importer
            # always supplies a value).
            conn.execute(
                "ALTER TABLE repeats ADD COLUMN nucleus_channel TEXT "
                "DEFAULT 'DAPI'"
            )
            self.logger.info(
                "Migrated: added nucleus_channel column to repeats "
                "(legacy rows default to 'DAPI')"
            )
        except duckdb.Error as err:
            self.logger.warning(
                f"Could not ensure nucleus_channel column: {err}"
            )

    def create_tables(self) -> None:
        """Create the database schema.

        Raises:
            DBError: If the database schema creation fails
        """
        try:
            conn = self.connect()

            # Start with a progress message
            self.ui.progress("Creating database tables")

            # Drop existing tables and sequences
            conn.execute("DROP TABLE IF EXISTS measurements")
            conn.execute("DROP TABLE IF EXISTS condition_variables")
            conn.execute("DROP TABLE IF EXISTS conditions")
            conn.execute("DROP TABLE IF EXISTS repeats")
            conn.execute("DROP TABLE IF EXISTS experiments")
            conn.execute("DROP TABLE IF EXISTS projects")
            conn.execute("DROP TABLE IF EXISTS templates")
            conn.execute("DROP SEQUENCE IF EXISTS project_id_seq")
            conn.execute("DROP SEQUENCE IF EXISTS experiment_id_seq")
            conn.execute("DROP SEQUENCE IF EXISTS repeat_id_seq")
            conn.execute("DROP SEQUENCE IF EXISTS condition_id_seq")
            conn.execute("DROP SEQUENCE IF EXISTS variable_id_seq")
            conn.execute("DROP SEQUENCE IF EXISTS measurement_id_seq")
            conn.execute("DROP SEQUENCE IF EXISTS template_id_seq")

            # Create sequences
            conn.execute("CREATE SEQUENCE project_id_seq START 1")
            conn.execute("CREATE SEQUENCE experiment_id_seq START 1")
            conn.execute("CREATE SEQUENCE repeat_id_seq START 1")
            conn.execute("CREATE SEQUENCE condition_id_seq START 1")
            conn.execute("CREATE SEQUENCE variable_id_seq START 1")
            conn.execute("CREATE SEQUENCE measurement_id_seq START 1")
            conn.execute("CREATE SEQUENCE template_id_seq START 1")

            # Create tables with auto-incrementing primary keys
            conn.execute("""
            CREATE TABLE projects (
                project_id INTEGER PRIMARY KEY DEFAULT nextval('project_id_seq'),
                project_name TEXT NOT NULL UNIQUE,
                description TEXT
            );

            CREATE TABLE experiments (
                experiment_id INTEGER PRIMARY KEY DEFAULT nextval('experiment_id_seq'),
                project_id INTEGER REFERENCES projects(project_id),
                experiment_name TEXT NOT NULL,
                description TEXT
            );

            CREATE TABLE repeats (
                repeat_id INTEGER PRIMARY KEY DEFAULT nextval('repeat_id_seq'),
                experiment_id INTEGER REFERENCES experiments(experiment_id),
                plate_id INTEGER,
                date DATE NOT NULL,
                lab_member TEXT,
                channel_0 TEXT NOT NULL,
                channel_1 TEXT,
                channel_2 TEXT,
                channel_3 TEXT,
                classifier TEXT,
                nucleus_channel TEXT NOT NULL DEFAULT 'DAPI',
                stitch_mode BOOLEAN NOT NULL DEFAULT false
            );

            CREATE TABLE conditions (
                condition_id INTEGER PRIMARY KEY DEFAULT nextval('condition_id_seq'),
                repeat_id INTEGER REFERENCES repeats(repeat_id),
                well TEXT NOT NULL,
                well_id TEXT NOT NULL,
                cell_line TEXT NOT NULL,
                antibody TEXT,
                antibody_1 TEXT,
                antibody_2 TEXT,
                antibody_3 TEXT,
                UNIQUE (repeat_id, well)
            );

            CREATE TABLE condition_variables (
                variable_id INTEGER PRIMARY KEY DEFAULT nextval('variable_id_seq'),
                condition_id INTEGER REFERENCES conditions(condition_id),
                variable_name TEXT NOT NULL,
                variable_value TEXT NOT NULL
            );

            CREATE TABLE templates (
                template_id   INTEGER PRIMARY KEY DEFAULT nextval('template_id_seq'),
                name          TEXT NOT NULL UNIQUE,
                description   TEXT,
                format        TEXT NOT NULL DEFAULT 'jupyter',
                path          TEXT NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                parent_template_id INTEGER REFERENCES templates(template_id)
            );

            CREATE TABLE measurements (
                measurement_id INTEGER PRIMARY KEY DEFAULT nextval('measurement_id_seq'),
                condition_id INTEGER REFERENCES conditions(condition_id),
                image_id INTEGER NOT NULL,
                timepoint INTEGER NOT NULL,
                cell_cycle TEXT,
                cell_cycle_detailed TEXT,

                label VARCHAR NOT NULL,
                area_nucleus FLOAT NOT NULL,
                "centroid-0-nuc" FLOAT NOT NULL,
                "centroid-1-nuc" FLOAT NOT NULL,
                intensity_min_DAPI_nucleus FLOAT NOT NULL,
                intensity_mean_DAPI_nucleus FLOAT NOT NULL,
                intensity_max_DAPI_nucleus FLOAT NOT NULL,
                integrated_int_DAPI_norm FLOAT,

                Cyto_ID INTEGER,
                area_cell FLOAT,
                "centroid-0-cell" FLOAT,
                "centroid-1-cell" FLOAT,

                intensity_min_DAPI_cell FLOAT,
                intensity_mean_DAPI_cell FLOAT,
                intensity_max_DAPI_cell FLOAT,

                area_cyto FLOAT,

                intensity_min_DAPI_cyto FLOAT,
                intensity_mean_DAPI_cyto FLOAT,
                intensity_max_DAPI_cyto FLOAT
            );
            """)

            # Show success message when tables are created
            self.ui.success("Tables created successfully")
        except duckdb.Error as err:
            self.ui.error(f"Failed to create database schema: {str(err)}")
            raise DBError(
                "Failed to create database schema",
                context={
                    "db_path": str(self.db_path),
                    "error": str(err),
                },
            ) from err
