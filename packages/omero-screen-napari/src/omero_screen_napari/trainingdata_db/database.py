"""Core database interface for training data storage.

This module provides the main TrainingDB class for interacting with the
training data database, including CRUD operations for classifiers, sessions,
and annotations.
"""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, cast

from omero_screen.config import get_logger

from .schema import (
    CREATE_INDEXES_SQL,
    CREATE_TABLES_SQL,
    QUERIES,
    SCHEMA_VERSION,
    get_filter_query,
)

logger = get_logger(__name__)


class TrainingDB:
    """Main interface to training data database.

    This class provides methods for managing classifiers, annotation sessions,
    and individual cell annotations. It handles database initialization,
    connection management, and all CRUD operations.

    Attributes:
        db_path: Path to the SQLite database file
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to database file. If None, uses default location.
                    Default location depends on TEST_DATABASE environment variable:
                    - If TEST_DATABASE=true: ~/omeroscreen_trainingdata/training_data_test.db
                    - Otherwise: ~/omeroscreen_trainingdata/training_data.db
                    Can also be overridden with OMERO_SCREEN_TRAINING_DB env var.
        """
        if db_path is None:
            # Check for environment variable override
            env_path = os.environ.get("OMERO_SCREEN_TRAINING_DB")
            if env_path:
                db_path = Path(env_path)
            elif os.getenv("TEST_DATABASE") == "true":
                # Use test database for development/testing
                db_path = (
                    Path.home()
                    / "omeroscreen_trainingdata"
                    / "training_data_test.db"
                )
            else:
                # Use production database
                db_path = (
                    Path.home()
                    / "omeroscreen_trainingdata"
                    / "training_data.db"
                )

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database if needed
        self._initialize_database()
        logger.info(f"Training database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections.

        Yields:
            sqlite3.Connection: Database connection with row factory
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_database(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._get_connection() as conn:
            # Create tables
            conn.executescript(CREATE_TABLES_SQL)
            # Create indexes
            conn.executescript(CREATE_INDEXES_SQL)

            # Check/set schema version
            cursor = conn.execute(QUERIES["get_schema_version"])
            result = cursor.fetchone()

            if result is None:
                # First initialization
                conn.execute(
                    QUERIES["insert_schema_version"], (SCHEMA_VERSION,)
                )
                logger.info(
                    f"Database initialized with schema version {SCHEMA_VERSION}"
                )
            elif result[0] < SCHEMA_VERSION:
                self._migrate_schema(conn, result[0])
            elif result[0] != SCHEMA_VERSION:
                logger.warning(
                    f"Schema version mismatch: DB={result[0]}, Code={SCHEMA_VERSION}"
                )

    def _migrate_schema(
        self, conn: sqlite3.Connection, current_version: int
    ) -> None:
        """Apply incremental schema migrations."""
        if current_version < 2:
            # v2: add centroid coordinates and source image ID to annotations
            for col, col_type in [
                ("centroid_row", "INTEGER"),
                ("centroid_col", "INTEGER"),
                ("source_image_id", "INTEGER"),
            ]:
                with suppress(sqlite3.OperationalError):
                    conn.execute(
                        f"ALTER TABLE annotations ADD COLUMN {col} {col_type}"
                    )
            conn.execute(QUERIES["insert_schema_version"], (2,))
            logger.info("Migrated database schema to version 2")

        if current_version < 3:
            # v3: drop UNIQUE(classifier_id, plate_id, well, image_id, timepoint)
            # from annotation_sessions so multiple sessions per well are allowed.
            # SQLite cannot DROP UNIQUE constraints; recreate the table.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotation_sessions_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    classifier_id INTEGER NOT NULL,
                    plate_id INTEGER NOT NULL,
                    well TEXT NOT NULL,
                    image_id INTEGER NOT NULL,
                    timepoint INTEGER DEFAULT 0,
                    user TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (classifier_id) REFERENCES classifiers(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                INSERT INTO annotation_sessions_new
                SELECT id, classifier_id, plate_id, well, image_id, timepoint,
                       user, created_at, updated_at, file_path, metadata_json
                FROM annotation_sessions
            """)
            conn.execute("DROP TABLE annotation_sessions")
            conn.execute(
                "ALTER TABLE annotation_sessions_new RENAME TO annotation_sessions"
            )
            conn.execute(QUERIES["insert_schema_version"], (3,))
            logger.info("Migrated database schema to version 3")

    # ========== Classifier Operations ==========

    def create_classifier(self, name: str, description: str = "") -> int:
        """Create a new classifier.

        Args:
            name: Unique classifier name
            description: Optional description

        Returns:
            Classifier ID

        Raises:
            sqlite3.IntegrityError: If classifier name already exists
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["insert_classifier"], (name, description)
            )
            classifier_id = cursor.lastrowid
            if classifier_id is None:
                raise RuntimeError("Failed to create classifier")
            logger.info(f"Created classifier '{name}' with ID {classifier_id}")
            return classifier_id

    def get_classifier(self, name: str) -> dict[str, Any] | None:
        """Get classifier by name.

        Args:
            name: Classifier name

        Returns:
            Dictionary with classifier info or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(QUERIES["get_classifier_by_name"], (name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_classifier_by_id(
        self, classifier_id: int
    ) -> dict[str, Any] | None:
        """Get classifier by ID.

        Args:
            classifier_id: Classifier ID

        Returns:
            Dictionary with classifier info or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["get_classifier_by_id"], (classifier_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_classifiers(self) -> list[dict[str, Any]]:
        """List all classifiers.

        Returns:
            List of classifier dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.execute(QUERIES["list_classifiers"])
            return [dict(row) for row in cursor.fetchall()]

    def delete_classifier(self, name: str) -> bool:
        """Delete a classifier and all associated data.

        Args:
            name: Classifier name

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(QUERIES["delete_classifier"], (name,))
            deleted = bool(cursor.rowcount > 0)
            if deleted:
                logger.info(
                    f"Deleted classifier '{name}' and all associated data"
                )
            return deleted

    # ========== Class Operations ==========

    def add_class(
        self, classifier_name: str, label: str, description: str = ""
    ) -> None:
        """Add a class label to a classifier.

        Args:
            classifier_name: Classifier name
            label: Class label
            description: Optional description

        Raises:
            ValueError: If classifier doesn't exist
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        with self._get_connection() as conn:
            conn.execute(
                QUERIES["insert_class"],
                (classifier["id"], label, description),
            )
            logger.debug(
                f"Added class '{label}' to classifier '{classifier_name}'"
            )

    def add_classes(
        self,
        classifier_name: str,
        labels: list[str],
        descriptions: list[str] | None = None,
    ) -> None:
        """Add multiple class labels to a classifier.

        Args:
            classifier_name: Classifier name
            labels: List of class labels
            descriptions: Optional list of descriptions (same length as labels)

        Raises:
            ValueError: If classifier doesn't exist or lists have different lengths
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        if descriptions is None:
            descriptions = [""] * len(labels)
        elif len(descriptions) != len(labels):
            raise ValueError(
                "Number of descriptions must match number of labels"
            )

        with self._get_connection() as conn:
            for label, desc in zip(labels, descriptions, strict=True):
                conn.execute(
                    QUERIES["insert_class"],
                    (classifier["id"], label, desc),
                )
            logger.info(
                f"Added {len(labels)} classes to classifier '{classifier_name}'"
            )

    def get_classes(self, classifier_name: str) -> list[str]:
        """Get all class labels for a classifier.

        Args:
            classifier_name: Classifier name

        Returns:
            List of class label strings

        Raises:
            ValueError: If classifier doesn't exist
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["get_classes_by_classifier"], (classifier["id"],)
            )
            return [row["label"] for row in cursor.fetchall()]

    # ========== Session Operations ==========

    def create_session(
        self,
        classifier_name: str,
        plate_id: int,
        well: str,
        image_id: int,
        timepoint: int,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
        user: str | None = None,
    ) -> int:
        """Create a new annotation session.

        Args:
            classifier_name: Classifier name
            plate_id: OMERO plate ID
            well: Well position (e.g., 'A1')
            image_id: OMERO image ID
            timepoint: Timepoint index
            file_path: Path to .npy file
            metadata: Optional metadata dictionary
            user: Optional username (defaults to os.getlogin())

        Returns:
            Session ID

        Raises:
            ValueError: If classifier doesn't exist
            sqlite3.IntegrityError: If session already exists
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        if user is None:
            try:
                user = os.getlogin()
            except OSError:
                user = "unknown"

        metadata_json = json.dumps(metadata) if metadata else None

        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["insert_session"],
                (
                    classifier["id"],
                    plate_id,
                    well,
                    image_id,
                    timepoint,
                    user,
                    str(file_path),
                    metadata_json,
                ),
            )
            session_id = cursor.lastrowid
            if session_id is None:
                raise RuntimeError("Failed to create session")
            logger.info(
                f"Created session {session_id} for {classifier_name}: "
                f"plate={plate_id}, well={well}, image={image_id}, t={timepoint}"
            )
            return session_id

    def get_session(
        self,
        classifier_name: str,
        plate_id: int,
        well: str,
        image_id: int,
        timepoint: int,
    ) -> dict[str, Any] | None:
        """Get annotation session by identifiers.

        Args:
            classifier_name: Classifier name
            plate_id: OMERO plate ID
            well: Well position
            image_id: OMERO image ID
            timepoint: Timepoint index

        Returns:
            Dictionary with session info or None if not found
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            return None

        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["get_session"],
                (classifier["id"], plate_id, well, image_id, timepoint),
            )
            row = cursor.fetchone()
            if row:
                session = dict(row)
                # Parse metadata JSON
                if session["metadata_json"]:
                    session["metadata"] = json.loads(session["metadata_json"])
                else:
                    session["metadata"] = None
                del session["metadata_json"]
                return session
            return None

    def get_session_by_file_path(
        self, file_path: str
    ) -> dict[str, Any] | None:
        """Get annotation session by its NPY file path.

        Args:
            file_path: Absolute path to the NPY file

        Returns:
            Dictionary with session info or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM annotation_sessions WHERE file_path = ?",
                (file_path,),
            )
            row = cursor.fetchone()
            if row:
                session = dict(row)
                if session["metadata_json"]:
                    session["metadata"] = json.loads(session["metadata_json"])
                else:
                    session["metadata"] = None
                del session["metadata_json"]
                return session
            return None

    def get_session_by_id(self, session_id: int) -> dict[str, Any] | None:
        """Get annotation session by ID.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with session info or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(QUERIES["get_session_by_id"], (session_id,))
            row = cursor.fetchone()
            if row:
                session = dict(row)
                # Parse metadata JSON
                if session["metadata_json"]:
                    session["metadata"] = json.loads(session["metadata_json"])
                else:
                    session["metadata"] = None
                del session["metadata_json"]
                return session
            return None

    def list_sessions(self, classifier_name: str) -> list[dict[str, Any]]:
        """List all sessions for a classifier.

        Args:
            classifier_name: Classifier name

        Returns:
            List of session dictionaries

        Raises:
            ValueError: If classifier doesn't exist
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["list_sessions_by_classifier"], (classifier["id"],)
            )
            sessions = []
            for row in cursor.fetchall():
                session = dict(row)
                if session["metadata_json"]:
                    session["metadata"] = json.loads(session["metadata_json"])
                else:
                    session["metadata"] = None
                del session["metadata_json"]
                sessions.append(session)
            return sessions

    def update_session(
        self,
        session_id: int,
        file_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update session file path and/or metadata.

        Args:
            session_id: Session ID
            file_path: New file path (if provided)
            metadata: New metadata (if provided)

        Returns:
            True if updated, False if session not found
        """
        # Get existing session to preserve unchanged fields
        session = self.get_session_by_id(session_id)
        if not session:
            return False

        new_file_path = str(file_path) if file_path else session["file_path"]
        new_metadata = (
            metadata if metadata is not None else session["metadata"]
        )
        metadata_json = json.dumps(new_metadata) if new_metadata else None

        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["update_session"],
                (new_file_path, metadata_json, session_id),
            )
            updated = bool(cursor.rowcount > 0)
            if updated:
                logger.info(f"Updated session {session_id}")
            return updated

    def delete_session(self, session_id: int) -> bool:
        """Delete an annotation session and all its annotations.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(QUERIES["delete_session"], (session_id,))
            deleted = bool(cursor.rowcount > 0)
            if deleted:
                logger.info(
                    f"Deleted session {session_id} and all annotations"
                )
            return deleted

    # ========== Annotation Operations ==========

    def add_annotations(
        self,
        session_id: int,
        annotations: list[tuple[int, str]],
        cell_meta: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple annotations to a session.

        Args:
            session_id: Session ID
            annotations: List of (cell_index, class_label) tuples
            cell_meta: Optional per-crop metadata list, same length as annotations.
                Each entry may contain centroid_row, centroid_col, image_id.

        Raises:
            ValueError: If session doesn't exist
        """
        session = self.get_session_by_id(session_id)
        if not session:
            raise ValueError(f"Session {session_id} does not exist")

        with self._get_connection() as conn:
            for i, (cell_index, class_label) in enumerate(annotations):
                meta = cell_meta[i] if cell_meta and i < len(cell_meta) else {}
                conn.execute(
                    QUERIES["insert_annotation"],
                    (
                        session_id,
                        cell_index,
                        class_label,
                        meta.get("centroid_row"),
                        meta.get("centroid_col"),
                        meta.get("image_id"),
                    ),
                )
            logger.info(
                f"Added {len(annotations)} annotations to session {session_id}"
            )

    def get_annotations(self, session_id: int) -> list[dict[str, Any]]:
        """Get all annotations for a session.

        Args:
            session_id: Session ID

        Returns:
            List of annotation dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["get_annotations_by_session"], (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_annotations_by_classifier(
        self,
        classifier_name: str,
        class_label: str | None = None,
        plate_id: int | None = None,
        well: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get annotations with optional filters.

        Args:
            classifier_name: Classifier name
            class_label: Filter by class label
            plate_id: Filter by plate ID
            well: Filter by well position
            date_from: Filter by date (ISO format)
            date_to: Filter by date (ISO format)

        Returns:
            List of annotation dictionaries with session info

        Raises:
            ValueError: If classifier doesn't exist
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        # Build query with filters
        filters = {
            "class_label": class_label,
            "plate_id": plate_id,
            "well": well,
            "date_from": date_from,
            "date_to": date_to,
        }
        # Remove None values
        filters = {k: v for k, v in filters.items() if v is not None}

        if filters:
            query, params = get_filter_query(
                QUERIES["get_annotations_filtered"], filters
            )
            params.insert(
                0, classifier["id"]
            )  # Add classifier_id as first param
        else:
            query = QUERIES["get_annotations_by_classifier"]
            params = [classifier["id"]]

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def delete_annotations(self, session_id: int) -> bool:
        """Delete all annotations for a session.

        Args:
            session_id: Session ID

        Returns:
            True if any annotations were deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["delete_annotations_by_session"], (session_id,)
            )
            deleted = bool(cursor.rowcount > 0)
            if deleted:
                logger.info(
                    f"Deleted all annotations for session {session_id}"
                )
            return deleted

    def get_used_centroids(
        self, classifier_name: str, plate_id: int, well: str
    ) -> set[tuple[int, int, int]]:
        """Return centroids already annotated for a classifier+plate+well.

        Args:
            classifier_name: Classifier name
            plate_id: OMERO plate ID
            well: Well position (e.g. 'A1')

        Returns:
            Set of (centroid_row, centroid_col, source_image_id) tuples.
            Only entries where all three values are non-NULL are included.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["get_used_centroids"],
                (classifier_name, plate_id, well),
            )
            return {
                (int(row[0]), int(row[1]), int(row[2]))
                for row in cursor.fetchall()
            }

    # ========== Statistics ==========

    def get_class_distribution(
        self,
        classifier_name: str,
        plate_id: int | None = None,
        well: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, int]:
        """Get class distribution (counts per class).

        Args:
            classifier_name: Classifier name
            plate_id: Filter by plate ID
            well: Filter by well position
            date_from: Filter by date (ISO format)
            date_to: Filter by date (ISO format)

        Returns:
            Dictionary mapping class labels to counts

        Raises:
            ValueError: If classifier doesn't exist
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        # Build query with filters
        filters = {
            "plate_id": plate_id,
            "well": well,
            "date_from": date_from,
            "date_to": date_to,
        }
        filters = {k: v for k, v in filters.items() if v is not None}

        if filters:
            base_query = QUERIES["get_class_distribution_filtered"]
            query, params = get_filter_query(base_query, filters)
            # Add GROUP BY and ORDER BY
            query += " GROUP BY a.class_label ORDER BY count DESC"
            params.insert(0, classifier["id"])
        else:
            query = QUERIES["get_class_distribution"]
            params = [classifier["id"]]

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return {
                row["class_label"]: row["count"] for row in cursor.fetchall()
            }

    def get_image_stats(self, classifier_name: str) -> list[dict[str, Any]]:
        """Get statistics per image/session.

        Args:
            classifier_name: Classifier name

        Returns:
            List of dicts containing:
            - plate_id, well, image_id, timepoint
            - total_cells
            - class_distribution: dict[str, int]
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["get_image_stats"], (classifier["id"],)
            )
            stats = []
            for row in cursor.fetchall():
                # Manually count class labels from GROUP_CONCAT
                class_labels = (
                    row["class_labels"].split(",")
                    if row["class_labels"]
                    else []
                )
                from collections import Counter

                dist = dict(Counter(class_labels))

                stats.append(
                    {
                        "session_id": row["session_id"],
                        "plate_id": row["plate_id"],
                        "well": row["well"],
                        "image_id": row["image_id"],
                        "timepoint": row["timepoint"],
                        "total_cells": row["total_cells"],
                        "class_distribution": dist,
                    }
                )
            return stats

    def get_session_count(self, classifier_name: str) -> int:
        """Get number of annotation sessions for a classifier.

        Args:
            classifier_name: Classifier name

        Returns:
            Number of sessions

        Raises:
            ValueError: If classifier doesn't exist
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["count_sessions_by_classifier"], (classifier["id"],)
            )
            result = cursor.fetchone()
            if result is None:
                raise RuntimeError("Expected result from count query")
            return cast(int, result[0])

    def get_total_annotations(self, classifier_name: str) -> int:
        """Get total number of annotations for a classifier.

        Args:
            classifier_name: Classifier name

        Returns:
            Total number of annotations

        Raises:
            ValueError: If classifier doesn't exist
        """
        classifier = self.get_classifier(classifier_name)
        if not classifier:
            raise ValueError(f"Classifier '{classifier_name}' does not exist")

        with self._get_connection() as conn:
            cursor = conn.execute(
                QUERIES["count_annotations_by_classifier"], (classifier["id"],)
            )
            result = cursor.fetchone()
            if result is None:
                raise RuntimeError("Expected result from count query")
            return cast(int, result[0])
