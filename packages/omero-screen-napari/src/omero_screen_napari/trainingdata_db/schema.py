"""Database schema definitions for training data storage.

This module defines the SQLite schema for storing training data metadata,
including classifiers, annotation sessions, and individual cell annotations.
"""

from typing import Any

# SQL schema version for migration tracking
SCHEMA_VERSION = 1

# Main schema creation SQL
CREATE_TABLES_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Classifier definitions
CREATE TABLE IF NOT EXISTS classifiers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Class labels for each classifier
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    classifier_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    description TEXT,
    FOREIGN KEY (classifier_id) REFERENCES classifiers(id) ON DELETE CASCADE,
    UNIQUE(classifier_id, label)
);

-- Annotation sessions (corresponds to one .npy file)
CREATE TABLE IF NOT EXISTS annotation_sessions (
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
    FOREIGN KEY (classifier_id) REFERENCES classifiers(id) ON DELETE CASCADE,
    UNIQUE(classifier_id, plate_id, well, image_id, timepoint)
);

-- Individual cell annotations
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    cell_index INTEGER NOT NULL,
    class_label TEXT NOT NULL,
    annotated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES annotation_sessions(id) ON DELETE CASCADE,
    UNIQUE(session_id, cell_index)
);
"""

# Indexes for common queries
CREATE_INDEXES_SQL = """
-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_sessions_classifier
    ON annotation_sessions(classifier_id);

CREATE INDEX IF NOT EXISTS idx_sessions_plate
    ON annotation_sessions(plate_id);

CREATE INDEX IF NOT EXISTS idx_sessions_updated
    ON annotation_sessions(updated_at);

CREATE INDEX IF NOT EXISTS idx_annotations_session
    ON annotations(session_id);

CREATE INDEX IF NOT EXISTS idx_annotations_class
    ON annotations(class_label);

CREATE INDEX IF NOT EXISTS idx_classes_classifier
    ON classes(classifier_id);
"""

# Common queries as constants
QUERIES = {
    # Classifier queries
    "get_classifier_by_name": """
        SELECT id, name, description, created_at
        FROM classifiers
        WHERE name = ?
    """,
    "get_classifier_by_id": """
        SELECT id, name, description, created_at
        FROM classifiers
        WHERE id = ?
    """,
    "list_classifiers": """
        SELECT c.id, c.name, c.description, c.created_at, GROUP_CONCAT(cl.label, ', ') as class_labels
        FROM classifiers c
        LEFT JOIN classes cl ON c.id = cl.classifier_id
        GROUP BY c.id
        ORDER BY c.created_at DESC
    """,
    "insert_classifier": """
        INSERT INTO classifiers (name, description)
        VALUES (?, ?)
    """,
    "delete_classifier": """
        DELETE FROM classifiers
        WHERE name = ?
    """,
    # Class queries
    "get_classes_by_classifier": """
        SELECT label, description
        FROM classes
        WHERE classifier_id = ?
        ORDER BY label
    """,
    "insert_class": """
        INSERT OR IGNORE INTO classes (classifier_id, label, description)
        VALUES (?, ?, ?)
    """,
    "delete_classes_by_classifier": """
        DELETE FROM classes
        WHERE classifier_id = ?
    """,
    # Session queries
    "get_session": """
        SELECT id, classifier_id, plate_id, well, image_id, timepoint,
               user, created_at, updated_at, file_path, metadata_json
        FROM annotation_sessions
        WHERE classifier_id = ? AND plate_id = ? AND well = ?
              AND image_id = ? AND timepoint = ?
    """,
    "get_session_by_id": """
        SELECT id, classifier_id, plate_id, well, image_id, timepoint,
               user, created_at, updated_at, file_path, metadata_json
        FROM annotation_sessions
        WHERE id = ?
    """,
    "list_sessions_by_classifier": """
        SELECT id, classifier_id, plate_id, well, image_id, timepoint,
               user, created_at, updated_at, file_path, metadata_json
        FROM annotation_sessions
        WHERE classifier_id = ?
        ORDER BY updated_at DESC
    """,
    "insert_session": """
        INSERT INTO annotation_sessions
        (classifier_id, plate_id, well, image_id, timepoint, user,
         file_path, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    "update_session": """
        UPDATE annotation_sessions
        SET updated_at = CURRENT_TIMESTAMP, file_path = ?, metadata_json = ?
        WHERE id = ?
    """,
    "delete_session": """
        DELETE FROM annotation_sessions
        WHERE id = ?
    """,
    "count_sessions_by_classifier": """
        SELECT COUNT(*)
        FROM annotation_sessions
        WHERE classifier_id = ?
    """,
    # Annotation queries
    "get_annotations_by_session": """
        SELECT id, cell_index, class_label, annotated_at
        FROM annotations
        WHERE session_id = ?
        ORDER BY cell_index
    """,
    "get_annotations_by_classifier": """
        SELECT a.id, a.session_id, a.cell_index, a.class_label, a.annotated_at,
               s.plate_id, s.well, s.image_id, s.timepoint, s.file_path
        FROM annotations a
        JOIN annotation_sessions s ON a.session_id = s.id
        WHERE s.classifier_id = ?
        ORDER BY a.annotated_at DESC
    """,
    "get_annotations_filtered": """
        SELECT a.id, a.session_id, a.cell_index, a.class_label, a.annotated_at,
               s.plate_id, s.well, s.image_id, s.timepoint, s.file_path
        FROM annotations a
        JOIN annotation_sessions s ON a.session_id = s.id
        WHERE s.classifier_id = ?
    """,
    "insert_annotation": """
        INSERT OR REPLACE INTO annotations
        (session_id, cell_index, class_label, annotated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """,
    "delete_annotations_by_session": """
        DELETE FROM annotations
        WHERE session_id = ?
    """,
    "count_annotations_by_session": """
        SELECT COUNT(*)
        FROM annotations
        WHERE session_id = ?
    """,
    "count_annotations_by_classifier": """
        SELECT COUNT(*)
        FROM annotations a
        JOIN annotation_sessions s ON a.session_id = s.id
        WHERE s.classifier_id = ?
    """,
    # Statistics queries
    "get_class_distribution": """
        SELECT a.class_label, COUNT(*) as count
        FROM annotations a
        JOIN annotation_sessions s ON a.session_id = s.id
        WHERE s.classifier_id = ?
        GROUP BY a.class_label
        ORDER BY count DESC
    """,
    "get_class_distribution_filtered": """
        SELECT a.class_label, COUNT(*) as count
        FROM annotations a
        JOIN annotation_sessions s ON a.session_id = s.id
        WHERE s.classifier_id = ?
    """,
    "get_image_stats": """
        SELECT s.plate_id, s.well, s.image_id, s.timepoint,
               COUNT(a.id) as total_cells,
               GROUP_CONCAT(a.class_label) as class_labels
        FROM annotation_sessions s
        LEFT JOIN annotations a ON s.id = a.session_id
        WHERE s.classifier_id = ?
        GROUP BY s.id
        ORDER BY s.plate_id, s.well, s.image_id
    """,
    # Version queries
    "get_schema_version": """
        SELECT version
        FROM schema_version
        ORDER BY version DESC
        LIMIT 1
    """,
    "insert_schema_version": """
        INSERT INTO schema_version (version)
        VALUES (?)
    """,
}


def get_filter_query(
    base_query: str, filters: dict[str, Any]
) -> tuple[str, list[Any]]:
    """Build dynamic SQL query with filters.

    Args:
        base_query: Base SQL query (must end with WHERE clause)
        filters: Dictionary of filter conditions
            - class_label: str or None
            - plate_id: int or None
            - well: str or None
            - date_from: str (ISO format) or None
            - date_to: str (ISO format) or None

    Returns:
        Tuple of (query_string, parameters_list)
    """
    conditions = []
    params = []

    if filters.get("class_label"):
        conditions.append("AND a.class_label = ?")
        params.append(filters["class_label"])

    if filters.get("plate_id"):
        conditions.append("AND s.plate_id = ?")
        params.append(filters["plate_id"])

    if filters.get("well"):
        conditions.append("AND s.well = ?")
        params.append(filters["well"])

    if filters.get("date_from"):
        conditions.append("AND s.updated_at >= ?")
        params.append(filters["date_from"])

    if filters.get("date_to"):
        conditions.append("AND s.updated_at <= ?")
        params.append(filters["date_to"])

    query = base_query + " " + " ".join(conditions)
    return query, params
