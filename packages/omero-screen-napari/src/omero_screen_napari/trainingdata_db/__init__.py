"""Training data database package.

This package provides a SQLite-based storage system for managing training data
annotations, including classifiers, annotation sessions, and individual cell
classifications.

Main classes:
    TrainingDB: Core database interface for all operations
    TrainingDataMigrator: Import existing .npy training data files

Example usage:
    >>> from omero_screen_napari.trainingdata_db import TrainingDB
    >>> db = TrainingDB()
    >>> db.create_classifier("cell-cycle", "Cell cycle phase classifier")
    >>> db.add_classes("cell-cycle", ["mitotic", "interphase", "apoptotic"])
    >>> session_id = db.create_session(
    ...     classifier_name="cell-cycle",
    ...     plate_id=123,
    ...     well="A1",
    ...     image_id=456,
    ...     timepoint=0,
    ...     file_path="~/path/to/data.npy"
    ... )
    >>> db.add_annotations(session_id, [(0, "mitotic"), (1, "interphase")])
    >>> distribution = db.get_class_distribution("cell-cycle")
    >>> print(distribution)
    {'mitotic': 45, 'interphase': 180, 'apoptotic': 20}

Migration example:
    >>> from omero_screen_napari.trainingdata_db import migrate_classifier
    >>> stats = migrate_classifier("/path/to/classifier-dir", dry_run=False)
"""

from .database import TrainingDB
from .migrator import (
    TrainingDataMigrator,
    migrate_all_classifiers,
    migrate_classifier,
)
from .schema import SCHEMA_VERSION

__all__ = [
    "TrainingDB",
    "TrainingDataMigrator",
    "migrate_classifier",
    "migrate_all_classifiers",
    "SCHEMA_VERSION",
]

__version__ = "0.1.0"
