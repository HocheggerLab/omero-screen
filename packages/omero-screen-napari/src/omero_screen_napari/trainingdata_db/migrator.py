"""Training data migration utilities.

This module provides tools for migrating existing .npy training data files
into the training data database.
"""

import json
import re
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from loguru import logger

from .database import TrainingDB


class MigrationStats(TypedDict):
    """Type definition for migration statistics."""

    classifiers_created: int
    sessions_imported: int
    annotations_imported: int
    sessions_skipped: int
    errors: list[str]


class TrainingDataMigrator:
    """Migrate existing .npy training data files to database.

    This class scans directories containing training data in the old format
    (.npy files with metadata.json) and imports them into the training database.

    Attributes:
        db: TrainingDB instance
        dry_run: If True, only report what would be imported without modifying database
    """

    def __init__(
        self, db: TrainingDB | None = None, dry_run: bool = False
    ) -> None:
        """Initialize migrator.

        Args:
            db: TrainingDB instance. If None, creates a new instance.
            dry_run: If True, only report what would be imported
        """
        self.db = db if db else TrainingDB()
        self.dry_run = dry_run
        self.stats: MigrationStats = {
            "classifiers_created": 0,
            "sessions_imported": 0,
            "annotations_imported": 0,
            "sessions_skipped": 0,
            "errors": [],
        }

    def migrate_directory(self, classifier_dir: Path) -> MigrationStats:
        """Migrate a single classifier directory.

        Expected structure:
            classifier_dir/
                metadata.json
                plate_well_image_timepoint.npy
                ...

        Args:
            classifier_dir: Path to classifier directory

        Returns:
            Dictionary with migration statistics for this classifier
        """
        classifier_dir = Path(classifier_dir)
        if not classifier_dir.exists():
            error = f"Directory does not exist: {classifier_dir}"
            logger.error(error)
            self.stats["errors"].append(error)
            return self.stats

        classifier_name = classifier_dir.name
        logger.info(f"Migrating classifier: {classifier_name}")

        # Check for metadata.json
        metadata_path = classifier_dir / "metadata.json"
        if not metadata_path.exists():
            error = f"No metadata.json found in {classifier_dir}"
            logger.error(error)
            self.stats["errors"].append(error)
            return self.stats

        # Load metadata
        try:
            with metadata_path.open("r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            error = f"Error parsing metadata.json in {classifier_dir}: {e}"
            logger.error(error)
            self.stats["errors"].append(error)
            return self.stats

        # Create classifier
        try:
            if not self.dry_run:
                # Check if classifier exists
                if not self.db.get_classifier(classifier_name):
                    self.db.create_classifier(
                        classifier_name, f"Migrated from {classifier_dir}"
                    )
                    logger.info(f"Created classifier: {classifier_name}")
                    self.stats["classifiers_created"] += 1

                # Always attempt to population classes if they are in metadata
                # Since db.add_classes uses insert OR IGNORE, this is safe
                if "class_options" in metadata:
                    class_labels = metadata["class_options"]
                    try:
                        self.db.add_classes(classifier_name, class_labels)
                        logger.info(
                            f"Added/Updated classes for {classifier_name}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not populate classes for {classifier_name}: {e}"
                        )
            else:
                logger.info(
                    f"[DRY RUN] Would create classifier: {classifier_name}"
                )
                if "class_options" in metadata:
                    logger.info(
                        f"[DRY RUN] Would add classes: {metadata['class_options']}"
                    )
        except Exception as e:
            error = f"Error creating classifier {classifier_name}: {e}"
            logger.error(error)
            self.stats["errors"].append(error)
            return self.stats

        # Find and import all .npy files
        npy_files = list(classifier_dir.glob("*.npy"))
        logger.info(f"Found {len(npy_files)} .npy files to import")

        for npy_file in npy_files:
            try:
                self._import_npy_file(
                    npy_file, classifier_name, metadata.get("user_data", {})
                )
            except Exception as e:
                error = f"Error importing {npy_file.name}: {e}"
                logger.error(error)
                self.stats["errors"].append(error)
                continue

        return self.stats

    def _import_npy_file(
        self,
        npy_file: Path,
        classifier_name: str,
        user_metadata: dict[str, Any],
    ) -> None:
        """Import a single .npy file into the database.

        Args:
            npy_file: Path to .npy file
            classifier_name: Name of classifier
            user_metadata: User metadata from metadata.json
        """
        # Parse filename: plate_well_image_timepoint.npy
        filename = npy_file.stem
        match = re.match(r"(\d+)_([A-Z]\d+)_(\d+)_(\d+)", filename)

        if not match:
            logger.warning(f"Could not parse filename: {filename}")
            return

        plate_id = int(match.group(1))
        well = match.group(2)
        image_id = int(match.group(3))
        timepoint = int(match.group(4))

        logger.debug(
            f"Importing {filename}: plate={plate_id}, well={well}, "
            f"image={image_id}, timepoint={timepoint}"
        )

        # Check if session already exists
        if not self.dry_run:
            existing_session = self.db.get_session(
                classifier_name, plate_id, well, image_id, timepoint
            )
            if existing_session:
                logger.info(f"Session already exists, skipping: {filename}")
                self.stats["sessions_skipped"] += 1
                return

        # Load .npy file
        try:
            data = np.load(npy_file, allow_pickle=True).item()
        except Exception as e:
            error = f"Error loading {npy_file}: {e}"
            logger.error(error)
            self.stats["errors"].append(error)
            return

        # Validate data structure
        if "data" not in data or "target" not in data:
            logger.warning(f"Invalid data structure in {npy_file}")
            return

        # Extract annotations
        crops, labels = data["data"]
        class_labels = data["target"]

        if len(class_labels) != len(crops):
            logger.warning(
                f"Mismatch in data lengths: {len(crops)} crops, "
                f"{len(class_labels)} labels in {filename}"
            )
            return

        num_annotations = len(class_labels)

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would import session: {filename} "
                f"with {num_annotations} annotations"
            )
            return

        # Create session
        try:
            session_id = self.db.create_session(
                classifier_name=classifier_name,
                plate_id=plate_id,
                well=well,
                image_id=image_id,
                timepoint=timepoint,
                file_path=str(npy_file),
                metadata=user_metadata,
            )
            self.stats["sessions_imported"] += 1
            logger.info(f"Created session {session_id} for {filename}")

            # Add annotations
            annotations = [
                (cell_idx, class_label)
                for cell_idx, class_label in enumerate(class_labels)
            ]
            self.db.add_annotations(session_id, annotations)
            self.stats["annotations_imported"] += num_annotations
            logger.info(
                f"Added {num_annotations} annotations to session {session_id}"
            )

        except Exception as e:
            error = f"Error creating session for {filename}: {e}"
            logger.error(error)
            self.stats["errors"].append(error)

    def migrate_all(self, base_dir: Path | None = None) -> MigrationStats:
        """Migrate all classifier directories in the base directory.

        Args:
            base_dir: Base directory containing classifier subdirectories.
                     Defaults to ~/omeroscreen_trainingdata/

        Returns:
            Dictionary with migration statistics
        """
        if base_dir is None:
            base_dir = Path.home() / "omeroscreen_trainingdata"

        base_dir = Path(base_dir)
        if not base_dir.exists():
            error = f"Base directory does not exist: {base_dir}"
            logger.error(error)
            self.stats["errors"].append(error)
            return self.stats

        logger.info(f"Scanning {base_dir} for classifier directories...")

        # Find all directories with metadata.json
        classifier_dirs: list[Path] = []
        classifier_dirs.extend(
            item
            for item in base_dir.iterdir()
            if item.is_dir()
            and (item / "metadata.json").exists()
            and not item.name.endswith(".db")
        )
        logger.info(
            f"Found {len(classifier_dirs)} classifier directories to migrate"
        )

        for classifier_dir in classifier_dirs:
            self.migrate_directory(classifier_dir)

        return self.stats

    def print_stats(self) -> None:
        """Print migration statistics."""
        print("\n" + "=" * 60)
        print("Migration Statistics")
        print("=" * 60)
        print(f"Classifiers created:    {self.stats['classifiers_created']}")
        print(f"Sessions imported:      {self.stats['sessions_imported']}")
        print(f"Sessions skipped:       {self.stats['sessions_skipped']}")
        print(f"Annotations imported:   {self.stats['annotations_imported']}")

        if self.stats["errors"]:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats["errors"][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.stats["errors"]) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more")
        else:
            print("\nNo errors encountered!")

        print("=" * 60 + "\n")


def migrate_classifier(
    classifier_dir: Path | str, dry_run: bool = False
) -> MigrationStats:
    """Convenience function to migrate a single classifier directory.

    Args:
        classifier_dir: Path to classifier directory
        dry_run: If True, only report what would be imported

    Returns:
        Dictionary with migration statistics
    """
    migrator = TrainingDataMigrator(dry_run=dry_run)
    stats = migrator.migrate_directory(Path(classifier_dir))
    migrator.print_stats()
    return stats


def migrate_all_classifiers(
    base_dir: Path | str | None = None, dry_run: bool = False
) -> MigrationStats:
    """Convenience function to migrate all classifiers.

    Args:
        base_dir: Base directory containing classifier subdirectories.
                 Defaults to ~/omeroscreen_trainingdata/
        dry_run: If True, only report what would be imported

    Returns:
        Dictionary with migration statistics
    """
    migrator = TrainingDataMigrator(dry_run=dry_run)
    stats = migrator.migrate_all(Path(base_dir) if base_dir else None)
    migrator.print_stats()
    return stats
