"""Unit tests for training data database.

Tests cover all CRUD operations for classifiers, sessions, and annotations,
as well as statistics and filtering functionality.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from omero_screen_napari.trainingdata_db import SCHEMA_VERSION, TrainingDB


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_training.db"
        db = TrainingDB(db_path=db_path)
        yield db
        # Cleanup is automatic with tempfile


@pytest.fixture
def sample_classifier(temp_db):
    """Create a sample classifier for testing."""
    classifier_id = temp_db.create_classifier(
        "test-classifier", "Test classifier for unit tests"
    )
    temp_db.add_classes(
        "test-classifier", ["class_a", "class_b", "class_c"], ["", "", ""]
    )
    return temp_db, classifier_id


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_create_database(self, temp_db):
        """Test database creation."""
        assert temp_db.db_path.exists()
        assert temp_db.db_path.is_file()

    def test_schema_version(self, temp_db):
        """Test schema version is set correctly."""
        with temp_db._get_connection() as conn:
            cursor = conn.execute("SELECT version FROM schema_version")
            version = cursor.fetchone()[0]
            assert version == SCHEMA_VERSION

    def test_tables_created(self, temp_db):
        """Test all required tables are created."""
        with temp_db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "schema_version",
            "classifiers",
            "classes",
            "annotation_sessions",
            "annotations",
        }
        assert expected_tables.issubset(tables)

    def test_indexes_created(self, temp_db):
        """Test indexes are created."""
        with temp_db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = {row[0] for row in cursor.fetchall()}

        expected_indexes = {
            "idx_sessions_classifier",
            "idx_sessions_plate",
            "idx_sessions_updated",
            "idx_annotations_session",
            "idx_annotations_class",
            "idx_classes_classifier",
        }
        assert expected_indexes.issubset(indexes)

    def test_foreign_keys_enabled(self, temp_db):
        """Test foreign key constraints are enabled."""
        with temp_db._get_connection() as conn:
            cursor = conn.execute("PRAGMA foreign_keys")
            enabled = cursor.fetchone()[0]
            assert enabled == 1


class TestClassifierOperations:
    """Tests for classifier CRUD operations."""

    def test_create_classifier(self, temp_db):
        """Test creating a classifier."""
        classifier_id = temp_db.create_classifier(
            "my-classifier", "My test classifier"
        )
        assert isinstance(classifier_id, int)
        assert classifier_id > 0

    def test_create_duplicate_classifier(self, temp_db):
        """Test that duplicate classifier names are rejected."""
        temp_db.create_classifier("duplicate", "First")
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.create_classifier("duplicate", "Second")

    def test_get_classifier(self, temp_db):
        """Test retrieving a classifier by name."""
        temp_db.create_classifier("test-clf", "Test")
        classifier = temp_db.get_classifier("test-clf")

        assert classifier is not None
        assert classifier["name"] == "test-clf"
        assert classifier["description"] == "Test"
        assert "id" in classifier
        assert "created_at" in classifier

    def test_get_nonexistent_classifier(self, temp_db):
        """Test retrieving a classifier that doesn't exist."""
        classifier = temp_db.get_classifier("nonexistent")
        assert classifier is None

    def test_get_classifier_by_id(self, temp_db):
        """Test retrieving a classifier by ID."""
        classifier_id = temp_db.create_classifier("test-clf", "Test")
        classifier = temp_db.get_classifier_by_id(classifier_id)

        assert classifier is not None
        assert classifier["id"] == classifier_id
        assert classifier["name"] == "test-clf"

    def test_list_classifiers(self, temp_db):
        """Test listing all classifiers."""
        temp_db.create_classifier("clf1", "First")
        temp_db.create_classifier("clf2", "Second")
        temp_db.create_classifier("clf3", "Third")

        classifiers = temp_db.list_classifiers()
        assert len(classifiers) == 3
        # All three should be present
        names = {c["name"] for c in classifiers}
        assert names == {"clf1", "clf2", "clf3"}

    def test_list_classifiers_empty(self, temp_db):
        """Test listing classifiers when database is empty."""
        classifiers = temp_db.list_classifiers()
        assert classifiers == []

    def test_delete_classifier(self, temp_db):
        """Test deleting a classifier."""
        temp_db.create_classifier("to-delete", "Will be deleted")
        deleted = temp_db.delete_classifier("to-delete")
        assert deleted is True

        classifier = temp_db.get_classifier("to-delete")
        assert classifier is None

    def test_delete_nonexistent_classifier(self, temp_db):
        """Test deleting a classifier that doesn't exist."""
        deleted = temp_db.delete_classifier("nonexistent")
        assert deleted is False


class TestClassOperations:
    """Tests for class label operations."""

    def test_add_class(self, temp_db):
        """Test adding a class to a classifier."""
        temp_db.create_classifier("test-clf", "Test")
        temp_db.add_class("test-clf", "positive", "Positive class")
        classes = temp_db.get_classes("test-clf")
        assert "positive" in classes

    def test_add_class_to_nonexistent_classifier(self, temp_db):
        """Test adding a class to a classifier that doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            temp_db.add_class("nonexistent", "class1")

    def test_add_duplicate_class(self, temp_db):
        """Test adding duplicate class labels (should be idempotent)."""
        temp_db.create_classifier("test-clf", "Test")
        temp_db.add_class("test-clf", "class1")
        temp_db.add_class("test-clf", "class1")  # Should not raise error
        classes = temp_db.get_classes("test-clf")
        assert classes.count("class1") == 1

    def test_add_multiple_classes(self, temp_db):
        """Test adding multiple classes at once."""
        temp_db.create_classifier("test-clf", "Test")
        labels = ["class1", "class2", "class3"]
        descriptions = ["First", "Second", "Third"]
        temp_db.add_classes("test-clf", labels, descriptions)

        classes = temp_db.get_classes("test-clf")
        assert set(classes) == set(labels)

    def test_add_classes_without_descriptions(self, temp_db):
        """Test adding classes without descriptions."""
        temp_db.create_classifier("test-clf", "Test")
        temp_db.add_classes("test-clf", ["class1", "class2"])
        classes = temp_db.get_classes("test-clf")
        assert len(classes) == 2

    def test_add_classes_mismatched_lengths(self, temp_db):
        """Test that mismatched label/description lengths raise error."""
        temp_db.create_classifier("test-clf", "Test")
        with pytest.raises(ValueError, match="must match"):
            temp_db.add_classes("test-clf", ["a", "b"], ["desc1"])

    def test_get_classes(self, temp_db):
        """Test retrieving all classes for a classifier."""
        temp_db.create_classifier("test-clf", "Test")
        temp_db.add_classes("test-clf", ["alpha", "beta", "gamma"])
        classes = temp_db.get_classes("test-clf")

        assert len(classes) == 3
        assert "alpha" in classes
        assert "beta" in classes
        assert "gamma" in classes

    def test_get_classes_empty(self, temp_db):
        """Test getting classes when none exist."""
        temp_db.create_classifier("test-clf", "Test")
        classes = temp_db.get_classes("test-clf")
        assert classes == []

    def test_get_classes_nonexistent_classifier(self, temp_db):
        """Test getting classes for nonexistent classifier."""
        with pytest.raises(ValueError, match="does not exist"):
            temp_db.get_classes("nonexistent")


class TestSessionOperations:
    """Tests for annotation session operations."""

    def test_create_session(self, sample_classifier):
        """Test creating an annotation session."""
        db, _ = sample_classifier
        session_id = db.create_session(
            classifier_name="test-classifier",
            plate_id=123,
            well="A1",
            image_id=456,
            timepoint=0,
            file_path="/path/to/file.npy",
            metadata={"crop_size": 64, "channels": [0, 1, 2]},
            user="testuser",
        )
        assert isinstance(session_id, int)
        assert session_id > 0

    def test_create_session_nonexistent_classifier(self, temp_db):
        """Test creating session for nonexistent classifier."""
        with pytest.raises(ValueError, match="does not exist"):
            temp_db.create_session(
                classifier_name="nonexistent",
                plate_id=1,
                well="A1",
                image_id=1,
                timepoint=0,
                file_path="/path/to/file.npy",
            )

    def test_create_multiple_sessions_same_well(self, sample_classifier):
        """Test that multiple sessions from the same well are allowed."""
        db, _ = sample_classifier
        session_id_1 = db.create_session(
            classifier_name="test-classifier",
            plate_id=123,
            well="A1",
            image_id=456,
            timepoint=0,
            file_path="/path/to/file.npy",
        )
        session_id_2 = db.create_session(
            classifier_name="test-classifier",
            plate_id=123,
            well="A1",
            image_id=456,
            timepoint=0,
            file_path="/path/to/file2.npy",
        )
        assert session_id_1 != session_id_2
        sessions = db.list_sessions("test-classifier")
        assert len(sessions) == 2

    def test_get_session(self, sample_classifier):
        """Test retrieving a session."""
        db, _ = sample_classifier
        metadata = {"crop_size": 64}
        db.create_session(
            classifier_name="test-classifier",
            plate_id=123,
            well="A1",
            image_id=456,
            timepoint=0,
            file_path="/path/to/file.npy",
            metadata=metadata,
        )

        session = db.get_session(
            classifier_name="test-classifier",
            plate_id=123,
            well="A1",
            image_id=456,
            timepoint=0,
        )

        assert session is not None
        assert session["plate_id"] == 123
        assert session["well"] == "A1"
        assert session["image_id"] == 456
        assert session["timepoint"] == 0
        assert session["file_path"] == "/path/to/file.npy"
        assert session["metadata"] == metadata

    def test_get_nonexistent_session(self, sample_classifier):
        """Test retrieving a session that doesn't exist."""
        db, _ = sample_classifier
        session = db.get_session(
            classifier_name="test-classifier",
            plate_id=999,
            well="Z9",
            image_id=999,
            timepoint=99,
        )
        assert session is None

    def test_get_session_by_id(self, sample_classifier):
        """Test retrieving a session by ID."""
        db, _ = sample_classifier
        session_id = db.create_session(
            classifier_name="test-classifier",
            plate_id=123,
            well="A1",
            image_id=456,
            timepoint=0,
            file_path="/path/to/file.npy",
        )

        session = db.get_session_by_id(session_id)
        assert session is not None
        assert session["id"] == session_id

    def test_list_sessions(self, sample_classifier):
        """Test listing all sessions for a classifier."""
        db, _ = sample_classifier
        db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path/file1.npy"
        )
        db.create_session(
            "test-classifier", 1, "A2", 11, 0, "/path/file2.npy"
        )
        db.create_session(
            "test-classifier", 2, "B1", 12, 0, "/path/file3.npy"
        )

        sessions = db.list_sessions("test-classifier")
        assert len(sessions) == 3

    def test_update_session(self, sample_classifier):
        """Test updating a session."""
        db, _ = sample_classifier
        session_id = db.create_session(
            classifier_name="test-classifier",
            plate_id=123,
            well="A1",
            image_id=456,
            timepoint=0,
            file_path="/old/path.npy",
            metadata={"old": "data"},
        )

        updated = db.update_session(
            session_id=session_id,
            file_path="/new/path.npy",
            metadata={"new": "data"},
        )
        assert updated is True

        session = db.get_session_by_id(session_id)
        assert session["file_path"] == "/new/path.npy"
        assert session["metadata"] == {"new": "data"}

    def test_update_nonexistent_session(self, temp_db):
        """Test updating a session that doesn't exist."""
        updated = temp_db.update_session(session_id=999, file_path="/path.npy")
        assert updated is False

    def test_delete_session(self, sample_classifier):
        """Test deleting a session."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )

        deleted = db.delete_session(session_id)
        assert deleted is True

        session = db.get_session_by_id(session_id)
        assert session is None

    def test_delete_nonexistent_session(self, temp_db):
        """Test deleting a session that doesn't exist."""
        deleted = temp_db.delete_session(999)
        assert deleted is False


class TestAnnotationOperations:
    """Tests for annotation operations."""

    def test_add_annotations(self, sample_classifier):
        """Test adding annotations to a session."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )

        annotations = [
            (0, "class_a"),
            (1, "class_b"),
            (2, "class_a"),
            (3, "class_c"),
        ]
        db.add_annotations(session_id, annotations)

        retrieved = db.get_annotations(session_id)
        assert len(retrieved) == 4

    def test_add_annotations_nonexistent_session(self, temp_db):
        """Test adding annotations to nonexistent session."""
        with pytest.raises(ValueError, match="does not exist"):
            temp_db.add_annotations(999, [(0, "class1")])

    def test_get_annotations(self, sample_classifier):
        """Test retrieving annotations for a session."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )

        db.add_annotations(session_id, [(0, "class_a"), (1, "class_b")])
        annotations = db.get_annotations(session_id)

        assert len(annotations) == 2
        assert annotations[0]["cell_index"] == 0
        assert annotations[0]["class_label"] == "class_a"
        assert annotations[1]["cell_index"] == 1
        assert annotations[1]["class_label"] == "class_b"

    def test_get_annotations_empty(self, sample_classifier):
        """Test getting annotations when none exist."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )

        annotations = db.get_annotations(session_id)
        assert annotations == []

    def test_get_annotations_by_classifier(self, sample_classifier):
        """Test getting all annotations for a classifier."""
        db, _ = sample_classifier

        # Create multiple sessions with annotations
        s1 = db.create_session("test-classifier", 1, "A1", 10, 0, "/path1.npy")
        s2 = db.create_session("test-classifier", 1, "A2", 11, 0, "/path2.npy")

        db.add_annotations(s1, [(0, "class_a"), (1, "class_b")])
        db.add_annotations(s2, [(0, "class_c"), (1, "class_a")])

        annotations = db.get_annotations_by_classifier("test-classifier")
        assert len(annotations) == 4

    def test_get_annotations_filtered_by_class(self, sample_classifier):
        """Test filtering annotations by class label."""
        db, _ = sample_classifier

        s1 = db.create_session("test-classifier", 1, "A1", 10, 0, "/path.npy")
        db.add_annotations(
            s1, [(0, "class_a"), (1, "class_b"), (2, "class_a")]
        )

        annotations = db.get_annotations_by_classifier(
            "test-classifier", class_label="class_a"
        )
        assert len(annotations) == 2
        assert all(a["class_label"] == "class_a" for a in annotations)

    def test_get_annotations_filtered_by_plate(self, sample_classifier):
        """Test filtering annotations by plate ID."""
        db, _ = sample_classifier

        s1 = db.create_session("test-classifier", 1, "A1", 10, 0, "/path1.npy")
        s2 = db.create_session("test-classifier", 2, "A1", 10, 0, "/path2.npy")

        db.add_annotations(s1, [(0, "class_a")])
        db.add_annotations(s2, [(0, "class_b")])

        annotations = db.get_annotations_by_classifier(
            "test-classifier", plate_id=1
        )
        assert len(annotations) == 1
        assert annotations[0]["plate_id"] == 1

    def test_get_annotations_multiple_filters(self, sample_classifier):
        """Test using multiple filters together."""
        db, _ = sample_classifier

        s1 = db.create_session("test-classifier", 1, "A1", 10, 0, "/path.npy")
        db.add_annotations(s1, [(0, "class_a"), (1, "class_b")])

        annotations = db.get_annotations_by_classifier(
            "test-classifier", class_label="class_a", plate_id=1, well="A1"
        )
        assert len(annotations) == 1
        assert annotations[0]["class_label"] == "class_a"
        assert annotations[0]["plate_id"] == 1
        assert annotations[0]["well"] == "A1"

    def test_delete_annotations(self, sample_classifier):
        """Test deleting all annotations for a session."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )
        db.add_annotations(session_id, [(0, "class_a"), (1, "class_b")])

        deleted = db.delete_annotations(session_id)
        assert deleted is True

        annotations = db.get_annotations(session_id)
        assert len(annotations) == 0

    def test_cascade_delete_annotations(self, sample_classifier):
        """Test that annotations are deleted when session is deleted."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )
        db.add_annotations(session_id, [(0, "class_a")])

        db.delete_session(session_id)

        # Annotations should be gone
        annotations = db.get_annotations(session_id)
        assert len(annotations) == 0


class TestUsedCentroids:
    """Tests for the already-annotated centroid lookup.

    The returned tuples are fed straight to ``CropPipeline`` as exclusion
    keys, so the element order is part of the contract: get it wrong and
    the exclusion silently matches nothing, letting the same cell be
    labelled again on every reload.
    """

    def test_returns_image_id_row_col_in_croppipeline_key_order(
        self, sample_classifier
    ):
        """Tuples are (image_id, centroid_row, centroid_col)."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )
        db.add_annotations(
            session_id,
            [(0, "class_a")],
            cell_meta=[
                {"centroid_row": 11, "centroid_col": 22, "image_id": 333}
            ],
        )

        assert db.get_used_centroids("test-classifier", 1, "A1") == {
            (333, 11, 22)
        }

    def test_scoped_to_classifier_plate_and_well(self, sample_classifier):
        """Annotations from another well don't leak into the exclusion."""
        db, _ = sample_classifier
        session_a1 = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/a1.npy"
        )
        session_b2 = db.create_session(
            "test-classifier", 1, "B2", 10, 0, "/b2.npy"
        )
        db.add_annotations(
            session_a1,
            [(0, "class_a")],
            cell_meta=[{"centroid_row": 1, "centroid_col": 2, "image_id": 10}],
        )
        db.add_annotations(
            session_b2,
            [(0, "class_b")],
            cell_meta=[{"centroid_row": 3, "centroid_col": 4, "image_id": 20}],
        )

        assert db.get_used_centroids("test-classifier", 1, "A1") == {(10, 1, 2)}
        assert db.get_used_centroids("test-classifier", 1, "B2") == {(20, 3, 4)}
        assert db.get_used_centroids("test-classifier", 2, "A1") == set()

    def test_accumulates_across_sessions_in_the_same_well(
        self, sample_classifier
    ):
        """A second pass over a well excludes the first pass's cells."""
        db, _ = sample_classifier
        first = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/first.npy"
        )
        second = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/second.npy"
        )
        db.add_annotations(
            first,
            [(0, "class_a")],
            cell_meta=[{"centroid_row": 1, "centroid_col": 1, "image_id": 10}],
        )
        db.add_annotations(
            second,
            [(0, "class_b")],
            cell_meta=[{"centroid_row": 2, "centroid_col": 2, "image_id": 10}],
        )

        assert db.get_used_centroids("test-classifier", 1, "A1") == {
            (10, 1, 1),
            (10, 2, 2),
        }

    def test_scoped_to_timepoint_when_given(self, sample_classifier):
        """A cell labelled at another timepoint stays available.

        In a timelapse the same coordinates one frame later are a
        different observation, so they must not be excluded.
        """
        db, _ = sample_classifier
        session_t10 = db.create_session(
            "test-classifier", 1, "A1", 10, 10, "/t10.npy"
        )
        session_t15 = db.create_session(
            "test-classifier", 1, "A1", 10, 15, "/t15.npy"
        )
        db.add_annotations(
            session_t10,
            [(0, "class_a")],
            cell_meta=[{"centroid_row": 1, "centroid_col": 2, "image_id": 10}],
        )
        db.add_annotations(
            session_t15,
            [(0, "class_b")],
            cell_meta=[{"centroid_row": 3, "centroid_col": 4, "image_id": 10}],
        )

        assert db.get_used_centroids(
            "test-classifier", 1, "A1", timepoint=10
        ) == {(10, 1, 2)}
        assert db.get_used_centroids(
            "test-classifier", 1, "A1", timepoint=15
        ) == {(10, 3, 4)}

    def test_same_coordinates_at_different_timepoints_are_independent(
        self, sample_classifier
    ):
        """An unmoved cell is still offered again in the next frame."""
        db, _ = sample_classifier
        session_t0 = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/t0.npy"
        )
        db.add_annotations(
            session_t0,
            [(0, "class_a")],
            cell_meta=[{"centroid_row": 5, "centroid_col": 5, "image_id": 10}],
        )

        assert db.get_used_centroids(
            "test-classifier", 1, "A1", timepoint=1
        ) == set()

    def test_omitting_timepoint_spans_all_timepoints(self, sample_classifier):
        """The default keeps the pre-timepoint behaviour."""
        db, _ = sample_classifier
        for tp, row in ((10, 1), (15, 3)):
            session_id = db.create_session(
                "test-classifier", 1, "A1", 10, tp, f"/t{tp}.npy"
            )
            db.add_annotations(
                session_id,
                [(0, "class_a")],
                cell_meta=[
                    {"centroid_row": row, "centroid_col": row, "image_id": 10}
                ],
            )

        assert db.get_used_centroids("test-classifier", 1, "A1") == {
            (10, 1, 1),
            (10, 3, 3),
        }

    def test_timepoint_zero_is_not_treated_as_missing(self, sample_classifier):
        """t=0 is a real timepoint, not a falsy 'no filter' sentinel."""
        db, _ = sample_classifier
        session_t0 = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/t0.npy"
        )
        session_t5 = db.create_session(
            "test-classifier", 1, "A1", 10, 5, "/t5.npy"
        )
        db.add_annotations(
            session_t0,
            [(0, "class_a")],
            cell_meta=[{"centroid_row": 1, "centroid_col": 1, "image_id": 10}],
        )
        db.add_annotations(
            session_t5,
            [(0, "class_b")],
            cell_meta=[{"centroid_row": 2, "centroid_col": 2, "image_id": 10}],
        )

        assert db.get_used_centroids(
            "test-classifier", 1, "A1", timepoint=0
        ) == {(10, 1, 1)}

    def test_annotations_without_centroids_are_ignored(
        self, sample_classifier
    ):
        """Pre-v2 rows have NULL centroids and must not crash the lookup."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )
        db.add_annotations(session_id, [(0, "class_a")])

        assert db.get_used_centroids("test-classifier", 1, "A1") == set()

    def test_annotation_without_image_id_is_ignored(self, sample_classifier):
        """A centroid with no source image can't form a valid key."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )
        db.add_annotations(
            session_id,
            [(0, "class_a")],
            cell_meta=[{"centroid_row": 5, "centroid_col": 6}],
        )

        assert db.get_used_centroids("test-classifier", 1, "A1") == set()


class TestStatistics:
    """Tests for statistics operations."""

    def test_get_class_distribution(self, sample_classifier):
        """Test getting class distribution."""
        db, _ = sample_classifier

        s1 = db.create_session("test-classifier", 1, "A1", 10, 0, "/path1.npy")
        s2 = db.create_session("test-classifier", 1, "A2", 11, 0, "/path2.npy")

        db.add_annotations(
            s1, [(0, "class_a"), (1, "class_a"), (2, "class_b")]
        )
        db.add_annotations(
            s2, [(0, "class_a"), (1, "class_c"), (2, "class_c")]
        )

        distribution = db.get_class_distribution("test-classifier")

        assert distribution == {"class_a": 3, "class_b": 1, "class_c": 2}

    def test_get_class_distribution_empty(self, sample_classifier):
        """Test class distribution when no annotations exist."""
        db, _ = sample_classifier
        distribution = db.get_class_distribution("test-classifier")
        assert distribution == {}

    def test_get_class_distribution_filtered(self, sample_classifier):
        """Test class distribution with filters."""
        db, _ = sample_classifier

        s1 = db.create_session("test-classifier", 1, "A1", 10, 0, "/path1.npy")
        s2 = db.create_session("test-classifier", 2, "A1", 10, 0, "/path2.npy")

        db.add_annotations(s1, [(0, "class_a"), (1, "class_b")])
        db.add_annotations(s2, [(0, "class_a"), (1, "class_a")])

        distribution = db.get_class_distribution(
            "test-classifier", plate_id=1
        )
        assert distribution == {"class_a": 1, "class_b": 1}

    def test_get_session_count(self, sample_classifier):
        """Test counting sessions for a classifier."""
        db, _ = sample_classifier

        db.create_session("test-classifier", 1, "A1", 10, 0, "/path1.npy")
        db.create_session("test-classifier", 1, "A2", 11, 0, "/path2.npy")
        db.create_session("test-classifier", 2, "B1", 12, 0, "/path3.npy")

        count = db.get_session_count("test-classifier")
        assert count == 3

    def test_get_session_count_empty(self, sample_classifier):
        """Test session count when no sessions exist."""
        db, _ = sample_classifier
        count = db.get_session_count("test-classifier")
        assert count == 0

    def test_get_total_annotations(self, sample_classifier):
        """Test counting total annotations for a classifier."""
        db, _ = sample_classifier

        s1 = db.create_session("test-classifier", 1, "A1", 10, 0, "/path1.npy")
        s2 = db.create_session("test-classifier", 1, "A2", 11, 0, "/path2.npy")

        db.add_annotations(s1, [(0, "class_a"), (1, "class_b")])
        db.add_annotations(s2, [(0, "class_a")])

        total = db.get_total_annotations("test-classifier")
        assert total == 3

    def test_get_total_annotations_empty(self, sample_classifier):
        """Test total annotations when none exist."""
        db, _ = sample_classifier
        total = db.get_total_annotations("test-classifier")
        assert total == 0


class TestCascadeDelete:
    """Tests for cascade delete behavior."""

    def test_delete_classifier_cascades(self, sample_classifier):
        """Test that deleting classifier deletes all related data."""
        db, _ = sample_classifier

        # Create sessions and annotations
        s1 = db.create_session("test-classifier", 1, "A1", 10, 0, "/path.npy")
        db.add_annotations(s1, [(0, "class_a")])

        # Delete classifier
        db.delete_classifier("test-classifier")

        # Session should be gone
        session = db.get_session_by_id(s1)
        assert session is None

        # Annotations should be gone
        annotations = db.get_annotations(s1)
        assert len(annotations) == 0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_metadata_serialization(self, sample_classifier):
        """Test that complex metadata is properly serialized."""
        db, _ = sample_classifier
        metadata = {
            "crop_size": 64,
            "channels": [0, 1, 2],
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        session_id = db.create_session(
            "test-classifier",
            1,
            "A1",
            10,
            0,
            "/path.npy",
            metadata=metadata,
        )

        session = db.get_session_by_id(session_id)
        assert session["metadata"] == metadata

    def test_special_characters_in_well(self, sample_classifier):
        """Test handling special characters in well positions."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1-2", 10, 0, "/path.npy"
        )
        assert session_id > 0

    def test_large_annotation_batch(self, sample_classifier):
        """Test adding many annotations at once."""
        db, _ = sample_classifier
        session_id = db.create_session(
            "test-classifier", 1, "A1", 10, 0, "/path.npy"
        )

        # Create 1000 annotations
        annotations = [(i, f"class_{i % 3}") for i in range(1000)]
        db.add_annotations(session_id, annotations)

        retrieved = db.get_annotations(session_id)
        assert len(retrieved) == 1000
