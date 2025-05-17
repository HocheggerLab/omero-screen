import pytest
from cellview.importers.experiments import ExperimentManager


class TestExperimentManager:
    """Tests for experiment management functionality."""

    def test_fetch_existing_experiments(self, db, test_projects):
        """Test fetching existing experiments for a project."""
        manager = ExperimentManager(db.conn)
        manager.state.project_id = test_projects[0][0]  # Set project ID

        # Add some test experiments
        db.conn.execute(
            """
            INSERT INTO experiments (project_id, experiment_name, description)
            VALUES (?, ?, ?)
            """,
            [test_projects[0][0], "Test Exp 1", "First test experiment"],
        )
        db.conn.execute(
            """
            INSERT INTO experiments (project_id, experiment_name, description)
            VALUES (?, ?, ?)
            """,
            [test_projects[0][0], "Test Exp 2", "Second test experiment"],
        )

        experiments = manager._fetch_existing_experiments()
        assert len(experiments) == 2
        assert experiments[0][1] == "Test Exp 1"
        assert experiments[1][1] == "Test Exp 2"

    def test_fetch_existing_experiments_no_project(self, db):
        """Test fetching experiments when no project is selected."""
        manager = ExperimentManager(db.conn)
        manager.state.project_id = None

        with pytest.raises(ValueError, match="No project selected"):
            manager._fetch_existing_experiments()

    def test_create_new_experiment(self, db, test_projects):
        """Test creating a new experiment."""
        manager = ExperimentManager(db.conn)
        manager.state.project_id = test_projects[0][0]

        experiment_id = manager._create_new_experiment("New Test Experiment")
        assert isinstance(experiment_id, int)

        # Verify experiment was created
        result = db.conn.execute(
            """
            SELECT experiment_name, description
            FROM experiments
            WHERE experiment_id = ?
            """,
            [experiment_id],
        ).fetchone()

        assert result is not None
        assert result[0] == "New Test Experiment"
        assert result[1] is None  # Description should be NULL by default

    def test_create_new_experiment_with_description(self, db, test_projects):
        """Test creating a new experiment with a description."""
        manager = ExperimentManager(db.conn)
        manager.state.project_id = test_projects[0][0]

        experiment_id = manager._create_new_experiment(
            "New Test Experiment", "Test description"
        )
        assert isinstance(experiment_id, int)

        # Verify experiment was created with description
        result = db.conn.execute(
            """
            SELECT experiment_name, description
            FROM experiments
            WHERE experiment_id = ?
            """,
            [experiment_id],
        ).fetchone()

        assert result is not None
        assert result[0] == "New Test Experiment"
        assert result[1] == "Test description"

    def test_create_new_experiment_no_project(self, db):
        """Test creating a new experiment when no project is selected."""
        manager = ExperimentManager(db.conn)
        manager.state.project_id = None

        with pytest.raises(ValueError, match="No project selected"):
            manager._create_new_experiment("New Test Experiment")

    def test_fetch_project_name(self, db, test_projects):
        """Test fetching project name."""
        manager = ExperimentManager(db.conn)
        manager.state.project_id = test_projects[0][0]

        project_name = manager._fetch_project_name()
        assert project_name == test_projects[0][1]

    def test_fetch_project_name_no_project(self, db):
        """Test fetching project name when no project is selected."""
        manager = ExperimentManager(db.conn)
        manager.state.project_id = None

        with pytest.raises(ValueError, match="No project selected"):
            manager._fetch_project_name()

    def test_fetch_project_name_invalid_id(self, db):
        """Test fetching project name with invalid project ID."""
        manager = ExperimentManager(db.conn)
        manager.state.project_id = 99999  # Non-existent project ID

        with pytest.raises(
            ValueError, match="Project with ID 99999 not found"
        ):
            manager._fetch_project_name()
