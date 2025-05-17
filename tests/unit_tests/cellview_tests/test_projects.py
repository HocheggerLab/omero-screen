import argparse
from unittest.mock import patch

import pytest
from cellview.importers.projects import ProjectManager
from cellview.utils import CellViewState
from cellview.utils.error_classes import DBError, StateError


def test_fetch_existing_projects(db, test_projects):
    """Test that existing projects are correctly fetched from the database."""
    manager = ProjectManager(db.conn)
    projects = manager._fetch_existing_projects()

    assert len(projects) == 3
    assert all(isinstance(p[0], int) for p in projects)  # IDs are integers
    assert all(isinstance(p[1], str) for p in projects)  # Names are strings
    assert all(
        isinstance(p[2], str) for p in projects
    )  # Descriptions are strings


def test_create_new_project(db):
    """Test creating a new project."""
    manager = ProjectManager(db.conn)
    new_id = manager._create_new_project("New Test Project")

    # Verify project was created
    result = db.conn.execute(
        "SELECT project_name FROM projects WHERE project_id = ?", [new_id]
    ).fetchone()

    assert result[0] == "New Test Project"


def test_create_duplicate_project(db, test_projects):
    """Test that creating a project with an existing name raises an error."""
    manager = ProjectManager(db.conn)

    with pytest.raises(DBError) as exc_info:
        manager._create_new_project("Existing Project")

    assert "Project name already exists" in str(exc_info.value)


def test_check_plate_exists(db, test_projects):
    """Test checking if a plate exists in the database."""
    manager = ProjectManager(db.conn)

    # First add a plate to test with
    db.conn.execute(
        """
        INSERT INTO experiments (project_id, experiment_name)
        VALUES (?, ?)
        """,
        [test_projects[0][0], "Test Experiment"],
    )
    experiment_id = db.conn.execute(
        "SELECT currval('experiment_id_seq')"
    ).fetchone()[0]

    db.conn.execute(
        """
        INSERT INTO repeats (experiment_id, plate_id, date, channel_0)
        VALUES (?, ?, CURRENT_DATE, 'DAPI')
        """,
        [experiment_id, 123],
    )

    # Test that checking for existing plate raises error
    with pytest.raises(DBError) as exc_info:
        manager._check_plate_exists(123)

    assert "Plate already exists" in str(exc_info.value)

    # Test that checking for non-existent plate doesn't raise error
    manager._check_plate_exists(999)  # Should not raise


@patch("rich.prompt.Prompt.ask")
def test_project_selection_existing_id(mock_prompt, db, test_projects):
    """Test selecting an existing project by ID."""
    manager = ProjectManager(db.conn)
    mock_prompt.return_value = str(
        test_projects[0][0]
    )  # Return first project's ID

    selected_id = manager._handle_project_selection(test_projects)
    assert selected_id == test_projects[0][0]


@patch("rich.prompt.Prompt.ask")
def test_project_selection_invalid_id(mock_prompt, db, test_projects):
    """Test selecting an invalid project ID."""
    manager = ProjectManager(db.conn)
    mock_prompt.return_value = "999"  # Invalid ID

    with pytest.raises(StateError) as exc_info:
        manager._handle_project_selection(test_projects)

    assert "Invalid project ID" in str(exc_info.value)


@patch("rich.prompt.Prompt.ask")
def test_project_selection_new_project(mock_prompt, db, test_projects):
    """Test creating a new project through selection."""
    manager = ProjectManager(db.conn)
    mock_prompt.return_value = "Brand New Project"

    new_id = manager._handle_project_selection(test_projects)

    # Verify new project was created
    result = db.conn.execute(
        "SELECT project_name FROM projects WHERE project_id = ?", [new_id]
    ).fetchone()

    assert result[0] == "Brand New Project"


@patch("rich.prompt.Prompt.ask")
def test_project_selection_existing_name(mock_prompt, db, test_projects):
    """Test selecting an existing project by name."""
    manager = ProjectManager(db.conn)
    mock_prompt.return_value = test_projects[0][1]  # First project's name

    selected_id = manager._handle_project_selection(test_projects)
    assert selected_id == test_projects[0][0]


def test_select_or_create_project_no_projects(db, sample_data_path):
    """Test project selection when no projects exist."""
    # Reset state to ensure clean slate
    state = CellViewState.get_instance()
    state.reset()

    manager = ProjectManager(db.conn)

    # Create args with required fields
    args = argparse.Namespace()
    args.csv = sample_data_path  # Use the sample data path

    # Initialize state properly
    state = CellViewState.get_instance(args)
    manager.state = state

    with patch("rich.prompt.Prompt.ask", return_value="First Project"):
        _project_id = manager.select_or_create_project()

    # Verify project was created
    result = db.conn.execute(
        "SELECT project_name FROM projects WHERE project_id = (SELECT currval('project_id_seq'))"
    ).fetchone()

    assert result[0] == "First Project"
