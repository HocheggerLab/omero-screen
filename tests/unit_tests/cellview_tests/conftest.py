from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cellview.db.db import CellViewDB
from rich.console import Console

# -------------------Data Base Fixtures--------------------------------


@pytest.fixture(scope="session")
def mock_console():
    """Provides a mock console to prevent output during tests."""
    console = Console(quiet=True)
    return console


@pytest.fixture(scope="function")
def db(mock_console):
    """Provides a fresh database instance with schema initialized.
    Using function scope to ensure complete isolation between tests.
    """
    db = CellViewDB(db_path=Path(":memory:"))
    db.console = mock_console
    db.create_tables()  # Initialize schema
    return db


@pytest.fixture(scope="function")
def uninitialized_db(mock_console):
    """Provides a fresh database instance without connecting or initializing schema.
    Useful for testing connection initialization.
    """
    db = CellViewDB(db_path=Path(":memory:"))
    db.console = mock_console
    return db


@pytest.fixture(autouse=True)
def setup_fresh_db(db):
    """Automatically ensures each test starts with a fresh schema.
    This fixture runs automatically for each test.
    """
    db.create_tables()
    yield db
    # Connection will be automatically closed when db goes out of scope
    # since we're using in-memory database


# -------------------Sample Data Fixtures--------------------------------
@pytest.fixture(scope="session")
def sample_data_path(tmp_path_factory) -> Path:
    """Create a sample CSV file with synthetic data matching the real data structure."""
    # Create a temporary directory for test data
    tmp_dir = tmp_path_factory.mktemp("test_data")
    csv_path = tmp_dir / "240326_test_data_cc.csv"

    # Generate sample data
    num_rows = 20  # 2 wells × 10 cells
    data = {
        "experiment": ["test_exp"] * num_rows,
        "well": ["A01", "A02"] * 10,  # Alternate between two wells
        "cell_line": ["MCF10A"] * num_rows,
        "clone": ["WT"] * num_rows,
        "nuclei4": range(1, num_rows + 1),
        "label": ["cell_" + str(i) for i in range(1, num_rows + 1)],
        "area_nucleus": np.random.uniform(100, 200, num_rows),
        "plate_id": [1] * num_rows,
        "well_id": [1, 2] * 10,
        "image_id": [1] * num_rows,
    }

    # Add intensity measurements for different channels (DAPI, Tub, p21, EdU)
    channels = ["DAPI", "Tub", "p21", "EdU"]
    regions = ["nucleus", "cell", "cyto"]
    stats = ["max", "min", "mean"]

    for channel in channels:
        for region in regions:
            for stat in stats:
                col_name = f"intensity_{stat}_{channel}_{region}"
                data[col_name] = np.random.uniform(100, 1000, num_rows)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture(scope="session")
def sample_data_missing_columns(tmp_path_factory) -> Path:
    """Create a sample CSV file with missing required columns for testing error handling."""
    tmp_dir = tmp_path_factory.mktemp("test_data")
    csv_path = tmp_dir / "240326_test_data_missing_cols.csv"

    # Generate minimal data without required columns
    data = {
        "experiment": ["test_exp"] * 10,
        "well": ["A01"] * 10,
        "label": ["cell_" + str(i) for i in range(1, 11)],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture(scope="session")
def sample_data_multiple_plates(tmp_path_factory) -> Path:
    """Create a sample CSV file with multiple plate_ids for testing error handling."""
    tmp_dir = tmp_path_factory.mktemp("test_data")
    csv_path = tmp_dir / "240326_test_data_multiple_plates.csv"

    # Copy structure from sample_data but with multiple plate_ids
    num_rows = 20
    data = {
        "experiment": ["test_exp"] * num_rows,
        "well": ["A01", "A02"] * 10,
        "plate_id": [1, 2] * 10,  # Alternate between two plate IDs
        "well_id": [1, 2] * 10,
        "image_id": [1] * num_rows,
    }

    # Add some basic intensity measurements
    for stat in ["max", "min", "mean"]:
        col_name = f"intensity_{stat}_DAPI_nucleus"
        data[col_name] = np.random.uniform(100, 1000, num_rows)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture(scope="function")
def test_projects(db):
    """Adds test projects to the database and returns their IDs and names."""
    projects = [
        ("Test Project 1", "First test project"),
        ("Test Project 2", "Second test project"),
        ("Existing Project", "Project that already exists"),
    ]

    project_ids = []
    for name, description in projects:
        result = db.conn.execute(
            """
            INSERT INTO projects (project_name, description)
            VALUES (?, ?)
            RETURNING project_id
            """,
            [name, description],
        ).fetchone()
        project_ids.append((result[0], name, description))

    return project_ids
