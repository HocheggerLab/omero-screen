import duckdb
import pytest
from cellview.db import CellViewDB


class TestCellViewDBConnection:
    """Tests for database connection and initialization functionality."""

    def test_in_memory_connection(self, uninitialized_db):
        """Test that in-memory database connection is established."""
        assert uninitialized_db.conn is None  # Connection not established yet
        conn = uninitialized_db.connect()
        assert conn is not None
        assert uninitialized_db.conn is conn  # Connection is cached

    def test_connection_reuse(self, uninitialized_db):
        """Test that the same connection is reused when calling connect multiple times."""
        conn1 = uninitialized_db.connect()
        conn2 = uninitialized_db.connect()
        assert conn1 is conn2

    def test_schema_initialization(self, db):
        """Test that the database schema is properly initialized."""
        conn = db.connect()
        assert db._is_initialized()

        # Verify essential tables exist
        result = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
        """).fetchall()
        table_names = {row[0] for row in result}

        expected_tables = {
            "projects",
            "experiments",
            "repeats",
            "conditions",
            "condition_variables",
            "measurements",
        }
        assert expected_tables.issubset(table_names)

    def test_file_based_connection(self, tmp_path, mock_console):
        """Test connection with a file-based database."""
        db_path = tmp_path / "test.duckdb"
        db = CellViewDB(db_path=db_path)
        db.console = mock_console

        # Connect and verify file creation
        db.connect()
        assert db_path.exists()
        assert db._is_initialized()

    def test_custom_path_creation(self, tmp_path, mock_console):
        """Test that custom database directory is created if it doesn't exist."""
        deep_path = tmp_path / "deep" / "nested" / "path" / "test.duckdb"
        db = CellViewDB(db_path=deep_path)
        db.console = mock_console

        # Directory should not exist yet
        assert not deep_path.exists()

        # Connect should create the directory
        db.connect()
        assert deep_path.exists()

    def test_is_initialized_with_new_db(self, uninitialized_db):
        """Test _is_initialized returns False for new database and True after initialization."""
        uninitialized_db.conn = duckdb.connect(":memory:")
        assert uninitialized_db.conn is not None

        # Should not be initialized since tables don't exist
        assert not uninitialized_db._is_initialized()

        # Create tables and verify initialization
        uninitialized_db.create_tables()
        assert uninitialized_db._is_initialized()

        # Verify we can query a table
        result = uninitialized_db.conn.execute(
            "SELECT COUNT(*) FROM projects"
        ).fetchone()
        assert result is not None
        assert result[0] == 0  # Empty table


class TestCellViewDBSchema:
    """Tests for database schema creation and validation."""

    def test_all_tables_created(self, db):
        """Test that all expected tables are created."""
        conn = db.connect()

        # Test each table exists by trying to select from it
        expected_tables = {
            "projects",
            "experiments",
            "repeats",
            "conditions",
            "condition_variables",
            "measurements",
        }

        for table in expected_tables:
            result = conn.execute(f"SELECT 1 FROM {table} LIMIT 0").fetchall()
            assert isinstance(result, list)  # Table exists if query succeeds

    def test_sequences_created(self, db):
        """Test that all expected sequences are created."""
        conn = db.connect()
        # Get two sequential IDs and verify they increment
        id1 = conn.execute("SELECT nextval('project_id_seq')").fetchone()[0]
        id2 = conn.execute("SELECT nextval('project_id_seq')").fetchone()[0]
        assert id2 == id1 + 1  # Sequence is incrementing

    def test_projects_table_schema(self, db):
        """Test the schema of the projects table."""
        conn = db.connect()

        # Test we can insert and retrieve data with expected types
        conn.execute("""
            INSERT INTO projects (project_name, description)
            VALUES ('test_project', 'test description')
        """)

        result = conn.execute("""
            SELECT project_id, project_name, description
            FROM projects
            WHERE project_name = 'test_project'
        """).fetchone()

        assert result is not None
        assert isinstance(result[0], int)  # project_id is INTEGER
        assert isinstance(result[1], str)  # project_name is TEXT/VARCHAR
        assert isinstance(result[2], str)  # description is TEXT/VARCHAR

    def test_foreign_key_relationships(self, db):
        """Test that foreign key relationships are properly set up."""
        conn = db.connect()

        # Get a new project ID first
        project_id = conn.execute(
            "SELECT nextval('project_id_seq')"
        ).fetchone()[0]

        # Insert test data and verify constraints
        conn.execute(f"""
            INSERT INTO projects (project_id, project_name)
            VALUES ({project_id}, 'test_project')
        """)

        # Should work - valid foreign key
        conn.execute(f"""
            INSERT INTO experiments (project_id, experiment_name)
            VALUES ({project_id}, 'test_exp')
        """)

        # Should fail - invalid foreign key
        with pytest.raises(duckdb.Error):
            conn.execute("""
                INSERT INTO experiments (project_id, experiment_name)
                VALUES (99999, 'test_exp')
            """)

    def test_measurements_table_columns(self, db):
        """Test that the measurements table has all required columns with correct types."""
        conn = db.connect()

        # Set up required foreign key relationships
        project_id = conn.execute(
            "SELECT nextval('project_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO projects (project_id, project_name)
            VALUES ({project_id}, 'test_project')
        """)

        experiment_id = conn.execute(
            "SELECT nextval('experiment_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO experiments (experiment_id, project_id, experiment_name)
            VALUES ({experiment_id}, {project_id}, 'test_exp')
        """)

        repeat_id = conn.execute("SELECT nextval('repeat_id_seq')").fetchone()[
            0
        ]
        conn.execute(f"""
            INSERT INTO repeats (repeat_id, experiment_id, date, channel_0)
            VALUES ({repeat_id}, {experiment_id}, CURRENT_DATE, 'DAPI')
        """)

        condition_id = conn.execute(
            "SELECT nextval('condition_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO conditions (condition_id, repeat_id, well, well_id, cell_line)
            VALUES ({condition_id}, {repeat_id}, 'A1', 'A1', 'HeLa')
        """)

        # Insert a test measurement with all required NOT NULL fields
        measurement_id = conn.execute(
            "SELECT nextval('measurement_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO measurements (
                measurement_id, condition_id, image_id, timepoint, label, area_nucleus,
                "centroid-0-nuc", "centroid-1-nuc",
                intensity_min_DAPI_nucleus, intensity_mean_DAPI_nucleus, intensity_max_DAPI_nucleus
            )
            VALUES (
                {measurement_id}, {condition_id}, 1, 1, 'cell_1', 100.5,
                10.0, 20.0,
                150.5, 200.5, 250.5
            )
        """)

        # Verify numeric types
        result = conn.execute("""
            SELECT area_nucleus, intensity_mean_DAPI_nucleus
            FROM measurements
            LIMIT 1
        """).fetchone()

        assert result is not None
        assert isinstance(result[0], float)  # area_nucleus is DOUBLE
        assert isinstance(result[1], float)  # intensity is DOUBLE

    def test_unique_constraints(self, db):
        """Test that unique constraints are properly set up."""
        conn = db.connect()

        # Test unique constraint on project_name
        project_id = conn.execute(
            "SELECT nextval('project_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO projects (project_id, project_name)
            VALUES ({project_id}, 'test_project')
        """)

        with pytest.raises(duckdb.Error):
            new_id = conn.execute(
                "SELECT nextval('project_id_seq')"
            ).fetchone()[0]
            conn.execute(f"""
                INSERT INTO projects (project_id, project_name)
                VALUES ({new_id}, 'test_project')
            """)

        # Test unique constraint on repeat_id + well in conditions
        # Create project
        project_id = conn.execute(
            "SELECT nextval('project_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO projects (project_id, project_name)
            VALUES ({project_id}, 'project1')
        """)

        # Create experiment
        experiment_id = conn.execute(
            "SELECT nextval('experiment_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO experiments (experiment_id, project_id, experiment_name)
            VALUES ({experiment_id}, {project_id}, 'exp1')
        """)

        # Create repeat
        repeat_id = conn.execute("SELECT nextval('repeat_id_seq')").fetchone()[
            0
        ]
        conn.execute(f"""
            INSERT INTO repeats (repeat_id, experiment_id, date, channel_0)
            VALUES ({repeat_id}, {experiment_id}, CURRENT_DATE, 'DAPI')
        """)

        # Create condition
        condition_id = conn.execute(
            "SELECT nextval('condition_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO conditions (condition_id, repeat_id, well, well_id, cell_line)
            VALUES ({condition_id}, {repeat_id}, 'A1', 'A1', 'HeLa')
        """)

        # Try to create duplicate condition (same repeat_id and well)
        with pytest.raises(duckdb.Error):
            new_condition_id = conn.execute(
                "SELECT nextval('condition_id_seq')"
            ).fetchone()[0]
            conn.execute(f"""
                INSERT INTO conditions (condition_id, repeat_id, well, well_id, cell_line)
                VALUES ({new_condition_id}, {repeat_id}, 'A1', 'A1', 'HeLa')
            """)


class TestCellViewDBErrorHandling:
    """Tests for error handling and edge cases."""

    def test_connection_to_invalid_path(self, tmp_path, mock_console):
        """Test connecting to an invalid path."""
        invalid_path = tmp_path / "nonexistent" / "invalid.duckdb"
        db = CellViewDB(db_path=invalid_path)
        db.console = mock_console

        # Should create parent directories automatically
        db.connect()
        assert invalid_path.exists()

    def test_null_handling(self, db):
        """Test handling of NULL values in optional fields."""
        conn = db.connect()

        # Insert project with NULL description
        project_id = conn.execute(
            "SELECT nextval('project_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO projects (project_id, project_name, description)
            VALUES ({project_id}, 'test_project', NULL)
        """)

        # Insert experiment with NULL description
        experiment_id = conn.execute(
            "SELECT nextval('experiment_id_seq')"
        ).fetchone()[0]
        conn.execute(f"""
            INSERT INTO experiments (
                experiment_id, project_id, experiment_name, description
            )
            VALUES ({experiment_id}, {project_id}, 'test_exp', NULL)
        """)

        # Insert repeat with NULL optional channels
        repeat_id = conn.execute("SELECT nextval('repeat_id_seq')").fetchone()[
            0
        ]
        conn.execute(f"""
            INSERT INTO repeats (
                repeat_id, experiment_id, date, channel_0,
                channel_1, channel_2, channel_3, lab_member
            )
            VALUES (
                {repeat_id}, {experiment_id}, CURRENT_DATE, 'DAPI',
                NULL, NULL, NULL, NULL
            )
        """)

        # Verify NULL values are preserved
        result = conn.execute("""
            SELECT p.description, e.description,
                   r.channel_1, r.channel_2, r.channel_3, r.lab_member
            FROM projects p
            JOIN experiments e ON e.project_id = p.project_id
            JOIN repeats r ON r.experiment_id = e.experiment_id
            WHERE p.project_name = 'test_project'
        """).fetchone()

        assert result is not None
        assert all(
            v is None for v in result
        )  # All selected values should be NULL

    def test_special_characters(self, db):
        """Test handling of special characters in text fields."""
        conn = db.connect()

        special_chars = "Test'Project\"With;Special,Chars!@#$%^&*()"
        project_id = conn.execute(
            "SELECT nextval('project_id_seq')"
        ).fetchone()[0]

        # Insert project with special characters using parameter binding
        conn.execute(
            """
            INSERT INTO projects (project_id, project_name, description)
            VALUES (?, ?, ?)
        """,
            [project_id, special_chars, "Description with unicode: 你好世界"],
        )

        # Verify special characters are preserved
        result = conn.execute(
            """
            SELECT project_name, description
            FROM projects
            WHERE project_id = ?
        """,
            [project_id],
        ).fetchone()

        assert result is not None
        assert result[0] == special_chars
        assert result[1] == "Description with unicode: 你好世界"

    def test_connection_cleanup(self, tmp_path, mock_console):
        """Test connection cleanup behavior."""
        # Use a temporary file-based database for this test
        db_path = tmp_path / "cleanup_test.duckdb"
        db = CellViewDB(db_path=db_path)
        db.console = mock_console
        db.create_tables()  # Initialize schema

        conn1 = db.connect()
        assert db.conn is conn1

        # Create some test data
        conn1.execute("""
            INSERT INTO projects (project_name)
            VALUES ('cleanup_test')
        """)

        # Close connection
        conn1.close()
        db.conn = None

        # Reconnect and verify data persists
        conn2 = db.connect()
        result = conn2.execute("""
            SELECT project_name
            FROM projects
            WHERE project_name = 'cleanup_test'
        """).fetchone()

        assert result is not None
        assert result[0] == "cleanup_test"

    def test_large_text_fields(self, db):
        """Test handling of large text fields."""
        conn = db.connect()

        # Create a large description (100KB)
        large_text = "x" * 100_000
        project_id = conn.execute(
            "SELECT nextval('project_id_seq')"
        ).fetchone()[0]

        # Insert project with large description
        conn.execute(f"""
            INSERT INTO projects (project_id, project_name, description)
            VALUES ({project_id}, 'large_text_test', '{large_text}')
        """)

        # Verify large text is preserved
        result = conn.execute(f"""
            SELECT description
            FROM projects
            WHERE project_id = {project_id}
        """).fetchone()

        assert result is not None
        assert len(result[0]) == 100_000
        assert result[0] == large_text
