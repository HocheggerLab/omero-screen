import os

import pytest

from omero_screen.config import find_project_root, set_env_vars


@pytest.mark.parametrize(
    "env, expected_host, expected_loglevel",
    [
        ("development", "localhost", "DEBUG"),
        ("production", "ome2.hpc.sussex.ac.uk", "WARNING"),
    ],
)
def test_env_specific_config_loads(
    env,
    expected_host,
    expected_loglevel,
    test_env_files,
    monkeypatch,
    clean_env,
):
    """Test that environment-specific config files are loaded correctly."""
    os.environ["ENV"] = env

    def mock_find_project_root():
        return test_env_files

    monkeypatch.setattr("omero_screen.config.find_project_root", mock_find_project_root)

    set_env_vars()
    host = os.getenv("HOST")
    loglevel = os.getenv("LOG_LEVEL")

    assert host == expected_host, "Host is not correct"
    assert loglevel == expected_loglevel, "Log level is not correct"


def test_default_to_development(test_env_files, monkeypatch, clean_env):
    """Test that when ENV is not set, it defaults to development environment."""

    def mock_find_project_root():
        return test_env_files

    monkeypatch.setattr("omero_screen.config.find_project_root", mock_find_project_root)

    set_env_vars()
    host = os.getenv("HOST")
    loglevel = os.getenv("LOG_LEVEL")

    assert host == "localhost", "Host should default to development value"
    assert loglevel == "DEBUG", "Log level should default to development value"


def test_fallback_to_default_env(tmp_path, monkeypatch, clean_env):
    """Test fallback to .env when environment-specific file doesn't exist."""
    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text("HOST=fallback-host\nLOG_LEVEL=INFO")

    def mock_find_project_root():
        return tmp_path

    monkeypatch.setattr("omero_screen.config.find_project_root", mock_find_project_root)

    # Set an environment that doesn't have a specific config
    os.environ["ENV"] = "staging"

    set_env_vars()
    assert os.getenv("HOST") == "fallback-host", (
        "Should use fallback .env values"
    )
    assert os.getenv("LOG_LEVEL") == "INFO", "Should use fallback .env values"


def test_no_config_files_error(tmp_path, monkeypatch, clean_env):
    """Test that appropriate error is raised when no config files exist."""

    def mock_find_project_root():
        return tmp_path

    monkeypatch.setattr("omero_screen.config.find_project_root", mock_find_project_root)

    with pytest.raises(OSError) as exc_info:
        set_env_vars()

    error_msg = str(exc_info.value)
    assert "No configuration found!" in error_msg
    assert "Current environment: development" in error_msg
    assert str(tmp_path / ".env.development") in error_msg
    assert str(tmp_path / ".env") in error_msg


def test_find_project_root_with_override(monkeypatch, tmp_path):
    """Test that OMERO_SCREEN_PROJECT_ROOT override works."""
    test_root = tmp_path / "custom_project"
    test_root.mkdir()

    monkeypatch.setenv("OMERO_SCREEN_PROJECT_ROOT", str(test_root))

    result = find_project_root()
    assert result == test_root.resolve()


def test_find_project_root_git_detection(monkeypatch, tmp_path):
    """Test that git repository detection works."""
    # Create a mock git repo
    git_root = tmp_path / "repo"
    git_root.mkdir()
    (git_root / ".git").mkdir()

    # Create subdir and change to it
    subdir = git_root / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    # Clear any override
    monkeypatch.delenv("OMERO_SCREEN_PROJECT_ROOT", raising=False)

    result = find_project_root()
    assert result == git_root


def test_find_project_root_project_markers(monkeypatch, tmp_path):
    """Test that project markers detection works."""
    # Create a project with markers
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("test")

    # Create subdir and change to it
    subdir = project_root / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    # Clear any override
    monkeypatch.delenv("OMERO_SCREEN_PROJECT_ROOT", raising=False)

    result = find_project_root()
    assert result == project_root
