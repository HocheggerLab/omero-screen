import os

import pytest

from omero_screen.config import set_env_vars


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

    def mock_resolve(self):
        return test_env_files

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

    set_env_vars()
    host = os.getenv("HOST")
    loglevel = os.getenv("LOG_LEVEL")

    assert host == expected_host, "Host is not correct"
    assert loglevel == expected_loglevel, "Log level is not correct"


def test_default_to_development(test_env_files, monkeypatch, clean_env):
    """Test that when ENV is not set, it defaults to development environment."""

    def mock_resolve(self):
        return test_env_files

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

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

    def mock_resolve(self):
        return tmp_path

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

    # Set an environment that doesn't have a specific config
    os.environ["ENV"] = "staging"

    set_env_vars()
    assert os.getenv("HOST") == "fallback-host", (
        "Should use fallback .env values"
    )
    assert os.getenv("LOG_LEVEL") == "INFO", "Should use fallback .env values"


def test_no_config_files_error(tmp_path, monkeypatch, clean_env):
    """Test that appropriate error is raised when no config files exist."""

    def mock_resolve(self):
        return tmp_path

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

    with pytest.raises(OSError) as exc_info:
        set_env_vars()

    error_msg = str(exc_info.value)
    assert "No configuration found!" in error_msg
    assert "Current environment: development" in error_msg
    assert str(tmp_path / ".env.development") in error_msg
    assert str(tmp_path / ".env") in error_msg
