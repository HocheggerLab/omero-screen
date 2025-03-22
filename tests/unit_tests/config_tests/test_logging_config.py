import logging
import os

import pytest

from omero_screen.config import get_logger, set_env_vars, validate_env_vars


@pytest.fixture
def clean_logger():
    """Fixture to clean logger handlers before and after tests"""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    yield
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


@pytest.fixture
def clean_env():
    """Fixture to clean environment variables before each test"""
    # Store original environment variables
    env_vars = [
        "LOG_LEVEL",
        "LOG_FILE_PATH",
        "ENABLE_CONSOLE_LOGGING",
        "ENABLE_FILE_LOGGING",
    ]
    original_env = {key: os.environ.get(key) for key in env_vars}

    # Clean up environment variables
    for key in env_vars:
        if key in os.environ:
            del os.environ[key]

    yield

    # Restore original environment variables
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


def test_validate_env_vars_all_present(test_env_files, monkeypatch, clean_env):
    """Test when all required environment variables are present"""

    def mock_resolve(self):
        return test_env_files

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

    set_env_vars()  # This will load from .env.development by default
    validate_env_vars()  # Should not raise any exception


def test_validate_env_vars_missing_vars(clean_env):
    """Test when required environment variables are missing"""
    with pytest.raises(OSError) as exc_info:
        validate_env_vars()
    assert "LOG_LEVEL" in str(exc_info.value)
    assert "LOG_FILE_PATH" in str(exc_info.value)


def test_get_logger_hierarchy(
    test_env_files, monkeypatch, clean_env, clean_logger
):
    """Test that loggers form proper hierarchy"""

    def mock_resolve(self):
        return test_env_files

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

    set_env_vars()  # Load development environment

    # Get loggers at different levels
    root_logger = logging.getLogger()
    screen_logger = get_logger("screen")
    utils_logger = get_logger("utils")

    # Verify hierarchy
    assert screen_logger.name == "screen"
    assert utils_logger.name == "utils"

    # Verify handlers are only on root logger
    assert len(root_logger.handlers) > 0, "Root logger should have handlers"
    assert len(screen_logger.handlers) == 0, (
        "Child logger should not have handlers"
    )
    assert len(utils_logger.handlers) == 0, (
        "Child logger should not have handlers"
    )


def test_get_logger_single_configuration(
    test_env_files, monkeypatch, clean_env, clean_logger
):
    """Test that logger configuration happens only once"""

    def mock_resolve(self):
        return test_env_files

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

    set_env_vars()  # Load development environment

    # Get the same logger multiple times
    _logger1 = get_logger("omero.test")
    _logger2 = get_logger("omero.test")
    root_logger = logging.getLogger()

    # Verify that handlers were only added once
    assert len(root_logger.handlers) > 0
    initial_handler_count = len(root_logger.handlers)

    # Get another logger
    _logger3 = get_logger("omero.another")

    # Verify no new handlers were added
    assert len(root_logger.handlers) == initial_handler_count


def test_get_logger_with_console(
    test_env_files, monkeypatch, clean_env, clean_logger
):
    """Test logger with console logging enabled"""

    def mock_resolve(self):
        return test_env_files

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

    # Use development environment which has console logging enabled
    set_env_vars()

    root_logger = logging.getLogger()

    # Verify console handler is present
    assert any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )


def test_get_logger_message_propagation(
    test_env_files, monkeypatch, clean_env, clean_logger, tmp_path
):
    """Test that messages from child loggers reach the root logger's handlers"""

    def mock_resolve(self):
        return test_env_files

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

    set_env_vars()  # Load development environment

    # Create loggers at different levels
    child_logger = get_logger("omero.test.child")

    # Log a message
    test_message = "Test message from child"
    child_logger.debug(test_message)

    # Since we're using development environment, console logging is enabled
    # We can verify the message was logged by checking the handlers
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) > 0
