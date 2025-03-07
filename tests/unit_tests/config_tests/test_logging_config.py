import logging
import os

import pytest

from omero_screen.config import get_logger, validate_env_vars


@pytest.fixture
def clean_logger():
    """Fixture to clean logger handlers before and after tests"""
    logger = logging.getLogger("omero")  # Clean the root logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    yield
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


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


def test_validate_env_vars_all_present(clean_env):
    """Test when all required environment variables are present"""
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_FILE_PATH"] = "/path/to/log"

    # Should not raise any exception
    validate_env_vars()


def test_validate_env_vars_missing_log_level(clean_env):
    """Test when LOG_LEVEL is missing"""
    os.environ["LOG_FILE_PATH"] = "/path/to/log"

    with pytest.raises(OSError) as exc_info:
        validate_env_vars()
    assert "LOG_LEVEL" in str(exc_info.value)


def test_validate_env_vars_missing_log_file_path(clean_env):
    """Test when LOG_FILE_PATH is missing"""
    os.environ["LOG_LEVEL"] = "DEBUG"

    with pytest.raises(OSError) as exc_info:
        validate_env_vars()
    assert "LOG_FILE_PATH" in str(exc_info.value)


def test_validate_env_vars_all_missing(clean_env):
    """Test when all required environment variables are missing"""
    with pytest.raises(OSError) as exc_info:
        validate_env_vars()
    assert "LOG_LEVEL" in str(exc_info.value)
    assert "LOG_FILE_PATH" in str(exc_info.value)


# --------------------------------test logger--------------------------------


def test_get_logger_hierarchy(clean_env, clean_logger):
    """Test that loggers form proper hierarchy"""
    # Set required environment variables
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_FILE_PATH"] = "test.log"
    os.environ["ENABLE_CONSOLE_LOGGING"] = (
        "true"  # Enable at least one handler
    )

    # Get loggers at different levels
    root_logger = get_logger("omero")
    screen_logger = get_logger("omero.screen")
    utils_logger = get_logger("omero.utils")

    # Verify hierarchy
    assert root_logger.name == "omero"
    assert screen_logger.name == "omero.screen"
    assert utils_logger.name == "omero.utils"

    # Verify handlers are only on root logger
    assert len(root_logger.handlers) > 0
    assert len(screen_logger.handlers) == 0
    assert len(utils_logger.handlers) == 0


def test_get_logger_single_configuration(clean_env, clean_logger, tmp_path):
    """Test that logger configuration happens only once"""
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["LOG_FILE_PATH"] = str(tmp_path / "test.log")
    os.environ["ENABLE_CONSOLE_LOGGING"] = "true"
    os.environ["ENABLE_FILE_LOGGING"] = "true"

    # Get the same logger multiple times
    _logger1 = get_logger(
        "omero.test"
    )  # Prefix with underscore to indicate intentionally unused
    _logger2 = get_logger(
        "omero.test"
    )  # Prefix with underscore to indicate intentionally unused
    root_logger = logging.getLogger("omero")

    # Verify that handlers were only added once
    assert len(root_logger.handlers) > 0
    initial_handler_count = len(root_logger.handlers)

    # Get another logger
    _logger3 = get_logger(
        "omero.another"
    )  # Prefix with underscore to indicate intentionally unused

    # Verify no new handlers were added
    assert len(root_logger.handlers) == initial_handler_count


def test_get_logger_with_console(clean_env, clean_logger):
    """Test logger with console logging enabled"""
    os.environ["ENABLE_CONSOLE_LOGGING"] = "true"
    os.environ["ENABLE_FILE_LOGGING"] = "false"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_FILE_PATH"] = "test.log"  # Required but won't be used

    _logger = get_logger(
        "omero.test"
    )  # Prefix with underscore to indicate intentionally unused
    root_logger = logging.getLogger("omero")

    # Verify console handler is present
    assert any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )
    assert not any(
        isinstance(h, logging.FileHandler) for h in root_logger.handlers
    )


def test_get_logger_with_file(clean_env, clean_logger, tmp_path):
    """Test logger with file logging enabled"""
    log_file = tmp_path / "test.log"
    os.environ["ENABLE_CONSOLE_LOGGING"] = "false"
    os.environ["ENABLE_FILE_LOGGING"] = "true"
    os.environ["LOG_FILE_PATH"] = str(log_file)
    os.environ["LOG_LEVEL"] = "DEBUG"

    _logger = get_logger(
        "omero.test"
    )  # Prefix with underscore to indicate intentionally unused
    root_logger = logging.getLogger("omero")

    # Verify file handler is present
    assert any(
        isinstance(h, logging.FileHandler) for h in root_logger.handlers
    )
    assert not any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )


def test_get_logger_message_propagation(clean_env, clean_logger, tmp_path):
    """Test that messages from child loggers reach the root logger's handlers"""
    log_file = tmp_path / "test.log"
    os.environ["ENABLE_FILE_LOGGING"] = "true"
    os.environ["LOG_FILE_PATH"] = str(log_file)
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Create loggers at different levels
    child_logger = get_logger("omero.test.child")

    # Log a message
    test_message = "Test message from child"
    child_logger.debug(test_message)

    # Verify message appears in log file
    with open(log_file) as f:
        log_content = f.read()
        assert test_message in log_content


def test_get_logger_log_level_inheritance(clean_env, clean_logger):
    """Test that child loggers inherit the log level from root logger"""
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["LOG_FILE_PATH"] = "test.log"
    os.environ["ENABLE_CONSOLE_LOGGING"] = "true"

    root_logger = get_logger("omero")
    child_logger = get_logger("omero.test")

    assert root_logger.level == logging.INFO
    assert (
        child_logger.level == logging.NOTSET
    )  # Indicates it inherits from parent
