import logging
import os

import pytest

from omero_screen.config import setup_logging, validate_env_vars

logger = setup_logging("omero_screen")


@pytest.fixture
def clean_logger():
    """Fixture to clean logger handlers before and after tests"""
    logger = logging.getLogger("omero_screen")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    yield
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


# @pytest.fixture
# def clean_env():
#     """Fixture to clean environment variables and logger handlers before each test"""
#     # Store original environment variables
#     original_env = {
#         key: value
#         for key, value in os.environ.items()
#         if key in ["LOG_LEVEL", "LOG_FILE_PATH"]
#     }

#     # Clean up environment variables
#     for key in ["LOG_LEVEL", "LOG_FILE_PATH"]:
#         if key in os.environ:
#             del os.environ[key]

#     # Clean up existing handlers
#     logger = logging.getLogger("omero_utils")
#     for handler in logger.handlers[:]:
#         logger.removeHandler(handler)

#     yield

#     # Restore original environment variables
#     for key, value in original_env.items():
#         os.environ[key] = value

#     # Clean up handlers after test
#     for handler in logger.handlers[:]:
#         logger.removeHandler(handler)


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


def test_setup_logging_default_config(clean_env, clean_logger, tmp_path):
    """Test setup_logging with default configuration"""
    # Set required env vars and explicitly disable handlers
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["LOG_FILE_PATH"] = str(tmp_path / "test.log")
    os.environ["ENABLE_CONSOLE_LOGGING"] = "false"
    os.environ["ENABLE_FILE_LOGGING"] = "false"

    logger = setup_logging()

    # Check logger configuration
    assert logger.name == "omero_screen"
    assert logger.level == logging.INFO
    assert not logger.propagate  # Should not propagate to root logger

    # By default, no handlers should be attached
    assert len(logger.handlers) == 0


def test_setup_logging_with_console(clean_env):
    """Test setup_logging with console logging enabled"""
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_FILE_PATH"] = "test.log"
    os.environ["ENABLE_CONSOLE_LOGGING"] = "true"
    os.environ["ENABLE_FILE_LOGGING"] = (
        "false"  # Explicitly disable file logging
    )

    logger = setup_logging()

    # Verify console handler is present
    console_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.DEBUG


def test_setup_logging_with_file(clean_env, tmp_path):
    """Test setup_logging with file logging enabled"""
    log_file = tmp_path / "test.log"
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["LOG_FILE_PATH"] = str(log_file)
    os.environ["ENABLE_FILE_LOGGING"] = "true"

    logger = setup_logging()

    # Verify file handler is present
    file_handlers = [
        h
        for h in logger.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    assert file_handlers[0].level == logging.INFO
    assert file_handlers[0].baseFilename == str(log_file)


def test_setup_logging_with_custom_format(clean_env, clean_logger):
    """Test setup_logging with custom log format"""
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["LOG_FILE_PATH"] = "test.log"
    os.environ["ENABLE_CONSOLE_LOGGING"] = "true"
    custom_format = "%(levelname)s - %(message)s"
    os.environ["LOG_FORMAT"] = custom_format

    logger = setup_logging()

    # Verify formatter
    handler = logger.handlers[0]
    assert handler.formatter._fmt == custom_format


def test_setup_logging_invalid_level(clean_env, clean_logger):
    """Test setup_logging with invalid log level defaults to DEBUG"""
    os.environ["LOG_LEVEL"] = "INVALID_LEVEL"
    os.environ["LOG_FILE_PATH"] = "test.log"

    logger = setup_logging()

    assert logger.level == logging.DEBUG


def test_setup_logging_file_rotation_config(clean_env, clean_logger, tmp_path):
    """Test file rotation settings"""
    log_file = tmp_path / "test.log"
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["LOG_FILE_PATH"] = str(log_file)
    os.environ["ENABLE_FILE_LOGGING"] = "true"
    os.environ["LOG_MAX_BYTES"] = "1000"
    os.environ["LOG_BACKUP_COUNT"] = "3"

    logger = setup_logging()

    file_handler = next(
        h
        for h in logger.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    )
    assert file_handler.maxBytes == 1000
    assert file_handler.backupCount == 3
