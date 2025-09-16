"""Configuration and Logging Utilities for OMERO Screen.

This module provides utilities for loading, validating, and managing environment variables and logging configuration for the OMERO Screen application. It supports loading environment variables from .env files (with environment-specific overrides), validates required variables, and configures logging with support for both console and file handlers.

Main Functions:
    - set_env_vars: Loads environment variables from .env files or the environment.
    - validate_env_vars: Ensures required environment variables are set.
    - get_logger: Returns a configured logger instance for the application/module.
    - configure_log_handler: Helper to configure logging handlers.
    - getenv_as_bool: Utility to parse boolean environment variables.

Attributes:
    project_root (Path): The root directory of the project, used to locate .env files.

Raises:
    OSError: If required configuration is missing or environment variables are not set.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

# Define project_root at module level
project_root = Path(__file__).parent.parent.parent.resolve()


def find_project_root() -> Path:
    """Find the project root directory using multiple strategies.

    Returns:
        Path to the project root directory.
    """
    # Strategy 1: Check for explicit override
    if override := os.environ.get("OMERO_SCREEN_PROJECT_ROOT"):
        override_path = Path(override)
        if override_path.exists():
            return override_path.resolve()

    # Strategy 2: Look for git repository root from current working directory
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    # Strategy 3: Look for common project markers from current working directory
    current = Path.cwd()
    while current != current.parent:
        # Look for project indicators
        if any(
            (current / marker).exists()
            for marker in [
                "pyproject.toml",
                "uv.lock",
                "CLAUDE.md",
                "packages",
            ]
        ):
            return current
        current = current.parent

    # Strategy 4: Traditional approach (relative to this file)
    # This works for development but may fail for installed packages
    file_based_root = Path(__file__).parent.parent.parent.resolve()
    if file_based_root.name != "site-packages":  # Avoid site-packages
        return file_based_root

    # Strategy 5: Fallback to current working directory
    return Path.cwd()


def set_env_vars() -> None:
    """Loads environment variables from configuration files or the environment.

    If the ENV variable is not set, defaults to 'development'. Attempts to load variables from a file named .env.{ENV} first; if not found, falls back to .env. If neither file exists, checks that all required environment variables are set in the environment.

    Raises:
        OSError: If no configuration file is found and required environment variables are missing.
    """
    # Determine the project root using robust discovery
    project_root = find_project_root()

    # Get environment, defaulting to development
    env = os.getenv("ENV", "development").lower()

    # Try environment-specific file first
    env_specific_path = project_root / f".env.{env}"
    if env_specific_path.exists():
        load_dotenv(env_specific_path)
        return

    # Fall back to default .env file
    default_env_path = project_root / ".env"
    if default_env_path.exists():
        load_dotenv(default_env_path)
        return

    # If no files found, check for required environment variables
    required_vars = [
        "ENV",
        "USERNAME",
        "PASSWORD",
        "HOST",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "ENABLE_CONSOLE_LOGGING",
        "ENABLE_FILE_LOGGING",
    ]

    if all(os.getenv(var) is not None for var in required_vars):
        # All required variables are present in environment
        return

    # If we get here, no configuration was found
    error_msg = "\n".join(
        [
            "No configuration found!",
            f"Current environment: {env}",
            f"Project root detected as: {project_root}",
            "Tried looking for:",
            f"  - {env_specific_path}",
            f"  - {default_env_path}",
            "And checked environment variables for:",
            f"  - {', '.join(required_vars)}",
            "\nSolutions:",
            f"  1. Create a .env.{env} file in {project_root}",
            "  2. Set OMERO_SCREEN_PROJECT_ROOT=/path/to/your/omero-screen",
            "  3. Set all required environment variables directly",
        ]
    )
    raise OSError(error_msg)


def validate_env_vars() -> None:
    """Validates that all required environment variables are set.

    Checks for the presence of required environment variables needed for logging configuration. Raises an OSError if any are missing.

    Raises:
        OSError: If one or more required environment variables are missing.
    """
    required_vars = ["LOG_LEVEL", "LOG_FILE_PATH"]
    if missing_vars := [var for var in required_vars if not os.getenv(var)]:
        raise OSError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def configure_log_handler(
    handler: logging.Handler,
    log_level: str,
    formatter: logging.Formatter,
    logger: logging.Logger,
) -> None:
    """Configures a logging handler with the specified settings and adds it to the given logger.

    Sets the log level and formatter for the provided handler, then attaches the handler to the specified logger instance.

    Args:
        handler (logging.Handler): The logging handler to configure.
        log_level (str): The logging level to set (e.g., 'DEBUG', 'INFO').
        formatter (logging.Formatter): The formatter to use for log messages.
        logger (logging.Logger): The logger to add the handler to.
    """
    handler.setLevel(getattr(logging, log_level, logging.DEBUG))
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the specified name, ensuring it is properly configured for the application.

    On the first call, this function sets up the root logger configuration, including log level,
    format, and handlers based on environment variables.
    Subsequent calls return loggers with the given name that inherit the root logger's configuration.

    Args:
        name (str): The logger name, typically __name__ from the calling module.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Handle the case when module is run directly (__main__)
    if name == "__main__":
        # Get the caller's file path
        import inspect

        frame = inspect.stack()[1]
        module_path = Path(frame.filename)
        try:
            # Get relative path from project root to the module
            rel_path = module_path.relative_to(project_root / "src")
            # Convert path to module notation (my_app.submodule.file)
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
            name = module_name
        except ValueError:
            # Fallback if file is not in src directory
            name = module_path.stem
    # Get or create the logger
    logger = logging.getLogger(name)

    # If the root logger isn't configured yet, configure it
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        validate_env_vars()

        # Retrieve logging configurations from environment variables
        LOG_LEVEL = os.getenv(
            "LOG_LEVEL", "INFO"
        ).upper()  # Default to INFO level
        LOG_FORMAT = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        )
        ENABLE_CONSOLE_LOGGING = getenv_as_bool("ENABLE_CONSOLE_LOGGING")
        ENABLE_FILE_LOGGING = getenv_as_bool("ENABLE_FILE_LOGGING")
        LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/app.log")
        LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 1048576))  # 1MB default
        LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

        # Configure the root logger
        root_logger.setLevel(
            getattr(logging, LOG_LEVEL, logging.DEBUG)
        )  # Use LOG_LEVEL from env

        # Prevent propagation beyond our root logger
        root_logger.propagate = False

        # Suppress specific external package logs
        omero_logger = logging.getLogger("omero")
        omero_logger.setLevel(logging.WARNING)
        omero_logger.propagate = True  # Allow OMERO logs to propagate to root

        # Formatter
        formatter = logging.Formatter(LOG_FORMAT)

        # Console Handler
        if ENABLE_CONSOLE_LOGGING:
            ch = logging.StreamHandler()
            ch.setLevel(
                getattr(logging, LOG_LEVEL, logging.DEBUG)
            )  # Set console handler to desired level
            configure_log_handler(ch, LOG_LEVEL, formatter, root_logger)

        # File Handler
        if ENABLE_FILE_LOGGING:
            log_path = Path(LOG_FILE_PATH)
            if log_dir := log_path.parent:
                log_dir.mkdir(parents=True, exist_ok=True)

            fh = RotatingFileHandler(
                LOG_FILE_PATH,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
            )
            fh.setLevel(
                getattr(logging, LOG_LEVEL, logging.DEBUG)
            )  # Set file handler to desired level
            configure_log_handler(fh, LOG_LEVEL, formatter, root_logger)

    # Suppress external logs
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("omero").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("cellpose").setLevel(logging.WARNING)

    return logger


def getenv_as_bool(name: str, default: bool = False) -> bool:
    """Get the boolean value of an environment variable.

    Args:
        name: Name of variable
        default: Default value
    Returns:
        True if the variable has value {true, 1, yes} (case insensitive)
    """
    v = os.getenv(name)
    return v.lower() in ["true", "1", "yes"] if v is not None else default
