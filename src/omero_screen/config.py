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
from rich.console import Console
from rich.logging import RichHandler

# Define project_root at module level
# Define project_root at module level
project_root = Path(__file__).parent.parent.parent.resolve()

_LOGGING_CONFIGURED = False

# Global console instance
_console: Console = Console()


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
        load_dotenv(env_specific_path, override=True)
        return

    # Fall back to default .env file
    default_env_path = project_root / ".env"
    if default_env_path.exists():
        load_dotenv(default_env_path, override=True)
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

    # Get the requested logger
    logger = logging.getLogger(name)

    # Configure logging system if not yet configured
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        _LOGGING_CONFIGURED = True

        validate_env_vars()

        # Load Config
        LOG_LEVEL = (
            os.getenv("LOG_LEVEL", "INFO").split("#")[0].strip().upper()
        )
        LOG_FORMAT = (
            os.getenv(
                "LOG_FORMAT",
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            )
            .split("#")[0]
            .strip()
        )
        ENABLE_CONSOLE_LOGGING = getenv_as_bool("ENABLE_CONSOLE_LOGGING")
        ENABLE_FILE_LOGGING = getenv_as_bool("ENABLE_FILE_LOGGING")
        LOG_FILE_PATH = (
            os.getenv("LOG_FILE_PATH", "logs/app.log").split("#")[0].strip()
        )

        # Ensure log path is absolute
        log_path_obj = Path(LOG_FILE_PATH)
        if not log_path_obj.is_absolute():
            log_path_obj = project_root / LOG_FILE_PATH
        LOG_FILE_PATH = str(log_path_obj)

        LOG_MAX_BYTES = getenv_as_int("LOG_MAX_BYTES", 1048576)
        LOG_BACKUP_COUNT = getenv_as_int("LOG_BACKUP_COUNT", 5)

        formatter = logging.Formatter(LOG_FORMAT)

        # Prepare Handlers
        handlers: list[logging.Handler] = []

        # Console Handler
        if ENABLE_CONSOLE_LOGGING:
            ch = RichHandler(console=get_console())
            ch.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
            # Rich console will add time, level and source file line to log messages.
            # Do not use the custom formatter.
            handlers.append(ch)

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
            fh.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
            fh.setFormatter(formatter)
            handlers.append(fh)

        root_logger = logging.getLogger()

        # Strategy Detection
        # If root logger has handlers, we assume we are running as a plugin (e.g. Napari)
        # and should avoid messing with the root logger to prevent console noise/duplication.
        is_plugin_mode = len(root_logger.handlers) > 0

        if is_plugin_mode:
            # Plugin Mode: Configure specific package loggers only and isolate them
            # We configure the top-level packages that we own
            packages_to_configure = [
                "omero_screen",
                "cellview",
                "omero_utils",
                "omero_screen_napari",
            ]

            for pkg_name in packages_to_configure:
                pkg_logger = logging.getLogger(pkg_name)
                pkg_logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
                pkg_logger.propagate = (
                    False  # Stop bubbling to Napari root logger
                )

                # Clear existing handlers to avoid duplication if re-run (though _LOGGING_CONFIGURED protects us)
                pkg_logger.handlers.clear()

                for h in handlers:
                    pkg_logger.addHandler(h)

        else:
            # Standalone Mode: Configure Root Logger
            root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
            root_logger.propagate = False

            # Suppress external logs
            logging.getLogger("omero").setLevel(logging.WARNING)
            logging.getLogger("omero").propagate = True

            for h in handlers:
                # Avoid adding duplicate handlers if they somehow exist
                if h not in root_logger.handlers:
                    root_logger.addHandler(h)

        # Common suppressions (apply to all modes)
        logging.getLogger("numba").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("omero").setLevel(logging.WARNING)
        logging.getLogger("fontTools").setLevel(logging.WARNING)
        logging.getLogger("cellpose").setLevel(logging.WARNING)

    return logger


def get_console() -> Console:
    """Get the global console.

    Returns:
        Rich console
    """
    return _console


def getenv_as_int(name: str, default: int) -> int:
    """Get the integer value of an environment variable, stripping comments.

    Args:
        name: Name of variable
        default: Default value
    Returns:
        Integer value or default if parsing fails
    """
    value = os.getenv(name)
    if value is None:
        return default

    # Remove inline comments and whitespace
    value = value.split("#")[0].strip()

    try:
        return int(value)
    except ValueError:
        return default


def getenv_as_bool(name: str, default: bool = False) -> bool:
    """Get the boolean value of an environment variable.

    Args:
        name: Name of variable
        default: Default value
    Returns:
        True if the variable has value {true, 1, yes} (case insensitive)
    """
    v = os.getenv(name)
    if v is None:
        return default
    v = v.split("#")[0].strip()
    return v.lower() in ["true", "1", "yes"]
