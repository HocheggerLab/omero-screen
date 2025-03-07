import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

# Define project_root at module level
project_root = Path(__file__).parent.parent.parent.resolve()


def set_env_vars() -> None:
    """
    Load environment variables based on the ENV variable.
    """
    # Determine the project root (adjust as necessary)
    project_root = Path(__file__).parent.parent.parent.resolve()
    # Set default environment variables if .env doesn't exist
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("LOG_FILE_PATH", "logs/app.log")
    os.environ.setdefault("ENABLE_CONSOLE_LOGGING", "True")
    os.environ.setdefault("ENABLE_FILE_LOGGING", "True")

    # Path to the minimal .env file (optional)
    minimal_env_path = project_root / ".env"
    print(minimal_env_path)
    # Load the minimal .env file to get the ENV variable (if exists)
    if minimal_env_path.exists():
        load_dotenv(minimal_env_path)

    # Retrieve the ENV variable, default to 'development' if not set
    ENV = os.getenv("ENV", "development").lower()

    # Path to the environment-specific .env file
    env_specific_path = project_root / f".env.{ENV}"

    # Load the environment-specific .env file if it exists
    if env_specific_path.exists():
        load_dotenv(env_specific_path, override=True)
    else:
        print(
            f"Warning: {env_specific_path} not found. Using default configurations."
        )


def validate_env_vars() -> None:
    """
    Validate that all required environment variables are set.
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
    """Configure a logging handler with the specified settings.

    Args:
        handler: The logging handler to configure
        log_level: The logging level to set
        formatter: The formatter to use for log messages
        logger: The logger to add the handler to
    """
    handler.setLevel(getattr(logging, log_level, logging.DEBUG))
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name, ensuring it's properly configured.
    If this is the first call, it will set up the root logger configuration.
    Subsequent calls will return appropriately named loggers that inherit the configuration.

    Args:
        name: The logger name, typically __name__ from the calling module

    Returns:
        logging.Logger: A configured logger instance
    """
    # Get or create the logger
    logger = logging.getLogger(name)

    # If the root logger isn't configured yet, configure it
    root_logger = logging.getLogger("omero")
    if not root_logger.handlers:
        validate_env_vars()

        # Retrieve logging configurations from environment variables
        LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
        LOG_FORMAT = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        )
        ENABLE_CONSOLE_LOGGING = os.getenv(
            "ENABLE_CONSOLE_LOGGING", "False"
        ).lower() in ["true", "1", "yes"]
        ENABLE_FILE_LOGGING = os.getenv(
            "ENABLE_FILE_LOGGING", "False"
        ).lower() in ["true", "1", "yes"]
        LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/app.log")
        LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 1048576))  # 1MB default
        LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

        # Configure the root logger
        root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))

        # Prevent propagation beyond our root logger
        root_logger.propagate = False

        # Formatter
        formatter = logging.Formatter(LOG_FORMAT)

        # Console Handler
        if ENABLE_CONSOLE_LOGGING:
            ch = logging.StreamHandler()
            configure_log_handler(ch, LOG_LEVEL, formatter, root_logger)

        # File Handler
        if ENABLE_FILE_LOGGING:
            if log_dir := os.path.dirname(LOG_FILE_PATH):
                os.makedirs(log_dir, exist_ok=True)

            fh = RotatingFileHandler(
                LOG_FILE_PATH,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
            )
            configure_log_handler(fh, LOG_LEVEL, formatter, root_logger)

    return logger
