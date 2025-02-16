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


def setup_logging(logger_name: str = "omero_screen") -> logging.Logger:
    """
    Configure logging based on environment variables.
    """
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

    # Configure the root logger to a less verbose level
    logging.basicConfig(level=logging.WARNING)

    # Create and configure your application's main logger
    app_logger_name = logger_name
    app_logger = logging.getLogger(app_logger_name)
    app_logger.setLevel(
        getattr(logging, LOG_LEVEL, logging.DEBUG)
    )  # Default to DEBUG if invalid level

    # Prevent propagation to the root logger
    app_logger.propagate = False

    # Formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Console Handler
    if ENABLE_CONSOLE_LOGGING:
        ch = logging.StreamHandler()
        configure_log_handler(ch, LOG_LEVEL, formatter, app_logger)
    # File Handler
    if ENABLE_FILE_LOGGING:
        if log_dir := os.path.dirname(LOG_FILE_PATH):
            os.makedirs(log_dir, exist_ok=True)

        fh = RotatingFileHandler(
            LOG_FILE_PATH, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        configure_log_handler(fh, LOG_LEVEL, formatter, app_logger)
    return app_logger


# TODO Rename this here and in `setup_logging`
def configure_log_handler(
    arg0: logging.Handler,
    LOG_LEVEL: str,
    formatter: logging.Formatter,
    app_logger: logging.Logger,
) -> None:
    arg0.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
    arg0.setFormatter(formatter)
    app_logger.addHandler(arg0)
