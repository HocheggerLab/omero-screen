"""Configuration and Logging Utilities for OMERO Screen.

This module provides utilities for loading, validating, and managing environment
variables and logging configuration for the OMERO Screen application. It supports
loading environment variables from .env files (with environment-specific
overrides), validates required variables, and configures logging on top of
`loguru` with support for both a Rich-rendered console sink and a rotating file
sink.

Logging architecture (loguru):
    - There is a single global `loguru` logger. Modules use it directly via
      ``from loguru import logger`` — there is no per-module logger and no
      ``get_logger`` seam. Because the logger is one process-global object, a
      single ``configure_logging()`` call governs logging for the whole process.
    - Configuration happens **once, at an application entry point** (a CLI
      ``main()`` or, for napari, a widget factory), never as an import side
      effect. Library packages (omero-utils, omero-screen-plots) simply emit;
      they don't configure or disable loguru, so their records flow into
      whatever sinks the application set up.
    - Production defaults are baked in (rotating file sink at INFO, console
      off); ``OMERO_SCREEN_LOG_LEVEL`` / ``OMERO_SCREEN_LOG_FILE`` provide
      optional overrides, read at call time (argument > env var > default).
    - Third-party stdlib logging (omero, cellpose, numba, matplotlib, fontTools)
      is redirected into loguru via an `InterceptHandler` on the stdlib root,
      except in plugin mode (napari) where the host owns the root.

Main Functions:
    - set_env_vars: Loads environment variables from .env files or the environment.
    - configure_logging: Configures the global loguru logger (call once at an
      entry point). ``configure_logging_once`` is the idempotent variant for
      activation points that may fire repeatedly (e.g. napari widgets).
    - is_level_enabled: Guard for expensive, level-gated computation.
    - get_console: Returns the shared Rich console.
    - getenv_as_bool / getenv_as_int: Parse typed environment variables.

Attributes:
    project_root (Path): The root directory of the project, used to locate .env files.

Raises:
    OSError: If required configuration is missing or environment variables are not set.
"""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger as _loguru_logger
from rich.console import Console
from rich.logging import RichHandler

# Define project_root at module level
project_root = Path(__file__).parent.parent.parent.resolve()

# Whether configure_logging() has run in this process.
_CONFIGURED = False

# Built-in defaults (production-safe): logging works with zero env/config.
_DEFAULT_LEVEL = "INFO"
_DEFAULT_LOG_FILE = "logs/app.log"

# Default loguru format for the file sink (the console sink delegates rendering
# to Rich). ``{name}``/``{line}`` are auto-derived by loguru from the call site.
_LOGURU_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}"
)

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

    # If no files found, check for required environment variables. Logging is
    # not listed: it is configured in code with safe defaults (see
    # configure_logging), so it never gates startup.
    required_vars = [
        "ENV",
        "USERNAME",
        "PASSWORD",
        "HOST",
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


# Third-party loggers we keep quiet (their records are intercepted into loguru).
_NOISY_LIBRARIES = ("omero", "cellpose", "numba", "matplotlib", "fontTools")


class _InterceptHandler(logging.Handler):
    """Stdlib handler that forwards every record into loguru.

    Installed on the stdlib root logger (standalone mode only) so that
    third-party libraries logging via the standard library are rendered through
    our loguru sinks with consistent formatting. This is the canonical loguru
    recipe (see the loguru docs, "Entirely compatible with standard logging").
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        # Map the stdlib level to a loguru level name, falling back to the
        # numeric level if loguru doesn't know the name.
        try:
            level: str | int = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk back out of the logging machinery so the originating module,
        # function and line are reported rather than this handler.
        frame: Any = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(
    *,
    level: str | None = None,
    console: bool = False,
    log_file: str | None = None,
    diagnose: bool = False,
    plugin: bool = False,
) -> None:
    """Configure the global loguru logger for an application entry point.

    Call this ONCE, explicitly, from an application's ``main()`` (or, for the
    napari plugin, from the top-level widget — see ``configure_logging_once``).
    Libraries must never call it. Because loguru's ``logger`` is a single
    process-global object, this one call governs logging for every module in
    every package in the process.

    Production defaults (zero config required): a rotating **file** sink at
    INFO, **console off** (the Rich progress display / panels are the user
    channel — a console log sink would corrupt them). Calling it again simply
    re-applies the configuration (``logger.remove()`` first), so it is safe to
    call from an idempotent widget hook.

    Resolution precedence for ``level``/``log_file`` is **argument > env var >
    default** (env vars ``OMERO_SCREEN_LOG_LEVEL`` / ``OMERO_SCREEN_LOG_FILE``,
    read here at call time — never at import).

    Args:
        level: Minimum level (e.g. ``"DEBUG"``). Falls back to
            ``OMERO_SCREEN_LOG_LEVEL`` then ``"INFO"``.
        console: Add a Rich console sink (sharing ``progress.py``'s ``Console``).
            Off by default; turn on for ``--verbose`` / interactive debugging.
        log_file: File-sink path. Falls back to ``OMERO_SCREEN_LOG_FILE`` then
            ``logs/app.log``. ``"none"``/``"off"``/``""`` disables the file sink.
        diagnose: loguru variable-value tracebacks. Off in production (can leak
            values/secrets into the log); on for ``--verbose``.
        plugin: When embedded in a host that owns the stdlib root logger
            (napari), skip installing the stdlib InterceptHandler.
    """
    global _CONFIGURED

    level = (
        level or os.getenv("OMERO_SCREEN_LOG_LEVEL") or _DEFAULT_LEVEL
    ).upper()
    if log_file is None:
        log_file = os.getenv("OMERO_SCREEN_LOG_FILE", _DEFAULT_LOG_FILE)

    # Drop loguru's default stderr sink (and anything from a prior call).
    _loguru_logger.remove()

    # File sink (production default) — native rotation/retention; enqueue makes
    # it safe for the pipeline's worker threads.
    if log_file and log_file.strip().lower() not in ("none", "off", ""):
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = project_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            str(log_path),
            level=level,
            format=_LOGURU_FILE_FORMAT,
            rotation="10 MB",
            retention=5,
            enqueue=True,
            backtrace=False,
            diagnose=diagnose,
        )

    # Console sink (opt-in) — reuse progress.py's shared Rich Console so logs
    # and the live progress display don't clobber each other.
    if console:
        _loguru_logger.add(
            RichHandler(console=get_console()),
            level=level,
            format="{message}",
            backtrace=False,
            diagnose=diagnose,
        )

    # Redirect third-party stdlib logging into loguru — unless a host (napari)
    # owns the root logger, in which case leave it alone.
    if not plugin:
        logging.basicConfig(
            handlers=[_InterceptHandler()], level=0, force=True
        )
    for lib in _NOISY_LIBRARIES:
        logging.getLogger(lib).setLevel(logging.WARNING)

    _CONFIGURED = True


def configure_logging_once(**kwargs: Any) -> None:
    """Configure logging only if no entry point has configured it yet.

    For activation points that may fire repeatedly and aren't the sole entry
    (e.g. a napari widget ``__init__``): the first call configures, later calls
    are no-ops, and an earlier explicit ``configure_logging()`` from a ``main()``
    always wins.
    """
    if not _CONFIGURED:
        configure_logging(**kwargs)


def get_console() -> Console:
    """Get the global console.

    Returns:
        Rich console
    """
    return _console


def is_level_enabled(level: str = "DEBUG") -> bool:
    """Return whether any configured sink would emit at ``level``.

    loguru has no public ``isEnabledFor``; this inspects the active sinks'
    minimum level. Use it to guard expensive, level-gated computation (e.g.
    building a diagnostic string only when DEBUG logging is on). Falls back to
    ``True`` if the level can't be determined, so guarded work is never wrongly
    skipped.

    Args:
        level: loguru level name (default ``"DEBUG"``).
    """
    try:
        # ``_core.min_level`` is the lowest level across active sinks (loguru
        # internal; not in the public type stub).
        core = _loguru_logger._core  # type: ignore[attr-defined]
        return bool(core.min_level <= _loguru_logger.level(level).no)
    except (AttributeError, ValueError):
        return True


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
