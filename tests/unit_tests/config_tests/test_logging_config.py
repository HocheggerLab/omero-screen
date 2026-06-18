"""Tests for the pure-loguru logging configuration in ``omero_screen.config``.

Logging is now configured once at an application entry point via
``configure_logging()`` (no ``get_logger`` seam). These tests assert the sink
architecture (file-only by default, console opt-in), the
argument > env-var > default resolution, the stdlib ``InterceptHandler`` /
plugin-mode behaviour, and third-party suppression.
"""

import logging

import pytest
from loguru import logger as loguru_logger

import omero_screen.config as config
from omero_screen.config import (
    _InterceptHandler,
    _NOISY_LIBRARIES,
    configure_logging,
    configure_logging_once,
    is_level_enabled,
)


@pytest.fixture
def reset_logging():
    """Reset loguru sinks, the configured flag and the stdlib root per test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_flag = config._CONFIGURED
    config._CONFIGURED = False
    loguru_logger.remove()
    yield
    loguru_logger.remove()
    config._CONFIGURED = saved_flag
    root.handlers[:] = saved_handlers


def _n_sinks() -> int:
    return len(loguru_logger._core.handlers)


# --------------------------------------------------------------------------- #
# Sinks                                                                       #
# --------------------------------------------------------------------------- #


def test_default_is_file_only(tmp_path, reset_logging):
    """Production default: exactly one (file) sink, no console."""
    configure_logging(log_file=str(tmp_path / "app.log"))
    assert _n_sinks() == 1


def test_console_adds_second_sink(tmp_path, reset_logging):
    """console=True adds a second sink alongside the file sink."""
    configure_logging(log_file=str(tmp_path / "app.log"), console=True)
    assert _n_sinks() == 2


def test_log_file_none_disables_file_sink(reset_logging):
    """log_file='none' disables the file sink (console only here)."""
    configure_logging(log_file="none", console=True)
    assert _n_sinks() == 1


def test_file_is_created(tmp_path, reset_logging):
    """The file sink opens its target immediately."""
    target = tmp_path / "app.log"
    configure_logging(log_file=str(target))
    assert target.exists()


def test_configure_logging_is_idempotent_in_effect(tmp_path, reset_logging):
    """Re-running configure_logging() re-applies, not accumulates, sinks."""
    configure_logging(log_file=str(tmp_path / "app.log"), console=True)
    configure_logging(log_file=str(tmp_path / "app.log"), console=True)
    assert _n_sinks() == 2


def test_configure_logging_once_is_noop_after_configured(
    tmp_path, reset_logging
):
    """configure_logging_once() does nothing once configured."""
    configure_logging(log_file=str(tmp_path / "app.log"))
    configure_logging_once(console=True)  # would add console if it ran
    assert _n_sinks() == 1


# --------------------------------------------------------------------------- #
# Level resolution: argument > env var > default                             #
# --------------------------------------------------------------------------- #


def test_default_level_is_info(tmp_path, reset_logging):
    configure_logging(log_file=str(tmp_path / "app.log"))
    assert is_level_enabled("INFO") is True
    assert is_level_enabled("DEBUG") is False


def test_level_argument(tmp_path, reset_logging):
    configure_logging(level="WARNING", log_file=str(tmp_path / "app.log"))
    assert is_level_enabled("WARNING") is True
    assert is_level_enabled("INFO") is False


def test_env_var_sets_level(tmp_path, monkeypatch, reset_logging):
    monkeypatch.setenv("OMERO_SCREEN_LOG_LEVEL", "ERROR")
    configure_logging(log_file=str(tmp_path / "app.log"))
    assert is_level_enabled("ERROR") is True
    assert is_level_enabled("WARNING") is False


def test_argument_overrides_env_var(tmp_path, monkeypatch, reset_logging):
    monkeypatch.setenv("OMERO_SCREEN_LOG_LEVEL", "ERROR")
    configure_logging(level="DEBUG", log_file=str(tmp_path / "app.log"))
    assert is_level_enabled("DEBUG") is True


def test_env_var_sets_log_file(tmp_path, monkeypatch, reset_logging):
    target = tmp_path / "from_env.log"
    monkeypatch.setenv("OMERO_SCREEN_LOG_FILE", str(target))
    configure_logging()
    assert target.exists()


# --------------------------------------------------------------------------- #
# stdlib interception / plugin mode / suppression                            #
# --------------------------------------------------------------------------- #


def test_standalone_installs_intercept_handler(tmp_path, reset_logging):
    logging.getLogger().handlers.clear()
    configure_logging(log_file=str(tmp_path / "app.log"))
    assert any(
        isinstance(h, _InterceptHandler)
        for h in logging.getLogger().handlers
    )


def test_plugin_mode_skips_intercept_handler(tmp_path, reset_logging):
    logging.getLogger().handlers.clear()
    configure_logging(log_file=str(tmp_path / "app.log"), plugin=True)
    assert not any(
        isinstance(h, _InterceptHandler)
        for h in logging.getLogger().handlers
    )


def test_noisy_libraries_pinned_to_warning(tmp_path, reset_logging):
    configure_logging(log_file=str(tmp_path / "app.log"))
    for lib in _NOISY_LIBRARIES:
        assert logging.getLogger(lib).level == logging.WARNING


# --------------------------------------------------------------------------- #
# is_level_enabled guard                                                      #
# --------------------------------------------------------------------------- #


def test_is_level_enabled_reflects_debug(tmp_path, reset_logging):
    configure_logging(level="DEBUG", log_file=str(tmp_path / "app.log"))
    assert is_level_enabled("DEBUG") is True
