"""Tests for the stdlib-to-loguru bridge in ``cellclass._logging``.

``configure_logging`` installs ``_InterceptHandler`` on the *root* logger at
level 0, so records from every third-party library flow through it. A handler
that raises therefore breaks unrelated callers rather than just losing a log
line, so the contract under test is that a malformed record is swallowed via
``handleError`` and never escapes ``emit``.
"""

from __future__ import annotations

import logging

import pytest

from cellclass._logging import _InterceptHandler, configure_logging


def _record(msg: str, args: tuple[object, ...]) -> logging.LogRecord:
    """Build a record the way ``Logger.log(level, msg, *args)`` would."""
    return logging.LogRecord(
        name="third_party.timer",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_unformattable_record_does_not_escape_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record whose msg and args do not %-format must not raise.

    This is the fontTools subsetting timer's shape: a message that is already
    formatted, plus a leftover args payload. Before the guard this propagated
    out of the handler and aborted matplotlib PDF saves.
    """
    handler = _InterceptHandler()
    handled: list[logging.LogRecord] = []
    monkeypatch.setattr(handler, "handleError", handled.append)

    record = _record("Took 0.000s to load font", ("unexpected", "extra"))

    with pytest.raises(TypeError):
        record.getMessage()  # the record really is malformed

    handler.emit(record)  # must not raise

    assert handled == [record]


def test_wellformed_record_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal lazy %-style path still renders and forwards the message."""
    handler = _InterceptHandler()
    monkeypatch.setattr(
        handler,
        "handleError",
        lambda record: pytest.fail("handleError called"),
    )

    logged: list[str] = []
    monkeypatch.setattr(
        "cellclass._logging.logger.opt",
        lambda **_: type(
            "_Sink", (), {"log": staticmethod(lambda _l, m: logged.append(m))}
        ),
    )

    handler.emit(_record("hello %s", ("world",)))

    assert logged == ["hello world"]


def test_configure_logging_does_not_enable_debug_globally() -> None:
    """The root level follows the requested level, not 0.

    ``level=0`` switched on DEBUG for every library in the process, which both
    spammed the CLI output and pushed third-party records into the handlers
    that would otherwise have been filtered at the logger.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging(logging.INFO)
        assert root.level == logging.INFO
        assert not logging.getLogger("fontTools.subset.timer").isEnabledFor(
            logging.DEBUG
        )

        configure_logging(logging.DEBUG)
        assert root.level == logging.DEBUG
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
