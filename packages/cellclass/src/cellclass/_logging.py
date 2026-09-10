"""loguru logging setup for the cellclass command-line tools.

The cellclass ``bin/`` scripts log via the standard library's module-level
``logging.info``/``logging.error`` functions. To render those through loguru
(consistent with the rest of the omero-screen monorepo) without rewriting every
call site, this module configures a loguru console sink and installs an
``InterceptHandler`` on the stdlib root so standard-library records are
forwarded into loguru. Lazy ``%``-style stdlib calls keep working because the
handler resolves them via ``record.getMessage()`` before forwarding.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from loguru import logger

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | <level>{message}</level>"
)


class _InterceptHandler(logging.Handler):
    """Forward standard-library log records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        # basicConfig() below installs this on the *root* logger at level 0, so
        # every third-party library's records pass through here. A record whose
        # msg and args do not %-format against each other then raises inside
        # emit(), and because logging does not wrap emit(), that exception
        # propagates into whatever unrelated code triggered the log call.
        # fontTools' subsetting timer does exactly this, which is enough to
        # abort a matplotlib PDF save. Fail like a stdlib handler instead:
        # report through handleError() and let the caller carry on.
        try:
            message = record.getMessage()
        except Exception:
            self.handleError(record)
            return
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame: Any = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        try:
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, message
            )
        except Exception:
            self.handleError(record)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure loguru for a cellclass CLI script.

    Args:
        level: Minimum level to emit (stdlib numeric level or loguru name).
            Defaults to ``logging.INFO``.

    """
    logger.remove()
    logger.add(sys.stderr, level=level, format=_CONSOLE_FORMAT)
    # Route stdlib logging (the scripts' module-level logging.* calls) into
    # loguru. Set the root level to the requested level, not 0: level=0 turns
    # on DEBUG for *every* library in the process, so a cellclass CLI would
    # emit third-party debug chatter that no one asked for, and records that
    # would normally be filtered out at the logger reach the handlers, where a
    # malformed one (fontTools' subsetting timer) can break unrelated callers.
    logging.basicConfig(
        handlers=[_InterceptHandler()], level=level, force=True
    )
