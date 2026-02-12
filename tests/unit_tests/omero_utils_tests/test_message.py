"""Tests for custom exception classes in omero_utils.message."""

import logging

import pytest

from omero_utils.message import (
    OmeroError,
    PlateDownloadError,
    StitchingError,
)


@pytest.fixture
def test_logger() -> logging.Logger:
    """Return a logger that does not propagate (avoids console noise)."""
    log = logging.getLogger("test_message")
    log.propagate = False
    log.addHandler(logging.NullHandler())
    return log


class TestPlateDownloadError:
    """Tests for PlateDownloadError."""

    def test_is_omero_error_subclass(self) -> None:
        """PlateDownloadError inherits from OmeroError."""
        assert issubclass(PlateDownloadError, OmeroError)

    def test_message_preserved(self, test_logger: logging.Logger) -> None:
        """Exception message is accessible via str()."""
        err = PlateDownloadError(
            "Failed to download image 42", test_logger
        )
        assert str(err) == "Failed to download image 42"

    def test_original_error_stored(
        self, test_logger: logging.Logger
    ) -> None:
        """Original error is preserved on the exception."""
        cause = ConnectionError("Ice connection lost")
        err = PlateDownloadError(
            "Download failed", test_logger, original_error=cause
        )
        assert err.original_error is cause

    def test_raises_and_catches(
        self, test_logger: logging.Logger
    ) -> None:
        """Can be raised and caught as OmeroError."""
        with pytest.raises(OmeroError):
            raise PlateDownloadError(
                "Image not found", test_logger
            )

    def test_raises_with_specific_type(
        self, test_logger: logging.Logger
    ) -> None:
        """Can be caught by its specific type."""
        with pytest.raises(PlateDownloadError) as exc_info:
            raise PlateDownloadError(
                "Timeout downloading well A1", test_logger
            )
        assert "well A1" in str(exc_info.value)


class TestStitchingError:
    """Tests for StitchingError."""

    def test_is_omero_error_subclass(self) -> None:
        """StitchingError inherits from OmeroError."""
        assert issubclass(StitchingError, OmeroError)

    def test_message_preserved(self, test_logger: logging.Logger) -> None:
        """Exception message is accessible via str()."""
        err = StitchingError(
            "No stage positions for well B3", test_logger
        )
        assert str(err) == "No stage positions for well B3"

    def test_original_error_stored(
        self, test_logger: logging.Logger
    ) -> None:
        """Original error is preserved on the exception."""
        cause = ValueError("Unsupported tile count: 7")
        err = StitchingError(
            "Grid derivation failed",
            test_logger,
            original_error=cause,
        )
        assert err.original_error is cause

    def test_raises_and_catches(
        self, test_logger: logging.Logger
    ) -> None:
        """Can be raised and caught as OmeroError."""
        with pytest.raises(OmeroError):
            raise StitchingError(
                "Tile grid failure", test_logger
            )

    def test_raises_with_specific_type(
        self, test_logger: logging.Logger
    ) -> None:
        """Can be caught by its specific type."""
        with pytest.raises(StitchingError) as exc_info:
            raise StitchingError(
                "Unsupported number of tiles: 7", test_logger
            )
        assert "7" in str(exc_info.value)
