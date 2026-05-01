# ------------------------------------------------------------------------------
# Tests for cellclass.bin_utils path-validation helpers.
# ------------------------------------------------------------------------------

"""Tests for bin_utils.file_path and bin_utils.dir_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from cellclass.bin_utils import dir_path, file_path


class TestFilePath:
    """Tests for file_path()."""

    def test_accepts_existing_file(self, tmp_path: Path) -> None:
        """file_path must return the path string when the file exists."""
        f = tmp_path / "data.txt"
        f.write_text("hello")
        result = file_path(str(f))
        assert result == str(f)

    def test_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        """file_path must raise FileNotFoundError for a missing path."""
        missing = str(tmp_path / "does_not_exist.txt")
        with pytest.raises(FileNotFoundError):
            file_path(missing)

    def test_raises_for_directory_path(self, tmp_path: Path) -> None:
        """file_path must raise FileNotFoundError when given a directory."""
        with pytest.raises(FileNotFoundError):
            file_path(str(tmp_path))


class TestDirPath:
    """Tests for dir_path()."""

    def test_accepts_existing_directory(self, tmp_path: Path) -> None:
        """dir_path must return the path string when the directory exists."""
        result = dir_path(str(tmp_path))
        assert result == str(tmp_path)

    def test_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        """dir_path must raise NotADirectoryError for a missing path."""
        missing = str(tmp_path / "ghost_dir")
        with pytest.raises(NotADirectoryError):
            dir_path(missing)

    def test_raises_when_path_is_a_file(self, tmp_path: Path) -> None:
        """dir_path must raise NotADirectoryError when the path points to a file."""
        f = tmp_path / "file.txt"
        f.write_text("data")
        with pytest.raises(NotADirectoryError):
            dir_path(str(f))
