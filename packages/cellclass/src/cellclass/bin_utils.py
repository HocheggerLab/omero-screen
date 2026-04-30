# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------

"""Utility functions for validating filesystem paths used by CLI scripts."""

import os


def file_or_dir_path(path: str) -> str:
    """Check if the path is a file or directory.

    Args:
        path: Path to check.

    Returns:
        The input path.

    Raises:
        FileNotFoundError: If not a file or directory.

    """
    if os.path.isfile(path) or os.path.isdir(path):
        return path
    raise FileNotFoundError(path)


def file_path(path: str) -> str:
    """Check if the path is a file.

    Args:
        path: Path to check.

    Returns:
        The input path.

    Raises:
        FileNotFoundError: If not a file.

    """
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(path)


def dir_path(path: str) -> str:
    """Check if the path is a directory.

    Args:
        path: Path to check.

    Returns:
        The input path.

    Raises:
        NotADirectoryError: If not a directory.

    """
    if os.path.isdir(path):
        return path
    raise NotADirectoryError(path)
