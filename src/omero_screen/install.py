"""Find and manage an install made by ``install.sh``.

``install.sh`` puts each release in ``<home>/<tag>/`` with its own ``.venv``,
points ``<home>/current`` at the active one and writes ``<home>/install.env``
(the bin folder and the linked commands). ``omero-screen self update`` and
``self uninstall`` run that script; this module finds the install, reads the
manifest and looks up the latest release.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = "HocheggerLab/omero-screen"
#: The installer from main, so an update also gets the newest install logic.
INSTALLER_URL = f"https://raw.githubusercontent.com/{REPO}/main/install.sh"
TAGS_URL = f"https://api.github.com/repos/{REPO}/tags?per_page=100"
TAG_PREFIX = "omero-screen-v"
MANIFEST = "install.env"
DEFAULT_BIN = Path("~/.local/bin").expanduser()


class InstallError(RuntimeError):
    """The install cannot be found or updated."""


@dataclass(frozen=True)
class Install:
    """An install made by ``install.sh``."""

    home: Path
    bin_dir: Path

    @property
    def installer(self) -> Path:
        """The ``install.sh`` shipped with the active release."""
        return self.home / "current" / "install.sh"


def find(prefix: str | Path | None = None) -> Install | None:
    """The install that the running interpreter belongs to, if any.

    ``prefix`` is the venv (default ``sys.prefix``): ``<home>/<tag>/.venv``.
    A source checkout, or a venv made some other way, returns None.
    """
    venv = Path(prefix if prefix is not None else sys.prefix).resolve()
    home = venv.parent.parent
    if venv.name != ".venv" or not (
        (home / MANIFEST).is_file() or (home / "current").is_symlink()
    ):
        return None
    manifest = read_manifest(home / MANIFEST)
    bin_dir = Path(manifest.get("INSTALLED_BIN_DIR") or DEFAULT_BIN)
    return Install(home=home, bin_dir=bin_dir)


def read_manifest(path: Path) -> dict[str, str]:
    """``KEY='value'`` lines of ``install.env`` (empty if there is none)."""
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([A-Z_]+)='([^']*)'", line.strip())
        if match:
            values[match[1]] = match[2]
    return values


def version_key(tag: str) -> tuple[int, ...]:
    """Sort key of a release tag (``omero-screen-v0.8.28`` -> (0, 8, 28))."""
    return tuple(
        int(part) for part in re.findall(r"\d+", tag.removeprefix(TAG_PREFIX))
    )


def latest_version(timeout: float = 10) -> str:
    """The highest released version, e.g. ``0.8.28``.

    Raises:
        InstallError: If GitHub cannot be reached.
    """
    try:
        with urllib.request.urlopen(TAGS_URL, timeout=timeout) as response:
            tags = [t["name"] for t in json.load(response)]
    except (OSError, ValueError) as err:
        raise InstallError(f"Could not reach GitHub: {err}") from err
    releases = [
        t for t in tags if re.fullmatch(rf"{TAG_PREFIX}\d+\.\d+\.\d+", t)
    ]
    if not releases:
        raise InstallError("No omero-screen release found on GitHub.")
    return max(releases, key=version_key).removeprefix(TAG_PREFIX)


def download_installer(dest: Path, timeout: float = 30) -> Path:
    """Fetch the current ``install.sh`` from main.

    Raises:
        InstallError: If it cannot be downloaded.
    """
    try:
        with urllib.request.urlopen(
            INSTALLER_URL, timeout=timeout
        ) as response:
            dest.write_bytes(response.read())
    except OSError as err:
        raise InstallError(f"Could not download the installer: {err}") from err
    return dest


def installer_env(install: Install, version: str | None) -> dict[str, str]:
    """Environment for ``install.sh`` to update this install in place."""
    env = {"OMERO_SCREEN_HOME": str(install.home)}
    if install.bin_dir != DEFAULT_BIN:
        env["OMERO_SCREEN_BIN"] = str(install.bin_dir)
    if version:
        tag = (
            version if version.startswith(TAG_PREFIX) else TAG_PREFIX + version
        )
        env["OMERO_SCREEN_VERSION"] = tag
    return env
