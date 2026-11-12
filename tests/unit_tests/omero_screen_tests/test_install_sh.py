"""``install.sh`` end to end, offline: ``curl`` and ``uv`` are stubs.

The stub ``curl`` serves a fake release tarball (or fails, like a network
outage); the stub ``uv sync`` makes a ``.venv`` with a fake ``omero-screen``.
"""

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "install.sh"

CURL = """#!/bin/sh
# Stub curl: -o FILE URL downloads, otherwise print to stdout.
out=""; url=""
while [ $# -gt 0 ]; do
    case "$1" in -o) out="$2"; shift ;; -*) ;; *) url="$1" ;; esac
    shift
done
[ -f "$FAKE/offline" ] && exit 6
case "$url" in
    *api.github.com*) cat "$FAKE/tags.json" ;;
    *.tar.gz) cp "$FAKE/$(basename "$url")" "$out" ;;
    *) exit 22 ;;
esac
"""

UV = """#!/bin/sh
# Stub uv: sync makes a venv with a fake omero-screen that logs its arguments.
case "$1" in
    tool) [ "$2" = dir ] && printf '%s\\n' "$FAKE/bin" ;;
    sync)
        mkdir -p .venv/bin
        printf '#!/bin/sh\\necho "$@" >> "%s/omero-screen.log"\\n' "$FAKE" > .venv/bin/omero-screen
        chmod +x .venv/bin/omero-screen ;;
esac
"""

# Stub keychain tools: never touch the real keychain; log the call, then
# report "no more items" so the uninstaller's delete loop ends.
KEYCHAIN = """#!/bin/sh
echo "$0 $*" >> "$FAKE/keychain.log"
exit 44
"""


def _release(fake: Path, tag: str) -> None:
    """A release tarball like GitHub's: one top folder holding the repo."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("uv.lock", "install.sh"):
            data = INSTALL_SH.read_bytes() if name == "install.sh" else b"lock"
            info = tarfile.TarInfo(f"omero-screen-{tag}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (fake / f"{tag}.tar.gz").write_bytes(buf.getvalue())


@pytest.fixture
def fake(tmp_path: Path) -> Path:
    fake = tmp_path / "fake"
    stubs = fake / "stubs"
    stubs.mkdir(parents=True)
    (fake / "bin").mkdir()
    stubbed = (
        ("curl", CURL), ("uv", UV), ("security", KEYCHAIN), ("secret-tool", KEYCHAIN)
    )
    for name, script in stubbed:
        (stubs / name).write_text(script)
        (stubs / name).chmod(0o755)
    tags = [f"omero-screen-v0.8.{n}" for n in (9, 10, 11)]
    (fake / "tags.json").write_text(json.dumps([{"name": t} for t in tags]))
    for tag in tags:
        _release(fake, tag)
    return fake


def _run(fake: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", str(INSTALL_SH), *args],
        env={
            "PATH": f"{fake / 'stubs'}:{fake / 'bin'}:/usr/bin:/bin",
            "HOME": str(fake / "home"),
            "FAKE": str(fake),
            "OMERO_SCREEN_NO_SETUP": "1",
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _home(fake: Path) -> Path:
    return fake / "home" / ".local" / "share" / "omero-screen"


def test_installs_the_latest_release_and_links_the_commands(fake: Path) -> None:
    result = _run(fake)
    assert result.returncode == 0, result.stderr
    home = _home(fake)
    assert os.readlink(home / "current") == str(home / "omero-screen-v0.8.11")
    assert os.readlink(fake / "bin" / "omero-screen").startswith(str(home))
    assert "INSTALLED_BIN_DIR='" + str(fake / "bin") in (
        home / "install.env"
    ).read_text()
    assert "models pull stock" in (fake / "omero-screen.log").read_text()


def test_a_failed_download_stops_cleanly(fake: Path) -> None:
    (fake / "offline").touch()
    result = _run(fake, OMERO_SCREEN_VERSION="0.8.10")
    assert result.returncode != 0
    assert "could not download" in result.stderr
    assert "pyproject" not in result.stderr
    assert not list(_home(fake).glob("omero-screen-v*"))


def test_an_update_keeps_only_the_previous_release(fake: Path) -> None:
    for version in ("0.8.9", "0.8.10", "0.8.11"):
        result = _run(fake, OMERO_SCREEN_VERSION=version)
        assert result.returncode == 0, result.stderr
    releases = sorted(p.name for p in _home(fake).glob("omero-screen-v*"))
    assert releases == ["omero-screen-v0.8.10", "omero-screen-v0.8.11"]


def test_uninstall_removes_only_our_links(fake: Path) -> None:
    assert _run(fake).returncode == 0
    own_napari = fake / "bin" / "napari"
    own_napari.write_text("#!/bin/sh\n")  # the user's own, not a link to us
    config = fake / "home" / ".config" / "omero-screen"
    config.mkdir(parents=True)

    result = _run(fake, "--uninstall")
    assert result.returncode == 0, result.stderr
    assert not _home(fake).exists()
    assert not (fake / "bin" / "omero-screen").exists()
    assert own_napari.exists()
    assert config.exists()  # kept without --purge


def test_uninstall_purge_removes_the_settings(fake: Path) -> None:
    assert _run(fake).returncode == 0
    config = fake / "home" / ".config" / "omero-screen"
    config.mkdir(parents=True)
    result = _run(fake, "--uninstall", "--purge")
    assert result.returncode == 0, result.stderr
    assert not config.exists()
    assert "omero-screen" in (fake / "keychain.log").read_text()


def test_uninstall_refuses_a_folder_that_is_not_an_install(fake: Path) -> None:
    other = fake / "home" / "precious"
    other.mkdir(parents=True)
    result = _run(fake, "--uninstall", OMERO_SCREEN_HOME=str(other))
    assert result.returncode != 0
    assert other.exists()
