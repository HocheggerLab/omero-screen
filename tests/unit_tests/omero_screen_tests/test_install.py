"""Finding an install made by install.sh, for ``omero-screen self``."""

from pathlib import Path

import pytest

from omero_screen import install


def _install(tmp_path: Path, manifest: str | None) -> Path:
    home = tmp_path / "omero-screen"
    venv = home / "omero-screen-v0.8.28" / ".venv"
    venv.mkdir(parents=True)
    (home / "current").symlink_to(venv.parent)
    if manifest is not None:
        (home / "install.env").write_text(manifest)
    return venv


def test_finds_the_install_and_its_bin_folder(tmp_path: Path) -> None:
    venv = _install(
        tmp_path, "# comment\nINSTALLED_BIN_DIR='/opt/bin'\nOTHER='x'\n"
    )
    found = install.find(venv)
    assert found is not None
    assert found.home == (tmp_path / "omero-screen").resolve()
    assert found.bin_dir == Path("/opt/bin")
    assert found.installer.name == "install.sh"


def test_an_install_without_a_manifest_uses_the_default_bin(
    tmp_path: Path,
) -> None:
    found = install.find(_install(tmp_path, None))
    assert found is not None and found.bin_dir == install.DEFAULT_BIN


def test_a_source_checkout_is_not_an_install(tmp_path: Path) -> None:
    venv = tmp_path / "omero-screen" / ".venv"
    venv.mkdir(parents=True)
    assert install.find(venv) is None


def test_versions_sort_numerically() -> None:
    tags = ["omero-screen-v0.8.9", "omero-screen-v0.8.28", "omero-screen-v0.10.0"]
    assert max(tags, key=install.version_key) == "omero-screen-v0.10.0"
    assert install.version_key("0.8.28") == install.version_key(
        "omero-screen-v0.8.28"
    )


@pytest.mark.parametrize("version", ["0.8.27", "omero-screen-v0.8.27"])
def test_installer_env_targets_this_install(tmp_path: Path, version: str) -> None:
    found = install.Install(home=tmp_path, bin_dir=Path("/opt/bin"))
    env = install.installer_env(found, version)
    assert env == {
        "OMERO_SCREEN_HOME": str(tmp_path),
        "OMERO_SCREEN_BIN": "/opt/bin",
        "OMERO_SCREEN_VERSION": "omero-screen-v0.8.27",
    }
    default = install.Install(home=tmp_path, bin_dir=install.DEFAULT_BIN)
    assert "OMERO_SCREEN_BIN" not in install.installer_env(default, None)
