"""Layered user configuration: site profile, user config, password store."""

import os
from pathlib import Path

import keyring
import pytest

from omero_screen import settings


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def user_config() -> Path:
    return settings.user_config_path()


def test_no_user_config_gives_empty_settings() -> None:
    cfg = settings.load()
    assert cfg.values == {} and cfg.site is None


def test_user_config_overrides_the_site_profile(user_config: Path) -> None:
    _write(
        user_config,
        'site = "sussex"\n[omero]\nusername = "ab123"\nport = 4065\n',
    )
    cfg = settings.load()
    assert cfg.get("omero", "host") == "ome2.hpc.sussex.ac.uk"
    assert cfg.get("omero", "port") == 4065
    assert cfg.sources[("omero", "host")] == "site:sussex"
    assert cfg.sources[("omero", "port")] == "user"
    assert "20x" in cfg.section("stitching")["calibrations"]


def test_site_profile_by_path(tmp_path: Path, user_config: Path) -> None:
    site = _write(tmp_path / "lab.toml", '[omero]\nhost = "omero.lab.org"\n')
    _write(user_config, f'site = "{site}"\n')
    assert settings.load().get("omero", "host") == "omero.lab.org"


def test_unknown_site_is_an_error(user_config: Path) -> None:
    _write(user_config, 'site = "atlantis"\n')
    with pytest.raises(settings.ConfigError, match="sussex"):
        settings.load()


def test_apply_never_replaces_a_set_variable(user_config: Path) -> None:
    _write(
        user_config,
        'site = "sussex"\n[omero]\nusername = "ab123"\n'
        '[paths]\ncache = "~/cache"\n[env]\nOMERO_SCREEN_USE_GPU = "0"\n',
    )
    environ = {"HOST": "localhost"}
    applied = settings.apply(environ=environ)
    assert environ["HOST"] == "localhost"
    assert environ["USERNAME"] == "ab123"
    assert environ["OMERO_SCREEN_CACHE_PATH"] == str(Path("~/cache").expanduser())
    assert environ["OMERO_SCREEN_USE_GPU"] == "0"
    assert "HOST" not in applied


def test_describe_names_the_source(user_config: Path) -> None:
    _write(user_config, 'site = "sussex"\n[omero]\nusername = "ab123"\n')
    environ: dict[str, str] = {"USERNAME": "someone-else"}
    settings.apply(environ=environ)
    rows = {var: source for var, _, source in settings.describe(environ=environ)}
    assert rows["HOST"] == "site:sussex"
    assert rows["USERNAME"] == "environment / .env"


def test_password_from_env_then_file_then_keychain(
    tmp_path: Path, user_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PASSWORD", raising=False)
    assert settings.get_password("ab123", "h") is None

    keyring.set_password(settings.KEYRING_SERVICE, "ab123@h", "from-keychain")
    assert settings.get_password("ab123", "h") == "from-keychain"
    assert settings.password_source("ab123", "h") == settings.LEGACY_KEYCHAIN

    # A configured password file wins over a (left-over) keychain entry.
    pw = _write(tmp_path / "pw", "from-file\n")
    pw.chmod(0o600)
    _write(user_config, f'[omero]\npassword_file = "{pw}"\n')
    assert settings.get_password("ab123", "h") == "from-file"
    assert settings.password_source("ab123", "h") == f"password file {pw}"

    monkeypatch.setenv("PASSWORD", "from-env")
    assert settings.get_password("ab123", "h") == "from-env"


def test_a_missing_password_file_falls_back_to_the_keychain(
    tmp_path: Path, user_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PASSWORD", raising=False)
    gone = tmp_path / "gone"
    _write(user_config, f'[omero]\npassword_file = "{gone}"\n')
    assert settings.password_source("ab123", "h") == (
        f"not found: password file {gone} is missing"
    )
    keyring.set_password(settings.KEYRING_SERVICE, "ab123@h", "from-keychain")
    assert settings.get_password("ab123", "h") == "from-keychain"


def test_password_file_readable_by_others_is_refused(tmp_path: Path) -> None:
    pw = _write(tmp_path / "pw", "secret")
    pw.chmod(0o644)
    with pytest.raises(settings.ConfigError, match="chmod 600"):
        settings.read_password_file(pw)


def test_login_reports_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HOST", "USERNAME", "PASSWORD", "OMERO_PORT", "OMERO_GROUP"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOST", "h")
    with pytest.raises(settings.ConfigError, match="user name.*password.*setup"):
        settings.login()
    monkeypatch.setenv("USERNAME", "u")
    monkeypatch.setenv("PASSWORD", "p")
    monkeypatch.setenv("OMERO_PORT", "4065")
    assert settings.login() == ("h", "u", 4065, None, "p")


def test_write_user_config_round_trips(user_config: Path) -> None:
    settings.write_user_config(
        {"site": "sussex", "omero": {"username": 'o"brien', "group": "lab"}}
    )
    cfg = settings.load()
    assert cfg.get("omero", "username") == 'o"brien'
    assert cfg.get("omero", "group") == "lab"
    assert "PASSWORD" not in user_config.read_text().upper().replace(
        "PASSWORD_FILE", ""
    )


def test_config_dir_follows_the_env_variable() -> None:
    assert settings.config_dir() == Path(os.environ["OMERO_SCREEN_CONFIG_DIR"])


def test_password_file_is_owner_only(tmp_path: Path) -> None:
    pw = settings.write_password_file(tmp_path / "cfg" / "pw", "x")
    assert pw.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "cfg").stat().st_mode & 0o777 == 0o700
    assert settings.read_password_file(pw) == "x"


def test_suggest_work_dir_takes_the_first_existing_match(tmp_path: Path) -> None:
    (tmp_path / "b" / "me").mkdir(parents=True)
    profile = {"cluster": {"work_dirs": [
        f"{tmp_path}/none/{{user}}", f"{tmp_path}/*/{{user}}",
    ]}}
    assert settings.suggest_work_dir(profile, "me") == tmp_path / "b" / "me"
    assert settings.suggest_work_dir({}, "me") is None


def test_the_sussex_profile_suggests_lustre() -> None:
    patterns = settings.load_site("sussex")["cluster"]["work_dirs"]
    assert patterns == ["/mnt/lustre/users/*/{user}"]
