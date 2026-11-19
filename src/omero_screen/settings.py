"""Layered user configuration: site profile, user config and the password store.

Everything in omero-screen reads its settings from environment variables
(``HOST``, ``USERNAME``, ``OMERO_SCREEN_*``, ...). This module fills those in
from two TOML files, so a new user needs neither a ``.env`` file nor a clone
of the repository:

``site profile``
    What a lab or facility shares: the OMERO server, microscope stitch
    calibrations, segmentation defaults. Shipped in ``omero_screen/sites``
    (``sussex.toml``) or any TOML file named by path.
``user config``
    ``~/.config/omero-screen/config.toml`` (or ``$OMERO_SCREEN_CONFIG_DIR``):
    the site, the user's OMERO name and group, paths, overrides. Written by
    ``omero-screen setup``.

Values already in the environment are never replaced, so precedence is, from
lowest to highest: built-in defaults < site profile < user config < ``.env``
file < environment variables < command-line flags.

The password is never written to either TOML file. ``setup`` keeps it in a
password file readable only by its owner (``[omero] password_file``). It is
looked up when a connection is made, in this order: the ``PASSWORD``
environment variable, the password file, then (only for setups made before
the password file, until the user runs ``setup`` again) the system keychain.
"""

from __future__ import annotations

import glob
import os
import shutil
import stat
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

#: Keychain service name under which older setups stored the password.
KEYRING_SERVICE = "omero-screen"

#: :func:`password_source` for a password still in the keychain.
LEGACY_KEYCHAIN = "system keychain (from an older setup)"

#: TOML key -> environment variable it sets.
ENV_KEYS: dict[tuple[str, str], str] = {
    ("omero", "host"): "HOST",
    ("omero", "port"): "OMERO_PORT",
    ("omero", "username"): "USERNAME",
    ("omero", "group"): "OMERO_GROUP",
    ("paths", "cache"): "OMERO_SCREEN_CACHE_PATH",
    ("paths", "cellview_db"): "DATABASE_PATH",
    ("paths", "training_db"): "OMERO_SCREEN_TRAINING_DB",
    ("paths", "cellpose_models"): "CELLPOSE_LOCAL_MODELS_PATH",
    ("paths", "log_file"): "OMERO_SCREEN_LOG_FILE",
}

#: Keys whose values are paths (``~`` is expanded).
PATH_KEYS = {key for key in ENV_KEYS if key[0] == "paths"}


class ConfigError(RuntimeError):
    """The configuration is missing or invalid."""


def config_dir() -> Path:
    """Directory of the user config (``$OMERO_SCREEN_CONFIG_DIR``)."""
    return Path(
        os.environ.get("OMERO_SCREEN_CONFIG_DIR", "~/.config/omero-screen")
    ).expanduser()


def user_config_path() -> Path:
    """Path of the user config file."""
    return config_dir() / "config.toml"


def available_sites() -> list[str]:
    """Names of the site profiles shipped with omero-screen."""
    folder = resources.files("omero_screen").joinpath("sites")
    return sorted(
        p.name.removesuffix(".toml")
        for p in folder.iterdir()
        if p.name.endswith(".toml")
    )


def load_site(site: str) -> dict[str, Any]:
    """A site profile, by shipped name (``"sussex"``) or by file path.

    Raises:
        ConfigError: If the profile does not exist.
    """
    path = Path(site).expanduser()
    if path.suffix == ".toml" and path.exists():
        return tomllib.loads(path.read_text())
    shipped = resources.files("omero_screen").joinpath("sites", f"{site}.toml")
    if shipped.is_file():
        return tomllib.loads(shipped.read_text())
    raise ConfigError(
        f"Unknown site profile {site!r}. Shipped profiles: "
        f"{', '.join(available_sites())}; or give the path to a .toml file."
    )


def _merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """Deep merge: tables merge, other values from ``top`` win."""
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Settings:
    """The merged site profile and user config.

    Attributes:
        values: Merged tables (``values["omero"]["host"]``, ...).
        sources: For each ``(table, key)``: ``"site:<name>"`` or ``"user"``.
        site: Name or path of the site profile, if any.
        user_file: The user config file, if it exists.
    """

    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[tuple[str, str], str] = field(default_factory=dict)
    site: str | None = None
    user_file: Path | None = None

    def get(self, table: str, key: str, default: Any = None) -> Any:
        """A value from the merged config."""
        return self.values.get(table, {}).get(key, default)

    def section(self, table: str) -> dict[str, Any]:
        """A whole table (``{}`` if absent)."""
        section = self.values.get(table, {})
        return dict(section) if isinstance(section, dict) else {}


def _record(
    sources: dict[tuple[str, str], str], data: dict[str, Any], label: str
) -> None:
    for table, content in data.items():
        if isinstance(content, dict):
            for key in content:
                sources[(table, key)] = label


def load(path: Path | None = None) -> Settings:
    """Read the user config and the site profile it names.

    Args:
        path: User config file; default :func:`user_config_path`.

    Returns:
        The merged settings; empty if there is no user config.
    """
    path = path or user_config_path()
    if not path.exists():
        return Settings()
    user = tomllib.loads(path.read_text())
    site_name = user.get("site")
    site = load_site(site_name) if site_name else {}
    sources: dict[tuple[str, str], str] = {}
    _record(sources, site, f"site:{site_name}")
    _record(sources, user, "user")
    return Settings(
        values=_merge(site, {k: v for k, v in user.items() if k != "site"}),
        sources=sources,
        site=site_name,
        user_file=path,
    )


def _env_value(key: tuple[str, str], value: Any) -> str:
    if key in PATH_KEYS:
        return str(Path(str(value)).expanduser())
    return str(value)


def apply(
    settings: Settings | None = None, environ: Any = None
) -> dict[str, str]:
    """Fill environment variables from the settings, never replacing one.

    Sets the variables in :data:`ENV_KEYS` and every entry of the ``[env]``
    table (an escape hatch for any ``OMERO_SCREEN_*`` variable).

    Returns:
        The variables that were set, by name.
    """
    settings = settings if settings is not None else load()
    environ = os.environ if environ is None else environ
    wanted: dict[str, str] = {}
    for key, var in ENV_KEYS.items():
        value = settings.get(*key)
        if value not in (None, ""):
            wanted[var] = _env_value(key, value)
    for var, value in settings.section("env").items():
        wanted[str(var)] = str(value)
    applied = {}
    for var, value in wanted.items():
        if var not in environ:
            environ[var] = value
            applied[var] = value
    return applied


# --- password ---------------------------------------------------------------


def _keyring_account(username: str, host: str) -> str:
    return f"{username}@{host}"


def _password_file(settings: Settings | None = None) -> Path | None:
    """The configured password file, if it exists."""
    settings = settings if settings is not None else load()
    if pw_file := settings.get("omero", "password_file"):
        path = Path(str(pw_file)).expanduser()
        if path.exists():
            return path
    return None


def get_password(
    username: str, host: str, settings: Settings | None = None
) -> str | None:
    """The OMERO password: environment, password file, then keychain.

    The keychain is only read for setups made before the password file;
    reading it can prompt (macOS asks whether Python may read the entry).
    """
    if password := os.environ.get("PASSWORD"):
        return password
    if path := _password_file(settings):
        return read_password_file(path)
    try:
        import keyring

        stored = keyring.get_password(
            KEYRING_SERVICE, _keyring_account(username, host)
        )
        if stored:
            return str(stored)
    except Exception:  # noqa: BLE001 - no usable keychain backend
        pass
    return None


def read_password_file(path: Path) -> str:
    """The password in ``path``, which must be readable only by its owner.

    Raises:
        ConfigError: If others can read the file.
    """
    path = Path(path).expanduser()
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ConfigError(
            f"{path} is readable by others; run: chmod 600 {path}"
        )
    return path.read_text().strip()


def password_source(username: str | None, host: str | None) -> str:
    """Where the password would come from, without revealing it."""
    if os.environ.get("PASSWORD"):
        return "PASSWORD environment variable"
    if not (username and host):
        return "not looked up (no user name or host)"
    if path := _password_file():
        return f"password file {path}"
    try:
        import keyring

        if keyring.get_password(
            KEYRING_SERVICE, _keyring_account(username, host)
        ):
            return LEGACY_KEYCHAIN
    except Exception:  # noqa: BLE001
        pass
    if pw_file := load().get("omero", "password_file"):
        return f"not found: password file {pw_file} is missing"
    return "not found: run `omero-screen setup`"


def describe(
    settings: Settings | None = None, environ: Any = None
) -> list[tuple[str, str, str]]:
    """``(variable, value, source)`` for every configured variable in effect.

    The source is ``"user config"``, ``"site:<name>"``, or
    ``"environment / .env"`` when the value did not come from the TOML files.
    """
    settings = settings if settings is not None else load()
    environ = os.environ if environ is None else environ
    rows = []
    for key, var in ENV_KEYS.items():
        if var not in environ:
            continue
        value = environ[var]
        configured = settings.get(*key)
        from_config = (
            configured is not None and _env_value(key, configured) == value
        )
        source = settings.sources.get(key, "") if from_config else ""
        rows.append((var, value, _label(source)))
    for var, value in settings.section("env").items():
        if environ.get(str(var)) == str(value):
            rows.append(
                (
                    str(var),
                    str(value),
                    _label(settings.sources.get(("env", str(var)), "")),
                )
            )
    return rows


def _label(source: str) -> str:
    return {"user": "user config", "": "environment / .env"}.get(
        source, source
    )


def default_password_file() -> Path:
    """Where ``setup`` puts the password when it uses a file."""
    return config_dir() / "omero-password"


def write_password_file(path: Path, password: str) -> Path:
    """Write the password to ``path``, readable only by its owner.

    The folder is created owner-only too; the file is never readable by
    others, not even briefly.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, password.encode())
    finally:
        os.close(fd)
    return path


def on_cluster() -> bool:
    """Whether this machine is part of a SLURM cluster."""
    return shutil.which("sbatch") is not None or any(
        key.startswith("SLURM_") for key in os.environ
    )


def suggest_work_dir(profile: dict[str, Any], user: str) -> Path | None:
    """The user's work directory for large data, from the site profile.

    ``[cluster] work_dirs`` lists glob patterns with ``{user}`` (the login
    name); the first existing directory wins.
    """
    patterns = profile.get("cluster", {}).get("work_dirs", [])
    for pattern in patterns:
        for match in sorted(glob.glob(str(pattern).format(user=user))):
            if Path(match).is_dir():
                return Path(match)
    return None


def login() -> tuple[str, str, int | None, str | None, str]:
    """``(host, username, port, group, password)`` for an OMERO connection.

    Raises:
        ConfigError: With what is missing and how to fix it.
    """
    host = os.environ.get("HOST")
    username = os.environ.get("USERNAME")
    port = os.environ.get("OMERO_PORT")
    group = os.environ.get("OMERO_GROUP") or None
    password = get_password(username, host) if host and username else None
    missing = [
        name
        for name, value in (
            ("server (HOST)", host),
            ("user name (USERNAME)", username),
            ("password", password),
        )
        if not value
    ]
    if missing:
        raise ConfigError(
            f"OMERO login incomplete: no {', '.join(missing)}. "
            "Run `omero-screen setup` (or `omero-screen doctor` to see what is "
            "configured)."
        )
    assert host and username and password
    return host, username, int(port) if port else None, group, password


# --- writing the user config -------------------------------------------------


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def read_user_config(path: Path | None = None) -> dict[str, Any]:
    """The user config as written, without the site profile (``{}`` if none)."""
    path = path or user_config_path()
    return tomllib.loads(path.read_text()) if path.exists() else {}


def _table_lines(name: str, table: dict[str, Any]) -> list[str]:
    """``[name]`` with its plain keys, then its sub-tables as ``[name.sub]``."""
    plain = {k: v for k, v in table.items() if not isinstance(v, dict)}
    lines: list[str] = []
    if plain:
        lines += ["", f"[{name}]"]
        lines += [
            f"{_toml_key(k)} = {_toml_value(v)}" for k, v in plain.items()
        ]
    for key, value in table.items():
        if isinstance(value, dict):
            lines += _table_lines(f"{name}.{_toml_key(key)}", value)
    return lines


def _toml_key(key: str) -> str:
    """A bare TOML key, quoted if it has characters a bare key can't."""
    if key and all(c.isalnum() or c in "-_" for c in key):
        return key
    return _toml_value(key)


def write_user_config(data: dict[str, Any], path: Path | None = None) -> Path:
    """Write the user config: top-level keys, then tables (nested allowed).

    Args:
        data: e.g. ``{"site": "sussex", "omero": {"username": "ab123"}}``.
        path: Target file; default :func:`user_config_path`.
    """
    path = path or user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# omero-screen user config (written by `omero-screen setup`).",
        "",
    ]
    for key, value in data.items():
        if not isinstance(value, dict):
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    for table, content in data.items():
        if isinstance(content, dict) and content:
            lines += _table_lines(_toml_key(table), content)
    path.write_text("\n".join(lines) + "\n")
    return path
