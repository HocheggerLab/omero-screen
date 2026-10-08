"""``omero-screen setup | doctor | config show``: configure and check an install.

``setup`` writes the user config (site profile, OMERO user, group) and stores
the password in the system keychain. ``doctor`` checks every piece a user
needs and says how to fix what is missing. ``config show`` prints the
settings in effect and where each came from, with the password masked.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from omero_screen import settings

#: Exit status of ``doctor`` when a required check fails.
DOCTOR_FAILED = 1


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Configure omero-screen and check the installation."""


# --- setup -------------------------------------------------------------------


def _try_login(
    host: str, port: int | None, username: str, password: str
) -> str:
    """Connect once; return the user's default group name."""
    from omero.gateway import BlitzGateway

    conn = BlitzGateway(username, password, host=host, port=port)
    try:
        if not conn.connect():
            raise click.ClickException(
                f"Login to {host} as {username} failed: wrong user name or "
                "password, or the server is unreachable (VPN?)."
            )
        return str(conn.getGroupFromContext().getName())
    finally:
        conn.close()


@cli.command()
@click.option(
    "--site",
    help="Site profile: a shipped name (e.g. sussex) or a path to a .toml.",
)
@click.option("--host", help="OMERO server, if the site profile has none.")
@click.option("--username", help="Your OMERO user name.")
@click.option(
    "--group",
    default=None,
    help="OMERO group to work in (default: your default group).",
)
@click.option(
    "--password-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Headless machines: read the password from this chmod-600 file "
    "instead of the keychain.",
)
@click.option(
    "--no-verify",
    is_flag=True,
    help="Do not test the login before saving.",
)
def setup(
    site: str | None,
    host: str | None,
    username: str | None,
    group: str | None,
    password_file: Path | None,
    no_verify: bool,
) -> None:
    """Write the user config and store the OMERO password."""
    sites = settings.available_sites()
    if site is None:
        site = click.prompt(
            "Site profile",
            default=sites[0] if sites else "",
            show_choices=True,
            type=click.Choice([*sites, "none"]) if sites else str,
        )
    site = None if site in ("none", "") else site
    profile = settings.load_site(site) if site else {}
    site_host = profile.get("omero", {}).get("host")
    port = profile.get("omero", {}).get("port")
    host = host or site_host or click.prompt("OMERO server (host name)")
    username = username or click.prompt("OMERO user name")
    assert host and username

    if password_file is not None:
        password = settings.read_password_file(password_file)
    else:
        password = click.prompt("OMERO password", hide_input=True)

    if not no_verify:
        click.echo(f"Checking the login to {host} ...")
        default_group = _try_login(host, port, username, password)
        click.echo(f"OK: logged in; your default group is {default_group!r}.")

    omero: dict[str, Any] = {"username": username}
    if not site_host or host != site_host:
        omero["host"] = host
    if group:
        omero["group"] = group
    if password_file is not None:
        omero["password_file"] = str(password_file)
    else:
        settings.store_password(username, host, password)
        click.echo("Password stored in the system keychain.")

    data: dict[str, Any] = {"site": site} if site else {}
    data["omero"] = omero
    path = settings.write_user_config(data)
    click.echo(f"Wrote {path}. Next: omero-screen doctor")


# --- doctor ------------------------------------------------------------------


@dataclass
class Check:
    """One doctor check: a name, whether it passed, details and a fix."""

    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    required: bool = True


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_python() -> Check:
    v = sys.version_info
    ok = (3, 12) <= (v.major, v.minor) < (3, 15)
    return Check(
        "Python", ok, f"{v.major}.{v.minor}.{v.micro}", "Use Python 3.12-3.14."
    )


def _check_ice() -> Check:
    ok = _module("Ice")
    return Check(
        "OMERO bindings (zeroc-ice)",
        ok,
        "importable" if ok else "missing",
        "Reinstall with the ice index: see the install guide.",
    )


def _check_config() -> Check:
    path = settings.user_config_path()
    cfg = settings.load()
    if path.exists():
        return Check(
            "User config", True, f"{path} (site: {cfg.site or 'none'})"
        )
    dev = os.environ.get("HOST") and os.environ.get("USERNAME")
    return Check(
        "User config",
        bool(dev),
        "none; using a .env file / environment" if dev else f"no {path}",
        "Run: omero-screen setup",
    )


def _check_login(connect: bool) -> list[Check]:
    try:
        host, username, port, group, password = settings.login()
    except settings.ConfigError as err:
        return [
            Check("OMERO login", False, str(err), "Run: omero-screen setup")
        ]
    checks = [
        Check(
            "OMERO login",
            True,
            f"{username}@{host}" + (f":{port}" if port else ""),
        )
    ]
    if not connect:
        return checks
    from omero.gateway import BlitzGateway

    conn = BlitzGateway(username, password, host=host, port=port, group=group)
    try:
        ok = bool(conn.connect())
        detail = (
            f"connected; group {conn.getGroupFromContext().getName()!r}"
            if ok
            else "login refused or server unreachable"
        )
    except Exception as err:  # noqa: BLE001
        ok, detail = False, f"{type(err).__name__}: {err}"
    finally:
        conn.close()
    checks.append(
        Check(
            "OMERO connection",
            ok,
            detail,
            "Check the VPN, then the password (omero-screen setup).",
        )
    )
    return checks


def _check_cellview() -> Check:
    path = Path(
        os.environ.get("DATABASE_PATH", "~/.cellview/cellview.duckdb")
    ).expanduser()
    if path.exists():
        return Check("CellView database", True, str(path))
    writable = os.access(
        path.parent if path.parent.exists() else Path.home(), os.W_OK
    )
    return Check(
        "CellView database",
        writable,
        f"{path} (created on first import)",
        f"Make {path.parent} writable.",
        required=False,
    )


def _check_napari() -> Check:
    ok = _module("napari") and _module("qtpy")
    return Check(
        "napari and Qt",
        ok,
        "importable" if ok else "missing",
        "Install the GUI group (included by the install script).",
    )


def _check_device() -> Check:
    try:
        import torch

        if torch.cuda.is_available():
            detail = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            detail = "Apple GPU (MPS)"
        else:
            detail = "CPU only: segmentation will be slow"
        return Check("Compute device", True, detail, required=False)
    except Exception as err:  # noqa: BLE001
        return Check("Compute device", False, str(err), "Reinstall torch.")


def _check_cache() -> Check:
    cache = Path(
        os.environ.get("OMERO_SCREEN_CACHE_PATH", "~/omero-cache")
    ).expanduser()
    probe = cache if cache.exists() else Path.home()
    free_gb = shutil.disk_usage(probe).free / 1e9
    return Check(
        "Image cache",
        free_gb > 20,
        f"{cache}, {free_gb:.0f} GB free",
        "Free disk space or set [paths] cache in the user config.",
        required=False,
    )


@cli.command()
@click.option(
    "--offline", is_flag=True, help="Skip the test connection to OMERO."
)
def doctor(offline: bool) -> None:
    """Check the installation and configuration; explain any fix."""
    probes: list[Callable[[], Check | list[Check]]] = [
        _check_python,
        _check_ice,
        _check_config,
        lambda: _check_login(connect=not offline),
        _check_cellview,
        _check_napari,
        _check_device,
        _check_cache,
    ]
    checks: list[Check] = []
    for probe in probes:
        result = probe()
        checks.extend(result if isinstance(result, list) else [result])
    failed = False
    for c in checks:
        mark = "OK  " if c.ok else ("FAIL" if c.required else "WARN")
        click.echo(f"[{mark}] {c.name}: {c.detail}")
        if not c.ok:
            click.echo(f"       fix: {c.fix}")
            failed |= c.required
    click.echo("")
    click.echo(
        "Some checks failed." if failed else "All required checks passed."
    )
    if failed:
        sys.exit(DOCTOR_FAILED)


# --- config show -------------------------------------------------------------


@cli.group()
def config() -> None:
    """Inspect the configuration."""


@config.command("show")
def config_show() -> None:
    """Print the settings in effect and where each came from."""
    cfg = settings.load()
    click.echo(f"user config:  {cfg.user_file or 'none'}")
    click.echo(f"site profile: {cfg.site or 'none'}")
    click.echo("")
    rows = settings.describe(cfg)
    for var, value, source in rows:
        click.echo(f"{var:28} = {value}   [{source}]")
    listed = {var for var, _, _ in rows}
    for var, value in sorted(os.environ.items()):
        if var.startswith("OMERO_SCREEN_") and var not in listed:
            click.echo(f"{var:28} = {value}   [environment / .env]")
    click.echo("")
    host, user = os.environ.get("HOST"), os.environ.get("USERNAME")
    click.echo(f"password: {settings.password_source(user, host)}")
    for table in ("segmentation", "features", "stitching"):
        if cfg.section(table):
            click.echo(f"[{table}] configured (site profile / user config)")


def run(argv: list[str]) -> None:
    """Run a subcommand (used by the ``omero-screen`` entry point)."""
    cli.main(args=argv, prog_name="omero-screen")
