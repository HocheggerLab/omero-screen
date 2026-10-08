"""``omero-screen setup | doctor | config show``."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from omero_screen import settings, setup_cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_setup_stores_the_password_in_the_keychain_only(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_cli, "_try_login", lambda *a: "lab")
    monkeypatch.delenv("PASSWORD", raising=False)  # set in CI; would win
    result = runner.invoke(
        setup_cli.cli,
        ["setup", "--site", "sussex", "--username", "ab123"],
        input="s3cret\n",
    )
    assert result.exit_code == 0, result.output
    text = settings.user_config_path().read_text()
    assert "s3cret" not in text and 'site = "sussex"' in text
    assert "host" not in text  # the site profile provides it
    assert settings.get_password("ab123", "ome2.hpc.sussex.ac.uk") == "s3cret"


def test_setup_with_a_password_file(
    runner: CliRunner, tmp_path: Path
) -> None:
    pw = tmp_path / "pw"
    pw.write_text("s3cret")
    pw.chmod(0o600)
    result = runner.invoke(
        setup_cli.cli,
        [
            "setup", "--site", "none", "--host", "omero.lab.org",
            "--username", "ab123", "--password-file", str(pw), "--no-verify",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = settings.load()
    assert cfg.get("omero", "host") == "omero.lab.org"
    assert cfg.get("omero", "password_file") == str(pw)


def test_doctor_fails_without_a_login(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("HOST", "USERNAME", "PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    result = runner.invoke(setup_cli.cli, ["doctor", "--offline"])
    assert result.exit_code == setup_cli.DOCTOR_FAILED
    assert "omero-screen setup" in result.output


def test_config_show_masks_the_password(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PASSWORD", "s3cret")
    result = runner.invoke(setup_cli.cli, ["config", "show"])
    assert result.exit_code == 0
    assert "s3cret" not in result.output
    assert "PASSWORD environment variable" in result.output
