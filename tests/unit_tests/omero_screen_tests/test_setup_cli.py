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


def _site_with_models(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """A site profile whose model set downloads from a local folder."""
    import hashlib

    store = tmp_path / "store"
    store.mkdir()
    for name, data in files.items():
        (store / name).write_bytes(data)
    sums = "\n".join(
        f'{name} = "{hashlib.sha256(data).hexdigest()}"' for name, data in files.items()
    )
    site = tmp_path / "site.toml"
    site.write_text(
        "[segmentation.model_sets.lab]\n"
        f'url = "file://{store}/{{name}}"\n'
        "[segmentation.model_sets.lab.models]\n"
        'nuclei = "Nuc"\n'
        "[segmentation.model_sets.lab.sha256]\n" + sums + "\n"
    )
    settings.write_user_config({"site": str(site)})
    return store


def test_models_pull_downloads_and_verifies(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _site_with_models(tmp_path, {"Nuc": b"weights"})
    target = tmp_path / "models"
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(target))

    result = runner.invoke(setup_cli.cli, ["models", "pull", "lab"])
    assert result.exit_code == 0, result.output
    assert (target / "Nuc").read_bytes() == b"weights"

    again = runner.invoke(setup_cli.cli, ["models", "pull", "lab"])
    assert "ok      Nuc" in again.output


def test_models_pull_rejects_a_bad_checksum(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _site_with_models(tmp_path, {"Nuc": b"weights"})
    (store / "Nuc").write_bytes(b"tampered")
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path / "models"))
    result = runner.invoke(setup_cli.cli, ["models", "pull", "lab"])
    assert result.exit_code != 0
    assert "Checksum mismatch" in result.output
    assert not (tmp_path / "models" / "Nuc").exists()


def test_model_set_selects_the_site_models(tmp_path: Path) -> None:
    """[segmentation] model_set fills MODEL_DICT from the site profile."""
    from omero_screen import _overrides_from_settings

    settings.write_user_config(
        {"site": "sussex", "segmentation": {"model_set": "hocheggerlab"}}
    )
    models = _overrides_from_settings()["MODEL_DICT"]
    assert models["nuclei"] == "Nuclei_Hoechst"


def test_unknown_model_set_is_an_error() -> None:
    from omero_screen import _overrides_from_settings

    settings.write_user_config(
        {"site": "sussex", "segmentation": {"model_set": "nope"}}
    )
    with pytest.raises(ValueError, match="hocheggerlab"):
        _overrides_from_settings()


def test_publish_needs_the_sidecar(runner: CliRunner, tmp_path: Path) -> None:
    pt = tmp_path / "m_c2_l2.pt"
    pt.write_bytes(b"x")
    result = runner.invoke(setup_cli.cli, ["models", "publish", str(pt)])
    assert result.exit_code != 0
    assert ".json" in result.output


def test_version_flags() -> None:
    """``--version`` works on the pipeline and cellclass CLIs (#26)."""
    from importlib.metadata import version

    from bin.run_omero_screen import cli as pipeline
    from cellclass.cli import cli as cellclass

    runner = CliRunner()
    out = runner.invoke(pipeline, ["--version"])
    assert out.exit_code == 0 and version("omero-screen") in out.output
    out = runner.invoke(cellclass, ["--version"])
    assert out.exit_code == 0 and version("cellclass") in out.output
