import os

import pytest

from omero_screen.config import set_env_vars


@pytest.mark.parametrize(
    "env, expected_host, expected_loglevel",
    [
        ("development", "localhost", "DEBUG"),
        ("production", "ome2.hpc.sussex.ac.uk", "WARNING"),
    ],
)
def test_set_env_vars_local(
    env, expected_host, expected_loglevel, test_env_files, monkeypatch
):
    os.environ["ENV"] = env
    # Patch the project root to point to our temporary directory
    monkeypatch.setattr("omero_screen.config.project_root", test_env_files)

    set_env_vars()
    host = os.getenv("HOST")
    loglevel = os.getenv("LOG_LEVEL")

    assert host == expected_host, "Host is not correct"
    assert loglevel == expected_loglevel, "Log level is not correct"
