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
    env,
    expected_host,
    expected_loglevel,
    test_env_files,
    monkeypatch,
    clean_env,
):
    os.environ["ENV"] = env

    # Patch the Path class's parent.parent.parent.resolve() chain
    def mock_resolve(self):
        return test_env_files

    monkeypatch.setattr("pathlib.Path.resolve", mock_resolve)

    set_env_vars()
    host = os.getenv("HOST")
    loglevel = os.getenv("LOG_LEVEL")

    assert host == expected_host, "Host is not correct"
    assert loglevel == expected_loglevel, "Log level is not correct"
