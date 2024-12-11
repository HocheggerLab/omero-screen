import os

import pytest
from omero_screen import set_env_vars


@pytest.mark.parametrize(
    "env, expected_host, expected_loglevel",
    [
        ("development", "localhost", "DEBUG"),
        ("production", "ome2.hpc.sussex.ac.uk", "WARNING"),
    ],
)
def test_set_env_vars_local(env, expected_host, expected_loglevel):
    os.environ["ENV"] = env
    set_env_vars()
    host = os.getenv("HOST")
    loglevel = os.getenv("LOG_LEVEL")

    assert host == expected_host, "Username is not correct"
    assert loglevel == expected_loglevel, "Log level is not correct"
