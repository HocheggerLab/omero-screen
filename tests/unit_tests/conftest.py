import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Automatically set up test environment for all tests.
    This ensures ENV is set to 'test' by default unless explicitly overridden.
    """
    old_env = os.environ.get("ENV")
    os.environ["ENV"] = "test"
    yield
    if old_env is None:
        del os.environ["ENV"]
    else:
        os.environ["ENV"] = old_env
