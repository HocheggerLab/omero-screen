"""Importing omero_utils must not load the pipeline or the OMERO gateway."""

import subprocess
import sys


def _loaded_after(statement: str) -> set[str]:
    code = (
        f"import sys; {statement}; "
        "print(','.join(m for m in ('omero_screen', 'omero.gateway') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return set(filter(None, out.split(",")))


def test_channels_import_is_light() -> None:
    """The channel rules load without core or the gateway."""
    assert _loaded_after("import omero_utils.channels") == set()


def test_package_import_is_light() -> None:
    """``import omero_utils`` exposes omero_connect without loading core."""
    assert _loaded_after("from omero_utils import omero_connect") == set()


def test_omero_connect_is_the_decorator() -> None:
    """The package attribute stays the function after the submodule loads."""
    import omero_utils
    import omero_utils.omero_connect  # noqa: F401

    assert callable(omero_utils.omero_connect)
