"""Tests for the explore command's argument handling."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cellview.cli import cli
from cellview.main import _parse_plate_ids


def _explore(argv):
    """Invoke `cellview explore ...` and return the handler's kwargs."""
    with patch("cellview.main.handle_explore") as mock:
        result = CliRunner().invoke(cli, ["explore", *argv])
    assert result.exit_code == 0, result.output
    return mock.call_args.kwargs


class TestExploreCLIArgs:
    """Tests for explore subcommand arguments."""

    def test_explore_single_plate(self) -> None:
        assert _explore(["12345"])["plate_ids"] == ["12345"]

    def test_explore_multiple_plates(self) -> None:
        kwargs = _explore(["12345", "12378", "12390"])
        assert kwargs["plate_ids"] == ["12345", "12378", "12390"]

    def test_explore_notebook_name(self) -> None:
        kwargs = _explore(["plates_3602_3603_3604"])
        assert kwargs["plate_ids"] == ["plates_3602_3603_3604"]

    def test_explore_experiment_by_name(self) -> None:
        kwargs = _explore(["--experiment", "palb_washout"])
        assert kwargs["experiment"] == "palb_washout"

    def test_explore_experiment_by_id(self) -> None:
        """The option stays a string; main resolves it to an int."""
        assert _explore(["--experiment", "6"])["experiment"] == "6"

    def test_fresh_flag(self) -> None:
        assert _explore(["12345", "--fresh"])["fresh"] is True

    def test_no_napari_flag(self) -> None:
        assert _explore(["12345", "--no-napari"])["no_napari"] is True

    def test_code_flag(self) -> None:
        assert _explore(["12345", "--code"])["code"] is True

    def test_explore_defaults(self) -> None:
        kwargs = _explore([])
        assert kwargs["plate_ids"] == []
        assert kwargs["experiment"] is None
        assert kwargs["fresh"] is False
        assert kwargs["no_napari"] is False
        assert kwargs["code"] is False
        assert kwargs["template"] == "cellcycle"


class TestParsePlateIds:
    """Tests for _parse_plate_ids helper."""

    def test_plain_integers(self) -> None:
        assert _parse_plate_ids(["3602", "3603", "3604"]) == [3602, 3603, 3604]

    def test_single_integer(self) -> None:
        assert _parse_plate_ids(["12345"]) == [12345]

    def test_notebook_name_multiple_plates(self) -> None:
        assert _parse_plate_ids(["plates_3602_3603_3604"]) == [3602, 3603, 3604]

    def test_notebook_name_single_plate(self) -> None:
        assert _parse_plate_ids(["plate_3602"]) == [3602]

    def test_notebook_name_with_explore_prefix(self) -> None:
        assert _parse_plate_ids(["explore_plates_3602_3603"]) == [3602, 3603]

    def test_notebook_name_with_ipynb_suffix(self) -> None:
        assert _parse_plate_ids(["plates_3602_3603.ipynb"]) == [3602, 3603]

    def test_notebook_name_full_filename(self) -> None:
        assert _parse_plate_ids(["explore_plates_3602_3603.ipynb"]) == [
            3602,
            3603,
        ]

    def test_results_are_sorted(self) -> None:
        assert _parse_plate_ids(["3604", "3602", "3603"]) == [3602, 3603, 3604]

    def test_invalid_name_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_plate_ids(["no_ids_here"])

    def test_invalid_mixed_args_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_plate_ids(["3602", "not_a_number"])
