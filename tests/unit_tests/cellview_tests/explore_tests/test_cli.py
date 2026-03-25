"""Tests for the explore CLI argument parsing."""

import pytest

from cellview.cli import get_parser
from cellview.main import _parse_plate_ids


class TestExploreCLIArgs:
    """Tests for explore subcommand arguments."""

    def test_explore_single_plate(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore", "12345"])
        assert args.command == "explore"
        assert args.plate_ids == ["12345"]

    def test_explore_multiple_plates(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore", "12345", "12378", "12390"])
        assert args.plate_ids == ["12345", "12378", "12390"]

    def test_explore_notebook_name(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore", "plates_3602_3603_3604"])
        assert args.plate_ids == ["plates_3602_3603_3604"]

    def test_explore_experiment_by_name(self) -> None:
        parser = get_parser()
        args = parser.parse_args(
            ["explore", "--experiment", "palb_washout"]
        )
        assert args.experiment == "palb_washout"

    def test_explore_experiment_by_id(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore", "--experiment", "6"])
        assert args.experiment == "6"

    def test_fresh_flag(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore", "12345", "--fresh"])
        assert args.fresh is True

    def test_no_napari_flag(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore", "12345", "--no-napari"])
        assert args.no_napari is True

    def test_explore_defaults(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore"])
        assert args.plate_ids == []
        assert args.experiment is None
        assert args.fresh is False
        assert args.no_napari is False
        assert args.template == "cellcycle"
        assert args.list_templates is False


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
        assert _parse_plate_ids(["explore_plates_3602_3603.ipynb"]) == [3602, 3603]

    def test_results_are_sorted(self) -> None:
        assert _parse_plate_ids(["3604", "3602", "3603"]) == [3602, 3603, 3604]

    def test_invalid_name_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_plate_ids(["no_ids_here"])

    def test_invalid_mixed_args_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_plate_ids(["3602", "not_a_number"])
