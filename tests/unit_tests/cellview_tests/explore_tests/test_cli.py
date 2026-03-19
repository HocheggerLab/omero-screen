"""Tests for the explore CLI argument parsing."""

from cellview.cli import get_parser


class TestExploreCLIArgs:
    """Tests for explore subcommand arguments."""

    def test_explore_single_plate(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore", "12345"])
        assert args.command == "explore"
        assert args.plate_ids == [12345]

    def test_explore_multiple_plates(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["explore", "12345", "12378", "12390"])
        assert args.plate_ids == [12345, 12378, 12390]

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
