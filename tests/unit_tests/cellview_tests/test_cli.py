"""Tests for the CellView CLI subcommand parser."""

from pathlib import Path

import pytest

from cellview.cli import get_parser


class TestNoCommand:
    """Tests for when no subcommand is given."""

    def test_no_args_gives_none_command(self) -> None:
        parser = get_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_global_db_option(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["--db", "/tmp/test.duckdb", "projects"])
        assert args.db == Path("/tmp/test.duckdb")
        assert args.command == "projects"

    def test_db_defaults_to_none(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["projects"])
        assert args.db is None


class TestDisplayCommands:
    """Tests for display subcommands."""

    def test_projects(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["projects"])
        assert args.command == "projects"

    def test_project(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["project", "5"])
        assert args.command == "project"
        assert args.id == 5

    def test_experiment(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["experiment", "3"])
        assert args.command == "experiment"
        assert args.id == 3

    def test_plate(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["plate", "12345"])
        assert args.command == "plate"
        assert args.id == 12345


class TestImportCommands:
    """Tests for import subcommands."""

    def test_import_csv(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["import", "csv", "/tmp/data.csv"])
        assert args.command == "import"
        assert args.import_command == "csv"
        assert args.path == Path("/tmp/data.csv")

    def test_import_plate_single(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["import", "plate", "12345"])
        assert args.import_command == "plate"
        assert args.ids == [12345]

    def test_import_plate_multiple(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["import", "plate", "12345", "12378"])
        assert args.ids == [12345, 12378]

    def test_import_plate_interactive(self) -> None:
        parser = get_parser()
        args = parser.parse_args(
            ["import", "plate", "12345", "--interactive"]
        )
        assert args.interactive is True

    def test_import_screen(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["import", "screen", "99"])
        assert args.import_command == "screen"
        assert args.id == 99

    def test_import_screen_interactive(self) -> None:
        parser = get_parser()
        args = parser.parse_args(
            ["import", "screen", "99", "--interactive"]
        )
        assert args.interactive is True

    def test_import_target_defaults_to_none(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["import", "plate", "12345"])
        assert args.project is None
        assert args.experiment is None

    @pytest.mark.parametrize(
        "argv",
        [
            ["import", "plate", "1", "2"],
            ["import", "screen", "99"],
            ["import", "csv", "/tmp/data.csv"],
        ],
    )
    def test_import_accepts_project_and_experiment(
        self, argv: list[str]
    ) -> None:
        parser = get_parser()
        args = parser.parse_args(
            [*argv, "--project", "3", "--experiment", "7"]
        )
        assert args.project == 3
        assert args.experiment == 7


class TestEditCommands:
    """Tests for edit subcommands."""

    def test_edit_project(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["edit", "project", "1"])
        assert args.command == "edit"
        assert args.edit_command == "project"
        assert args.id == 1

    def test_edit_experiment(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["edit", "experiment", "2"])
        assert args.edit_command == "experiment"
        assert args.id == 2


class TestExportCommand:
    """Tests for export subcommand."""

    def test_export(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["export", "42"])
        assert args.command == "export"
        assert args.id == 42


class TestDeleteCommand:
    """Tests for delete subcommand."""

    def test_delete_plate(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["delete", "plate", "12345"])
        assert args.command == "delete"
        assert args.delete_command == "plate"
        assert args.ids == [12345]

    def test_delete_multiple_plates(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["delete", "plate", "1", "2", "3"])
        assert args.delete_command == "plate"
        assert args.ids == [1, 2, 3]

    def test_delete_plate_requires_at_least_one_id(self) -> None:
        parser = get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["delete", "plate"])


class TestCleanCommand:
    """Tests for clean subcommand."""

    def test_clean(self) -> None:
        parser = get_parser()
        args = parser.parse_args(["clean"])
        assert args.command == "clean"
