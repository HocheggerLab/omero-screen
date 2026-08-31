"""Behaviour tests for the CellView Click CLI.

These drive the exported ``cli`` group through ``CliRunner``. Each command's
work happens in a handler that is patched out here, so the assertions are
about the command surface — names, nesting, types, defaults, variadics and
exit codes — not about the database.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cellview.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


# --------------------------------------------------------------------------
# Command surface
# --------------------------------------------------------------------------


class TestNoCommand:
    """A bare `cellview` printed help and exited 0 under argparse."""

    def test_no_args_prints_help_and_exits_zero(self, runner) -> None:
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_top_level_help_lists_every_command(self, runner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for command in (
            "projects",
            "project",
            "experiment",
            "plate",
            "import",
            "edit",
            "export",
            "delete",
            "clean",
            "explore",
            "template",
        ):
            assert command in result.output

    def test_unknown_command_is_a_usage_error(self, runner) -> None:
        assert runner.invoke(cli, ["nosuchcommand"]).exit_code == 2

    @pytest.mark.parametrize("group", ["import", "edit", "delete"])
    def test_bare_group_prints_help_and_exits_zero(
        self, runner, group
    ) -> None:
        """argparse dispatched to `<group> --help`, which exits 0."""
        result = runner.invoke(cli, [group])
        assert result.exit_code == 0
        assert "Usage:" in result.output


class TestGlobalDbOption:
    """--db is a group-level option, given before the subcommand."""

    def test_db_reaches_the_context(self, runner) -> None:
        with patch("cellview.db.display.display_projects") as mock_display:
            result = runner.invoke(
                cli, ["--db", "/tmp/test.duckdb", "projects"]
            )
        assert result.exit_code == 0
        mock_display.assert_called_once()

    def test_db_after_subcommand_is_rejected(self, runner) -> None:
        result = runner.invoke(cli, ["projects", "--db", "/tmp/x.duckdb"])
        assert result.exit_code == 2


# --------------------------------------------------------------------------
# Display commands
# --------------------------------------------------------------------------


class TestDisplayCommands:
    """Tests for the display subcommands."""

    def test_projects(self, runner) -> None:
        with patch("cellview.db.display.display_projects") as mock:
            assert runner.invoke(cli, ["projects"]).exit_code == 0
        mock.assert_called_once()

    def test_project_takes_an_int_id(self, runner) -> None:
        with patch("cellview.db.display.display_single_project") as mock:
            assert runner.invoke(cli, ["project", "5"]).exit_code == 0
        assert mock.call_args.args[1] == 5

    def test_experiment_takes_an_int_id(self, runner) -> None:
        with patch("cellview.db.display.display_experiment") as mock:
            assert runner.invoke(cli, ["experiment", "3"]).exit_code == 0
        assert mock.call_args.args[1] == 3

    def test_plate_takes_an_int_id(self, runner) -> None:
        with patch("cellview.db.display.display_plate_summary") as mock:
            assert runner.invoke(cli, ["plate", "12345"]).exit_code == 0
        assert mock.call_args.args[0] == 12345

    def test_non_integer_id_is_rejected(self, runner) -> None:
        assert runner.invoke(cli, ["project", "abc"]).exit_code == 2


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------


class TestImportCommands:
    """Tests for the nested import subcommands."""

    def test_import_csv(self, runner) -> None:
        with patch("cellview.main.handle_import_csv") as mock:
            result = runner.invoke(cli, ["import", "csv", "/tmp/data.csv"])
        assert result.exit_code == 0
        kwargs = mock.call_args.kwargs
        assert kwargs["path"] == Path("/tmp/data.csv")
        assert kwargs["nucleus_channel"] is None
        assert kwargs["project"] is None
        assert kwargs["experiment"] is None

    def test_import_plate_single(self, runner) -> None:
        with patch("cellview.main.handle_import_plate") as mock:
            result = runner.invoke(cli, ["import", "plate", "12345"])
        assert result.exit_code == 0
        assert mock.call_args.kwargs["ids"] == [12345]

    def test_import_plate_multiple(self, runner) -> None:
        with patch("cellview.main.handle_import_plate") as mock:
            result = runner.invoke(cli, ["import", "plate", "12345", "12378"])
        assert result.exit_code == 0
        assert mock.call_args.kwargs["ids"] == [12345, 12378]

    def test_import_plate_requires_at_least_one_id(self, runner) -> None:
        assert runner.invoke(cli, ["import", "plate"]).exit_code == 2

    def test_import_plate_interactive(self, runner) -> None:
        with patch("cellview.main.handle_import_plate") as mock:
            result = runner.invoke(
                cli, ["import", "plate", "12345", "--interactive"]
            )
        assert result.exit_code == 0
        assert mock.call_args.kwargs["interactive"] is True

    def test_import_screen(self, runner) -> None:
        with patch("cellview.main.handle_import_screen") as mock:
            result = runner.invoke(cli, ["import", "screen", "99"])
        assert result.exit_code == 0
        assert mock.call_args.kwargs["screen_id"] == 99

    def test_nucleus_channel_on_every_route(self, runner) -> None:
        cases = [
            (["import", "csv", "/tmp/d.csv"], "cellview.main.handle_import_csv"),
            (["import", "plate", "1"], "cellview.main.handle_import_plate"),
            (["import", "screen", "9"], "cellview.main.handle_import_screen"),
        ]
        for argv, target in cases:
            with patch(target) as mock:
                result = runner.invoke(
                    cli, [*argv, "--nucleus-channel", "H2B_RFP"]
                )
            assert result.exit_code == 0, argv
            assert mock.call_args.kwargs["nucleus_channel"] == "H2B_RFP"

    def test_project_and_experiment_on_every_route(self, runner) -> None:
        cases = [
            (["import", "plate", "1", "2"], "cellview.main.handle_import_plate"),
            (["import", "screen", "99"], "cellview.main.handle_import_screen"),
            (["import", "csv", "/tmp/data.csv"], "cellview.main.handle_import_csv"),
        ]
        for argv, target in cases:
            with patch(target) as mock:
                result = runner.invoke(
                    cli, [*argv, "--project", "3", "--experiment", "7"]
                )
            assert result.exit_code == 0, argv
            assert mock.call_args.kwargs["project"] == 3
            assert mock.call_args.kwargs["experiment"] == 7


# --------------------------------------------------------------------------
# Edit, export, delete, clean
# --------------------------------------------------------------------------


class TestEditCommands:
    """Tests for the nested edit subcommands."""

    def test_edit_project(self, runner) -> None:
        with patch("cellview.db.edit.edit_project") as mock:
            assert runner.invoke(cli, ["edit", "project", "1"]).exit_code == 0
        assert mock.call_args.args[0] == 1

    def test_edit_experiment(self, runner) -> None:
        with patch("cellview.db.edit.edit_experiment") as mock:
            result = runner.invoke(cli, ["edit", "experiment", "2"])
        assert result.exit_code == 0
        assert mock.call_args.args[0] == 2


class TestExportCommand:
    """Tests for the export subcommand."""

    def test_export(self, runner) -> None:
        with patch(
            "cellview.exporters.db_to_pandas.export_pandas_df"
        ) as mock:
            mock.return_value = ([], ["a", "b"])
            result = runner.invoke(cli, ["export", "42"])
        assert result.exit_code == 0
        assert mock.call_args.args[0] == 42
        assert "Exported plate 42" in result.output


class TestDeleteCommand:
    """Tests for the delete subcommand."""

    def test_delete_single_plate(self, runner) -> None:
        with (
            patch("cellview.db.clean_up.del_measurements_by_plate_id") as mock,
            patch("cellview.db.clean_up.clean_up_db") as mock_clean,
        ):
            result = runner.invoke(cli, ["delete", "plate", "12345"])
        assert result.exit_code == 0
        assert mock.call_count == 1
        assert mock.call_args.args[2] == 12345
        mock_clean.assert_called_once()

    def test_delete_multiple_plates_cleans_up_once(self, runner) -> None:
        with (
            patch("cellview.db.clean_up.del_measurements_by_plate_id") as mock,
            patch("cellview.db.clean_up.clean_up_db") as mock_clean,
        ):
            result = runner.invoke(cli, ["delete", "plate", "1", "2", "3"])
        assert result.exit_code == 0
        assert [c.args[2] for c in mock.call_args_list] == [1, 2, 3]
        mock_clean.assert_called_once()

    def test_delete_plate_requires_at_least_one_id(self, runner) -> None:
        assert runner.invoke(cli, ["delete", "plate"]).exit_code == 2


class TestCleanCommand:
    """Tests for the clean subcommand."""

    def test_clean(self, runner) -> None:
        with patch("cellview.db.clean_up.clean_up_db") as mock:
            assert runner.invoke(cli, ["clean"]).exit_code == 0
        mock.assert_called_once()


# --------------------------------------------------------------------------
# Explore
# --------------------------------------------------------------------------


class TestExploreCommand:
    """Explore is the one command that must not open the database."""

    def test_explore_defaults(self, runner) -> None:
        with patch("cellview.main.handle_explore") as mock:
            assert runner.invoke(cli, ["explore"]).exit_code == 0
        kwargs = mock.call_args.kwargs
        assert kwargs["plate_ids"] == []
        assert kwargs["template"] == "cellcycle"
        assert kwargs["fresh"] is False
        assert kwargs["no_napari"] is False
        assert kwargs["code"] is False
        assert kwargs["json_output"] is False

    def test_explore_full_contract(self, runner) -> None:
        with patch("cellview.main.handle_explore") as mock:
            result = runner.invoke(
                cli,
                [
                    "explore",
                    "12",
                    "34",
                    "--experiment",
                    "DNA-damage",
                    "--template",
                    "cellcycle",
                    "--fresh",
                    "--no-napari",
                    "--code",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        assert mock.call_args.kwargs == {
            "plate_ids": ["12", "34"],
            "experiment": "DNA-damage",
            "template": "cellcycle",
            "fresh": True,
            "no_napari": True,
            "code": True,
            "json_output": True,
        }

    def test_explore_does_not_open_the_database(self, runner) -> None:
        with (
            patch("cellview.main.handle_explore"),
            patch("cellview.db.db.CellViewDB") as mock_db,
        ):
            assert runner.invoke(cli, ["explore", "12"]).exit_code == 0
        mock_db.assert_not_called()


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


class TestTemplateCommands:
    """Tests for the nested template subcommands."""

    def test_bare_template_lists(self, runner) -> None:
        """argparse treated a bare `template` as `template list`."""
        with patch("cellview.main.handle_template_list") as mock:
            assert runner.invoke(cli, ["template"]).exit_code == 0
        mock.assert_called_once()

    def test_template_list(self, runner) -> None:
        with patch("cellview.main.handle_template_list") as mock:
            assert runner.invoke(cli, ["template", "list"]).exit_code == 0
        mock.assert_called_once()

    def test_template_sync(self, runner) -> None:
        with patch("cellview.main.handle_template_sync") as mock:
            assert runner.invoke(cli, ["template", "sync"]).exit_code == 0
        mock.assert_called_once()

    def test_template_add_full_contract(self, runner) -> None:
        with patch("cellview.main.handle_template_add") as mock:
            result = runner.invoke(
                cli,
                [
                    "template",
                    "add",
                    "/tmp/analysis.py",
                    "--name",
                    "custom",
                    "--description",
                    "Custom analysis",
                ],
            )
        assert result.exit_code == 0
        kwargs = mock.call_args.kwargs
        assert kwargs["path"] == Path("/tmp/analysis.py")
        assert kwargs["name"] == "custom"
        assert kwargs["description"] == "Custom analysis"

    def test_template_remove(self, runner) -> None:
        with patch("cellview.main.handle_template_remove") as mock:
            result = runner.invoke(cli, ["template", "remove", "old"])
        assert result.exit_code == 0
        assert mock.call_args.kwargs["name"] == "old"

    def test_template_show(self, runner) -> None:
        with patch("cellview.main.handle_template_show") as mock:
            result = runner.invoke(cli, ["template", "show", "cellcycle"])
        assert result.exit_code == 0
        assert mock.call_args.kwargs["name"] == "cellcycle"
