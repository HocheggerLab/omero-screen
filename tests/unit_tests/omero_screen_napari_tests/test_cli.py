"""Behaviour tests for the ``omero-train`` Click CLI.

These exercise the exported ``cli`` group through ``CliRunner`` rather than
inspecting a parser, plus direct tests of the handler functions. The parser
contract they replace was recorded on the argparse implementation; the
invocations below are the same ones users and scripts actually type.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from omero_screen_napari.trainingdata_db.cli import (
    cli,
    handle_export,
    handle_list,
    handle_migrate,
    handle_stats,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_db():
    with patch(
        "omero_screen_napari.trainingdata_db.database.TrainingDB"
    ) as MockDB:
        db_instance = MockDB.return_value
        yield db_instance


@pytest.fixture
def mock_console():
    with patch(
        "omero_screen_napari.trainingdata_db.cli.console"
    ) as mock_console:
        yield mock_console


# --------------------------------------------------------------------------
# Command surface
# --------------------------------------------------------------------------


def test_group_help_lists_every_command(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("migrate", "list", "stats", "export", "delete"):
        assert command in result.output


def test_no_command_is_an_error(runner):
    """Argparse printed help and exited non-zero; Click must not exit 0."""
    result = runner.invoke(cli, [])
    assert result.exit_code != 0


def test_help_does_not_import_heavy_dependencies():
    """``--help`` and Great Docs discovery must stay cheap.

    Importing the CLI module must not drag in pandas, napari or the database
    layer — those belong inside the command callbacks.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "import omero_screen_napari.trainingdata_db.cli;"
        "heavy = [m for m in ('pandas', 'napari', 'torch') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.stdout.strip() == "", f"heavy imports at module scope: {out.stdout}"


# --------------------------------------------------------------------------
# Parsing contract — the invocations scripts rely on
# --------------------------------------------------------------------------


@patch("omero_screen_napari.trainingdata_db.cli.handle_migrate")
def test_migrate_parsing(mock_handle, runner):
    result = runner.invoke(cli, ["migrate", "--dry-run", "--path", "/tmp/data"])
    assert result.exit_code == 0
    mock_handle.assert_called_once_with(path=Path("/tmp/data"), dry_run=True)


@patch("omero_screen_napari.trainingdata_db.cli.handle_migrate")
def test_migrate_defaults(mock_handle, runner):
    result = runner.invoke(cli, ["migrate"])
    assert result.exit_code == 0
    mock_handle.assert_called_once_with(path=None, dry_run=False)


@patch("omero_screen_napari.trainingdata_db.cli.handle_list")
def test_list_parsing(mock_handle, runner):
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    mock_handle.assert_called_once_with()


@patch("omero_screen_napari.trainingdata_db.cli.handle_stats")
def test_stats_parsing(mock_handle, runner):
    result = runner.invoke(cli, ["stats", "MyClassifier"])
    assert result.exit_code == 0
    mock_handle.assert_called_once_with(classifier="MyClassifier")


@patch("omero_screen_napari.trainingdata_db.cli.handle_export")
def test_export_parsing_defaults(mock_handle, runner):
    result = runner.invoke(cli, ["export", "MyClassifier", "--format", "json"])
    assert result.exit_code == 0
    mock_handle.assert_called_once_with(
        classifier="MyClassifier",
        output_format="json",
        output=None,
        plate=None,
        well=None,
    )


@patch("omero_screen_napari.trainingdata_db.cli.handle_export")
def test_export_full_contract(mock_handle, runner):
    result = runner.invoke(
        cli,
        [
            "export",
            "MyClassifier",
            "--format",
            "parquet",
            "--output",
            "/tmp/training.parquet",
            "--plate",
            "1234",
            "--well",
            "A1",
        ],
    )
    assert result.exit_code == 0
    mock_handle.assert_called_once_with(
        classifier="MyClassifier",
        output_format="parquet",
        output=Path("/tmp/training.parquet"),
        plate=1234,
        well="A1",
    )


@patch("omero_screen_napari.trainingdata_db.cli.handle_export")
def test_export_short_output_flag(mock_handle, runner):
    result = runner.invoke(cli, ["export", "C1", "-o", "/tmp/out.csv"])
    assert result.exit_code == 0
    assert mock_handle.call_args.kwargs["output"] == Path("/tmp/out.csv")


def test_export_rejects_unknown_format(runner):
    result = runner.invoke(cli, ["export", "C1", "--format", "xlsx"])
    assert result.exit_code == 2


@patch("omero_screen_napari.trainingdata_db.cli.handle_delete")
def test_delete_full_contract_argparse_variadic_spelling(mock_handle, runner):
    """``--plate 1234 5678`` is the argparse spelling and must keep working."""
    result = runner.invoke(
        cli,
        [
            "delete",
            "classifier-a",
            "classifier-b",
            "--yes",
            "--plate",
            "1234",
            "5678",
        ],
    )
    assert result.exit_code == 0
    mock_handle.assert_called_once_with(
        identifiers=["classifier-a", "classifier-b"],
        yes=True,
        plate=[1234, 5678],
    )


@patch("omero_screen_napari.trainingdata_db.cli.handle_delete")
def test_delete_repeated_option_spelling(mock_handle, runner):
    """The native Click spelling must work too."""
    result = runner.invoke(
        cli, ["delete", "c1", "-y", "--plate", "1234", "--plate", "5678"]
    )
    assert result.exit_code == 0
    mock_handle.assert_called_once_with(
        identifiers=["c1"], yes=True, plate=[1234, 5678]
    )


@patch("omero_screen_napari.trainingdata_db.cli.handle_delete")
def test_delete_without_plate_deletes_whole_classifier(mock_handle, runner):
    result = runner.invoke(cli, ["delete", "c1"])
    assert result.exit_code == 0
    mock_handle.assert_called_once_with(
        identifiers=["c1"], yes=False, plate=None
    )


def test_delete_requires_an_identifier(runner):
    result = runner.invoke(cli, ["delete"])
    assert result.exit_code == 2


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


@patch("omero_screen_napari.trainingdata_db.migrator.migrate_all_classifiers")
def test_handle_migrate(mock_migrate, mock_console):
    handle_migrate(path=Path("/tmp"), dry_run=True)
    mock_migrate.assert_called_once_with(base_dir=Path("/tmp"), dry_run=True)
    mock_console.print.assert_called()


def test_handle_list(mock_db, mock_console):
    mock_db.list_classifiers.return_value = [
        {"id": 1, "name": "C1", "description": "D1", "created_at": "2023-01-01"}
    ]
    mock_db.get_session_count.return_value = 5
    mock_db.get_total_annotations.return_value = 100

    handle_list()

    mock_db.list_classifiers.assert_called_once()
    mock_console.print.assert_called()


def test_handle_list_empty(mock_db, mock_console):
    mock_db.list_classifiers.return_value = []
    handle_list()
    mock_console.print.assert_any_call(
        "[yellow]No classifiers found.[/yellow]"
    )


def test_handle_stats(mock_db, mock_console):
    mock_db.get_classifier.return_value = {"id": 1, "name": "C1"}
    mock_db.get_session_count.return_value = 10
    mock_db.get_total_annotations.return_value = 500
    mock_db.get_classes.return_value = ["A", "B"]
    mock_db.get_class_distribution.return_value = {"A": 300, "B": 200}

    handle_stats(classifier="C1")

    mock_db.get_classifier.assert_called_with("C1")
    mock_db.get_class_distribution.assert_called_with("C1")


def test_handle_stats_not_found(mock_db, mock_console):
    mock_db.get_classifier.return_value = None

    with pytest.raises(SystemExit):
        handle_stats(classifier="C1")


def test_handle_stats_by_id(mock_db, mock_console):
    mock_db.get_classifier_by_id.return_value = {"id": 1, "name": "C1"}
    mock_db.get_session_count.return_value = 10
    mock_db.get_total_annotations.return_value = 500
    mock_db.get_classes.return_value = ["A", "B"]
    mock_db.get_class_distribution.return_value = {"A": 300, "B": 200}

    handle_stats(classifier="1")

    mock_db.get_classifier_by_id.assert_called_with(1)
    # Shouldn't fall back to a name lookup when the ID resolves.
    mock_db.get_classifier.assert_not_called()
    mock_console.print.assert_called()


def test_handle_stats_not_found_id(mock_db, mock_console):
    mock_db.get_classifier_by_id.return_value = None
    mock_db.get_classifier.return_value = None

    with pytest.raises(SystemExit):
        handle_stats(classifier="999")

    mock_db.get_classifier_by_id.assert_called_with(999)
    mock_db.get_classifier.assert_called_with("999")


@patch("pandas.DataFrame")
def test_handle_export(mock_df_cls, mock_db, mock_console, tmp_path):
    mock_db.get_classifier.return_value = {"id": 1, "name": "C1"}
    mock_db.get_annotations_by_classifier.return_value = [{"col": "val"}]
    mock_df = MagicMock()
    mock_df_cls.return_value = mock_df
    mock_df.__len__.return_value = 1

    handle_export(
        classifier="C1",
        output_format="csv",
        output=None,
        plate=None,
        well=None,
    )

    mock_db.get_annotations_by_classifier.assert_called_once()
    mock_df.to_csv.assert_called_once()


def test_handle_export_no_data(mock_db, mock_console):
    mock_db.get_classifier.return_value = {"id": 1, "name": "C1"}
    mock_db.get_annotations_by_classifier.return_value = []

    handle_export(
        classifier="C1",
        output_format="csv",
        output=None,
        plate=None,
        well=None,
    )

    mock_console.print.assert_any_call(
        "[yellow]No data found to export.[/yellow]"
    )
