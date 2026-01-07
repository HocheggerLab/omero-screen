
import argparse
import pytest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
from pathlib import Path

from omero_screen_napari.trainingdata_db.cli import (
    setup_parser,
    handle_migrate,
    handle_list,
    handle_stats,
    handle_export,
    main,
)


@pytest.fixture
def parser():
    return setup_parser()


@pytest.fixture
def mock_db():
    with patch("omero_screen_napari.trainingdata_db.cli.TrainingDB") as MockDB:
        db_instance = MockDB.return_value
        yield db_instance


@pytest.fixture
def mock_console():
    with patch("omero_screen_napari.trainingdata_db.cli.console") as mock_console:
        yield mock_console


def test_parser_migrate(parser):
    args = parser.parse_args(["migrate", "--dry-run", "--path", "/tmp/data"])
    assert args.command == "migrate"
    assert args.dry_run is True
    assert args.path == Path("/tmp/data")


def test_parser_list(parser):
    args = parser.parse_args(["list"])
    assert args.command == "list"


def test_parser_stats(parser):
    args = parser.parse_args(["stats", "MyClassifier"])
    assert args.command == "stats"
    assert args.classifier == "MyClassifier"


def test_parser_export(parser):
    args = parser.parse_args(["export", "MyClassifier", "--format", "json"])
    assert args.command == "export"
    assert args.classifier == "MyClassifier"
    assert args.format == "json"


@patch("omero_screen_napari.trainingdata_db.cli.migrate_all_classifiers")
def test_handle_migrate(mock_migrate, mock_console):
    args = argparse.Namespace(path=Path("/tmp"), dry_run=True)
    handle_migrate(args)
    mock_migrate.assert_called_once_with(base_dir=Path("/tmp"), dry_run=True)
    mock_console.print.assert_called()


def test_handle_list(mock_db, mock_console):
    mock_db.list_classifiers.return_value = [
        {"id": 1, "name": "C1", "description": "D1", "created_at": "2023-01-01"}
    ]
    mock_db.get_session_count.return_value = 5
    mock_db.get_total_annotations.return_value = 100

    handle_list(argparse.Namespace())

    mock_db.list_classifiers.assert_called_once()
    mock_console.print.assert_called()


def test_handle_list_empty(mock_db, mock_console):
    mock_db.list_classifiers.return_value = []
    handle_list(argparse.Namespace())
    mock_console.print.assert_any_call("[yellow]No classifiers found.[/yellow]")


def test_handle_stats(mock_db, mock_console):
    mock_db.get_classifier.return_value = {"id": 1, "name": "C1"}
    mock_db.get_session_count.return_value = 10
    mock_db.get_total_annotations.return_value = 500
    mock_db.get_classes.return_value = ["A", "B"]
    mock_db.get_class_distribution.return_value = {"A": 300, "B": 200}

    args = argparse.Namespace(classifier="C1")
    handle_stats(args)

    mock_db.get_classifier.assert_called_with("C1")
    mock_db.get_class_distribution.assert_called_with("C1")


def test_handle_stats_not_found(mock_db, mock_console):
    mock_db.get_classifier.return_value = None
    args = argparse.Namespace(classifier="C1")

    with pytest.raises(SystemExit):
        handle_stats(args)


@patch("omero_screen_napari.trainingdata_db.cli.pd.DataFrame")
def test_handle_export(mock_df_cls, mock_db, mock_console, tmp_path):
    # Setup
    mock_db.get_classifier.return_value = {"id": 1}
    mock_db.get_annotations_by_classifier.return_value = [{"col": "val"}]
    mock_df = MagicMock()
    mock_df_cls.return_value = mock_df
    mock_df.__len__.return_value = 1

    args = argparse.Namespace(
        classifier="C1", format="csv", output=None, plate=None, well=None
    )

    # Execute
    handle_export(args)

    # Assert
    mock_db.get_annotations_by_classifier.assert_called_once()
    mock_df.to_csv.assert_called_once()


def test_handle_export_no_data(mock_db, mock_console):
    mock_db.get_classifier.return_value = {"id": 1}
    mock_db.get_annotations_by_classifier.return_value = []

    args = argparse.Namespace(
        classifier="C1", format="csv", output=None, plate=None, well=None
    )

    handle_export(args)

    mock_console.print.assert_any_call("[yellow]No data found to export.[/yellow]")
