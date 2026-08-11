"""Tests for --project / --experiment resolution on the import commands."""

import argparse

import pytest
from cellview.main import _resolve_import_target
from cellview.utils.error_classes import DBError
from cellview.utils.state import create_cellview_state


@pytest.fixture
def populated_conn(db):
    """A connection with two projects and one experiment in each."""
    conn = db.connect()
    conn.execute(
        "INSERT INTO projects (project_id, project_name) VALUES "
        "(1, 'project one'), (2, 'project two')"
    )
    conn.execute(
        "INSERT INTO experiments (experiment_id, project_id, experiment_name) "
        "VALUES (10, 1, 'exp in one'), (20, 2, 'exp in two')"
    )
    return conn


def _args(project=None, experiment=None):
    return argparse.Namespace(project=project, experiment=experiment)


class TestResolveImportTarget:
    """Tests for _resolve_import_target."""

    def test_neither_given_returns_none(self, populated_conn) -> None:
        assert _resolve_import_target(_args(), populated_conn) == (None, None)

    def test_missing_attributes_are_tolerated(self, populated_conn) -> None:
        # Namespaces built elsewhere may not carry the flags at all.
        assert _resolve_import_target(
            argparse.Namespace(), populated_conn
        ) == (None, None)

    def test_project_only(self, populated_conn) -> None:
        assert _resolve_import_target(_args(project=1), populated_conn) == (
            1,
            None,
        )

    def test_experiment_implies_its_project(self, populated_conn) -> None:
        assert _resolve_import_target(_args(experiment=20), populated_conn) == (
            2,
            20,
        )

    def test_matching_pair(self, populated_conn) -> None:
        assert _resolve_import_target(
            _args(project=1, experiment=10), populated_conn
        ) == (1, 10)

    def test_unknown_project_raises(self, populated_conn) -> None:
        with pytest.raises(DBError, match="Project 99 does not exist"):
            _resolve_import_target(_args(project=99), populated_conn)

    def test_unknown_experiment_raises(self, populated_conn) -> None:
        with pytest.raises(DBError, match="Experiment 99 does not exist"):
            _resolve_import_target(_args(experiment=99), populated_conn)

    def test_mismatched_pair_raises(self, populated_conn) -> None:
        with pytest.raises(DBError, match="belongs to project 2"):
            _resolve_import_target(
                _args(project=1, experiment=20), populated_conn
            )


class TestStateCarriesTarget:
    """The resolved IDs must reach the state, or the prompts still fire."""

    def test_ids_are_set_on_state(self) -> None:
        state = create_cellview_state(
            argparse.Namespace(
                csv=None,
                plate_id=None,
                nucleus_channel=None,
                project_id=3,
                experiment_id=7,
            )
        )
        assert state.project_id == 3
        assert state.experiment_id == 7

    def test_ids_default_to_none(self) -> None:
        state = create_cellview_state(
            argparse.Namespace(csv=None, plate_id=None, nucleus_channel=None)
        )
        assert state.project_id is None
        assert state.experiment_id is None
