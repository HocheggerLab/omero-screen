"""Tests for the explore notebook registry."""

from pathlib import Path

import pytest

from cellview.explore._registry import (
    experiment_notebook_exists,
    notebook_path_for_experiment,
    notebook_path_for_plates,
    notebooks_for_plate,
)


class TestNotebookPathForPlates:
    """Tests for notebook_path_for_plates."""

    def test_single_plate(self) -> None:
        path = notebook_path_for_plates([12345])
        assert path.name == "explore_plate_12345.ipynb"

    def test_multiple_plates_sorted(self) -> None:
        path = notebook_path_for_plates([12378, 12345])
        assert path.name == "explore_plates_12345_12378.ipynb"

    def test_returns_path_in_explore_dir(self) -> None:
        path = notebook_path_for_plates([12345])
        assert path.parent == Path.home() / ".cellview" / "explore"


class TestNotebookPathForExperiment:
    """Tests for notebook_path_for_experiment."""

    def test_returns_correct_name(self) -> None:
        path = notebook_path_for_experiment(6)
        assert path.name == "explore_exp_6.ipynb"


class TestNotebooksForPlate:
    """Tests for notebooks_for_plate using temp directory."""

    def test_returns_empty_when_dir_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cellview.explore._registry.EXPLORE_DIR",
            tmp_path / "nonexistent",
        )
        assert notebooks_for_plate(12345) == []

    def test_finds_single_plate_notebook(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        (tmp_path / "explore_plate_12345.ipynb").touch()
        result = notebooks_for_plate(12345)
        assert result == ["plate_12345"]

    def test_finds_multi_plate_notebook(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        (tmp_path / "explore_plates_12345_12378.ipynb").touch()
        result = notebooks_for_plate(12345)
        assert result == ["plates_12345_12378"]

    def test_finds_multiple_notebooks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        (tmp_path / "explore_plate_12345.ipynb").touch()
        (tmp_path / "explore_plates_12345_12378.ipynb").touch()
        result = notebooks_for_plate(12345)
        assert len(result) == 2
        assert "plate_12345" in result
        assert "plates_12345_12378" in result

    def test_does_not_match_unrelated_plate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        (tmp_path / "explore_plate_99999.ipynb").touch()
        assert notebooks_for_plate(12345) == []


class TestExperimentNotebookExists:
    """Tests for experiment_notebook_exists."""

    def test_returns_false_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        assert experiment_notebook_exists(6) is False

    def test_returns_true_when_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        (tmp_path / "explore_exp_6.ipynb").touch()
        assert experiment_notebook_exists(6) is True
