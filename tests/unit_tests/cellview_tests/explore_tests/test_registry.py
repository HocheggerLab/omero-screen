"""Tests for the explore notebook registry."""

from pathlib import Path

import pytest
from cellview.explore._registry import (
    experiment_notebook_exists,
    legacy_notebook_path_for_experiment,
    notebook_path_for_experiment,
    notebook_path_for_plates,
    notebooks_for_plate,
    sanitize_folder_name,
)


class TestNotebookPathForPlates:
    """Tests for notebook_path_for_plates."""

    def test_single_plate(self) -> None:
        path = notebook_path_for_plates([12345])
        assert path.name == "explore_plate_12345.ipynb"
        assert path.parent.name == "12345"
        assert path.parent.parent.name == "plates"

    def test_multiple_plates_sorted(self) -> None:
        path = notebook_path_for_plates([12378, 12345])
        assert path.name == "explore_plates_12345_12378.ipynb"
        assert path.parent.name == "12345_12378"

    def test_folder_override(self) -> None:
        path = notebook_path_for_plates([12345], folder_name="Project Name/Exp A")
        assert path.name == "explore_plate_12345.ipynb"
        assert path.parent.name == "Exp_A"
        assert path.parent.parent.name == "Project_Name"


class TestNotebookPathForExperiment:
    """Tests for notebook_path_for_experiment."""

    def test_returns_correct_name(self) -> None:
        path = notebook_path_for_experiment(6)
        assert path.name == "explore_exp_6.ipynb"

    def test_uses_readable_folder_name(self) -> None:
        path = notebook_path_for_experiment(6, folder_name="Project/Experiment Name")
        assert path.name == "explore_exp_6.ipynb"
        assert path.parent.name == "Experiment_Name"
        assert path.parent.parent.name == "Project"


class TestSanitizeFolderName:
    """Tests for folder-name normalization."""

    def test_replaces_spaces_and_separators(self) -> None:
        assert sanitize_folder_name(" Project Name / Experiment:1 ") == "Project_Name_Experiment_1"


class TestNotebooksForPlate:
    """Tests for notebooks_for_plate using temp directory."""

    def test_returns_empty_when_dir_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "cellview.explore._registry.EXPLORE_DIR",
            tmp_path / "nonexistent",
        )
        assert notebooks_for_plate(12345) == []

    def test_finds_single_plate_notebook(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        (tmp_path / "plates" / "12345").mkdir(parents=True)
        (tmp_path / "plates" / "12345" / "explore_plate_12345.ipynb").touch()
        result = notebooks_for_plate(12345)
        assert result == ["plate_12345"]

    def test_finds_multi_plate_notebook(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        (tmp_path / "plates" / "12345_12378").mkdir(parents=True)
        (tmp_path / "plates" / "12345_12378" / "explore_plates_12345_12378.ipynb").touch()
        result = notebooks_for_plate(12345)
        assert result == ["plates_12345_12378"]

    def test_finds_multiple_notebooks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        (tmp_path / "plates" / "12345").mkdir(parents=True)
        (tmp_path / "plates" / "12345" / "explore_plate_12345.ipynb").touch()
        (tmp_path / "plates" / "12345_12378").mkdir(parents=True)
        (tmp_path / "plates" / "12345_12378" / "explore_plates_12345_12378.ipynb").touch()
        result = notebooks_for_plate(12345)
        assert len(result) == 2
        assert "plate_12345" in result
        assert "plates_12345_12378" in result

    def test_does_not_match_unrelated_plate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        (tmp_path / "plates" / "99999").mkdir(parents=True)
        (tmp_path / "plates" / "99999" / "explore_plate_99999.ipynb").touch()
        assert notebooks_for_plate(12345) == []


class TestExperimentNotebookExists:
    """Tests for experiment_notebook_exists."""

    def test_returns_false_when_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        assert experiment_notebook_exists(6) is False

    def test_returns_true_when_present_in_nested_folder(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("cellview.explore._registry.EXPLORE_DIR", tmp_path)
        path = notebook_path_for_experiment(6, folder_name="Project/Exp")
        path.parent.mkdir(parents=True)
        path.touch()
        assert experiment_notebook_exists(6) is True

    def test_legacy_path_helper_is_flat(self) -> None:
        path = legacy_notebook_path_for_experiment(6)
        assert path.name == "explore_exp_6.ipynb"
        assert path.parent == Path.home() / ".cellview" / "explore"
