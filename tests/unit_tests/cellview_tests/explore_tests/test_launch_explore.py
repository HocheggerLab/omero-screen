"""Tests for launch-time explore behavior."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from cellview.explore import _cli


@pytest.fixture
def template_file(tmp_path: Path) -> Path:
    template = tmp_path / "template.ipynb"
    template.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
    return template


class TestLaunchExplore:
    """Tests for notebook creation, migration, and editor launching."""

    def test_migrates_legacy_notebook_before_opening(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        legacy_path = tmp_path / "explore_plate_12345.ipynb"
        target_path = tmp_path / "plates" / "12345" / "explore_plate_12345.ipynb"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text("legacy")

        _cli._migrate_legacy_notebook(legacy_path, target_path)

        assert not legacy_path.exists()
        assert target_path.exists()
        assert target_path.read_text() == "legacy"

    def test_code_flag_opens_parent_folder(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        template_file: Path,
    ) -> None:
        notebook_path = tmp_path / "Project" / "Experiment" / "explore_exp_6.ipynb"
        launches: list[list[str]] = []

        monkeypatch.setattr(
            _cli,
            "_resolve_target",
            lambda **_: _cli.ExploreTarget(
                notebook_path=notebook_path,
                legacy_path=tmp_path / "explore_exp_6.ipynb",
                plate_ids=[12345],
                label=6,
            ),
        )
        monkeypatch.setattr(
            _cli,
            "get_template",
            lambda name: SimpleNamespace(path=template_file, name=name, fmt="jupyter"),
        )
        monkeypatch.setattr(
            _cli.subprocess,
            "Popen",
            lambda args, **kwargs: launches.append(list(args)),
        )

        _cli.launch_explore(experiment=6, no_napari=True, code=True)

        assert notebook_path.exists()
        assert launches == [["code", str(notebook_path.parent)]]

    def test_jupyter_launches_notebook_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CELLVIEW_EDITOR", raising=False)
        notebook_path = tmp_path / "plates" / "12345" / "explore_plate_12345.ipynb"
        notebook_path.parent.mkdir(parents=True)
        notebook_path.write_text("existing")
        launches: list[list[str]] = []

        monkeypatch.setattr(
            _cli,
            "_resolve_target",
            lambda **_: _cli.ExploreTarget(
                notebook_path=notebook_path,
                legacy_path=tmp_path / "explore_plate_12345.ipynb",
                plate_ids=[12345],
                label="12345",
            ),
        )
        monkeypatch.setattr(
            _cli.subprocess,
            "Popen",
            lambda args, **kwargs: launches.append(list(args)),
        )

        _cli.launch_explore(plate_ids=[12345], no_napari=True)

        assert launches == [["jupyter", "lab", str(notebook_path)]]


class TestFolderResolution:
    """Tests for folder-name selection."""

    def test_plate_folder_falls_back_when_no_shared_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeConn:
            def execute(self, query: str, params: list[int]) -> "FakeConn":
                self.query = query
                self.params = params
                return self

            def fetchall(self) -> list[tuple[str | None, str | None]]:
                return [("Project A", "Exp 1"), ("Project B", "Exp 2")]

            def close(self) -> None:
                return None

        class FakeDB:
            def connect(self) -> FakeConn:
                return FakeConn()

        monkeypatch.setattr("cellview.db.db.CellViewDB", FakeDB)
        assert _cli._folder_name_for_plates([1, 2]) is None

    def test_plate_folder_prefers_shared_experiment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeConn:
            def execute(self, query: str, params: list[int]) -> "FakeConn":
                self.query = query
                self.params = params
                return self

            def fetchall(self) -> list[tuple[str | None, str | None]]:
                return [("Project A", "Exp 1")]

            def close(self) -> None:
                return None

        class FakeDB:
            def connect(self) -> FakeConn:
                return FakeConn()

        monkeypatch.setattr("cellview.db.db.CellViewDB", FakeDB)
        assert _cli._folder_name_for_plates([1, 2]) == "Project A/Exp 1"
