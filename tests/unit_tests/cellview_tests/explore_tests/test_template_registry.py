"""Tests for the explore template registry."""

import json
from pathlib import Path

import pytest

from cellview.explore._template_registry import (
    BUILTIN_TEMPLATE_DIR,
    create_notebook_from_template,
    get_template,
    list_templates,
)


def _make_template(
    path: Path, plate_ids_line: str = "PLATE_IDS: list[int] = []"
) -> None:
    """Create a minimal .ipynb template file."""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Test Template"],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from cellview import cellview_load_data\n",
                    f"{plate_ids_line}\n",
                    "df, vars = cellview_load_data(*PLATE_IDS)\n",
                ],
                "outputs": [],
                "execution_count": None,
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f)


class TestListTemplates:
    """Tests for list_templates."""

    def test_finds_builtin_templates(self) -> None:
        templates = list_templates()
        names = [t.name for t in templates]
        assert "cellcycle" in names

    def test_builtin_has_description(self) -> None:
        templates = list_templates()
        cellcycle = next(t for t in templates if t.name == "cellcycle")
        assert cellcycle.description != ""
        assert cellcycle.source == "built-in"

    def test_user_template_overrides_builtin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "cellview.explore._template_registry.USER_TEMPLATE_DIR",
            tmp_path,
        )
        _make_template(tmp_path / "cellcycle.ipynb")
        templates = list_templates()
        cellcycle = next(t for t in templates if t.name == "cellcycle")
        assert cellcycle.source == "user"

    def test_discovers_user_templates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "cellview.explore._template_registry.USER_TEMPLATE_DIR",
            tmp_path,
        )
        _make_template(tmp_path / "my_custom.ipynb")
        templates = list_templates()
        names = [t.name for t in templates]
        assert "my_custom" in names


class TestGetTemplate:
    """Tests for get_template."""

    def test_finds_builtin(self) -> None:
        tmpl = get_template("cellcycle")
        assert tmpl is not None
        assert tmpl.source == "built-in"

    def test_returns_none_for_missing(self) -> None:
        assert get_template("nonexistent_template_xyz") is None

    def test_user_takes_priority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "cellview.explore._template_registry.USER_TEMPLATE_DIR",
            tmp_path,
        )
        _make_template(tmp_path / "cellcycle.ipynb")
        tmpl = get_template("cellcycle")
        assert tmpl is not None
        assert tmpl.source == "user"


class TestCreateNotebookFromTemplate:
    """Tests for create_notebook_from_template."""

    def test_injects_plate_ids(self, tmp_path: Path) -> None:
        template = tmp_path / "template.ipynb"
        output = tmp_path / "output.ipynb"
        _make_template(template)

        create_notebook_from_template(template, output, [12345, 12378])

        with open(output) as f:
            nb = json.load(f)

        code_cell = nb["cells"][1]
        source = "".join(code_cell["source"])
        assert "12345" in source
        assert "12378" in source

    def test_preserves_other_cells(self, tmp_path: Path) -> None:
        template = tmp_path / "template.ipynb"
        output = tmp_path / "output.ipynb"
        _make_template(template)

        create_notebook_from_template(template, output, [12345])

        with open(output) as f:
            nb = json.load(f)

        assert nb["cells"][0]["cell_type"] == "markdown"
        assert "Test Template" in "".join(nb["cells"][0]["source"])

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        template = tmp_path / "template.ipynb"
        output = tmp_path / "subdir" / "output.ipynb"
        _make_template(template)

        create_notebook_from_template(template, output, [12345])
        assert output.exists()

    def test_handles_string_source_format(self, tmp_path: Path) -> None:
        """Handles notebooks where source is a single string."""
        template = tmp_path / "template.ipynb"
        output = tmp_path / "output.ipynb"

        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": "PLATE_IDS = []\ndf = load(*PLATE_IDS)",
                    "outputs": [],
                    "execution_count": None,
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(template, "w") as f:
            json.dump(nb, f)

        create_notebook_from_template(template, output, [99999])

        with open(output) as f:
            result = json.load(f)
        assert "99999" in result["cells"][0]["source"]

    def test_copies_as_is_without_marker(self, tmp_path: Path) -> None:
        """Template without PLATE_IDS marker is just copied."""
        template = tmp_path / "template.ipynb"
        output = tmp_path / "output.ipynb"

        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": ["print('hello')"],
                    "outputs": [],
                    "execution_count": None,
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(template, "w") as f:
            json.dump(nb, f)

        create_notebook_from_template(template, output, [12345])
        assert output.exists()

    def test_sorts_plate_ids(self, tmp_path: Path) -> None:
        template = tmp_path / "template.ipynb"
        output = tmp_path / "output.ipynb"
        _make_template(template)

        create_notebook_from_template(template, output, [12378, 12345])

        with open(output) as f:
            nb = json.load(f)

        source = "".join(nb["cells"][1]["source"])
        assert "[12345, 12378]" in source


class TestBuiltinCellcycleTemplate:
    """Validate the shipped cellcycle.ipynb template."""

    def test_template_exists(self) -> None:
        assert (BUILTIN_TEMPLATE_DIR / "cellcycle.ipynb").exists()

    def test_template_is_valid_json(self) -> None:
        with open(BUILTIN_TEMPLATE_DIR / "cellcycle.ipynb") as f:
            nb = json.load(f)
        assert nb["nbformat"] == 4

    def test_template_has_plate_ids_marker(self) -> None:
        with open(BUILTIN_TEMPLATE_DIR / "cellcycle.ipynb") as f:
            nb = json.load(f)
        all_source = ""
        for cell in nb["cells"]:
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            all_source += source
        assert "PLATE_IDS" in all_source
