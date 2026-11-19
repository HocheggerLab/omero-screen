"""``scripts/build_skill_zips.py``: one desktop-app zip per skill."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "build_skill_zips", REPO_ROOT / "scripts" / "build_skill_zips.py"
)
assert _spec and _spec.loader
build_skill_zips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_skill_zips)


def _skill(root: Path, name: str) -> Path:
    skill = root / name
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    (skill / "references" / "a.md").write_text("ref")
    (skill / "__pycache__").mkdir()
    (skill / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    (skill / ".DS_Store").write_bytes(b"\0")
    return skill


def test_zip_has_the_skill_folder_at_its_top(tmp_path: Path) -> None:
    skill = _skill(tmp_path / "skills", "demo")
    target = build_skill_zips.zip_skill(skill, tmp_path / "out")
    with zipfile.ZipFile(target) as zf:
        names = sorted(zf.namelist())
    assert names == ["demo/SKILL.md", "demo/references/a.md"]


def test_zips_are_reproducible(tmp_path: Path) -> None:
    skill = _skill(tmp_path / "skills", "demo")
    first = build_skill_zips.zip_skill(skill, tmp_path / "a").read_bytes()
    second = build_skill_zips.zip_skill(skill, tmp_path / "b").read_bytes()
    assert first == second


def test_build_all_skips_folders_without_a_skill(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    _skill(skills, "one")
    _skill(skills, "two")
    (skills / "notes").mkdir()
    written = build_skill_zips.build_all(tmp_path / "out", skills)
    assert [p.name for p in written] == ["one.zip", "two.zip"]


def test_a_folder_without_skill_md_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(build_skill_zips.SkillZipError):
        build_skill_zips.zip_skill(tmp_path / "empty", tmp_path / "out")


def test_the_real_plugin_skills_zip(tmp_path: Path) -> None:
    written = build_skill_zips.build_all(tmp_path)
    names = {p.name for p in written}
    assert {"omero-screen.zip", "cellclass-train.zip", "track-review.zip"} <= names
    with zipfile.ZipFile(tmp_path / "omero-screen.zip") as zf:
        assert "omero-screen/SKILL.md" in zf.namelist()
