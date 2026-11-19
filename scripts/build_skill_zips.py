"""Zip each plugin skill for upload into the Claude desktop app.

The desktop app does not read plugin marketplaces; users add a skill by
uploading a ``.zip`` whose top level is the skill folder, with ``SKILL.md``
inside it. The docs build calls this to publish one zip per skill under
``downloads/`` on the documentation site.

Usage:
    python scripts/build_skill_zips.py <out_dir> [--skills-dir DIR]
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "plugins" / "omero-screen" / "skills"

#: Never shipped: caches and editor or OS clutter.
EXCLUDED_NAMES = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}

#: A fixed timestamp, so the same skill always gives the same zip.
ZIP_DATE = (2020, 1, 1, 0, 0, 0)


class SkillZipError(RuntimeError):
    """A skill folder cannot be zipped."""


def _shipped(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return not any(p in EXCLUDED_NAMES for p in parts) and (
        path.suffix not in EXCLUDED_SUFFIXES
    )


def zip_skill(skill_dir: Path, out_dir: Path) -> Path:
    """Write ``<out_dir>/<skill>.zip`` with ``<skill>/SKILL.md`` at its top.

    Raises:
        SkillZipError: If the folder has no ``SKILL.md``.
    """
    if not (skill_dir / "SKILL.md").is_file():
        raise SkillZipError(f"{skill_dir} has no SKILL.md")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{skill_dir.name}.zip"
    files = sorted(
        p
        for p in skill_dir.rglob("*")
        if p.is_file() and _shipped(p, skill_dir)
    )
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = (
                f"{skill_dir.name}/{path.relative_to(skill_dir).as_posix()}"
            )
            info = zipfile.ZipInfo(arcname, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return target


def build_all(out_dir: Path, skills_dir: Path = SKILLS_DIR) -> list[Path]:
    """Zip every skill folder (a folder holding a ``SKILL.md``)."""
    skills = sorted(
        d for d in skills_dir.iterdir() if (d / "SKILL.md").is_file()
    )
    if not skills:
        raise SkillZipError(f"No skills found in {skills_dir}")
    return [zip_skill(d, out_dir) for d in skills]


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="Where to write the zips")
    parser.add_argument(
        "--skills-dir", type=Path, default=SKILLS_DIR, help="Skills folder"
    )
    args = parser.parse_args()
    for path in build_all(args.out_dir, args.skills_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
