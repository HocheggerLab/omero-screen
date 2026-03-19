"""Template discovery and notebook generation from .ipynb templates.

Templates are plain Jupyter notebooks stored in two locations:
1. Built-in: ``packages/cellview/src/cellview/explore/templates/``
2. User: ``~/.cellview/templates/``

User templates take priority over built-in templates with the same name.

Convention: templates should contain a code cell with ``PLATE_IDS = []``
which gets patched with the actual plate IDs at generation time.
"""

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BUILTIN_TEMPLATE_DIR = Path(__file__).parent / "templates"
USER_TEMPLATE_DIR = Path.home() / ".cellview" / "templates"

# Pattern to match PLATE_IDS assignment (with optional type hint)
_PLATE_IDS_PATTERN = re.compile(
    r"^(PLATE_IDS\s*(?::\s*list\[int\]\s*)?=\s*)(\[.*?\])", re.MULTILINE
)


@dataclass
class TemplateInfo:
    """Metadata about an available template.

    Attributes:
        name: Template name (filename without .ipynb).
        path: Full path to the .ipynb file.
        source: Either "built-in" or "user".
        description: First markdown cell content (truncated), if available.
    """

    name: str
    path: Path
    source: str
    description: str


def list_templates() -> list[TemplateInfo]:
    """Discover all available templates (user overrides built-in).

    Returns:
        List of TemplateInfo sorted by name.
    """
    templates: dict[str, TemplateInfo] = {}

    # Built-in first (will be overridden by user templates with same name)
    for directory, source in [
        (BUILTIN_TEMPLATE_DIR, "built-in"),
        (USER_TEMPLATE_DIR, "user"),
    ]:
        if not directory.exists():
            continue
        for nb_path in sorted(directory.glob("*.ipynb")):
            name = nb_path.stem
            description = _extract_description(nb_path)
            templates[name] = TemplateInfo(
                name=name,
                path=nb_path,
                source=source,
                description=description,
            )

    return sorted(templates.values(), key=lambda t: t.name)


def get_template(name: str) -> TemplateInfo | None:
    """Look up a template by name (user dir checked first).

    Args:
        name: Template name (without .ipynb extension).

    Returns:
        TemplateInfo if found, None otherwise.
    """
    # User templates take priority
    for directory, source in [
        (USER_TEMPLATE_DIR, "user"),
        (BUILTIN_TEMPLATE_DIR, "built-in"),
    ]:
        path = directory / f"{name}.ipynb"
        if path.exists():
            return TemplateInfo(
                name=name,
                path=path,
                source=source,
                description=_extract_description(path),
            )
    return None


def create_notebook_from_template(
    template_path: Path,
    output_path: Path,
    plate_ids: list[int],
) -> None:
    """Copy a template notebook and inject plate IDs.

    Finds a cell containing ``PLATE_IDS = [...]`` and replaces the
    list with the actual plate IDs.

    Args:
        template_path: Path to the source template .ipynb.
        output_path: Path where the patched notebook will be written.
        plate_ids: Plate IDs to inject.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(template_path) as f:
        nb = json.load(f)

    ids_str = ", ".join(str(pid) for pid in sorted(plate_ids))
    replacement = rf"\g<1>[{ids_str}]"

    patched = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        # Handle both string and list-of-lines formats
        if isinstance(source, list):
            joined = "".join(source)
            if _PLATE_IDS_PATTERN.search(joined):
                new_source = _PLATE_IDS_PATTERN.sub(replacement, joined)
                cell["source"] = new_source.splitlines(keepends=True)
                patched = True
                break
        else:
            if _PLATE_IDS_PATTERN.search(source):
                cell["source"] = _PLATE_IDS_PATTERN.sub(replacement, source)
                patched = True
                break

    if not patched:
        # No PLATE_IDS cell found — just copy as-is
        shutil.copy2(template_path, output_path)
        return

    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)


def init_user_template_dir() -> None:
    """Create the user template directory if it doesn't exist."""
    USER_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


def _extract_description(nb_path: Path) -> str:
    """Extract the first markdown cell as a description.

    Args:
        nb_path: Path to a .ipynb file.

    Returns:
        First 80 chars of the first markdown cell, or empty string.
    """
    try:
        with open(nb_path) as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                # Strip markdown heading markers
                text = source.strip().lstrip("#").strip()
                if text:
                    return str(text[:80])
    except (json.JSONDecodeError, KeyError):
        pass
    return ""
