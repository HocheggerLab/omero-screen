"""Template discovery and notebook generation from .ipynb templates.

Templates are plain Jupyter notebooks stored in two locations:
1. Built-in: ``packages/cellview/src/cellview/explore/templates/``
2. User: ``~/.cellview/templates/``

User templates take priority over built-in templates with the same name.

Convention: templates should contain a code cell with ``PLATE_IDS = []``
which gets patched with the actual plate IDs at generation time.

DB integration
--------------
The ``templates`` DB table is the registration layer: it stores format,
description, and path for each template so the agentic skill can discover
templates without touching the filesystem.  The filesystem remains the
authoritative source for the actual file content.

Call :func:`sync_filesystem_to_db` to register all currently-visible
templates into the DB.  The DB-aware variants :func:`list_templates_from_db`
and :func:`get_template_from_db` use the DB as primary source, with the
filesystem as fallback for path validation.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BUILTIN_TEMPLATE_DIR = Path(__file__).parent / "templates"
USER_TEMPLATE_DIR = Path.home() / ".cellview" / "templates"

# Extensions we recognise as template files
_TEMPLATE_EXTENSIONS = ("*.ipynb", "*.py")

# Pattern to match PLATE_IDS assignment (with optional type hint)
_PLATE_IDS_PATTERN = re.compile(
    r"^(PLATE_IDS\s*(?::\s*list\[int\]\s*)?=\s*)(\[.*?\])", re.MULTILINE
)


@dataclass
class TemplateInfo:
    """Metadata about an available template.

    Attributes:
        name: Template name (filename without extension).
        path: Full path to the template file.
        source: ``"built-in"``, ``"user"``, or ``"db"`` (registered via CLI).
        description: First markdown cell / docstring content (truncated).
        fmt: ``"jupyter"`` for ``.ipynb`` files, ``"marimo"`` for ``.py`` files.
    """

    name: str
    path: Path
    source: str
    description: str
    fmt: str = "jupyter"


def list_templates() -> list[TemplateInfo]:
    """Discover all available templates from the filesystem (user overrides built-in).

    Scans both ``.ipynb`` and ``.py`` template files.  This function does
    **not** require a DB connection and is safe to call from any context.

    Returns:
        List of :class:`TemplateInfo` sorted by name.
    """
    templates: dict[str, TemplateInfo] = {}

    # Built-in first (will be overridden by user templates with same name)
    for directory, source in [
        (BUILTIN_TEMPLATE_DIR, "built-in"),
        (USER_TEMPLATE_DIR, "user"),
    ]:
        if not directory.exists():
            continue
        for pattern in _TEMPLATE_EXTENSIONS:
            for tmpl_path in sorted(directory.glob(pattern)):
                if tmpl_path.stem.startswith("_"):
                    continue  # skip __init__.py and private helpers
                name = tmpl_path.stem
                fmt = _fmt_from_path(tmpl_path)
                description = _extract_description(tmpl_path)
                templates[name] = TemplateInfo(
                    name=name,
                    path=tmpl_path,
                    source=source,
                    description=description,
                    fmt=fmt,
                )

    return sorted(templates.values(), key=lambda t: t.name)


def get_template(name: str) -> TemplateInfo | None:
    """Look up a template by name from the filesystem (user dir checked first).

    Checks both ``.ipynb`` and ``.py`` extensions.  Does **not** require a
    DB connection.

    Args:
        name: Template name (without extension).

    Returns:
        :class:`TemplateInfo` if found, ``None`` otherwise.
    """
    for directory, source in [
        (USER_TEMPLATE_DIR, "user"),
        (BUILTIN_TEMPLATE_DIR, "built-in"),
    ]:
        for ext, fmt in [(".ipynb", "jupyter"), (".py", "marimo")]:
            path = directory / f"{name}{ext}"
            if path.exists():
                return TemplateInfo(
                    name=name,
                    path=path,
                    source=source,
                    description=_extract_description(path),
                    fmt=fmt,
                )
    return None


# ---------------------------------------------------------------------------
# DB-aware functions
# ---------------------------------------------------------------------------


def sync_filesystem_to_db(conn: duckdb.DuckDBPyConnection) -> int:  # type: ignore[name-defined]  # noqa: F821
    """Register all filesystem-visible templates into the DB.

    Scans both built-in and user template directories and upserts each
    discovered template into the ``templates`` table.  Templates that are
    already registered are updated if their path or description changed.

    Args:
        conn: Active DuckDB connection.

    Returns:
        Number of templates registered or updated.
    """
    import duckdb

    from cellview.db.templates import upsert_template

    count = 0
    for tmpl in list_templates():
        try:
            upsert_template(
                conn,
                name=tmpl.name,
                path=tmpl.path,
                fmt=tmpl.fmt,
                description=tmpl.description or None,
            )
            count += 1
        except duckdb.Error:
            pass  # non-fatal; log if needed
    return count


def list_templates_from_db(
    conn: duckdb.DuckDBPyConnection,  # type: ignore[name-defined]  # noqa: F821
) -> list[TemplateInfo]:
    """Return all registered templates from the DB, validated against the filesystem.

    Templates whose file no longer exists on disk are included but marked with
    source ``"db-only"`` so callers can warn the user.

    Args:
        conn: Active DuckDB connection.

    Returns:
        List of :class:`TemplateInfo` sorted by name.
    """
    from cellview.db.templates import list_template_records

    records = list_template_records(conn)
    result: list[TemplateInfo] = []
    for rec in records:
        p = Path(rec.path)
        source = "db" if p.exists() else "db-only"
        result.append(
            TemplateInfo(
                name=rec.name,
                path=p,
                source=source,
                description=rec.description or "",
                fmt=rec.format,
            )
        )
    return result


def get_template_from_db(
    conn: duckdb.DuckDBPyConnection,  # type: ignore[name-defined]  # noqa: F821
    name: str,
) -> TemplateInfo | None:
    """Look up a template by name, preferring the DB record.

    Falls back to filesystem discovery if the name is not in the DB.

    Args:
        conn: Active DuckDB connection.
        name: Template name (without extension).

    Returns:
        :class:`TemplateInfo` if found, ``None`` otherwise.
    """
    from cellview.db.templates import get_template_record

    rec = get_template_record(conn, name)
    if rec is not None:
        p = Path(rec.path)
        return TemplateInfo(
            name=rec.name,
            path=p,
            source="db" if p.exists() else "db-only",
            description=rec.description or "",
            fmt=rec.format,
        )
    # DB miss — fall back to filesystem
    return get_template(name)


def create_notebook_from_template(
    template_path: Path,
    output_path: Path,
    plate_ids: list[int],
) -> None:
    """Copy a template and inject plate IDs.

    Supports both Jupyter (``.ipynb``) and Marimo (``.py``) templates.
    Finds a ``PLATE_IDS = [...]`` assignment and replaces the list with
    the actual plate IDs.

    Args:
        template_path: Path to the source template file.
        output_path: Path where the patched file will be written.
        plate_ids: Plate IDs to inject.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if template_path.suffix == ".py":
        _create_marimo_from_template(template_path, output_path, plate_ids)
    else:
        _create_jupyter_from_template(template_path, output_path, plate_ids)


def _create_jupyter_from_template(
    template_path: Path,
    output_path: Path,
    plate_ids: list[int],
) -> None:
    """Patch a ``.ipynb`` template and write to output_path."""
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
        shutil.copy2(template_path, output_path)
        return

    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)


def _create_marimo_from_template(
    template_path: Path,
    output_path: Path,
    plate_ids: list[int],
) -> None:
    """Patch a Marimo ``.py`` template and write to output_path.

    Replaces the ``PLATE_IDS = [...]`` assignment directly in the source
    text — no AST manipulation needed since the pattern is unambiguous.
    """
    source = template_path.read_text()
    ids_str = ", ".join(str(pid) for pid in sorted(plate_ids))
    replacement = rf"\g<1>[{ids_str}]"

    patched_source = _PLATE_IDS_PATTERN.sub(replacement, source)
    if patched_source == source:
        # No PLATE_IDS found — copy as-is
        shutil.copy2(template_path, output_path)
        return

    output_path.write_text(patched_source)


def init_user_template_dir() -> None:
    """Create the user template directory if it doesn't exist."""
    USER_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


def _fmt_from_path(path: Path) -> str:
    """Return ``"jupyter"`` for ``.ipynb`` files, ``"marimo"`` for ``.py``."""
    return "marimo" if path.suffix == ".py" else "jupyter"


def _extract_description(tmpl_path: Path) -> str:
    """Extract a short description from a template file.

    For ``.ipynb`` files: returns the first 80 chars of the first markdown cell.
    For ``.py`` files: returns the first 80 chars of the module docstring.

    Args:
        tmpl_path: Path to a template file (``.ipynb`` or ``.py``).

    Returns:
        Description string, or empty string if none found.
    """
    if tmpl_path.suffix == ".py":
        return _extract_py_description(tmpl_path)
    return _extract_nb_description(tmpl_path)


def _extract_nb_description(nb_path: Path) -> str:
    """Extract the first markdown cell from a Jupyter notebook.

    Args:
        nb_path: Path to a ``.ipynb`` file.

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
                text = source.strip().lstrip("#").strip()
                if text:
                    return str(text[:80])
    except (json.JSONDecodeError, KeyError):
        pass
    return ""


def _extract_py_description(py_path: Path) -> str:
    """Extract the module-level docstring from a Python template file.

    Args:
        py_path: Path to a ``.py`` file.

    Returns:
        First 80 chars of the module docstring, or empty string.
    """
    import ast

    try:
        tree = ast.parse(py_path.read_text())
        doc = ast.get_docstring(tree)
        if doc:
            return str(doc[:80])
    except (SyntaxError, OSError):
        pass
    return ""
