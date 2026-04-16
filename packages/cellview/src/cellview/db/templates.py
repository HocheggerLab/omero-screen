"""CRUD operations for the templates table.

The templates table is the DB-backed registry of analysis notebook templates.
Each row points to a file on disk (built-in or user-managed) and carries
enough metadata for the agentic skill to select the right template.

Template formats:
- ``"jupyter"`` — ``.ipynb`` notebook, opened with JupyterLab or VS Code.
- ``"marimo"``  — Marimo ``.py`` app, opened with ``marimo edit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass
class TemplateRecord:
    """A row from the templates table.

    Attributes:
        template_id: Auto-assigned primary key.
        name: Unique short name (e.g. ``"cellcycle"``, ``"sirna"``).
        description: Human-readable summary shown in listings.
        format: Either ``"jupyter"`` or ``"marimo"``.
        path: Absolute path to the template file on disk.
        parent_template_id: FK to another template this was derived from, or None.
    """

    template_id: int
    name: str
    description: str | None
    format: str
    path: str
    parent_template_id: int | None


def upsert_template(
    conn: duckdb.DuckDBPyConnection,
    *,
    name: str,
    path: Path | str,
    fmt: str = "jupyter",
    description: str | None = None,
    parent_template_id: int | None = None,
) -> int:
    """Insert or update a template record.

    If a template with ``name`` already exists its ``path``, ``fmt``,
    ``description``, and ``updated_at`` are refreshed; other fields are left
    unchanged.

    Args:
        conn: Active DuckDB connection.
        name: Unique template name.
        path: Absolute path to the template file.
        fmt: ``"jupyter"`` or ``"marimo"``.
        description: Optional description shown in listings.
        parent_template_id: Optional FK to a parent template.

    Returns:
        The ``template_id`` of the inserted or updated row.
    """
    if fmt not in ("jupyter", "marimo"):
        raise ValueError(f"fmt must be 'jupyter' or 'marimo', got {fmt!r}")

    existing = conn.execute(
        "SELECT template_id FROM templates WHERE name = ?", [name]
    ).fetchone()

    if existing:
        conn.execute(
            """
            UPDATE templates
               SET path = ?,
                   format = ?,
                   description = ?,
                   updated_at = now()
             WHERE name = ?
            """,
            [str(path), fmt, description, name],
        )
        return int(existing[0])

    row = conn.execute(
        """
        INSERT INTO templates (name, description, format, path, parent_template_id)
        VALUES (?, ?, ?, ?, ?)
        RETURNING template_id
        """,
        [name, description, fmt, str(path), parent_template_id],
    ).fetchone()
    assert row is not None
    return int(row[0])


def get_template_record(
    conn: duckdb.DuckDBPyConnection, name: str
) -> TemplateRecord | None:
    """Fetch a single template by name.

    Args:
        conn: Active DuckDB connection.
        name: Template name to look up.

    Returns:
        A :class:`TemplateRecord`, or ``None`` if not found.
    """
    row = conn.execute(
        """
        SELECT template_id, name, description, format, path, parent_template_id
          FROM templates
         WHERE name = ?
        """,
        [name],
    ).fetchone()
    if row is None:
        return None
    return TemplateRecord(
        template_id=row[0],
        name=row[1],
        description=row[2],
        format=row[3],
        path=row[4],
        parent_template_id=row[5],
    )


def list_template_records(
    conn: duckdb.DuckDBPyConnection,
) -> list[TemplateRecord]:
    """Return all registered templates ordered by name.

    Args:
        conn: Active DuckDB connection.

    Returns:
        List of :class:`TemplateRecord` instances.
    """
    rows = conn.execute(
        """
        SELECT template_id, name, description, format, path, parent_template_id
          FROM templates
         ORDER BY name
        """
    ).fetchall()
    return [
        TemplateRecord(
            template_id=r[0],
            name=r[1],
            description=r[2],
            format=r[3],
            path=r[4],
            parent_template_id=r[5],
        )
        for r in rows
    ]


def delete_template(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    """Remove a template record by name.

    This does **not** delete the file from disk.

    Args:
        conn: Active DuckDB connection.
        name: Template name to remove.

    Returns:
        ``True`` if a row was deleted, ``False`` if name was not found.
    """
    before = conn.execute(
        "SELECT COUNT(*) FROM templates WHERE name = ?", [name]
    ).fetchone()
    if not before or before[0] == 0:
        return False
    conn.execute("DELETE FROM templates WHERE name = ?", [name])
    return True
