"""JSON context snapshot for the agentic explore workflow.

``explore_json()`` is the primary entry point.  It returns a structured dict
describing a plate (or experiment) in enough detail for an agent to write
correct pandas condition-prep code without having to query the DB itself.

Emitted structure::

    {
      "plates": [
        {
          "plate_id": 123,
          "date": "2025-03-01",
          "lab_member": "Alice",
          "channels": ["DAPI", "EdU", "Tub", null],
          "classifier": null
        }
      ],
      "experiment": {"experiment_id": 1, "name": "palb_washout"},
      "project":    {"project_id": 1,    "name": "BRCA2"},
      "schema": {
        "numeric":   ["area_nucleus", "intensity_mean_DAPI_nucleus", ...],
        "text":      ["cell_cycle", "cell_cycle_detailed", "label"],
        "condition": ["cell_line", "well", "antibody", "siRNA", "drug"]
      },
      "conditions": {
        "cell_line": ["RPE", "HeLa"],
        "antibody":  [null],
        "condition_variables": {
          "siRNA": ["ctrl", "PALB2"],
          "drug":  ["DMSO", "olaparib"]
        }
      },
      "stats": {
        "total_cells": 45000,
        "cells_per_condition": {
          "ctrl": {"RPE": 12000, "HeLa": 9000},
          "PALB2": {"RPE": 11500, "HeLa": 8700}
        },
        "cells_per_plate": {"123": 22000, "456": 23000}
      },
      "notebooks": [
        {"path": "~/.cellview/explore/plates/123/explore_plate_123.ipynb",
         "created": "2025-03-15T14:23:00"}
      ]
    }
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from cellview.explore._registry import (
    EXPLORE_DIR,
    notebooks_for_plate,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explore_json(
    conn: duckdb.DuckDBPyConnection,
    *,
    plate_ids: list[int] | None = None,
    experiment: str | int | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable context snapshot for one or more plates.

    Exactly one of ``plate_ids`` or ``experiment`` must be supplied.

    Args:
        conn: Active DuckDB connection.
        plate_ids: One or more plate IDs to describe.
        experiment: Experiment name (str) or ID (int); resolves to its plates.

    Returns:
        A nested dict ready for ``json.dumps()``.

    Raises:
        SystemExit: If the experiment or plates are not found in the DB.
    """
    resolved_plate_ids = _resolve_plate_ids(conn, plate_ids, experiment)

    plates_info = _plates_info(conn, resolved_plate_ids)
    exp_info, proj_info = _experiment_project_info(conn, resolved_plate_ids)
    schema = _schema_info(conn)
    conditions = _conditions_info(conn, resolved_plate_ids)
    stats = _stats_info(conn, resolved_plate_ids, conditions)
    notebooks = _notebooks_info(resolved_plate_ids)

    return {
        "plates": plates_info,
        "experiment": exp_info,
        "project": proj_info,
        "schema": schema,
        "conditions": conditions,
        "stats": stats,
        "notebooks": notebooks,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_plate_ids(
    conn: duckdb.DuckDBPyConnection,
    plate_ids: list[int] | None,
    experiment: str | int | None,
) -> list[int]:
    """Return a non-empty list of plate IDs or exit with an error message."""
    if experiment is not None:
        if isinstance(experiment, int):
            rows = conn.execute(
                "SELECT DISTINCT plate_id FROM repeats WHERE experiment_id = ? ORDER BY plate_id",
                [experiment],
            ).fetchall()
        else:
            row = conn.execute(
                "SELECT experiment_id FROM experiments WHERE experiment_name = ?",
                [experiment],
            ).fetchone()
            if row is None:
                print(
                    f"[cellview] Experiment '{experiment}' not found.",
                    file=sys.stderr,
                )
                sys.exit(1)
            rows = conn.execute(
                "SELECT DISTINCT plate_id FROM repeats WHERE experiment_id = ? ORDER BY plate_id",
                [row[0]],
            ).fetchall()

        ids = [r[0] for r in rows if r[0] is not None]
        if not ids:
            print(
                f"[cellview] No plates found for experiment '{experiment}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        return ids

    if plate_ids:
        return sorted(plate_ids)

    print("[cellview] Provide plate_ids or experiment.", file=sys.stderr)
    sys.exit(1)


def _plates_info(
    conn: duckdb.DuckDBPyConnection, plate_ids: list[int]
) -> list[dict[str, Any]]:
    """Return per-plate metadata rows."""
    placeholders = ", ".join("?" for _ in plate_ids)
    rows = conn.execute(
        f"""
        SELECT plate_id,
               date,
               lab_member,
               channel_0, channel_1, channel_2, channel_3,
               classifier
          FROM repeats
         WHERE plate_id IN ({placeholders})
         ORDER BY plate_id
        """,
        plate_ids,
    ).fetchall()

    result = []
    for r in rows:
        channels = [r[3], r[4], r[5], r[6]]
        # Trim trailing nulls but keep at least channel_0
        while len(channels) > 1 and channels[-1] is None:
            channels.pop()
        result.append(
            {
                "plate_id": r[0],
                "date": str(r[1]) if r[1] else None,
                "lab_member": r[2],
                "channels": channels,
                "classifier": r[7],
            }
        )
    return result


def _experiment_project_info(
    conn: duckdb.DuckDBPyConnection, plate_ids: list[int]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return experiment and project dicts, or None if plates span multiple."""
    placeholders = ", ".join("?" for _ in plate_ids)
    rows = conn.execute(
        f"""
        SELECT DISTINCT
               e.experiment_id, e.experiment_name,
               p.project_id,   p.project_name
          FROM repeats r
          LEFT JOIN experiments e ON e.experiment_id = r.experiment_id
          LEFT JOIN projects    p ON p.project_id    = e.project_id
         WHERE r.plate_id IN ({placeholders})
        """,
        plate_ids,
    ).fetchall()

    # Single experiment → report it; multiple → report None
    unique_exps = {(r[0], r[1]) for r in rows if r[0] is not None}
    unique_projs = {(r[2], r[3]) for r in rows if r[2] is not None}

    exp_info: dict[str, Any] | None = None
    if len(unique_exps) == 1:
        eid, ename = next(iter(unique_exps))
        exp_info = {"experiment_id": eid, "name": ename}

    proj_info: dict[str, Any] | None = None
    if len(unique_projs) == 1:
        pid, pname = next(iter(unique_projs))
        proj_info = {"project_id": pid, "name": pname}

    return exp_info, proj_info


def _schema_info(conn: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """Categorise measurement columns into numeric, text, and condition groups."""
    table_info = conn.execute("PRAGMA table_info(measurements)").fetchall()
    # columns: cid, name, type, notnull, dflt_value, pk
    numeric_types = {"FLOAT", "DOUBLE", "INTEGER", "BIGINT", "DECIMAL", "REAL"}
    text_types = {"TEXT", "VARCHAR"}

    numeric: list[str] = []
    text: list[str] = []
    skip = {"measurement_id", "condition_id"}

    for row in table_info:
        name = row[1]
        dtype = str(row[2]).upper().split("(")[0]
        if name in skip:
            continue
        if dtype in numeric_types:
            numeric.append(name)
        elif dtype in text_types:
            text.append(name)

    condition_cols = [
        "cell_line",
        "well",
        "antibody",
        "antibody_1",
        "antibody_2",
        "antibody_3",
    ]

    return {"numeric": numeric, "text": text, "condition": condition_cols}


def _conditions_info(
    conn: duckdb.DuckDBPyConnection, plate_ids: list[int]
) -> dict[str, Any]:
    """Return unique values for every condition dimension."""
    placeholders = ", ".join("?" for _ in plate_ids)

    # Fixed condition columns
    rows = conn.execute(
        f"""
        SELECT DISTINCT c.cell_line, c.antibody, c.antibody_1, c.antibody_2, c.antibody_3
          FROM repeats r
          JOIN conditions c ON r.repeat_id = c.repeat_id
         WHERE r.plate_id IN ({placeholders})
        """,
        plate_ids,
    ).fetchall()

    cell_lines: list[str | None] = sorted(
        {r[0] for r in rows}, key=lambda x: (x is None, x)
    )
    antibodies: list[str | None] = sorted(
        {r[1] for r in rows}, key=lambda x: (x is None, x)
    )

    # Condition variables (dynamic key-value pairs)
    var_rows = conn.execute(
        f"""
        SELECT DISTINCT cv.variable_name, cv.variable_value
          FROM repeats r
          JOIN conditions c  ON r.repeat_id  = c.repeat_id
          JOIN condition_variables cv ON c.condition_id = cv.condition_id
         WHERE r.plate_id IN ({placeholders})
         ORDER BY cv.variable_name, cv.variable_value
        """,
        plate_ids,
    ).fetchall()

    cond_vars: dict[str, list[str]] = {}
    for vname, vval in var_rows:
        cond_vars.setdefault(vname, []).append(vval)

    return {
        "cell_line": cell_lines,
        "antibody": antibodies,
        "condition_variables": cond_vars,
    }


def _stats_info(
    conn: duckdb.DuckDBPyConnection,
    plate_ids: list[int],
    conditions: dict[str, Any],
) -> dict[str, Any]:
    """Return total cell counts and breakdowns by condition and plate."""
    placeholders = ", ".join("?" for _ in plate_ids)

    # Total cells
    total_row = conn.execute(
        f"""
        SELECT COUNT(*)
          FROM repeats r
          JOIN conditions c ON r.repeat_id = c.repeat_id
          JOIN measurements m ON c.condition_id = m.condition_id
         WHERE r.plate_id IN ({placeholders})
        """,
        plate_ids,
    ).fetchone()
    total_cells = int(total_row[0]) if total_row else 0

    # Cells per plate
    plate_rows = conn.execute(
        f"""
        SELECT r.plate_id, COUNT(*) AS n
          FROM repeats r
          JOIN conditions c ON r.repeat_id = c.repeat_id
          JOIN measurements m ON c.condition_id = m.condition_id
         WHERE r.plate_id IN ({placeholders})
         GROUP BY r.plate_id
         ORDER BY r.plate_id
        """,
        plate_ids,
    ).fetchall()
    cells_per_plate = {str(r[0]): int(r[1]) for r in plate_rows}

    # Cells per condition variable × cell_line (only when variables exist)
    cells_per_condition: dict[str, Any] = {}
    cond_vars: dict[str, list[str]] = conditions.get("condition_variables", {})

    for var_name in cond_vars:
        breakdown_rows = conn.execute(
            f"""
            SELECT cv.variable_value, c.cell_line, COUNT(*) AS n
              FROM repeats r
              JOIN conditions c ON r.repeat_id = c.repeat_id
              JOIN measurements m ON c.condition_id = m.condition_id
              JOIN condition_variables cv ON c.condition_id = cv.condition_id
             WHERE r.plate_id IN ({placeholders})
               AND cv.variable_name = ?
             GROUP BY cv.variable_value, c.cell_line
             ORDER BY cv.variable_value, c.cell_line
            """,
            [*plate_ids, var_name],
        ).fetchall()
        var_counts: dict[str, dict[str, int]] = {}
        for val, cell_line, n in breakdown_rows:
            var_counts.setdefault(val, {})[cell_line] = int(n)
        cells_per_condition[var_name] = var_counts

    return {
        "total_cells": total_cells,
        "cells_per_plate": cells_per_plate,
        "cells_per_condition": cells_per_condition,
    }


def _notebooks_info(plate_ids: list[int]) -> list[dict[str, str | None]]:
    """Return existing explore notebooks that reference any of the plate IDs."""
    seen: set[Path] = set()
    result: list[dict[str, str | None]] = []

    for pid in plate_ids:
        for short_name in notebooks_for_plate(pid):
            # Reconstruct candidate paths from both flat and nested layouts
            candidates = [
                EXPLORE_DIR / f"explore_{short_name}.ipynb",
                EXPLORE_DIR
                / "plates"
                / short_name
                / f"explore_{short_name}.ipynb",
            ]
            for nb in candidates:
                if nb in seen:
                    continue
                if nb.exists():
                    seen.add(nb)
                    mtime = nb.stat().st_mtime
                    created = datetime.fromtimestamp(mtime).isoformat(
                        timespec="seconds"
                    )
                    result.append(
                        {
                            "path": str(nb).replace(str(Path.home()), "~"),
                            "created": created,
                        }
                    )

    return result
