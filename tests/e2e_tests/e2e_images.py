"""End-to-end test of ``omero-screen-images`` on a per-field test plate.

Runs OMERO Screen on the plate (segmentation masks and measurements),
imports the measurements into a throwaway CellView database, then renders
galleries, whole-well overviews and a batch through the console script, and
checks every requested image was written. This is the path a stitched plate
with a zarr cache never takes: per-field images and masks loaded through the
plate disk cache and stitched in memory.

The CellView database lives in a temporary directory. The e2etest
environment sets no ``DATABASE_PATH``, so without one CellView would fall
back to the default (production) database.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from omero.gateway import BlitzGateway

from tests.e2e_tests.e2e_omero_screen import (
    clean_omero_screen_run,
    run_omero_screen_test,
)


def run_images_test(
    conn: BlitzGateway,
    teardown: bool = True,
    plate_id: int = 1,
    tub: bool = True,
) -> None:
    """Render galleries, overviews and a batch for a processed per-field plate."""
    from omero_screen_napari.plate_cache import delete_plate_from_cache

    # A cached label map from an earlier run would point at deleted masks.
    delete_plate_from_cache(plate_id, remove_images=False)
    run_omero_screen_test(conn, teardown=False, plate_id=plate_id, tub=tub)
    try:
        with tempfile.TemporaryDirectory(prefix="e2e_images_") as tmp:
            work = Path(tmp)
            env = {
                **os.environ,
                "ENV": "e2etest",
                "DATABASE_PATH": str(work / "cellview.duckdb"),
            }
            experiment_id = _make_experiment(work / "cellview.duckdb")
            _run(
                env,
                "cellview",
                "import",
                "plate",
                str(plate_id),
                "--experiment",
                str(experiment_id),
            )
            wells = _wells(conn, plate_id)
            print(f"Rendering wells {wells} of plate {plate_id}")

            gallery = _images(
                env,
                "gallery",
                str(plate_id),
                "--wells",
                "All",
                "--channels",
                "DAPI",
                "--grid",
                "3x3",
                "--fmt",
                "png",
                "--out",
                str(work / "gallery"),
            )
            _check(gallery, wells, "gallery")
            assert gallery["source"] == "fields", gallery["source"]

            overview = _images(
                env,
                "well",
                str(plate_id),
                "--wells",
                "All",
                "--layers",
                "DAPI,nuclei_masks",
                "--out",
                str(work / "well"),
            )
            _check(overview, wells, "well")
            for well, entry in overview["wells"].items():
                assert not entry["missing_masks"], (well, entry)

            plan = work / "plan.csv"
            plan.write_text(
                "plate_id,well,render\n"
                + "".join(
                    f"{plate_id},{w},{r}\n"
                    for w in wells
                    for r in ("well", "gallery")
                )
            )
            batch = _images(
                env,
                "batch",
                str(plan),
                "--channels",
                "DAPI",
                "--out",
                str(work / "batch"),
            )
            assert all(not run.get("error") for run in batch["runs"]), batch
            assert sum(run["written"] for run in batch["runs"]) == 2 * len(
                wells
            )
            print("omero-screen-images e2e: all images written")
    finally:
        if teardown:
            clean_omero_screen_run(conn, plate_id)
            delete_plate_from_cache(plate_id, remove_images=False)


def _make_experiment(db_path: Path) -> int:
    """Create a project and experiment to import the test plate into."""
    from cellview.db.db import CellViewDB

    db = CellViewDB(db_path)
    conn = db.connect()
    conn.execute("INSERT INTO projects (project_name) VALUES ('e2e')")
    conn.execute(
        "INSERT INTO experiments (project_id, experiment_name) "
        "SELECT project_id, 'images' FROM projects WHERE project_name = 'e2e'"
    )
    row = conn.execute("SELECT max(experiment_id) FROM experiments").fetchone()
    conn.close()
    assert row is not None
    return int(row[0])


def _wells(conn: BlitzGateway, plate_id: int) -> list[str]:
    plate = conn.getObject("Plate", plate_id)
    assert plate is not None, f"Plate {plate_id} not found"
    return sorted(well.getWellPos() for well in plate.listChildren())


def _run(env: dict[str, str], command: str, *args: str) -> str:
    """Run a console script from this environment; fail with its output."""
    exe = Path(sys.executable).parent / command
    result = subprocess.run(
        [str(exe), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{command} {' '.join(args)} failed ({result.returncode}):\n"
            f"{result.stderr[-3000:]}"
        )
    return result.stdout


def _images(env: dict[str, str], *args: str) -> dict:
    return json.loads(_run(env, "omero-screen-images", *args, "--json"))


def _check(manifest: dict, wells: list[str], what: str) -> None:
    exported = {
        well: entry.get("exported")
        for well, entry in manifest["wells"].items()
    }
    assert exported == dict.fromkeys(wells, True), (
        f"{what}: {manifest['wells']}"
    )
