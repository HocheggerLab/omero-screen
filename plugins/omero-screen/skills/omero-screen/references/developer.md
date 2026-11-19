# Developer checkout

**Only for a source checkout; never suggest any of this to an installed
user.** A checkout is a git clone of HocheggerLab/omero-screen run with
`uv run`. If `command -v omero-screen` points into `~/.local/bin` and there is
no clone, the user has the installed release: use the other references.

Full guide: https://hocheggerlab.github.io/omero-screen/ (Install & configure →
Developer install; Development).

---

## Set up

```bash
git clone https://github.com/HocheggerLab/omero-screen.git
cd omero-screen
uv sync --dev
uv run omero-screen setup
pre-commit install
```

uv only, never pip. `zeroc-ice` comes from the lab's wheel index, configured
in `pyproject.toml`. After a `git pull`, run `uv sync` (jobs submitted with
`uv run python bin/sbatch_omero_screen.py` use the checkout's `.venv` as is).

The workspace has the root package plus `packages/omero-utils`,
`omero-screen-napari`, `omero-screen-plots`, `cellview`, `cellclass` and
`omero-screen-track`. Add a dependency with `uv add [--package NAME] pkg`.

## `.env` files

`.env.development` (default), `.env.production` (`ENV=production` or
`--env production`) and `.env.e2etest` at the repository root override the
user config: `USERNAME`, `PASSWORD`, `HOST`, optionally `DATABASE_PATH`,
`OMERO_SCREEN_CACHE_PATH`. They hold the password in plain text; never print
one or paste it anywhere.

Give the checkout its own CellView database and image cache
(`DATABASE_PATH`, `OMERO_SCREEN_CACHE_PATH` in `.env.development`): the cache
is keyed by image ID, not server, so a test server and production can collide.

A release run inside the checkout folder picks up the checkout's `.env` files;
run the release from elsewhere.

## Quality and tests

```bash
uv run ruff check . && uv run ruff format .
uv run ty check                 # type checker (pinned); ignores as `# ty: ignore[rule]`
uv run pytest tests/unit_tests/<package>
```

Pre-commit runs ruff, ty and nbstripout; fix failures, never `--no-verify`.

## Commits and releases

Issue → branch → PR (`Closes #N`) → green CI → squash merge. Branch commits
are `dev(<scope>): …`. The PR title decides the release: `fix(…)` patch,
`feat(…)` minor, `docs`/`chore`/`ci`/`dev` none. The release workflow bumps
the version and tags it; never run `cz bump` by hand.

## Test OMERO server

A throwaway OMERO + Postgres in Docker on **127.0.0.2:4064** (`root`/`omero`),
separate from any other OMERO on 127.0.0.1.

```bash
sudo ifconfig lo0 alias 127.0.0.2          # macOS, once per boot
./scripts/manage_test_server.sh start|stop|status|restart
./scripts/load_plates.sh -d /path/to/plates -x
```

(`scripts/download_sample_data.py` fetches the omero-screen-plots example CSV,
not plates.)

`.env.e2etest` points at it (`HOST=127.0.0.2`, `USERNAME=root`,
`PASSWORD=omero`). Integration tests, with the server running and data
loaded:

```bash
uv run omero-integration-test e2e_connection
uv run omero-integration-test e2e_omero_screen     # full pipeline, several minutes
```

Watch out: `.env.e2etest` has no `DATABASE_PATH`, so CellView imports from it
go to the production CellView database; use a temporary one.

## Code-level pointers

- OMERO connections: the `@omero_connect` decorator (`omero_utils`), which
  passes `conn=`; it reads the login from the settings like the CLI.
- Stitch calibrations: `STITCH_CALIBRATIONS` in `omero_utils.stitching`;
  check a plate's seams with
  `uv run python scripts/diagnose_stitch_seams.py <plate_id>` (a peak at *d*
  px means `overlap` is short by *d*).
- HPC job submission from the checkout:
  `uv run python bin/sbatch_omero_screen.py <plate_id> …` (same options as
  `omero-screen-sbatch`).
