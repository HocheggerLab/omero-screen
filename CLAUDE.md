# OMERO-Screen

High-content IF microscopy monorepo: OMERO → Cellpose segmentation → feature extraction → cell-cycle analysis → DuckDB (CellView) → publication figures. `uv` workspace, Python 3.12, six packages.

GitHub: https://github.com/Helfrid/omero-screen · Docs: https://hocheggerlab.github.io/omero-screen/

## Memory lives in the Obsidian vault

Architecture and design detail are **not** in this file — they live in the vault (`/Users/hh65/Notes`), pulled on demand via the **obsidian-vault-manager** skill. A copy here only goes stale.

**Session start:** load the obsidian-vault-manager skill → run the MOC scan → anchor to [[@OmeroScreen]] and read [[@OmeroScreen_progresslog]] → read the in-scope package sub-MOC (table below), pulling linked notes via layered retrieval only as needed. Create a missing MOC/log before proceeding (MOCs use What/Why/How/When; no Bases embed unless asked).

**Logging (append-only, Did / Decided / Next):** satellite-package work → its `&Package_progresslog`; main-pipeline (core) and global / cross-package changes → [[@OmeroScreen_progresslog]]. Log decisions, commits, results, and session end; update the active MOC's `next_task` when it changes.

### Package → vault map

| Package | Path | Sub-MOC | Log to |
|---|---|---|---|
| omero-screen (core / main pipeline) | `src/omero_screen/` | [[&OmeroScreenCore]] | [[@OmeroScreen_progresslog]] |
| omero-utils | `packages/omero-utils/` | [[&OmeroUtils]] | [[&OmeroUtils_progresslog]] |
| cellview | `packages/cellview/` | [[&Cellview]] | [[&Cellview_progresslog]] |
| omero-screen-plots | `packages/omero-screen-plots/` | [[&OmeroScreenPlots]] | [[&OmeroScreenPlots_progresslog]] |
| omero-screen-napari | `packages/omero-screen-napari/` | [[&OmeroScreenNapari]] | [[&OmeroScreenNapari_progresslog]] |
| cellclass | `packages/cellclass/` | [[&CellClass]] | [[&CellClass_progresslog]] |

Live-cell tracking: [[&OmeroScreenTracking]].

## Operational quick reference

- **Tooling:** `uv` only, never pip — `uv sync --dev`, `uv run <cmd>`. `ruff check . && ruff format .` · `mypy .` · `pytest`. Conventional commits via `cz commit`; don't bypass pre-commit hooks.
- **Run:** `omero-screen <plate_id> [--env production] [--segmentation] [--inference model.pth] [--stitch] [--track [model]]` · `cellview display|import-csv|import-plate|export` · `pytest tests/unit_tests/<package>` · `omero-integration-test e2e_connection`.
- **Tracking (Trackastra):** `--track [model]` (default `general_2d`) relabels nuclei with a stable `track_id` across T; requires `--stitch` and a timelapse (T>1), no-op on single-timepoint. Adds `track_id`/`parent_track_id` (+`_raw`) to measurements → CellView. `--track-mode greedy|greedy_nodiv|ilp`.
- **Test server:** parallel OMERO at 127.0.0.2:4064 (root/omero) — `./scripts/manage_test_server.sh start|stop|status`.
- **Env:** `.env.{development,production,e2etest}` via `ENV` (default development). Required: OMERO `USERNAME`/`PASSWORD`/`HOST`, CellView `DATABASE_PATH`/`TEST_DATABASE`. Optional: `OMERO_SCREEN_CONFIG`, `OMERO_SCREEN_INFERENCE_MODEL`, `OMERO_SCREEN_CLEAR_BORDER`, `OMERO_SCREEN_TRACKING_MODEL`, `OMERO_SCREEN_TRACKING_MODE`.
- **Workspace members:** `.`, `packages/{omero-utils,omero-screen-napari,omero-screen-plots,cellview,cellclass}`.
