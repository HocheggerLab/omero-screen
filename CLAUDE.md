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
- **Run:** `omero-screen <plate_id> [--env production] [--segmentation] [--inference model.pth] [--stitch] [--stream-stitch] [--track [model]]` · `cellview display|import-csv|import-plate|export` · `pytest tests/unit_tests/<package>` · `omero-integration-test e2e_connection`.
- **Stitched-well host RAM:** flatfield-corrected fields are float32 (not float64) — halves the canvas. For long multi-channel timelapses that still OOM during load, `--stream-stitch` (`OMERO_SCREEN_STITCH_STREAMING`) stitches one timepoint at a time so peak ≈ canvas + one frame's fields (vs all fields + canvas), at the cost of `n_fields × T` OMERO reads. Requires `--stitch`. (Feature extraction still holds the full canvas — see [[&OmeroScreenTracking]] for the remaining Peak-B work.)
- **Tracking (Trackastra):** `--track [model]` (default `general_2d`) relabels nuclei with a stable `track_id` across T; requires `--stitch` and a timelapse (T>1), no-op on single-timepoint. Adds `track_id`/`parent_track_id` (+`_raw`) to measurements → CellView. `--track-mode greedy|greedy_nodiv|ilp`. GPU-memory controls for dense stitched wells (Trackastra's attention builds a dense `(heads, N, N)` spatial-bias matrix, `N ≈ window × detections_per_frame`, so memory is O(N²) and can exceed A40 VRAM regardless of batch size): `--track-device cpu` runs the identical computation in host RAM (slower, no VRAM ceiling, no accuracy loss — the accuracy-first option); `--track-window N` shrinks the temporal window (memory ~quadratic, trades temporal context); `--track-batch-size N` (default 4) only helps when there are many windows (long timelapses). A diagnostic log line reports frames / objects-per-frame / effective window / detections-per-window.
- **Test server:** parallel OMERO at 127.0.0.2:4064 (root/omero) — `./scripts/manage_test_server.sh start|stop|status`.
- **Env:** `.env.{development,production,e2etest}` via `ENV` (default development). Required: OMERO `USERNAME`/`PASSWORD`/`HOST`, CellView `DATABASE_PATH`/`TEST_DATABASE`. Optional: `OMERO_SCREEN_CONFIG`, `OMERO_SCREEN_INFERENCE_MODEL`, `OMERO_SCREEN_CLEAR_BORDER`, `OMERO_SCREEN_TRACKING_MODEL`, `OMERO_SCREEN_TRACKING_MODE`, `OMERO_SCREEN_TRACKING_BATCH_SIZE`, `OMERO_SCREEN_TRACKING_DEVICE`, `OMERO_SCREEN_TRACKING_WINDOW`, `OMERO_SCREEN_STITCH_STREAMING`.
- **Workspace members:** `.`, `packages/{omero-utils,omero-screen-napari,omero-screen-plots,cellview,cellclass}`.
