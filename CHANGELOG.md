## omero-screen-v0.8.19 (2026-10-08)

### Fix

- **cellview**: do not import the stitch_mode marker as a condition variable (#61)

## omero-screen-v0.8.18 (2026-10-08)

### Fix

- **pipeline**: record versions, settings and models on each processed plate (#60)

## omero-screen-v0.8.17 (2026-10-08)

### Fix

- **cli**: --version for omero-screen and cellclass (#59)
- **install**: find the latest version from release tags (#58)

## omero-screen-v0.8.16 (2026-10-08)

### Fix

- **install**: one-line installer for test users (#54)

## omero-screen-v0.8.15 (2026-10-08)

### Fix

- **segmentation**: device-aware model defaults, opt-in lab models, models pull/publish (#51)

## omero-screen-v0.8.14 (2026-10-08)

### Fix

- **config**: user config with site profiles, keychain password, setup/doctor/config show (#50)

## omero-screen-v0.8.13 (2026-10-08)

### Fix

- remove scratch.py and ignore scratch files (#55)

## omero-screen-v0.8.12 (2026-10-07)

### Fix

- **napari**: hide debris, agent-alone runs and proposal confidence (#47)

## omero-screen-v0.8.11 (2026-10-07)

### Fix

- **cellview**: warn when stationary cells still express a reporter (#46)

## omero-screen-v0.8.10 (2026-10-06)

### Fix

- **napari**: mask edits with re-measurement, and absorb/drop (#45)

## omero-screen-v0.8.9 (2026-10-06)

### Fix

- **napari**: agent review tools through the napari-mcp plugin hook (#44)

## omero-screen-v0.8.8 (2026-10-06)

### Fix

- **napari**: review queue sampling and Track Review v2 (#43)
- **napari**: follow tracked cells through time and render filmstrips (#42)

## omero-screen-v0.8.7 (2026-10-06)

### Fix

- **cellview**: FUCCI-gated track repair, per-cell fate walker and Track Review widget (#39)

## omero-screen-v0.8.6 (2026-10-05)

### Fix

- **napari**: add omero-screen-images batch command, docs and e2e test (#35)

## omero-screen-v0.8.5 (2026-10-05)

### Fix

- **napari**: add omero-screen-images well command for whole-well overviews (#34)

## omero-screen-v0.8.4 (2026-10-05)

### Fix

- **napari**: add omero-screen-images gallery command for headless galleries (#33)

## omero-screen-v0.8.3 (2026-10-03)

### Fix

- **napari**: survive transient OMERO errors during zarr cache builds (#24)

## omero-screen-v0.8.2 (2026-10-02)

### Fix

- **stitch**: select stitch calibration by objective (pixel size) (#25)

## omero-screen-v0.8.1 (2026-10-02)

### Fix

- **export**: reuse the previous timestamp when Operetta left none (#23)

## omero-screen-v0.8.0 (2026-10-02)

### Feat

- **deps**: support Python 3.12-3.14, napari 0.9 and PyQt6 (#14)

## omero-screen-v0.7.2 (2026-09-28)

### Fix

- **pipeline**: link nuclei to cells per frame and keep tracked nuclei
- **napari**: drop lineage edges to parent tracks with no rows

## omero-screen-v0.7.1 (2026-09-01)

### Fix

- **ci**: keep uv.lock in step with the version bump

## omero-screen-v0.7.0 (2026-08-31)

### Feat

- **cellclass**: unify command-line interface with Click

### Fix

- **ci**: stop the two doc workflows cancelling each other

## omero-screen-v0.6.0 (2026-08-26)

### Feat

- **plots**: make the per-cell scatter marker size configurable
- **plots**: one colour-matched star row per annotated class
- **plots**: report effect size and gate significance markers on it

## omero-screen-v0.5.0 (2026-07-21)

## omero-screen-v0.4.0 (2026-06-16)

### Feat

- auto-manage tracking VRAM and stitch host-RAM (fewer flags)
- **loops**: opt-in per-timepoint streaming stitch to bound host RAM
- **loops**: per-timepoint progress logs in stitched segmentation
- **tracking**: add device + window controls for GPU-OOM-safe tracking
- **tracking**: configurable batch size to cap GPU memory
- **tracking**: forward --track/--track-mode through SLURM submission
- **tracking**: add Trackastra nucleus tracking (Stage 1 + 2)

### Fix

- **cellcycle**: classify H3P-positive cells as M before DNA/EdU gates
- **napari**: align Tracks layer with OME-Zarr physical-unit scale
- **napari**: try swapped centroid-0/centroid-1 axis order in Tracks loader
- **napari**: default Tracks widget well to empty + auto-fill from loaded data
- **cellview**: exempt track columns from classifier auto-rename
- **cellview**: persist Trackastra track columns through import
- **tracking**: drop from __future__ import annotations in tracks widget

### Perf

- **loops**: cast flatfield-corrected fields to float32

## omero-screen-v0.3.5 (2026-05-27)

### Refactor

- **omero-screen-napari**: make TrainingDataSaver derive paths lazily

## omero-screen-v0.3.4 (2026-04-15)

### Refactor

- **cellview**: new data structure for notebooks

## omero-screen-v0.3.3 (2026-02-25)

### Fix

- **omero-screen-napari**: add Blosc compression to plate cache (~2× size reduction)

## omero-screen-v0.3.2 (2026-02-23)

### Fix

- **omero-screen-napari**: adapt tests to new training data handling and test new functions

## omero-screen-v0.3.1 (2026-02-11)

### Fix

- **omero-screen-napari**: adapt tests to new training data handling and test new functions

## omero-screen-v0.3.0 (2026-02-06)

### Feat

- Add support for cellpose 3 and 4 models

## omero-screen-v0.2.6 (2026-01-12)

### Fix

- training-db with cli and links to plugins finalised and tests running
- training-db with cli and links to plugins finalised

## omero-screen-v0.2.5 (2025-12-21)

### Fix

- synchronize package versions and simplify release workflow

## omero-screen-v0.2.4 (2025-12-10)

### Fix

- default model for unknow cell line, loops.py line 89

## omero-screen-v0.2.3 (2025-10-08)

### Fix

- various

## omero-screen-v0.2.2 (2025-09-16)

### Fix

- pathfinding for .env file in set_env_vars function config.py

## omero-screen-v0.2.1 (2025-09-16)

### Fix

- **cellview**: adapt release CI and bump cellview version
- **cellview**: enhance OMERO import

## omero-screen-v0.2.0 (2025-09-15)

### Fix

- resolve remaining conflict markers in pyproject.toml
- **omero-screen-plots**: corrected release ci and upgrade version to 0.1.1
- **omero-screen-plots**: corrected release ci and upgrade version to 0.1.1
- **omero-screen-plots**: finalised package with new architecture
- bug with detecting repeats in prop_pivot cellcycleplots.py fixed with nunique check

### Refactor

- cellcycle plot factory with comprehensive features
- migrate from singleton pattern to dependency injection

## omero-screen-v0.2.0 (2025-09-10)

### Fix

- bug with detecting repeats in prop_pivot cellcycleplots.py fixed with nunique check

### Refactor

- cellcycle plot factory with comprehensive features
- migrate from singleton pattern to dependency injection

## omero-screen-v0.1.5 (2025-05-19)

### Fix

- **cellview**: merge cellview branch to main

## omero-screen-v0.1.4 (2025-05-13)

### Fix

- address mypy errors

## omero-screen-v0.1.3 (2025-02-17)

### Fix

- finalise testsetup with session scope connection and plates
- setup unit tests with omero-server with plate setup in test_metadata
- setup unit tests with omero-server
- setup unit tests with omero-server

## omero-screen-v0.1.2 (2025-01-06)

### Fix

- finalise omero-connect update README.md
- **omero-utils**: set up and test omero_connect decorator

## omero-screen-v0.1.1 (2024-12-12)

### Fix

- **omero-screen**: finalise env logging and CI setup attempt 1
- **omero-screen**: add omero-connect and tests
- **omero-screen**: set env variables in ci.yml
- **omero-screen**: fourth correction of pyproject.toml
- **omero-screen**: third correction of pyproject.toml
- **omero-screen**: 2nd correction of pyproject.toml
- **omero-screen**: update linux compatibility for zero-ice in pyproject.toml
- **omero-screen**: set up logger and env variables)
