---
description: Build and verify the documentation for omero-screen-plots
---

# Build and Verify Documentation

This workflow builds the Sphinx documentation for `omero-screen-plots` and verifies that the build was successful.

## Prerequisites
- The `docs` dependency group must be installed.
- `scipy` must be in `packages/omero-screen-plots/docs/requirements.txt`.
- `omero_screen` must be mocked in `conf.py` (or installed).

## Steps

1.  **Install Dependencies**
    Ensure the documentation dependencies are installed.
    ```bash
    uv sync --group docs
    ```

2.  **Clean Build Directory**
    Remove previous builds to ensure a clean state.
    ```bash
    rm -rf packages/omero-screen-plots/docs/_build
    ```

3.  **Build Documentation**
    Run the Sphinx build command using `uv` to ensure the correct environment.
    **Note:** We set `BUILD_PLOTS=true` to ensure plots are regenerated.
    ```bash
    BUILD_PLOTS=true uv run --group docs make -C packages/omero-screen-plots/docs html
    ```

4.  **Verify Build**
    Check if the index file exists.
    ```bash
    ls -l packages/omero-screen-plots/docs/_build/html/index.html
    ```

5.  **Serve Documentation (Optional)**
    To preview the documentation, you can run a python server.
    ```bash
    # python3 -m http.server --directory packages/omero-screen-plots/docs/_build/html
    ```
