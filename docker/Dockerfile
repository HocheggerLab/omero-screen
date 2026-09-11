# Headless omero-screen runtime.
#
# Two build variants, selected by TORCH_EXTRA:
#   cpu    -> torch 2.9.1+cpu, no CUDA payload (linux/arm64 and amd64)
#   cu128  -> torch 2.9.1+cu128, needs `--gpus all` and a host driver
#             supporting CUDA 12.8
#
# Neither installs napari: `--no-default-groups` drops the `gui` group, which is
# the whole point — napari[all] pulls pyqt5, pyqt5-qt5 and triangle, none of
# which ship linux/arm64 wheels. The core pipeline never imports napari.
#
#   docker build -f docker/Dockerfile --build-arg TORCH_EXTRA=cu128 -t omero-screen:cu128 .
#   docker run --rm --gpus all --env-file .env.production omero-screen:cu128 omero-screen --help

# ---------------------------------------------------------------- builder ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ARG TORCH_EXTRA=cpu

# numcodecs (via trackastra -> geff) ships no linux/arm64 wheel and compiles
# from source, so the builder needs a C/C++ toolchain. It stays in this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Dependency layer first, so editing source does not re-resolve or re-download
# the multi-GB CUDA wheels. Only the manifests are needed:
# --no-install-workspace skips building the workspace members themselves.
COPY pyproject.toml uv.lock ./
COPY packages/omero-utils/pyproject.toml packages/omero-utils/
COPY packages/omero-screen-napari/pyproject.toml packages/omero-screen-napari/
COPY packages/omero-screen-plots/pyproject.toml packages/omero-screen-plots/
COPY packages/cellview/pyproject.toml packages/cellview/
COPY packages/cellclass/pyproject.toml packages/cellclass/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --extra "${TORCH_EXTRA}" --no-install-workspace

# Then the source, and a second pass to install the workspace packages.
# An explicit allowlist rather than `COPY . .`: with a blanket copy the
# .dockerignore is the only thing keeping credentials out of the image, and a
# secret whose name the ignore list does not happen to match would bake into a
# layer and ship to any registry. Name what the build needs instead.
COPY README.md ./
COPY src/ src/
COPY bin/ bin/
COPY tests/ tests/
COPY packages/ packages/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --extra "${TORCH_EXTRA}"

# ---------------------------------------------------------------- runtime ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

# opencv needs libGL and glib at import time; everything else is self-contained
# in the venv. No compiler here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    CELLPOSE_LOCAL_MODELS_PATH=/models

WORKDIR /app
COPY --from=builder /app /app

CMD ["omero-screen", "--help"]
