# Headless omero-screen runtime.
#
# torch comes from PyPI: on linux/arm64 that wheel is CPU-only, on
# linux/amd64 it carries the CUDA stack and works with `--gpus all`. Selecting
# a build via the PyTorch indexes was tried and reverted — that index serves a
# torchvision 0.24.1+d801a34 with only win_arm64 wheels, which uv picks as the
# highest version and which then fails to install anywhere else.
#
# The image does not install napari: `--no-default-groups` drops the `gui` group, which is
# the whole point — napari[all] pulls pyqt5, pyqt5-qt5 and triangle, none of
# which ship linux/arm64 wheels. The core pipeline never imports napari.
#
#   docker build -f docker/Dockerfile -t omero-screen .
#   docker run --rm --gpus all -v "$PWD/.env:/app/.env:ro" omero-screen omero-screen --help

# ---------------------------------------------------------------- builder ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

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
    uv sync --locked --no-default-groups --no-install-workspace

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
COPY skills/ skills/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups

# ---------------------------------------------------------------- runtime ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

# opencv needs libGL and glib at import time; everything else is self-contained
# in the venv. No compiler here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pi, the coding agent the in-container agent tests drive. It is an npm global
# (@earendil-works/pi-coding-agent) needing node >= 22.19, so the image carries
# a second runtime: node is copied from the official image rather than added
# via a distro package, because bookworm ships node 18. Pinned for the same
# reason torch is — an agent harness changing under the tests is not a
# variable we want.
COPY --from=node:22-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-bookworm-slim /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && npm install -g @earendil-works/pi-coding-agent@0.85.1 \
    && npm cache clean --force

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    CELLPOSE_LOCAL_MODELS_PATH=/models

WORKDIR /app
COPY --from=builder /app /app

# Register the repo's omero-screen skill where pi looks for it. A symlink, not
# a copy, so mounting the repo over /app/skills during development changes what
# the agent sees without a rebuild. cellclass-train is deliberately not
# registered: it drives model training, which is not what these tests exercise.
RUN mkdir -p /root/.pi/agent/skills \
    && ln -s /app/skills/omero-screen /root/.pi/agent/skills/omero-screen

# A self-contained pi configuration. Nothing is mounted from a host ~/.pi: the
# container must behave like a fresh install, so what the agent tests exercise
# is this file, not whatever a developer happens to have configured locally.
#
# The only provider is the tailnet Ollama box, and apiKey is the literal string
# "ollama" that Ollama ignores — so there is no credential here and no
# auth.json to mount. The tailnet address is reachable from inside the
# container because Docker routes container traffic through the host, which
# holds the tailnet route; a MagicDNS name would not resolve, hence the raw IP.
ARG OLLAMA_BASE_URL=http://100.120.147.41:11434/v1
COPY docker/pi/models.json docker/pi/settings.json /root/.pi/agent/
RUN sed -i "s|__OLLAMA_BASE_URL__|${OLLAMA_BASE_URL}|" /root/.pi/agent/models.json

CMD ["omero-screen", "--help"]
