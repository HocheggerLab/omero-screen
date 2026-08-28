#!/usr/bin/env bash
#
# Build the complete OMERO-Screen documentation site.
#
# Great Docs documents exactly one package and one Click CLI per site, so the
# platform docs are six separate builds assembled into one deployable tree:
#
#   _site/                 root umbrella  — landing page, Learn, core reference
#   _site/cellview/        cellview
#   _site/cellclass/       cellclass      — includes generated CLI reference
#   _site/plots/           omero-screen-plots
#   _site/napari/          omero-screen-napari
#   _site/utils/           omero-utils
#
# Usage:
#   ./scripts/build_docs.sh            # build everything into ./_site
#   ./scripts/build_docs.sh --serve    # build, then serve on localhost:8000
#
set -euo pipefail

GREAT_DOCS_VERSION="0.17.0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/_site"

cd "${ROOT}"

# package directory -> output subdirectory
SUBSITES=(
  "packages/cellview:cellview"
  "packages/cellclass:cellclass"
  "packages/omero-screen-plots:plots"
  "packages/omero-screen-napari:napari"
  "packages/omero-utils:utils"
)

# --offline reuses the cached environment. Without it uv re-resolves the
# project on every call, which needs to reach GitHub for the zeroc-ice wheel
# and fails outright when the network is down.
gd() {
  uv run --offline --with "great-docs==${GREAT_DOCS_VERSION}" great-docs "$@"
}

echo "==> Cleaning ${OUT}"
rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "==> Building umbrella site (omero_screen)"
gd build --no-refresh
cp -R "${ROOT}/great-docs/_site/." "${OUT}/"

for entry in "${SUBSITES[@]}"; do
  pkg="${entry%%:*}"
  sub="${entry##*:}"
  echo "==> Building ${pkg} -> /${sub}/"
  gd build --no-refresh --project-path "${pkg}"
  mkdir -p "${OUT}/${sub}"
  cp -R "${ROOT}/${pkg}/great-docs/_site/." "${OUT}/${sub}/"
done

echo
echo "==> Done. $(find "${OUT}" -name '*.html' | wc -l | tr -d ' ') HTML pages in ${OUT}"

if [[ "${1:-}" == "--serve" ]]; then
  echo "==> Serving on http://localhost:8000 (Ctrl-C to stop)"
  cd "${OUT}"
  python3 -m http.server 8000
fi
