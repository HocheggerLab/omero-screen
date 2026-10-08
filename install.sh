#!/bin/sh
# Install omero-screen for a user: pipeline, napari plugin, CellView, plots and
# cellclass, from a release of github.com/HocheggerLab/omero-screen.
#
#   curl -LsSf https://raw.githubusercontent.com/HocheggerLab/omero-screen/main/install.sh | sh
#
# Options (as environment variables):
#   OMERO_SCREEN_VERSION   release tag (default: the highest omero-screen-v* tag)
#   OMERO_SCREEN_BRANCH    install a branch instead of a release (for testing)
#   OMERO_SCREEN_HOME      install location (default: ~/.local/share/omero-screen)
#   OMERO_SCREEN_BIN       where commands are linked (default: ~/.local/bin)
#   OMERO_SCREEN_NO_SETUP  set to 1 to skip `omero-screen setup` at the end
#
# Running it again with a newer version updates the install; your settings
# (~/.config/omero-screen) and data are kept.
set -eu

REPO="HocheggerLab/omero-screen"
HOME_DIR="${OMERO_SCREEN_HOME:-$HOME/.local/share/omero-screen}"
BIN_DIR="${OMERO_SCREEN_BIN:-$HOME/.local/bin}"
# Commands linked onto the PATH.
COMMANDS="omero-screen omero-screen-export omero-screen-images omero-train cellview cellclass cellclass-train cellclass-dataset cellclass-extract napari"

say() { printf '%s\n' "==> $*"; }
die() { printf '%s\n' "error: $*" >&2; exit 1; }

case "$(uname -s)" in
    Darwin|Linux) ;;
    *) die "omero-screen supports macOS and Linux; see the install guide for Windows." ;;
esac
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar >/dev/null 2>&1 || die "tar is required"

# 1. uv (installs Python as needed)
if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi

# 2. The release (or a branch, for testing)
if [ -n "${OMERO_SCREEN_BRANCH:-}" ]; then
    VERSION="branch-$(printf '%s' "$OMERO_SCREEN_BRANCH" | tr '/' '-')"
    ARCHIVE="https://github.com/$REPO/archive/refs/heads/$OMERO_SCREEN_BRANCH.tar.gz"
    rm -rf "$HOME_DIR/$VERSION"   # a branch moves: always fetch it again
else
    VERSION="${OMERO_SCREEN_VERSION:-}"
    if [ -z "$VERSION" ]; then
        # Releases are git tags (omero-screen-vX.Y.Z); take the highest version.
        VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/tags?per_page=100" \
            | sed -n 's/.*"name": *"\(omero-screen-v[0-9][^"]*\)".*/\1/p' \
            | sort -t . -k 1,1 -k 2,2n -k 3,3n | tail -n 1)
        [ -n "$VERSION" ] || die "could not find the latest release; set OMERO_SCREEN_VERSION"
    fi
    ARCHIVE="https://github.com/$REPO/archive/refs/tags/$VERSION.tar.gz"
fi
TARGET="$HOME_DIR/$VERSION"
if [ ! -f "$TARGET/uv.lock" ]; then
    say "Downloading $VERSION"
    mkdir -p "$TARGET"
    curl -fsSL "$ARCHIVE" | tar -xz -C "$TARGET" --strip-components 1
fi

# 3. Install exactly the locked versions, with the napari GUI, without dev tools
say "Installing packages (a few minutes the first time)"
(cd "$TARGET" && uv sync --frozen --no-default-groups --group gui --compile-bytecode)
ln -sfn "$TARGET" "$HOME_DIR/current"

# 4. Commands on the PATH
mkdir -p "$BIN_DIR"
for cmd in $COMMANDS; do
    if [ -x "$HOME_DIR/current/.venv/bin/$cmd" ]; then
        ln -sf "$HOME_DIR/current/.venv/bin/$cmd" "$BIN_DIR/$cmd"
    fi
done
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) say "Add $BIN_DIR to your PATH (e.g. in ~/.zshrc): export PATH=\"$BIN_DIR:\$PATH\""
       PATH="$BIN_DIR:$PATH"; export PATH ;;
esac
say "Installed $VERSION in $TARGET"

# 5. Configure and check
if [ "${OMERO_SCREEN_NO_SETUP:-0}" != "1" ]; then
    if [ -f "${OMERO_SCREEN_CONFIG_DIR:-$HOME/.config/omero-screen}/config.toml" ]; then
        say "Existing configuration kept (change it with: omero-screen setup)"
    elif (: </dev/tty) 2>/dev/null; then
        "$BIN_DIR/omero-screen" setup </dev/tty || say "Setup did not finish; run: omero-screen setup"
    else
        say "Next: omero-screen setup"
    fi
    "$BIN_DIR/omero-screen" doctor || true
fi
