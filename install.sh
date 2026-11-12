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
#   OMERO_SCREEN_BIN       where commands are linked (default: uv's tool bin dir)
#   OMERO_SCREEN_NO_SETUP  set to 1 to skip `omero-screen setup` at the end
#
# Running it again updates the install (or `omero-screen self update`); your
# settings (~/.config/omero-screen) and data are kept. The previous release is
# kept for a rollback (OMERO_SCREEN_VERSION=<old tag>); older ones are removed.
#
# Uninstall (or `omero-screen self uninstall`):
#   curl -LsSf https://raw.githubusercontent.com/HocheggerLab/omero-screen/main/install.sh | sh -s -- --uninstall
# --purge also deletes the settings and the keychain password. Your CellView
# database, image cache and Cellpose models are never deleted.
set -eu

REPO="HocheggerLab/omero-screen"
HOME_DIR="${OMERO_SCREEN_HOME:-$HOME/.local/share/omero-screen}"
# Commands linked onto the PATH.
COMMANDS="omero-screen omero-screen-export omero-screen-images omero-train cellview cellclass cellclass-train cellclass-dataset cellclass-extract napari"

say() { printf '%s\n' "==> $*"; }
die() { printf '%s\n' "error: $*" >&2; exit 1; }

UNINSTALL=0
PURGE=0
for arg in "$@"; do
    case "$arg" in
        --uninstall) UNINSTALL=1 ;;
        --purge) PURGE=1 ;;
        *) die "unknown option: $arg (options: --uninstall [--purge])" ;;
    esac
done
[ "$PURGE" = 0 ] || [ "$UNINSTALL" = 1 ] || die "--purge goes with --uninstall"

# Remove the install, the commands linked to it and, with --purge, the
# settings and the keychain password. User data stays.
uninstall() {
    [ -f "$HOME_DIR/install.env" ] || [ -L "$HOME_DIR/current" ] \
        || die "no omero-screen install in $HOME_DIR (set OMERO_SCREEN_HOME if it is elsewhere)"
    INSTALLED_BIN_DIR=""
    INSTALLED_COMMANDS=""
    # shellcheck disable=SC1091  # written by this script
    if [ -f "$HOME_DIR/install.env" ]; then . "$HOME_DIR/install.env"; fi
    bin_dir="${INSTALLED_BIN_DIR:-${OMERO_SCREEN_BIN:-$(uv tool dir --bin 2>/dev/null || printf '%s' "$HOME/.local/bin")}}"
    for cmd in ${INSTALLED_COMMANDS:-$COMMANDS}; do
        link="$bin_dir/$cmd"
        # Only our links: a napari (or anything) of the user's own stays.
        if [ -L "$link" ]; then
            case "$(readlink "$link")" in
                "$HOME_DIR"/*) rm -f "$link" ;;
            esac
        fi
    done
    say "Removed the commands from $bin_dir"
    rm -rf "$HOME_DIR"
    say "Removed $HOME_DIR"
    if [ "$PURGE" = 1 ]; then
        rm -rf "${OMERO_SCREEN_CONFIG_DIR:-$HOME/.config/omero-screen}"
        if command -v security >/dev/null 2>&1; then
            while security delete-generic-password -s omero-screen >/dev/null 2>&1; do :; done
        elif command -v secret-tool >/dev/null 2>&1; then
            secret-tool clear service omero-screen 2>/dev/null || true
        fi
        say "Removed your settings and the keychain password"
    else
        say "Kept your settings (~/.config/omero-screen); --purge removes them"
    fi
    for data in "$HOME/.cellview" "$HOME/omero-cache" "$HOME/.cellpose/models"; do
        if [ -e "$data" ]; then
            say "Kept your data in $data (delete it yourself if you no longer need it)"
        fi
    done
    say "uv was left installed (remove it with: uv self uninstall)"
    exit 0
}

case "$(uname -s)" in
    Darwin|Linux) ;;
    *) die "omero-screen supports macOS and Linux; see the install guide for Windows." ;;
esac
[ "$UNINSTALL" = 0 ] || uninstall
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar >/dev/null 2>&1 || die "tar is required"

# 1. uv (installs Python as needed)
if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi
UV_BIN_DIR=$(uv tool dir --bin 2>/dev/null || printf '%s' "$HOME/.local/bin")
BIN_DIR="${OMERO_SCREEN_BIN:-$UV_BIN_DIR}"

# 2. The release (or a branch, for testing)
if [ -n "${OMERO_SCREEN_BRANCH:-}" ]; then
    VERSION="branch-$(printf '%s' "$OMERO_SCREEN_BRANCH" | tr '/' '-')"
    ARCHIVE="https://github.com/$REPO/archive/refs/heads/$OMERO_SCREEN_BRANCH.tar.gz"
    FETCH=1   # a branch moves: always fetch it again
else
    VERSION="${OMERO_SCREEN_VERSION:-}"
    if [ -z "$VERSION" ]; then
        # Releases are git tags (omero-screen-vX.Y.Z); take the highest version.
        TAGS=$(curl -fsSL "https://api.github.com/repos/$REPO/tags?per_page=100") \
            || die "could not reach GitHub; check the network (VPN, proxy) and run the installer again"
        VERSION=$(printf '%s\n' "$TAGS" \
            | sed -n 's/.*"name": *"\(omero-screen-v[0-9][^"]*\)".*/\1/p' \
            | sort -t . -k 1,1 -k 2,2n -k 3,3n | tail -n 1)
        [ -n "$VERSION" ] || die "could not find the latest release; set OMERO_SCREEN_VERSION"
    fi
    case "$VERSION" in omero-screen-v*) ;; *) VERSION="omero-screen-v$VERSION" ;; esac
    ARCHIVE="https://github.com/$REPO/archive/refs/tags/$VERSION.tar.gz"
fi
TARGET="$HOME_DIR/$VERSION"
if [ "${FETCH:-0}" = 1 ] || [ ! -f "$TARGET/uv.lock" ]; then
    say "Downloading $VERSION"
    mkdir -p "$HOME_DIR"
    # Download and unpack beside the target, then move it into place: a failed
    # download never leaves a half-made release behind.
    TMP=$(mktemp -d "$HOME_DIR/.download.XXXXXX")
    trap 'rm -rf "$TMP"' EXIT
    curl -fsSL -o "$TMP/release.tar.gz" "$ARCHIVE" \
        || die "could not download $ARCHIVE; check the network (VPN, proxy) and run the installer again"
    mkdir "$TMP/src"
    tar -xzf "$TMP/release.tar.gz" -C "$TMP/src" --strip-components 1 \
        || die "the download of $VERSION is damaged; run the installer again"
    [ -f "$TMP/src/uv.lock" ] || die "the download of $VERSION is incomplete; run the installer again"
    rm -rf "$TARGET"
    mv "$TMP/src" "$TARGET"
fi

# 3. Install exactly the locked versions, with the napari GUI, without dev tools
say "Installing packages (a few minutes the first time)"
(cd "$TARGET" && uv sync --frozen --no-default-groups --group gui --compile-bytecode)
PREVIOUS=$(readlink "$HOME_DIR/current" 2>/dev/null || true)
ln -sfn "$TARGET" "$HOME_DIR/current"

# Keep this release and the previous one (for a rollback); each is several GB.
for old in "$HOME_DIR"/omero-screen-v* "$HOME_DIR"/branch-*; do
    [ -d "$old" ] || continue
    if [ "$old" = "$TARGET" ] || [ "$old" = "$PREVIOUS" ]; then continue; fi
    say "Removing the old release $(basename "$old")"
    rm -rf "$old"
done

# 4. Commands on the PATH
mkdir -p "$BIN_DIR"
for cmd in $COMMANDS; do
    if [ -x "$HOME_DIR/current/.venv/bin/$cmd" ]; then
        ln -sf "$HOME_DIR/current/.venv/bin/$cmd" "$BIN_DIR/$cmd"
    fi
done
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        PATH="$BIN_DIR:$PATH"
        export PATH
        if [ -z "${OMERO_SCREEN_BIN:-}" ]; then
            say "Adding $BIN_DIR to your shell PATH"
            if uv tool update-shell; then
                say "PATH updated automatically; open a new terminal window before using the commands"
            else
                say "Could not update your shell PATH automatically; run: uv tool update-shell"
            fi
        else
            say "$BIN_DIR is not on your PATH; add it in your shell configuration"
        fi
        ;;
esac
# What --uninstall and `omero-screen self` need to find again.
cat > "$HOME_DIR/install.env" <<EOF
# Written by install.sh; read by install.sh --uninstall and omero-screen self.
INSTALLED_VERSION='$VERSION'
INSTALLED_BIN_DIR='$BIN_DIR'
INSTALLED_COMMANDS='$COMMANDS'
EOF
say "Installed $VERSION in $TARGET"

# 5. The stock Cellpose models, so a network problem shows now, not mid-plate
"$HOME_DIR/current/.venv/bin/omero-screen" models pull stock \
    || say "Could not download the Cellpose models now; later run: omero-screen models pull stock"

# 6. Configure and check
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
