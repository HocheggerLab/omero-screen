#!/usr/bin/env python3
"""Dev server for the documentation: rebuild on save, reload the browser.

The `npm run dev` equivalent. Edit index.qmd, user_guide/*.qmd, great-docs.yml
or anything in assets/, save, and the page in your browser reloads itself once
the rebuild finishes.

Why this exists rather than `great-docs build --watch`:
Great Docs copies the sources into the ephemeral `great-docs/` directory and
runs `quarto preview` from there, so Quarto watches the *copies*. Editing the
real files at the repo root never triggers a rebuild. Each rebuild also
recreates that directory, which orphans the running preview and moves its
port. This script watches the actual sources and serves from a fixed port.

Usage:
    ./scripts/dev_docs.py                          # root site on :8000
    ./scripts/dev_docs.py --port 8080
    ./scripts/dev_docs.py --package packages/cellview
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

GREAT_DOCS_VERSION = "0.17.0"
ROOT = Path(__file__).resolve().parent.parent

# Bumped after every successful rebuild. The browser polls it and reloads
# when it changes.
GENERATION = 0
_OFFLINE_CACHED: bool | None = None
BUILDING = False

RELOAD_SNIPPET = b"""
<script>
(function () {
  var current = null;
  async function poll() {
    try {
      var r = await fetch('/__dev/stamp', {cache: 'no-store'});
      var t = (await r.text()).trim();
      if (current === null) { current = t; }
      else if (t !== current) { location.reload(); }
    } catch (e) { /* server restarting; ignore */ }
  }
  setInterval(poll, 700);
  poll();
})();
</script>
"""


def _offline_ok() -> bool:
    """True when great-docs is already in the uv cache.

    ``--offline`` avoids re-resolving the project on every rebuild, but it
    cannot work against a cold cache, so probe once before relying on it.
    """
    global _OFFLINE_CACHED
    if _OFFLINE_CACHED is None:
        _OFFLINE_CACHED = (
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--offline",
                    "--with",
                    f"great-docs=={GREAT_DOCS_VERSION}",
                    "great-docs",
                    "--version",
                ],
                cwd=ROOT,
                capture_output=True,
            ).returncode
            == 0
        )
    return _OFFLINE_CACHED


def watched_files(project: Path) -> list[Path]:
    """Every source file whose change should trigger a rebuild."""
    found: list[Path] = []
    for name in ("index.qmd", "great-docs.yml", "README.md"):
        candidate = project / name
        if candidate.is_file():
            found.append(candidate)
    for folder in ("user_guide", "assets", "custom"):
        directory = project / folder
        if directory.is_dir():
            found.extend(p for p in directory.rglob("*") if p.is_file())
    return found


def fingerprint(project: Path) -> tuple[tuple[str, float], ...]:
    """A cheap signature of the watched files' paths and modification times."""
    out = []
    for path in watched_files(project):
        try:
            out.append((str(path), path.stat().st_mtime))
        except OSError:
            continue
    return tuple(sorted(out))


def build(project: Path) -> bool:
    """Rebuild the site. Returns True when the build succeeded."""
    global BUILDING
    BUILDING = True
    print(f"\n\033[1;34m==> rebuilding\033[0m {time.strftime('%H:%M:%S')}")
    try:
        result = subprocess.run(
            [
                # Prefer the cached environment: without --offline, uv
                # re-resolves the project on every rebuild, which needs the
                # network for the zeroc-ice wheel. Falls back to a networked
                # run when the cache has no great-docs yet.
                "uv",
                "run",
                *(["--offline"] if _offline_ok() else []),
                "--with",
                f"great-docs=={GREAT_DOCS_VERSION}",
                "great-docs",
                "build",
                "--no-refresh",
                "--project-path",
                str(project),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        BUILDING = False

    if result.returncode != 0:
        print("\033[1;31m==> BUILD FAILED — the page is unchanged\033[0m")
        for line in (result.stdout + result.stderr).splitlines():
            if any(k in line for k in ("FAIL", "Error", "error:", "WARN")):
                print("   ", line.strip())
        return False

    for line in result.stdout.splitlines():
        if "WARN:" in line:
            print("   ", line.strip())
    return True


def watcher(project: Path) -> None:
    """Rebuild whenever a watched file changes."""
    global GENERATION
    last = fingerprint(project)
    while True:
        time.sleep(1)
        now = fingerprint(project)
        if now == last:
            continue
        last = now
        if build(project):
            GENERATION += 1
            print("\033[1;32m==> reloaded\033[0m")


def make_handler(site: Path) -> type[http.server.SimpleHTTPRequestHandler]:
    """A static handler that injects the reload poller into every page."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(site), **kwargs)

        def log_message(self, *args: Any, **kwargs: Any) -> None:
            """Silence the per-request logging."""

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if self.path.startswith("/__dev/stamp"):
                state = f"{GENERATION}:{'building' if BUILDING else 'idle'}"
                self._send_bytes(state.encode(), "text/plain; charset=utf-8")
                return

            path = Path(self.translate_path(self.path))
            if path.is_dir():
                path = path / "index.html"

            if path.suffix == ".html" and path.is_file():
                data = path.read_bytes()
                if b"</body>" in data:
                    data = data.replace(
                        b"</body>", RELOAD_SNIPPET + b"</body>", 1
                    )
                else:
                    data += RELOAD_SNIPPET
                self._send_bytes(data, "text/html; charset=utf-8")
                return

            super().do_GET()

    return Handler


def main() -> None:
    """Build once, then serve and watch until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--package",
        default=".",
        help="Project to serve, e.g. packages/cellview (default: root site)",
    )
    args = parser.parse_args()

    project = (ROOT / args.package).resolve()
    site = project / "great-docs" / "_site"

    if not build(project) and not site.is_dir():
        sys.exit("initial build failed and no previous site exists")

    handler = make_handler(site)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        threading.Thread(target=watcher, args=(project,), daemon=True).start()
        rel = os.path.relpath(project, ROOT)
        print(
            f"\n\033[1;32m==> {rel} live on "
            f"http://localhost:{args.port}\033[0m"
        )
        print(
            "==> watching index.qmd, great-docs.yml, user_guide/, assets/ "
            "— the browser reloads itself. Ctrl-C to stop.\n"
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
