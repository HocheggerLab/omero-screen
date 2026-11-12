"""Download the stock Cellpose 3 models, from our mirror first.

Cellpose 3 fetches its stock models from cellpose.org on first use. When that
server is down, a fresh install cannot segment anything. The models used by
default (``nuclei``, ``cyto3``) are mirrored as assets of a GitHub release of
omero-screen and checked against a sha256; cellpose.org remains the fallback,
and the only source for the other stock models.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

#: GitHub release holding the mirrored model files.
MIRROR_TAG = "cellpose3-models-v1"
MIRROR_URL = (
    "https://github.com/HocheggerLab/omero-screen/releases/download/"
    f"{MIRROR_TAG}/{{name}}"
)
CELLPOSE_URL = "https://www.cellpose.org/models/{name}"

#: sha256 of each mirrored file: the models behind ``cp3:nuclei`` and
#: ``cp3:cyto3``, and their size models.
MIRRORED_SHA256 = {
    "nucleitorch_0": "89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303",
    "size_nucleitorch_0.npy": "a79107c9f284b569dd65f3a7363014d613a3403241c062822f07b6f026b4d3ef",
    "cyto3": "2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c",
    "size_cyto3.npy": "87f91feb48019b4dfb0400e7bbb89aca5873368686815aa7b4431c0faee97fc5",
}

#: Seconds to wait for a server before trying the next one.
TIMEOUT = 60


class ModelDownloadError(RuntimeError):
    """No source could provide a Cellpose model file."""


def model_dir() -> Path:
    """Folder Cellpose reads its models from (``$CELLPOSE_LOCAL_MODELS_PATH``)."""
    return Path(
        os.environ.get("CELLPOSE_LOCAL_MODELS_PATH", "~/.cellpose/models")
    ).expanduser()


def sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sources(name: str) -> list[str]:
    """URLs to try for a model file, in order."""
    urls = [CELLPOSE_URL.format(name=name)]
    if name in MIRRORED_SHA256:
        urls.insert(0, MIRROR_URL.format(name=name))
    return urls


def fetch(name: str, dest: Path, urls: Iterable[str] | None = None) -> None:
    """Download model file ``name`` to ``dest``, trying each source in turn.

    The file is written next to ``dest`` and moved into place only when it is
    complete and, for mirrored files, its checksum matches.

    Raises:
        ModelDownloadError: If every source failed; the message lists why and
            how to get the file another way.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    expected = MIRRORED_SHA256.get(name)
    errors = []
    for url in urls if urls is not None else sources(name):
        logger.info(f"Downloading Cellpose model {name} from {url}")
        try:
            with (
                urllib.request.urlopen(url, timeout=TIMEOUT) as response,
                partial.open("wb") as fh,
            ):
                while block := response.read(1 << 20):
                    fh.write(block)
        except OSError as err:  # URLError, HTTPError and timeouts
            partial.unlink(missing_ok=True)
            errors.append(f"{url}: {err}")
            continue
        if expected and sha256(partial) != expected:
            partial.unlink()
            errors.append(f"{url}: checksum mismatch")
            continue
        partial.replace(dest)
        return
    raise ModelDownloadError(
        f"Could not download the Cellpose model {name!r}:\n  "
        + "\n  ".join(errors)
        + "\nRun `omero-screen models pull stock` when the network is back, "
        f"or copy {name} from another machine's {model_dir()} into "
        f"{dest.parent}."
    )


def missing(names: Iterable[str] = MIRRORED_SHA256) -> list[str]:
    """Model files not yet in the Cellpose model folder."""
    return [name for name in names if not (model_dir() / name).exists()]
