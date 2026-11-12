"""Stock Cellpose models: mirror first, cellpose.org as the fallback."""

import hashlib
from pathlib import Path

import pytest

from omero_screen import stock_models


def _source(tmp_path: Path, name: str, data: bytes) -> str:
    src = tmp_path / "server" / name
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(data)
    return src.as_uri()


def test_mirrored_models_try_the_mirror_first() -> None:
    urls = stock_models.sources("cyto3")
    assert urls[0].startswith("https://github.com/HocheggerLab/")
    assert urls[-1] == "https://www.cellpose.org/models/cyto3"
    assert stock_models.sources("tissuenet_cp3") == [
        "https://www.cellpose.org/models/tissuenet_cp3"
    ]


def test_falls_back_when_the_first_source_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"weights"
    monkeypatch.setitem(
        stock_models.MIRRORED_SHA256, "m", hashlib.sha256(data).hexdigest()
    )
    dest = tmp_path / "models" / "m"
    down = (tmp_path / "missing").as_uri()
    stock_models.fetch("m", dest, [down, _source(tmp_path, "m", data)])
    assert dest.read_bytes() == data
    assert not dest.with_name("m.part").exists()


def test_a_bad_checksum_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(stock_models.MIRRORED_SHA256, "m", "0" * 64)
    dest = tmp_path / "models" / "m"
    with pytest.raises(stock_models.ModelDownloadError, match="checksum"):
        stock_models.fetch("m", dest, [_source(tmp_path, "m", b"tampered")])
    assert not dest.exists()
    assert not dest.with_name("m.part").exists()


def test_the_error_says_how_to_recover(tmp_path: Path) -> None:
    with pytest.raises(stock_models.ModelDownloadError) as err:
        stock_models.fetch(
            "cyto3", tmp_path / "cyto3", [(tmp_path / "gone").as_uri()]
        )
    assert "omero-screen models pull stock" in str(err.value)


def test_missing_lists_absent_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path))
    (tmp_path / "cyto3").write_bytes(b"x")
    assert "cyto3" not in stock_models.missing()
    assert "nucleitorch_0" in stock_models.missing()
