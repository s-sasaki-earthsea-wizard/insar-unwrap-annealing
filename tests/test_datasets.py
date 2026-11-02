import hashlib
import shutil

import pytest

from insar_unwrap.io.datasets import (
    DatasetEntry,
    DatasetError,
    Registry,
    fetch_dataset,
    load_registry,
    verify,
)

from .conftest import REPO_ROOT


def _entry(path, payload, **overrides):
    fields = {
        "description": "test",
        "filename": "sample.bin",
        "url": path.as_uri(),
        "size_bytes": len(payload),
        "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "checksum_source": "test",
        "requires_login": False,
    }
    # model_construct skips validation so that a file:// URL can stand in for HTTPS.
    return DatasetEntry.model_construct(**(fields | overrides))


def test_repository_registry_is_valid():
    registry = load_registry(REPO_ROOT / "configs" / "datasets.yaml")
    assert "nisar_sample_alos1_gunw_d00404" in registry.datasets
    for entry in registry.datasets.values():
        assert entry.url.startswith("https://")
        assert entry.md5 or entry.sha256


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"Bad-Name": {}}, "snake_case|Field required"),
        (
            {
                "ok": {
                    "description": "d",
                    "filename": "f.h5",
                    "url": "https://x/f.h5",
                    "size_bytes": 1,
                    "checksum_source": "s",
                    "requires_login": False,
                }
            },
            "md5 and/or sha256",
        ),
    ],
)
def test_registry_validation_errors(tmp_path, data, fragment):
    import yaml

    path = tmp_path / "reg.yaml"
    path.write_text(yaml.safe_dump({"datasets": data}), encoding="utf-8")
    with pytest.raises(DatasetError, match=fragment):
        load_registry(path)


def test_verify_reports_size_and_checksum_mismatch(tmp_path):
    payload = b"abc"
    path = tmp_path / "f.bin"
    path.write_bytes(payload)
    assert verify(path, _entry(path, payload)).size_bytes == 3
    with pytest.raises(DatasetError, match="size 3 != 4.*sha256"):
        verify(path, _entry(path, payload, size_bytes=4, sha256="0" * 64))


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl not installed")
def test_fetch_downloads_verifies_and_then_reports_present(tmp_path):
    payload = b"wrapped phase bytes"
    src = tmp_path / "remote.bin"
    src.write_bytes(payload)
    registry = Registry.model_construct(datasets={"sample": _entry(src, payload)})
    dest = tmp_path / "raw"

    first = fetch_dataset("sample", registry, dest)
    assert first.status == "downloaded"
    assert first.path.read_bytes() == payload
    assert "sha256" in first.provenance.read_text(encoding="utf-8")
    assert fetch_dataset("sample", registry, dest).status == "present"


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl not installed")
def test_fetch_keeps_mismatching_download_as_part_file(tmp_path):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"tampered")
    registry = Registry.model_construct(datasets={"sample": _entry(src, b"original")})
    with pytest.raises(DatasetError, match="mismatch|!="):
        fetch_dataset("sample", registry, tmp_path / "raw")
    assert (tmp_path / "raw" / "sample" / "sample.bin.part").exists()
    assert not (tmp_path / "raw" / "sample" / "sample.bin").exists()


def test_fetch_rejects_unknown_name(tmp_path):
    registry = Registry.model_construct(datasets={})
    with pytest.raises(DatasetError, match="unknown dataset"):
        fetch_dataset("nope", registry, tmp_path)
