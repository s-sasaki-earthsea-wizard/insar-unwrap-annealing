"""Registry of approved sample datasets, download and integrity checks.

Only datasets listed in the registry (``configs/datasets.yaml``) can be fetched.
Each entry records the URL, the expected size and at least one checksum.
Downloads go through ``curl``: it follows the Earthdata Login redirect chain
with credentials from ``~/.netrc`` and a temporary cookie jar, which the
standard library does not handle cleanly. Credentials are never read or
logged by this module.

A file is moved into place only after its size and checksums match. Existing
files are verified, never overwritten or deleted. Every verified file gets a
``<file>.provenance.yaml`` sidecar.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from insar_unwrap import __version__

DEFAULT_REGISTRY = Path("configs/datasets.yaml")
DEFAULT_DEST = Path("data/raw")
_CHUNK = 8 * 1024 * 1024
_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class DatasetError(Exception):
    """Raised for registry problems, failed downloads or integrity mismatches."""


class DatasetEntry(BaseModel):
    """One approved dataset.

    Attributes:
        description: What the data is, including caveats (e.g. not a NISAR
            observation).
        filename: File name as delivered.
        url: HTTPS download URL.
        size_bytes: Expected size.
        md5: Expected MD5 (lower-case hex), e.g. as published in CMR.
        sha256: Expected SHA-256 (lower-case hex).
        checksum_source: Where the checksums come from.
        requires_login: Whether the URL needs Earthdata Login.
        landing_page: Human-readable source page.
        collection: Collection short name and concept id, if any.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    filename: str = Field(min_length=1, pattern=r"^[^/\\]+$")
    url: str = Field(pattern=r"^https://")
    size_bytes: Annotated[int, Field(gt=0)]
    md5: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checksum_source: str = Field(min_length=1)
    requires_login: bool
    landing_page: str | None = None
    collection: str | None = None

    @model_validator(mode="after")
    def _needs_a_checksum(self) -> DatasetEntry:
        if self.md5 is None and self.sha256 is None:
            raise ValueError("each dataset needs md5 and/or sha256")
        return self


class Registry(BaseModel):
    """All approved datasets, keyed by a short snake_case name."""

    model_config = ConfigDict(extra="forbid")

    datasets: dict[str, DatasetEntry]

    @model_validator(mode="after")
    def _check_names(self) -> Registry:
        bad = [n for n in self.datasets if not _NAME.match(n)]
        if bad:
            raise ValueError(f"dataset names must be snake_case: {bad}")
        return self


@dataclass(frozen=True)
class Digest:
    """Size and checksums of a local file."""

    size_bytes: int
    md5: str
    sha256: str


@dataclass(frozen=True)
class FetchResult:
    """Outcome of :func:`fetch_dataset`.

    Attributes:
        path: Verified file.
        status: ``"downloaded"`` or ``"present"`` (already there and verified).
        provenance: Sidecar path.
    """

    path: Path
    status: str
    provenance: Path


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> Registry:
    """Load and validate the dataset registry.

    Raises:
        DatasetError: If the file is missing or invalid.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Registry.model_validate(raw)
    except FileNotFoundError as exc:
        raise DatasetError(f"registry not found: {path}") from exc
    except (yaml.YAMLError, ValidationError) as exc:
        raise DatasetError(f"invalid registry {path}: {exc}") from exc


def file_digest(path: Path) -> Digest:
    """Stream a file once and return its size, MD5 and SHA-256."""
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
    return Digest(size, md5.hexdigest(), sha.hexdigest())


def verify(path: Path, entry: DatasetEntry) -> Digest:
    """Check size and every checksum given in ``entry``.

    Returns:
        The digest of the file.

    Raises:
        DatasetError: On any mismatch.
    """
    digest = file_digest(path)
    problems = []
    if digest.size_bytes != entry.size_bytes:
        problems.append(f"size {digest.size_bytes} != {entry.size_bytes}")
    if entry.md5 is not None and digest.md5 != entry.md5:
        problems.append(f"md5 {digest.md5} != {entry.md5}")
    if entry.sha256 is not None and digest.sha256 != entry.sha256:
        problems.append(f"sha256 {digest.sha256} != {entry.sha256}")
    if problems:
        raise DatasetError(f"{path}: " + "; ".join(problems))
    return digest


def _curl_version(curl: str) -> str:
    out = subprocess.run([curl, "--version"], capture_output=True, text=True, check=False)
    return out.stdout.splitlines()[0] if out.stdout else "unknown"


def _download(url: str, target: Path, *, curl: str, requires_login: bool) -> None:
    cmd = [curl, "--location", "--fail", "--silent", "--show-error", "--output", str(target)]
    with tempfile.TemporaryDirectory(prefix="insar-unwrap-fetch-") as tmp:
        if requires_login:
            jar = str(Path(tmp) / "cookies.txt")
            cmd += ["--netrc-optional", "--cookie", jar, "--cookie-jar", jar]
        proc = subprocess.run([*cmd, url], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        hint = " (Earthdata Login: check ~/.netrc for urs.earthdata.nasa.gov)" * requires_login
        raise DatasetError(f"curl failed ({proc.returncode}): {proc.stderr.strip()}{hint}")


def _write_provenance(
    path: Path, name: str, entry: DatasetEntry, digest: Digest, status: str, curl_info: str
) -> Path:
    sidecar = path.with_name(path.name + ".provenance.yaml")
    record = {
        "dataset": name,
        "description": entry.description,
        "file": path.name,
        "url": entry.url,
        "collection": entry.collection,
        "size_bytes": digest.size_bytes,
        "md5": digest.md5,
        "sha256": digest.sha256,
        "checksum_source": entry.checksum_source,
        "verified_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "obtained_by": (
            f"insar-unwrap {__version__} fetch-sample ({curl_info})"
            if status == "downloaded"
            else "present before fetch-sample; verified only"
        ),
    }
    sidecar.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    return sidecar


def fetch_dataset(
    name: str,
    registry: Registry,
    dest_root: Path = DEFAULT_DEST,
    *,
    curl: str = "curl",
) -> FetchResult:
    """Download one approved dataset into ``dest_root/<name>/`` and verify it.

    An existing file is verified instead of downloaded. A provenance sidecar is
    written next to the file (refreshed for an existing file only if missing).

    Args:
        name: Registry key.
        registry: Loaded registry.
        dest_root: Root directory for raw products.
        curl: curl executable.

    Returns:
        Path, status and provenance sidecar.

    Raises:
        DatasetError: Unknown name, missing curl, failed download or mismatch.
            A mismatching download stays as ``<file>.part`` for inspection.
    """
    if name not in registry.datasets:
        known = ", ".join(sorted(registry.datasets))
        raise DatasetError(f"unknown dataset {name!r}; known: {known}")
    entry = registry.datasets[name]
    dest = Path(dest_root) / name
    path = dest / entry.filename
    sidecar = path.with_name(path.name + ".provenance.yaml")
    if path.exists():
        digest = verify(path, entry)
        if not sidecar.exists():
            _write_provenance(path, name, entry, digest, "present", "")
        return FetchResult(path, "present", sidecar)

    if shutil.which(curl) is None:
        raise DatasetError(f"curl executable not found: {curl}")
    dest.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    _download(entry.url, part, curl=curl, requires_login=entry.requires_login)
    digest = verify(part, entry)
    part.rename(path)
    _write_provenance(path, name, entry, digest, "downloaded", _curl_version(curl))
    return FetchResult(path, "downloaded", sidecar)
