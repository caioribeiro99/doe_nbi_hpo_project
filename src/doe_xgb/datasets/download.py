"""Shared utilities for the per-dataset downloader scripts.

Pure standard library + lazy imports of pandas / openpyxl / xlrd /
sklearn so that importing this module never triggers a download or a
heavy dependency. The downloader scripts under ``scripts/`` import
from here.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from zipfile import ZipFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3]


def data_root() -> Path:
    return _repo_root() / "data" / "source"


def dataset_dir(dataset_id: str) -> Path:
    return data_root() / dataset_id


def raw_dir(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "raw"


def processed_dir(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "processed"


def manifest_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "manifest.json"


def checksums_path() -> Path:
    return data_root() / "CHECKSUMS.txt"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


class DownloadError(RuntimeError):
    pass


def download_url(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    timeout: float = 60.0,
    user_agent: str = "doe-xgb/0.2",
) -> Path:
    """Download ``url`` to ``dest``. Idempotent unless ``force=True``.

    Raises :class:`DownloadError` on network failure.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:  # noqa: S310
            shutil.copyfileobj(resp, out)
    except urllib.error.URLError as e:
        raise DownloadError(f"failed to download {url}: {e}") from e
    return dest


# ---------------------------------------------------------------------------
# ZIP / XLS helpers
# ---------------------------------------------------------------------------


def extract_zip(src: Path, dest_dir: Path) -> list[Path]:
    src = Path(src)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with ZipFile(src) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            target = dest_dir / Path(name).name
            with zf.open(name) as src_f, target.open("wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)
            out.append(target)
    return out


def read_xlsx(path: Path):
    """Read an XLSX (or XLS) file lazily."""
    import pandas as pd  # noqa: PLC0415

    return pd.read_excel(path)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class FileEntry:
    path: str
    sha256: str
    size_bytes: int


@dataclass
class DatasetManifest:
    dataset_id: str
    source_url: str | None
    openml_id: int | None
    target_column: str
    target_transform: str | None
    notes: str | None
    raw_files: list[FileEntry] = field(default_factory=list)
    processed_files: list[FileEntry] = field(default_factory=list)
    generated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["raw_files"] = [asdict(f) for f in self.raw_files]
        out["processed_files"] = [asdict(f) for f in self.processed_files]
        return out


def _file_entry(path: Path) -> FileEntry:
    p = Path(path)
    return FileEntry(
        path=str(p.relative_to(data_root())) if p.is_absolute() and str(p).startswith(str(data_root())) else str(p),
        sha256=sha256_file(p),
        size_bytes=int(p.stat().st_size),
    )


def write_manifest(
    dataset_id: str,
    *,
    raw_files: Iterable[Path],
    processed_files: Iterable[Path],
    source_url: str | None,
    openml_id: int | None,
    target_column: str,
    target_transform: str | None,
    notes: str | None,
) -> Path:
    mp = manifest_path(dataset_id)
    mp.parent.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        source_url=source_url,
        openml_id=openml_id,
        target_column=target_column,
        target_transform=target_transform,
        notes=notes,
        raw_files=[_file_entry(p) for p in raw_files],
        processed_files=[_file_entry(p) for p in processed_files],
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    mp.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return mp


def read_manifest(dataset_id: str) -> DatasetManifest | None:
    mp = manifest_path(dataset_id)
    if not mp.exists():
        return None
    blob = json.loads(mp.read_text(encoding="utf-8"))
    return DatasetManifest(
        dataset_id=blob["dataset_id"],
        source_url=blob.get("source_url"),
        openml_id=blob.get("openml_id"),
        target_column=blob.get("target_column", ""),
        target_transform=blob.get("target_transform"),
        notes=blob.get("notes"),
        raw_files=[FileEntry(**f) for f in blob.get("raw_files", [])],
        processed_files=[FileEntry(**f) for f in blob.get("processed_files", [])],
        generated_at=blob.get("generated_at"),
    )


def manifest_matches_disk(dataset_id: str) -> bool:
    """Return True if every file in the manifest exists with the recorded
    SHA-256."""
    m = read_manifest(dataset_id)
    if m is None:
        return False
    for entry in (*m.raw_files, *m.processed_files):
        candidate = data_root() / entry.path if not Path(entry.path).is_absolute() else Path(entry.path)
        if not candidate.exists():
            return False
        if sha256_file(candidate) != entry.sha256:
            return False
    return True


# ---------------------------------------------------------------------------
# Aggregated CHECKSUMS.txt
# ---------------------------------------------------------------------------


def update_checksums_txt(dataset_id: str) -> Path:
    """Re-write the CHECKSUMS.txt block for one dataset; preserves blocks
    for other datasets."""
    cp = checksums_path()
    cp.parent.mkdir(parents=True, exist_ok=True)
    block_marker_start = f"# >>> {dataset_id}\n"
    block_marker_end = f"# <<< {dataset_id}\n"
    existing = cp.read_text(encoding="utf-8") if cp.exists() else ""
    # Strip any prior block.
    if block_marker_start in existing and block_marker_end in existing:
        a = existing.index(block_marker_start)
        b = existing.index(block_marker_end) + len(block_marker_end)
        existing = existing[:a] + existing[b:]
    if not existing.startswith("# SHA-256 checksums"):
        header = (
            "# SHA-256 checksums for canonical datasets used by the article-track configs.\n"
            "# Auto-managed by scripts/fetch_<dataset>_dataset.py via\n"
            "# doe_xgb.datasets.download.update_checksums_txt.\n"
            "# Format: <sha256>  <relative_path>\n\n"
        )
        existing = header + existing
    m = read_manifest(dataset_id)
    if m is None:
        return cp
    block_lines = [block_marker_start]
    block_lines.append(f"# generated_at: {m.generated_at}\n")
    if m.source_url:
        block_lines.append(f"# source_url: {m.source_url}\n")
    if m.openml_id is not None:
        block_lines.append(f"# openml_id: {m.openml_id}\n")
    for entry in (*m.raw_files, *m.processed_files):
        block_lines.append(f"{entry.sha256}  {entry.path}\n")
    block_lines.append(block_marker_end)
    block_lines.append("\n")
    new = existing.rstrip() + "\n\n" + "".join(block_lines)
    cp.write_text(new, encoding="utf-8")
    return cp


def verify_checksums(dataset_id: str | None = None) -> dict[str, bool]:
    """Verify all manifest files for ``dataset_id`` (or every manifest)."""
    if dataset_id is None:
        ids = sorted(p.parent.name for p in data_root().glob("*/manifest.json"))
    else:
        ids = [dataset_id]
    return {did: manifest_matches_disk(did) for did in ids}


__all__ = [
    "DownloadError",
    "DatasetManifest",
    "FileEntry",
    "checksums_path",
    "data_root",
    "dataset_dir",
    "download_url",
    "extract_zip",
    "manifest_path",
    "manifest_matches_disk",
    "processed_dir",
    "raw_dir",
    "read_manifest",
    "read_xlsx",
    "sha256_bytes",
    "sha256_file",
    "update_checksums_txt",
    "verify_checksums",
    "write_manifest",
]
