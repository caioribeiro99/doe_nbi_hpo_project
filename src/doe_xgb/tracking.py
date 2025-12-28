from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


PathLike = Union[str, Path]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_name(s: str) -> str:
    """Simple slug for folder/file naming."""
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "."):
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "item"


def build_replica_dir(
    experiments_root: Path,
    dataset_path: Path,
    design_path: Path,
    replica: int,
) -> Path:
    ds = safe_name(dataset_path.stem)
    design = safe_name(design_path.stem)
    rep = f"replica_{replica:02d}"
    out = experiments_root / ds / design / rep
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_manifest(
    path: PathLike,
    *,
    replica: Optional[int] = None,
    seed: Optional[int] = None,
    dataset_path: Optional[Path] = None,
    design_path: Optional[Path] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Flexible manifest writer.

    Supports:
      1) Experiment-level manifest (only `path` + `extra`)
      2) Replica-level manifest (with replica/seed/dataset/design metadata)
    """

    p = Path(path)

    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Replica-level metadata (optional)
    if replica is not None:
        manifest["replica"] = int(replica)
    if seed is not None:
        manifest["seed"] = int(seed)

    if dataset_path is not None:
        manifest["dataset"] = {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path) if dataset_path.exists() else None,
        }

    if design_path is not None:
        manifest["design"] = {
            "path": str(design_path),
            "sha256": sha256_file(design_path) if design_path.exists() else None,
        }

    # Extra free-form metadata
    if extra:
        manifest.update(dict(extra))

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
