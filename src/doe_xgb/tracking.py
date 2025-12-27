from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


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
    out_dir: Path,
    *,
    replica: int,
    seed: int,
    dataset_path: Path,
    design_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replica": replica,
        "seed": seed,
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path) if dataset_path.exists() else None,
        },
        "design": {
            "path": str(design_path),
            "sha256": sha256_file(design_path) if design_path.exists() else None,
        },
    }
    if extra:
        manifest.update(extra)

    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
