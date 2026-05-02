"""Base helper for the per-dataset fetch scripts.

Every ``scripts/fetch_<id>_dataset.py`` defines a dataset-specific
``process(raw_dir) -> path-to-processed-csv`` callable and delegates
the rest to :func:`run_fetcher` here.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb.datasets.download import (  # noqa: E402
    DownloadError,
    download_url,
    manifest_matches_disk,
    processed_dir,
    raw_dir,
    update_checksums_txt,
    write_manifest,
)
from doe_xgb.datasets.registry import REGISTRY  # noqa: E402

ProcessFn = Callable[[Path, Path], list[Path]]
"""Process callable: (raw_dir, processed_dir) -> list of processed file paths."""


def run_fetcher(
    *,
    dataset_id: str,
    raw_url: str | None,
    raw_filename: str | None,
    process_fn: ProcessFn,
    notes: str | None = None,
    extra_raw_files: list[Path] | None = None,
    argv: list[str] | None = None,
) -> int:
    """Standard fetch entry point.

    Parameters
    ----------
    dataset_id:
        Key in :data:`doe_xgb.datasets.registry.REGISTRY`.
    raw_url:
        URL to download; ``None`` means the script's ``process_fn``
        produces the raw files itself (e.g. via OpenML).
    raw_filename:
        Local filename inside ``raw/`` (defaults to URL basename).
    process_fn:
        Callable that converts the raw files into one or more
        normalized files and returns the list of processed paths.
    notes:
        Free-form notes recorded in the manifest.
    """
    parser = argparse.ArgumentParser(description=f"Fetch the {dataset_id!r} dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the local cache and checksums match.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip the URL download; useful when the raw files are already in place.",
    )
    args = parser.parse_args(argv)

    if dataset_id not in REGISTRY:
        print(f"unknown dataset id: {dataset_id}", file=sys.stderr)
        return 2
    meta = REGISTRY[dataset_id]

    rd = raw_dir(dataset_id)
    pd_dir = processed_dir(dataset_id)
    rd.mkdir(parents=True, exist_ok=True)
    pd_dir.mkdir(parents=True, exist_ok=True)

    raw_files: list[Path] = []
    if raw_url is not None and not args.no_network:
        target_name = raw_filename or raw_url.rsplit("/", 1)[-1]
        target = rd / target_name
        try:
            download_url(raw_url, target, force=args.force)
        except DownloadError as e:
            print(f"download failed for {dataset_id}: {e}", file=sys.stderr)
            return 1
        raw_files.append(target)
    if extra_raw_files:
        raw_files.extend(extra_raw_files)

    if manifest_matches_disk(dataset_id) and not args.force:
        print(f"{dataset_id}: manifest checksums match; nothing to do.")
        return 0

    try:
        processed_files = process_fn(rd, pd_dir)
    except FileNotFoundError as e:
        print(f"{dataset_id}: missing input file -- {e}", file=sys.stderr)
        return 1
    except Exception as e:  # pragma: no cover - defensive
        print(f"{dataset_id}: processing failed -- {e}", file=sys.stderr)
        return 1

    raw_observed = sorted(rd.glob("*"))
    write_manifest(
        dataset_id,
        raw_files=raw_observed if raw_observed else raw_files,
        processed_files=processed_files,
        source_url=meta.source_url,
        openml_id=meta.openml_id,
        target_column=meta.target_column or "",
        target_transform=meta.target_transform,
        notes=notes or meta.notes,
    )
    update_checksums_txt(dataset_id)
    print(f"{dataset_id}: wrote manifest and updated CHECKSUMS.txt.")
    return 0
