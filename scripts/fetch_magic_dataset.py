#!/usr/bin/env python
"""Download UCI MAGIC Gamma Telescope and verify SHA-256.

The dataset itself is not committed to the repository; this script
fetches it and saves it under ``data/source/magic.csv`` for use by the
article-track configurations.

Usage:
    python scripts/fetch_magic_dataset.py
    python scripts/fetch_magic_dataset.py --out data/source/magic.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
EXPECTED_SHA256 = "f44a90e0fb1b6df30bd3d8c2b06e2dd66c8a64b94f97f88158f37d4988ccca35"  # placeholder; verify on first download


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/source/magic.csv"),
        help="Destination path (default: data/source/magic.csv).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 verification (do this only when intentionally updating the checksum).",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MAGIC from {URL} ...")
    try:
        urllib.request.urlretrieve(URL, args.out)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 2

    sha = _sha256(args.out)
    print(f"saved {args.out} ({args.out.stat().st_size} bytes), sha256={sha}")
    if args.no_verify:
        return 0
    if sha != EXPECTED_SHA256:
        print(
            f"WARNING: SHA-256 mismatch. expected={EXPECTED_SHA256} actual={sha}\n"
            "Update data/source/CHECKSUMS.txt or pass --no-verify after manually validating.",
            file=sys.stderr,
        )
        return 1
    print("Checksum OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
