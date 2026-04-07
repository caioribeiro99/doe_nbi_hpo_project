#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _ensure_file(expected: Path, candidates: list[Path], *, required: bool = True) -> None:
    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists():
        return
    found = _first_existing(candidates)
    if found is not None:
        shutil.copy2(found, expected)
        print(f"copied: {found} -> {expected}")
        return
    if required:
        raise FileNotFoundError(
            f"Required dataset not found for expected path {expected}. "
            f"Checked candidates: {[str(c) for c in candidates]}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare local dataset layout expected by the 3-base finance fairness suite.")
    p.parse_args()

    _ensure_file(
        REPO_ROOT / "data/source/bank/bank-additional-full.csv",
        [
            REPO_ROOT / "data/source/bank/bank-additional-full.csv",
            REPO_ROOT / "data/source/bank-additional-full.csv",
            REPO_ROOT / "data/source/fairness/bank-additional-full.csv",
            REPO_ROOT / "bank-additional-full.csv",
            REPO_ROOT.parent / "bank-additional-full.csv",
            REPO_ROOT.parent.parent / "bank-additional-full.csv",
        ],
    )

    _ensure_file(
        REPO_ROOT / "data/source/fairness/german_credit.csv",
        [
            REPO_ROOT / "data/source/fairness/german_credit.csv",
            REPO_ROOT / "data/source/german_credit.csv",
            REPO_ROOT / "german_credit.csv",
            REPO_ROOT.parent / "german_credit.csv",
            REPO_ROOT.parent.parent / "german_credit.csv",
        ],
    )

    _ensure_file(
        REPO_ROOT / "data/source/fairness/credit_card_default.csv",
        [
            REPO_ROOT / "data/source/fairness/credit_card_default.csv",
            REPO_ROOT / "data/source/credit_card_default.csv",
            REPO_ROOT / "credit_card_default.csv",
            REPO_ROOT.parent / "credit_card_default.csv",
            REPO_ROOT.parent.parent / "credit_card_default.csv",
        ],
    )

    print("✅ Dataset layout ready for 3-base finance suite.")


if __name__ == "__main__":
    main()
