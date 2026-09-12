#!/usr/bin/env python
"""Numerical claim audit: every number in the manuscript must exist in a source artifact.

Extracts each sentence of the compiled PDF that carries a quantitative claim, pulls the
numbers out of it, and checks each against the union of values present in the frozen R = 30
tables, the statistics outputs and the NSGA-II tables. A number that appears nowhere in any
artifact is flagged for manual resolution.

This is a screen, not a proof: it catches transcription drift and invented figures, not a
number that is real but attached to the wrong claim. Matching is by value with tolerance, so
a flagged item may still be legitimate (a ratio computed in prose, a count of datasets, a
year). Every flag must be resolved by hand and recorded in claims_and_evidence.md.

Usage:  python audit_numbers.py [--pdf /tmp/texout/main.pdf]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REP = REPO / "reports" / "pco213_postwork_benchmark"
LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "—": "-", "–": "-", "×": "x"}

# numbers a reader would not expect to find in a results table
WHITELIST = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 24, 25, 30, 40, 50,
    66, 80, 90, 95, 100, 120, 200, 400,           # protocol constants and counts
    1958, 1963, 1979, 1996, 1998, 1999, 2000, 2002, 2003, 2004, 2005, 2006, 2007, 2009,
    2010, 2011, 2013, 2014, 2015, 2016, 2017, 2019, 2020, 2021, 2022, 2024, 2025, 2026,
    0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.1,     # thresholds and conventions
}

# offsets documented in the seed policy, and derived quantities stated in prose
SEED_LIKE = {20260904, 20260906, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
             10000, 12000, 13000, 770000, 21030904}


def artifact_values() -> np.ndarray:
    """Every numeric value present in any committed source artifact (CSV or JSON)."""
    import json as _json

    vals: list[float] = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            vals.append(float(o))

    for f in [REP / "summary.json"] + sorted((REP / "manifests").glob("*.json")) + \
             sorted((REP / "nsga2").glob("*.json")) + \
             [Path(__file__).parent / "nsga2_preregistered_config.json"]:
        if f.exists():
            try:
                walk(_json.loads(f.read_text()))
            except Exception:
                pass
    globs = [REP / "tables", REP / "statistics", REP / "nsga2",
             Path(__file__).parent / "tables"]
    for g in globs:
        if not g.exists():
            continue
        for f in sorted(g.glob("*.csv")):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            for c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                vals.extend(s.tolist())
    return np.unique(np.asarray(vals, dtype=float))


def sentences(pdf: Path) -> list[str]:
    import pymupdf

    d = pymupdf.open(pdf)
    t = "".join(p.get_text() for p in d)
    for k, v in LIG.items():
        t = t.replace(k, v)
    t = t.split("\nReferences\n")[0]
    t = re.sub(r"\s+", " ", t)
    # join thousands separators so "91,456" is one number, not "91" and "456"
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    return [s.strip() for s in re.split(r"(?<=[.!?]) ", t) if s.strip()]


NUM = re.compile(r"(?<![A-Za-z0-9._-])(\d+(?:\.\d+)?)(?![A-Za-z0-9._-])")
# a bare x/y count, NOT part of a win/tie/loss triple x/y/z and not a date
FRAC = re.compile(r"(?<![\d/])(\d{1,3})\s*/\s*(\d{1,3})(?![\d/])")
WTL = re.compile(r"\b\d{1,3}/\d{1,3}/\d{1,3}\b")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="/tmp/texout/main.pdf")
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args()

    vals = artifact_values()
    print(f"source values loaded: {len(vals):,} distinct numbers from "
          f"{sum(1 for g in [REP/'tables', REP/'statistics', REP/'nsga2'] if g.exists())} artifact directories")

    flagged, checked, frac_claims = [], 0, 0
    for s in sentences(Path(args.pdf)):
        nums = [float(x) for x in NUM.findall(s)]
        if not nums:
            continue
        # x/30, x/20, x/10 style counts: check the denominator is a real replication count
        for a, b in FRAC.findall(WTL.sub(" ", s)):
            frac_claims += 1
            if int(b) not in (5, 7, 8, 10, 12, 20, 23, 24, 30, 40, 50, 66, 120):
                flagged.append(("odd-denominator", f"{a}/{b}", s[:160]))
        for n in nums:
            if n in WHITELIST or (n.is_integer() and abs(n) <= 30):
                continue
            if n.is_integer() and 1900 <= n <= 2030:      # citation years
                continue
            if n.is_integer() and n in SEED_LIKE:          # documented seed offsets
                continue
            checked += 1
            rel = np.abs(vals - n)
            tol = max(args.tol, abs(n) * 1e-3)
            if not (rel <= tol).any():
                flagged.append(("not-in-artifacts", f"{n}", s[:160]))

    print(f"quantitative sentences scanned; {checked} non-trivial numbers checked; "
          f"{frac_claims} x/y claims checked")
    print(f"FLAGGED: {len(flagged)}")
    seen = set()
    for kind, val, ctx in flagged:
        key = (kind, val)
        if key in seen:
            continue
        seen.add(key)
        print(f"  [{kind}] {val}\n      ...{ctx}...")
    print(f"\n{len(seen)} distinct flags to resolve by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
