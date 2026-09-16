#!/usr/bin/env python
"""The frozen primary statistical family, and the boundary-control analysis.

Implements EXPERIMENT_PROTOCOL.md sections 10 and 11.1 exactly as frozen:

  * primary indicator   : hv_ratio against the CORE reference
  * primary family      : the three identifying contrasts, HISTORICAL-WS -> WS-S,
                          WS-S -> NBI-S, NBI-S -> NBI-R
  * correction          : Holm WITHIN each dataset; NO pooling across datasets
  * primary evidence    : the descriptive triple -- median paired difference with a
                          percentile bootstrap interval, win fraction with a Wilson
                          interval, matched-pairs rank-biserial correlation
  * the test            : SECONDARY. Wilcoxon signed-rank, and the Nadeau-Bengio
                          corrected resampled t as a further sensitivity
  * panel               : MAGIC, Adult, Bank Marketing carry the primary family;
                          Spambase is reported separately as the pre-specified
                          boundary geometry control
  * sensitivity         : every comparison repeated against the AUGMENTED reference

Sign convention: a paired difference is (second arm - first arm). For hv_ratio,
larger is better, so a POSITIVE difference favours the second arm named in the
contrast.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.runner import (BOUNDARY_CONTROLS, DATASETS,  # noqa: E402
                                     N_REPLICATIONS, PRIMARY_GEOMETRY_PANEL)

ANALYSIS = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis"
BOOT = 10000
SEED = 20260915

# The primary family, in the order protocol/method_arms.md declares them.
PRIMARY_CONTRASTS = [("HISTORICAL-WS", "WS-S", "normalization / historical reconstruction"),
                     ("WS-S", "NBI-S", "front-construction geometry"),
                     ("NBI-S", "NBI-R", "anchor / payoff provenance")]
SECONDARY_CONTRASTS = [("HISTORICAL-WS-asrun", "HISTORICAL-WS", "shared specification vs bit-faithful"),
                       ("NBI-S", "ANCHOR-INJECTION-CONTROL", "set composition at fixed geometry"),
                       ("NBI-S", "GRID", "vs evaluation-matched grid"),
                       ("NBI-S", "RANDOM", "vs evaluation-matched random"),
                       ("NBI-S", "NSGA2-MATCHED", "vs evaluation-matched NSGA-II")]
R_ = N_REPLICATIONS
TEST_TRAIN = 0.25          # the protocol's 80/20 outer split


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_median_ci(x: np.ndarray, rng, reps: int = BOOT) -> tuple[float, float]:
    idx = rng.integers(0, len(x), (reps, len(x)))
    meds = np.median(x[idx], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial: (W+ - W-) / sum of ranks, ties excluded."""
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    r = stats.rankdata(np.abs(nz))
    tot = r.sum()
    return float((r[nz > 0].sum() - r[nz < 0].sum()) / tot)


def nadeau_bengio_t(d: np.ndarray) -> dict:
    """SE_corr^2 = (1/R + n_test/n_train) s^2. Reported as a sensitivity only."""
    n = len(d)
    s2 = float(np.var(d, ddof=1))
    se = float(np.sqrt((1.0 / n + TEST_TRAIN) * s2))
    if se == 0:
        return {"t": float("nan"), "p": float("nan"), "se_corrected": 0.0,
                "se_inflation": float("nan")}
    t = float(np.mean(d) / se)
    p = float(2 * stats.t.sf(abs(t), df=n - 1))
    naive = float(np.sqrt(s2 / n))
    return {"t": t, "p": p, "se_corrected": se,
            "se_inflation": float(se / naive) if naive else float("nan")}


def compare(df: pd.DataFrame, ds: str, a: str, b: str, ref: str, rng) -> dict:
    col = f"{ref}__hv_ratio"
    xa = (df[(df.dataset == ds) & (df.entity == a)].sort_values("replication")[col]
          .to_numpy(float))
    xb = (df[(df.dataset == ds) & (df.entity == b)].sort_values("replication")[col]
          .to_numpy(float))
    assert len(xa) == len(xb) == R_, f"{ds} {a}->{b}: {len(xa)} vs {len(xb)} pairs"
    d = xb - xa                                     # positive favours b
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    ties = int((d == 0).sum())
    lo, hi = boot_median_ci(d, rng)
    wlo, whi = wilson(wins, R_)
    try:
        w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        wp = float(w.pvalue)
    except ValueError:
        wp = float("nan")
    return {"dataset": ds, "reference": ref, "contrast": f"{a} -> {b}",
            "first": a, "second": b, "n_pairs": R_,
            "median_a": float(np.median(xa)), "median_b": float(np.median(xb)),
            "median_diff": float(np.median(d)),
            "median_diff_ci95": [lo, hi],
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "mean_diff": float(np.mean(d)),
            "wins_for_second": wins, "losses": losses, "ties": ties,
            "win_fraction": wins / R_, "win_fraction_ci95": [wlo, whi],
            "win_ci_excludes_half": bool(whi < 0.5 or wlo > 0.5),
            "rank_biserial": rank_biserial(d),
            "wilcoxon_p": wp,
            "nadeau_bengio": nadeau_bengio_t(d)}


def holm(rows: list[dict], key: str = "wilcoxon_p") -> list[dict]:
    """Holm-Bonferroni WITHIN a dataset family."""
    order = sorted(range(len(rows)), key=lambda i: (np.isnan(rows[i][key]), rows[i][key]))
    m = len(rows)
    prev = 0.0
    for rank, i in enumerate(order):
        p = rows[i][key]
        adj = float("nan") if np.isnan(p) else min(1.0, max(prev, (m - rank) * p))
        rows[i]["holm_p"] = adj
        rows[i]["holm_significant_at_05"] = bool(adj < 0.05) if adj == adj else False
        if adj == adj:
            prev = adj
    return rows


def main() -> int:
    df = pd.read_csv(ANALYSIS / "indicators_long.csv")
    rng = np.random.default_rng(SEED)
    out = {"primary_family": {}, "boundary_control": {}, "secondary": {},
           "convention": {
               "indicator": "hv_ratio", "primary_reference": "core",
               "sensitivity_reference": "augmented",
               "sign": "difference is (second - first); positive favours the second arm",
               "correction": "Holm within each dataset; no pooling across datasets",
               "primary_evidence": ("median paired difference with percentile bootstrap "
                                    "interval, win fraction with Wilson interval, "
                                    "matched-pairs rank-biserial correlation"),
               "test_status": "secondary; Nadeau-Bengio corrected t a further sensitivity",
               "bootstrap_resamples": BOOT, "seed": SEED}}

    for ref in ("core", "augmented"):
        for ds in PRIMARY_GEOMETRY_PANEL:
            rows = [compare(df, ds, a, b, ref, rng) for a, b, _ in PRIMARY_CONTRASTS]
            for r, (_, _, mech) in zip(rows, PRIMARY_CONTRASTS):
                r["mechanism"] = mech
            out["primary_family"].setdefault(ref, {})[ds] = holm(rows)
        for ds in BOUNDARY_CONTROLS:
            rows = [compare(df, ds, a, b, ref, rng) for a, b, _ in PRIMARY_CONTRASTS]
            for r, (_, _, mech) in zip(rows, PRIMARY_CONTRASTS):
                r["mechanism"] = mech
            out["boundary_control"].setdefault(ref, {})[ds] = holm(rows)
        for ds in DATASETS:
            rows = [compare(df, ds, a, b, ref, rng) for a, b, _ in SECONDARY_CONTRASTS]
            for r, (_, _, mech) in zip(rows, SECONDARY_CONTRASTS):
                r["mechanism"] = mech
            out["secondary"].setdefault(ref, {})[ds] = rows   # descriptive, no correction

    (ANALYSIS / "primary_analysis.json").write_text(json.dumps(out, indent=2))

    print("PRIMARY FAMILY — hv_ratio against the CORE reference, Holm within dataset\n")
    hdr = f"{'dataset':15} {'contrast':26} {'med Δ':>9} {'95% CI':>20} {'win':>7} {'rb':>6} {'holm p':>8}"
    print(hdr); print("-" * len(hdr))
    for ds in PRIMARY_GEOMETRY_PANEL:
        for r in out["primary_family"]["core"][ds]:
            ci = f"[{r['median_diff_ci95'][0]:+.4f},{r['median_diff_ci95'][1]:+.4f}]"
            print(f"{ds:15} {r['contrast']:26} {r['median_diff']:+9.4f} {ci:>20} "
                  f"{r['wins_for_second']:3}/{R_:<3} {r['rank_biserial']:+6.2f} "
                  f"{r['holm_p']:8.4f}")
    print("\nBOUNDARY CONTROL — Spambase, reported separately, not in the primary family\n")
    for ds in BOUNDARY_CONTROLS:
        for r in out["boundary_control"]["core"][ds]:
            ci = f"[{r['median_diff_ci95'][0]:+.4f},{r['median_diff_ci95'][1]:+.4f}]"
            print(f"{ds:15} {r['contrast']:26} {r['median_diff']:+9.4f} {ci:>20} "
                  f"{r['wins_for_second']:3}/{R_:<3} {r['rank_biserial']:+6.2f} "
                  f"{r['holm_p']:8.4f}")
    print(f"\nwrote {ANALYSIS/'primary_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
