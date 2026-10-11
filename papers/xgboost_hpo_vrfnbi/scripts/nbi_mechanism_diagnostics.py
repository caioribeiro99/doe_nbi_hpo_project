#!/usr/bin/env python
"""Per-unit NBI mechanism table: one row per NBI arm-unit (4 datasets x 30 x NBI-S/NBI-R).

Collects, for every NBI-S and NBI-R run of the confirmatory campaign, the solver and
payoff-matrix diagnostics the arm recorded (payoff-matrix rank and condition number,
anchor coincidence, certified fraction, maximum equality residual, maximum rounding
displacement on the fitted surfaces, distinct realized configurations, dominated shares,
largest t, first quasi-normal component), the row-wise range of the recorded payoff
matrix, the CHIM extent (Euclidean length of the segment between the two anchor images,
the distance between the columns of the recorded payoff matrix), the unit's NBI-R/NBI-S
extent ratio, the arm's CORE hypervolume ratio and the unit's NBI-R minus NBI-S CORE gap.
For NBI-R it also records whether each empirical anchor (empirical_anchors.json x_star,
realized coded) is among the arm's realized candidates, with the nearest coded distance
(maximum absolute coordinate difference, the "_linf" columns); "among" means within
1e-9, the 9-decimal identity the arms use to count distinct configurations. NBI-S anchor
vectors were not persisted, so its anchor columns are empty. Unit-level columns (extent
ratio, HV gap) repeat on both rows of a unit.

Every recorded value is checked against its recomputation from the stored candidates,
and the per-dataset medians are asserted equal to the committed
analysis/nbi_r_mechanism.json and analysis/nbi_r_chim_collapse.json; the CORE hypervolume
ratios are asserted equal to analysis/indicators_long.csv.

Provenance label: retrospective (recorded diagnostics); anchor containment and CHIM extent post hoc.

Inputs (read-only): experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/
nbi_s.json, nbi_r.json, empirical_anchors.json and metrics_by_method.json;
papers/xgboost_hpo_vrfnbi/analysis/nbi_r_mechanism.json, nbi_r_chim_collapse.json and
indicators_long.csv (checks only). Writes analysis/nbi_mechanism_per_unit.csv.

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage: python papers/xgboost_hpo_vrfnbi/scripts/nbi_mechanism_diagnostics.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.arms import EQUALITY_TOLERANCE, N_WEIGHTS  # noqa: E402
from doe_xgb.campaign.runner import DATASETS, N_REPLICATIONS  # noqa: E402

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
ANALYSIS = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis"
OUT = ANALYSIS / "nbi_mechanism_per_unit.csv"
ARMS = (("NBI-S", "nbi_s"), ("NBI-R", "nbi_r"))
SAME = 1e-9           # 9-decimal identity of realized coded configurations (arms.py)
TOL = 1e-12
COLUMNS = ["dataset", "replication", "arm", "n_candidates", "payoff_rank",
           "payoff_condition_number", "anchors_coincide", "certified_fraction",
           "max_equality_residual", "max_realization_displacement",
           "distinct_realized_configurations", "payoff_span_quality", "payoff_span_cost",
           "dominated_share_of_returned_set", "dominated_share_among_certified", "t_max",
           "quasi_normal_quality", "chim_extent", "chim_extent_ratio_nbi_r_over_nbi_s",
           "core_hv_ratio", "core_hv_gap_nbi_r_minus_nbi_s",
           "empirical_anchor_quality_returned", "empirical_anchor_quality_nearest_coded_linf",
           "empirical_anchor_cost_returned", "empirical_anchor_cost_nearest_coded_linf"]
# committed analysis/nbi_r_mechanism.json field -> column whose per-dataset median it is
MECHANISM_FIELDS = {"cond": "payoff_condition_number", "payoff_span_q": "payoff_span_quality",
                    "payoff_span_c": "payoff_span_cost", "certified": "certified_fraction",
                    "dom_all": "dominated_share_of_returned_set",
                    "dom_cert": "dominated_share_among_certified", "t_hi": "t_max",
                    "maxres": "max_equality_residual",
                    "maxdisp": "max_realization_displacement",
                    "distinct": "distinct_realized_configurations",
                    "qn0": "quasi_normal_quality"}


def arm_row(ds: str, rep: int, arm: str, run: dict) -> dict:
    """Recorded diagnostics of one NBI arm-unit, each checked against the candidates."""
    dg, cands = run["diagnostics"], run["candidates"]
    phi = np.asarray(dg["payoff_matrix"], dtype=float)       # columns F*_j - utopia
    cert = np.mean([bool(c["solver_success"]) and c["equality_residual"] is not None
                    and c["equality_residual"] < EQUALITY_TOLERANCE for c in cands])
    distinct = len({tuple(np.round(np.asarray(c["x_realized_coded"], float), 9))
                    for c in cands})
    assert len(cands) == N_WEIGHTS == dg["n_weights"], (ds, rep, arm)
    assert round(float(cert), 4) == dg["certified_fraction"], (ds, rep, arm)
    assert (max(c["equality_residual"] for c in cands)
            == dg["max_equality_residual"]), (ds, rep, arm)
    assert (max(c["realization_displacement"] for c in cands)
            == dg["max_realization_displacement"]), (ds, rep, arm)
    assert distinct == dg["distinct_realized_configurations"], (ds, rep, arm)
    assert int(np.linalg.matrix_rank(phi)) == dg["payoff_rank"], (ds, rep, arm)
    cond = dg["payoff_condition_number"]          # recorded as null when Phi is rank-deficient
    assert (cond is None) == (dg["payoff_rank"] < phi.shape[0]), (ds, rep, arm)
    assert cond is None or abs(float(np.linalg.cond(phi)) - cond) < 1e-9 * max(1.0, cond), (
        ds, rep, arm)
    dom_cert = dg["dominated_share_among_certified"]
    return {"dataset": ds, "replication": rep, "arm": arm, "n_candidates": len(cands),
            "payoff_rank": int(dg["payoff_rank"]),
            "payoff_condition_number": float(cond) if cond is not None else float("nan"),
            "anchors_coincide": bool(dg["anchors_coincide"]),
            "certified_fraction": float(dg["certified_fraction"]),
            "max_equality_residual": float(dg["max_equality_residual"]),
            "max_realization_displacement": float(dg["max_realization_displacement"]),
            "distinct_realized_configurations": int(dg["distinct_realized_configurations"]),
            "payoff_span_quality": float(np.ptp(phi[0])),
            "payoff_span_cost": float(np.ptp(phi[1])),
            "dominated_share_of_returned_set": float(dg["dominated_share_of_returned_set"]),
            "dominated_share_among_certified": (float(dom_cert) if dom_cert is not None
                                                else float("nan")),
            "t_max": float(dg["t_range"][1]),
            "quasi_normal_quality": float(dg["quasi_normal"][0]),
            "chim_extent": float(np.linalg.norm(phi[:, 0] - phi[:, 1]))}


def anchor_containment(run: dict, anchors: dict) -> dict:
    """For each empirical anchor: returned by NBI-R, and nearest coded distance."""
    xs = np.asarray(anchors["x_star"], dtype=float)                # (2, 7), realized coded
    cx = np.asarray([c["x_realized_coded"] for c in run["candidates"]], dtype=float)
    out = {}
    for j, name in enumerate(("quality", "cost")):
        d = float(np.abs(cx - xs[j]).max(axis=1).min())
        out[f"empirical_anchor_{name}_returned"] = bool(d <= SAME)
        out[f"empirical_anchor_{name}_nearest_coded_linf"] = d
    return out


def collect() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for rep in range(N_REPLICATIONS):
            u = ROOT / ds / f"rep_{rep:02d}"
            runs = {key: json.loads((u / f"{key}.json").read_text()) for _, key in ARMS}
            mbm = json.loads((u / "metrics_by_method.json").read_text())["methods"]
            ea = json.loads((u / "empirical_anchors.json").read_text())
            # NBI-R's recorded payoff matrix is the anchor search's own measurements.
            f_star = np.asarray(ea["F_star"], dtype=float)
            assert np.abs(np.asarray(runs["nbi_r"]["diagnostics"]["payoff_matrix"])
                          - (f_star - np.diag(f_star)[:, None])).max() < TOL, (ds, rep)
            unit = {label: arm_row(ds, rep, label, runs[key]) for label, key in ARMS}
            hv = {label: float(mbm[key]["core"]["hv_ratio"]) for label, key in ARMS}
            ratio = unit["NBI-R"]["chim_extent"] / unit["NBI-S"]["chim_extent"]
            gap = hv["NBI-R"] - hv["NBI-S"]
            for label, key in ARMS:
                row = {**unit[label], "chim_extent_ratio_nbi_r_over_nbi_s": ratio,
                       "core_hv_ratio": hv[label], "core_hv_gap_nbi_r_minus_nbi_s": gap}
                if key == "nbi_r":
                    row.update(anchor_containment(runs[key], ea))
                rows.append(row)
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    order = {"dataset": {ds: i for i, ds in enumerate(DATASETS)},
             "arm": {label: i for i, (label, _) in enumerate(ARMS)}}
    return (df.sort_values(["dataset", "replication", "arm"],
                           key=lambda s: s.map(order[s.name]) if s.name in order else s)
            .reset_index(drop=True))


def cross_check(df: pd.DataFrame) -> float:
    """Largest absolute difference from the committed per-dataset artifacts (asserted 0)."""
    mech = json.loads((ANALYSIS / "nbi_r_mechanism.json").read_text())
    chim = json.loads((ANALYSIS / "nbi_r_chim_collapse.json").read_text())
    ind = pd.read_csv(ANALYSIS / "indicators_long.csv")
    worst = 0.0
    for ds in DATASETS:
        for label, key in ARMS:
            sub = df[(df.dataset == ds) & (df.arm == label)]
            for field, col in MECHANISM_FIELDS.items():
                worst = max(worst, abs(float(sub[col].median()) - mech[ds][field][key]))
            ref = (ind[(ind.dataset == ds) & (ind.entity == label)]
                   .set_index("replication")["core__hv_ratio"])
            worst = max(worst, float(np.abs(sub.set_index("replication")["core_hv_ratio"]
                                            - ref.loc[sub.replication]).max()))
        s = df[(df.dataset == ds) & (df.arm == "NBI-S")].set_index("replication")
        r = df[(df.dataset == ds) & (df.arm == "NBI-R")].set_index("replication")
        er, gap = r["chim_extent_ratio_nbi_r_over_nbi_s"], r["core_hv_gap_nbi_r_minus_nbi_s"]
        c = chim[ds]
        assert c["n"] == len(r) == N_REPLICATIONS, ds
        for mine, theirs in ((s["chim_extent"].median(), c["chim_extent_nbi_s_median"]),
                             (r["chim_extent"].median(), c["chim_extent_nbi_r_median"]),
                             (er.median(), c["extent_ratio_median"]),
                             (er.min(), c["extent_ratio_range"][0]),
                             (er.max(), c["extent_ratio_range"][1]),
                             (gap.median(), c["hv_gap_median"]),
                             (spearmanr(er, gap).statistic,
                              c["spearman_extent_ratio_vs_hv_gap"])):
            worst = max(worst, abs(float(mine) - float(theirs)))
    assert worst < TOL, f"disagreement with a committed artifact: {worst:.3e}"
    return worst


def main() -> int:
    df = collect()
    worst = cross_check(df)
    df.to_csv(OUT, index=False)

    r = df[df.arm == "NBI-R"]
    near = pd.concat([r["empirical_anchor_quality_nearest_coded_linf"],
                      r["empirical_anchor_cost_nearest_coded_linf"]])
    returned = int(r["empirical_anchor_quality_returned"].sum()
                   + r["empirical_anchor_cost_returned"].sum())
    off = df[df.distinct_realized_configurations != N_WEIGHTS]
    print(f"arm-units {len(df)}; payoff rank 2 in {int((df.payoff_rank == 2).sum())}; "
          f"anchors coincide in {int(df.anchors_coincide.sum())}")
    print(f"{N_WEIGHTS} distinct realized configurations in {len(df) - len(off)} of {len(df)}; "
          "exceptions: " + ", ".join(f"{x.dataset} rep_{x.replication:02d} {x.arm} "
                                     f"({x.distinct_realized_configurations})"
                                     for x in off.itertuples()))
    print(f"certified fraction median {df.certified_fraction.median():.3f}, below 1: "
          + ", ".join(f"{x.dataset} rep_{x.replication:02d} {x.arm} {x.certified_fraction:.3f}"
                      for x in df[df.certified_fraction < 1].itertuples()))
    print(f"max equality residual: median {df.max_equality_residual.median():.2e}, "
          f"max {df.max_equality_residual.max():.2e}")
    print(f"empirical anchors returned by NBI-R: {returned} of {len(near)}; nearest coded "
          f"distance min {near.min():.3f}, median {near.median():.3f}")
    print(f"largest difference from nbi_r_mechanism.json, nbi_r_chim_collapse.json and "
          f"indicators_long.csv: {worst:.1e}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
