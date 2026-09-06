#!/usr/bin/env python3
"""Replication-level statistical analysis of the PCO213 post-work benchmark
(R = 30 robustness extension; also runnable on the R = 10 tables).

Statistical unit: one outer replication (train/holdout partition) of one
dataset. Within a dataset the R partitions overlap, so they measure
sensitivity to data partitioning, not independent samples; the DATASET is
the unit for cross-dataset generalization. Nothing here pools the 4 x R
replications as independent observations.

Primary pre-specified comparisons (paired by replication, weighted cost):
  NBI-B vs NBI-A, NBI-C vs NBI-B, NBI-C vs random scalarization,
  NBI-C vs budget-matched random Dirichlet search.
Primary endpoints: IGD+ (lower is better) and hypervolume ratio (higher).
Sign convention: delta > 0 always means the SECOND-named (new) method is
better: dHV = HV_new - HV_ref ; dIGD+ = IGD+_ref - IGD+_new.

Per dataset x comparison x endpoint: paired-difference summaries, percentile
bootstrap 95% CI (Monte Carlo stability under the observed resampling
process, not population sampling), Wilson/Jeffreys intervals for
proportions, win/tie/loss, matched-pairs rank-biserial effect size,
Nadeau-Bengio corrected resampled t (rho = n_test/n_train = 0.25) with Holm
correction within the dataset family, plus Wilcoxon signed-rank as a
distribution-free check. Everything else is descriptive.

Reads  <reports>/tables/*.csv (from pco213_postwork_benchmark_report.py) and
       raw per-replication artifacts under <root> for blends and class balance
Writes <reports>/statistics/*.csv and <figures>/r30_*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ["lr", "gnb", "knn", "rf", "xgb"]
DATASETS = ["santander", "bnp", "porto", "uci_credit"]
SETS = ["nbi_A", "nbi_B", "nbi_C", "ws_random_scalarization", "random_dirichlet_budget", "design_runs"]
PRIMARY = [("nbi_B", "nbi_A"), ("nbi_C", "nbi_B"), ("nbi_C", "ws_random_scalarization"), ("nbi_C", "random_dirichlet_budget")]
ENDPOINTS = {"igd_plus": "lower", "hv_ratio": "higher"}
RHO = 0.25          # n_test / n_train for the 80/20 outer split
TIE_TOL = 1e-4      # |delta| below this counts as a tie for win/tie/loss
N_BOOT = 10_000
SEED = 20260906


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def boot_ci(x: np.ndarray, stat=np.mean, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    b = stat(x[idx], axis=1)
    return (float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2))))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def jeffreys(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5) if k < n else 1.0
    return (float(lo), float(hi))


def nadeau_bengio(d: np.ndarray, rho: float = RHO) -> dict:
    """Corrected resampled t-test (Nadeau & Bengio 2003) on paired differences
    over J overlapping resamples; variance inflated by (1/J + rho)."""
    d = np.asarray(d, dtype=float); d = d[np.isfinite(d)]
    J = d.size
    if J < 3:
        return {"t_nb": float("nan"), "p_nb": float("nan"), "df": J - 1}
    m = d.mean(); v = d.var(ddof=1)
    if v == 0:
        return {"t_nb": float("inf") if m != 0 else 0.0, "p_nb": 0.0 if m != 0 else 1.0, "df": J - 1}
    t = m / np.sqrt(v * (1.0 / J + rho))
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return {"t_nb": float(t), "p_nb": float(p), "df": J - 1}


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014): (T+ - T-) / (T+ + T-)."""
    d = np.asarray(d, dtype=float); d = d[np.isfinite(d)]; d = d[np.abs(d) > 0]
    if d.size == 0:
        return float("nan")
    r = stats.rankdata(np.abs(d))
    tp = r[d > 0].sum(); tm = r[d < 0].sum()
    return float((tp - tm) / (tp + tm))


def wilcoxon_p(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float); d = d[np.isfinite(d)]
    if d.size < 5 or np.all(d == 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float); n = len(p)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    adj = np.full(n, np.nan); running = 0.0
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        val = min(1.0, (n - rank) * p[i]); running = max(running, val); adj[i] = running
    return adj.tolist()


def summarize_delta(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float); d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return {"n": 0}
    lo, hi = boot_ci(d); mlo, mhi = boot_ci(d, stat=np.median)
    wins = int((d > TIE_TOL).sum()); losses = int((d < -TIE_TOL).sum()); ties = n - wins - losses
    wl, wh = wilson(wins, n)
    return {"n": n, "mean": float(d.mean()), "median": float(np.median(d)), "sd": float(d.std(ddof=1)) if n > 1 else 0.0,
            "q25": float(np.percentile(d, 25)), "q75": float(np.percentile(d, 75)), "min": float(d.min()), "max": float(d.max()),
            "ci95_mean_lo": lo, "ci95_mean_hi": hi, "ci95_median_lo": mlo, "ci95_median_hi": mhi,
            "wins": wins, "ties": ties, "losses": losses, "win_frac": wins / n, "win_frac_wilson_lo": wl, "win_frac_wilson_hi": wh,
            "rank_biserial": rank_biserial(d), **nadeau_bengio(d), "p_wilcoxon": wilcoxon_p(d)}


def bimodality_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]; n = x.size
    if n < 4:
        return float("nan")
    g = stats.skew(x, bias=False); k = stats.kurtosis(x, bias=False)
    return float((g * g + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def gmm_bic_delta(x: np.ndarray) -> float:
    """BIC(1 component) - BIC(2 components); > 0 favours two components (exploratory)."""
    from sklearn.mixture import GaussianMixture
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)].reshape(-1, 1)
    if x.shape[0] < 8 or np.std(x) == 0:
        return float("nan")
    b1 = GaussianMixture(1, random_state=0).fit(x).bic(x); b2 = GaussianMixture(2, random_state=0, n_init=3).fit(x).bic(x)
    return float(b1 - b2)


# ---------------------------------------------------------------------------
# analysis blocks
# ---------------------------------------------------------------------------

def paired_primary(pq: pd.DataFrame, cost: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, tests = [], []
    for ds in DATASETS:
        fam = []
        for new, ref in PRIMARY:
            for ep, direction in ENDPOINTS.items():
                a = pq[(pq.dataset == ds) & (pq.cost == cost) & (pq.set == new)].set_index("rep")[ep]
                b = pq[(pq.dataset == ds) & (pq.cost == cost) & (pq.set == ref)].set_index("rep")[ep]
                j = a.index.intersection(b.index)
                d = (a.loc[j] - b.loc[j]) if direction == "higher" else (b.loc[j] - a.loc[j])
                s = summarize_delta(d.to_numpy())
                row = {"dataset": ds, "cost": cost, "comparison": f"{new} vs {ref}", "new": new, "ref": ref, "endpoint": ep,
                       "sign_convention": "delta > 0 = new better", **s}
                rows.append(row); fam.append(row)
        pn = holm([r["p_nb"] for r in fam]); pw = holm([r["p_wilcoxon"] for r in fam])
        for r, a1, a2 in zip(fam, pn, pw):
            tests.append({"dataset": ds, "cost": cost, "comparison": r["comparison"], "endpoint": r["endpoint"], "n": r["n"],
                          "mean_delta": r["mean"], "t_nadeau_bengio": r["t_nb"], "df": r["df"], "p_nadeau_bengio": r["p_nb"],
                          "p_nb_holm_family_dataset": a1, "p_wilcoxon": r["p_wilcoxon"], "p_wilcoxon_holm": a2,
                          "rank_biserial": r["rank_biserial"], "win_frac": r["win_frac"], "rho": RHO,
                          "family": f"{ds}: 4 comparisons x 2 endpoints"})
    return pd.DataFrame(rows), pd.DataFrame(tests)


def win_tie_loss(pq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost in ("weighted", "support"):
        for ds in DATASETS:
            for ep, direction in ENDPOINTS.items():
                for s1, s2 in combinations(SETS, 2):
                    a = pq[(pq.dataset == ds) & (pq.cost == cost) & (pq.set == s1)].set_index("rep")[ep]
                    b = pq[(pq.dataset == ds) & (pq.cost == cost) & (pq.set == s2)].set_index("rep")[ep]
                    j = a.index.intersection(b.index)
                    d = (a.loc[j] - b.loc[j]) if direction == "higher" else (b.loc[j] - a.loc[j])
                    d = d.to_numpy(); d = d[np.isfinite(d)]
                    w = int((d > TIE_TOL).sum()); l = int((d < -TIE_TOL).sum()); t = d.size - w - l
                    wl, wh = wilson(w, d.size)
                    rows.append({"cost": cost, "dataset": ds, "endpoint": ep, "set_a": s1, "set_b": s2, "n": d.size,
                                 "a_wins": w, "ties": t, "b_wins": l, "a_win_frac": w / d.size if d.size else np.nan,
                                 "wilson_lo": wl, "wilson_hi": wh, "rank_biserial_a_over_b": rank_biserial(d)})
    return pd.DataFrame(rows)


def reliability_gate(so: pd.DataFrame, pq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel = so[so.selected]
    rows = []
    for ds in DATASETS:
        for resp in ("roc_auc", "log_loss", "brier", "pr_auc"):
            sub = sel[(sel.dataset == ds) & (sel.response == resp)]
            k = int(sub["reliable"].astype(bool).sum()); n = len(sub)
            wl, wh = wilson(k, n); jl, jh = jeffreys(k, n)
            lo, hi = boot_ci(sub["r2_external"].to_numpy())
            rows.append({"dataset": ds, "response": resp, "n": n, "pass": k, "p_pass": k / n if n else np.nan,
                         "wilson_lo": wl, "wilson_hi": wh, "jeffreys_lo": jl, "jeffreys_hi": jh,
                         "r2_external_median": float(sub["r2_external"].median()), "r2_external_mean": float(sub["r2_external"].mean()),
                         "r2_external_ci95_lo": lo, "r2_external_ci95_hi": hi, "spearman_median": float(sub["spearman_external"].median()),
                         "orders": sub["order"].value_counts().to_dict()})
    gate = pd.DataFrame(rows)
    # conditional gain from real anchors (B - A), by gate outcome, per dataset (descriptive; n varies)
    crow = []
    for ds in DATASETS:
        for resp in ("roc_auc", "log_loss"):
            flag = sel[(sel.dataset == ds) & (sel.response == resp)].set_index("rep")["reliable"].astype(bool)
            for cost in ("weighted", "support"):
                for ep, direction in ENDPOINTS.items():
                    a = pq[(pq.dataset == ds) & (pq.cost == cost) & (pq.set == "nbi_B")].set_index("rep")[ep]
                    b = pq[(pq.dataset == ds) & (pq.cost == cost) & (pq.set == "nbi_A")].set_index("rep")[ep]
                    j = a.index.intersection(b.index).intersection(flag.index)
                    d = (a.loc[j] - b.loc[j]) if direction == "higher" else (b.loc[j] - a.loc[j])
                    for cond in (True, False):
                        dd = d[flag.loc[j] == cond].to_numpy()
                        s = summarize_delta(dd) if dd.size else {"n": 0}
                        crow.append({"dataset": ds, "gate_response": resp, "gate_passed": cond, "cost": cost, "endpoint": ep,
                                     "comparison": "nbi_B vs nbi_A", **{k: v for k, v in s.items() if k in ("n", "mean", "median", "sd", "ci95_mean_lo", "ci95_mean_hi", "wins", "ties", "losses", "win_frac")}})
    return gate, pd.DataFrame(crow)


def coefficient_stability(sb: pd.DataFrame, mp: pd.DataFrame, root: Path, fs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # 50/50 blend check needs raw OOF: compute per dataset x rep for all 10 pairs (cheap)
    from mixens import fastmetrics as fm
    blend_cache: dict[tuple, dict] = {}
    for ds in DATASETS:
        reps = sorted(sb[sb.dataset == ds]["rep"].unique())
        for r in reps:
            d = root / ds / f"rep_{r:02d}"
            if not (d / "oof.npz").exists():
                continue
            z = np.load(d / "oof.npz"); P = z["P"].astype(np.float32); y = z["y_train"]; yb = y.astype(bool)
            names = [str(s) for s in z["model_names"]]
            auc_v = {n: fm.fast_auc_no_ties(yb, P[:, i]) for i, n in enumerate(names)}
            ll_v = {n: fm.log_loss_vec(y, P[:, i]) for i, n in enumerate(names)}
            for i, j in combinations(range(len(names)), 2):
                w = np.zeros(len(names), np.float32); w[i] = w[j] = 0.5; p = P @ w
                blend_cache[(ds, r, f"{names[i]}*{names[j]}")] = {
                    "blend_auc": fm.fast_auc_no_ties(yb, p), "best_member_auc": max(auc_v[names[i]], auc_v[names[j]]),
                    "auc_gap": abs(auc_v[names[i]] - auc_v[names[j]]),
                    "blend_ll": fm.log_loss_vec(y, p), "best_member_ll": min(ll_v[names[i]], ll_v[names[j]]),
                    "ll_gap": abs(ll_v[names[i]] - ll_v[names[j]])}
    for ds in DATASETS:
        for resp in ("roc_auc", "log_loss"):
            sub = sb[(sb.dataset == ds) & (sb.response == resp)]
            reps = sorted(sub["rep"].unique()); n = len(reps)
            piv = sub.pivot(index="rep", columns="term", values="beta")
            inter = [t for t in piv.columns if "*" in t]
            ranks = piv[inter].abs().rank(axis=1, ascending=False)
            for term in piv.columns:
                v = piv[term].to_numpy()
                lo, hi = boot_ci(v)
                kpos = int((v > 0).sum()); wl, wh = wilson(kpos, n)
                row = {"dataset": ds, "response": resp, "term": term, "kind": "interaction" if "*" in term else "linear", "n": n,
                       "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "cv": float(v.std(ddof=1) / abs(v.mean())) if v.mean() != 0 else np.nan,
                       "median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                       "ci95_lo": lo, "ci95_hi": hi, "sign_pos_freq": kpos / n, "sign_pos_wilson_lo": wl, "sign_pos_wilson_hi": wh,
                       "sign_stable": bool(kpos in (0, n))}
                if term in inter:
                    rk = ranks[term]
                    row.update({"rank_mean": float(rk.mean()), "rank_min": int(rk.min()), "rank_max": int(rk.max()),
                                "top1_freq": float((rk == 1).mean()), "top3_freq": float((rk <= 3).mean())})
                    key = "auc" if resp == "roc_auc" else "ll"
                    bl = [blend_cache.get((ds, r, term)) for r in reps]; bl = [b for b in bl if b]
                    if bl:
                        if key == "auc":
                            row.update({"vertex_gap_mean": float(np.mean([b["auc_gap"] for b in bl])),
                                        "blend50_minus_best_member_mean": float(np.mean([b["blend_auc"] - b["best_member_auc"] for b in bl])),
                                        "blend50_beats_best_member_freq": float(np.mean([b["blend_auc"] > b["best_member_auc"] for b in bl]))})
                        else:
                            row.update({"vertex_gap_mean": float(np.mean([b["ll_gap"] for b in bl])),
                                        "blend50_minus_best_member_mean": float(np.mean([b["blend_ll"] - b["best_member_ll"] for b in bl])),
                                        "blend50_beats_best_member_freq": float(np.mean([b["blend_ll"] < b["best_member_ll"] for b in bl]))})
                    a, b_ = term.split("*")
                    ref = fs[(fs.dataset == ds) & (fs.cost == "weighted") & (fs.set == "empirical_reference")]
                    if len(ref):
                        both = ref[ref.support.str.split("+").apply(lambda s: a in s and b_ in s)]["count"].sum()
                        row["pareto_pair_participation"] = float(both / ref["count"].sum())
                rows.append(row)
    return pd.DataFrame(rows)


def r10_vs_r30(T: dict, r10_max: int) -> pd.DataFrame:
    rows = []

    def add(metric, ds, series_all: pd.Series, extra=None):
        s30 = series_all.dropna(); s10 = s30[s30.index <= r10_max] if isinstance(s30.index, pd.Index) else s30
        if len(s30) == 0:
            return
        e10, e30 = float(s10.mean()), float(s30.mean()); lo, hi = boot_ci(s30.to_numpy())
        rows.append({"metric": metric, "dataset": ds, **(extra or {}), "n_r10": int(len(s10)), "n_r30": int(len(s30)),
                     "r10_estimate": e10, "r30_estimate": e30, "abs_change": e30 - e10,
                     "rel_change": (e30 - e10) / abs(e10) if e10 != 0 else np.nan, "r30_ci95_lo": lo, "r30_ci95_hi": hi,
                     "r10_median": float(s10.median()), "r30_median": float(s30.median())})

    so = T["scheffe_orders"]; sel = so[so.selected]
    for ds in DATASETS:
        for resp in ("roc_auc", "log_loss"):
            sub = sel[(sel.dataset == ds) & (sel.response == resp)].set_index("rep")
            add("scheffe_r2_external", ds, sub["r2_external"], {"response": resp})
            add("reliability_gate_pass", ds, sub["reliable"].astype(float), {"response": resp})
    sb = T["scheffe_quadratic_beta"]
    for ds in DATASETS:
        for resp in ("roc_auc", "log_loss"):
            sub = sb[(sb.dataset == ds) & (sb.response == resp)]
            piv = sub.pivot(index="rep", columns="term", values="beta")
            for term in piv.columns:
                if "*" not in term:
                    add("beta_linear", ds, piv[term], {"term": term})
            inter = [t for t in piv.columns if "*" in t]
            top = piv[inter].abs().mean().sort_values(ascending=False).index[:3]
            for term in top:
                add("beta_interaction_top3", ds, piv[term], {"term": term})
                add("beta_interaction_sign_pos_freq", ds, (piv[term] > 0).astype(float), {"term": term})
    pq = T["pareto_quality"]
    for ds in DATASETS:
        for cost in ("weighted", "support"):
            for st in SETS:
                sub = pq[(pq.dataset == ds) & (pq.cost == cost) & (pq.set == st)].set_index("rep")
                for m in ("igd_plus", "hv_ratio", "joint_nondominated_fraction_all_valid", "n_front"):
                    if m in sub:
                        add(m, ds, sub[m], {"set": st, "cost": cost})
    nb = T["nbi_runs"]
    for ds in DATASETS:
        for v in ("A", "B", "C"):
            sub = nb[(nb.dataset == ds) & (nb.variant == v)].set_index("rep")
            add("nbi_success_rate", ds, sub["n_success"] / sub["n_subproblems"], {"set": f"nbi_{v}"})
            add("nbi_seconds", ds, sub["seconds"], {"set": f"nbi_{v}"})
    rd = T["reference_diagnostics"]
    for ds in DATASETS:
        sub = rd[rd.dataset == ds].set_index("rep")
        add("reference_front_weighted_size", ds, sub["front_weighted_size"]); add("reference_front_support_size", ds, sub["front_support_size"])
        add("reference_displaced_fraction", ds, sub["displaced_fraction"])
    mp = T["mcdm_picks_holdout"]
    for ds in DATASETS:
        for st in ("nbi_A", "nbi_B", "nbi_C", "ws_random_scalarization"):
            sub = mp[(mp.dataset == ds) & (mp.cost == "weighted") & (mp.rule == "knee") & (mp.set == st)].set_index("rep")
            add("holdout_minus_oof_auc_knee", ds, sub["holdout_roc_auc"] - sub["oof_roc_auc"], {"set": st})
    st_ = T["stage_times"]
    for ds in DATASETS:
        sub = st_[st_.dataset == ds].groupby("rep")["seconds"].sum()
        add("replication_seconds", ds, sub)
    return pd.DataFrame(rows)


def holdout_transfer(mp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sets = ["nbi_A", "nbi_B", "nbi_C", "ws_random_scalarization"]
    for cost in ("weighted", "support"):
        for ds in DATASETS:
            sub = mp[(mp.dataset == ds) & (mp.cost == cost) & (mp.rule == "knee") & (mp.set.isin(sets))]
            for st in sets:
                s2 = sub[sub.set == st].set_index("rep")
                for metric, oof_col, hold_col in (("roc_auc", "oof_roc_auc", "holdout_roc_auc"), ("log_loss", "oof_log_loss", "holdout_log_loss")):
                    d = (s2[hold_col] - s2[oof_col]).to_numpy(); lo, hi = boot_ci(d)
                    rows.append({"cost": cost, "dataset": ds, "set": st, "metric": metric, "n": len(d), "oof_mean": float(s2[oof_col].mean()),
                                 "holdout_mean": float(s2[hold_col].mean()), "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
                                 "delta_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "delta_ci95_lo": lo, "delta_ci95_hi": hi,
                                 "delta_min": float(d.min()), "delta_max": float(d.max()), "frac_abs_delta_gt_0.005": float((np.abs(d) > 0.005).mean())})
            # ranking agreement: best set by OOF AUC == best set by holdout AUC per rep
            piv_o = sub.pivot(index="rep", columns="set", values="oof_roc_auc"); piv_h = sub.pivot(index="rep", columns="set", values="holdout_roc_auc")
            common = piv_o.dropna().index.intersection(piv_h.dropna().index)
            agree = (piv_o.loc[common].idxmax(axis=1) == piv_h.loc[common].idxmax(axis=1))
            k, n = int(agree.sum()), int(len(agree)); wl, wh = wilson(k, n)
            rows.append({"cost": cost, "dataset": ds, "set": "ALL(knee picks)", "metric": "ranking_agreement_auc", "n": n, "delta_mean": k / n if n else np.nan,
                         "delta_ci95_lo": wl, "delta_ci95_hi": wh, "wins": k})
    return pd.DataFrame(rows)


def auc_logloss_conflict(rf: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        a = rf[(rf.dataset == ds) & (rf.method == "direct_auc_search")].set_index("rep")
        b = rf[(rf.dataset == ds) & (rf.method == "slsqp_direct_logloss")].set_index("rep")
        j = a.index.intersection(b.index)
        wa = a.loc[j, [f"w_{m}" for m in MODELS]].to_numpy(); wb = b.loc[j, [f"w_{m}" for m in MODELS]].to_numpy()
        l1 = np.abs(wa - wb).sum(axis=1); supp_a = wa > 1e-3; supp_b = wb > 1e-3
        jacc = np.array([len(set(np.where(x)[0]) & set(np.where(y)[0])) / max(len(set(np.where(x)[0]) | set(np.where(y)[0])), 1) for x, y in zip(supp_a, supp_b)])
        for name, d in (("delta_auc_directauc_minus_slsqp", (a.loc[j, "oof_roc_auc"] - b.loc[j, "oof_roc_auc"]).to_numpy()),
                        ("delta_logloss_directauc_minus_slsqp", (a.loc[j, "oof_log_loss"] - b.loc[j, "oof_log_loss"]).to_numpy()),
                        ("holdout_delta_auc", (a.loc[j, "holdout_roc_auc"] - b.loc[j, "holdout_roc_auc"]).to_numpy()),
                        ("holdout_delta_logloss", (a.loc[j, "holdout_log_loss"] - b.loc[j, "holdout_log_loss"]).to_numpy()),
                        ("weight_l1_distance", l1), ("support_jaccard", jacc),
                        ("delta_cost_weighted", (a.loc[j, "cost_weighted"] - b.loc[j, "cost_weighted"]).to_numpy()),
                        ("delta_cost_support", (a.loc[j, "cost_support"] - b.loc[j, "cost_support"]).to_numpy()),
                        ("ensembling_gain_auc_slsqp_minus_best_single", (b.loc[j, "oof_roc_auc"] - rf[(rf.dataset == ds) & (rf.method == "best_single")].set_index("rep").loc[j, "oof_roc_auc"]).to_numpy())):
            lo, hi = boot_ci(d)
            rows.append({"dataset": ds, "quantity": name, "n": len(d), "mean": float(np.mean(d)), "median": float(np.median(d)), "sd": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                         "ci95_lo": lo, "ci95_hi": hi, "min": float(np.min(d)), "max": float(np.max(d))})
    return pd.DataFrame(rows)


def cost_definition_sensitivity(pq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for ep in ("hv_ratio", "igd_plus"):
            w = pq[(pq.dataset == ds) & (pq.cost == "weighted") & (pq.set.isin(SETS))].pivot(index="rep", columns="set", values=ep)
            s = pq[(pq.dataset == ds) & (pq.cost == "support") & (pq.set.isin(SETS))].pivot(index="rep", columns="set", values=ep)
            common = w.dropna().index.intersection(s.dropna().index)
            best_w = w.loc[common].idxmax(axis=1) if ep == "hv_ratio" else w.loc[common].idxmin(axis=1)
            best_s = s.loc[common].idxmax(axis=1) if ep == "hv_ratio" else s.loc[common].idxmin(axis=1)
            rho = [stats.spearmanr(w.loc[i], s.loc[i]).correlation for i in common]
            diff = int((best_w != best_s).sum()); n = len(common); wl, wh = wilson(diff, n)
            rows.append({"dataset": ds, "endpoint": ep, "n": n, "best_set_differs_count": diff, "best_set_differs_frac": diff / n if n else np.nan,
                         "wilson_lo": wl, "wilson_hi": wh, "median_rank_spearman_weighted_vs_support": float(np.nanmedian(rho)),
                         "best_weighted_freq": best_w.value_counts().to_dict(), "best_support_freq": best_s.value_counts().to_dict()})
        # NBI-C support-cost HV distribution (BNP flip diagnostic)
        c = pq[(pq.dataset == ds) & (pq.cost == "support") & (pq.set == "nbi_C")]["hv_ratio"].to_numpy()
        rows.append({"dataset": ds, "endpoint": "nbi_C_support_hv_ratio", "n": len(c), "mean": float(c.mean()), "median": float(np.median(c)),
                     "frac_below_0.8": float((c < 0.8).mean()), "bimodality_coefficient": bimodality_coefficient(c), "gmm_bic_delta": gmm_bic_delta(c)})
    return pd.DataFrame(rows)


def bimodality(pq: pd.DataFrame, nb: pd.DataFrame, so: pd.DataFrame, mp: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    prev = {}
    for ds in DATASETS:
        for d in sorted((root / ds).glob("rep_*")):
            f = d / "oof.npz"
            if f.exists():
                z = np.load(f); prev[(ds, int(d.name[4:]))] = float(z["y_train"].mean())
    sel = so[so.selected]
    for ds in DATASETS:
        for st in ("nbi_A", "nbi_B", "nbi_C"):
            for cost in ("weighted", "support"):
                for m in ("hv_ratio", "igd_plus"):
                    sub = pq[(pq.dataset == ds) & (pq.cost == cost) & (pq.set == st)].set_index("rep")[m].dropna()
                    x = sub.to_numpy()
                    rows.append({"dataset": ds, "set": st, "cost": cost, "metric": m, "n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
                                 "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "q10": float(np.percentile(x, 10)), "q90": float(np.percentile(x, 90)),
                                 "bimodality_coefficient": bimodality_coefficient(x), "gmm_bic_delta": gmm_bic_delta(x),
                                 "bc_gt_0.555": bool(bimodality_coefficient(x) > 0.555) if np.isfinite(bimodality_coefficient(x)) else None})
            v = st[-1]
            s2 = nb[(nb.dataset == ds) & (nb.variant == v)].set_index("rep")
            rate = (s2["n_success"] / s2["n_subproblems"]).to_numpy()
            rows.append({"dataset": ds, "set": st, "cost": "-", "metric": "success_rate", "n": len(rate), "mean": float(rate.mean()), "median": float(np.median(rate)),
                         "sd": float(rate.std(ddof=1)) if len(rate) > 1 else 0.0, "q10": float(np.percentile(rate, 10)), "q90": float(np.percentile(rate, 90)),
                         "bimodality_coefficient": bimodality_coefficient(rate), "gmm_bic_delta": gmm_bic_delta(rate)})
            for col in ("anchor_auc_cost", "anchor_ll_cost"):
                x = s2[col].to_numpy()
                rows.append({"dataset": ds, "set": st, "cost": "-", "metric": col, "n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
                             "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "q10": float(np.percentile(x, 10)), "q90": float(np.percentile(x, 90)),
                             "bimodality_coefficient": bimodality_coefficient(x), "gmm_bic_delta": gmm_bic_delta(x)})
        # regime association (exploratory): NBI-A weighted HV vs observables
        a = pq[(pq.dataset == ds) & (pq.cost == "weighted") & (pq.set == "nbi_A")].set_index("rep")["hv_ratio"]
        obs = {"r2ext_auc": sel[(sel.dataset == ds) & (sel.response == "roc_auc")].set_index("rep")["r2_external"],
               "r2ext_ll": sel[(sel.dataset == ds) & (sel.response == "log_loss")].set_index("rep")["r2_external"],
               "spearman_auc": sel[(sel.dataset == ds) & (sel.response == "roc_auc")].set_index("rep")["spearman_external"],
               "anchor_auc_cost_A": nb[(nb.dataset == ds) & (nb.variant == "A")].set_index("rep")["anchor_auc_cost"],
               "anchor_ll_cost_A": nb[(nb.dataset == ds) & (nb.variant == "A")].set_index("rep")["anchor_ll_cost"],
               "split_prevalence": pd.Series({r: p for (dd, r), p in prev.items() if dd == ds})}
        for k, v in obs.items():
            j = a.index.intersection(v.index)
            if len(j) >= 5:
                rho = stats.spearmanr(a.loc[j], v.loc[j])
                rows.append({"dataset": ds, "set": "nbi_A", "cost": "weighted", "metric": f"spearman(hv_ratio, {k})", "n": len(j), "mean": float(rho.correlation), "p_value_descriptive": float(rho.pvalue)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(pq, eff, gate, cs, stab, ht, nb, wtl, figdir: Path) -> None:
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    comps = [("nbi_B", "nbi_A"), ("nbi_C", "nbi_B"), ("nbi_C", "ws_random_scalarization")]
    for ep, direction, fname in (("igd_plus", "lower", "r30_fig01_paired_delta_igd_plus"), ("hv_ratio", "higher", "r30_fig02_paired_delta_hv")):
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.6 * len(DATASETS), 3.8), sharey=False)
        for ax, ds in zip(np.atleast_1d(axes), DATASETS):
            data = []
            for new, ref in comps:
                a = pq[(pq.dataset == ds) & (pq.cost == "weighted") & (pq.set == new)].set_index("rep")[ep]
                b = pq[(pq.dataset == ds) & (pq.cost == "weighted") & (pq.set == ref)].set_index("rep")[ep]
                j = a.index.intersection(b.index)
                data.append(((a.loc[j] - b.loc[j]) if direction == "higher" else (b.loc[j] - a.loc[j])).to_numpy())
            ax.boxplot(data, tick_labels=["B−A", "C−B", "C−scal."], showfliers=False)
            for i, d in enumerate(data, 1):
                ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(d)), d, s=9, alpha=0.6)
            ax.axhline(0, color="k", lw=0.6); ax.set_title(ds)
        np.atleast_1d(axes)[0].set_ylabel(f"paired Δ{ep} (>0 = second method better)")
        fig.suptitle(f"Paired replication differences, {ep} (weighted cost)"); fig.tight_layout(); fig.savefig(figdir / f"{fname}.png", bbox_inches="tight"); plt.close(fig)
    # 3. R10 vs R30 stability dumbbells for key metrics
    key = stab[stab.metric.isin(["igd_plus", "hv_ratio"]) & stab.set.isin(["nbi_A", "nbi_B", "nbi_C"]) & (stab.cost == "weighted")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, m in zip(axes, ("hv_ratio", "igd_plus")):
        sub = key[key.metric == m].reset_index(drop=True)
        yy = np.arange(len(sub))
        ax.hlines(yy, sub.r10_estimate, sub.r30_estimate, color="gray", lw=1)
        ax.scatter(sub.r10_estimate, yy, color="tab:orange", label="R=10", zorder=3); ax.scatter(sub.r30_estimate, yy, color="tab:blue", label="R=30", zorder=3)
        ax.errorbar(sub.r30_estimate, yy, xerr=[np.maximum(sub.r30_estimate - sub.r30_ci95_lo, 0), np.maximum(sub.r30_ci95_hi - sub.r30_estimate, 0)], fmt="none", ecolor="tab:blue", alpha=0.5)
        ax.set_yticks(yy); ax.set_yticklabels([f"{r.dataset} {r.set}" for r in sub.itertuples()], fontsize=7); ax.set_xlabel(m); ax.legend(fontsize=7)
    fig.suptitle("R=10 vs R=30 mean estimates (bars: R=30 bootstrap 95% CI)"); fig.tight_layout(); fig.savefig(figdir / "r30_fig03_r10_vs_r30_stability.png", bbox_inches="tight"); plt.close(fig)
    # 4. gate pass frequencies with Wilson intervals
    fig, ax = plt.subplots(figsize=(9, 3.6))
    g = gate[gate.response.isin(["roc_auc", "log_loss", "pr_auc"])].reset_index(drop=True)
    x = np.arange(len(g)); ax.bar(x, g.p_pass, color=["tab:red" if r == "roc_auc" else "tab:blue" if r == "log_loss" else "tab:green" for r in g.response])
    ax.errorbar(x, g.p_pass, yerr=[np.maximum(g.p_pass - g.wilson_lo, 0), np.maximum(g.wilson_hi - g.p_pass, 0)], fmt="none", ecolor="k", capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([f"{r.dataset}\n{r.response}" for r in g.itertuples()], fontsize=7); ax.set_ylim(0, 1.05); ax.set_ylabel("P(reliability gate pass)")
    fig.suptitle("Reliability-gate pass frequency with Wilson 95% intervals"); fig.tight_layout(); fig.savefig(figdir / "r30_fig04_gate_pass.png", bbox_inches="tight"); plt.close(fig)
    # 5. beta_ij distributions
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(3.8 * len(DATASETS), 6.5))
    for c, ds in enumerate(DATASETS):
        for r_, resp in enumerate(("roc_auc", "log_loss")):
            sub = cs[(cs.dataset == ds) & (cs.response == resp) & (cs.kind == "interaction")].sort_values("mean")
            ax = axes[r_, c]; yy = np.arange(len(sub))
            ax.errorbar(sub["mean"], yy, xerr=[np.maximum(sub["mean"] - sub["ci95_lo"], 0), np.maximum(sub["ci95_hi"] - sub["mean"], 0)], fmt="o", ms=3)
            ax.set_yticks(yy); ax.set_yticklabels(sub["term"], fontsize=6); ax.axvline(0, color="k", lw=0.5); ax.set_title(f"{ds} {resp}", fontsize=8)
    fig.suptitle("Scheffé quadratic β_ij: mean and bootstrap 95% CI over replications"); fig.tight_layout(); fig.savefig(figdir / "r30_fig05_beta_ij_stability.png", bbox_inches="tight"); plt.close(fig)
    # 6. win/tie/loss heatmap (weighted, hv_ratio)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.6 * len(DATASETS), 3.6))
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        sub = wtl[(wtl.dataset == ds) & (wtl.cost == "weighted") & (wtl.endpoint == "hv_ratio")]
        mat = np.full((len(SETS), len(SETS)), np.nan)
        for r in sub.itertuples():
            i, j = SETS.index(r.set_a), SETS.index(r.set_b); mat[i, j] = r.a_win_frac; mat[j, i] = 1 - r.a_win_frac - (r.ties / r.n if r.n else 0)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1); ax.set_xticks(range(len(SETS))); ax.set_yticks(range(len(SETS)))
        ax.set_xticklabels([s[:6] for s in SETS], rotation=90, fontsize=6); ax.set_yticklabels([s[:10] for s in SETS], fontsize=6); ax.set_title(ds)
        for i in range(len(SETS)):
            for j in range(len(SETS)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.suptitle("Fraction of replications in which the row set beats the column set (HV ratio, weighted cost)"); fig.tight_layout(); fig.savefig(figdir / "r30_fig06_win_tie_loss.png", bbox_inches="tight"); plt.close(fig)
    # 7. holdout transfer distributions
    mp_sets = ["nbi_A", "nbi_B", "nbi_C", "ws_random_scalarization"]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.6 * len(DATASETS), 3.6), sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        data = [ht_raw[(ds, s)] for s in mp_sets if (ds, s) in ht_raw]
        ax.boxplot(data, tick_labels=[s[:5] for s in mp_sets][:len(data)], showfliers=True); ax.axhline(0, color="k", lw=0.6); ax.set_title(ds)
    np.atleast_1d(axes)[0].set_ylabel("holdout − OOF AUC (knee pick)"); fig.suptitle("OOF → holdout transfer of knee picks (weighted cost)")
    fig.tight_layout(); fig.savefig(figdir / "r30_fig07_holdout_transfer.png", bbox_inches="tight"); plt.close(fig)
    # 8. weighted vs support: best set frequency
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.6 * len(DATASETS), 3.4))
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        w = pq[(pq.dataset == ds) & (pq.cost == "weighted") & (pq.set.isin(SETS))].pivot(index="rep", columns="set", values="hv_ratio").idxmax(axis=1).value_counts()
        s = pq[(pq.dataset == ds) & (pq.cost == "support") & (pq.set.isin(SETS))].pivot(index="rep", columns="set", values="hv_ratio").idxmax(axis=1).value_counts()
        idx = SETS; xx = np.arange(len(idx))
        ax.bar(xx - 0.2, [w.get(k, 0) for k in idx], width=0.4, label="weighted"); ax.bar(xx + 0.2, [s.get(k, 0) for k in idx], width=0.4, label="support")
        ax.set_xticks(xx); ax.set_xticklabels([k[:6] for k in idx], fontsize=6, rotation=90); ax.set_title(ds)
    np.atleast_1d(axes)[0].set_ylabel("replications where set has the best HV"); np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Best set by hypervolume under weighted vs support cost"); fig.tight_layout(); fig.savefig(figdir / "r30_fig08_cost_definition_ranking.png", bbox_inches="tight"); plt.close(fig)
    # 9. anchor-cost distributions
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.6 * len(DATASETS), 3.4))
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        for v, col in (("A", "tab:red"), ("B", "tab:blue")):
            s2 = nb[(nb.dataset == ds) & (nb.variant == v)]
            ax.scatter(np.full(len(s2), 0) + (0 if v == "A" else 1) + np.random.default_rng(1).normal(0, 0.05, len(s2)), s2["anchor_auc_cost"], s=10, color=col, label=f"AUC anchor {v}")
            ax.scatter(np.full(len(s2), 2) + (0 if v == "A" else 1) + np.random.default_rng(2).normal(0, 0.05, len(s2)), s2["anchor_ll_cost"], s=10, color=col, marker="s", label=f"LL anchor {v}")
        ax.set_yscale("log"); ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["AUC-A", "AUC-B", "LL-A", "LL-B"], fontsize=7); ax.set_title(ds)
    np.atleast_1d(axes)[0].set_ylabel("anchor weighted cost (ms/1k)"); fig.suptitle("Anchor cost: surrogate (A) vs real (B) anchors")
    fig.tight_layout(); fig.savefig(figdir / "r30_fig09_anchor_costs.png", bbox_inches="tight"); plt.close(fig)
    # 10. NBI success-rate distributions (ECDF)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.6 * len(DATASETS), 3.4), sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        for v, col in (("A", "tab:red"), ("B", "tab:blue"), ("C", "tab:green")):
            s2 = nb[(nb.dataset == ds) & (nb.variant == v)]; x = np.sort((s2["n_success"] / s2["n_subproblems"]).to_numpy())
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, label=f"NBI-{v}")
        ax.set_xlim(0, 1.02); ax.set_title(ds); ax.set_xlabel("subproblem success/feasibility rate")
    np.atleast_1d(axes)[0].set_ylabel("ECDF over replications"); np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Distribution of NBI subproblem outcome rates"); fig.tight_layout(); fig.savefig(figdir / "r30_fig10_nbi_success_ecdf.png", bbox_inches="tight"); plt.close(fig)


ht_raw: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "experiments" / "pco213_postwork_benchmark"))
    ap.add_argument("--reports", default=str(REPO / "reports" / "pco213_postwork_benchmark"))
    ap.add_argument("--figures", default=str(REPO / "figures" / "pco213_postwork_benchmark"))
    ap.add_argument("--r10-max-rep", type=int, default=9)
    args = ap.parse_args()
    root, repdir, figdir = Path(args.root), Path(args.reports), Path(args.figures)
    out = repdir / "statistics"; out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    T = {k: pd.read_csv(repdir / "tables" / f"{k}.csv") for k in
         ("pareto_quality", "scheffe_orders", "scheffe_quadratic_beta", "single_objective_refs", "nbi_runs", "mcdm_picks_holdout",
          "reference_diagnostics", "stage_times", "model_performance", "front_supports")}
    pq = T["pareto_quality"]
    print("replications per dataset:", pq.groupby("dataset")["rep"].nunique().to_dict())

    eff_w, tests_w = paired_primary(pq, "weighted"); eff_s, tests_s = paired_primary(pq, "support")
    eff = pd.concat([eff_w, eff_s], ignore_index=True); tests = pd.concat([tests_w, tests_s], ignore_index=True)
    eff.to_csv(out / "paired_primary_effects.csv", index=False); tests.to_csv(out / "paired_primary_tests.csv", index=False)
    wtl = win_tie_loss(pq); wtl.to_csv(out / "win_tie_loss.csv", index=False)
    gate, cond = reliability_gate(T["scheffe_orders"], pq)
    gate.to_csv(out / "reliability_gate_r30.csv", index=False); cond.to_csv(out / "reliability_gate_conditional_gain_r30.csv", index=False)
    cs = coefficient_stability(T["scheffe_quadratic_beta"], T["model_performance"], root, T["front_supports"])
    cs.to_csv(out / "coefficient_stability_r30.csv", index=False)
    stab = r10_vs_r30(T, args.r10_max_rep); stab.to_csv(out / "r10_vs_r30_stability.csv", index=False)
    ht = holdout_transfer(T["mcdm_picks_holdout"]); ht.to_csv(out / "holdout_transfer_r30.csv", index=False)
    for ds in DATASETS:
        for st in ("nbi_A", "nbi_B", "nbi_C", "ws_random_scalarization"):
            sub = T["mcdm_picks_holdout"]; s2 = sub[(sub.dataset == ds) & (sub.cost == "weighted") & (sub.rule == "knee") & (sub.set == st)]
            ht_raw[(ds, st)] = (s2["holdout_roc_auc"] - s2["oof_roc_auc"]).to_numpy()
    conf = auc_logloss_conflict(T["single_objective_refs"], root); conf.to_csv(out / "auc_logloss_conflict_r30.csv", index=False)
    cds = cost_definition_sensitivity(pq); cds.to_csv(out / "cost_definition_sensitivity_r30.csv", index=False)
    bim = bimodality(pq, T["nbi_runs"], T["scheffe_orders"], T["mcdm_picks_holdout"], root); bim.to_csv(out / "bimodality_regimes_r30.csv", index=False)
    # proportion intervals (gate pass, win fractions, sign consistency, NBI feasibility >= 0.9)
    prop = []
    for r in gate.itertuples():
        prop.append({"quantity": "reliability_gate_pass", "dataset": r.dataset, "level": r.response, "k": r.pass_ if hasattr(r, "pass_") else r._4, "n": r.n, "p": r.p_pass, "wilson_lo": r.wilson_lo, "wilson_hi": r.wilson_hi, "jeffreys_lo": r.jeffreys_lo, "jeffreys_hi": r.jeffreys_hi})
    for r in eff[eff.cost == "weighted"].itertuples():
        prop.append({"quantity": f"win_frac[{r.comparison}, {r.endpoint}]", "dataset": r.dataset, "level": "weighted", "k": r.wins, "n": r.n, "p": r.win_frac, "wilson_lo": r.win_frac_wilson_lo, "wilson_hi": r.win_frac_wilson_hi, **dict(zip(("jeffreys_lo", "jeffreys_hi"), jeffreys(int(r.wins), int(r.n))))})
    nb = T["nbi_runs"]
    for ds in DATASETS:
        for v in ("A", "B", "C"):
            s2 = nb[(nb.dataset == ds) & (nb.variant == v)]; k = int(((s2["n_success"] / s2["n_subproblems"]) >= 0.9).sum()); n = len(s2)
            prop.append({"quantity": f"nbi_{v}_feasibility_rate>=0.9", "dataset": ds, "level": v, "k": k, "n": n, "p": k / n if n else np.nan, **dict(zip(("wilson_lo", "wilson_hi"), wilson(k, n))), **dict(zip(("jeffreys_lo", "jeffreys_hi"), jeffreys(k, n)))})
    for r in cs[(cs.kind == "interaction") & (cs.get("top3_freq", pd.Series(dtype=float)).notna() if "top3_freq" in cs else True)].itertuples():
        prop.append({"quantity": f"beta_sign_positive[{r.term}, {r.response}]", "dataset": r.dataset, "level": r.response, "k": int(round(r.sign_pos_freq * r.n)), "n": r.n, "p": r.sign_pos_freq, "wilson_lo": r.sign_pos_wilson_lo, "wilson_hi": r.sign_pos_wilson_hi, **dict(zip(("jeffreys_lo", "jeffreys_hi"), jeffreys(int(round(r.sign_pos_freq * r.n)), int(r.n))))})
    pd.DataFrame(prop).to_csv(out / "proportion_intervals.csv", index=False)
    make_figures(pq, eff, gate, cs, stab, ht, nb, wtl, figdir)
    summary = {"replications_per_dataset": pq.groupby("dataset")["rep"].nunique().to_dict(), "r10_max_rep": args.r10_max_rep,
               "primary_comparisons": [f"{a} vs {b}" for a, b in PRIMARY], "endpoints": ENDPOINTS, "rho_nadeau_bengio": RHO, "tie_tol": TIE_TOL,
               "n_boot": N_BOOT, "seed": SEED, "holm_family": "per dataset: 4 primary comparisons x 2 endpoints (8 tests)",
               "sign_convention": "delta > 0 means the new (second-named) method is better"}
    (out / "analysis_config.json").write_text(json.dumps(summary, indent=2))
    print("statistics written to", out)


if __name__ == "__main__":
    main()
