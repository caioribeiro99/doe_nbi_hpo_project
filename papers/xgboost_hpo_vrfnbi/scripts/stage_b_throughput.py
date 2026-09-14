#!/usr/bin/env python
"""Stage B: parallel-throughput benchmark. Engineering only.

Chooses the campaign's execution layout by measuring it. No arm is run, no method
is compared, and no configuration is chosen because an optimizer liked it: the
benchmark set is pre-specified from the design's own structure.

What is measured, per layout:
  evaluations per second, wall clock for a fixed batch, CPU utilization, peak
  resident memory, swap activity, load average, sustained-interval degradation,
  and whether results are bit-identical across layouts.

Layouts are (process workers) x (XGBoost threads per fit). Combinations whose
product exceeds the physical core count are skipped, because nested
oversubscription measures the scheduler rather than the workload.

Writes STAGE_B_THROUGHPUT.md and audits/stage_b_throughput.json.

Usage:
  python stage_b_throughput.py [--dataset magic] [--repeats 1] [--sustained-min 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi"
DESIGN = REPO / "data" / "design" / "hyperparameter_design.csv"

PARAMS = ["subsample", "colsample_bytree", "colsample_bylevel",
          "learning_rate", "max_depth", "gamma", "n_estimators"]
INTS = {"max_depth", "n_estimators"}
BOUNDS = {"subsample": (0.05, 1.0), "colsample_bytree": (0.05, 1.0),
          "colsample_bylevel": (0.05, 1.0), "learning_rate": (0.01, 0.30),
          "max_depth": (3, 18), "gamma": (0.05, 5.0), "n_estimators": (50, 700)}

_G: dict = {}          # per-worker process state, filled by _init


# --------------------------------------------------------------------- workload

def benchmark_configs() -> pd.DataFrame:
    """A pre-specified set spanning the box, taken from the design's own structure.

    Composition: the centre; the cheapest and the most expensive corner; six
    further factorial corners chosen by index, not by outcome; and six axial runs.
    Nothing here is selected because it performs well.
    """
    lo = np.array([BOUNDS[p][0] for p in PARAMS])
    hi = np.array([BOUNDS[p][1] for p in PARAMS])

    def un(c):
        return lo + (np.asarray(c, float) + 1.0) * (hi - lo) / 2.0

    k = len(PARAMS)
    rows, labels = [], []
    rows.append(un(np.zeros(k))); labels.append("centre")
    cheap = np.array([-1, -1, -1, -1, -1, +1, -1], float)   # small, shallow, pruned, few trees
    rows.append(un(cheap)); labels.append("cheap corner")
    expensive = np.array([+1, +1, +1, +1, +1, -1, +1], float)
    rows.append(un(expensive)); labels.append("expensive corner")
    # six further corners from the design's own half fraction, by position
    design = pd.read_csv(DESIGN, sep=";", decimal=",", encoding="utf-8-sig")
    design.columns = [str(c).strip().strip('"') for c in design.columns]
    C = 2 * (design[PARAMS].to_numpy(float) - lo) / (hi - lo) - 1
    corners = C[(np.abs(np.abs(C) - 1) < 1e-9).all(axis=1)]
    for idx in (0, 12, 25, 37, 50, 62):
        rows.append(un(corners[idx])); labels.append(f"design corner #{idx}")
    for i in (0, 2, 4, 6):
        for s in (-1.0, +1.0):
            c = np.zeros(k); c[i] = s
            rows.append(un(c)); labels.append(f"axial {PARAMS[i]} {s:+.0f}")
    df = pd.DataFrame(rows, columns=PARAMS)
    for p in INTS:
        df[p] = df[p].round().astype(int)
    df.insert(0, "label", labels)
    return df


def _init(dataset_id: str, threads: int, seed: int) -> None:
    """Per-worker setup. The dataset is loaded once per process, not per fit."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(threads)
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from doe_xgb.datasets.loaders import load
    d = load(dataset_id)
    X = d.X.copy()
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat:
        X = pd.get_dummies(X, columns=cat, dummy_na=False)
    X = X.astype(float).to_numpy()
    y = np.asarray(d.y, dtype=int)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    _G.update(X=Xtr, y=ytr, threads=threads, seed=seed,
              kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))


def _evaluate(params: dict) -> dict:
    """One real evaluation: five folds, both cost quantities persisted."""
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    X, y, kf = _G["X"], _G["y"], _G["kf"]
    leaves, times, acc, auc, ll = [], [], [], [], []
    for tr, va in kf.split(X, y):
        mdl = XGBClassifier(**params, eval_metric="logloss", verbosity=0,
                            tree_method="hist", n_jobs=_G["threads"],
                            random_state=_G["seed"])
        t0 = time.perf_counter()
        mdl.fit(X[tr], y[tr])
        prob = mdl.predict_proba(X[va])[:, 1]
        times.append(time.perf_counter() - t0)
        leaves.append(sum(s.count("leaf=") for s in mdl.get_booster().get_dump()))
        acc.append(accuracy_score(y[va], (prob >= 0.5).astype(int)))
        auc.append(roc_auc_score(y[va], prob))
        ll.append(log_loss(y[va], np.clip(prob, 1e-7, 1 - 1e-7)))
    return {"leaves": float(np.mean(leaves)), "time": float(np.mean(times)),
            "acc": float(np.mean(acc)), "auc": float(np.mean(auc)),
            "logloss": float(np.mean(ll))}


def _task(row: dict) -> dict:
    p = {k: (int(row[k]) if k in INTS else float(row[k])) for k in PARAMS}
    r = _evaluate(p)
    r["label"] = row["label"]
    return r


# -------------------------------------------------------------------- measuring

class Sampler:
    """Samples system state on a background thread while a batch runs."""

    def __init__(self, period: float = 0.5) -> None:
        self.period, self.samples, self._stop = period, [], False

    def __enter__(self):
        import threading
        psutil.cpu_percent(interval=None)
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self) -> None:
        proc = psutil.Process()
        while not self._stop:
            try:
                rss = proc.memory_info().rss
                for c in proc.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except psutil.Error:
                        pass
                self.samples.append({
                    "cpu": psutil.cpu_percent(interval=None),
                    "rss": rss,
                    "swap_used": psutil.swap_memory().used,
                    "load1": os.getloadavg()[0],
                })
            except Exception:
                pass
            time.sleep(self.period)

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        s = self.samples
        return {"cpu_mean_pct": round(float(np.mean([x["cpu"] for x in s])), 1),
                "cpu_max_pct": round(float(np.max([x["cpu"] for x in s])), 1),
                "peak_rss_gb": round(float(np.max([x["rss"] for x in s])) / 2**30, 2),
                "swap_used_delta_mb": round(
                    (float(s[-1]["swap_used"]) - float(s[0]["swap_used"])) / 2**20, 1),
                "load1_max": round(float(np.max([x["load1"] for x in s])), 1),
                "samples": len(s)}


def run_layout(dataset: str, workers: int, threads: int, configs: pd.DataFrame,
               repeats: int, seed: int) -> dict:
    rows = configs.to_dict("records") * repeats
    with Sampler() as smp:
        t0 = time.perf_counter()
        if workers == 1:
            _init(dataset, threads, seed)
            results = [_task(r) for r in rows]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init,
                          initargs=(dataset, threads, seed)) as pool:
                results = pool.map(_task, rows, chunksize=1)
        wall = time.perf_counter() - t0
    return {"workers": workers, "threads": threads, "evaluations": len(rows),
            "wall_seconds": round(wall, 2),
            "evals_per_second": round(len(rows) / wall, 4),
            "seconds_per_evaluation": round(wall / len(rows), 3),
            "system": smp.summary(),
            "results": results}


def fingerprint(results: list[dict]) -> str:
    """Deterministic signature of a layout's numerical output.

    Leaf count and the metrics must not depend on how the work was distributed.
    Measured time must, so it is excluded.
    """
    import hashlib
    keyed = sorted(results, key=lambda r: r["label"])
    blob = "|".join(f"{r['label']}:{r['leaves']:.6f}:{r['acc']:.10f}:"
                    f"{r['auc']:.10f}:{r['logloss']:.10f}" for r in keyed)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="magic")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sustained-min", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    configs = benchmark_configs()
    print(f"Stage B throughput benchmark\n  machine: {platform.platform()}, "
          f"{cores} physical cores, {psutil.virtual_memory().total/2**30:.0f} GB\n"
          f"  dataset: {args.dataset}, {len(configs)} pre-specified configurations "
          f"x {args.repeats} repeat(s)\n")

    worker_grid = (1, 2, 4, 6, 8, 10, 12, 14)
    thread_grid = (1, 2, 4)
    layouts = [(w, t) for w, t in itertools.product(worker_grid, thread_grid)
               if w * t <= cores]
    report = {"machine": {"platform": platform.platform(), "physical_cores": cores,
                          "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
              "dataset": args.dataset, "seed": args.seed,
              "configurations": configs.to_dict("records"),
              "layouts": [], "skipped_oversubscribed":
                  [f"{w}x{t}" for w, t in itertools.product(worker_grid, thread_grid)
                   if w * t > cores]}

    print(f"{'layout':>10}{'evals/s':>10}{'s/eval':>9}{'wall s':>9}"
          f"{'cpu%':>7}{'peakGB':>8}{'swapMB':>8}{'fingerprint':>18}")
    base_fp = None
    for w, t in layouts:
        r = run_layout(args.dataset, w, t, configs, args.repeats, args.seed)
        r["fingerprint"] = fingerprint(r.pop("results"))
        base_fp = base_fp or r["fingerprint"]
        r["deterministic_vs_first_layout"] = (r["fingerprint"] == base_fp)
        report["layouts"].append(r)
        s = r["system"]
        print(f"{w}w x {t}t".rjust(10)
              + f"{r['evals_per_second']:10.3f}{r['seconds_per_evaluation']:9.3f}"
              + f"{r['wall_seconds']:9.1f}{s.get('cpu_mean_pct', 0):7.1f}"
              + f"{s.get('peak_rss_gb', 0):8.2f}{s.get('swap_used_delta_mb', 0):8.1f}"
              + f"{r['fingerprint']:>18}"
              + ("" if r["deterministic_vs_first_layout"] else "  <-- DIFFERS"))

    best = max(report["layouts"], key=lambda r: r["evals_per_second"])
    band = [r for r in report["layouts"]
            if r["evals_per_second"] >= 0.95 * best["evals_per_second"]]
    # within 5% of the best, prefer the simpler layout: fewer workers, then fewer threads
    chosen = sorted(band, key=lambda r: (r["workers"], r["threads"]))[0]
    report["best_throughput"] = {"workers": best["workers"], "threads": best["threads"],
                                 "evals_per_second": best["evals_per_second"]}
    report["within_5pct_band"] = [f"{r['workers']}x{r['threads']}" for r in band]
    report["chosen_layout"] = {"workers": chosen["workers"], "threads": chosen["threads"],
                               "evals_per_second": chosen["evals_per_second"],
                               "rule": "highest throughput, then simplest layout within 5%"}
    print(f"\nbest {best['workers']}w x {best['threads']}t at "
          f"{best['evals_per_second']:.3f} evals/s; within 5%: "
          f"{report['within_5pct_band']}")
    print(f"chosen: {chosen['workers']} workers x {chosen['threads']} threads")

    # sustained interval at the chosen layout, to expose thermal or memory drift
    if args.sustained_min > 0:
        print(f"\nsustained run at the chosen layout for ~{args.sustained_min:.0f} min")
        reps = max(1, int(args.sustained_min * 60 * chosen["evals_per_second"] / len(configs)))
        segs = []
        with Sampler() as smp:
            t_start = time.perf_counter()
            for seg in range(4):
                r = run_layout(args.dataset, chosen["workers"], chosen["threads"],
                               configs, max(1, reps // 4), args.seed)
                r.pop("results")
                segs.append({"segment": seg, "evals_per_second": r["evals_per_second"]})
                print(f"    segment {seg}: {r['evals_per_second']:.3f} evals/s")
            total = time.perf_counter() - t_start
        first, last = segs[0]["evals_per_second"], segs[-1]["evals_per_second"]
        report["sustained"] = {"minutes": round(total / 60, 1), "segments": segs,
                               "degradation_pct": round(100 * (first - last) / first, 1),
                               "system": smp.summary()}
        print(f"  sustained {total/60:.1f} min, degradation "
              f"{report['sustained']['degradation_pct']:+.1f}%")

    (OUT / "audits").mkdir(parents=True, exist_ok=True)
    (OUT / "audits" / "stage_b_throughput.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT/'audits'/'stage_b_throughput.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
