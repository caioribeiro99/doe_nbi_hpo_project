#!/usr/bin/env python
"""Provenance reproduction of the dissertation DoE + factor-analysis stage on MAGIC.

Runs the code frozen at tag ``v0.1.0-dissertation`` -- not the article-track
rewrite -- over the version-controlled 88-run CCDFC design, so that the
historical pipeline can be re-executed and its outputs inspected rather than
taken on trust.

The dissertation code is extracted to a scratch directory and imported from
there, so nothing on the working tree shadows it and no historical artifact is
modified.

Outputs, under papers/xgboost_hpo_vrfnbi/audits/provenance/:
    doe_results.csv              88 design rows with their CV metrics
    factor_loadings.csv          what the dissertation calls loadings
    factor_scores.csv            rotated scores plus Score_Quality/Score_Cost
    rsm_coefficients_*.csv       backward-eliminated quadratic surfaces
    provenance_run.json          environment, seeds, checksums, timings

The dissertation pinned ``pandas>=2.1``. Under pandas 3 the frozen factor stage
raises, because copy-on-write makes the array it writes into read-only. Rather
than patch a historical artifact, run this script on an era-appropriate
interpreter:

    uv venv --python 3.11 /tmp/venv-diss
    VIRTUAL_ENV=/tmp/venv-diss uv pip install 'numpy<2' 'pandas<3' \
        'scikit-learn<2' 'xgboost>=2.0' 'scipy>=1.11' 'statsmodels>=0.14' tqdm
    /tmp/venv-diss/bin/python provenance_reproduce_dissertation.py

Usage:  python provenance_reproduce_dissertation.py [--frozen-src DIR] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "provenance"
DESIGN = REPO / "data" / "design" / "hyperparameter_design.csv"
RAW = REPO / "data" / "source" / "magic" / "raw" / "magic04.data"
DISS_TAG = "v0.1.0-dissertation"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_frozen(dest: Path) -> Path:
    """Extract the dissertation source tree at its tag into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "archive", DISS_TAG, "src/doe_xgb"],
                         cwd=REPO, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar.stdout, check=True)
    return dest / "src"


MAGIC_COLS = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
              "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]


def load_magic_raw() -> tuple[pd.DataFrame, pd.Series]:
    """Read magic04.data exactly as the manifest describes it: headerless CSV,
    ten numeric features, a final class column with {g, h} mapped to {0, 1}."""
    df = pd.read_csv(RAW, header=None, names=MAGIC_COLS)
    y = df.pop("class").map({"g": 0, "h": 1})
    if y.isna().any():
        raise ValueError("unmapped class labels in magic04.data")
    return df.astype(float), y.astype(int)


def load_design() -> pd.DataFrame:
    """The Minitab-exported CCDFC, written with ';' separators and ',' decimals."""
    df = pd.read_csv(DESIGN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-src", default="/tmp/diss_frozen_prov")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--reuse-doe", dest="reuse_doe", action="store_true", default=True,
                    help="reuse an existing doe_results.csv instead of re-evaluating (default)")
    ap.add_argument("--no-reuse-doe", dest="reuse_doe", action="store_false")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # The frozen tree and the article tree both provide a package called doe_xgb,
    # so only one can be importable. The frozen one wins: this is a reproduction of
    # the historical pipeline, and the data is read here directly rather than through
    # the article-track loader so that nothing from the rewrite enters the run.
    src = extract_frozen(Path(args.frozen_src))
    sys.path.insert(0, str(src))

    from doe_xgb.doe_runner import run_doe                    # frozen
    from doe_xgb.factor_analysis import run_factor_analysis   # frozen
    from doe_xgb.rsm import fit_rsm_backward                  # frozen
    from doe_xgb.config import PARAM_NAMES                    # frozen

    for mod in ("doe_xgb.doe_runner", "doe_xgb.factor_analysis", "doe_xgb.rsm"):
        got = sys.modules[mod].__file__
        assert str(src) in got, f"{mod} resolved to {got}, not the frozen tree"

    X, y = load_magic_raw()
    design = load_design()
    print(f"design {design.shape} | MAGIC X {X.shape} prevalence {float(y.mean()):.4f}")

    doe_csv = OUT / "doe_results.csv"
    if args.reuse_doe and doe_csv.exists():
        res = pd.read_csv(doe_csv)
        doe_seconds = 0.0
        print(f"reusing {doe_csv} ({len(res)} runs); pass --no-reuse-doe to re-evaluate")
    else:
        t_doe = time.time()
        res = run_doe(design, X, y, seed=args.seed, n_splits=args.n_splits)
        doe_seconds = time.time() - t_doe
        res.to_csv(doe_csv, index=False)
        print(f"DOE stage: {doe_seconds/60:.1f} min for {len(res)} runs")
    res = res.apply(pd.to_numeric, errors="coerce")
    # The frozen factor stage writes into the array returned by
    # ``np.asarray(df.loc[:, metrics])``. Under pandas 2.x, which the dissertation
    # pinned, that array is writable; under pandas 3 copy-on-write it is read-only and
    # the code raises. Run this script on an era-appropriate interpreter rather than
    # patching the frozen source -- see the header for how to build one.
    if tuple(int(v) for v in pd.__version__.split(".")[:1]) >= (3,):
        raise SystemExit(
            f"pandas {pd.__version__} cannot execute the frozen dissertation code "
            "(copy-on-write makes its metric array read-only). Build the era "
            "environment and rerun:\n"
            "    uv venv --python 3.11 /tmp/venv-diss\n"
            "    VIRTUAL_ENV=/tmp/venv-diss uv pip install 'numpy<2' 'pandas<3' \\\n"
            "        'scikit-learn<2' 'xgboost>=2.0' 'scipy>=1.11' 'statsmodels>=0.14' tqdm\n"
            "    /tmp/venv-diss/bin/python " + str(Path(__file__).name)
        )

    fa = run_factor_analysis(res)
    fa.loadings.reset_index().rename(columns={"index": "Metric"}).to_csv(
        OUT / "factor_loadings.csv", index=False)
    fa.scores.to_csv(OUT / "factor_scores.csv", index=False)

    scored = res.join(fa.scores)
    factors_df = scored[PARAM_NAMES].copy()
    models = {}
    for resp in ("Score_Quality", "Score_Cost"):
        m = fit_rsm_backward(factors_df, scored[resp], response_name=resp, alpha=0.05)
        pd.DataFrame({"Term": list(m.terms), "Coef": list(m.coefs)}).to_csv(
            OUT / f"rsm_coefficients_{resp.split('_')[1].lower()}.csv", index=False)
        models[resp] = {"r2": float(m.r2), "r2_adj": float(m.r2_adj),
                        "n_terms": int(len(m.terms))}
        print(f"{resp}: R2={m.r2:.4f} R2adj={m.r2_adj:.4f} terms={len(m.terms)}")

    manifest = {
        "purpose": "provenance reproduction of the dissertation DoE + FA + RSM stage",
        "dissertation_tag": DISS_TAG,
        "dissertation_commit": subprocess.run(
            ["git", "rev-parse", DISS_TAG + "^{commit}"], cwd=REPO,
            capture_output=True, text=True).stdout.strip(),
        "dataset": {"name": "magic", "rows": int(X.shape[0]), "features": int(X.shape[1]),
                    "prevalence": float(y.mean()), "raw_sha256": sha256(RAW)},
        "design": {"path": str(DESIGN.relative_to(REPO)), "runs": int(len(design)),
                   "sha256": sha256(DESIGN)},
        "protocol": {"seed": args.seed, "n_splits": args.n_splits,
                     "cv": "StratifiedKFold(shuffle=True)"},
        "factor_analysis": {"quality_factor": int(fa.quality_factor),
                            "cost_factor": int(fa.cost_factor),
                            "n_factors_returned": int(fa.loadings.shape[1])},
        "rsm": models,
        "timing_seconds": {"doe": doe_seconds, "total": time.time() - t0},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }
    try:
        import xgboost, sklearn
        manifest["environment"]["xgboost"] = xgboost.__version__
        manifest["environment"]["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    (OUT / "provenance_run.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUT}")
    print(f"total {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
