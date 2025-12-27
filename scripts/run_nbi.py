#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from doe_xgb.nbi import load_coefficients_csv, run_nbi_weighted_sum, save_nbi_candidates


def main():
    p = argparse.ArgumentParser(description="Generate NBI candidates from RSM coefficients")
    p.add_argument("--replica_dir", required=True, help="Replica output directory (contains doe_results_with_scores + rsm_coefficients_*.csv)")
    p.add_argument("--seed", type=int, required=True, help="Replica seed")
    p.add_argument("--beta_step", type=float, default=0.05)
    p.add_argument("--n_starts", type=int, default=10)
    args = p.parse_args()

    rep_dir = Path(args.replica_dir)
    scored_path = rep_dir / "doe_results_with_scores.csv"
    coef_q_path = rep_dir / "rsm_coefficients_quality.csv"
    coef_c_path = rep_dir / "rsm_coefficients_cost.csv"
    if not scored_path.exists():
        raise FileNotFoundError(scored_path)
    if not coef_q_path.exists() or not coef_c_path.exists():
        raise FileNotFoundError("Missing RSM coefficients. Run scripts/run_fa_rsm.py first.")

    df = pd.read_csv(scored_path, sep=';', decimal=',')

    utopia = (float(df["Score_Quality"].max()), float(df["Score_Cost"].max()))
    nadir = (float(df["Score_Quality"].min()), float(df["Score_Cost"].min()))

    m1 = load_coefficients_csv(str(coef_q_path))
    m2 = load_coefficients_csv(str(coef_c_path))

    candidates = run_nbi_weighted_sum(
        m1,
        m2,
        observed_utopia=utopia,
        observed_nadir=nadir,
        beta_step=args.beta_step,
        seed=args.seed,
        n_starts=args.n_starts,
        constrain_pred_range=True,
    )

    out_path = rep_dir / "nbi_candidates.csv"
    save_nbi_candidates(candidates, str(out_path))
    print(f"✅ NBI candidates saved: {out_path}")


if __name__ == "__main__":
    main()
