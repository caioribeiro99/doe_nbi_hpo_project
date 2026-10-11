#!/usr/bin/env python
"""Per-replication factor refit against the frozen model: Tucker congruence (S4.4).

Purpose. EXPERIMENT_PROTOCOL.md section 7.2 and amendment 14 freeze one factor model
per dataset, applied to every replication, and declare the per-replication refit as
a reported sensitivity: the Tucker congruence between the frozen model's rotated
loadings and those of the model each replication's own 88 design rows would have
produced. The refit is never applied to anything. This script tabulates that
sensitivity per dataset in two forms.

* ``greedy_recorded``: the congruences the runner recorded in each unit's
  ``factor_model.json`` (``per_replication_refit_sensitivity``), computed by the
  frozen ``tucker_congruence``. That function matches columns greedily and returns
  the signed congruence of each pick, so it does not resolve factor sign and a
  negative value can be an artifact of the matching. Values are read, not recomputed;
  the script only checks that ``fit_factor_model`` on the stored design rows
  reproduces them.
* ``optimal`` (post hoc): the refit (``fit_factor_model`` on the unit's stored 88
  design rows, the same call the runner made) is matched to the frozen columns by the
  assignment, over all 3! permutations, that maximizes the total absolute
  congruence; each matched refit column is then oriented to agree in sign with its
  frozen column before the congruence is computed.

Definitions of the ``optimal`` fields, per dataset over its replications:
  min_phi                minimum oriented congruence over replications and factors;
  n_below_0_95, _0_85    replications whose smallest oriented congruence is below
                         0.95 (conventional equivalence) or 0.85 (fair similarity);
  n_negative             replications in which an optimally matched pair had a
                         negative congruence before the orientation step, that is,
                         where the refit's own role-block orientation (amendment 3b)
                         disagreed with the frozen model's;
  cost_factor_recovered  replications in which the optimal assignment pairs the
                         frozen cost factor with the refit's own cost-role factor.
``greedy_recorded.replications_with_negative_phi`` lists the zero-based replication
indices (``rep_XX``) with any negative recorded congruence.

Provenance label: retrospective (declared in the frozen protocol, section 7.2 and
amendment 14; computed after the campaign from stored outputs); the
optimal-assignment matching rule is post hoc.

Inputs (repo-relative):
    experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/design.json
    experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/factor_model.json
    papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json
Output:
    papers/xgboost_hpo_vrfnbi/analysis/tucker_congruence_refit.json

Reads stored outputs only; runs no learner, optimizer or surrogate fit. The refit is
the frozen PCA/Varimax arithmetic on stored design-row responses.

Usage, from the repository root:
    python -B papers/xgboost_hpo_vrfnbi/scripts/tucker_congruence_refit.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.factor_model import (fit_factor_model,  # noqa: E402
                                           load_reference_factor_model,
                                           tucker_congruence)
from doe_xgb.campaign.runner import DATASETS, N_REPLICATIONS  # noqa: E402

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis" / "tucker_congruence_refit.json"
SCRIPT = pathlib.Path(__file__).resolve().relative_to(REPO).as_posix()
LABEL = ("retrospective (declared in the frozen protocol, section 7.2 and amendment 14; "
         "computed after the campaign from stored outputs); the optimal-assignment "
         "matching rule is post hoc")
EQUIVALENCE, FAIR_SIMILARITY = 0.95, 0.85    # the conventional reading the runner records
REPRODUCTION_TOL = 1e-9                      # recomputed greedy vs recorded greedy


def congruence_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C[j, i]: Tucker congruence of frozen column j with refit column i.

    The normalized inner product the frozen ``tucker_congruence`` computes before
    its greedy pick, kept whole so that every assignment can be scored.
    """
    na = np.linalg.norm(a, axis=0)
    nb = np.linalg.norm(b, axis=0)
    return (a.T @ b) / np.outer(np.where(na == 0, 1, na), np.where(nb == 0, 1, nb))


def optimal_assignment(C: np.ndarray) -> tuple[int, ...]:
    """The permutation p maximizing sum_j |C[j, p[j]]|, by enumeration (k = 3)."""
    k = C.shape[0]
    return max(itertools.permutations(range(k)),
               key=lambda p: sum(abs(C[j, p[j]]) for j in range(k)))


def oriented_congruence(a: np.ndarray, b: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    """Flip each matched refit column to agree in sign with its frozen column, then
    compute the congruence of every matched pair."""
    out = np.zeros(a.shape[1])
    for j, i in enumerate(perm):
        bj = b[:, i] * (np.sign(a[:, j] @ b[:, i]) or 1.0)
        out[j] = float(congruence_matrix(a[:, [j]], bj[:, None])[0, 0])
    return out


def assert_frozen_model_applied(unit_model: dict, frozen) -> None:
    """The unit's factor_model.json must be the committed per-dataset model."""
    for key, value in (("rotated_loadings", frozen.rotated_loadings),
                       ("varimax_rotation", frozen.rotation),
                       ("pca_components", frozen.components),
                       ("standardization_mean", frozen.mu),
                       ("standardization_sd", frozen.sd)):
        if not np.array_equal(np.asarray(unit_model[key], dtype=float), value):
            raise SystemExit(f"stored factor model differs from the frozen one in {key}")
    if int(unit_model["cost_factor_index"]) != frozen.cost_index:
        raise SystemExit("stored cost factor index differs from the frozen model")


def main() -> int:
    out = {"script": SCRIPT, "label": LABEL, "datasets": {}}
    max_dev = 0.0
    for ds in DATASETS:
        frozen = load_reference_factor_model(ds)
        A = frozen.rotated_loadings
        rec_min_abs, rec_negative = [], []
        opt_min, opt_negative, cost_recovered = [], 0, 0
        for rep in range(N_REPLICATIONS):
            unit = ROOT / ds / f"rep_{rep:02d}"
            unit_model = json.loads((unit / "factor_model.json").read_text())
            assert_frozen_model_applied(unit_model, frozen)
            sens = unit_model["per_replication_refit_sensitivity"]
            phi_rec = np.asarray(sens["tucker_congruence"], dtype=float)
            rec_min_abs.append(float(sens["min_abs_congruence"]))
            if (phi_rec < 0).any():
                rec_negative.append(rep)

            # The runner's own call, repeated on the stored design rows.
            design_df = pd.DataFrame(json.loads((unit / "design.json").read_text())["rows"])
            refit = fit_factor_model(design_df)
            B = refit.rotated_loadings
            max_dev = max(max_dev, float(np.abs(tucker_congruence(A, B) - phi_rec).max()))
            if int(sens["cost_index_refit"]) != refit.cost_index:
                raise SystemExit(f"{ds} rep {rep}: refit cost index not reproduced")

            C = congruence_matrix(A, B)
            perm = optimal_assignment(C)
            if any(C[j, i] < 0 for j, i in enumerate(perm)):
                opt_negative += 1
            phi_opt = oriented_congruence(A, B, perm)
            opt_min.append(float(phi_opt.min()))
            cost_recovered += int(perm[frozen.cost_index] == refit.cost_index)

        if max_dev > REPRODUCTION_TOL:
            raise SystemExit(f"{ds}: recomputed greedy congruence deviates from the "
                             f"recorded values by {max_dev:.3g}")
        opt_min_a = np.asarray(opt_min)
        out["datasets"][ds] = {
            "n_replications": N_REPLICATIONS,
            "greedy_recorded": {
                "min_abs_phi": float(min(rec_min_abs)),
                "replications_with_negative_phi": rec_negative},
            "optimal": {
                "min_phi": float(opt_min_a.min()),
                "n_below_0_95": int((opt_min_a < EQUIVALENCE).sum()),
                "n_below_0_85": int((opt_min_a < FAIR_SIMILARITY).sum()),
                "n_negative": int(opt_negative),
                "cost_factor_recovered": int(cost_recovered)}}

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"max |recomputed greedy - recorded| over {N_REPLICATIONS * len(DATASETS)} "
          f"units: {max_dev:.3g}\n")
    print(f"{'dataset':15} {'greedy min|phi|':>15} {'greedy neg reps':>18} "
          f"{'opt min':>8} {'<0.95':>6} {'<0.85':>6} {'neg':>4} {'cost':>5}")
    for ds in DATASETS:
        v = out["datasets"][ds]
        g, o = v["greedy_recorded"], v["optimal"]
        print(f"{ds:15} {g['min_abs_phi']:15.3f} "
              f"{str(g['replications_with_negative_phi']):>18} {o['min_phi']:8.3f} "
              f"{o['n_below_0_95']:6} {o['n_below_0_85']:6} {o['n_negative']:4} "
              f"{o['cost_factor_recovered']:5}")
    print(f"\nwrote {OUT.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
