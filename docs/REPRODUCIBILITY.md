# Reproducibility guide

This branch is engineered to be reproducible end-to-end at three
fidelity levels.

## Level 1 — Synthetic smoke

Goal: ≤ 1 minute, no external data. Used by CI and as a sanity check.

```bash
make install-dev
make test-unit
make test-methodology
make smoke
```

## Level 2 — Reduced reproduction

Goal: a few minutes. Three replicas, deterministic XGBoost, small CV
budget. Demonstrates the pipeline qualitatively on the canonical
dataset (or a synthetic stand-in if MAGIC is not available).

```bash
make data            # downloads MAGIC, verifies SHA-256
make repro-mini      # = python -m doe_xgb.cli run --config configs/reduced_repro.yaml
```

## Level 3 — Full reproduction

Goal: matches the dissertation tables (within determinism caveats).
Long: 30 replicas, full benchmark suite, `tree_method="exact"` for the
canonical headline run.

```bash
make repro-full      # = python -m doe_xgb.cli run --config configs/dissertation_baseline_xgb_magic.yaml
make tables          # regenerates aggregated tables under experiments/
```

## End-to-end article-track smoke (Commit 23)

A reduced article-track smoke is committed for sanity checks:

```bash
python scripts/run_article_true_nbi_magic_smoke.py
```

Single replica on MAGIC + XGBoost, q=2 NBI, simplex_lattice {2, 10}.
Drives the full pipeline (DOE → FA → RSM → true N-objective NBI →
confirmation → conditional MBPA), uses
`evaluation.assert_metric_set_compatible_with_task` for the binary
guardrail, and writes
`experiments/_v1_smoke/article_true_nbi_magic_smoke.{json,md}` with
per-stage timings, NBI residual statistics, and the MBPA decision.
Total wall-clock ~86 s on Apple Silicon. The legacy weighted-sum
scalarization is never invoked.

## What every replica writes

```
experiments/<dataset>/<design>/replica_XX/
├── manifest.json                       # seeds, sha256, config, system fingerprint
├── doe_results.csv
├── doe_results_with_scores.csv
├── factor_loadings.csv
├── factor_scores.csv
├── factor_diagnostics.json
├── rsm_coefficients_<obj>.csv
├── nbi_anchors.csv
├── nbi_chim.json
├── nbi_candidates.csv
├── nbi_subproblem_diagnostics.csv
├── nbi_candidate_evaluations.csv
├── confirmation_vrf.csv
├── frontier_quality.json
├── post_optimization_diagnostics.json   # always written; describes whether MBPA fired
├── post_optimization_mixture_fit.csv    # only if MBPA fired
├── post_optimization_refined_candidate.csv
├── confirmation_summary.csv
├── fold_metrics.csv
└── run_replica.log
```

## System fingerprint

`manifest.json` includes platform, CPU model, RAM, OS, Python version,
and a `pip freeze` snapshot. This makes "I cannot reproduce" tickets
diagnosable.

## Deterministic mode

For bit-stable headline tables, set `experiment.deterministic: true` in
the YAML config or pass `--deterministic` on the CLI. This forces
`tree_method="exact"` and `n_jobs=1`. It roughly doubles wall-time on
laptop-class hardware and is the recommended setting for the article
tables.
