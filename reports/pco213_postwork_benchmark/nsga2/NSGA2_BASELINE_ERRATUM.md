# Erratum to `NSGA2_BASELINE_REPORT.md`

Dated 2026-10-11, revision branch `revision/paper1-matheus-review`. The report beside this file, the executed code,
`papers/surrogate_nbi_ensemble/nsga2_preregistered_config.json` and every tag are unchanged; this note records how
they must be read.

**The NSGA-II baseline was not evaluation-matched to NBI-C.** Its termination target is NBI-C's recorded counter
`n_real_objective_evals`, which rises by two per weight-vector evaluation: one ROC-AUC computation and one log-loss
computation (`scripts/pco213_run_postwork_benchmark.py`, lines 557-562, the `f_auc` and `f_ll` closures passed to
`src/mixens/nbi.py`). NSGA-II's own counter rises by one per weight vector, each of which computes both objectives
(`src/mixens/nsga2_baseline.py`, line 92: `counter.n += len(W)`).

The realized ratio 0.999996 (50,150,166 / 50,150,376) reported in Section 2 is therefore a ratio of weight vectors to
scalar objective computations. NSGA-II evaluated 1.999992 times as many weight vectors as NBI-C's solve (per run
between 1.99985 and 2.00020), and about 1.7 times as many (1.672 to 1.681 per dataset) if NBI-C is also charged the
ROC-AUC anchor search recorded in `papers/surrogate_nbi_ensemble/tables/tab05_compute.csv`. Sources:
`nsga2_runs.csv` (actual evaluations) and `reports/pco213_postwork_benchmark/tables/nbi_runs.csv`
(`n_real_objective_evals`).

Read accordingly:

- the title, the opening question, Section 2, check #1, check #9 and Section 9 ("matched real-objective budget",
  "equal evaluation count", "evaluation-matched"): the comparison is matched neither in weight-vector evaluations nor
  in wall clock;
- the about 2.2 times wall-clock ratio (median 2.157 to 2.252 per dataset, `nsga2_budget_runtime.csv`) is twice the
  weight-vector evaluations at a similar time per evaluation (per-vector ratio 1.08 to 1.13), not an amortization
  effect;
- the headline describes NSGA-II under this budget. Whether it leads at an equal number of weight-vector evaluations
  was not tested.

The docstrings and comments of `src/mixens/nsga2_baseline.py` (lines 1, 17-18, 161),
`scripts/pco213_run_nsga2_baseline.py` (lines 2-6, 94), `scripts/pco213_nsga2_stats.py` (line 2) and
`scripts/pco213_nsga2_launch.sh` (line 2) carry the same description; they are executed code and are left unchanged.
The manuscript (Section 5.5, Table 8 and supplementary Section S13) states the corrected units.
