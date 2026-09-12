# Code availability statement

**Statement for the submission form.** "All code is openly available in a public Git repository under version
control, with annotated tags freezing the exact states used for the reported results. The analysis is checkpointed
and resumable, every stochastic component is seeded, and the library carries 83 unit tests."

## Repository and frozen states

Repository: `caioribeiro99/doe_nbi_hpo_project`.

| Tag | Freezes |
|---|---|
| `pco213-postwork-r10` | the ten-replication study, retained as the robustness reference |
| `pco213-postwork-r30` | the thirty-replication study reported here; every main number derives from this commit |
| `paper-draft-v1-pre-nsga2` | the manuscript as it stood before the external baseline was run |
| `paper-submission-v1` | the submitted state |

## Components

| Path | Contents |
|---|---|
| `src/mixens/` | mixture designs, Scheffe models, NBI on the simplex, fast metrics, Pareto tools, the NSGA-II baseline |
| `scripts/pco213_run_postwork_benchmark.py` | the checkpointed ten-stage benchmark runner |
| `scripts/pco213_postwork_benchmark_report.py`, `..._stats.py` | aggregation and replication-level statistics |
| `scripts/pco213_run_nsga2_baseline.py`, `pco213_nsga2_analysis.py`, `pco213_nsga2_stats.py` | the external baseline and its two-reference scoring |
| `papers/surrogate_nbi_ensemble/build_assets.py` | regenerates every manuscript figure and table from the frozen artifacts |
| `papers/surrogate_nbi_ensemble/audit_numbers.py` | screens every number in the compiled PDF against the source artifacts |
| `tests/mixens/` | 83 unit tests |

## Environment

Python 3.11.15 with NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, scikit-learn 1.9.0, statsmodels 0.15.0, XGBoost 3.2.0 and
pymoo 0.6.2, on macOS arm64.

## Reproduction cost

Reproducing the aggregate results from the published tables requires only the repository. Reproducing from raw data
requires the four datasets, the pinned environment and roughly 81 hours of single-workstation compute for the main
study plus 15 hours for the external baseline, or proportionally less for a subset of datasets or replications, since
both runners are resumable and parameterized by dataset and replication count.
