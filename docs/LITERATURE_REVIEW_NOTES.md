# Literature review notes — HPO families & comparative protocol

Internal notes from the literature pass that drove
`docs/COMPARATIVE_PROTOCOL.md` and the rewrite of
`article/sections/02_related_work.tex`. Not part of the manuscript;
kept under `docs/` so future commits can extend the citation set
without re-doing the survey.

## Family-by-family map

### A. Classical HPO

| key reference | citation key | role |
|---|---|---|
| Bergstra & Bengio, *JMLR* 13, 2012 | `bergstra2012random` | random search baseline |
| Bergstra et al., *NeurIPS* 24, 2011 | `bergstra2011tpe` | TPE algorithm itself |
| Snoek, Larochelle & Adams, *NeurIPS* 25, 2012 | `snoek2012bayesian` | practical GP-BO |
| Shahriari et al., *Proc. IEEE* 104(1), 2016 | `shahriari2016bayesian` | BO survey |
| Frazier, *arXiv* 1807.02811, 2018 | `frazier2018tutorial` | BO tutorial |
| Probst, Boulesteix & Bischl, *JMLR* 20, 2019 | `probst2019tunability` | GBDT tunability |

Implementation choice for the protocol: **Optuna TPESampler**
\citep{akiba2019optuna} as the canonical TPE backend; **scipy.stats**
uniform draws as the random-search backend.

### B. SMAC family

| key reference | citation key | role |
|---|---|---|
| Hutter, Hoos & Leyton-Brown, *LION 5*, 2011 | `hutter2011smac` | original SMAC paper |
| Lindauer et al., *JMLR* 23, 2022 | `lindauer2022smac3` | SMAC3 implementation |

Implementation: **SMAC3** (`smac>=2.0`). RF surrogate; intensification
handles noisy CV evaluations naturally.

### C. Multi-fidelity HPO

| key reference | citation key | role |
|---|---|---|
| Li et al., *JMLR* 18(185), 2018 | `li2017hyperband` | Hyperband |
| Li et al., *MLSys* 2, 2020 | `li2020asha` | ASHA |
| Falkner, Klein & Hutter, *PMLR* 80, 2018 | `falkner2018bohb` | BOHB |
| Awad, Mallik & Hutter, *IJCAI* 2021 | `awad2021dehb` | DEHB |

Fidelity dimension for GBDTs: number of boosting iterations
(`n_estimators` in XGBoost / LightGBM, `iterations` in CatBoost).
Open question pending freeze: ASHA vs Hyperband for the protocol
(ASHA recommended for parallelism on the dedicated Mac).

### D. Evolutionary / derivative-free HPO

| key reference | citation key | role |
|---|---|---|
| Storn & Price, *J. Global Optim.*, 1997 | `storn1997differential` | DE |
| Hansen & Ostermeier, *Evol. Comput.*, 2001 | `hansen2001cmaes` | CMA-ES |
| Rapin & Teytaud, GitHub, 2018 | `rapin2018nevergrad` | Nevergrad framework |

DE underpins DEHB; CMA-ES is cited but not directly benchmarked in
the protocol (we already have evolutionary-MOO coverage via NSGA-II;
adding CMA-ES would inflate the campaign without adding methodological
contrast). Nevergrad is referenced as a framework, not a baseline.

### E. Multi-objective HPO

| key reference | citation key | role |
|---|---|---|
| Deb et al., *IEEE TEC* 6(2), 2002 | `deb2002nsga2` | NSGA-II |
| Deb & Jain, *IEEE TEC* 18(4), 2014 | `deb2014nsga3` | NSGA-III (ref-points) |
| Knowles, *IEEE TEC* 10(1), 2006 | `knowles2006parego` | ParEGO |
| Ozaki et al., *GECCO* 2020 | `ozaki2020motpe` | MOTPE |
| Das & Dennis, *SIAM J. Optim.* 8(3), 1998 | `das1998nbi` | true NBI (proposed method) |
| Morales-Hernández et al., *AIR* 56, 2023 | `morales2022survey` | MOO HPO survey |
| Blank & Deb, *IEEE Access* 8, 2020 | `blank2020pymoo` | pymoo implementation |
| Pereira et al., *EAAI* 162, 2025 | `pereira2025eaai` | VRF + FMSE + NBI antecedent |

Implementation choices: **pymoo** for NSGA-II (and NSGA-III later if
needed); **Optuna MOTPESampler** for MOTPE; ParEGO via SMAC's MO
extension or pymoo, on the CC18 subset only.

### F. AutoML systems (contextual)

| key reference | citation key | role |
|---|---|---|
| Feurer et al., *NeurIPS* 28, 2015 | `feurer2015autosklearn` | Auto-sklearn |
| Wang et al., *MLSys* 3, 2021 | `wang2021flaml` | FLAML |
| Erickson et al., *arXiv* 2003.06505, 2020 | `erickson2020autogluon` | AutoGluon |

All three are pipeline-AutoML or wider-than-GBDT AutoML systems.
**Auto-sklearn** is excluded from the protocol because its search
space is not aligned with per-algorithm GBDT HPO. **AutoGluon** is
similarly excluded. **FLAML** is the only one of the three that can
be coerced into single-algorithm GBDT tuning under a wall-clock
budget; whether to include it is the most consequential open item
in `docs/COMPARATIVE_PROTOCOL.md`.

## OpenML / OpenML-CC18 anchor

| key reference | citation key | role |
|---|---|---|
| Vanschoren et al., *ACM SIGKDD Explor.* 15(2), 2014 | `vanschoren2014openml` | OpenML platform |
| Bischl et al., *NeurIPS Datasets & Benchmarks* 1, 2021 | `bischl2021openmlbenchmark` | OpenML benchmark suites incl. CC18 |

Protocol relies on these for: (i) task identifiers (so a "task"
means the same thing across implementations), (ii) suite definition
(`suite_id = 99`), (iii) per-task default folds.

## TODO references (verify before submission)

The following entries in `article/references.bib` carry a `TODO`
marker because at least one of {volume, number, pages, DOI, venue}
could not be quickly verified:

- `bischl2021openmlbenchmark` — exact volume / pages of the *NeurIPS
  Datasets and Benchmarks* track entry.
- `rapin2018nevergrad` — the appropriate cite-able artifact for
  Nevergrad changes over time; settle on the release tag at
  submission.

The following entries are confident enough to ship without further
verification but should still be sanity-checked at proof stage:

- All other entries with explicit DOIs.

## Methods we *did not* add

Considered and excluded, with reason:

- **TPE-Hyperopt directly** — superseded by `optuna_tpesampler`
  (same algorithm, cleaner integration, more active maintenance).
- **scikit-optimize `BayesSearchCV`** — superseded by SMAC3 (RF
  surrogate handles mixed-type spaces more naturally) and by Optuna
  GP samplers if a GP-BO baseline is later required.
- **CMA-ES directly** — DEHB already exercises an evolutionary inner
  loop within a multi-fidelity schedule, and NSGA-II covers
  evolutionary multi-objective search; adding CMA-ES alone would
  inflate the campaign without changing the family coverage.
- **Direct Hyperopt** — same algorithm as `tpe_optuna` from a
  reviewer's standpoint; pick one.

## Open items

These are also captured at the bottom of
`docs/COMPARATIVE_PROTOCOL.md` as the freeze gate for the SQLite
shard generator:

1. FLAML inclusion (literature-only vs single-algorithm GBDT
   baseline).
2. ASHA vs Hyperband for the multi-fidelity slot.
3. ParEGO subset definition.
4. Verify TODO references in `article/references.bib`.
