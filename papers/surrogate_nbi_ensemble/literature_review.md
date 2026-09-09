# Literature review — surrogate-assisted NBI over a classifier-ensemble weight simplex

Prepared 2026-09-09 from the seven verified cluster reviews in `lit/` (`clusterA_ensemble_weighting.md`,
`clusterB_moo_ensemble.md`, `clusterC_mixture_ml.md`, `clusterDG_surrogate_moo.md`, `clusterE_nbi_mixture_rsm.md`,
`clusterF_cost_aware.md`, `foundational_methods.md`) and from `predecessor_lineage.md`, which documents the authors'
own prior work. Every bibkey used below exists in one of those cluster `.bib` files and in the manuscript's merged
`references.bib`.

---

## 1. Scope of the search

**Clusters searched.** Seven, run as independent verified passes:

| Cluster | Question it was asked |
|---|---|
| A — ensemble weight optimization | Are classifier-combination weights already optimized as simplex-constrained variables, and with what search? |
| B — multiobjective ensembles | Has anyone built a Pareto front over ensemble *combination weights* (as opposed to member generation or selection)? |
| C — mixture designs / Scheffé in machine learning | Has a formal mixture design and a Scheffé polynomial been used on ML weights or proportions? |
| D+G — surrogate-assisted MOO, surrogate and anchor failure | What is known about surrogate error, payoff-matrix/anchor error and NBI's geometric limits? |
| E — NBI with mixture DoE and RSM | What exactly does the de Paiva/Balestrassi lineage already do, and who outside it combines NBI with a weight simplex? |
| F — cost-aware / latency-aware ensembles | Is inference or deployment cost already a Pareto objective for weighted ensembles, and how is it modelled? |
| Foundational methods | The verified reference list for the method, indicators, statistics, software and data. |

**Sources used.** Crossref (per-DOI records and bibliographic search), OpenAlex (per-DOI records and
`abstract_inverted_index` reconstruction), Semantic Scholar Graph (per-paper, `batch`, and `tldr` endpoints), arXiv
API, OpenAIRE, DataCite, publisher landing pages and first-page PDFs (Elsevier `sdfe` endpoint, JoVE, SciELO, USENIX,
JMLR, PMLR), Google Books full-text snippet search, OpenLibrary, and the UNIFEI institutional repository. Publisher
sites were frequently bot-blocked (Cloudflare/Incapsula challenges on Wiley, Springer IdP redirects, ScienceDirect
CAPTCHAs); where no abstract could be retrieved from any source, the cluster reviews say so explicitly and the
method-level fields are marked "not determinable from the retrieved record" rather than guessed. Those cases are
flagged again below wherever they affect an overlap judgement.

**Coverage window.** Foundational work (Brier 1950; Wilcoxon 1945; Scheffé 1958, 1963) through September 2026, with
2024, 2025 and 2026 items searched explicitly. The most recent directly relevant items are `maier2026hapens`,
`akkerman2026pace`, `galvan2026simultaneous`, `moradpour2026ensemble`, `herrmann2026nonextreme`, `xu2026pseo`,
`hassan2026weighted` and `karl2026svemnet` (2026), and `rocha2025ensemblestlf`, `leal2025mixtureweights`,
`ding2025moeec` and the LLM data-mixing line (2025).

**Counts.** 137 records were verified across the seven clusters plus the predecessor lineage. Of these, **60 are
directly relevant** — they share at least one of: combination weights as decision variables, a mixture design or
Scheffé model over a simplex, NBI, surrogate-assisted multiobjective optimization with an anchor construction, or
inference/deployment cost as an optimization criterion — and appear as rows of the comparison table in §2. A further
~60 foundational method, indicator, statistics, software and dataset citations are listed in §3. About 15 further
verified records (multiobjective ensemble *selection* and *generation* papers, engineering NBI variants, cascade and
budgeted-learning papers) are discussed in the prose of §4 but are not tabulated, because their decision variables are
neither combination weights nor simplex proportions.

**Records that could not be fully read.** `kwon2024ensemble` (no abstract in any index; read from the publisher's
Korean-language HTML, one constraint rendered only as an image), `onan2016multiobjective` and `zhang2011sparse`
(Elsevier blocked; `zhang2011sparse` was later read from a first-page PDF in cluster F), `galvan2026simultaneous`
(MDPI 403 — the MOEA name and whether the voting weights are normalized are **unverified**),
`streitenberger2022nbifa` and `vasquezramos2025rsmxgboost` (title-level only), `karl2026svemnet` and
`incerti/paula2019gatuning` abstracts (recovered for the latter via OpenAIRE). Overlap judgements that depend on these
records are hedged accordingly.

**Bibkey hygiene note.** Six papers were independently verified by two cluster agents and carry two bibkeys apiece.
The manuscript's `references.bib` has already resolved them; the canonical keys are `das1998nbi` (not
`dasdennis1998nbi`), `messac2003normalized` (not `messac2003nnc`), `ishibuchi2015modified` (not `ishibuchi2015igdplus`),
`boxdraper2007response` (not `boxdraper2007ridge`), `rocha2025ensemblestlf` (not `rocha2025ensemble`) and
`paula2019gatuning` (not `incerti2019gatuning`). Only the canonical keys are used here.

---

## 2. Comparison table — the 60 directly relevant studies

Legend for the short cells: **Y** = yes, **N** = no, **P** = partial, **N.A.** = not applicable, **n/s** = the
verified record does not state it (never inferred). "EW" = ensemble/combination weights are decision variables.
Rows marked **OWN** are the authors' own prior work; rows marked **GRP** are the de Paiva/Balestrassi group lineage
the study inherits from.

| Citation | Research domain | Decision variables | Objectives | Optimization algorithm | Ensemble weights? | Weights on simplex? | Mixture DoE? | Scheffe/RSM? | Surrogate MOO? | NBI? | Real anchors? | Real-objective revalidation? | Deployment cost? | Replication protocol | Datasets/problems | Closest overlap | Key difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pereira2025hybrid` **OWN** | Hard turning; post-Pareto MCDM | 3 machining factors (CCD) | 3 rotated factor-score (VRF/FMSE) surfaces | NBI on RSM, then 2nd NBI on Scheffé models of GD/entropy | N (β only) | Y (β) | Y (over β) | Y (RSM + Scheffé 4th order) | Y | Y | N (surrogate payoff) | n/s | N (process cost) | Single 19-run CCD, one case | AISI H13 turning | The DoE+RSM+NBI pipeline, lattice β, Scheffé on a simplex | Mixture design acts on scalarization weights, not decision variables; no ML metrics, gate, reference or replication |
| `ribeiro2026dissertation` **OWN** | XGBoost hyperparameter MOO | XGBoost hyperparameters (CCD) | Quality and cost factor scores | NBI on quadratic RSM, 66 lattice weights | N | Y (β, lattice {3,10}) | Y (over β) | RSM Y; Scheffé not named | Y | Y | N | **Y** (candidates re-run on the real model) | P (training/exec runtime) | Replicated runs + 2 extra datasets | 1 canonical + 2 tabular | 66-point {3,10} lattice; "re-evaluate on the real model"; ML metrics | Hyperparameters, not simplex weights; no AUC/log-loss over w, no real anchors, no NBI-C, no gate, no reference |
| `bacci2019nbi_mixture` **GRP** | Time-series forecast combination | Forecast-combination weights | Rotated factor scores of residual metrics | NBI on Scheffé mixture models; entropy+GPE | **Y** | **Y** | **Y** (simplex-lattice) | **Y** | Y | **Y** | n/s | n/s | N | 3 real + simulated series | Brazilian coffee demand | Simplex weights as mixture components + Scheffé + NBI — the structural precedent | Forecasts, not classifiers; latent factor objectives; no real anchors, reference, gate, cost or replication |
| `mendes2016portfolio_mde` **GRP** | Portfolio optimization (ARMA–GARCH) | Asset proportions | Return, risk, entropy | Desirability on mixture models | P (portfolio) | Y | Y | Y | Y | N | N.A. | n/s | N | Single dataset | Weekly crude-oil prices | Mixture DoE over simplex proportions | Finance; desirability not NBI; no ML, anchors or reference |
| `moreira2021ann_ensemble_mde` **GRP** | PV forecasting; ANN ensembles | ANN ensemble weights | MAPE (single) | Mixture-DoE modelling of the weights | **Y** | **Y** | **Y** | Y (implied) | N | N | N.A. | n/s | N | n/s | PV series, Minas Gerais | Ensemble weights as mixture components optimized through a mixture design | Single-objective regression; no MOO, NBI, cost or reference |
| `leal2022portfolio_doptimal_mde` **GRP** | Energy-asset portfolio | 5 asset weights | Return, variance, entropy | Desirability on FA/FMSE-reduced mixture models | P | Y | Y (D-optimal subset of {5,10}) | Y | Y | N | N.A. | n/s | N | Single dataset | Energy price/return series | Large simplex-lattice on 5 components + budget reduction | Desirability; finance; no anchors, reference or replication |
| `pereira2026postpareto` **GRP** | Hard turning | Vc, f, ap | 3 rotated-factor surfaces | NBI on RSM + entropy/GPE; vs NSGA-II, MOEA/D, WS, MOLA | N | Y (β) | n/s | RSM Y | Y | Y | N | n/s | N | Single case + weight sensitivity | AISI H13 turning | HV/IGD/spacing used to compare NBI against MOEAs | Indicators compare optimizers, not revalidated candidates against an empirical reference |
| `azevedo2026nbivrf` **GRP** | CFD hydrodynamics | CFD/hydraulic parameters | Rotated-factor RSM surfaces | NBI + simplex-lattice weights; vs NSGA-II, MOEA/D | N | Y (β) | Y (over β) | RSM Y | Y | Y | n/s | P (prediction–experiment correlation) | N | Single case | Hydraulic piping CFD | Lattice-weighted NBI-VRF as a current group template | Scalarization weights; no ML, anchors or reference front |
| `oliveira2011portfolio` **GRP** | Electricity-contract portfolio | Contract proportions | CVaR-adjusted mean and variance | Desirability on mixture models | P | Y | Y | Y | Y | N | N.A. | n/s | N | Single case | Brazilian energy market | The group's founding "mixture design over a weight simplex" | Financial responses; no NBI, gate or revalidation |
| `lopes2016rpdmnbi` **GRP** | Robust parameter design, end milling | Machining factors + noise | PCA-based mean/variance surrogates | Multivariate NBI on RSM | N | Y (β) | N | RSM Y | Y | Y | N | **Y** (L9 confirmation at 3 weight settings) | N | Single CCD + L9 array | End milling, combined arrays | Physical confirmation of surrogate-NBI points — the lineage precedent for revalidation | Three points confirmed, not every candidate; no reference front; no simplex decision space |
| `rocha2017robustmcdm` **GRP** | Hard turning, robust MCDM | Vc, f, d | Tool life, Ra, MRR/Fc (RSM) | NBI **with mixture DoE**, entropic selection | N | Y (β) | **Y** | RSM Y; Scheffé in substance | Y | Y | N | n/s | N | Single design | AISI H13 with PCBN wiper | Explicit pre-2025 "NBI along with MDE" statement in the lineage | MDE on scalarization weights; no ML objectives, revalidation or reference |
| `monticeli2017portfolio` **GRP** | Power-generation portfolio | Generation-asset proportions | Cost, risk (+entropy) | Desirability on mixture models | P | Y | Y | Y | Y | N | N.A. | n/s | N | **Computational replicas per design point** + moving windows | California market | Replicated evaluation of each simplex design point; entropy on weights | Replicas capture series volatility, not learner resampling; no NBI, gate or reference |
| `paula2019gatuning` **GRP** | GA tuning + welding optimization | Objective weights (mixture) × 3 GA hyperparameters | 4 welding responses via global criterion | GA, analysed by a mixture-process DoE | N | Y | Y (mixture–process) | Y | P | N | N.A. | n/s | N | n/s | FCAW welding | Mixture DoE whose response is an *algorithm's* output | Components are objective weights; no ensemble, NBI or Pareto revalidation |
| `rocha2021robustpoint` **GRP** | Multiresponse RSM, post-Pareto MCDM | Process factors; mixture vars = scalarization weights | Entropy/error, diversity/error, prediction variance | RSM MOO + mixture metamodels of MCDM indicators | N | Y | Y | Y | Y | not named | N | n/s | N | n/s | RSM case study | Direct antecedent of the predecessor's MBPA step | Mixture surfaces model a *decision indicator*, never the objectives as functions of simplex decision variables |
| `almeida2022taguchinbi` **GRP** | Welding/cladding quality–cost | Welding parameters | PCA-based Taguchi loss functions | NBI on RSM/PCA + total-loss (cost) post-Pareto rule | N | Y (β) | N | RSM Y | Y | Y | N | n/s | P (process cost in the selection rule) | Single design | FCAW cladding | Cost used to choose among Pareto solutions | Manufacturing cost, used for selection only, not a third optimized objective |
| `gellerich2023doenbi` | Machine-setting optimization (heat sealing) | Machine settings | Measured quality figures (incl. discrete/binary) | NBI whose subproblems are physical experiments | N | Y (β) | N | **N (deliberately model-free)** | **N** | **Y** | **Y** (real experimental optima) | N.A. | N | n/s | Industrial heat sealing | The only located metamodel-free NBI — the idea behind NBI-C, with real anchors | No comparison against surrogate NBI on the same problem; no simplex decision variables, ML, indicators or replication |
| `leal2025mixtureweights` | Gaussian-mixture weight estimation | Mixture weights (2 components, CI-bounded) | Sample log-likelihood (single) | Constrained optimization of a quadratic RSM over the weight simplex | P (statistical mixture) | **Y** | **Y** (simplex-lattice) | **Y** | N | N | N.A. | Y (ML re-estimation on real data) | P (runtime as a benefit) | 27 Monte Carlo scenarios + 2 real datasets | Old Faithful; PV production | DoE/RSM surrogate over a weight simplex replacing an iterative optimizer | Single objective; statistical mixture, not model combination; no NBI, gate or cost objective |
| `breiman1996stacked` | Regression model combination | Combination coefficients a_k | CV squared error | Non-negative least squares on level-one data | **Y** | **Y** (non-negativity operational; sum-to-one analysed) | N | N | N | N | N.A. | N.A. | N | LOO CV + simulation replicates | 2 regression DBs + simulation | THE precedent for the decision space: simplex-constrained weights on cross-validated predictions | Regression/MSE only; weights are the argmin of a convex program, not a designed experiment |
| `zhou2002ensembling` | NN ensembles / pruning | GA-evolved weight per network, then a subset | Generalization error | Genetic algorithm | Y | N | N | N | N | N | N.A. | N.A. | P (smaller ensembles as a benefit) | "Large empirical study" | n/s | Evolved weights → deployed subset — the support mechanism our cost prices | Weights are a selection heuristic; cost is never priced or optimized |
| `caruana2004ensemble` | Ensemble construction from model libraries | Which models to add (forward stepwise) | Any single metric, incl. **ROC Area** and **cross entropy** | Greedy forward stepwise selection | P | n/s | N | N | N | N | N.A. | N.A. | N | 7 problems × 10 metrics | 7 test problems | Tuning a heterogeneous pool directly to ROC Area / cross entropy | One metric at a time; greedy search; no design, surrogate, front or cost |
| `vanderlaan2007super` | Statistics / prediction (Super Learner) | Weights of the combination of candidate learners | V-fold cross-validated risk | Minimization of the CV loss over the weights | **Y** | P (constraint set not stated in the record) | N | N | N | N | N.A. | N.A. | N | V-fold CV + simulations | Simulated distributions | Identical data structure: weights fitted on OOF predictions of heterogeneous learners | Single objective; no design, surrogate, front, cost or partition replication |
| `zhang2011sparse` | Classifier fusion, sparse combination | Combination weight vector (LP variables) | Hinge loss + 1-norm (scalarized) | Linear programming | Y | P | N | N | N | N | N.A. | N.A. | P (sparsity) | Train/test | UCI + radar HRRP | An L1 relaxation used to control the deployed support of a weight vector | Never audits the relaxation against the step cost; single scalarized objective |
| `ledell2016auc` | Biostatistics / metalearning | Super Learner metalearner weights | Cross-validated ROC-AUC (single) | Benchmarked nonlinear optimizers | **Y** | P | N | N | N | N | N.A. | N.A. | N | CV across imbalance levels | Binary datasets | AUC as an explicit function of the ensemble weights, with its non-smoothness acknowledged | Single objective; no log-loss/cost trade-off, design, surrogate or front |
| `yao2018stacking` | Bayesian model averaging | Weights over predictive distributions | LOO expected log predictive density | Maximization of the LOO utility (PSIS) | Y | P | N | N | N | N | N.A. | N.A. | N | Simulations + real data | Bayesian comparison examples | A proper scoring rule (our log-loss) optimized over combination weights | Single objective; no DoE, surrogate, front or cost |
| `large2019probabilistic` | Heterogeneous classifier ensembles (CAWPE) | None optimized (weights = accuracy^α) | Accuracy / probabilistic performance | None (heuristic weighting) | Y | P (non-negative by construction) | N | N | N | N | N.A. | N.A. | P (five fast classifiers by design) | Extensive resampling on UCI + external UCR archive, ablation | UCI, UCR | **Five heterogeneous fast classifiers combined at the probability level with CV-derived weights, across many datasets** | Weights are a fixed heuristic, never optimized; no simplex search, front or cost objective |
| `shahhosseini2022optimizing` | Regression stacking | Ensemble weights + base hyperparameters (nested) | Prediction error (single) | Nested optimization with Bayesian search | Y | P | N | N | P | N | N.A. | P (BO evaluates the true objective) | N | 10 public datasets | 10 regression datasets | Surrogate-accelerated search over ensemble weights | Regression, single objective; a GP over a search, not a validated polynomial model of the weight space |
| `kwon2024ensemble` | Industrial ML — mixture-design ensemble weighting | **Weights of 5 heterogeneous base classifiers as mixture components** | Accuracy **or** F1, one at a time | **Scheffé polynomial + backward elimination, then constrained maximization (SLSQP among others) under Σw = 1** | **Y** | **Y** (Σw=1 verbatim; w≥0 inferred) | **Y** | **Y (Scheffé)** | N | N | N.A. | Y (verification on test data; no external validation of the polynomial) | N | Train/test with CV; no repeated partitions or paired tests | 11 binary industrial/medical/financial datasets | **The closest third-party prior art**: mixture DoE + Scheffé + simplex-constrained optimization of classifier-ensemble weights | Single objective; no ROC-AUC/log-loss; no cost; no NBI or Pareto front; no external reliability gate; no empirical reference; no replicated partitions or corrected tests |
| `xu2026pseo` | AutoML post-hoc stacking | Base-model selection (binary QP) + strategy hyperparameters | Performance, with diversity inside the QP | Binary QP + hyperparameter search | P | N | N | N | P | N | N.A. | P | N | 80 datasets, 16 methods | 80 public datasets | The 2026 state of the art in post-hoc combination of a model pool | Binary selection + strategy hyperparameters, not a continuous composition; no design, surrogate, front or cost |
| `vrugt2006multi` | Forecast post-processing (BMA) | BMA parameters incl. mixture weights | Several forecast-skill diagnostics | Unnamed numerical algorithm for the Pareto set | **Y** | **Y** (by BMA construction) | N | N | N | N | N.A. | N.A. | N | 2 case studies | Surface temperature, sea-level pressure | The closest structural precedent for MOO over simplex-constrained ensemble weights | Geoscience skill metrics; no cost/parsimony, metamodel, NBI, indicators or replication |
| `ekbal2011weighted` | NLP (NER) | Per-(classifier, class) vote weights | F-measure (single) | Genetic algorithm | Y | N | N | N | N | N | N.A. | N.A. | N | Per-language train/test | Bengali, Hindi, Telugu, Oriya, CoNLL-2003 | Continuous combination weights of a fixed heterogeneous pool | Single objective; no simplex, design, surrogate, front or cost |
| `ekbal2011multiobjective` | NLP (NER) | Per-(classifier, class) vote weights | Recall vs precision | **AMOSA** (multiobjective simulated annealing) | **Y** | N | N | N | N | N | N.A. | N.A. | N | One train/test per language | Bengali, Hindi, Telugu | MOO of continuous ensemble weights with two conflicting predictive objectives | No simplex, no cost objective, no metamodel/DoE/NBI, no reference indicators, no replication |
| `saha2013combining` | NLP (NER) | **Binary eligibility vs real vote weights, head to head** | Recall vs precision | SOO GA and MOO | **Y** | N | N | N | N | N | N.A. | N.A. | N | n/s | Bengali, Hindi, Telugu | Establishes that continuous weights beat binary membership under MOO | No probability simplex, calibration or cost objective; no surrogate, NBI, reference or replication |
| `qian2015pareto` | Ensemble pruning theory | Binary membership vector | Validation error vs **number of base learners** | Bi-objective Pareto EA (GSEMO-style) | N | N.A. | N | N | N | N | N.A. | N.A. | **P (ensemble size = unit-cost support proxy)** | n/s | UCI-style benchmarks | Explicit bi-objective performance-vs-size formulation — nearest prior art to the support cost | Cardinality over binary membership with unit costs, not the L0 count of a continuous simplex vector with heterogeneous c_i; no calibration objective, metamodel or NBI |
| `onan2016multiobjective` | Text sentiment + tabular classification | Continuous per-(classifier, class) voting weights after greedy selection | Precision vs recall | **Multiobjective differential evolution** | **Y** | n/s | N | N | N | N | N.A. | N.A. | N | Several task families | Sentiment, defect prediction, credit risk, spam | **Five heterogeneous base classifiers combined by continuous, multiobjectively optimized weights** — the nearest architectural twin | Threshold-dependent objectives; per-class, unconstrained weights; stochastic EA front; no cost, metamodel validation, reference indicators or corrected tests |
| `zhao2018multiobjective` | Sparse ensemble learning | Sparse continuous combination coefficients | fpr, fnr, **sparsity ratio** | Several EMO algorithms (ADET convex hull) | Y/P | n/s | N | N | N | N | N.A. | N.A. | **Y (parsimony)** | n/s | MNIST; remote-sensing change detection | Three-objective ensemble optimization with two error objectives + explicit parsimony — structurally closest to our (AUC, log-loss, cost) | Unit-cost sparsity, operating-point error rates; no simplex, DoE, NBI, reference or replication |
| `ribeiro2020ensemble` | Imbalanced classification; MO ensemble design | Branch-dependent: generation, selection, **combination weights** | Imbalance-aware predictive criteria | MOOD / EMO (spMODE lineage) | Y/P | n/s | N | N | N | N | N.A. | N.A. | N | Benchmark suite + one real case | Imbalanced benchmarks; water-quality anomalies | **Names and legitimises "multi-objective ensemble member combination"** — the slot this study occupies | A taxonomy plus an EMO comparison, not a designed-experiment/metamodel/NBI pipeline; no simplex geometry, cost or metamodel validation |
| `galvan2026simultaneous` | Heterogeneous regression ensembles | Estimator activation (binary) + per-learner feature subsets + **continuous voting weights** | RMSE vs mean feature cardinality | Mixed-variable MOEA (name **unverified**) | **Y** | **unverified** | N | N | N | N | N.A. | N.A. | P (feature cardinality) | **Nested CV, 16 baselines, Wilcoxon, ablation** | 5 real-world regression benchmarks | The most recent direct competitor on the weights axis, with a leakage-safe protocol | Regression RMSE; complexity is feature count, not weighted inference cost; stochastic EA front; no mixture design, Scheffé, NBI, real anchors or empirical reference |
| `rocha2025ensemblestlf` | Short-term load forecasting; NN ensembles | **Ensemble mixing weights on the probability simplex** (3 networks) | FA/PCA factors of correlated error metrics | **NBI** on the factor models + entropy/GPE rule | **Y** | **Y** | **Y ({3,5} simplex-lattice)** | **Y** (canonical form not named) | **Y** | **Y** | n/s | n/s | N | 4 substations × 3 horizons + significance tests | Brazilian distribution substations | **The same core construction**: simplex-lattice design over ensemble weights, surrogate, NBI, entropy-based selection | Three neural regressors with error metrics; no inference cost, reliability gate, metamodel-free or real-anchor variant, empirical reference, or partition replication |
| `karl2023svem` | Pharmaceutical formulation (LNP) | Lipid molar ratios + process factors | Multiple LNP quality responses | SVEM-fitted models + numerical optimization | P (SVEM is the *modelling* ensemble) | Y | Y (mixture–process, space-filling) | Y | Y | N | N.A. | **Y (confirmation runs)** | N | SVEM self-validation + confirmation runs | LNP formulation | ML modelling inside a mixture design with explicit confirmation of the surrogate-selected optimum | The ensemble is the modelling device, not the object being weighted; no Pareto machinery, NBI or cost |
| `ye2025mixinglaws` | LLM pretraining data mixing | Domain mixture proportions | Validation loss (single) | Fit a parametric mixing law on sampled mixtures, then optimize it | N | **Y** | P (sampled, not a classical design) | N | P | N | N.A. | **Y (confirmation training run)** | N | Nested small-scale runs + one confirmation run | RedPajama; Pile-CC | Fit a law over simplex proportions, predict *unseen* mixtures, optimize the fitted law | Single objective; no classical design or Scheffé form; no pass/fail reliability gate, front or ensembles |
| `liu2025regmix` | LLM pretraining | Corpus mixture proportions (Dirichlet-sampled) | Validation loss (single) | Regression surrogate + argmax over a large candidate set | N | **Y** | P (random Dirichlet plan) | N | P | N | N.A. | **Y (retrained at 1B/7B against 64 candidates)** | N | 512 proxy runs + confirmation runs | The Pile domains | "Regression over simplex proportions, then validate on the real objective"; explicit finding that pairwise interactions are non-obvious | Random rather than designed sampling; black-box regressor; single objective; no front, NBI, cost or ensemble weights |
| `chen2025aioli` | LLM data mixing | Mixture proportions, updated online | Average test perplexity (single) | Online estimation of mixing-law parameters | N | **Y** | N | N | P | N | N.A. | Y (full training runs) | N | 6 datasets vs a stratified-sampling baseline | 6 pretraining corpora | **The surrogate-fidelity argument**: a mixture method's value depends on how faithful its law over the simplex is; and a trivial baseline often wins | Online single-objective LLM data mixing; no designed experiment, polynomial model, front or ensemble weights |
| `isermann1988payoff` | Multiple objective linear programming | Continuous LP variables | Multiple linear criteria | Simplex-based procedure over the efficient set | N | N.A. | N | N | N | N (pre-dates) | **Y (the object dissected)** | N.A. | N | "Computational experience" on MOLP instances | MOLP instances | The classical demonstration that payoff-table (anchor-derived) minima can be far from the true minima over the efficient set | Error from degenerate/alternative LP optima, not metamodel error; no NBI, revalidation or ML |
| `jin2001metamodelling` | Engineering metamodelling methodology | Test-problem design variables | Metamodel quality criteria | Comparative study (polynomial, MARS, RBF, kriging) | N | N.A. | N | **Y (polynomial RSM among the four)** | N | N | N.A. | P (held-out scoring under several criteria) | N | 14 test problems, several sample sizes | Analytic/engineering problems | The argument that a metamodel must be judged on *multiple* criteria, not one in-sample statistic — the rationale for a reliability gate | No optimization on the metamodels; no rank-agreement gate on a simplex; no front or anchors |
| `jin2002framework` | EC with approximate fitness | Continuous design variables | Single fitness | Evolution strategy + NN surrogate + evolution control | N | N.A. | N | N | Surrogate-assisted, single-objective | N | N.A. | **Y (in-loop true evaluations driven by model fidelity)** | N | Repeated ES runs | Benchmarks + aerodynamic design | **"Incorrect convergence will occur if the approximate model has false optima"** — the mechanism behind surrogate-NBI failure, and the remedy of true-objective evaluation | Single objective; in-loop control rather than post-hoc revalidation of a whole front; no anchors, NBI or indicators |
| `knowles2006parego` | Expensive MOO / Bayesian optimization | Continuous design variables (2–8 D) | 2–3 objectives | ParEGO: GP + EI on a randomly weighted Tchebycheff scalarization | N | Y (scalarization weights) | N (DoE-inspired initialization only) | N | **Y** | N | P (reference from observed true minima) | **Y (in-loop)** | N | Repeated runs, 9 benchmarks, fixed budgets | 9 test functions | Random simplex-weight scalarization + surrogate + true evaluation — the Bayesian form of our random-scalarization comparator | Iterative GP infill, not a one-shot mixture-design surrogate; no NBI, anchor contrast, empirical reference or ML objectives |
| `deb2010nadir` | EMO | Generic | 2–20 objectives | Modified EMO + reference-point local search | N | N.A. | N | N | N | N | **Y (reliable estimation of extreme values)** | N.A. | N | Multiple runs on benchmark suites | DTLZ-style + one engineering problem | Establishes that the extreme points defining the normalization box are *estimates* that can be wrong, and that better estimation changes results | Estimation error from search difficulty, not from a metamodel; no NBI, surrogate-vs-true contrast or ML |
| `zhang2010moeadego` | Expensive MOO | Continuous benchmark variables | 2–3 objectives | MOEA/D-EGO (decomposition + GP + EI batch infill) | N | **Y (MOEA/D weight vectors)** | N | N | **Y** | N (PBI is NBI-style) | P | **Y (in-loop)** | N | Repeated runs | Expensive-MOP benchmarks | A uniformly spread set of simplex weight vectors driving surrogate-based front generation | Per-subproblem GPs with adaptive infill, not one-shot Scheffé polynomials on a fixed design; no anchors, reference or ML |
| `he2021normalization` | EMO (survey of normalization) | Generic | Generic, differently scaled | Survey | N | Y (reference vectors) | N | N | N | N | N.A. (systematic account of ideal/nadir estimation error) | N.A. | N | N.A. | N.A. | The mechanism we investigate: a mis-estimated ideal/anchor set distorting the geometry that generates candidates | The error arises inside an EMO run on true objectives; no metamodel-derived anchors and no comparison against true-objective anchors |
| `abolghasemian2022haulage` | Mining simulation optimization | Truck counts of four capacities (CCD-coded) | Total extraction (max), haulage time (min) | **Modified NBI on regression metamodels** | N | Y (NBI weights) | N (CCD) | **Y** | **Y** | **Y** | **N (surrogate payoff)** | **N (metamodels validated by PRESS/R²; candidates never re-simulated)** | N | Single CCD, one case | Copper open-pit haulage | The best external example of the NBI-A pattern: DoE → metamodel → validate the metamodel → run NBI on it, never revalidate the candidates | No true-objective re-evaluation, real anchors, empirical reference, indicators or replication; decision variables are truck counts |
| `herrmann2026nonextreme` | MOO method development / decision making | Generic | n_J objectives | Weighted-sum scalarizations to locate non-extreme individual minima | N | Y (normalized weights) | N | N | N | Not by name (produces the utopia–nadir hyperbox NBI needs) | **Y (true-objective minima, deliberately replaced)** | N.A. | N | Two case studies | One convex academic, one non-convex application | The only located work treating **the choice of anchor points** as a design decision affecting sampling and normalization | It swaps extreme for non-extreme *true-objective* minima; it never contemplates surrogate-derived anchors and never revalidates candidates |
| `xu2012greedy` | Budgeted learning | Boosted trees and the features they may use | Loss + feature-cost penalty (scalarized) | Stage-wise gradient boosting | P | N | N | N | N | N | N.A. | N.A. | **Y (test-time CPU)** | Benchmark experiments | n/s | The canonical accuracy-vs-test-time-cost trade-off curve | Cost on features, not ensemble members; a scalarized penalty, not a Pareto front; no simplex, NBI or surrogate |
| `chen2020frugalml` | ML-as-a-service, API selection | Sequential API-calling strategy | Accuracy under an average monetary budget | Sparsity-exploiting optimization | N | n/s | N | N | N | N | N.A. | N.A. | **Y (real per-call prices)** | Systematic experiments across providers | Emotion, sentiment, speech APIs | Costs attached to individual models with real price tags — the same economics as our c_i | Adaptive per-example calling; a budget constraint, not a Pareto coordinate; no simplex, DoE or metamodel |
| `erickson2020autogluon` | AutoML for tabular data | Which models to train, stacking architecture, final ensemble weights | Predictive performance under a **training-time** budget | Multi-layer stacking + ensembling | **Y** | n/s | N | N | N | N | N.A. | N.A. | **N (training budget only)** | 50 tasks + Kaggle competitions | 50 classification/regression tasks | The same object: a post-hoc ensemble of heterogeneous tabular models | Single objective; greedy selection; no inference-cost objective, front, design, metamodel or NBI |
| `gunasekaran2022cocktail` | Cloud model serving | The set of models in the serving ensemble + autoscaling | Minimize dollar cost subject to accuracy and latency | Runtime model selection + proactive autoscaling | P (membership, not weights) | N | N | N | N | N | N.A. | N.A. | **Y (real dollars, measured latency)** | AWS EC2 prototype across workloads | Serving workloads | The clearest statement that a deployed ensemble's cost is driven by how many models stay online — our support cost, with real money | Systems paper with runtime adaptive selection; no continuous weights, front, mixture design or weighted-cost contrast |
| `wang2022committees` | Efficient deep models | Which pre-trained models form the committee; cascade thresholds | Accuracy vs FLOPs/latency (trade-off curve) | Not stated ("most simplistic method") | P (mostly uniform) | P | N | N | N | N | N.A. | N.A. | **Y (FLOPs, wall-clock)** | Large empirical study across tasks and architecture families | ImageNet-style, video, segmentation | **Puts two cost accountings (all-members-run vs early-exit) side by side on the same models** | The two accountings are ensemble-vs-cascade *architectures*, not two cost functions on one weight vector; no simplex weights, front, NBI or metamodel |
| `borchert2022paretoselect` | Time-series forecasting model selection | Which model/default configuration (discrete) | Accuracy **and** latency (Pareto) | Learned surrogate mapping model→metrics + non-dominated filtering | N | N.A. | N | N | **Y** | N | N.A. | P (benchmark ground truth exists for all candidates) | **Y** | 44 datasets, 13 methods, all evaluations released | 44 forecasting datasets | A metamodel of the objectives used to locate the Pareto front cheaply, with cost as a genuine Pareto coordinate | Selection among single models rather than weighting a mixture; no simplex, mixture DoE, NBI, anchors or empirical reference |
| `ji2023pruning` | Ensemble pruning for deep forest | Binary selection per cascade layer | Accuracy, independent diversity, coupled diversity | Tri-objective layer-wise pruning | N | N | N | N | N | N | N.A. | N.A. | P (storage/prediction time as motivation and reported outcome, never an objective) | 15 UCI datasets | 15 UCI datasets | A multiobjective treatment of which members to keep, motivated by deployment cost | Binary membership; diversity instead of a cost coordinate; no surrogate, NBI or cost-model comparison |
| `maier2024hardware` | AutoML post-hoc ensembling, tabular | Ensemble composition (greedy multiplicities inducing weights) | Predictive accuracy **and inference time** (Pareto) | Quality-diversity optimization over ensemble candidates | **Y** | Y in effect (searched combinatorially, not designed) | N | N | N (direct evaluation) | N | N.A. | N.A. | **Y (support-style, a Pareto objective)** | 83 classification datasets | 83 tabular datasets | Accuracy vs inference cost as a Pareto front over post-hoc weighted ensembles of tabular base models | Combinatorial QD/greedy search rather than mixture design + Scheffé + NBI; two objectives; no empirical reference or indicators; no reliability gate; no weighted-vs-support contrast |
| `maier2026hapens` | AutoML post-hoc ensembling, tabular | Ensemble composition over cached predictions | Predictive performance **and resource usage** (Pareto; memory the effective metric) | Multi-objective / quality-diversity search + greedy variant with static objective weighting | **Y** | Y in effect | N | N | N | N | N.A. | N.A. | **Y (explicitly "deployment cost")** | 83 datasets + ablations | 83 tabular datasets | **The direct contemporary competitor**: post-hoc weighted ensembling on cached tabular predictions with deployment cost as an explicit Pareto objective | No mixture design, Scheffé metamodel or reliability gate; no NBI (hence no anchor question); no empirical reference with IGD+/HV; no weighted-vs-support cost contrast |

---

## 3. Foundational method, indicator, statistics, software and data citations

These are cited for definitions, tools and protocol, not as comparable studies, and are therefore kept out of the
table above.

**Optimization method.** `das1998nbi` (NBI: payoff matrix, utopia point, CHIM, quasi-normal subproblems);
`messac2003normalized` and `messac2004nc` (Normalized Normal Constraint — the anchor-based normalization and
even-representation arguments, and the standard NBI comparison); `marler2004survey` (a priori / a posteriori /
no-articulation taxonomy); `miettinen1998nonlinear` (ideal and nadir vectors, payoff tables);
`deb2002nsga2`, `zhang2007moead`, `deb2014nsga3`, `bandyopadhyay2008amosa` (the EMO alternatives).

**Mixture design and response surfaces.** `scheffe1958mixtures` (simplex-lattice designs, canonical polynomials);
`scheffe1963simplexcentroid` (simplex-centroid design and the special-cubic model); `cornell2002mixtures` (the
authority on the interpretation of β_ij as departure from *linear blending*); `piepel1982component` (mixture
coefficients are not readable as component effects under Σx=1); `boxdraper2007response` (second-order models on
linearly restricted regions); `myers2016rsm` and `smith2005formulation` (RSM diagnostics; constrained-simplex
formulation and collinearity).

**Ensemble background.** `hansen1990neural` (ensembles beat single models); `wolpert1992stacked` (the level-one /
out-of-fold prediction matrix our pipeline caches); `perrone1995networks` (optimal linear combination in closed form);
`leblanc1996combining` (combination weights in regression and classification); `kittler1998combining` (the sum rule =
our uniform-average baseline, and the sensitivity-to-estimation-error argument); `fumera2005theoretical` (why a
non-uniform non-negative weight vector has headroom over the simple average); `dietterich2000ensemble`,
`kuncheva2014combining`, `kuncheva2003measures` (surveys and diversity measures).

**Performance indicators.** `zitzler1999multiobjective`, `zitzler2003performance` (hypervolume and indicator theory);
`ishibuchi2015modified` (IGD+ and its weak Pareto compliance; note that a reference set must be supplied);
`vanveldhuizen2000multiobjective` (generational distance); `schott1995fault` (spacing); `audet2021performance`
(indicator survey); `coello2004parallelization` (cited in the literature for IGD but see the metadata note in
`foundational_methods.md` — prefer `ishibuchi2015modified` or `audet2021performance` for the definition).

**Statistics.** `nadeau2003inference` (correction for the overlap between resampled partitions);
`bouckaert2004replicability`, `dietterich1998approximate`, `demsar2006statistical`, `holm1979simple`,
`wilcoxon1945individual`, `cureton1956rank`, `kerby2014simple` (rank-biserial effect size), `wilson1927probable` and
`brown2001interval` (binomial intervals), `efron1993bootstrap`.

**Metrics and calibration.** `brier1950verification`, `hanley1982meaning`, `niculescu2005predicting`,
`guo2017calibration`, `davis2006relationship`.

**Software and data.** `pedregosa2011scikit`, `chen2016xgboost`, `virtanen2020scipy`, `harris2020numpy`,
`kraft1988software` (SLSQP); `yeh2009comparisons` and `yeh2009ucidataset` (UCI credit default), `kagglesantander`,
`kagglebnp`, `kaggleporto`.

**Framing.** `turney2000types` (there is more than one kind of cost in inductive learning);
`schwartz2020greenai` (efficiency as a reported evaluation criterion).

---

## 4. Cluster reviews

### 4.1 Cluster A — Ensemble weight optimization (single objective)

The decision space of this study is not new. `wolpert1992stacked` created the out-of-fold ("level one") prediction
matrix on which everything here operates, and `breiman1996stacked` put the combination coefficients under
non-negativity constraints, explicitly analysing the non-negativity *plus* sum-to-one case and showing that any such
coefficient vector produces an interpolating predictor bounded between the minimum and maximum member predictions.
`perrone1995networks` gives the closed-form optimum for the linear combination in the MSE sense, `leblanc1996combining`
frames the same problem statistically for regression and classification, and `vanderlaan2007super` supplies the
canonical cross-validated weighted combination of heterogeneous learners — the exact data structure of our log-loss
branch. `kittler1998combining` is the theoretical statement of the uniform-average (sum-rule) baseline, and
`fumera2005theoretical` explains, under the Tumer–Ghosh framework, why an interior simplex point can beat that
baseline and how the gain depends on the correlation between member outputs.

Metric-targeted weighting is equally established. `caruana2004ensemble` optimizes an ensemble drawn from a library of
models *directly* to a chosen metric, and names ROC Area and cross entropy among the ten metrics used — the two
predictive objectives of the present study, taken one at a time. `ledell2016auc` goes further and formulates
AUC maximization over Super Learner weights as a constrained nonlinear program, benchmarking many optimizers precisely
because AUC as a function of the weights is hard; that is the same non-smoothness that forces our direct-AUC-search
comparator and the reduced solver settings of the metamodel-free arm. `yao2018stacking` optimizes a proper scoring
rule (the log score) over weights on predictive distributions. On the search side, the literature is a catalogue of
metaheuristics: a genetic algorithm on the weight vector followed by thresholding (`zhou2002ensembling`), particle
swarm on a combined accuracy-and-diversity fitness (`you2020weighted`), simulated annealing on cached decision outputs
(`choi2023combining`), a manta-ray optimizer for voting weights as recently as 2026 (`hassan2026weighted`), and nested
Bayesian search over weights and base hyperparameters (`shahhosseini2022optimizing`). `large2019probabilistic` is the
"same setup, no optimization" reference point: exactly five heterogeneous fast classifiers combined at the probability
level with weights that are a closed-form function of cross-validated accuracy, benchmarked with heavy resampling over
UCI and externally validated on a separate archive. `xu2026pseo` is the 2026 AutoML answer — binary quadratic
programming for selection plus a searched hyperparameter space of ensemble strategies.

The decisive record in this cluster is `kwon2024ensemble`. It takes five heterogeneous base classifiers (kNN, SVC,
logistic regression, naive Bayes, decision tree), treats their weights explicitly as mixture components, runs a mixture
design of experiments over those weights, fits a Scheffé canonical polynomial (linear plus binary-interaction terms) to
Accuracy or F1, reduces it by backward elimination at α = 0.05, and maximizes the reduced polynomial subject to
Σw_i = 1 using SciPy solvers including SLSQP, on eleven binary tabular datasets. Three of the seventeen elements this
manuscript enumerates — weights as mixture components, a mixture design over classifier weights, and a Scheffé
surrogate of ensemble performance — are therefore prior art from outside the authors' group. The cluster A audit
records that the record for this paper is a publisher HTML page in Korean whose Eq. (5) is an image; the sum-to-one
constraint is quotable, the non-negativity constraint is inferred from the mixture framing. That caveat should be
stated if the paper is leaned on heavily.

`zhang2011sparse` deserves separate mention because it is the classical antecedent of the cost half of this study: a
linear program with a 1-norm penalty over the combination weights, deliberately producing a sparse vector so that
"we only ensemble classifiers with nonzero weight coefficients". It uses a continuous weighted quantity as the
relaxation of a support count without ever asking whether the two agree.

**What cluster A leaves open.** Everything in this cluster is single-objective. No entry builds a Pareto front, none
introduces an inference or deployment cost objective, none externally validates the fitted weight-response surface on
unseen compositions before optimizing it, and none uses a scalarization-based even-spread front generator. The one
paper that fits a Scheffé model over classifier weights (`kwon2024ensemble`) optimizes a single classification metric
and reports no trade-off, no Pareto reference, no replicated partitions and no corrected paired tests. What is left
open is therefore not "can weights be optimized on a simplex", which is settled, but whether a *multiobjective*
treatment of the same surface can be trusted — and specifically whether the surrogate that makes the multiobjective
treatment cheap is faithful enough to place the extreme points on which the whole construction depends.

### 4.2 Cluster B — Multiobjective ensembles

This cluster splits cleanly on one axis: whether the decision variables are continuous combination weights, binary
membership, or the parameters of the members themselves. Member generation dominates historically —
`abbass2003pareto` harvests the Pareto front of a bi-objective training problem as the ensemble,
`chandra2006ensemble` and `chandra2006hybrid` evolve accuracy and diversity as separate pressures (DIVACE,
DIVACE-II), and `jin2008pareto` and `gu2015multi` survey the field with generation as its centre of gravity. Selection
is the second large group: `dossantos2008dynamic`, `qian2015pareto`, `onan2017hybrid`, `rosalesperez2017evolutionary`,
`peimankar2017evolutionary`, `fletcher2020nonspecialized`, `sun2021constructing`, `wegier2022multicriteria`,
`grzyb2023svm`, `ekbal2012multiobjective` and the review chapter `heywood2024evolutionary`. None of these optimizes a
weight vector, so none moves the novelty boundary of the present study, but two are directly relevant as comparators:
`qian2015pareto` is an explicit bi-objective error-versus-ensemble-size formulation with theoretical support — the
closest prior art to a support-based deployment cost, albeit with unit costs on a binary membership set — and
`fletcher2020nonspecialized` shows that multiobjective selection spontaneously yields small, cheap ensembles.

The weights branch is smaller but pointed. `ekbal2011weighted` optimizes per-(classifier, class) vote weights with a
single-objective GA; `ekbal2011multiobjective` and `ekbal2013combining` move the same weights under AMOSA with recall
and precision as separate objectives; `saha2013combining` puts binary and real vote weights head to head in both
frameworks and concludes in favour of the continuous formulation — a result the present study relies on rather than
claims. `onan2016multiobjective` is the nearest architectural twin in the entire review: five heterogeneous base
classifiers, continuous voting weights, a multiobjective differential-evolution search, and evaluation across several
task families including credit risk. `rezaei2019multi` does the same on three classifiers for brain-tumour CT.
`zhao2018multiobjective` is structurally closest to the objective vector of the present study: three objectives, two
error rates plus an explicit sparsity ratio. `ribeiro2020ensemble` supplies the taxonomy — generation, selection,
**combination**, and combinations thereof — that names the slot this study occupies and shows that combination is the
least studied of the three. `galvan2026simultaneous` is the most recent competitor: activation, per-learner feature
subsets and continuous voting weights optimized jointly under RMSE versus feature cardinality, with nested
cross-validation, Wilcoxon tests and an ablation that identifies the continuous weighting mechanism as pivotal. The
cluster B agent could not read its full text (MDPI 403), so neither the MOEA name nor whether the weights are
normalized is verified. `vrugt2006multi` sits slightly outside the ML literature but is the cleanest precedent of all
for the geometry: a Pareto set over Bayesian-model-averaging mixture weights, which are simplex-constrained by
construction, calibrated against several forecast-skill diagnostics instead of one.

Two negative results from this cluster matter. First, after Crossref bibliographic search, OpenAlex full-text search
and four arXiv queries that all returned zero hits, the cluster reports **no verified paper applying NSGA-II, NSGA-III
or MOEA/D to simplex-constrained ensemble combination weights**; that negative should be reported as "no verified
competitor was found", not as "none exists". Second, whether the Saha/Ekbal vote weights are normalized could not be
settled — their abstracts do not say and the full texts are paywalled — so the claim that no prior multiobjective
ensemble-weighting study uses a probability simplex rests on an unread record.

**What cluster B leaves open.** Multiobjective optimization of continuous ensemble weights is established, and has
been for fifteen years. What is absent is the geometry and the audit: no entry constrains the weights to the
probability simplex *and* treats that simplex as a designed experimental region; every front is generated by a
stochastic population method rather than by a deterministic even-spread construction; no entry fits a metamodel of the
objectives over the weight space, so the question of whether the front reflects the surrogate or the data never
arises; no entry scores its front against an independently constructed empirical reference; the objectives are almost
always two threshold-dependent predictive measures rather than a threshold-free ranking metric, a calibration metric
and a deployment cost; and replication is typically one train/test split per dataset, with `galvan2026simultaneous`
and `moradpour2026ensemble` the exceptions that use nested CV and non-parametric multi-method tests respectively.

### 4.3 Cluster C — Mixture designs and Scheffé models in machine learning

The design machinery comes from `scheffe1958mixtures` (simplex-lattice designs and the canonical polynomial forms
obtained by eliminating the constant term through Σx_i = 1) and `scheffe1963simplexcentroid` (the 2^q − 1 centroid
design and its regression equation); `cornell2002mixtures` is the standard monograph and the source of the vocabulary
— synergistic, antagonistic, linear blending — that this study uses when interpreting β_ij; `piepel1982component`
is the classical demonstration that mixture-model coefficients are *not* directly readable as component effects
because of the Σx_i = 1 constraint. `smith2005formulation` names collinearity under the mixture constraint as an
explicit topic, and `boxdraper2007response` covers second-order models on linearly restricted regions.

The transfer of this machinery into machine learning has happened in three places. First, and closest, is
`rocha2025ensemblestlf`: a {3,5} simplex-lattice mixture design **on the probability simplex of ensemble weights**
combining a linear network, an MLP and an RBF network; factor analysis over correlated error metrics; NBI on the
resulting factors to produce a Pareto frontier of ensemble configurations; and an entropy/GPE rule to pick the
operating point. Its own stated contributions are "casting ensemble-weight definition as a structured mixture-design
problem" and "integrating factor analysis/PCA with NBI". Any claim of the form "we cast ensemble-weight selection as a
mixture-design problem" or "we are the first to run NBI over the ensemble-weight simplex" is pre-empted by this paper.
Its first author is not an author of the present manuscript, but Balestrassi is a UNIFEI colleague of the group; it
should be treated as adjacent-group, not own, prior work. Second is `kwon2024ensemble`, discussed above, which is the
classifier-side instance. Third is the mixture-DoE-in-statistics line: `leal2025mixtureweights` fits a quadratic
response surface to the sample log-likelihood over a (CI-bounded) weight simplex and replaces EM with one constrained
optimization, validated on 27 Monte Carlo scenarios — the clearest recent statement that a DoE/RSM surrogate over a
weight simplex can substitute for an iterative optimizer, and the closest analogue of the surrogate stage of the
present study minus the multiobjective part.

`karl2023svem` and `karl2026svemnet` represent the modern statistical-ML direction inside mixture DoE: self-validated
ensemble models replace the classical polynomial as the response model on a simplex-constrained design, with explicit
confirmation runs on the candidate optima. Note the inversion relative to this study — there the ensemble is the
*modelling device* and the mixture proportions are the decision variables; here the ensemble weights *are* the mixture
proportions. `paula2019gatuning` is the lineage's bridge from formulation to algorithms: a mixture design crossed with
process variables where the mixture components are objective weights and the process variables are GA hyperparameters,
i.e. the response of the design is an algorithm's output.

The fourth, and most active, body of work is LLM data mixing, which is the same mathematics in a different costume.
`xie2023doremi` optimizes domain proportions with group DRO on a small proxy model and transfers them to a large one.
`ye2025mixinglaws` discovers "the quantitative predictability of model performance regarding the mixture proportions in
function forms", fits such laws on sampled mixtures, and — the point that matters here — uses them to predict
performance at *unseen* mixtures before running them. `liu2025regmix` samples Dirichlet mixtures, fits a regression
surrogate, optimizes it and confirms the winner with real training runs, and reports that domain interactions
"often contradict common sense". `chen2025aioli` shows that the value of a data-mixing method tracks the *fidelity* of
its assumed law over the simplex, and that no existing method consistently beat a stratified-sampling baseline.
`thudi2025mixmin` proves that the bi-level mixing objective becomes convex as the model class grows and applies the
result to XGBoost on bioassay data — a tabular, non-LLM instance.

**What cluster C leaves open.** The design family, the polynomial family and even the application of both to ensemble
weights are all prior art. What is not present anywhere in this cluster is an *admissibility* criterion: the LLM
mixing papers validate the mixture surrogate by predicting or retraining at the selected point, and `chen2025aioli`
argues that fidelity determines value, but none defines a pre-registered pass/fail gate on held-out compositions that
decides whether a surface may be optimized at all. None of the ML-side entries is multiobjective; none carries a
deployment-cost objective; none contrasts anchors or runs a metamodel-free control. On the interpretive question — is
"a large β_ij does not imply the blend beats its better pure component" already known? — the cluster's answer is that
it follows immediately from the classical definition, which is stated against the linear-blending chord (Cornell's
worked pages compare observed blend responses against "the average of the yields of the two pure blends"), but the
explicit caveat in the form this manuscript needs could not be located in any accessible page, and Cornell's relevant
pages are view-restricted. That should be reported as "the classical reading is restated, the ML consequence is
demonstrated", not as a correction of mixture-design theory.

### 4.4 Clusters D and G — Surrogate-assisted MOO, and surrogate / anchor failure

NBI itself is `das1998nbi`: individual minima of each objective form a payoff matrix whose diagonal defines the utopia
point; the convex hull of individual minima is traversed by an evenly spaced set of convex parameters; each subproblem
pushes along the quasi-normal. The abstract proves scale independence and even spread. Nothing in the verified record
suggests the paper contemplates how those individual minima are obtained — they are exact by assumption. The same
assumption runs through the Normal Constraint family (`messac2003normalized`, `messac2004nc`), whose contribution is
precisely a linear mapping *anchored on* the individual minima, and through the NBI repair literature:
`siddiqui2012improving` reformulates the sweep as one optimization problem, `motta2012modified` fixes deficiencies for
more than two objectives, `wagner2025nbiplus` (2025) extends coverage in many-objective space, and
`muellergritschneder2009bounded` computes bounded fronts by first establishing trade-off limits. All of them repair the
*sweep* or the *coverage*; none repairs the *anchors*.

The anchor literature exists, but on the other side of the surrogate divide. `isermann1988payoff` is the classical
demonstration that payoff-table minima can differ substantially from the true minima over the efficient set, and that
the field needs something better than payoff tables. `deb2010nadir` shows that the nadir vector requires information
about the whole front, is genuinely hard to estimate, and that better estimation changes results.
`he2021normalization` surveys ideal- and nadir-point estimation across MOEAs and states plainly that inaccurate
estimation of the front range degrades performance. `herrmann2026nonextreme` (2026) is the nearest neighbour: it treats
the *choice* of individual minima as a design decision, replacing extreme minima with "non-extreme individual minima"
to obtain a refined utopia–nadir hyperbox and better knee selection. In every one of these the error source is search
difficulty or degeneracy on the *true* objectives; a metamodel is never the culprit, and no paper computes anchors two
ways and compares the fronts.

The surrogate literature says clearly that surrogates mislead. `jin2002framework` states it in one sentence —
"incorrect convergence will occur if the approximate model has false optima" — and prescribes evolution control, i.e.
periodic evaluation on the true function governed by estimated model fidelity. `deb2019taxonomy` argues that building
one metamodel per objective and per constraint is not the most efficient approach because "the cumulative effect of
errors from each metamodel may turn out to be detrimental for the accuracy of the overall optimization procedure", and
`deb2020surrogate` adds an adaptive scheme that switches among ten metamodeling frameworks on the basis of a
statistical comparison of their accuracy. `jin2001metamodelling` is the methodological argument for judging a
metamodel on multiple criteria rather than one in-sample statistic. `tabatabaei2015survey` is the most directly useful
survey: it names the *sequential* framework — build the surrogate once from a fixed design, then optimize it — which
is exactly this study's pipeline, contrasts it with the *adaptive* framework, and recommends the latter, i.e. it
identifies our configuration as the one most exposed to surrogate error. `chugh2019survey`, `jin2011surrogate`,
`liang2025survey` and `rojasgonzalez2020kriging` complete the map; the last is the reference for surrogate MOO under
*noisy* objective estimates, which is our regime, and calls heterogeneous noise the field's major open problem. The
canonical algorithms — `knowles2006parego`, `zhang2010moeadego`, `ponweiser2008smsego`, `chugh2018krvea` — all
evaluate their infill points on the true objectives inside the loop, which is a different discipline from re-scoring a
completed front afterwards.

`abolghasemian2022haulage` is the single best external example of the failure mode this study is designed to expose: a
central composite design supplies the runs, nonlinear regression metamodels are fitted, **model adequacy is checked
with PRESS and R²**, and a modified NBI is then solved on those metamodels — with the payoff matrix taken from the
metamodels and no re-simulation of the resulting Pareto points. That is the NBI-A configuration with a metamodel
validation gate and without revalidation. At the other extreme, `gellerich2023doenbi` runs NBI entirely without a
metamodel, deriving the experimental plan from the NBI subproblems themselves and evaluating them by physical tests,
so its anchors are real experimental optima. The two papers bracket the present study's NBI-A and NBI-C arms, but no
source runs both on the same problem.

**What clusters D and G leave open.** The cluster's search log states the negative result explicitly: no source
consulted (Crossref, OpenAlex, OpenAIRE, Semantic Scholar, publisher pages) yields a paper that (a) computes NBI or NC
anchors from a surrogate and separately from the true objectives and compares the resulting frontiers, or (b)
revalidates a complete surrogate-generated Pareto set on the true objectives and scores it with IGD+ and hypervolume
against an *empirically constructed* — rather than analytically known — reference front. The literature knows that
anchors can be wrong and that surrogates can be wrong; it has not put the two together, and it has never asked the
question in a setting where the objectives are noisy out-of-fold classifier metrics on a simplex.

### 4.5 Cluster E — NBI with mixture design and RSM

This cluster is largely the authors' own lineage, and it is the reason this manuscript claims no part of the
DoE+RSM+NBI construction. The pattern begins with `oliveira2011portfolio`, where energy-contract proportions are
treated as mixture components and CVaR-adjusted responses are modelled by mixture polynomials; it continues through
`mendes2016portfolio_mde`, `monticeli2017portfolio` — which adds computational replicas at each design point — and
`leal2022portfolio_doptimal_mde`, which reduces a {5,10} simplex-lattice to 200 D-optimal runs. In parallel the group
develops NBI on RSM surrogates for manufacturing: `brito2014nbimse`, `costa2016nbimmse`, `costa2016nbipcasnr`,
`lopes2016rpdmnbi`, `naves2017nbirfs`, `belinato2019mnbipca`, `luz2021weldingnbi`, `almeida2022taguchinbi` and
`streitenberger2022nbifa`. Two of these run confirmation experiments on the surrogate-selected points —
`lopes2016rpdmnbi` with an L9 array at three weight settings, `naves2017nbirfs` against prediction intervals — which
is the lineage's precedent for real-objective revalidation, applied to one or a few selected solutions rather than to
every candidate.

The two strands meet on the *weight* simplex. `rocha2015entropynbi` introduces the entropy/GPE weight-selection rule;
`rocha2017robustmcdm` states verbatim that "the normal boundary intersection (NBI) method along with the mixture
design of experiments (MDE) are used to optimize these responses simultaneously"; `rocha2021robustpoint` makes the
construction explicit — Shannon-entropy/error, diversity/error and unscaled prediction variance are "experimentally
modeled using mixture design over the weights", then maximized to choose the weights. This is the direct antecedent of
the predecessor's MBPA step. `bacci2019nbi_mixture` is the case where the weights being designed over are genuinely
combination weights: a Simplex-Lattice {m,q} over forecast-combination weights, PCFA of the combined-residual metrics,
Scheffé-type mixture models of the factor scores, NBI on those models, and entropy/GPE selection.
`moreira2021ann_ensemble_mde` uses a mixture design to set ANN ensemble weights for PV forecasting.
`pereira2025hybrid` — the authors' own 2025 EAAI paper — generalizes the post-optimization step: a simplex-lattice
schedules the NBI weights, generalized distance and Shannon entropy are recorded along the frontier and modelled by
Scheffé fourth-order polynomials in those weights, and NBI is reapplied to those polynomials to obtain interpolated,
non-lattice weights. `pereira2026postpareto` and `azevedo2026nbivrf` continue the template into 2026 and fix how the
group uses hypervolume, IGD and spacing — to compare optimizers head to head. The first author's own dissertation
`ribeiro2026dissertation` transfers the pipeline to XGBoost hyperparameters, uses the 66-point Simplex-Lattice {3,10}
weight arrangement this study reuses, and establishes the "candidates are re-evaluated on the real model" step.

Outside the lineage, `gellerich2023doenbi` is the only metamodel-free NBI found, and `leal2025mixtureweights` and
`vasquezramos2025rsmxgboost` show DoE/RSM entering ML from other directions (mixture-weight estimation; XGBoost
hyperparameters — the latter verified only at metadata level plus one clause).

**What cluster E leaves open.** The cluster's own critical question — does any paper in this lineage apply NBI or
mixture design to classifier or regressor *ensemble* weights? — is answered yes, three times, all on the
regression/forecasting side: `rocha2025ensemblestlf`, `bacci2019nbi_mixture` and `moreira2021ann_ensemble_mde`. What
is absent is any application to classifier ensembles with ROC-AUC and log-loss objectives; any inference or deployment
cost objective; any external validation of the mixture surrogate on unseen compositions with an admissibility rule;
any real-objective anchor variant or metamodel-free control; any empirical Pareto reference independent of the
surrogate; and any replication across resampled partitions with corrected paired inference. Every lineage paper is a
single design on a single case study. It is also worth recording that no full text in this cluster was read — all
method fields come from abstracts — so statements that the lineage uses surrogate anchors follow the papers' own
descriptions of the payoff step and are marked "not stated" wherever the abstract is silent.

### 4.6 Cluster F — Cost-aware and latency-aware ensembles

The cluster imposes a three-way taxonomy that the manuscript should adopt: **continuous weighted** cost (a smooth
function of a mixing variable, Σ w_i c_i), **support/step** cost (a model is paid for in full the moment it is used at
all, Σ c_i·1[w_i > ε], ensemble size, memory footprint), and **cascade/early-exit** cost (per-example expected cost
under a gated policy). It also records, for each entry, whether cost is a Pareto objective, a constraint, a scalarized
penalty, or only a reported outcome — a distinction the literature blurs constantly.

The cascade branch is the oldest and the largest: `trapeznikov2013multistage`, `xu2013cstc`, `nan2015budgeted`,
`nan2017adaptive` and `chen2020frugalml` all price per-example adaptive execution, usually as a budget constraint.
`chen2020frugalml` matters most here because its costs attach to *individual models* with real prices, which is the
economics of our c_i. `xu2012greedy` is the canonical scalarized accuracy-versus-test-time-cost formulation, with the
cost living in feature space. The support branch is where the present study sits. `gunasekaran2022cocktail` is the
clearest statement that a deployed ensemble's cost is what you keep online, with real AWS dollars and measured
latency; `ji2023pruning` runs a tri-objective pruning of deep forests motivated by storage and prediction time but —
tellingly — puts diversity, not cost, in the objective vector; `akkerman2026pace` controls the number of retained weak
learners under a faithfulness target; `gudipaty2025mel` and `moreira2026energy` budget the aggregate footprint and the
energy of the models actually deployed. `zhou2002ensembling` and `zhang2011sparse` are the implicit ancestors: the
support of the weight vector is what gets deployed, and an L1 penalty is used as its relaxation.

Two 2024–2026 papers are the direct competitors. `maier2024hardware` integrates inference time into post-hoc ensemble
selection by extending a quality-diversity ensemble-selection framework, producing a Pareto front of accurate and
efficient ensembles over 83 tabular classification datasets. `maier2026hapens` extends the same line, names the
objective "deployment cost", identifies memory usage as the most effective metric in an ablation, and shows that even a
greedy ensembling algorithm improves markedly under static multi-objective weighting. Both operate on cached base-model
predictions for tabular classification, both use support-style costs, and both are Pareto rather than
constraint-based. `borchert2022paretoselect` is the closest thing to a surrogate-assisted cost-aware Pareto method: a
learned mapping from model to metrics is used to select from the accuracy–latency front without training candidates.
`erickson2020autogluon` and `feurer2022autosklearn` are the AutoML baselines whose budgets are on *training* time, not
inference. `ding2025moeec` is included as evidence that mainstream multiobjective ensemble *selection* still uses
diversity and similarity rather than cost in its objective vector as of 2025. `turney2000types` and
`schwartz2020greenai` supply the framing: there is more than one kind of cost, and the cost of running models should be
reported alongside accuracy.

**What cluster F leaves open.** The cluster answers its own critical question unambiguously: across all verified
entries, **no prior work explicitly contrasts the continuous weighted-cost relaxation Σ w_i c_i against the
support/step cost Σ c_i·1[w_i > ε] for the same weighted ensemble, and none shows the two disagreeing about which
method wins.** The near misses are named and are worth citing precisely because they are near misses:
`zhang2011sparse` uses the L1 relaxation of a support count without auditing agreement; a companion of
`nan2015budgeted` proves via total unimodularity that a 0-1 support-cost programme is solved exactly by its LP
relaxation — a result in the *opposite* direction, i.e. the two agree there; and `wang2022committees` shows that
switching cost accountings reorders which committee looks best, but across ensemble-versus-cascade *architectures*
rather than across two cost functions on one weight vector. Also absent: any coupling of a cost objective to a mixture
design or a Scheffé surrogate, any three-objective (ranking, calibration, cost) formulation, and any cost-aware
ensemble study that scores its front against an independently built empirical reference.

---

## 5. Ten closest prior works

Ranked by how much of this manuscript's construction they already contain.

**1. `rocha2025ensemblestlf` — Rocha, Rotella, Balestrassi, Melgani & Zambroni de Souza (2025), *IEEE Access* 13:207903–207915.**
Shares the entire core construction: a simplex-lattice mixture design on the probability simplex of ensemble weights,
a fitted surrogate over that simplex, NBI on the resulting objectives to produce a Pareto frontier of ensemble
configurations, and an entropy-based post-Pareto rule. Its stated first contribution is "casting ensemble-weight
definition as a structured mixture-design problem", which is why no version of that sentence may appear as a novelty
claim here. What separates it: three neural regressors and forecast-error metrics rather than five heterogeneous
probabilistic classifiers with ROC-AUC and log-loss; objectives are factor-analytic composites rather than the raw
metrics; and it has no inference-cost objective, no external reliability gate, no real-anchor or metamodel-free
variant, no empirical Pareto reference, and no replication over resampled partitions.

**2. `pereira2025hybrid` — Pereira, Tertuliano Ribeiro, Mendes, Campos & de Paiva (2025), *EAAI* 162:112510. OWN PRIOR WORK.**
The direct methodological parent. It supplies the DoE + RSM + NBI pipeline, the simplex-lattice schedule of NBI
weights, the use of Scheffé polynomials over a simplex, the payoff-matrix construction from individual optima of the
*surrogate* surfaces — which is exactly this study's NBI-A control condition — and the post-Pareto machinery. The
present study inherits all of that and claims none of it. What separates them: there the mixture design acts on
scalarization weights while the decision variables are three machining factors; here the mixture design is over the
decision variables themselves. The predecessor has no ML objectives, no external validation of the surrogate on unseen
compositions, no real-objective anchors, no metamodel-free variant, no empirical Pareto reference, no inference cost,
and a single 19-run CCD on one case study.

**3. `bacci2019nbi_mixture` — Bacci, Mello, Incerti, de Paiva & Balestrassi (2019), *IJPE* 212:186–211.**
The group's earliest statement of the combination this manuscript uses: a Simplex-Lattice over genuine *combination*
weights, mixture (Scheffé-type) models of accuracy metrics as functions of those weights, NBI on the mixture models to
build the Pareto frontier of optimal weight sets, and entropy/GPE selection. It establishes that "mixture DoE over
combination weights + surrogate + NBI" is inherited group methodology rather than new. What separates it: the
components are forecasting methods, the objectives are rotated factor scores of residual metrics, and it has no
anchors analysis, reliability gate, empirical reference, cost objective or replication.

**4. `kwon2024ensemble` — Kwon, Lee & Lee (2024), *J. Soc. Korea Ind. Syst. Eng.* 47(4):161–170.**
The closest third-party prior art on the classifier side: five heterogeneous base classifiers whose weights are
mixture components, a mixture design over those weights, a Scheffé canonical polynomial fitted to Accuracy or F1 and
reduced by backward elimination, and constrained maximization under Σw = 1 with SciPy solvers including SLSQP, over
eleven binary tabular datasets against stacking baselines. It independently establishes elements 1–3 of this study's
novelty matrix as known. What separates it: strictly single-objective with no trade-off analysis; no ROC-AUC or
log-loss; no cost; no NBI or Pareto front; no external validation of the polynomial on unseen compositions and no
reliability gate; no empirical Pareto reference or indicators; and no replicated partitions or corrected paired tests.
(Verified from the publisher's HTML; one constraint is rendered as an image and could not be quoted.)

**5. `ribeiro2026dissertation` — Ribeiro (2026), UNIFEI M.Sc. dissertation. OWN PRIOR WORK.**
The first author's own antecedent, and the source of two concrete inherited elements: the 66-point Simplex-Lattice
{3,10} weight arrangement, and the discipline of re-evaluating NBI candidate solutions on the real model rather than
trusting the surrogate. It also establishes the group's DoE+RSM+FA+NBI pipeline on machine-learning metrics under a
fixed evaluation budget, benchmarked against grid search, random search, Bayesian optimization and Hyperopt and
replicated on additional datasets. What separates it: the decision variables are XGBoost hyperparameters in a CCD
space, not simplex ensemble weights; the objectives are factor scores of accuracy-family metrics and runtime, not
ROC-AUC, log-loss and inference cost over a composition; and it has no real anchors, no metamodel-free NBI, no
reliability gate and no empirical Pareto reference.

**6. `maier2026hapens` — Maier & Purucker (2026), arXiv:2603.10582 (with `maier2024hardware`, AutoML 2024 workshop).**
The direct contemporary competitor on the cost axis: post-hoc weighted ensembling over cached tabular base-model
predictions, with deployment cost as an explicit Pareto objective, evaluated on 83 classification datasets, plus an
ablation identifying memory usage as the most effective cost metric and a study of static multi-objective weighting.
Together with `maier2024hardware` it means "inference cost as a Pareto objective for post-hoc tabular ensembles" is
prior art and must not be claimed. What separates them: a quality-diversity/greedy combinatorial search rather than a
mixture design, a Scheffé metamodel and NBI; two objectives rather than three; no reliability gate; no empirical
Pareto reference scored with IGD+ and hypervolume; and — decisively — no formulation of the continuous weighted-cost
relaxation and therefore no comparison of the two cost accountings on the same weight vector.

**7. `onan2016multiobjective` — Onan, Korukoğlu & Bulut (2016), *ESWA* 62:1–16.**
The nearest architectural twin: five heterogeneous base classifiers whose continuous voting weights are set by a
multiobjective differential-evolution search, evaluated across sentiment analysis, software defect prediction, credit
risk modelling, spam filtering and semantic mapping. It is the strongest single threat to any broad claim that
"multiobjective optimization of ensemble weights is new". What separates it: precision and recall, which are
threshold-dependent, instead of a threshold-free ranking metric plus a calibration metric plus a cost; per-class,
unconstrained weights rather than a single probability-simplex vector; a stochastic EA front rather than a
deterministic even-spread construction over a fitted mixture surrogate; and no cost objective, metamodel validation,
reference-front indicators or corrected paired tests. (Its abstract could not be retrieved from any index; the
characterization rests on the CoLab-mirrored publisher abstract read in cluster B.)

**8. `ekbal2011multiobjective` and `saha2013combining` — Ekbal & Saha (2011), *ESWA* 38(12):14760–14772; Saha & Ekbal (2013), *DKE* 85:15–39.**
Together they establish that continuous ensemble *weights* — not just membership — can be optimized under a genuine
Pareto formulation, and that the continuous ("real vote") formulation beats the binary one. `saha2013combining` runs
that comparison head to head across seven heterogeneous classifiers. The present study relies on this result rather
than re-deriving it. What separates them: the weights are per-(classifier, class) reliabilities with no stated
normalization, so no probability simplex; the objectives are recall and precision; the search is an archive-based
metaheuristic with no metamodel, no design over the weight space and no even-spread front generator; there is no cost
objective, no reference front, and no replication across datasets and partitions. Whether their weights are normalized
could not be settled from the available records.

**9. `abolghasemian2022haulage` — Abolghasemian, Ghane Kanafi & Daneshmand-Mehr (2022), *Complexity* 2022:3540736.**
The best external instance of exactly the configuration this study puts on trial: a central composite design, nonlinear
regression metamodels, metamodel adequacy checked with PRESS and R², and a modified NBI solved *on the metamodels*
with the payoff matrix taken from them. It is NBI-A with a surrogate-validation gate, published and unremarked. What
separates it from the present study: the generated Pareto solutions are never re-run through the simulation, there are
no real-objective anchors and no metamodel-free control, there is no empirical reference front and no indicator
scoring, there is no replication, and the decision variables are truck counts rather than simplex weights.

**10. `moreira2021ann_ensemble_mde` — Moreira, Balestrassi, Paiva, Ribeiro & Bonatto (2021), *RSER* 135:110450.**
Establishes that mixture DoE has already been used to determine the weights of a model ensemble — here an ANN ensemble
for photovoltaic generation forecasting — inside the same research group. It is the reason element 1 of the novelty
matrix is marked as inherited rather than new. What separates it: a single error objective (MAPE), regression rather
than classification, no multiobjective or NBI stage, no cost objective, no surrogate validation and no reference front.

**Runners-up, in rough order.** `galvan2026simultaneous` (the most recent joint selection-plus-continuous-weights
competitor, with the strongest evaluation protocol in cluster B); `maier2024hardware` (grouped with entry 6);
`qian2015pareto` (the bi-objective error-versus-size formulation that a reviewer will raise against the support cost);
`vrugt2006multi` (Pareto sets over BMA mixture weights — simplex by construction); `gellerich2023doenbi` (the only
metamodel-free NBI, with real anchors); `herrmann2026nonextreme` (the only work treating anchor choice as a design
decision); `isermann1988payoff` (the classical proof that payoff-table anchors are unreliable);
`leal2025mixtureweights` (a quadratic RSM over a weight simplex replacing an iterative estimator);
`liu2025regmix` and `chen2025aioli` (surrogate-over-the-simplex with revalidation, and the surrogate-fidelity
argument); `large2019probabilistic` (five heterogeneous classifiers, probability-level combination, heavy resampling —
the heuristic-weighting baseline this study must beat).
