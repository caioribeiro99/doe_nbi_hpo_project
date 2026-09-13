# Literature review for Paper 2

**Purpose.** Fix the novelty boundary before the protocol is frozen, so the experiment is designed
to test something the literature does not already establish.

**Citation rule.** Every reference below was resolved against Crossref, arXiv or the publisher's own
record during this review, and the fields shown are the ones those records return. Nothing here is
cited from memory. Where a record is incomplete, it says so.

**Status of the base bibliography.** Paper 1 (`papers/surrogate_nbi_ensemble/references.bib`,
frozen at `paper-submission-v2`) already holds 180 verified entries covering Normal Boundary
Intersection, the Varimax-rotated-factor lineage, mixture design of experiments, surrogate-assisted
multiobjective optimization and ensemble weighting. Paper 2 inherits it. This review adds the
hyperparameter-optimization literature that Paper 1 did not need.

---

## Cluster A. Design of experiments and response surfaces for hyperparameter tuning

The idea that hyperparameter tuning is a designed experiment is established, and the closest work is
recent.

- **Lujan-Moreno, Howard, Rojas and Montgomery (2018).** *Design of experiments and response surface
  methodology to tune machine learning hyperparameters, with a random forest case-study.* Expert
  Systems with Applications 109, 195–205. DOI `10.1016/j.eswa.2018.05.024`.
  Screening factorial then response surface, on random forest. Single objective. Montgomery, a
  co-author, is the standard design-of-experiments reference, which makes this the canonical
  statement of the idea.

- **Vasquez-Ramos, Ruiz-Sandoval, Oliva, Ramos-Soto, Ramos-Frutos, Sharawi and Pérez-Cisneros
  (2025).** *Response surface-driven hyperparameter optimization for XGBoost.* The Journal of
  Supercomputing 81(10), article 1112. DOI `10.1007/s11227-025-07600-4`.
  **This is the closest prior art on the surface of the problem:** response surface methodology,
  applied to XGBoost hyperparameters, published the year before this work. It uses a Box–Behnken
  design and a single response. It does not build a Pareto front, does not reduce objectives by
  factor analysis, and does not use any scalarization of several objectives.

**Consequence.** "Design of experiments and a response surface to tune XGBoost hyperparameters" is
**prior art and must not be claimed.** It was claimed by neither the dissertation nor Paper 2's
plan, but the manuscript must cite Vasquez-Ramos et al. and say plainly what it adds.

## Cluster B. The Varimax-rotated-factor Normal Boundary Intersection lineage

The combination of principal-component or factor extraction, Varimax rotation, and NBI over the
resulting factor scores is a mature line of work from the authors' own group, on manufacturing
processes.

- **Costa, Paula, Silva and Paiva (2016).** IJAMT 87(1–4), 825–834. DOI `10.1007/s00170-016-8478-7`.
  NBI on principal components with Taguchi signal-to-noise ratios, turning of 12L14 steel.
- **Luz, Romão, Streitenberger, Mancilha, de Paiva and Balestrassi (2021).** IJAMT 117(5–6),
  1517–1534. DOI `10.1007/s00170-021-07761-5`. NBI with multivariate techniques, welding.
- **Streitenberger, Romão, Paiva, Balestrassi, Freitas and Paes (2022).** Journal of Cleaner
  Production 333, 129915. DOI `10.1016/j.jclepro.2021.129915`. NBI with a factor-analysis approach,
  cladding, stochastic objectives.
- **Pereira, Tertuliano Ribeiro, Mendes, Campos and de Paiva (2025).** Engineering Applications of
  Artificial Intelligence 162, 112510. DOI `10.1016/j.engappai.2025.112510`. Hybrid multivariate NBI
  with mixture-design post-optimization. **Two of its authors are authors here, and the first author
  of this work is a co-author there.**
- **de Azevedo, Pereira, Cesário and de Paiva (2026).** Thermal Science and Engineering Progress 74,
  104722. DOI `10.1016/j.tsep.2026.104722`. Explicitly named the **NBI-VRF** method, applied to
  computational fluid dynamics for hydrodynamic systems.

**Consequence.** "Varimax-rotated factor scores as NBI objectives" is **the group's own established
method and must not be claimed as new.** `de Azevedo et al. (2026)` names it. Paper 2 inherits it.

## Cluster C. Hyperparameter optimization baselines

What a reviewer will expect the proposed method to be compared against.

- **Bergstra and Bengio (2012).** *Random Search for Hyper-Parameter Optimization.* Journal of
  Machine Learning Research 13(10), 281–305. The reason random search is the mandatory floor.
- **Hutter, Hoos and Leyton-Brown (2011).** *Sequential Model-Based Optimization for General
  Algorithm Configuration.* Lecture Notes in Computer Science, 507–523. DOI
  `10.1007/978-3-642-25566-3_40`. SMAC.
- **Falkner, Klein and Hutter (2018).** *BOHB: Robust and Efficient Hyperparameter Optimization at
  Scale.* ICML 2018. arXiv:1807.01774.
- **Akiba, Sano, Yanase, Ohta and Koyama (2019).** *Optuna.* KDD '19, 2623–2631. DOI
  `10.1145/3292500.3330701`.
- **Bischl, Binder, Lang, Pielok, Richter, Coors, Thomas, Ullmann, Becker, Boulesteix and others
  (2023).** *Hyperparameter optimization: foundations, algorithms, best practices, and open
  challenges.* WIREs Data Mining and Knowledge Discovery 13(2), e1484. DOI `10.1002/widm.1484`.

The dissertation compared against coarse grid search, random search, Bayesian optimization via
`scikit-optimize` and the tree-structured Parzen estimator via `hyperopt`. That comparator set is
reasonable and covers the floor and the standard model-based methods. It omits multi-fidelity
methods entirely, which is the visible gap (`baseline_gap_assessment.md`).

## Cluster D. Multiobjective hyperparameter optimization

The field this work actually sits in, and the one that decides its novelty.

- **Morales-Hernández, Van Nieuwenhuyse and Rojas Gonzalez (2023).** *A survey on multi-objective
  hyperparameter optimization algorithms for machine learning.* Artificial Intelligence Review
  56(8), 8043–8093. DOI `10.1007/s10462-022-10359-2`.
- **Karl, Pielok, Moosbauer, Pfisterer, Coors, Binder, Schneider, Thomas, Richter, Lang and others
  (2023).** *Multi-Objective Hyperparameter Optimization in Machine Learning — An Overview.* ACM
  Transactions on Evolutionary Learning and Optimization 3(4), 1–50. DOI `10.1145/3610536`.
- **Guerrero-Viu, Hauns, Izquierdo, Miotto, Schrodi, Biedenkapp, Elsken, Deng, Lindauer and Hutter
  (2021).** *Bag of Baselines for Multi-objective Joint Neural Architecture Search and Hyperparameter
  Optimization.* arXiv:2105.01015. Establishes what a credible multiobjective HPO baseline set looks
  like.
- **Eggensperger, Müller, Mallik, Feurer, Sass, Klein, Awad, Lindauer and Hutter (2021).**
  *HPOBench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO.* NeurIPS
  Datasets and Benchmarks Track 2021. arXiv:2109.06716.
- **Pfisterer, Schneider, Moosbauer, Binder and Bischl (2022).** *YAHPO Gym — An Efficient
  Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization.* First Conference on
  Automated Machine Learning. arXiv:2109.03670.

**Consequence, and it is the important one.** Accuracy against training or inference cost is the
**standard** multiobjective HPO problem, treated in both surveys, with dedicated benchmark suites
(HPOBench, YAHPO Gym) and an agreed baseline set (Guerrero-Viu et al.). Paper 2 therefore **cannot**
claim novelty for:

- posing hyperparameter optimization as multiobjective;
- trading predictive quality against training or inference cost;
- producing a Pareto front of hyperparameter configurations.

It also means a reviewer from the automated-machine-learning community will expect comparison
against that community's baselines, evaluated the way that community evaluates. The dissertation's
comparator set predates none of this but engages with none of it either.

## Cluster E. What no located work does

Searches for Normal Boundary Intersection combined with hyperparameter optimization returned the
NBI methodological literature and the HPO literature separately, and nothing joining them. The
searches run were: normal boundary intersection with hyperparameter optimization; normal boundary
intersection with XGBoost, gradient boosting, Pareto, training time and accuracy; design of
experiments with response surface methodology and multiobjective hyperparameter optimization; and
the Varimax-rotated-factor NBI lineage.

**This is a negative result from a bounded search, not a proof of absence,** and the manuscript must
say so in those words. What it supports is narrow: no located work applies NBI, or the VRF-NBI
construction, to hyperparameter optimization.

It does **not** support calling that application the contribution. Transferring an established
method to a new application domain is thin ground for a methods journal, and the reviewer question
"why would NBI be the right tool here?" has to be answered with evidence rather than with novelty.

## Cluster F. Reproducibility and implementation auditing

The literature that gives the audit work its footing.

The Paper-1 bibliography already carries the replicated-evaluation apparatus (Nadeau and Bengio's
corrected test, Bouckaert and Frank, Dietterich, Demšar, Holm). What this review did not find is an
established genre of published audits comparing a paper's stated method against its released code in
optimization. That absence is itself worth one careful paragraph in the manuscript, and worth no
more than that: the audits here serve the experiment's design, they are not the paper's claim.

---

## Where this leaves the novelty boundary

Established prior art, not claimable:

1. Design of experiments and response surfaces for hyperparameter tuning (Cluster A).
2. Response surfaces for XGBoost hyperparameters specifically (Vasquez-Ramos et al. 2025).
3. Varimax-rotated factor scores as NBI objectives (Cluster B, the authors' own group).
4. Multiobjective hyperparameter optimization with a cost objective (Cluster D).
5. Normal Boundary Intersection itself (Das and Dennis 1998).

What is left, and what `novelty_matrix.md` has to defend, is not a new combination. It is what the
controlled comparison of the four arms measures, and what the audits make measurable.
