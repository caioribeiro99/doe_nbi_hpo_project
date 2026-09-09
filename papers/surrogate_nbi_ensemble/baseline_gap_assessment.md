# Baseline gap assessment: is an evolutionary multiobjective baseline required?

**Question.** Will reviewers require a canonical evolutionary multiobjective optimizer (NSGA-II, NSGA-III, MOEA/D,
AMOSA) as a comparator, and if so, which one, at what budget, and answering which scientific question?

**Verdict: USEFUL, trending to LIKELY REQUIRED at Applied Soft Computing and at Engineering Applications of
Artificial Intelligence.** Not needed to support any claim currently made in the manuscript, but cheap enough that
declining to run it is harder to defend than running it. Recommendation: **run NSGA-II** as a supplementary arm before
submission, reusing the cached out-of-fold predictions, and report it in the supplementary material with a short
pointer from the results. Measured cost is **1.3 hours** at a candidate-matched budget and **18.4 hours** at a budget
matched to NBI-C, against the 81.1 hours already invested.

---

## 1. What the study already contains

The comparator set was designed to decompose one pipeline, not to rank optimizers. It contains:

| Arm | Role |
|---|---|
| NBI-A | the inherited construction: surrogate objectives, surrogate anchors |
| NBI-B | isolates anchor misplacement (surrogate objectives, real anchors) |
| NBI-C | isolates residual surrogate error (real objectives, real anchors) |
| Random weighted scalarization | the cheap classical alternative on the *same* surfaces with the *same* real anchors |
| 66-point random Dirichlet sample | floor check on the reference geometry (candidate-count matched only) |
| 66-run mixture design | the unoptimized design itself, as a further floor |
| SLSQP log-loss optimum, direct AUC search | exact single-objective references and the source of the real anchors |
| Empirical Pareto reference | ≥ 10⁵ points, independent of every method, with a convergence check |

Every claim in the manuscript is a *paired contrast within this set*, and none of them would change if an evolutionary
arm were added. The empirical reference already provides what an evolutionary baseline is usually asked to provide:
an answer to "how far is this front from the best attainable?"

## 2. The case that it is not needed

1. **The research question is diagnostic, not competitive.** The paper asks under which conditions surrogate-assisted
   NBI can be trusted. Adding NSGA-II answers "is NBI better than NSGA-II?", which is a different question and one
   the paper explicitly declines to ask.
2. **The reference front is a stronger yardstick than any single optimizer.** A method that reaches hypervolume ratio
   0.98 against a 10⁵-point empirical reference is characterized more informatively than by beating one evolutionary
   run.
3. **The group's sibling paper already does this comparison.** Pereira et al. (2026, *Int. J. Adv. Manuf. Technol.*
   144:5335–5361) benchmarks the same NBI machinery against NSGA-II, MOEA/D, weighted sum and MOLA on hypervolume,
   IGD and spacing. The present paper can cite that rather than repeat it in a different domain.
4. **Budget matching is genuinely hard here and would invite its own criticism.** NBI-A and NBI-B consume zero real
   evaluations; NBI-C consumes 4.2 × 10⁵. Any single budget makes some arm look artificially good or bad, and a
   reviewer who wanted the comparison would likely also dispute the budget chosen.

## 3. The case that it is needed

1. **Readership expectation.** At Applied Soft Computing the readership is evolutionary-computation centred and the
   absence of NSGA-II is the first thing a reviewer will notice. At EAAI it is a common, though not universal, request.
2. **The paper makes claims about front quality.** Statements such as "NBI-C is best or tied-best on the primary
   endpoints" are made within a comparator set that a reviewer may consider self-selected, since it contains no
   method from outside the DoE/NBI lineage other than random search.
3. **It would strengthen an existing negative result.** The manuscript reports that NBI's characteristic even spacing
   does not survive revalidation on real objectives. NSGA-II, whose crowding distance explicitly targets spread, is
   the natural test of whether *any* method achieves good spacing here or whether the objective geometry prevents it.
   This is the strongest scientific argument for running it.
4. **It is very cheap.** See below. The cost is small enough that "we did not run it" reads as an omission rather
   than a choice.
5. **It removes the weakest comparator's burden.** The 66-point random Dirichlet arm currently carries the role of
   "external floor" and is degenerate under the weighted cost. An evolutionary arm would be a far more credible
   external reference point and would let the Dirichlet arm be demoted to the supplementary material.

## 4. If run: exact specification

**Algorithm: NSGA-II.** Chosen over MOEA/D and NSGA-III because the problem has three objectives (NSGA-III's
advantage appears at higher dimensions), because NSGA-II is the algorithm reviewers actually name, and because
MOEA/D's decomposition weights would introduce a second scalarization-weight design that confuses the comparison with
NBI's β lattice. AMOSA would be the choice if the reviewer question were specifically about the ensemble-weighting
literature (where AMOSA-based classifier-ensemble weighting is established); NSGA-II covers the general expectation.

**Encoding.** Real-valued vector of length 5 on the simplex, with the same free-variable parameterization and
projection used by NBI, or equivalently a normalized encoding `w = u / Σu` with `u ∈ [0,1]^5`. Simulated binary
crossover and polynomial mutation, standard parameters (η_c = 20, η_m = 20, p_c = 0.9, p_m = 1/5).

**Objectives.** Exactly the three real out-of-fold objectives used by NBI-C — negative ROC-AUC, log-loss and the
weighted cost — computed by the same `evaluate_weights` routine, so no implementation asymmetry is introduced.

**Reuse of cached predictions.** Yes. NSGA-II evaluates weight vectors against the cached out-of-fold probability
matrix `P` exactly as NBI-C does. No model refitting is required, which is why the arm is cheap. It runs on the same
120 replications with the same seeds, so it is paired by partition like everything else.

**Two budgets, both reported.**

| Budget | Rationale | Setting |
|---|---|---|
| **Candidate-matched** | returns a front comparable in size to the NBI sets | population 100, 300 generations = 30,000 evaluations |
| **NBI-C-matched** | equal real-objective budget to the most expensive arm | population 200, 2,100 generations = 420,000 evaluations |

**Measured cost.** Wall-clock per real objective evaluation, measured on the actual cached matrices with eight worker
threads on the study workstation:

| Dataset | Rows (OOF) | ms per evaluation | 30k evals: 30 partitions | 420k evals: 30 partitions |
|---|---|---|---|---|
| Santander | 160,000 | 2.13 | 0.53 h | 7.5 h |
| BNP Paribas | 91,456 | 0.99 | 0.25 h | 3.5 h |
| Porto Seguro | 160,000 | 1.73 | 0.43 h | 6.1 h |
| UCI credit | 24,000 | 0.37 | 0.09 h | 1.3 h |
| **Total** | | | **1.3 h** | **18.4 h** |

NSGA-II's own overhead (non-dominated sorting and crowding distance at population 100–200) is negligible beside the
objective evaluations. Both budgets together cost under 20 hours, against 81.1 hours already spent on the study.

**Scientific questions it would answer.**

1. Does *any* method attain even spacing on the revalidated front, or is the poor spacing of all current methods a
   property of the objective geometry? (The strongest reason to run it.)
2. How close does a standard evolutionary optimizer get to the empirical reference at a comparable evaluation budget,
   and does it match NBI-C's ≥ 0.97 hypervolume ratio?
3. Does the anchor finding have an analogue outside NBI — that is, is the failure specific to CHIM-based methods that
   require a payoff matrix, or does any surrogate-driven method inherit it? (Answerable by additionally running
   NSGA-II *on the Scheffé surfaces*, at negligible extra cost, which would be a clean surrogate-versus-real contrast
   for a method with no anchors at all. **This variant is the most scientifically interesting addition and should be
   included.**)

**What it would not answer, and must not be claimed.** It would not establish that NBI is better or worse than
evolutionary MOO in general, on one problem class with four datasets and one parameter setting. The framing must be:
an external reference point that confirms (or bounds) the front-quality claims, not a horse race.

## 5. Decision and sequencing

**Run it before submission**, in this order:

1. NSGA-II on the real out-of-fold objectives at the candidate-matched budget (1.3 h). This is the minimum that
   answers a reviewer.
2. NSGA-II on the Scheffé surfaces at the same budget (negligible additional cost, since surrogate evaluations are
   essentially free). This gives the anchor-free surrogate-versus-real contrast.
3. Only if a reviewer asks: the NBI-C-matched budget (18.4 h).

**Constraint on execution.** The R = 30 experiment is frozen at tag `pco213-postwork-r30` and must not be modified.
An evolutionary arm is an *additive* stage that reads the existing cached out-of-fold artifacts and writes to a new
output directory, leaving every existing artifact and every reported number untouched. It would be tagged separately
and reported as a supplementary comparison, with the frozen tag remaining the reference for all primary results.

**Do not launch it as part of the current manuscript-drafting work.** The user's standing instruction is that no new
experiments run unless the literature review identifies a genuine reviewer-level gap. This document records that the
gap is real but not blocking, that the cost is now measured rather than estimated, and that the decision to run is the
author's to make.
