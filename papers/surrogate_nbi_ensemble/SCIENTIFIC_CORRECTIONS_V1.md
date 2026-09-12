# Scientific corrections applied before the NSGA-II baseline

Applied 2026-09-11 to the manuscript at `ea328eb`. No experimental artifact was touched; the R = 30 benchmark remains
frozen at tag `pco213-postwork-r30` (commit `b3ed050`). Every corrected number was recomputed from the authoritative
statistics CSVs rather than carried over from prose. Verification was run on the **compiled PDF text** (with ligature
normalization, since `ﬁ`/`ﬀ` otherwise defeat a naive string search) so that the check covers what a reader actually
sees, not only the sources.

**Status: all seven CLOSED.**

---

## Issue 1 — The empirical reference was described as method-independent

**Old claim.** Several locations said or implied that the reference used for scoring is independent of the methods
under test: "an empirical Pareto reference independent of the methods under test", "scoring against an empirical
reference the methods cannot influence", "built independently of the surrogate", "an independently constructed
reference", "an independent reference front".

**Why it was wrong.** The methodology already states correctly that the final reference is the non-dominated union of
the sampled core *with every candidate set*, and that NBI-B and NBI-C contribute 30–40% of its points. The manuscript
therefore contradicted itself: the *core* is method-independent, the *final augmented reference* is not.

**New claim.** The two layers are now always distinguished. The **sampled core** (≥ 10⁵ Dirichlet points, simplex
lattice, vertices, edges and a 40-cap ε-constraint sweep) is constructed independently of the surrogate and of every
candidate method. The **final augmented reference** used for scoring is the union of that core with all candidate
sets. The self-grading caveat is preserved and now travels with the claim.

**Evidence.** `src` of the reference stage in `scripts/pco213_run_postwork_benchmark.py`; front-source composition in
`tables/reference_diagnostics.csv`; the existing methodology paragraph in §3.6.

**Locations changed.** `00_abstract.tex` (reference description), `01_introduction.tex` (approach paragraph and
Contribution 4), `02_related_work.tex` (four places: §2.2, §2.4 Kwon paragraph, §2.5, §2.7 gap list),
`06_discussion.tex` (the checklist in §6.7), `08_conclusion.tex` (opening paragraph), `claims_and_evidence.md`
(new binding wording rule in C11), `research_lineage.md`, `novelty_matrix.md`.

**Status: CLOSED.** Compiled-text scan finds none of the five forbidden phrasings; "sampled core", "augmented with all
candidate sets" and "partly graded against itself" are all present.

---

## Issue 2 — The real-anchor win counts were stated as a single number per dataset

**Old claim.** "Real anchors improve both primary endpoints in 30 of 30 partitions on the two datasets whose ROC-AUC
surfaces almost never pass the gate, and in 24 of 30 on the other two." Repeated in five locations.

**Why it was wrong.** The two endpoints do not share a win count on Porto Seguro or UCI credit, and UCI credit's
hypervolume comparison contains a tie.

**Authoritative counts** (recomputed from `statistics/paired_primary_effects.csv`, cost = weighted, comparison
`nbi_B vs nbi_A`, as wins/ties/losses):

| Dataset | IGD⁺ | Hypervolume |
|---|---|---|
| Santander | 30/0/0 | 30/0/0 |
| BNP Paribas | 30/0/0 | 30/0/0 |
| Porto Seguro | **23/0/7** | 24/0/6 |
| UCI credit | 24/0/6 | **24/1/5** |

**New claim.** Unanimous on Santander and BNP Paribas; 23–24 of 30 on Porto Seguro and 24 of 30 on UCI credit. Where
the text is endpoint-specific it now names each count, including the UCI tie ("24 wins with one tie and five losses").

**Locations changed.** `00_abstract.tex`, `01_introduction.tex` (Contribution 2), `05_results.tex` (§5.3 primary
paragraph and the extension-batch qualifier, now labelled as hypervolume counts), `06_discussion.tex` (§6.1),
`08_conclusion.tex` (Geometry paragraph), `claims_and_evidence.md` (C1 table plus a binding wording rule).

**Status: CLOSED.**

---

## Issue 3 — "Pre-registered" was used without a formal preregistration

**Old claim.** "a pre-registered reliability gate", "pre-registered as the two primary endpoints", "by the
pre-registered rule", and equivalents in the supporting documents.

**Check performed.** No OSF entry, AsPredicted entry, registered report or other externally timestamped
preregistration exists for this study. The evidence of advance specification is the version-controlled commit history:
the gate thresholds, primary endpoints, comparison families and analysis rules were committed before the replicated
benchmark was launched.

**New claim.** "a reliability gate pre-specified before the replicated benchmark"; "whose thresholds were fixed and
committed to version control before the replicated benchmark was run"; "fixed in advance as the two primary
endpoints". The methodology now states explicitly: *"We describe this as pre-specification rather than preregistration:
there is no external registry entry, and the evidence is the version-controlled commit history."*

**Locations changed.** `00_abstract.tex`, `01_introduction.tex`, `03_methodology.tex` (two places, plus the new
disclaimer sentence), `04_experimental_protocol.tex` (two places), `05_results.tex`, and five supporting documents.

**Status: CLOSED.** No instance of "pre-registered", "preregistered" or "preregistration" remains as a claim about this
study anywhere in the manuscript or the supporting documents.

---

## Issue 4 — The "anchors dominate" claim was stated universally

**Old claim.** "Anchor construction, not interior surrogate accuracy, is the first-order failure mode."

**Why it was too strong.** The study supports this for failures *mediated by the surrogate pipeline*, in the regimes
where surrogate anchoring fails. UCI credit is a direct counterexample: its surfaces pass the gate in 30/30
partitions, yet the residual NBI-C versus NBI-B gap there is a subproblem-convergence effect of NBI-B (median 29 of 66
certified; Spearman(ΔHV, converged count) = −0.92), not a surrogate-accuracy effect. Solver geometry can dominate
where the surrogate is excellent.

**New claim.** "Among failures attributable to the surrogate pipeline, anchor misplacement had the largest observed
effect." The UCI counterexample is now stated explicitly in the abstract, in Contribution 2 and in the conclusion, so
the narrower claim is visible wherever the strong one used to be.

**Locations changed.** `00_abstract.tex`, `01_introduction.tex` (Contribution 2 title and body), `06_discussion.tex`
(§6.1 opening), `08_conclusion.tex` (Geometry paragraph). The mechanism discussion in §6.1 and the UCI analysis in
§5.5 already carried the correct qualification and were left intact.

**Status: CLOSED.**

---

## Issue 5 — The methods section asserted the spacing property the paper later falsifies

**Old claim.** "NBI generates an even spread of Pareto points" (§3.4); "generates well-distributed Pareto points"
(§2.4).

**Why it was wrong.** §5.5 reports that no spacing advantage survives revalidation on the real objectives: NBI-C's
size-matched spacing percentile is 0.99, 0.54, 0.96 and 0.41, while the random Dirichlet sample scores 1.00, 1.00,
1.00 and 0.93. The methods section should not assert what the results overturn.

**New claim.** "NBI was introduced to obtain a distributed representation of the Pareto boundary, by advancing from the
convex hull of individual minima along quasi-normal directions", with an immediate forward reference: whether that
distribution survives the map into revalidated objective space is one of the questions this paper tests, and in our
setting it does not. The results section now names the classical motivation it is testing rather than introducing it
as new.

**Locations changed.** `02_related_work.tex` §2.4, `03_methodology.tex` §3.4, `05_results.tex` §5.5 (negative-result
paragraph rewritten to link back to §3.4).

**Status: CLOSED.**

---

## Issue 6 — "Differ in exactly two binary choices, and in nothing else"

**Old claim.** The three NBI variants "differ in exactly two binary choices --- where the objectives come from, and
where the anchors come from --- and in nothing else".

**Why it was wrong.** NBI-C necessarily uses different solver semantics. Because the real ROC-AUC is piecewise
constant, SLSQP cannot certify optimality, so NBI-C accepts an iterate whose equality residual is below 10⁻³ under a
reduced budget (2 starts, maxiter 120, fd-eps 10⁻³), whereas NBI-A and NBI-B require full certification on smooth
polynomials. "In nothing else" was false.

**New claim.** "The variants are defined by two methodological factors --- the source of the objectives and the source
of the anchors." The causal reading is retained but bounded: A versus B isolates the anchor source with objectives and
solver held fixed and is a clean contrast; B versus C tests removal of the surrogate once real anchors are supplied
but is *not* a perfectly identical numerical experiment, because changing the objective from a smooth polynomial to a
piecewise-constant rank statistic necessarily changes solver semantics. Solver outcomes are reported separately per
arm.

**Locations changed.** `01_introduction.tex` (Contribution 1), `03_methodology.tex` §3.4 (variant paragraph and the
closing asymmetry note). The limitations section already carried the solver-semantics caveat and was left intact.

**Status: CLOSED.**

---

## Issue 7 — The feasibility-recovery step was called a nearest-point projection

**Old claim.** "a projection step returning the nearest feasible composition when an iterate leaves the simplex".

**Implementation audited.** `src/mixens/nbi.py::project_free_vars` (lines 377–392), called at line 511:

```python
z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
s = float(z.sum())
if s <= 1.0 + atol:
    return lift_simplex(z), True
w = np.concatenate([z / s, [0.0]])
validate_weights(w[None, :])
return w, False
```

and the consumer marks the candidate: `"success": bool(c.success) and feasible`.

**Why the old wording was wrong.** This clips the free variables to [0, 1] and, if they still sum above one,
renormalizes them onto the face `w_M = 0`. That is a deterministic repair, **not** the Euclidean projection onto the
simplex; the two differ in general. No code was changed to make the old prose true.

**New claim.** "a deterministic feasibility-recovery step for iterates that leave the simplex: the free variables are
clipped to [0,1] and, if they still sum above one, renormalized onto the face `w_M = 0`. This is a documented repair,
not a Euclidean projection onto the simplex, and a candidate recovered this way is flagged infeasible and excluded
from the certified set."

**Locations changed.** `03_methodology.tex` §3.4.

**Status: CLOSED.**

---

## Verification performed

| Check | Result |
|---|---|
| Tests | 68 passed |
| Compilation | clean; 0 undefined citations, 0 undefined cross-references |
| Compiled-text scan for the 10 forbidden phrasings | all absent |
| Compiled-text scan for the 11 replacement phrasings | all present |
| Anchor counts recomputed from `paired_primary_effects.csv` | match the corrected prose exactly |
| `project_free_vars` read line by line | prose now matches code |
| Preregistration search (OSF, AsPredicted, registered report) | none exists; terminology changed accordingly |
| Code changed to suit prose | none |

Page count 38, body 17,886 words including tables. Compression is Phase 17, after the NSGA-II baseline.
