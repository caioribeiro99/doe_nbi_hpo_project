# Novelty matrix

Every element the paper could conceivably claim, with its actual status. An element marked KNOWN
PRIOR ART must never appear as a positive claim.

**Rule.** Nothing is marked NEW on the strength of "we could not find it". A bounded literature
search establishes that something was not located, which is weaker than absence and much weaker than
a contribution. Elements that survive do so because of what the campaign **measures**, not because
of what the search failed to find.

---

## The matrix

| # | Element | Status | Owner, or what it depends on |
|---|---|---|---|
| 1 | Normal Boundary Intersection as a scalarization | KNOWN PRIOR ART | Das and Dennis (1998) |
| 2 | PCA or factor analysis to decorrelate responses before optimization | KNOWN PRIOR ART | Costa et al. (2016) and the line that follows |
| 3 | Varimax-rotated factor scores as NBI objectives (NBI-VRF) | KNOWN PRIOR ART | the authors' own group; named in de Azevedo et al. (2026) |
| 4 | Design of experiments and response surfaces for hyperparameter tuning | KNOWN PRIOR ART | Lujan-Moreno, Howard, Rojas and Montgomery (2018) |
| 5 | Response surfaces for XGBoost hyperparameters | KNOWN PRIOR ART | Vasquez-Ramos et al. (2025), unrelated group |
| 6 | Multiobjective hyperparameter optimization | KNOWN PRIOR ART | Morales-Hernández et al. (2023); Karl et al. (2023) |
| 7 | Predictive quality against training or inference cost as the objective pair | KNOWN PRIOR ART | both surveys; HPOBench; YAHPO Gym |
| 8 | A Pareto front of hyperparameter configurations | KNOWN PRIOR ART | the field |
| 9 | Applying NBI, or NBI-VRF, to hyperparameter optimization | NOT LOCATED | a bounded search; **not claimable as the contribution** |
| 10 | Re-evaluating every returned candidate on the real objectives | INHERITED | the dissertation's own protocol |
| 11 | An external reliability gate on surrogate adequacy | INHERITED | the authors' Paper 1 |
| 12 | Evaluation-matched budgeting against an evolutionary baseline | INHERITED | the authors' Paper 1 |
| 13 | The anchor-injection control separating set composition from geometry | INHERITED | the authors' Paper 1 |
| 14 | **The four-arm decomposition isolating reference construction, scalarization geometry and anchor provenance on one problem** | CANDIDATE NEW | the campaign must produce a measurable difference |
| 15 | **A reproducible identity audit of a published optimizer against its released code** | CANDIDATE NEW, WEAK | see below |
| 16 | **The finding that the historical method's cost objective is not reproducible from its seeds** | CANDIDATE NEW, NARROW | measured; see below |
| 17 | **The measured sensitivity of a VRF quality composite to its aggregation weighting** | CANDIDATE NEW | measured: Spearman 0.374, best row changes |
| 18 | Whatever the four-arm campaign measures | UNDETERMINED | the point of running it |

## The candidates, assessed honestly

### 14. The four-arm decomposition

The strongest candidate, and the reason the arm set has four members
(`protocol/method_arms.md`). No located work computes, on one problem, a weighted sum and a
canonical NBI over the **same** surrogates with the **same** weight grid, separating the
normalization box from the CHIM geometry, and then separates surrogate anchors from real anchors on
top of that.

**What could sink it.** If the three contrasts all measure approximately zero, the paper has a
carefully controlled null result on one problem family. That is publishable but it is a different
paper, and the discussion must be drafted for that outcome before the campaign runs, not after.
Paper 1's experience is directly relevant: its most useful finding came from reporting a result
against interest.

**It also is not new in kind.** Paper 1 already runs a three-arm decomposition on two methodological
factors. Paper 2 adds a third factor and a different problem. The manuscript must say that plainly
rather than presenting the design as an innovation.

### 15. The identity audit

`literature_review.md` Cluster F found no established genre of published method-versus-code audits in
optimization. That is a thin basis for a claim, and the audit here has a specific weakness: **the
discrepancy it documents was already recorded by the author**, in `docs/METHODOLOGY_DECISIONS.md` D1,
before the audit began. The audit confirms and makes reproducible; it does not discover.

**Recommendation: do not claim this as a contribution.** Report it in the methodology as the reason
the arm set is what it is. A reviewer who reads it as "the authors audited their own earlier code and
found it did not match the text" will be right, and the appropriate register is matter-of-fact.
There is to be no sensationalizing of the discrepancy, and no framing that implies concealment.

### 16. The non-reproducible cost objective

Measured: running the frozen pipeline under two environments gives a quality surface agreeing to six
decimals and a cost surface that does not, because `Time_MeanFold` is wall-clock training time.

Real, and consequential for anyone replicating this line of work. But "wall-clock timing is not
reproducible" is not a surprising statement, and the contribution is the demonstration on a specific
published pipeline rather than the insight. **Report as a methodological caution with the numbers
attached. One paragraph, not a section.**

### 17. The aggregation sensitivity

Measured on the reproduced MAGIC design: the unweighted mean of z-scored factor scores ranks the
design at Spearman 0.374 against a variance-weighted composite, shares 1 of 10 top design rows, and
picks a different best row.

This is the most interesting of the audit findings because it is not about a bug. It is about a
modelling choice inside an established method that nobody reports and nobody varies, and it moves
the objective the whole pipeline optimizes.

**Strengthening it is cheap and should be pre-registered:** run the sensitivity across the whole
panel and all 30 replications rather than on one design, and report how often the two weightings
disagree about the returned front, not only about the design ranking. That converts an anecdote into
a result. It is in `protocol/EXPERIMENT_PROTOCOL.md` as a planned secondary analysis.

## Claim blacklist

Must not appear as positive claims:

1. That this work introduces NBI, VRF objectives, or their combination.
2. That this work is the first to apply design of experiments or response surfaces to hyperparameter
   tuning, or to XGBoost hyperparameters.
3. That this work introduces multiobjective hyperparameter optimization or the quality-versus-cost
   trade-off.
4. That applying NBI-VRF to hyperparameter optimization is, by itself, the contribution.
5. That the dissertation's results are invalid. They are the results of a weighted-sum method, which
   is a legitimate method; what was wrong was the name.
6. That **their code implements canonical NBI**, of any implementation not audited here.
7. That the audit discovered the weighted-sum discrepancy. It confirmed an existing record.
8. Any framing of the discrepancy as concealment, scandal or misconduct.
9. That NBI produces more uniformly spaced fronts, unless measured here and found to.
10. That a bounded literature search establishes absence.

## The contribution statement to defend

Draft, to be revised once the campaign reports:

> Hyperparameter optimization by design of experiments, response surfaces and Normal Boundary
> Intersection over Varimax-rotated factor objectives is an established construction, and this paper
> claims no part of it. What it contributes is a controlled decomposition of that construction into
> the three choices it silently bundles — how the objective reference is built, which scalarization
> geometry is used, and where the anchors come from — measured on one problem with everything else
> held fixed, and an account of which of those choices the outcome is actually sensitive to.

Whether that statement survives depends on element 18.
