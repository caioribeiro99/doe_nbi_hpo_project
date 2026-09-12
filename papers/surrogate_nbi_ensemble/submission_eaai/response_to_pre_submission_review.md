# Response to the internal pre-submission review

Before submission the manuscript was put through a three-reviewer adversarial simulation: a multiobjective-
optimization specialist, a machine-learning evaluation specialist and an associate editor assessing novelty and
scope. Each was instructed to check claims against the underlying CSVs rather than the prose, and to recommend
rejection if warranted. A fourth agent verified every criticism against the artifacts before triage. The full record,
including all three reviews and the verification verdicts, is in `REVIEWER_SIMULATION.md`.

**Outcome: MAJOR from all three. 21 major comments — 17 valid, 4 partially valid, none invalid. All 20 must-fix items
have been applied; none required touching the frozen experiment.**

## The two corrections that changed what the paper claims

**1. The anchor contrast was not a single-factor experiment.** Reviewer 1 observed that vertex values of the
scalarization weight return their anchor directly, so the real-anchor arm's candidate set *contains* the three real
single-objective optima, which are also points of the reference. We reproduced this independently and added a
control that rescores the surrogate-anchored arm's own set augmented with the same anchors. Injection alone closes a
median 80% of the gap on Santander, 90% on BNP Paribas, 49% on Porto Seguro and 1% on UCI credit. The manuscript now
reports this decomposition, no longer describes the contrast as clean, and enumerates the four things that change
together. A reproducible script and CSV accompany it.

**2. The compute accounting understated the real-anchor arm.** The previously quoted ratios were solver-stage only,
but both real-anchor arms depend on the stage that computes the anchors, which is dominated by a 4 × 10⁴-evaluation
direct search. Charged standalone, the metamodel-free premium is 2.3–6.2×, not 3–73×, and real anchors are not free.
Three previously inconsistent statements now agree.

## Corrections of fact and attribution

- Five verified references were in the bibliography but cited nowhere. Each is now cited and the surrounding claim
  narrowed: metamodel-free NBI on measured objectives is published (Gellerich et al. 2023); that misplaced anchors
  damage a scalarization is established (Isermann 1988; Deb et al. 2010; Herrmann et al. 2026) — what is new is the
  *surrogate* provenance and the controlled contrast; that a change of cost accounting reorders comparisons is known
  across architectures (Wang et al. 2022); inference cost as a Pareto objective for post-hoc weighted tabular
  ensembles is prior art (Maier and Purucker 2024, 2026).
- The conclusions are now explicitly scoped to **one-shot, no-infill** surrogate pipelines, since the
  surrogate-assisted literature we cite refits during search.
- A real table-generation bug was found: medians were paired with *mean* bootstrap intervals, printing point
  estimates outside their own intervals. Fixed, and the numerical audit now asserts interval consistency structurally.
- The sample-core reference is described precisely: it contains no output of the five compared methods, but it does
  carry ε-constraint points and the single-objective references.
- The reliability gate's non-independence from order selection is now disclosed — both use the same 100 unseen
  compositions.

## Claims weakened or withdrawn

- "Real anchors matter most where the surrogate is unreliable" — withdrawn; the medians invert that ordering.
- The BNP Paribas hypervolume/IGD⁺ "genuine split" — reference-qualified; it reverses under the sample-core reference.
- "Three of four datasets" framings of the anchor result — replaced by the accurate pattern (two clean, one near-tie,
  one reversing under the support cost).
- Non-smoothness demoted from *the* explanation to a contributing factor.
- Median effect sizes now appear beside every win count, so 30/30 cannot be read as a large effect.

## Deliberately not done

Three reviewer requests would require reopening the frozen protocol and are recorded as reviewer-response material
rather than acted on: a fifth dataset, a larger replication count, and a second evolutionary optimizer (MOEA/D or
NSGA-III). We judged that adding a second evolutionary algorithm would not change any claim, since the first already
outperforms the pipeline's best arm and the paper's contribution is diagnostic.
