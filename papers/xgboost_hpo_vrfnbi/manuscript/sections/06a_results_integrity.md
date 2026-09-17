## 6. Results

### 6.1 Protocol integrity, accounting and the screened panel

The confirmatory campaign executed **120 of 120 units with 0 failures and 0 methodological
failures**, in 9 h 46 min of wall clock on 14 workers at one thread each. A unit is one
`(dataset, replication)` pair — 4 datasets $\times$ $R = 30$ replicated outer partitions — and every
arm, control and baseline ran at every unit.

The scientific budget is a **request ledger**: every request is charged whether or not it is served
from cache, so duplicate candidates, rounding and cache behaviour cannot reduce a method's charge.
The campaign spent **396,120 logical evaluations**, exactly the declared method-stage
registry total, with **every one of the 120 units reconciling individually**. Physical execution
required **377,316 unique physical fits**, a 4.75% cache hit rate. The two are kept distinct in
every statement below: only the logical figure enters a fairness comparison, and the physical figure
is an engineering quantity that enters none. The 15,360 evaluations of the
unmatched NSGA-II context baseline are **already inside the four `rep_00` ledgers** summing to
396,120; they are not added again. A total that adds them a second time is
wrong by construction.

Objective definitions were frozen before execution and **applied, never refitted**. One factor model
per dataset was fitted on that dataset's 88 face-centred central composite design rows and nothing
else, then applied unchanged to all 30 replications; the per-replication refit is reported only as a
Tucker-congruence sensitivity and never scores a front. Fitting once per dataset keeps the
30 paired indicator values inside one objective space, which a per-replication refit would not, and
the 78-point external validation set is disjoint from the fitting sample in coded space on all four
datasets, so the surrogate gate audits a surface against rows that never helped define its target.

The primary endpoint is the **CORE-relative hypervolume ratio**. The CORE reference is a finite,
method-independent empirical set — the 88 design rows plus the 200 anchor-search rows, **288 points
before Pareto filtering, present in every unit** — that no compared method contributes to. It is not
the true Pareto front, and a ratio above 1 is not an error: it means the returned set improved on
that finite reference.

Dataset roles were assigned by prospective screening and not revised (see Supplement S5).
MAGIC, Adult and Bank Marketing met the nonlinear-front criterion and form the primary geometry
panel; Spambase failed it and was retained in advance as a **boundary geometry control**, executed
in full under identical arms, budgets and seeds, and excluded only from the primary inferential
family. Dataset is the generalization unit; nothing is pooled.

Two integrity items are reported rather than smoothed. First, the campaign output contains **140
non-finite cells**, being Schott spacing and its coefficient of variation on the **70
method-by-reference blocks whose front has one point**. Schott spacing requires at least two gaps and
is undefined on a single-point front; `NaN` is the mathematically correct value there. Those blocks
are excluded from spacing summaries rather than propagated through a median. Second, five independent
verifiers recomputed eleven headline claims from the raw per-unit artifacts, and the central contrast
was additionally rebuilt from `candidate_revalidation` rows through the frozen factor model with an
independent dominance filter and a two-dimensional hypervolume sweep, reproducing all 480 published
per-unit values to $0.000\mathrm{e}{+}00$. Nine claims reproduced exactly; two were corrected, and
both were the author's.

### 6.2 The historical reconstruction: specification and normalization

A code-level reconstruction of the archived dissertation implementation established that its front
construction is a min–max normalized weighted scalarization over component-wise observed extrema of
the design rows, although the historical text uses Normal Boundary Intersection terminology. That
fact fixes the arm set: the historical method enters as a weighted-sum arm, and the geometry question
needs a separate, specification-matched contrast.

**HISTORICAL-WS $\rightarrow$ WS-S isolates normalization alone**: the same weighted sum, the same
surrogates, the same symmetric weight grid, differing only in whether the normalization box comes
from observed design-row extrema or from the payoff matrix. The effect is undetectable on every
primary dataset (Table 3).

| dataset | median difference | bootstrap interval | win/tie/loss | Holm $p$ |
|---|---:|---|---:|---:|
| MAGIC | +0.0000 | [-0.0022, +0.0056] | 14/3/13 | 0.5165 |
| Adult | -0.0006 | [-0.0033, +0.0000] | 9/4/17 | 0.2087 |
| Bank Marketing | +0.0000 | [-0.0088, +0.0000] | 8/8/14 | 0.3896 |

**Table 3.** The historical specification contrast, HISTORICAL-WS $\rightarrow$ WS-S, CORE reference, $R=30$.

This is a **non-detection at $R = 30$**, not a demonstration of no effect: exact ties occur in 3, 4
and 8 of the 30 replications on MAGIC, Adult and Bank Marketing, and the frozen design resolves only
medium effects. It does establish that this contrast — the one place a normalization-reference
hypothesis would have to appear — does not carry the historical-to-canonical difference.

The **as-run to shared-specification gap** is a descriptive control quantity rather than a member of
the frozen primary family, and carries no multiplicity correction. It compares the bit-faithful
frozen dissertation solver, with its own asymmetric weight grid, against the same weighted sum under
this campaign's surrogates, coding and symmetric grid: median paired difference in
CORE-relative hypervolume ratio +0.0994 [+0.0623, +0.1287] on MAGIC, +0.1480 [-0.0815, +0.2702] on
Adult, +0.3126 [+0.1981, +0.4484] on Bank Marketing and +0.5404 [+0.1526, +0.8736] on Spambase, the
panel's largest. That gap is the whole historical reconstruction effect, and it is confounded by
construction: surrogate identity, coding and weight-grid symmetry move together in it. Given the
non-detection above, the normalization reference is not what explains it, and we attribute it no
further (Figure 1).
