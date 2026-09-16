## 6.3 Front-construction geometry: WS-S versus NBI-S

The second contrast in the frozen decomposition holds the surrogate models, the
canonicalized objective set, the scaling, the decision space, the candidate realizer,
the weight grid and the payoff reference fixed, and changes only the scalarization
geometry: a weighted sum of min–max normalized surrogate objectives (WS-S) against a
canonical NBI construction with a CHIM, a quasi-normal direction and a $\max t$
subproblem (NBI-S). Isolation was enforced at run time by a contrast fingerprint over
the solver configuration, the realizer, the weight grid, the surrogate identities and
the reference. Every returned candidate was re-evaluated on the real learner before any
indicator was computed. Differences are paired by replication and formed as
(second − first), with $R = 30$ per dataset; Holm correction is applied within each
dataset over the family of three primary contrasts, and the dataset — not the
replication — is the generalization unit.

Under these conditions, canonical NBI front construction produced higher
real-revalidated CORE-relative hypervolume than specification-matched weighted
scalarization on all three primary datasets. The medians are **+0.0374** on MAGIC,
**+0.2549** on Adult and **+0.2281** on Bank Marketing; the win/tie/loss counts are
26/0/4, 30/0/0 and 24/0/6; the percentile bootstrap intervals on the median are
$[+0.0144, +0.0687]$, $[+0.1740, +0.5614]$ and $[+0.1670, +0.4741]$, none of them
spanning zero; and the matched-pairs rank-biserial correlation is $+0.92$, $+1.00$ and
$+0.88$. The secondary tests are Holm-significant within every primary dataset
($p = 9.42\times10^{-7}$, $5.59\times10^{-9}$ and $9.72\times10^{-6}$). This is the
primary positive result of the study, and its scope is exactly these datasets, this
surrogate architecture, two objectives and this budget. It is not a claim that NBI
recovers a true Pareto front: the endpoint is a ratio against CORE, a finite
method-independent empirical reference of 288 points before Pareto filtering, so a
ratio above 1 means only that the candidate set improved on that finite reference
([TAB:primary_contrasts]).

Two qualifications belong in the same breath as the result. First, the geometry
advantage did not require a surrogate that passed the study's own external reliability
criterion: on Adult and Bank Marketing the composite-quality gate passed in 0 of 30
replications, and the geometry effect is Holm-significant on both. The gate was
diagnostic rather than adaptive — every arm ran at every replication regardless of its
outcome — and absolute surrogate trustworthiness and relative front-construction
geometry are different questions, of which this study answers only the second. Second,
at the frozen comparator budget an evaluation-matched coarse grid attained a higher
median CORE-relative hypervolume than every surrogate-assisted arm on every dataset in
the panel (NBI-S 0.9430 vs grid 1.0068 on MAGIC; 0.9951 vs 1.0476 on Adult; 0.9979 vs
1.0625 on Bank Marketing; 1.0812 vs 1.0192 on Spambase). The budget rule is matched to
the most expensive arm (NBI-R, 386 evaluations), while WS-S and NBI-S cost 186
standalone, so the comparators received roughly twice the real evaluations those arms
require; the rule was frozen before any result existed and is not revised here. The
geometry contrast compares two front constructions at a fixed surrogate budget; it is
not a demonstration that the surrogate-assisted family is preferable to direct search.
Section 6.7 reports this baseline in full.

**[TAB:primary_contrasts] Primary paired contrasts on CORE-relative hypervolume ratio,
$R = 30$, differences formed as (second − first).**

| Contrast | Dataset | Median diff. | Bootstrap interval | Win/tie/loss | Rank-biserial | Holm $p$ |
|---|---|---|---|---|---|---|
| HISTORICAL-WS → WS-S | MAGIC | +0.0000 | [−0.0022, +0.0056] | 14/3/13 | — | 0.5165 |
| HISTORICAL-WS → WS-S | Adult | −0.0006 | [−0.0033, +0.0000] | 9/4/17 | — | 0.2087 |
| HISTORICAL-WS → WS-S | Bank Marketing | +0.0000 | [−0.0088, +0.0000] | 8/8/14 | — | 0.3896 |
| WS-S → NBI-S | MAGIC | **+0.0374** | [+0.0144, +0.0687] | 26/0/4 | +0.92 | 9.42e−07 |
| WS-S → NBI-S | Adult | **+0.2549** | [+0.1740, +0.5614] | 30/0/0 | +1.00 | 5.59e−09 |
| WS-S → NBI-S | Bank Marketing | **+0.2281** | [+0.1670, +0.4741] | 24/0/6 | +0.88 | 9.72e−06 |
| WS-S → NBI-S | Spambase (boundary) | +0.0500 | [−0.0043, +0.4369] | 18/2/10 | — | 0.1175 |
| NBI-S → NBI-R | MAGIC | **−0.0741** | [−0.0957, −0.0482] | 2/0/28 | — | 0.0000 |
| NBI-S → NBI-R | Adult | −0.0192 | [−0.1610, +0.0299] | 11/0/19 | — | 0.1607 |
| NBI-S → NBI-R | Bank Marketing | **−0.4093** | [−0.5821, −0.2059] | 7/0/23 | — | 0.0000 |

Spambase is shown for completeness and is excluded from the primary inferential family
by prospective designation; it is treated in Section 6.6.

### 6.3.1 The geometry effect is indicator-specific

The advantage does not extend uniformly across the frozen secondary indicators, and
that limitation is reported here rather than deferred. On convergence and coverage
geometry the direction matches hypervolume: median IGD⁺ is 0.0511 for NBI-S against
0.0930 for WS-S on MAGIC, 0.0579 against 0.1688 on Adult, and 0.0915 against 0.1948 on
Bank Marketing, lower being better. On the joint non-dominated fraction it does not:
the median differences are +0.0208 on MAGIC, +0.0714 on Adult and +0.0000 on Bank
Marketing, and the comparison is non-significant on all four datasets, including
Spambase (raw $p$ = 0.160, 0.093, 0.627 and 0.586). Secondary indicators are reported
descriptively, with intervals and without multiplicity-adjusted tests, and none of them
is described as significant. The defensible statement is therefore narrow: under the
frozen study conditions the advantage is specifically an improvement in convergence and
coverage geometry as measured by hypervolume and IGD⁺. The data do not support a
general better-coverage claim, and in particular do not support a claim that NBI
construction returned more non-dominated solutions [FIG:indicator_panel].

## 6.4 Anchor and payoff provenance: NBI-S versus NBI-R

The third contrast holds the NBI geometry fixed and changes only the provenance of the
payoff matrix and anchors, from surrogate-derived (NBI-S) to the pre-specified
empirical-real procedure that obtains each anchor by direct search on the true
objectives (NBI-R). The quasi-normal direction is derived from each arm's own payoff
matrix and so moves with the reference by construction; that displacement is the
intervention, not a confound.

Replacing surrogate-derived payoff information with the frozen empirical-real anchor
procedure did not improve the confirmatory NBI fronts, and degraded CORE-relative
hypervolume on two of the three primary datasets. The medians are **−0.0741** on MAGIC
(2/0/28, $[-0.0957, -0.0482]$, Holm $p$ = 0.0000), **−0.4093** on Bank Marketing
(7/0/23, $[-0.5821, -0.2059]$, Holm $p$ = 0.0000) and **−0.0192** on Adult (11/0/19,
$[-0.1610, +0.0299]$, Holm $p$ = 0.1607), the last a non-detection under CORE with an
interval spanning zero. This negative result carries the same evidentiary weight as the
geometry result of Section 6.3 and is reported at the same prominence: spending real
evaluations to relocate the payoff matrix onto empirically obtained anchors did not buy
better fronts under these conditions, and no alternative formulation under which it
does is offered here.

### 6.4.1 What differs mechanically, stated as association

Two prospectively frozen diagnostics narrow where the deficit comes from. The
anchor-injection control inserts NBI-R's empirical anchors into NBI-S's revalidated
candidate set and changes nothing else; because the control set is a superset of
NBI-S's, its effect is non-negative by construction. Its median effect is +0.0133 on
MAGIC and +0.0000 on the other three datasets, against full NBI-S → NBI-R gaps of
−0.0741, −0.0192, −0.4093 and −0.6932. The deficit is therefore associated with the
geometric consequences of the relocated payoff matrix rather than with anchor
point-set composition, though the control does not establish that the negative is
entirely geometric. Solver health is identical between the two arms — certified
fraction 1.000, maximum equality residual 0.000, and 20 distinct realized
configurations for both on every dataset — so the deficit is not solver failure, not
rounding and not candidate collapse.

The second diagnostic is the CHIM extent $\|\Phi_{:,0} - \Phi_{:,1}\|$, which at $q = 2$
is the segment along which the subproblems distribute their targets. Under empirical
anchors it contracts to 20–81% of its surrogate-derived extent: ratio 0.811 on MAGIC,
0.204 on Adult, 0.272 on Bank Marketing and 0.370 on Spambase. Within-dataset Spearman
correlations between the per-replication extent ratio and the per-replication
hypervolume gap are +0.683 on Adult, +0.678 on Bank Marketing, +0.320 on Spambase and
**−0.111 on MAGIC**. This is an association measured across 30 replications within a
dataset, present on three datasets and absent on the fourth; it is not a demonstrated
cause of the deficit, and it is not presented as one [FIG:chim_contraction].

## 6.5 Reference sensitivity

Every dataset-by-contrast conclusion was recomputed under both the primary
method-independent CORE reference and the mandatory AUGMENTED sensitivity reference.
**Eleven of the twelve conclusions agree in direction and in significance, and no cell
anywhere flips direction.** The single disagreement is Adult's anchor-provenance
contrast: under CORE the median is −0.0192 with Holm $p$ = 0.1607, non-significant;
under AUGMENTED the median is −0.0535 with Holm $p$ = 0.0174, significant. The sign is
negative under both references, so the disagreement is one of resolution, not of
direction. Under AUGMENTED that cell is significant only on the secondary test: the
primary descriptive triple remains equivocal, with the bootstrap interval on the median
spanning zero.

CORE remains primary because it was prospectively frozen as method-independent before
any result existed; AUGMENTED remains a mandatory sensitivity, not an alternative from
which a reference may be selected after the fact. The reported Adult conclusion is
accordingly the CORE non-detection, with the AUGMENTED result stated alongside it here
and in [TAB:reference_sensitivity].
