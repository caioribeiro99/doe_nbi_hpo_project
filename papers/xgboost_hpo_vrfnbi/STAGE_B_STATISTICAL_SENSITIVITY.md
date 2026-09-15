# What R = 30 can resolve

**This is not a power analysis used to choose a replication count.** R = 30 is fixed, inherited from
the dissertation and from Paper 1, and it does not move. The question here is the other one: given
that R, what is the smallest paired effect this design can distinguish, and how wide will the
intervals be?

It exists because the adversarial review found that Stage B had been silently descoped. `§12` of the
protocol specified Stage B as "all four arms and all comparators on one partition of one dataset",
producing the paired standard deviation of each primary endpoint and the resulting detectable effect.
What ran was a throughput benchmark. Two of three deliverables did not exist while a protocol
amendment was being decided. Recorded as amendment 11.

Reproduce: `python scripts/stage_b_statistical_sensitivity.py`. Record:
`audits/stage_b_sensitivity.json`.

---

## What this uses, and what it does not

Only pre-campaign information: the 166 committed Stage A evaluations per dataset, and analytical
results that depend on no arm outcome. **No arm has been run.**

The paired standard deviations come from a **resampling proxy**, and it is labelled as one
everywhere. The 166 evaluated configurations of a dataset are repeatedly split into two disjoint
halves; each half's non-dominated set is scored against a common reference; and the spread of the
resulting paired indicator differences is taken as a stand-in for between-replication variability.

Two honest caveats. Half-sized sets are noisier than the campaign's returned sets, so the proxy is
**conservative** — the real paired standard deviation should be smaller and the real detectable
effect finer. And a resampling split is not a fresh outer partition, so this bounds the sampling
component of variability and not the partition component. The campaign measures the real quantity;
this establishes what to expect.

## What R = 30 resolves

Two-sided paired test at α = 0.05 and 80% power, 200 resampling draws per dataset:

| Dataset | indicator | paired sd (proxy) | minimum detectable difference | expected 95% interval half-width | detectable at n_test/n_train = 0.25 (80/20) |
|---|---|---:|---:|---:|---:|
| MAGIC | hypervolume ratio | 0.0150 | **0.0079** | 0.0056 | 0.0231 |
| MAGIC | IGD⁺ | 0.0098 | 0.0052 | 0.0037 | 0.0152 |
| Spambase | hypervolume ratio | 0.0522 | **0.0276** | 0.0195 | 0.0805 |
| Spambase | IGD⁺ | 0.0379 | 0.0201 | 0.0141 | 0.0585 |
| Adult | hypervolume ratio | 0.0277 | **0.0147** | 0.0103 | 0.0428 |
| Adult | IGD⁺ | 0.0184 | 0.0097 | 0.0069 | 0.0283 |
| Bank Marketing | hypervolume ratio | 0.0417 | **0.0221** | 0.0156 | 0.0644 |
| Bank Marketing | IGD⁺ | 0.0196 | 0.0104 | 0.0073 | 0.0302 |

The standardized effect R = 30 detects at 80% power is **0.532 paired standard deviations**, which
is a medium effect. The design cannot resolve small ones.

**Win fractions.** The smallest majority out of 30 whose Wilson 95% interval excludes one half is
**21 of 30, that is 70%**. A 19-of-30 or 20-of-30 majority is not evidence of a direction at this
replication count, however suggestive it looks, and will not be reported as one.

## What this means for the study, stated before any result exists

**The hypervolume-ratio differences the study can resolve are between about 0.8 and 2.8 percentage
points of the reference hypervolume** under the uncorrected paired test, depending on the dataset,
and between **2.3 and 8.1 percentage points** under the corrected test at n_test/n_train = 0.25.
Magic is the most sensitive, by a factor of 3.5, and Spambase the least. Every figure in this
paragraph is regenerated from `audits/stage_b_sensitivity.json`; an earlier version published 0.6 to
2.4 and 2.0 to 8.0, computed on the superseded factor algebra.

Two consequences follow, and both are written down now rather than after the campaign.

**If a primary contrast comes back smaller than the figures above, the correct report is an interval
that includes zero, not a null result.** "No significant difference" and "no difference" are not the
same statement, and at these resolutions the study is only entitled to the first.

**The three identifying contrasts are not equally likely to clear the bar.** Paper 1 measured the
anchor-provenance effect as the largest of its methodological factors, by a wide margin. If the
geometry contrast here, NBI-S against WS-S, is genuinely of the order of a percentage point of
hypervolume ratio, this design will resolve it on MAGIC and may not on Spambase. That is a property
of the design, it was knowable in advance, and it is knowable in advance because this analysis was
run before the campaign rather than after.

## Treatment of the corrected test, and a correction to it

The final protocol review found that the quantity reported here as the Nadeau and Bengio correction
**was not theirs**. Their result is that for overlapping resamples the variance of the mean
difference is `σ²(1/n + ρ/(1−ρ))`, so the standard error is `sd·√(1/n + ρ/(1−ρ))`. The earlier
version instead inflated `sd/√n` by `√(1 + ρ/(1−ρ))`, which divides the correction term by `n` and
understates it badly at R = 30.

Corrected, the cost of the correction is much larger than previously reported: the standard error
inflates by **sqrt(8.5) = 2.9155×** at n_test/n_train = 0.25, not 1.15×. An earlier version of this sentence said 3.31× “at ρ = 0.25”, which is the equicorrelation variant and corresponds to a 75/25 split, not this protocol's 80/20. The detectable hypervolume-ratio difference under the
corrected test is therefore **0.023 to 0.081** depending on dataset, not 0.008 to 0.028.

That is a material change and it is reported rather than buried. It is also why the correction is a
**sensitivity and never the primary test**: it was derived for the generalization error of a learner
under repeated resampling, and a Pareto quality indicator computed on a returned set is not that
quantity. Applying it here is a transfer, and at n_test/n_train = 0.25 it would leave this design able to
resolve only fairly large differences.

Primary evidence remains the paired effect distribution: the median with its bootstrap interval, the
win, tie and loss counts with Wilson intervals, and the matched-pairs rank-biserial correlation.

## What this analysis is forbidden from changing

The primary endpoints, the comparison family, the gate thresholds, the effect directions, the
datasets, and R. None was chosen with reference to these numbers, and none may be revised because of
them. If the design turns out to be underpowered for a contrast, that is reported as a limitation of
the study, not repaired by changing the study.
