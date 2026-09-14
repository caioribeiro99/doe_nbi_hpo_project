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

Two-sided paired test at α = 0.05 and 80% power, 150 resampling draws per dataset:

| Dataset | indicator | paired sd (proxy) | minimum detectable difference | expected 95% interval half-width | detectable at ρ = 0.25 |
|---|---|---:|---:|---:|---:|
| MAGIC | hypervolume ratio | 0.0118 | **0.0063** | 0.0044 | 0.0072 |
| MAGIC | IGD⁺ | 0.0096 | 0.0051 | 0.0036 | 0.0059 |
| Spambase | hypervolume ratio | 0.0423 | **0.0224** | 0.0158 | 0.0259 |
| Spambase | IGD⁺ | 0.0172 | 0.0091 | 0.0064 | 0.0105 |
| Adult | hypervolume ratio | 0.0334 | **0.0177** | 0.0125 | 0.0204 |
| Adult | IGD⁺ | 0.0165 | 0.0087 | 0.0062 | 0.0101 |
| Bank Marketing | hypervolume ratio | 0.0297 | **0.0157** | 0.0111 | 0.0182 |
| Bank Marketing | IGD⁺ | 0.0152 | 0.0080 | 0.0057 | 0.0093 |

The standardized effect R = 30 detects at 80% power is **0.532 paired standard deviations**, which
is a medium effect. The design cannot resolve small ones.

**Win fractions.** The smallest majority out of 30 whose Wilson 95% interval excludes one half is
**21 of 30, that is 70%**. A 19-of-30 or 20-of-30 majority is not evidence of a direction at this
replication count, however suggestive it looks, and will not be reported as one.

## What this means for the study, stated before any result exists

**The hypervolume-ratio differences the study can resolve are between about 0.6 and 2.2 percentage
points of the reference hypervolume**, depending on the dataset. MAGIC is the most sensitive by a
factor of three and Spambase the least.

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

## Treatment of the corrected test

The Nadeau and Bengio correction inflates the variance for overlapping resamples. It is reported as
a **sensitivity, not as the primary test**, for a reason the protocol now states: the correction was
derived for the generalization error of a learner under repeated resampling, and a Pareto quality
indicator computed on a returned set is not that quantity. The table gives the detectable effect at
ρ = 0.25 so the reader can see how much the correction costs — between 14% and 16% coarser
resolution — without the paper resting on it.

Primary evidence remains the paired effect distribution: the median with its bootstrap interval, the
win, tie and loss counts with Wilson intervals, and the matched-pairs rank-biserial correlation.

## What this analysis is forbidden from changing

The primary endpoints, the comparison family, the gate thresholds, the effect directions, the
datasets, and R. None was chosen with reference to these numbers, and none may be revised because of
them. If the design turns out to be underpowered for a contrast, that is reported as a limitation of
the study, not repaired by changing the study.
