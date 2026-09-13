# Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have
appeared to influence the work reported in this paper.

## Disclosure of related prior work by the present authors

This manuscript is a deliberate follow-up to, and a stress test of, a framework developed by an overlapping author
team and published in this journal:

> Pereira, M. C., Tertuliano Ribeiro, C., Mendes, R. R. A., Campos, P. H. S., de Paiva, A. P. (2025). A hybrid
> multivariate normal boundary intersection approach with post-optimization assisted by mixture design of
> experiments. *Engineering Applications of Artificial Intelligence*, 162, 112510.
> DOI 10.1016/j.engappai.2025.112510.

**Two of the three authors of the present manuscript (M. C. Pereira, first author of that paper, and A. P. de Paiva,
its senior author) are co-authors here, and the present first author was a co-author there.** The overlap is
therefore substantial and deliberate, and we bring it to the editor's attention rather than leaving it to be
discovered.

Also relevant: the first author's master's dissertation (Ribeiro, C. T., 2026, Universidade Federal de Itajubá, open
access, no DOI), which applies the same framework to hyperparameter optimization and whose repository this study
builds on.

### Why we consider this an appropriate follow-up rather than a self-citation exercise

- The manuscript claims **no part** of the DoE–RSM–NBI construction as new, and says so in the abstract, the
  introduction, a dedicated related-work subsection and the contribution list. The construction is prior art four
  times over, including in Kwon, Lee and Lee (2024) for classifier ensembles specifically, by an unrelated group.
- One of the three configurations studied here, **NBI-A — surrogate objectives with anchors taken from the
  surrogate's own optima — is the construction of the 2025 paper, and we report it failing on two of our four
  datasets.** The manuscript's central corrective finding is directed at our own prior method.
- We additionally report an external optimizer (NSGA-II, at a matched real-objective evaluation budget)
  outperforming the pipeline's strongest arm on all four datasets.
- No prose, figure or table is reused from any prior publication.

A dimension-by-dimension self-overlap analysis across thirteen dimensions accompanies the submission as a
supplementary document.

### Request to the editor

Because two of the five authors of the predecessor are co-authors here, and a third (R. R. A. Mendes) is a frequent
collaborator, we ask that reviewers be selected from outside that collaboration network, and in particular that the
remaining co-authors of the 2025 paper and of Rocha et al. (2025, IEEE Access) not be invited.
