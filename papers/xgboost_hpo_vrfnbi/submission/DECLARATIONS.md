# Submission declarations — extracted from `paper2-manuscript-v5`

Every block below is reproduced verbatim from the manuscript at the frozen tag
by `scripts/build_submission_package.py`. None is retyped. If a publisher form
requires one of these fields, paste it from here rather than from memory.

## Title

Separating Scalarization Specification, Pareto Geometry, and Anchor Provenance in Surrogate-Assisted Multiobjective XGBoost Hyperparameter Optimization: A Replicated Study

## Abstract

Surrogate-assisted multiobjective hyperparameter optimization pipelines routinely
change scalarization specification, Pareto-front construction geometry, and anchor
provenance together, so an observed difference cannot be attributed to any one of
them. We report a prospectively frozen, provenance-preserving reconstruction that
separates the three: seven XGBoost hyperparameters on an 88-run face-centred central
composite design, two minimized objectives — a Varimax-rotated composite quality factor
and a leaf-count model-complexity factor — four datasets, 30 replicated outer
partitions, and every returned candidate revalidated on the real learner (396,120
logical evaluations).

Replacing the archived observed-extrema normalization with a payoff-matrix reference
produced no detectable change. Holding the surrogate model, scaling, decision space,
candidate realization and budget fixed, canonical Normal Boundary Intersection improved
CORE-relative hypervolume over specification-matched weighted scalarization on all three geometry-positive datasets: $+0.037$ (26/30 wins),
$+0.255$ (30/30) and $+0.228$ (24/30), Holm-significant within each dataset. Replacing
surrogate-derived payoff information with a pre-specified empirical-real anchor
procedure gave no benefit and degraded the front on two of the three ($-0.074$;
$-0.019$, not significant under the CORE reference; $-0.409$); the deficit is associated, on three of four
datasets, with contraction of the convex hull of individual minima rather than with
anchor point-set composition. CORE and AUGMENTED references agree in 11 of 12 cells. Under the frozen
direct-search comparator budget the median paired difference favoured a coarse grid
over NBI-S on all four datasets, although the grid received 386 real evaluations
against 186 standalone for the scalarization arms. A prospectively
designated boundary dataset did not resolve: a larger point estimate than the smallest
significant effect, at roughly thirty times its dispersion.

## **Keywords:** multiobjective hyperparameter optimization; Normal Boundary
Intersection; weighted scalarization; response surface methodology; design of
experiments; XGBoost; replicated evaluation; Pareto indicators

## Funding

**Funding.** The master's research underlying this study received support from the
Fundação de Amparo à Pesquisa do Estado de Minas Gerais (FAPEMIG), project BPD-01045-22,
and from the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES).
Anderson Paulo de Paiva acknowledges research support from the Conselho Nacional de
Desenvolvimento Científico e Tecnológico (CNPq), process 312844/2023-9.

## Acknowledgements

**Acknowledgements.** The authors acknowledge the Universidade Federal de Itajubá
(UNIFEI) for institutional support.

## Competing interests

**Competing interests.** The authors declare no competing interests.

## Data and code availability

**Data and code availability.** The protocol, the frozen factor models, every analysis
artifact and every script are version-controlled and tagged; see §9. Raw datasets are
public and are retrieved by checksum-verified loaders; evaluation caches are
deliberately unversioned.

## Author contributions (CRediT)

**Author contributions (CRediT).**

*Caio Tertuliano Ribeiro* — Conceptualization; Methodology; Software; Validation;
Formal analysis; Investigation; Data curation; Project administration; Writing –
original draft; Visualization.

*Matheus Costa Pereira* — Methodology; Writing – review & editing.

*Anderson Paulo de Paiva* — Conceptualization; Supervision; Funding acquisition;
Writing – review & editing.

The CRediT role *Resources* is not assigned. The taxonomy does not require every role to
be represented, and no author is credited with a contribution the work did not involve.
