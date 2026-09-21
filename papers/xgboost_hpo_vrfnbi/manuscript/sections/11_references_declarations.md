## Declarations

**Prior dissemination and relationship to a dissertation.** This study extends work
first developed in the first author's master's dissertation (Ribeiro, 2026, UNIFEI,
unpublished), which applied a PCA/Varimax latent-objective construction to XGBoost
hyperparameter optimization. The present work reconstructs that pipeline from its
archived implementation, separates three mechanisms it changed together, and evaluates
them under a protocol frozen before any comparative result existed. The archived
implementation is reproduced by calling its frozen code unmodified rather than by
reimplementation, and is reported neutrally: a code-level reconstruction established
that it performs normalized weighted scalarization for front construction although the
historical text used NBI terminology. No claim is made about the correctness of the
dissertation's reported results, whose objective space differs from this study's
(§3.5).

**Relationship to companion work by the same authors.** Three instruments used here
were developed in a companion manuscript by the present authors (Ribeiro, Pereira and
de Paiva, 2026, unpublished, frozen and not submitted): the external surrogate
reliability gate, the comparator-budget rule — which matches direct-search
comparators to the most expensive arm rather than pairwise to each arm — and the
anchor-injection control. They are applied here, not introduced. The methodological
lineage of the latent-objective construction is set out in §2.7 and includes Pereira
et al. (2025) and de Azevedo et al. (2026), with which this work shares co-authors.

**Overlap statement.** The datasets, the frozen protocol, the confirmatory campaign and
every result reported here are specific to this study. No figure, table or result is
reproduced from the dissertation or from the companion manuscript.

**Data and code availability.** The protocol, the frozen factor models, every analysis
artifact and every script are version-controlled and tagged; see §9. Raw datasets are
public and are retrieved by checksum-verified loaders; evaluation caches are
deliberately unversioned.

**Competing interests.** The authors declare no competing interests.

**Funding.** The master's research underlying this study received support from the
Fundação de Amparo à Pesquisa do Estado de Minas Gerais (FAPEMIG), project BPD-01045-22,
and from the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES).
Anderson Paulo de Paiva acknowledges research support from the Conselho Nacional de
Desenvolvimento Científico e Tecnológico (CNPq), process 312844/2023-9.

**Acknowledgements.** The authors acknowledge the Universidade Federal de Itajubá
(UNIFEI) for institutional support.

**Author contributions (CRediT).**

*Caio Tertuliano Ribeiro* — Conceptualization; Methodology; Software; Validation;
Formal analysis; Investigation; Data curation; Project administration; Writing –
original draft; Visualization.

*Matheus Costa Pereira* — Methodology; Writing – review & editing.

*Anderson Paulo de Paiva* — Conceptualization; Supervision; Funding acquisition;
Writing – review & editing.

The CRediT role *Resources* is not assigned. The taxonomy does not require every role to
be represented, and no author is credited with a contribution the work did not involve.

---

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used Anthropic Claude and OpenAI ChatGPT in order to assist with manuscript drafting, language refinement, consistency checking and editorial review. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

This declaration concerns manuscript preparation only. The study's computational methods are described in the Methods section, and every reported result derives from the committed, version-controlled artifacts referenced there.

## References

Every entry below was verified against an authoritative source — Crossref, the
publisher's proceedings page, arXiv, or JMLR — at the DOI or identifier shown.
Machine-readable records are in `manuscript/references.bib`. Fields a source did not
state are omitted rather than inferred; two such omissions are noted explicitly.

Bergstra, J. and Bengio, Y. (2012). Random search for hyper-parameter optimization.
*Journal of Machine Learning Research*, 13, 281–305.

Costa, D. M. D., Paula, T. I., Silva, P. A. P. and Paiva, A. P. (2016). Normal boundary
intersection method based on principal components and Taguchi's signal-to-noise ratio
applied to the multiobjective optimization of 12L14 free machining steel turning
process. *The International Journal of Advanced Manufacturing Technology*, 87(1–4),
825–834. https://doi.org/10.1007/s00170-016-8478-7

Das, I. and Dennis, J. E. (1998). Normal-Boundary Intersection: a new method for
generating the Pareto surface in nonlinear multicriteria optimization problems.
*SIAM Journal on Optimization*, 8(3), 631–657.
https://doi.org/10.1137/S1052623496307510

de Azevedo, T. M., Pereira, M. C., Cesário, M. de C. and de Paiva, A. P. (2026).
Multiobjective and multivariate rationalization of CFD simulations for hydrodynamic
systems using the NBI-VRF method. *Thermal Science and Engineering Progress*, 74,
104722. https://doi.org/10.1016/j.tsep.2026.104722

Eggensperger, K., Müller, P., Mallik, N., Feurer, M., Sass, R., Klein, A., Awad, N.,
Lindauer, M. and Hutter, F. (2021). HPOBench: a collection of reproducible
multi-fidelity benchmark problems for HPO. Published at the NeurIPS Datasets and
Benchmarks Track, 2021. arXiv:2109.06716

Guerrero-Viu, J., Hauns, S., Izquierdo, S., Miotto, G., Schrodi, S., Biedenkapp, A.,
Elsken, T., Deng, D., Lindauer, M. and Hutter, F. (2021). Bag of baselines for
multi-objective joint neural architecture search and hyperparameter optimization.
arXiv:2105.01015. *(The arXiv record states no workshop venue, so none is asserted.)*

Ishibuchi, H., Masuda, H., Tanigaki, Y. and Nojima, Y. (2015). Modified distance
calculation in generational distance and inverted generational distance. In
*Evolutionary Multi-Criterion Optimization (EMO 2015)*, Lecture Notes in Computer
Science, Springer, 110–125. https://doi.org/10.1007/978-3-319-15892-1_8 *(The LNCS
series volume is omitted: sources do not state it consistently and it was not confirmed
by a second source. The DOI identifies the chapter unambiguously.)*

Karl, F., Pielok, T., Moosbauer, J., Pfisterer, F., Coors, S., Binder, M., Schneider,
L., Thomas, J., Richter, J., Lang, M., Garrido-Merchán, E. C., Branke, J. and Bischl,
B. (2023). Multi-objective hyperparameter optimization in machine learning — an
overview. *ACM Transactions on Evolutionary Learning and Optimization*, 3(4), Article
16, 1–50. https://doi.org/10.1145/3610536

Lujan-Moreno, G. A., Howard, P. R., Rojas, O. G. and Montgomery, D. C. (2018). Design
of experiments and response surface methodology to tune machine learning
hyperparameters, with a random forest case-study. *Expert Systems with Applications*,
109, 195–205. https://doi.org/10.1016/j.eswa.2018.05.024

Luz, E. R., Romão, E. L., Streitenberger, S. C., Mancilha, L. R., de Paiva, A. P. and
Balestrassi, P. P. (2021). A multiobjective optimization of the welding process in
aluminum alloy (AA) 6063 T4 tubes used in corona rings through normal boundary
intersection and multivariate techniques. *The International Journal of Advanced
Manufacturing Technology*, 117(5–6), 1517–1534.
https://doi.org/10.1007/s00170-021-07761-5

Morales-Hernández, A., Van Nieuwenhuyse, I. and Rojas Gonzalez, S. (2023). A survey on
multi-objective hyperparameter optimization algorithms for machine learning.
*Artificial Intelligence Review*, 56(8), 8043–8093.
https://doi.org/10.1007/s10462-022-10359-2

Nadeau, C. and Bengio, Y. (2003). Inference for the generalization error.
*Machine Learning*, 52(3), 239–281. https://doi.org/10.1023/A:1024068626366

Pereira, M. C., Ribeiro, C. T., Mendes, R. R. A., Campos, P. H. da S. and de Paiva,
A. P. (2025). A hybrid multivariate normal boundary intersection approach with
post-optimization assisted by mixture design of experiments. *Engineering Applications
of Artificial Intelligence*, 162, 112510.
https://doi.org/10.1016/j.engappai.2025.112510

Pfisterer, F., Schneider, L., Moosbauer, J., Binder, M. and Bischl, B. (2022). YAHPO
Gym — an efficient multi-objective multi-fidelity benchmark for hyperparameter
optimization. In *Proceedings of the First International Conference on Automated
Machine Learning*, PMLR 188, 3/1–39.
https://proceedings.mlr.press/v188/pfisterer22a.html

Ribeiro, C. T. (2026). *Multiobjective optimization of XGBoost hyperparameters using
design of experiments and latent objective construction*. MSc dissertation,
Universidade Federal de Itajubá (UNIFEI). Unpublished.

Ribeiro, C. T., Pereira, M. C. and de Paiva, A. P. (2026). *Surrogate reliability and
evaluation-matched comparison in multivariate response-surface optimization*.
Unpublished manuscript, frozen and not submitted.

Streitenberger, S. C., Romão, E. L., Paiva, A. P., Balestrassi, P. P., Freitas,
J. H. G. and Paes, V. C. (2022). Normal Boundary Intersection with factor analysis
approach for multiobjective stochastic optimization of a cladding process focusing on
reduction of energy consumption and rework. *Journal of Cleaner Production*, 333,
129915. https://doi.org/10.1016/j.jclepro.2021.129915

Vasquez-Ramos, J., Ruiz-Sandoval, M. G., Oliva, D., Ramos-Soto, O., Ramos-Frutos, J.,
Sharawi, M. and Pérez-Cisneros, M. (2025). Response surface-driven hyperparameter
optimization for XGBoost. *The Journal of Supercomputing*, 81(10), Article 1112.
https://doi.org/10.1007/s11227-025-07600-4
