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
(§3.4).

**Relationship to companion work by the same authors.** Three instruments used here
were developed in a companion manuscript by the present authors (Ribeiro, Pereira and
de Paiva, 2026, unpublished, frozen and not submitted): the external surrogate
reliability gate, the evaluation-matched comparator-budget rule, and the
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

**Funding.** _To be completed by the authors._

**Author contributions.** _To be completed by the authors._

---

## References

Entries below are given with the bibliographic detail recorded in the project's
lineage register. Entries marked **[complete before submission]** require full
bibliographic data that is not held in the project record and must be supplied by the
authors; they are cited in the text and must not be dropped.

Das, I. and Dennis, J. E. (1998). Normal-Boundary Intersection: a new method for
generating the Pareto surface in nonlinear multicriteria optimization problems.
*SIAM Journal on Optimization*, 8(3), 631–657.

Pereira, J. H. F., Tertuliano Ribeiro, C., Mendes, M. H. S., Campos, P. H. S. and
de Paiva, A. P. (2025). Hybrid multivariate Normal Boundary Intersection with
mixture-design post-optimization. *Engineering Applications of Artificial
Intelligence*, 162, 112510.

de Azevedo, R., Pereira, J. H. F., Cesário, A. and de Paiva, A. P. (2026). NBI-VRF
applied to computational fluid dynamics. *Thermal Science and Engineering Progress*,
74, 104722.

Ribeiro, C. T. (2026). *Multiobjective optimization of XGBoost hyperparameters using
design of experiments and latent objective construction*. MSc dissertation,
Universidade Federal de Itajubá (UNIFEI). Unpublished.

Ribeiro, C. T., Pereira, J. H. F. and de Paiva, A. P. (2026). *Surrogate reliability
and evaluation-matched comparison in multivariate response-surface optimization*.
Unpublished manuscript, frozen and not submitted.

Lujan-Moreno, G. A., Howard, P. R., Rojas, O. G. and Montgomery, D. C. (2018).
Design of experiments and response surface methodology to tune machine learning
hyperparameters, with a random forest case study. **[complete before submission]**

Vasquez-Ramos, et al. (2025). Response surface methodology for XGBoost hyperparameter
tuning. **[complete before submission]**

Ishibuchi, H., Masuda, H., Tanigaki, Y. and Nojima, Y. (2015). Modified distance
calculation in generational distance and inverted generational distance.
**[complete before submission]**

Nadeau, C. and Bengio, Y. (2003). Inference for the generalization error.
*Machine Learning*, 52(3), 239–281.

Bergstra, J. and Bengio, Y. (2012). Random search for hyper-parameter optimization.
*Journal of Machine Learning Research*, 13, 281–305.

Morales-Hernández, A., Van Nieuwenhuyse, I. and Rojas Gonzalez, S. (2023). A survey on
multi-objective hyperparameter optimization algorithms for machine learning.
**[complete before submission]**

Karl, F., et al. (2023). Multi-objective hyperparameter optimization in machine
learning — an overview. **[complete before submission]**

Costa, et al. (2016); Luz, et al. (2021); Streitenberger, et al. (2022). Prior
applications of the latent-objective construction by the same research group.
**[complete before submission]**

Eggensperger, K., et al. (2021); Pfisterer, F., et al. (2022); Guerrero-Viu, J., et al.
(2021). Benchmark suites and baseline sets for hyperparameter optimization.
**[complete before submission]**
