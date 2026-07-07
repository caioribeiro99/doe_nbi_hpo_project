"""mixens — Mixture-design ensemble weighting study (PCO213 final project).

Self-contained package: treats M base classifiers as mixture components,
generates weight designs on the simplex, evaluates them on out-of-fold
probability matrices, fits Scheffé canonical polynomials as interpretable
metamodels of ensemble performance, and optimizes/validates the weights.

Parts of this package are ported, with attribution, from the dissertation
repository branch ``repo-publication-readiness`` (commit 0465466) of
``doe_nbi_hpo_project`` (Caio Tertuliano Ribeiro, MIT License). The
dissertation package ``doe_xgb`` is NOT imported and NOT modified.
"""

__version__ = "0.1.0"
