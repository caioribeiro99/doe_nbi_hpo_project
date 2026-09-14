# The two confirmatory objectives, defined exactly

Frozen for the confirmatory campaign. Nothing here changes after any arm-level result is observed.

Neither objective is referred to in the manuscript as bare "quality" or "cost" without the
definition below being in scope. Both are **latent factor scores**, not raw metrics, and the
complexity objective is **not** wall-clock training time.

---

## 0. Notation

For one dataset and one replication, the design is the version-controlled 88-run face-centred central
composite in the seven XGBoost hyperparameters. Write a configuration in **coded** units as
`x ∈ [−1, 1]⁷`, mapped to natural units by

    natural_i(x) = lo_i + (x_i + 1)(hi_i − lo_i) / 2

with the bounds of `protocol/original_thesis_protocol.md` §2. `max_depth` and `n_estimators` are
integers; every other factor is continuous.

## 1. The seven measured responses

One **real evaluation** is one stratified 5-fold cross-validation of one configuration on one outer
partition, that is five XGBoost fits. Each evaluation records, as the mean over folds:

| # | Response | Definition | Direction |
|---|---|---|---|
| 1 | `Accuracy_Mean` | accuracy at decision threshold 0.5 | higher is better |
| 2 | `Precision_Mean` | precision at 0.5 | higher is better |
| 3 | `Recall_Mean` | recall at 0.5 | higher is better |
| 4 | `Specificity_Mean` | true negatives / (true negatives + false positives) at 0.5 | higher is better |
| 5 | `RocAuc_Mean` | area under the receiver operating characteristic curve | higher is better |
| 6 | `LogLoss_Mean` | logarithmic loss, probabilities clipped to [1e−7, 1−1e−7] | lower is better |
| 7 | `Leaves_Mean` | total leaf count of the fitted ensemble, summed over all trees | lower is better |

An eighth quantity, `Time_MeanFold`, the mean per-fold wall-clock fit-and-predict time, is recorded
on **every** evaluation. **It is a secondary audit variable and enters no objective.**

## 2. Transform and orientation, declared per response

Applied in this order, and neither step is inferred from the data:

**2.1 Transform.** `Leaves_Mean` is transformed by `log1p`. Every other response is untransformed.
The transform is not cosmetic: leaf count spans 591× to 4,273× over the design box, and a quadratic
surface fitted to it raw reaches in-sample R² of only 0.49 to 0.65 against 0.95 transformed.
Untransformed, backward elimination stripped the cost surface to an intercept on two datasets.

**2.2 Orientation.** Every response is canonicalized to **minimization** by multiplying by −1 where
"higher is better". Write the resulting matrix `M`, with columns in the order of §1.

    M[:, j] = s_j · g_j(response_j),    s_j = +1 if the response is minimized, −1 otherwise
    g_7 = log1p,  g_1..g_6 = identity

## 3. Standardization

Column-wise, with the mean and the sample standard deviation (`ddof = 1`) **computed on the design
side only**:

    Z = (M − μ) / σ,    σ_j ← 1 wherever σ_j = 0

`μ` and `σ` are part of the frozen transformation of §7.2 and are persisted per dataset.

## 4. Principal components and Varimax rotation

**4.1 Extraction.** Principal component analysis of `Z`, retaining `k = 3` components. `k` is a
**fixed, protocol-level latent representation** and is discussed in §8.

**4.2 Loadings.** The loading matrix is the eigenvector matrix scaled by the square roots of the
eigenvalues:

    Λ = V · diag(√λ₁, √λ₂, √λ₃)

This is the matrix that is rotated and the matrix that is reported. Reporting the unscaled
eigenvector matrix as "loadings" is the defect the PCA/Varimax audit found in the dissertation code.

**4.3 Rotation.** Varimax on `Λ`, giving rotated loadings `Λ_R = Λ · R` and scores `S = Z · V · R`,
then standardized column-wise to `S_z`.

**4.4 Role assignment, deterministic.** The **cost factor** is the component with the largest
absolute rotated loading on `Leaves_Mean`. This uses an absolute value and is therefore independent
of orientation. The other two components are the **quality factors**.

**4.5 Sign orientation, deterministic.** A principal component's sign is arbitrary and Varimax does
not fix it, so an explicit rule is required or the objective's sign is arbitrary.

- The **cost factor** is oriented so that the rotated loading of `Leaves_Mean` on it is positive.
- Each **quality factor** is oriented so that the **mean rotated loading over the six quality
  responses** is positive.

Orientation is by the role-block mean, **not** by the single largest loading. Specificity
anti-correlates with accuracy, recall and the area under the curve across the decision threshold and
dominates the leading quality factor on three of four datasets; the largest-loading rule therefore
pointed the composite at specificity-badness and reported the objectives as agreeing on datasets
whose raw metrics plainly conflict.

After orientation, **a larger score is a worse configuration on every factor**.

## 5. Objective 1 — the quality composite

    f_quality(x) = Σ_{j ∈ Q} w_j · S_z[:, j],      w_j = λ_j / Σ_{i ∈ Q} λ_i

where `Q` is the two quality factor indices and `λ` their eigenvalues, so the weights are each
component's **share of explained variance among the quality factors**, summing to one.

Minimized. The alternative, an unweighted mean of the same z-scored scores, is the dissertation's
choice and is the sensitivity analysis declared in advance in §11.1. Under this factor stage the two
weightings agree at Spearman 0.865 to 0.964 across the panel.

## 6. Objective 2 — the leaf-count complexity objective

    f_cost(x) = S_z[:, c],      c = the cost factor index of §4.4

Minimized. On the panel this factor loads on `Leaves_Mean` at 0.90 to 1.00 and is effectively the
transformed leaf count expressed on the shared latent scale.

**Terminology, binding.** This is the **leaf-count complexity objective**, a *deterministic
model-complexity proxy*. It is **never** called training time, wall-clock cost, computational time or
computational cost without qualification. It tracks measured training time at Spearman ≈ 0.86, which
is a strong association and is not identity. Enforced by `scripts/check_claim_blacklist.py`.

## 7. From a real XGBoost evaluation to objective space

The mapping is **fit once on the design side and applied everywhere else**, so that every point a
replication scores lives in one objective space.

**7.1 Fitted on the design side only**, in this order, from the 88 evaluated design rows:

    raw responses → transforms (§2.1) → orientation (§2.2) → standardization (§3, μ and σ)
    → PCA fit (§4.1) → fixed k = 3 → Varimax fit (§4.3, R) → role assignment (§4.4)
    → sign orientation (§4.5) → factor-score standardization (μ_S, σ_S) → weights (§5)

The tuple `(μ, σ, V, R, roles, signs, μ_S, σ_S, w)` is then **frozen** and persisted.

**7.2 Applied, never refitted**, to: the 78 external validation runs, every arm's revalidated
candidates, every direct baseline's evaluations, the reference sets, and the holdout confirmation.
PCA and Varimax are never refitted outside the design side. A test asserts that adding validation
rows cannot alter the design-side mapping.

**7.3 So, for any real evaluation** producing a response vector `r`:

    f(r) = ( w ⊙ standardize(pca_rotate(standardize(orient(transform(r))))) )

evaluated with the frozen parameters, giving the pair `(f_quality, f_cost)`, both minimized.

**7.4 Surrogate fitting.** The response surface is a full quadratic in **coded** units, fitted per
objective by ordinary least squares with backward elimination at α = 0.05 and hierarchy enforced,
**on the 88 design rows only**. The surrogate is fitted to the objective as defined above, that is to
the factor score, not to a raw response. The `log1p` of §2.1 is inside the objective, so the cost
surrogate is fitted on the transformed scale by construction.

## 8. On `k = 3`

`k = 3` is a **fixed, protocol-level latent representation**, chosen for provenance and comparability
with the reconstructed thesis framework. It is **not** a per-dataset, Kaiser-selected dimensionality,
and describing it as criterion-supported would not be defensible.

Measured on the panel, the Kaiser criterion (eigenvalue > 1 of the correlation matrix of the seven
canonicalized responses) retains:

| Dataset | eigenvalues (first four) | Kaiser retains | λ₃/λ₄ |
|---|---|---:|---:|
| MAGIC | 3.988, 1.659, 0.891, 0.304 | 2 | 2.93 |
| Spambase | 4.455, 1.092, 0.865, 0.412 | 2 | 2.10 |
| Adult | 4.775, 0.851, 0.701, 0.522 | 1 | 1.34 |
| Bank Marketing | 4.664, 0.853, 0.625, 0.588 | 1 | **1.06** |

Never three, on any dataset. On Bank Marketing the third and fourth components differ by 6%, so
which subspace is retained there is effectively arbitrary.

**The lineage reason for fixing k.** The dissertation's frozen `run_factor_analysis` sets
`n_factors = 3` with `force_time_factor = True`, which raises the count to at least three so that a
dedicated time factor exists; the EAAI 2025 formulation this work decomposes likewise works with a
fixed factor architecture rather than a per-dataset criterion. Paper 2 exists to decompose that
construction, so it inherits its dimensionality rather than re-choosing it.

**Consequences, fixed before launch.** `k` does **not** become data-adaptive during the campaign and
does **not** vary by dataset. Eigenvalues, variance explained and the Kaiser-retained count are
reported per dataset per replication as **descriptive diagnostics**. Component count is never chosen
by which value gives the better optimization result.

## 9. What is not an objective

`Time_MeanFold` is persisted on every evaluation and is a secondary audit variable. The campaign
reports the association between the leaf-count objective and measured time over all 120
dataset-by-replication units, which is the first measurement of that association at scale. It is
never optimized, in either direction, before or after results are seen.
