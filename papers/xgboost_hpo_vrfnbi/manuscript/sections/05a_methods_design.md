## 5.1 Datasets and their prospectively assigned roles

Four public binary-classification datasets are used, each obtainable under CC BY 4.0 with a
published SHA-256. MAGIC Gamma Telescope is the continuity dataset, the one the historical
pipeline used; Adult / Census Income contributes scale and mixed types; Bank Marketing
contributes class imbalance with independently constructed categorical features; Spambase is
small and entirely numeric.

Roles were assigned by a screening rule frozen before the campaign, measured on the 88 design
rows of each dataset on one partition, before any arm ran. Two of its four criteria are
structural: a raw-response conflict between the six quality responses and the raw leaf-count
response, and a non-dominated set of design rows with detectable curvature. MAGIC, Adult and
Bank Marketing met all four and form the **primary geometry panel**. Spambase failed criterion 2
with a two-point non-dominated set, which has no interior, so every scalarization returns the
same two extreme points and no interior geometry remains for the weighted-sum-versus-NBI
contrast to separate. It was therefore **retained prospectively as a boundary geometry
control**: the replacement rule was considered and deliberately not exercised, no replacement
dataset was selected, and Spambase runs every arm, budget, seed and all $R = 30$ replications,
excluded only from the primary inferential family. These are pre-campaign screening
measurements, not study results [TAB:panel_screening]. The dataset is the unit of
generalization; nothing is pooled.

## 5.2 Decision space

Seven XGBoost hyperparameters are optimized, with the bounds of the reconstructed
historical configuration retained unchanged [TAB:hyperparameter_bounds].

| Hyperparameter | Low | High | Type |
|---|---:|---:|---|
| `subsample` | 0.05 | 1.00 | continuous |
| `colsample_bytree` | 0.05 | 1.00 | continuous |
| `colsample_bylevel` | 0.05 | 1.00 | continuous |
| `learning_rate` | 0.01 | 0.30 | continuous |
| `max_depth` | 3 | 18 | integer |
| `gamma` | 0.05 | 5.00 | continuous |
| `n_estimators` | 50 | 700 | integer |

A configuration is written in coded units as $x \in [-1, 1]^7$ and mapped to natural units
by $\text{natural}_i(x) = lo_i + (x_i + 1)(hi_i - lo_i)/2$. `max_depth` and `n_estimators`
are cast by `int(round(·))` at evaluation time, so every surrogate is fitted on a continuous
relaxation whose evaluated points are rounded; the gap is carried rather than repaired, with
each subproblem reporting the objective displacement induced by rounding beside its residual.

## 5.3 The 88-run face-centred central composite design

The design is a version-controlled face-centred central composite in the seven factors, 88 runs,
reused byte-identically across datasets and replications and checksummed. It comprises 64
factorial points (a half fraction of $2^7$), 14 axial points and 10 centre points; face-centred
means an axial distance of $\alpha = 1$, so no run leaves the coded box.

One **real evaluation** is one stratified, shuffled, seeded 5-fold cross-validation of one
configuration on one outer partition — five XGBoost fits. Outer partitions are resampled $R = 30$
times per dataset on an 80/20 split, and every method sees the same split at a given replication,
so comparisons are paired by partition.

A separate 78-point set — the design's complementary half fraction plus 14 axial runs at half
the design's axial distance — is evaluated on the real objectives and is **audit-only**:
disjoint from the 88 design rows by construction, reaching the surrogate reliability gate and
nothing else. The gate passes when external $R^2 \ge 0.5$ and Spearman $\rho \ge 0.9$ per
response, and is **diagnostic, not adaptive** — every arm runs at every replication whatever it
returns, and its pass rate is reported as a covariate, never used as a filter.

## 5.4 Responses, transforms and orientation

Seven responses are recorded per evaluation as the mean over folds. Six carry the `quality`
role — `Accuracy_Mean`, `Precision_Mean`, `Recall_Mean`, `Specificity_Mean`, `RocAuc_Mean`,
`LogLoss_Mean`, the threshold metrics computed at 0.5 and log-loss probabilities clipped to
$[10^{-7}, 1 - 10^{-7}]$ — and one carries the `cost` role, `Leaves_Mean`, the total leaf count
of the fitted ensemble summed over all trees. Mean per-fold wall-clock time, `Time_MeanFold`, is
recorded on every evaluation and is a **secondary audit variable entering no objective**.

Transform and orientation are declared per response, not inferred from data.
`Leaves_Mean` is transformed by $\log 1p$; every other response is untransformed. The transform
is not cosmetic: leaf count spans $591\times$ to $4,356\times$ over the design box, a
quadratic fitted to it raw reaches in-sample $R^2$ of only 0.49 to 0.65 against 0.95
transformed, and untransformed it strips the cost surface to an intercept on two datasets.
Responses are then canonicalized to minimization by multiplying by $-1$ where
higher is better, giving $M$, whose columns are standardized to $Z$ using the mean and sample
standard deviation ($\text{ddof} = 1$) computed **on the design side only**, with $\sigma_j
\leftarrow 1$ wherever $\sigma_j = 0$.

## 5.5 Latent objective construction

**Extraction and rotation.** PCA of $Z$ retains $k = 3$ components. Loadings are the eigenvector
matrix scaled by the square roots of the eigenvalues, $\Lambda = V
\operatorname{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_2}, \sqrt{\lambda_3})$; this is the matrix
that is rotated and reported, and Varimax gives $\Lambda_R = \Lambda R$. Scores are formed from
**standardized** component scores,

$$S = Z\,V\,\operatorname{diag}(1/\sqrt{\lambda})\,R,$$

then standardized column-wise to $S_z$. The $\operatorname{diag}(1/\sqrt{\lambda})$ term is
load-bearing: raw component scores have variances equal to the eigenvalues, so rotating them
orthogonally mixes axes of unequal scale and yields correlated factors. On this panel the
maximum off-diagonal correlation of the nominally orthogonal factors was 0.379, 0.541, 0.698 and
0.678, and is below $10^{-15}$ under the algebra above, which also makes $\Lambda_R$ the loading
matrix of the scores actually in use, so the role and sign rules below are read off a matrix
that describes what is being oriented [FIG:rotated_loadings].

**Role assignment and sign, both deterministic.** The **cost factor** is the component with the
largest absolute rotated loading on `Leaves_Mean`; the other two are **quality factors**. The cost
factor is oriented so that the rotated loading of `Leaves_Mean` on it is positive, each quality
factor so that the **mean rotated loading over the six quality responses** is positive — the
role-block mean rather than the single largest loading, because specificity anti-correlates with
accuracy, recall and AUC and dominates the leading quality factor on three of four datasets.
After orientation, a larger score is a worse configuration on every factor.

**Objective 1, composite quality.** $f_{\text{quality}}(x) = \sum_{j \in Q} w_j S_z[:, j]$
with $w_j = h_j / \sum_{i \in Q} h_i$ and $h_j = \sum_i \Lambda_R[i, j]^2$ — the **rotated**
sums of squared loadings, normalized over the two quality factors. Varimax redistributes variance
across components, so an unrotated eigenvalue $\lambda_j$ and rotated component $j$ are not the
same object and indexing one by the other pairs unrelated quantities. On this panel the rotated
shares are 0.556/0.444, 0.775/0.225, 0.547/0.453 and 0.587/0.413 against unrotated 0.706/0.294,
0.837/0.163, 0.872/0.128 and 0.845/0.155; on Adult the second factor's weight differs by a
factor of 3.5; both shares are persisted under distinct names. Equal weighting — the historical
choice — is the sensitivity declared in advance, and the two agree at Spearman 0.987 to 0.991.

**Objective 2, leaf-count complexity.** $f_{\text{cost}}(x) = S_z[:, c]$, the cost-role factor,
which loads on `Leaves_Mean` at 0.90 to 1.00 on this panel. It is a **deterministic
model-complexity proxy**, never described as training time: it tracks measured per-fold time at
Spearman $\approx 0.86$, an association and not an identity. Both objectives are minimized.

**$k = 3$ is fixed at protocol level, not criterion-selected.** It is inherited from the
historical construction so that this study decomposes that construction rather than re-choosing
its dimensionality. The Kaiser criterion retains 2, 2, 1 and 1 components on MAGIC, Spambase,
Adult and Bank Marketing, with $\lambda_3/\lambda_4$ of 2.93, 2.10, 1.34 and 1.06 — never three.
Eigenvalues and Kaiser counts are reported as diagnostics; the component count is never chosen by
which value optimizes better.

**One frozen model per dataset, fitted on the 88 design rows only.** The tuple $(\mu, \sigma, V,
R, \text{roles}, \text{signs}, \mu_S, \sigma_S, w)$ is fitted once per dataset on the 88 design
rows and **nothing else**, then applied unchanged to the 78 audit-only rows, every arm's
revalidated candidates, every direct baseline, the reference sets and the holdout confirmation
[TAB:factor_model_provenance]. Two constraints force this scope at once. A per-replication refit
makes the objective a different variable in every pair, so 30 paired indicator values would not
live in one objective space. And an earlier specification adding the 78-point complement — 166
points — was withdrawn, because that complement *is* the audit-only external construction (64 of
its 78 coded points identical to the external set, the other 14 the same axial runs up to integer
rounding of `max_depth`), which would let the gate validate a surface against data that helped
define that surface's target; the intersection is verified as zero in coded space on all four
datasets. The per-replication refit is computed and reported as a Tucker-congruence sensitivity,
flips reported rather than corrected, and is never applied.

## 5.6 Response surface models

Each objective is modelled by a full quadratic response surface in **coded** units, fitted by
ordinary least squares on the 88 design rows only, with backward elimination at $\alpha = 0.05$
and hierarchy enforced. The surfaces are fitted to the objective of §5.5 — the factor score, not
a raw response — so the $\log 1p$ of §5.4 sits inside the objective and the cost surface is fitted
on the transformed scale by construction. The same procedure, elimination rule and coded
parameterization are used identically by every arm that consumes a surrogate, which is what
allows the arm contrasts of §5.7 to vary one mechanism at a time. Model order is fixed by this
rule and never revised after the fact. One consequence is carried rather than smoothed over: a
backward-eliminated quadratic is often minimized on a box corner, so two objectives' surrogate
anchors can coincide and leave the payoff matrix rank-deficient.

The archived historical implementation fits its surfaces in uncoded natural units and constructs
its front by normalized weighted scalarization, although the historical text uses NBI
terminology; it is reproduced unmodified in the as-run historical arm rather than repaired.

Everything specified here is scoped to these four datasets, this surrogate architecture, two
objectives and this budget.
