## 5. Methods

### 5.1 Datasets and their prospectively assigned roles

Four public binary-classification datasets are used, each obtainable under CC BY 4.0 with a
published SHA-256. MAGIC Gamma Telescope is the continuity dataset, the one the historical
pipeline used; Adult / Census Income contributes scale and mixed types; Bank Marketing
contributes class imbalance with independently constructed categorical features; Spambase is
small and entirely numeric.

Roles were assigned by a screening rule frozen before the campaign, applied to the 88 design rows
of each dataset on one partition, which were evaluated before any arm ran; that partition came from a pre-campaign pilot,
is not one of the 30 confirmatory partitions, and its 88 evaluations are also the fitting sample of
the frozen factor model (Section 5.5). Two of its four criteria are structural: a raw-response
conflict between the six quality responses and the raw leaf-count response (criterion 1), and a
non-dominated set of design rows with detectable curvature (criterion 2, the front-geometry screen).
MAGIC, Adult and Bank Marketing met all four and form the *primary geometry panel*. Spambase failed
criterion 2 with a two-point non-dominated set, which the screening rule treated as leaving no
interior geometry for the front-construction contrast to act on. Whether that premise held in
execution is a result, reported in Section 6.6 together with the other three datasets' design-row
fronts on the 30 confirmatory partitions, not an assumption carried forward here. Spambase was
retained as a boundary geometry control by a pre-campaign amendment adopted after this failure was
known (amendment 22): the replacement rule frozen at protocol v1, before the pilot, required a
dataset failing criterion 1 or 2 to be replaced from the registry, and retaining Spambase departs
from that rule. No replacement dataset was selected, and Spambase runs every arm, budget, seed and
all $R = 30$ replications, excluded only from the primary inferential family. These are pre-campaign
screening measurements, not study results (Supplement S5). The dataset is the unit of
generalization; nothing is pooled.

### 5.2 Decision space

Seven XGBoost hyperparameters are optimized, with the bounds of the reconstructed
historical configuration retained unchanged (Table 1).

| Hyperparameter | Low | High | Type |
|---|---:|---:|---|
| `subsample` | 0.05 | 1.00 | continuous |
| `colsample_bytree` | 0.05 | 1.00 | continuous |
| `colsample_bylevel` | 0.05 | 1.00 | continuous |
| `learning_rate` | 0.01 | 0.30 | continuous |
| `max_depth` | 3 | 18 | integer |
| `gamma` | 0.05 | 5.00 | continuous |
| `n_estimators` | 50 | 700 | integer |

**Table 1.** The seven XGBoost hyperparameters and their frozen bounds. All enter
the design in coded units on $[-1,1]$; integers are cast by `int(round(·))` at
evaluation time.

A configuration is written in coded units as $x \in [-1, 1]^7$ and mapped to natural units
by $\text{natural}_i(x) = lo_i + (x_i + 1)(hi_i - lo_i)/2$. `max_depth` and `n_estimators`
are cast by `int(round(·))` at evaluation time, so every surrogate is fitted on a continuous
relaxation whose evaluated points are rounded; the gap is carried rather than repaired, with
each subproblem reporting the objective displacement induced by rounding beside its residual.

### 5.3 The 88-run face-centred central composite design

The design is the historical pipeline's face-centred central composite in the seven factors, 88
runs, version-controlled, reused byte-identically across datasets and replications and checksummed.
It comprises 64 factorial points, 14 axial points and 10 centre points; face-centred means an axial
distance of $\alpha = 1$, so no run leaves the coded box. The factorial points are the resolution
VII half fraction of $2^7$ with generator $G = ABCDEF$ and defining relation $I = +ABCDEFG$, where
$A$ to $G$ are the factors in the order of Table 1 (so $G$ is `n_estimators`): the 64 corners whose
seven coded levels multiply to $+1$. In this fraction each main effect is aliased only with a
six-factor interaction and each two-factor interaction only with a five-factor interaction, so no
two linear or interaction terms of the full quadratic of Section 5.6 are aliased with each other,
and the axial runs make its squared terms separately estimable.

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

### 5.4 Responses, transforms and orientation

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

### 5.5 Latent objective construction

**Extraction and rotation.** PCA of $Z$ retains $k = 3$ components. Loadings are the eigenvector
matrix scaled by the square roots of the eigenvalues, $\Lambda = V
\operatorname{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_2}, \sqrt{\lambda_3})$; this is the matrix
that is rotated and reported, and Varimax gives $\Lambda_R = \Lambda R$. Scores are formed from
*standardized* component scores,

$$S = Z\,V\,\operatorname{diag}(1/\sqrt{\lambda})\,R, \tag{1}$$

then standardized column-wise to $S_z$. The $\operatorname{diag}(1/\sqrt{\lambda})$ term is
load-bearing: raw component scores have variances equal to the eigenvalues, so rotating them
orthogonally mixes axes of unequal scale and yields correlated factors. On this panel the maximum
off-diagonal correlation of the nominally orthogonal factors ranged from 0.379 to 0.698 when raw
scores were rotated, and is below $10^{-15}$ under the algebra above, which also makes $\Lambda_R$
the loading matrix of the scores actually in use, so the role and sign rules below are read off a
matrix that describes what is being oriented (Supplement S4.2).

**Role assignment and sign, both deterministic.** The *cost factor* is the component with the
largest absolute rotated loading on `Leaves_Mean`; the other two are *quality factors*. The cost
factor is oriented so that the rotated loading of `Leaves_Mean` on it is positive, each quality
factor so that the mean rotated loading over the six quality responses is positive: the role-block
mean rather than the single largest loading, because specificity anti-correlates with accuracy,
recall and AUC and dominates the leading quality factor on three of four datasets. After
orientation, a larger score is worse on average over the factor's role block (the six quality
responses, or `Leaves_Mean` for the cost factor), not on every response. Every oriented loading is
reported in Supplement S4.2; each is a correlation on the 88 pilot design rows on which the model is
fitted, where a positive loading marks a response that improves as the score decreases. On those
rows the leading quality factor of Spambase, Adult and Bank Marketing sets canonicalized specificity
($-0.92$, $-0.93$, $-0.89$) against canonicalized recall ($+0.92$, $+0.86$, $+0.87$), and the second
quality factor of MAGIC sets precision and specificity against recall and accuracy. On the same
rows, a lower value of the quality composite defined next goes with higher recall and lower
specificity on Spambase, Adult and Bank Marketing, and with better values of all six quality
responses on MAGIC.

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
(Supplement S4.1). Two constraints force this scope at once. A per-replication refit
makes the objective a different variable in every pair, so 30 paired indicator values would not
live in one objective space. And an earlier specification adding the 78-point complement — 166
points — was withdrawn, because that complement *is* the audit-only external construction (64 of
its 78 coded points identical to the external set, the other 14 the same axial runs up to integer
rounding of `max_depth`), which would let the gate validate a surface against data that helped
define that surface's target; the intersection is verified as zero in coded space on all four
datasets. The per-replication refit is computed and reported as a Tucker-congruence sensitivity,
flips reported rather than corrected, and is never applied.

### 5.6 Response surface models

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

One arm is the exception to this shared procedure: HISTORICAL-WS-asrun (Section 5.7), whose surfaces
this study's code fits in the archived uncoded natural-unit parameterization (Table 2), on this
study's objectives.

Everything specified here is scoped to these four datasets, this surrogate architecture, two
objectives and this budget.
