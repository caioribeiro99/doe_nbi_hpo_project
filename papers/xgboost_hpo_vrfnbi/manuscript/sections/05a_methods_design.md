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

Seven XGBoost hyperparameters are optimized, with the bounds of the archived historical
configuration retained unchanged (Table 2). Every other learner setting is fixed and identical for
every method: `XGBClassifier` with the binary logistic objective, `tree_method="hist"`, one thread
per fit and no early stopping, so every fit runs all `n_estimators` boosting rounds; there is no
class reweighting, the training seed is the replication's seed (Section 5.13), the remaining
parameters that affect training keep their library defaults, and categorical features are one-hot
encoded.

| Hyperparameter | Low | High | Type |
|---|---:|---:|---|
| `subsample` | 0.05 | 1.00 | continuous |
| `colsample_bytree` | 0.05 | 1.00 | continuous |
| `colsample_bylevel` | 0.05 | 1.00 | continuous |
| `learning_rate` | 0.01 | 0.30 | continuous |
| `max_depth` | 3 | 18 | integer |
| `gamma` | 0.05 | 5.00 | continuous |
| `n_estimators` | 50 | 700 | integer |

**Table 2.** The seven XGBoost hyperparameters and their frozen bounds. All enter
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
$A$ to $G$ are the factors in the order of Table 2 (so $G$ is `n_estimators`): the 64 corners whose
seven coded levels multiply to $+1$. In this fraction each main effect is aliased only with a
six-factor interaction and each two-factor interaction only with a five-factor interaction, so no
two linear or interaction terms of the full quadratic of Section 5.6 are aliased with each other,
and the axial runs make its squared terms separately estimable.

One *real evaluation* is one stratified, shuffled, seeded 5-fold cross-validation of one
configuration on the 80% training part of one outer partition, that is, five XGBoost fits; the only
exception is the holdout confirmation of Section 5.13, a single fit on that whole training part
scored on the held-out 20%. Outer partitions are resampled $R = 30$ times per dataset on an 80/20
split, and every method sees the same split at a given replication, so comparisons are paired by
partition.

A separate 78-point set, made of the design's complementary half fraction ($I = -ABCDEFG$) and 14
axial runs at half the design's axial distance, is evaluated on the real objectives and is
*audit-only*: disjoint from the 88 design rows by construction, it reaches the surrogate reliability
gate and nothing else. The gate passes when external $R^2 \ge 0.5$ and Spearman $\rho \ge 0.9$ per
objective. Both cut-offs are fixed at protocol level, not calibrated for this problem: they were
carried over unchanged from the companion manuscript in which the gate was first used (Declarations),
frozen before the measurement-validation pilot (Section 9.1) and never moved. The audit set, by
contrast, was adapted to this problem, because in that pilot uniformly sampled points carried too
little response spread for external $R^2$ to be informative. The gate is diagnostic, not adaptive:
every arm runs at every replication whatever it returns, and its pass rate is reported as a
covariate (Supplement S11), never used as a filter.

### 5.4 Responses, transforms and orientation

Seven responses are recorded per evaluation as the mean over folds. Six carry the `quality` role:
`Accuracy_Mean`, `Precision_Mean`, `Recall_Mean`, `Specificity_Mean`, `RocAuc_Mean` and
`LogLoss_Mean`. The first four are computed at a fixed decision threshold of 0.5, and log-loss
probabilities are clipped to $[10^{-7}, 1 - 10^{-7}]$. One response carries the `cost` role:
`Leaves_Mean`, the total leaf count of the fitted ensemble summed over all trees. Mean per-fold
wall-clock time, `Time_MeanFold`, is recorded on every evaluation and is a secondary audit variable
entering no objective. The learner is fitted without class weighting (Section 5.2), and precision is
set to 0 in a fold with no predicted positive. The positive class is 35.2% of MAGIC, 39.4% of
Spambase, 24.9% of Adult and 12.7% of Bank Marketing. Among the 88 design rows on which the factor
model of Section 5.5 is fitted, 13 on Adult, 16 on Bank Marketing, one on Spambase and none on MAGIC
predict no positive case at 0.5, all at the design's lowest learning rate (0.01) and tree count
(50). Their recall is 0, their precision 0 by that convention, their specificity 1 and their
accuracy the majority-class rate, although they still rank cases well above chance (ROC AUC 0.77 to
0.94 on Adult and Bank Marketing, 0.95 on the Spambase row; Section 8.5).

The six quality responses measure different properties. The four threshold metrics, kept for
continuity with the archived pipeline, describe one operating point at 0.5: accuracy is, fold by
fold, the prevalence-weighted mean of recall and specificity, so it adds no dimension of its own,
and on the 88 design rows recall and specificity are negatively rank-correlated on every dataset.
ROC AUC is threshold-free and measures ranking; log loss scores the predicted probabilities and so
reflects calibration as well as discrimination. The responses receive no hand-set weights: the
factor stage of Section 5.5 combines them, together with leaf count, through loadings fitted once on
the 88 design rows, so what the resulting quality composite tracks differs by dataset (Supplement
S4.2).

Transform and orientation are declared per response, not inferred from data. `Leaves_Mean` is
transformed by $\log(1 + \cdot)$; every other response is untransformed. The transform is not
cosmetic: leaf count varies by a factor of 591 to 4,356 over the design box, a quadratic fitted to
it raw reaches in-sample $R^2$ of only 0.49 to 0.65 against 0.95 transformed, and untransformed it
strips the cost surface to an intercept on two datasets. Responses are then canonicalized to
minimization by multiplying by $-1$ where higher is better, giving $M$, whose columns are
standardized to $Z$ using the mean and sample standard deviation (with the $n - 1$ divisor,
$n = 88$) computed on the design side only (the pilot design rows of Section 5.5), with
$\sigma_j \leftarrow 1$ wherever $\sigma_j = 0$.

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

**Objective 1, composite quality.** $f_{\text{quality}}(x) = \sum_{j \in Q} w_j\, s_j(x)$, where $Q$
is the index set of the two quality factors and $s_j(x)$ is the score on factor $j$ obtained by
applying the frozen mapping that produces $S_z$ to the seven responses measured at configuration $x$
(column $j$ of $S_z$ holds $s_j$ at the 88 design rows), with $w_j = h_j / \sum_{l \in Q} h_l$ and
$h_j = \sum_{i=1}^{7} (\Lambda_R)_{ij}^2$: the *rotated* sums of squared loadings, normalized over
the two quality factors. Varimax redistributes variance across components, so an unrotated
eigenvalue $\lambda_j$ and rotated component $j$ are not the same object and indexing one by the
other pairs unrelated quantities. On this panel the rotated shares are 0.556/0.444, 0.775/0.225,
0.547/0.453 and 0.587/0.413 against unrotated 0.706/0.294, 0.837/0.163, 0.872/0.128 and 0.845/0.155;
on Adult the second factor's weight differs by a factor of 3.5; both shares are persisted under
distinct names. Equal weighting, the historical choice, is the sensitivity declared in advance. The
two composites rank the 88 design rows at Spearman 0.987 to 0.991; that is rank agreement on the
fitting sample, not the declared estimand. The declared comparison of returned fronts, with its 20%
rule, was computed after the campaign from stored outputs by rescoring each arm's revalidated
candidates with the equal-weight composite (a post hoc operationalization; no arm was re-solved),
and is reported in Supplement S4.5.

**Objective 2, leaf-count complexity.** $f_{\text{cost}}(x) = s_c(x)$, where $c$ is the index of the
cost-role factor, whose rotated loading on `Leaves_Mean` is 0.89 to 0.98 on this panel: a latent
factor dominated by log leaf count. The `cost` role of Section 5.4 is named after the archived
pipeline's cost response, wall-clock time (Section 3.5); here the role designates model complexity,
measured as total leaf count. This objective is a deterministic model-complexity proxy, never
described as training time: before the campaign, total leaf count tracked measured per-fold time at
Spearman $\approx 0.86$ on 22 of the 88 MAGIC design configurations, an association and not an
identity, and measured time itself differed by a median 1.3% between two fits of one configuration.
The association over all 120 campaign units and the comparison of winning arms under the two cost
definitions, both declared in the frozen protocol, were computed after the campaign from stored
outputs and are reported in Supplement S4.6 (association) and S4.7 (winning arms). Both objectives
are minimized.

**What the two objectives contain.** Each factor score is computed from all of $Z$, so both
objectives are linear scores of all seven standardized responses (Supplement S4.2). The coefficient
of $f_{\text{quality}}$ on the standardized, log-transformed leaf count is $+0.01$ on MAGIC and
$+0.17$ to $+0.19$ on Spambase, Adult and Bank Marketing, so at equal quality responses the
composite prefers fewer leaves. On the 88-row fitting sample the two objectives are linearly
uncorrelated by construction (absolute Pearson correlation below $10^{-15}$), because the quality
factors and the cost factor are orthogonal there. Their rank correlation on that sample, $-0.20$,
$+0.32$, $+0.10$ and $-0.02$ on MAGIC, Spambase, Adult and Bank Marketing, is a nonlinearity
residual whose sign differs across datasets, not a measure of objective conflict; the objective
conflict the study reports is the raw one between the six quality responses and leaf count
(Supplement S5). How closely the composite tracks raw quality, on and beyond its fitting sample, is
reported in Section 8.5 and Supplement S4.3.

**$k = 3$ is fixed at protocol level, not criterion-selected.** It is inherited from the historical
construction, which forced at least three components so that its cost response, wall-clock time, had
a factor of its own beside the quality factors; keeping it lets this study decompose that
construction rather than re-choose its dimensionality. The Kaiser criterion retains 2, 2, 1 and 1
components on MAGIC, Spambase, Adult and Bank Marketing, with $\lambda_3/\lambda_4$ of 2.93, 2.10,
1.34 and 1.06, never three. Adopting those counts would leave a single component to serve both
objectives on Adult and Bank Marketing and would drop MAGIC's third component, which holds most of
the leaf count's variance. Three components carry 87.7% to 93.4% of the total variance of $Z$, and
two carry 78.8% to 80.7%. On Bank Marketing the third and fourth eigenvalues nearly tie, so the
third retained direction is poorly determined. The first four eigenvalues, cumulative variance and
Kaiser counts are reported in Supplement S4.1 as descriptive diagnostics; the component count is
chosen neither by them nor by which value optimizes better.

**One frozen model per dataset, fitted on the 88 design rows only.** The tuple
$(\mu, \sigma, V, \lambda, R, \text{roles}, \text{signs}, \mu_S, \sigma_S, w)$ is fitted once per
dataset on the 88 design rows as evaluated in the pre-campaign pilot, and nothing else: one
cross-validated evaluation of each run, as in Section 5.3, on the 80% training part of a single
stratified 80/20 partition of the dataset, the partition on which the screening of Section 5.1 was
measured. That partition is not one of the 30 confirmatory partitions, so no replication's
evaluations enter the fit; the model's standardization and factor orthogonality are exact on these
pilot rows only. Every replication re-evaluates the same 88 configurations on its own partition, and
the model is applied unchanged to those rows, on which the surfaces of Section 5.6 are fitted, and
to the 78 audit-only rows, every arm's revalidated candidates, every direct baseline, the reference
sets and the holdout confirmation (Supplement S4.1). Two constraints force this scope at once. A
per-replication refit makes the objective a different variable in every pair, so 30 paired indicator
values would not live in one objective space; for the same reason the four datasets, each with its
own fitted model, do not share an objective space, and every contrast is computed within a dataset
(Section 5.15). And an earlier specification that added the 78-point complement, for 166 fitting
points in all, was withdrawn, because that complement *is* the audit-only external construction (64
of its 78 coded points identical to the external set, the other 14 the same axial runs up to integer
rounding of `max_depth`), which would let the gate validate a surface against data that helped
define that surface's target; the intersection is verified as zero in coded space on all four
datasets. The per-replication refit, the model a replication's own design rows would have produced,
is computed only to measure its Tucker congruence with the pilot-fitted model and is never applied.
Its congruences, declared in the frozen protocol and computed after the campaign from stored
outputs, are reported in Supplement S4.4, both under the runner's greedy factor matching, which does
not resolve factor sign, and under a post hoc optimal-assignment matching.

### 5.6 Response surface models

Each objective is modelled by a full quadratic response surface in *coded* units, fitted by ordinary
least squares on the 88 design rows only, with backward elimination at $\alpha = 0.05$ and hierarchy
enforced. Evaluation is deterministic within a replication, so the 10 centre points return identical
responses and provide no estimate of pure error; the elimination's $t$-tests nonetheless count all
88 rows, 79 of them distinct, with $88 - p$ residual degrees of freedom for a $p$-term model. The
surfaces are fitted to the objective of Section 5.5, the factor score rather than a raw response, so
the $\log(1 + \cdot)$ transform of Section 5.4 sits inside the objective and the cost surface is
fitted on the transformed scale by construction. The same procedure, elimination rule and coded
parameterization are used identically by every arm that consumes a surrogate, which is what allows
the arm contrasts of Sections 5.8–5.11 to vary one mechanism at a time. Model order is fixed by this
rule and never revised after the fact. Two of its consequences are carried rather than smoothed
over. First, the rule is re-applied in every replication, so the retained terms vary across
replications. Second, a hyperparameter eliminated from a surface leaves that surface flat along it,
so that objective's surrogate anchor (Section 5.10) takes its value of that hyperparameter from the
optimizer's starting point rather than from either objective; the payoff matrix, pseudo-nadir and
CHIM (Sections 5.9–5.10) inherit that value, which in general does not minimize the other objective.
The external gate (Section 5.3) measures neither consequence.

One arm is the exception to this shared procedure: HISTORICAL-WS-asrun (Section 5.7), whose surfaces
this study's code fits in the archived uncoded natural-unit parameterization (Table 1), on this
study's objectives.

Everything specified here is scoped to these four datasets, this surrogate architecture, two
objectives and this budget.
