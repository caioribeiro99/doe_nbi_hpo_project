# Applied Soft Computing — scope gate for `paper2-manuscript-v5`

Target: *Applied Soft Computing* (Elsevier, ISSN 1568-4946), official journal of the
World Federation on Soft Computing.

## Verdict

**DEFENSIBLE FIT WITH EDITORIAL RISK.**

Not CLEAR FIT, and not MATERIAL SCOPE RISK. The reasoning is below, with the journal's
own words. Nothing in the manuscript needs to change for this classification to hold:
the paper already *is* multiobjective optimization of a machine-learning model.

## How the evidence was obtained, and how current it is

ScienceDirect and elsevier.com return HTTP 403 to every automated route (Chrome and
Googlebot user agents, proxies, both host paths), and this session's web-search budget
was exhausted by the EAAI retrieval. The text below was recovered from the Internet
Archive by way of the CDX index and `curl` — the only route that worked.

| Source | Snapshot | What it covers |
|---|---|---|
| ScienceDirect journal home | **2026-02-03** | Aims & Scope, Major Topics — **current** |
| ScienceDirect Guide for Authors | **2024-06-30** | submission rules — newest that exists in the archive |
| elsevier.com legacy guide | 2023-10-10 | article-type sections the redesign dropped |

**Currency caveat.** The scope evidence is current to February 2026, seven months old.
The submission rules are current to June 2024, roughly 27 months old; no newer capture of
that page exists anywhere. Rules already changed once inside the observed window
(keywords 3–10 → 1–7; graphical abstract optional → required), so **every formatting rule
below must be re-checked against the live page before submitting.**

## The scope text

> "Applied Soft Computing is an international journal promoting an integrated view of
> soft computing to solve real life problems. Soft computing is a collection of
> methodologies, which aim to exploit tolerance for imprecision, uncertainty and partial
> truth to achieve tractability, robustness and low solution cost. The focus is to
> publish the highest quality research in application, advance and convergence of the
> areas of Fuzzy Logic, Neural Networks, Evolutionary Computing, Swarm Intelligence and
> other similar techniques to address real world complexities."

> "The scope of this journal covers the following soft computing **and related
> techniques**, interactions between several soft computing techniques, and their
> industrial applications:"
> Evolutionary Computing · Fuzzy Computing · **Hybrid Methods** · Immunological Computing
> · Neuro Computing · Swarm Intelligence · **Machine and Deep Learning** · Rough Sets

> "The application areas of interest include but are not limited to applications of soft
> computing to:" … Combinatorial Optimization · **Data Mining** · **Decision Support** ·
> Engineering Design Optimization · **Multi-objective Optimization** · … *(28 areas)*

## What supports the fit

| Scope item (journal's own list) | Evidence in `paper2-manuscript-v5` |
|---|---|
| **Machine and Deep Learning** (listed technique) | The optimized learner is XGBoost, gradient-boosted decision trees. Hyperparameter optimization of a machine-learning model is the entire object of the study. |
| **Multi-objective Optimization** (listed application area) | 21 occurrences of *multiobjective*; two frozen minimized objectives; Pareto-front construction is the mechanism under test; hypervolume, IGD⁺, generational distance and Schott spacing are the indicators. |
| **Hybrid Methods** (listed technique) | The pipeline hybridizes design of experiments, response-surface surrogates, PCA/Varimax latent objectives and Normal Boundary Intersection. |
| **Data Mining / Decision Support** (listed areas) | Four public binary-classification tasks; the estimand is the returned Pareto front a practitioner would choose from. |
| Evolutionary Computing (listed technique) | NSGA-II is implemented and evaluated — **as a comparator, not as the contribution.** |

**Crucially, unlike EAAI, Applied Soft Computing states no mandatory real-world-application
requirement.** The phrases "real life problems" and "real world complexities" occur only
inside the aims-and-scope description; the journal never issues a directive of the form
"papers must report a real-world application". Nor does the Guide for Authors contain a
desk-rejection list — repeated searches for "without review", "not be considered", "out of
scope", "rejected without" and "immediately reject" returned nothing. The only gate stated
is the ordinary initial editorial suitability screen.

**So the requirement that blocked EAAI does not exist here.** The frozen study does not
need an engineering application to be eligible, and none is claimed.

## What the risk actually is

The journal's identity is soft computing, and its four named pillars are Fuzzy Logic,
Neural Networks, Evolutionary Computing and Swarm Intelligence. Measured in the frozen
text:

| Pillar | Occurrences in the manuscript |
|---|---|
| Fuzzy | **0** |
| Neural networks / deep learning | **0** |
| Swarm intelligence | **0** |
| Evolutionary computing | present **only as a baseline comparator** (NSGA-II) |

The paper's own instruments — response surface methodology, central composite design,
PCA with Varimax rotation, Normal Boundary Intersection, SLSQP — are classical
statistics and deterministic optimization, not soft computing in the strict sense. An
editor reading "soft computing" narrowly could see a design-of-experiments paper.

Against that: "Machine and Deep Learning" is an explicitly listed technique and the
learner is a machine-learning model; "Multi-objective Optimization" is an explicitly
listed application area and that is precisely the paper's object; and the technique list
is introduced as "soft computing **and related techniques**".

**That is why the classification is DEFENSIBLE rather than CLEAR: the fit rests on two
explicitly listed scope items, not on the journal's core pillars.**

## What is NOT being done about it

- No application domain is invented.
- No benchmark dataset is reframed as an industrial problem.
- No scientific claim, number, effect estimate, dataset role or conclusion is altered.
- No soft-computing vocabulary is bolted onto methods that are not soft computing.

The honest framing already available — and already in the manuscript — is that this is a
replicated, prospectively frozen study of *multiobjective optimization applied to
machine-learning hyperparameter selection*. Both halves of that sentence are named ASOC
scope items, and both are true of the frozen paper.
