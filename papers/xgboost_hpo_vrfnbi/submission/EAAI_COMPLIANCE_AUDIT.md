# EAAI compliance audit of `paper2-manuscript-v5`

Target: *Engineering Applications of Artificial Intelligence* (Elsevier, ISSN 0952-1976).
Guide for Authors retrieved 2026-09-18. ScienceDirect returns HTTP 403 to automated
fetches; the wording below was recovered by search against `sciencedirect.com` and
`elsevier.com` and each rule was then independently re-retrieved by a second agent.
Rules that could not be quoted are listed as UNRETRIEVED and are **not** asserted.

The manuscript is unchanged. Nothing in this audit has been applied.

---

## 1. The blocking finding: the engineering-application requirement

EAAI's scope gate, quoted:

> "Engineering Applications of Artificial Intelligence provides an international forum
> for rapid publication of work describing the practical application of AI methods in
> all branches of engineering."

> "Submitted papers should report novel aspects of AI used for a **real-world
> engineering application** and also validated using public data sets for easy
> replicability of the research results."

and, as one of four explicit desk-rejection conditions:

> "The abstract should clearly specify which is the contribution in AI, and which is
> the application in engineering."

| Conjunct | Status | Evidence in `paper2-manuscript-v5` |
|---|---|---|
| novel aspects of AI | **met** | The decomposition of scalarization specification, front-construction geometry and anchor provenance is a methodological contribution to surrogate-assisted multiobjective hyperparameter optimization. |
| validated on public data sets | **met** | Four public binary-classification datasets, each under CC BY 4.0 with a published SHA-256. |
| **a real-world engineering application** | **NOT MET** | See below. |

**The study establishes no engineering application, and the manuscript does not claim
one.** Measured against the frozen text:

- The word *engineering* occurs four times. **Every occurrence** is the phrase "an
  engineering quantity", describing the physical-fit and cache-hit bookkeeping that is
  deliberately excluded from every method comparison. It is software accounting, not an
  application domain.
- *industrial*, *process optimization* and *real-world deployment* occur **zero** times.
- *manufacturing* and *machining* occur only inside reference titles (Costa et al. 2016;
  Luz et al. 2021), i.e. in the methodological lineage, never in this study's own work.
- §5.1 selects the four datasets for **statistical** properties, not domain relevance:
  MAGIC Gamma Telescope for continuity with the historical pipeline, Adult / Census
  Income for "scale and mixed types", Bank Marketing for "class imbalance", Spambase as
  "small and entirely numeric". None is presented as an engineering problem.

A submission stating an engineering application would therefore be stating something the
evidence does not support. Supplying one is **new science**, which is out of scope for a
submission pass, and inventing one is out of the question.

**This is a go/no-go decision for the author, not an editorial fix.** It is recorded here
rather than resolved. The options are set out in §5.

---

## 2. Desk-rejection conditions, checked one by one

EAAI states four, plus a length/size cap. "Submissions that do not meet these
requirements will be subject to desk rejection."

| # | Condition (quoted) | Status | Detail |
|---|---|---|---|
| 1 | "The abstract should clearly specify which is the contribution in AI, and which is the application in engineering." | **FAIL** | AI contribution is clear. No engineering application exists to state. See §1. |
| 2 | "The use of undefined acronyms in the title and in the abstract is forbidden." | **FAIL** | Four undefined tokens. See §3. |
| 3 | "The papers must be formatted in single-column format." | **PASS** | The compiled package is single-column throughout. |
| 4 | Metaphor-based metaheuristics are out of scope. | **PASS** | Not applicable: the comparators are NSGA-II, Bayesian optimization, TPE, grid and random search, none framed metaphorically. |
| — | "All submissions must not exceed 50 pages, and the manuscript file size should be under 100 MB." | **PASS, with a format caveat** | 32 pages, 1.4 MB as built. See §4. |

---

## 3. Undefined acronyms in the title and abstract (desk-rejection condition 2)

| Token | Where | Defined at first use? |
|---|---|---|
| `XGBoost` | title **and** abstract | **No.** Expansion "Extreme Gradient Boosting" appears nowhere in either. |
| `CORE` | abstract, "CORE-relative hypervolume" | **No.** CORE is the name of the method-independent reference set, defined only in §5.14. |
| `AUGMENTED` | abstract, "CORE and AUGMENTED references" | **No.** Defined only in §5.14. |
| `NBI-S` | abstract, "favoured a coarse grid over NBI-S" | **No.** *Normal Boundary Intersection* is spelled out earlier in the abstract, but the arm identifier and its `-S` suffix are not defined. |

All four are repairable by expansion at first use, with **no change of meaning and no
change to any number**. The repair is drafted but **not applied**, because condition 1
governs the same two pieces of text and is unresolved.

Also noted: the abstract is **257 words** against a stated limit of **250**. Expanding
four acronyms makes it longer, so the abstract needs a net reduction of roughly 30-40
words. That is achievable by tightening connective prose without touching a claim or a
figure, but it is a rewrite of the abstract and is therefore held pending §1.

---

## 4. Length under Elsevier submission formatting

The 50-page cap is a desk-rejection trigger. The package is 32 pages **as currently
built** (10.5 pt, single-spaced, single column). EAAI's "Your Paper Your Way" allows a
single PDF in any reasonable format for first submission, so 32 pages is compliant today.

The exposure is at revision, when a "correct format" is requested. At 16,800 words:

| Format | Estimated text pages | Plus floats | Total |
|---|---|---|---|
| double-spaced 12 pt | ~67 | ~11 | **~78** |
| 1.5-spaced 12 pt | ~50 | ~11 | **~61** |
| single-spaced 12 pt | ~33 | ~11 | ~44 |
| as built (10.5 pt) | ~28 | ~11 | ~39 |

Whether the 50-page cap counts references, appendices and supplementary material is
**UNRETRIEVED** and should be confirmed before a reformat. If double spacing is required
at revision, the manuscript exceeds the cap by a wide margin.

---

## 5. Options for the scope mismatch — author decision

None of these is applied. They are listed so the decision is explicit.

1. **Submit to EAAI as-is and accept the risk.** Honest, and likely desk-rejected on
   condition 1: there is no engineering application to name in the abstract.
2. **Add an engineering application.** Requires running the frozen protocol on a
   genuine engineering problem. That is new science and is outside a submission pass.
3. **Reframe within what the evidence supports.** The study audits a design-of-experiments
   and response-surface optimization pipeline of the kind used in engineering process
   optimization, and its lineage (Costa et al. 2016, steel turning; Streitenberger et al.
   2022; Pereira et al. 2025 in EAAI itself) is engineering. That is an honest statement
   about the *method's* provenance — but the present study's own object is the pipeline,
   not an engineering artifact, and an editor applying condition 1 may still see no
   application. This reduces the risk; it does not remove it.
4. **Target a journal whose scope matches a methodology and measurement study.** The
   contribution is a replicated, prospectively frozen decomposition of an optimization
   pipeline. Venues for methodology and empirical rigour fit the evidence without any
   reframing.

---

## 6. Remaining compliance items, once §1 is settled

| Item | Requirement | Status |
|---|---|---|
| Double-anonymized review | "submit your title page (including author details) and anonymized manuscript (excluding author details) as separate files" | **Not met.** The package is one PDF containing author details. Needs splitting. |
| Reference style | Author-date, not numbered | **PASS** — the manuscript already uses author-date. |
| Section numbering | "1.1 (then 1.1.1, 1.1.2, ...)" and use it for cross-references | **PASS** — numbering and cross-references were repaired in v3/v4. |
| Declaration of competing interest | Required at submission | **PASS** — present verbatim. |
| CRediT author statement | Required | **PASS** — present, author-confirmed. |
| Funding sources | Required, with role of the sponsor if any | **PASS on the statement**; the *role of the funding source* sentence is not present and is an author decision. |
| Data availability | Must state availability at submission | **PASS** — present. |
| Declaration of generative AI | Owed for generative-AI help in **writing**; explicitly *not* for AI used to analyze data | **AUTHOR DECISION** — only the authors know whether generative AI assisted the writing. The XGBoost machinery is research-process AI and is carved out. |
| Highlights | Optional; if supplied, 3-5 bullets, max 85 characters each including spaces, separate editable file named "Highlights" | Not prepared — depends on the framing decision. |
| Graphical abstract | Optional | Not prepared. |
| Cover letter | Must disclose how the paper differs from any prior conference paper | No prior conference paper is recorded, so the disclosure is not triggered. |
| ORCID | **UNRETRIEVED** whether EAAI mandates it | None exists in the repository for any author. |

---

## 7. What could not be retrieved

Not asserted anywhere above: whether the 50-page cap includes references and
supplementary material; per-article-type length limits; an explicit desk-rejection ground
for out-of-scope or purely methodological work; whether a section literally headed "Data
availability" is mandatory; ORCID requirement; keyword count and placement; abstract
single-paragraph or structured; title length limit; accepted supplementary file formats;
ethics statements for human/animal subjects (not applicable here).
