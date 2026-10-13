# Revision R1 (coauthor review by Matheus Costa Pereira): decisions and constraints

Recorded 2026-10-11 on branch `revision/paper1-matheus-review`, cut from `submission/paper1-eaai` (422098a).
The frozen scientific master `paper-submission-v2` and every other tag are unchanged; no release tag is created
by this revision.

## Author decisions (corresponding author, 2026-10-11)

| ID | Decision | How it is applied |
|---|---|---|
| C1 | Approved | All verified workstream A (research integrity, factual) and B (scientific communication) corrections of the canonical patch plan. |
| C2 | Variant A | Paper 2 only. |
| C3 | Keep EAAI | Six keywords; no engineering application is invented beyond the measured inference-cost setting. |
| C4 | Elsevier LaTeX workflow | The main text uses the `elsarticle` class (single column, author-year). |
| C5 | Defer optional experiments | Optional analyses (gate-threshold sweep, reference-dependence counts, vector-matched NSGA-II, truncation table, re-clip, batch re-scoring) are not run. |
| C6 | Editorial revision with constraints | Editorial patches are applied where they improve clarity without hiding a limitation, removing a negative result or changing meaning. |

## Author-only facts (docket Q1-Q4)

No answer was available on 2026-10-11, so the record-derived fallback texts of the patch plan are used. None of
them asserts an event the record does not show.

| Question | Text used | Still to be confirmed by the authors |
|---|---|---|
| Q1.1 reason for extending R = 10 to R = 30 | Chronology only (Section 4.1, supplementary S1) | The reason, if one can be documented |
| Q1.2 machine conditions of the two batches | Conditions stated as unrecorded; the batch difference is exploratory and unexplained (Limitations) | Machine, power and thermal state, concurrent load, thread variables |
| Q1.3 base-model hyperparameters | Committed on 2026-07-06 before the replicated study, unchanged afterwards (git record) | Any further rationale |
| Q1.4 dataset mirrors | Mirrors were used; no reason is given | The reason, if documented |
| Q2 gate thresholds | No source is cited; a single-dataset Santander pilot had been completed before they were committed | Whether the pilot informed the two values |
| Q3.1 generative AI | Declaration names Anthropic's Claude (Claude Opus 5 through Claude Code, from the commit trailers) and OpenAI's ChatGPT (stated by the corresponding author); research-code assistance stated in the Reproducibility statement as the trailers record it (Claude Fable 5, Claude Fable 5.1, Claude Opus 5) | Each author's attestation; ChatGPT version and whether it was used for this paper |
| Q3.2 CRediT and funding | Unchanged | Confirmation |
| Q3.3 references | Corrections backed by Crossref, DBLP, OpenAlex or the publisher's proceedings record | One author's check of the whole list before upload |

## Journal requirements verified on 2026-10-11 (Engineering Applications of Artificial Intelligence)

The ScienceDirect Guide for Authors returns HTTP 403 to automated retrieval and the newest public archive capture
dates from 2024-02-18. The requirements below come from the search-engine index of the official guide page
(retrieved 2026-10-11) and from the repository's own EAAI audit of 2026-09-18
(`submission/paper2-journal:papers/xgboost_hpo_vrfnbi/submission/EAAI_COMPLIANCE_AUDIT.md`). They must be
re-read on the live page at upload.

- Abstract at most 250 words, stating the purpose, procedures, main findings and conclusions; it must state the AI
  contribution and the engineering application (a desk-rejection condition); undefined acronyms in the title and
  abstract are forbidden (a desk-rejection condition); unavoidable abbreviations are defined at first mention in the
  abstract and in a footnote; no references.
- At most 6 keywords, American spelling, avoiding general and plural terms.
- Highlights optional: 3 to 5 bullet points of at most 85 characters each, in a separate editable file.
- Single-column format (a desk-rejection condition); at most 50 pages; file size under 100 MB.
- Double-anonymized review: a separate title page with the author details, and an anonymized manuscript.
- Generative AI (Elsevier policy): a separate declaration before the references, titled "Declaration of generative
  AI and AI-assisted technologies in the manuscript preparation process"; AI used in the research process is
  described in the methods.

## Patch policy

- A and B patches are applied after their anchors are re-verified at the branch tip; a stale anchor is rejected, not
  force-matched.
- Patches that print numbers from the new read-only scripts are applied only after the script and its output are
  committed.
- Workstream D patches (optional analyses) are deferred under C5.
- The frozen reports, statistical plan, analysis configuration and NSGA-II report are not edited; the NSGA-II
  budget-unit error is recorded in the new file
  `reports/pco213_postwork_benchmark/nsga2/NSGA2_BASELINE_ERRATUM.md`.

## Presentation QA patch (QA2, 2026-10-12)

Requested by the corresponding author after a page-by-page inspection of the review PDFs; presentation and scope
wording only, no analysis, artifact or reported result changes.

- Figure 3 (surrogate validation) is a 2 x 2 grid, so that the panel titles no longer collide; Figure 5 (paired
  differences) uses the short comparison labels B − A, C − B and C − Scal., defined in its caption. The plotted
  points, lines, limits and scales are unchanged; only the automatic tick positions of Figure 3 differ.
- Supplementary Table S4 prints the reliable count 1 where the aggregation had printed the boolean True (an
  object-typed column summed over one recorded value); the source CSV is unchanged and every other count is equal.
- The anchor recommendation of Section 5.3, and its repetition in Section 6.1, are confined to the studied setting:
  one-shot weighting of this ensemble without infill, with affordable real evaluations. The cost-conditional advice
  of Section 6.5 is made only for the cheap regime studied, the expensive regime being stated as untested, and the
  Conclusion notes that the anchor check was affordable here because a real evaluation is cheap.
- The cover letter carries the rebuilt page counts (manuscript 45, anonymized copy 44).

## Coauthor review R2 (2026-10-13)

A second, short review by the same coauthor. Bounded scope: no experiment, rerun, new dataset or change to a frozen
result; two read-only retrospective diagnostics from stored records, both labelled post hoc, each with its definition
committed before its first run.

- Gate and order selection (P1-R2-A). `scripts/pco213_gate_split_sample.py` re-reads the gate on disjoint halves of
  the 100 stored compositions (200 stratified halves, both directions; same-half control at equal size). Extreme
  verdicts are unchanged; intermediate pass counts fall (Porto Seguro ROC-AUC 15.0 to 11.7 of 30, log-loss 11.1 to
  7.2). The Discussion no longer says the gate "is worth its 100 extra evaluations": it now describes diagnostic
  separation of extreme cases conditionally on the selected order, not validated decision utility. New Limitations
  paragraph; supplementary Section S6 and Table S6-split.
- Dependence correction (P1-R2-B). rho = 0.25 is stated as an assumed correction, not an estimated correlation;
  the overlap argument says it may understate the dependence, not that it does. Abstract and Conclusion significance
  statements carry "at the assumed" qualifiers. No rho-sensitivity table was added (author option).
- Anchor contrast (P1-R2-C). "A versus B isolates anchor misplacement" and "B versus C isolates" are replaced by
  the anchor-provenance intervention / joint effect wording; "anchor effect" is defined once as shorthand for that
  joint effect; the injection control "separates" rather than "isolates".
- Endpoints (P1-R2-D). The Abstract now says out-of-fold real-objective revalidation and a confirmation holdout, and
  that NSGA-II returned larger sets. The body already stated both; nothing recomputed.
- Batches (P1-R2-E). One bounded provenance check of the existing records (supplementary Section S11): identical
  `src/` for partitions 1-29, identical recorded environment block, one seed rule, configuration differing only in
  the replication count and two flags that compute nothing; no partition computed twice; the one available
  recomputation (UCI credit partition 0, NBI stage) identical. The difference stays unexplained. "Pinned environment"
  corrected to "recorded environment (no lockfile)".
- Support-cost reference (P1-R2-F). `scripts/pco213_support_reference_convergence.py` computes, from the stored
  reference samples, a held-out-block and a prefix diagnostic for both fronts. The support-cost front met the 5%
  tolerance in 6 to 12 of 30 partitions per dataset against 9 to 27 for the weighted front under the same
  (pessimistic) definition. The step-cost conclusions are qualified in Results, Limitations, Conclusion and Abstract.
  The S8 table now shows the front sizes it had silently dropped (column-name mismatch).
- Reproducibility wording. The stale branch name is removed, the cost-file path corrected, the raw artifacts described
  as not deposited (their out-of-fold files carry Kaggle training labels), and the data-availability statement no
  longer says "all derived artifacts" are available.
- Page counts: manuscript 45, anonymized 45, supplement 44 (24 tables).
