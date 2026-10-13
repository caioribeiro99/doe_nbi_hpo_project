# Revision R1 (coauthor review by Matheus Costa Pereira): decisions and constraints

Recorded 2026-10-11 on branch `revision/paper2-matheus-review`, cut from `submission/paper2-asoc` (d40eb0b).
The frozen scientific master `paper2-manuscript-v5`, the protocol tags and the results tag are unchanged; no
release tag is created by this revision. The package tag named by the generators, `paper2-manuscript-v6`, is a
forward reference by construction (see `scripts/build_supplement.py`); it is created only after author sign-off.

## Author decisions (corresponding author, 2026-10-11)

| ID | Decision | How it is applied |
|---|---|---|
| C1 | Approved | All verified workstream A and B corrections of the canonical patch plan. |
| C2 | Variant A | The protocol-declared analyses that the Methods describe as reported are computed after the campaign from stored outputs by committed read-only scripts and reported, favourable or not, with their chronology (declared in the frozen protocol, computed after the campaign; any choice the protocol did not fix is labelled post hoc). Procedures that would need new runs (fixed-integer re-solve, true-nadir sensitivity, restricted-*t* run) are disclosed as not executed. |
| C3 | Keep ASOC | No hybrid soft-computing novelty claim is introduced. |
| C4 | Elsevier LaTeX workflow | The main text is provided as an `elsarticle` source; figures as separate vector files. |
| C5 | Defer optional analyses | Grid truncated to 186 evaluations, Spambase dispersion mechanism, threshold-free re-scoring, HV reference-point sensitivity and every new run are not performed; their patches are deferred. |
| C6 | Editorial revision with constraints | Applied where it improves clarity without hiding a limitation or changing meaning. The title change (dropping ": A Replicated Study") is NOT applied: the title was a recorded decision of all three authors (cbefdcf) and needs their agreement. |

## Author-only facts (docket Q1-Q4)

| Question | Text used | Still to be confirmed |
|---|---|---|
| Q2 gate thresholds | Inherited unchanged from the companion study, as a transfer, not a calibration | Whether the companion study's Santander pilot informed them |
| Q3.1 generative AI | Anthropic's Claude (Claude Opus 4.7 and Claude Opus 5 through Claude Code, from the commit trailers of the manuscript, protocol, code, review and verification commits) and OpenAI's ChatGPT (named in the frozen text and by the corresponding author); uses never narrower than the record | Each author's attestation; ChatGPT version |
| Q3.3 references | The unsupported "every entry was verified" sentence is removed | One author's check of every entry before upload |
| Q4.1 dissertation | Title, handle, open-access status and date verified on 2026-10-11 against the UNIFEI repository record (handle 123456789/4372; dc.title as cited; dc.rights "Acesso Aberto"; issued 2026-02-19) | The dissertation's two additional datasets: the record says only that the pipeline was also validated on a balanced and a highly imbalanced dataset; if either is Adult, Bank Marketing or Spambase, the overlap statement must name it |
| Q4.2 archived grid step | States that the archived code does not establish which step produced the dissertation's numbers | Which step was used |
| Q4.3 timing of the signed-rank choice | Labelled post hoc | Any dated record of an earlier choice |
| Q4.4 reimplemented surface fit | Stated without a reason | The reason, if any |

## Journal requirements verified on 2026-10-11 (Applied Soft Computing)

The ScienceDirect Guide for Authors returns HTTP 403 to automated retrieval; the newest public archive capture is
the 2024-06-30 copy kept in `submission/_source_asoc_guide_20240630.txt`. The live guide, as indexed by search on
2026-10-11, differs from that capture on the abstract: it now asks for "a concise and factual abstract which does not
exceed 250 words" stating the purpose, the basic procedures, the main findings with their statistical significance
where applicable, and the principal conclusions. `submission/ASOC_REQUIREMENTS.md` (which records "no word limit")
is corrected accordingly. Re-read the live page at upload.

- Abstract at most 250 words; stands alone; no references; non-standard abbreviations defined at first mention.
- 1 to 7 keywords in English, avoiding multi-word keywords built with "and" or "of".
- Highlights encouraged: 3 to 5 bullet points of at most 85 characters each, in a separate editable file.
- Graphical abstract required, as a separate file.
- Editable source required: `.tex` for LaTeX ("A PDF is not an acceptable source file"); the Elsevier LaTeX template
  is encouraged.
- Generative AI (Elsevier policy): a separate declaration before the references, titled "Declaration of generative
  AI and AI-assisted technologies in the manuscript preparation process"; AI used in the research process is
  described in the methods.

## Patch policy

- A and B patches are applied after their anchors are re-verified at the branch tip; a stale anchor is rejected.
- Patches that print or read numbers from the new read-only scripts are applied only after those scripts and their
  outputs are committed.
- The tests that read files at the v6 tag (P2-T-33 to P2-T-35) are deferred to the retag commit.
- `PROTOCOL_AMENDMENTS.md`, the protocol, `PROTOCOL_V3_FREEZE_REPORT.md` and every committed analysis artifact are
  not edited; the post-campaign corrections are filed in the new file `PROTOCOL_ERRATA_POST_CAMPAIGN.md`.

## Dispositions recorded during implementation

- Section 3 (P2-014 E1/E3): retained before Section 4 because RQ1 is defined on it, retitled "The historical method";
  the stated-once, D1-citation and no-audit-heading conditions are met (P2-L-09, P2-L-10, P2-L-12).
- `docs/METHODOLOGY_DECISIONS.md` D1 (P2-014 E7): the entry has no numbered open item 8; a dated addendum records how
  its four decision items were carried out, and the dated entry is unchanged.
- Tables are numbered in order of appearance: the Section 3 table is Table 1 and the hyperparameter table Table 2.
- The ASOC package files (`submission/ASOC_*.md`) are generated from the committed manuscript; until the v6 tag exists
  they are regenerated from the revision branch head as a draft (`ASOC_PACKAGE_SOURCE_REF`), never by hand.

## Presentation QA patch (QA2, 2026-10-12)

Requested by the corresponding author after a page-by-page inspection of the review PDFs; presentation and
provenance wording only, no analysis, artifact or reported result changes.

- Figures are numbered in the order of their first citation, which is also the order in which the LaTeX build places
  them: anchor provenance is Figure 3, reference sensitivity Figure 4 and boundary dispersion Figure 5 (previously
  4, 5 and 3, which the LaTeX build printed under the placement numbers 3, 4 and 5 while the text cited the old
  ones). The introduction's contribution list cites Section 6.4 instead of the anchor figure, so that no figure is
  cited before Figure 1. `build_latex.py` now fails if the first-citation order and the numbering differ.
- Figure 6: both legends sit above their axes, with the space reserved, so that no bar is covered.
- Figure 4 (reference sensitivity): the axis limits follow the data. The fixed +/-0.55 had clipped the Spambase
  anchor-provenance cell (CORE -0.6932, AUGMENTED -0.3464) out of view, so the chart showed eleven of the twelve
  cells its caption describes; the plotted values are unchanged.
- The supplement is a review draft until the authors approve it: S17 names `paper2-manuscript-v6` as the planned
  tag, not yet created. The release build sets `PAPER2_PACKAGE_RELEASE=paper2-manuscript-v6`, which restores the
  final-package wording that `audit_package_tag.py` checks at the tag.

## Coauthor review R2 (2026-10-13)

A second, short review by the same coauthor. No experiment, rerun, dataset or change to a frozen result; two
retrospective diagnostics from stored outputs, labelled post hoc, with their definitions committed before the
first run (centre-point criteria at 933422a). Patch identifiers below are those cited in
`audits/scientific_revision_v6.json`; every R2 numeric movement there is an addition, none removes or alters an
existing number.

- Centre points (P2-R2-03, -07, -12, -16, -19, -21, -27). `scripts/center_point_sensitivity.py` reproduces the
  stored 88-row surrogate stage exactly and refits it on the 79 distinct configurations; it also refits the
  factor models on the 79 distinct pilot configurations. Premise verified: the ten centre rows are one cached
  evaluation in every unit (79 computed, 9 served) and ten identical deterministic executions in the pilot. Factor
  models: unchanged in every respect measured (min Tucker 0.999). Surfaces: predictions nearly unchanged, terms
  often differ by one or two, gate decisions flip for MAGIC quality (8 -> 1 passes) and Bank Marketing cost
  (27 -> 23). Surrogate anchors moved beyond the pre-declared tolerance in more than half of the replications
  on every dataset, so the pre-declared class is HIGH, by the anchor rule alone. A descriptive control, added
  after the first run and outside the classification, shows that restarting the frozen multistart moves the
  quality anchor about as often and almost always to an equally good minimum (the flat-direction mechanism
  already stated in Section 5.6); the cost-anchor movement is mainly a centre-row effect. Handling, as the HIGH
  rule requires: the frozen protocol and results are untouched; the affected claim, the post hoc CHIM-contraction
  association, is no longer carried into the Discussion summary or the Conclusion and is qualified in 6.4.1, 7.5
  and the Figure 3 caption; Section 8.5 states that the surrogate payoff carries a component set by the
  minimizer's starting points and that the contrasts' behaviour under the alternative specification was not
  tested; the decision whether to run a new post hoc experiment is left to the authors.
- Signed-rank status (P2-R2-01, -02, -05, -06, -11, -19). The Wilcoxon signed-rank test is labelled secondary and
  post hoc wherever its significance is claimed, beside the pre-declared Nadeau-Bengio result (Adult only for the
  geometry contrast, MAGIC only for the anchor contrast). Abstract kept at 250 words.
- Grid at the arms' budget (P2-R2-01, -08, -13, -15, -25). `scripts/grid_budget_prefix.py` scores the first 186
  stored grid evaluations, verified to be the frozen 186-point grid, with the frozen code: the grid's lead and its
  21-of-30 status are unchanged on every dataset. No grid point evaluated.
- Holdout (P2-R2-04, -09, -23). "Replication-level holdout" adopted; the overlap with the pilot that fixed the
  objective mapping is stated as overlap with the measurement definition, not label leakage.
- Instability and scope (P2-R2-10, -13, -14, -19, -24). 6.10 and the Conclusion distinguish changed candidate sets
  from contrast directions and keep the Adult reversal under measured time; the recommendation sentence of 7.7 is
  scoped; Spambase and the primary panel are stated as conditional on a one-partition screen.
- Availability (P2-R2-18, -20, -22, -26). Public repository named; public, independently obtainable, commit-pinned
  and unversioned material distinguished; the draft header no longer implies the frozen material is inaccessible;
  the ASOC statement no longer promises an existing Zenodo archive.
