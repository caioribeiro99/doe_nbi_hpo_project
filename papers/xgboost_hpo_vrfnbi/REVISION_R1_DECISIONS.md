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
