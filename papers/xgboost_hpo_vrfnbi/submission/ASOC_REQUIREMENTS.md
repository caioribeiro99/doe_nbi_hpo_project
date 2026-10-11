# Applied Soft Computing — submission requirements

Quoted from the journal's own pages, recovered from the Internet Archive (ScienceDirect
and elsevier.com return HTTP 403 to every automated route; this session's web-search
budget was spent on the EAAI retrieval). The retrieved page text is committed alongside
this file as `_source_asoc_guide_20240630.txt` and `_source_asoc_scope_20260203.txt` so
every quote can be checked without re-fetching.

> **Currency warning.** Scope text is current to **2026-02-03**. The Guide for Authors
> text is from **2024-06-30**, the newest capture that exists. Two rules demonstrably
> changed inside the observed window — keywords went 3–10 → 1–7, and the graphical
> abstract went optional → required — so treat every formatting rule here as
> **to-be-reconfirmed against the live page before submitting**.

## Review model

> "This journal follows a **single anonymized** review process. Your submission will
> initially be assessed by our editors to determine suitability for publication in this
> journal. If your submission is deemed suitable, it will typically be sent to a minimum
> of two reviewers…"

**No anonymized manuscript and no separate blinded file are required** — the opposite of
EAAI. Author details stay in the manuscript.

## Requirements that the frozen package does not yet meet

| Item | Journal's words | Status |
|---|---|---|
| **Keywords** | "You are required to provide **1 to 7 keywords** for indexing purposes… Please try to avoid keywords consisting of multiple words (using 'and' or 'of')." | **Was 8.** Reduced to 7 — see below. |
| **Graphical abstract** | "You are **required** to provide a graphical abstract at submission… Ensure the image is a minimum of **531 x 1328 pixels (h x w)**… readable at a size of 5 x 13 cm… preferred file types are TIFF, EPS, PDF or MS Office files." | **Built** from the study's own frozen artifacts. |
| **Vitae** | "Please submit a short (**maximum 100 words**) biography of each author and include a **passport-type photograph** as a separate figure." | **AUTHOR INPUT REQUIRED** — biographies and photographs cannot be written or sourced here. |
| **Declaration of Interests tool** | "The Declaration of Interests tool should always be completed… The resulting Word document… should be uploaded… saved in the .doc/.docx file format." | **AUTHOR ACTION** — the tool output is generated in the submission system. |
| **Research data (Option C)** | "You are required to: Deposit your research data in a relevant data repository. Cite and link to this dataset in your article. If this is not possible, make a statement explaining why research data cannot be shared." | **AUTHOR DECISION** — the code is on GitHub; a DOI-bearing deposit would satisfy this cleanly. |
| **Editable source files** | "We ask you to provide editable source files for your entire submission (including figures, tables and text graphics)… Save files in an editable format, using the extension .doc/.docx for Word files and .tex for LaTeX files. A PDF is not an acceptable source file." (guide:246-247); "You will be asked to provide all relevant editable source files upon submission or revision." (guide:252) | **NOT MET in v5**: the manuscript exists only as Markdown printed to PDF through HTML/CSS (`scripts/build_pdfs.py`); no `.tex` or `.docx` is built. Planned (decision C4, default): LaTeX in Elsevier's elsarticle template, which the guide encourages (guide:252). Recheck on the live page whether a PDF is accepted at first submission. |
| **Figures** | "Figures, images, artwork, diagrams and other graphical media must be supplied as separate files along with the manuscript." (guide:302); vector EPS or PDF, or bitmaps of at least 300, 500 or 1000 dpi by type (guide:311-314); "Provide captions in a separate file." (guide:320) | **NOT MET in v5**: `manuscript/figures/fig1.png` to `fig6.png` are 200-dpi rasters and are not in the package. Export Figure_1 to Figure_6 as vector files and supply the captions (`sections/12_figure_captions.md`) as a separate file. |
| **Abbreviations in the abstract** | "Avoid non-standard or uncommon abbreviations. If any are essential to include, ensure they are defined within your abstract at first mention." (guide:268); "Abbreviations which are essential to include in your abstract should be defined at first mention in your abstract, as well as in a footnote on the first page of your article." (guide:393) | **OPEN**: v5 used CORE, AUGMENTED and NBI-S undefined and never introduced NBI. The revised abstract defines NBI, NBI-S, CORE and AUGMENTED at first mention; the first-page footnote must be added in the editable source. |

## Requirements the frozen package already meets

| Item | Journal's words | Status |
|---|---|---|
| Abstract length | The 2024-06-30 capture states no limit (guide:264-268). The live guide, as indexed by search on 2026-10-11 (direct retrieval returns HTTP 403), asks for "a concise and factual abstract which does not exceed 250 words", stating the purpose, basic procedures, main findings with their statistical significance where applicable, and principal conclusions. | **MET for the live rule**: the revised abstract is 247 words (whitespace tokens of the source), defines CORE and NBI-S at first mention and carries the weighting-conditionality sentence of protocol section 11. Recheck the live page at upload. |
| Reference formatting at submission | "This journal does not set strict requirements on reference formatting at submission… References can be in any style or format as long as the style is consistent." | **PASS** — author-date, consistent. |
| Reference style at proof | "Indicate references by adding a number within square brackets in the text… Number references in the order they appear." | Applied by the publisher after acceptance. |
| Section numbering | "Subsections should be numbered 1.1 (then 1.1.1, 1.1.2, …)… Use this numbering also for internal cross-referencing." | **PASS** — repaired in v3/v4. |
| Competing interests | "All authors must disclose any financial and personal relationships…" | **PASS** — statement present. |
| CRediT | "Corresponding authors are required to acknowledge co-author contributions using CRediT roles." | **PASS** — present, author-confirmed, unchanged from v5. |
| Funding | "Authors must disclose any funding sources…" | **PASS** — present, unchanged from v5. |
| Data availability | "you are required to state the availability of any data at submission." | **PASS** — statement present (see Option C above for the stronger deposit rule). |
| Highlights | "Highlights are optional." (The 2023 legacy page contradicted itself; the 2024 page resolves it as encouraged.) 3–5 bullets, ≤85 characters each. | Optional — drafted. |
| Page limit | "Authors are invited to submit technical papers (20-30 pages, at most 50 pages)" (guide:131); no file-size limit is stated. | **Within the maximum**: v5 is 32 pp as built, above the 20-30 norm; recount after conversion to the editable source. |
| Desk-rejection list | None exists. Only the ordinary initial editorial suitability screen. | — |

## Generative AI — exact required wording

> "The use of generative AI and AI-assisted technologies in scientific writing must be
> declared by adding a statement at the end of the manuscript when the paper is first
> submitted. The statement will appear in the published work and should be placed in a
> new section before the references list. An example:
> **Title of new section: Declaration of generative AI and AI-assisted technologies in
> the writing process.**
> Statement: During the preparation of this work the author(s) used [NAME TOOL / SERVICE]
> in order to [REASON]. After using this tool/service, the author(s) reviewed and edited
> the content as needed and take(s) full responsibility for the content of the published
> article.
> The declaration does not apply to the use of basic tools, such as tools used to check
> grammar, spelling and references. **If you have nothing to disclose, you do not need to
> add a statement.**"

Two further rules:

> "Authors must not list or cite AI and AI-assisted technologies as an author or
> co-author on the manuscript."

> "We do not permit the use of Generative AI or AI-assisted tools to create or alter
> images in submitted manuscripts. The only exception is if the use of AI or AI-assisted
> tools is part of the research design or methods… The use of generative AI or
> AI-assisted tools in the production of artwork such as for graphical abstracts is not
> permitted."

**The distinction that matters for this paper.** XGBoost, the surrogates, NSGA-II,
Bayesian optimization and TPE are the *research methodology*. They are described in the
Methods and are **not** what this declaration covers. Generative-AI assistance is a
different matter, and the project record shows more of it than manuscript drafting: most
commits that touch the analysis code (`scripts/`, `src/doe_xgb`) up to `d40eb0b` carry a
`Co-Authored-By: Claude` trailer. Whether the adversarial pre-campaign reviews and the
verification passes were also AI-run is for the authors to state.

**Status.** The authors supplied a declaration naming Anthropic Claude and OpenAI ChatGPT
(2026-09-21); it is carried in the manuscript and, extracted from it, on the title page. Its
scope is an author attestation (decision Q3): the words "manuscript preparation only" can stand
only if the authors confirm that no AI-assisted analysis code, review or verification needs
disclosure; otherwise the declaration, and the Reproducibility statement for the code, name
those uses. Paper 1's declaration is to be worded consistently.

On the artwork rule: the graphical abstract built here is a `matplotlib` rendering
produced deterministically from the study's committed artifacts by a script in this
repository. It is programmatic plotting of the paper's own data, the same mechanism that
produced Figures 1–6 — not generative-AI artwork.

## Keywords: what changed and why

v5 carried **eight**; the journal allows at most seven, and asks authors to avoid
keywords built with "and" or "of".

Removed: **"design of experiments"** — it was the only keyword containing "of", and
"response surface methodology" (retained) covers the same literature. Nothing else
changed; the remaining seven are v5's own words in v5's order.

This is a reversible indexing choice, not a scientific one. If the author prefers to keep
"design of experiments", drop one of the others instead.

## Unretrieved: not asserted anywhere above

Whether the 2024 formatting rules still hold in September 2026; whether the "4 potential
referees" requirement (2023 capture only) still applies; ORCID policy; cover-letter
requirement for regular articles; accepted supplementary-file formats.

## Recheck on the live pages immediately before upload

Every rule above comes from the 2024-06-30 archive capture, except the abstract-length rule, which was
confirmed on 2026-10-11 from the search-engine index of the live page (the page itself returns HTTP 403 to
automated retrieval). Before uploading, confirm on the journal's live pages:

- whether a PDF is accepted at first submission, and the editable-source, figure-file,
  resolution and caption-file rules;
- the abstract word limit (250 words on 2026-10-11; the revised abstract is 247 words), and the
  abbreviation rule with its first-page footnote;
- the page guidance (technical papers 20-30 pages, at most 50);
- the highlights and graphical-abstract rules;
- the current generative-AI policy wording and section title, against the declaration as
  revised by the authors (decision Q3).
