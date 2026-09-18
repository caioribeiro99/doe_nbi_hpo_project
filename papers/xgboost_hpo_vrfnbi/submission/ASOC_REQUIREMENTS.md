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

## Requirements the frozen package already meets

| Item | Journal's words | Status |
|---|---|---|
| Abstract length | No word limit is stated anywhere in the guide. Searched exhaustively. | **PASS** — 257 words, unconstrained. |
| Acronyms in title/abstract | No prohibition exists. (This was an EAAI rule and does **not** transfer.) | **PASS** — no expansion forced. |
| Reference formatting at submission | "This journal does not set strict requirements on reference formatting at submission… References can be in any style or format as long as the style is consistent." | **PASS** — author-date, consistent. |
| Reference style at proof | "Indicate references by adding a number within square brackets in the text… Number references in the order they appear." | Applied by the publisher after acceptance. |
| Section numbering | "Subsections should be numbered 1.1 (then 1.1.1, 1.1.2, …)… Use this numbering also for internal cross-referencing." | **PASS** — repaired in v3/v4. |
| Competing interests | "All authors must disclose any financial and personal relationships…" | **PASS** — statement present. |
| CRediT | "Corresponding authors are required to acknowledge co-author contributions using CRediT roles." | **PASS** — present, author-confirmed, unchanged from v5. |
| Funding | "Authors must disclose any funding sources…" | **PASS** — present, unchanged from v5. |
| Data availability | "you are required to state the availability of any data at submission." | **PASS** — statement present (see Option C above for the stronger deposit rule). |
| Highlights | "Highlights are optional." (The 2023 legacy page contradicted itself; the 2024 page resolves it as encouraged.) 3–5 bullets, ≤85 characters each. | Optional — drafted. |
| Page / file-size limit | No page limit and no file-size limit is stated. (Both were EAAI rules.) | **PASS** — 32 pp, 1.4 MB. |
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
Methods and are **not** what this declaration covers — the policy is explicitly about
"scientific writing". The declaration is owed only if generative AI assisted in *writing
the manuscript*.

**This is a factual question only the authors can answer, and it is not established by
the project record.** It is therefore listed as an author decision, not filled in. Note
the policy's own escape: if there is nothing to disclose, no statement is added.

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

## Unretrieved — not asserted anywhere above

Whether the 2024 formatting rules still hold in September 2026; whether the "4 potential
referees" requirement (2023 capture only) still applies; ORCID policy; cover-letter
requirement for regular articles; accepted supplementary-file formats.
