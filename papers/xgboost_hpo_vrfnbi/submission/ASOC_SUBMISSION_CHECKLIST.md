# Applied Soft Computing — submission checklist

Every mutable journal rule carries its verification status. An archived rule is never silently promoted to a live one.

| Requirement | Status | Evidence | This package |
|---|---|---|---|
| Generative-AI declaration: section title, template, placement | **VERIFIED LIVE 2026-09-21** | Elsevier policy page, last updated June 2026, fetched directly | Done — in the manuscript before the references, and on the title page |
| Generative AI not an author; not used for graphical-abstract artwork | **VERIFIED LIVE 2026-09-21** | same page | Complies — abstract figure is matplotlib from committed data |
| AI in research methodology needs no declaration | **VERIFIED LIVE 2026-09-21** | same page | XGBoost, NSGA-II, BO, TPE described in Methods only |
| Journal identity, ISSN, hybrid-subscription, submission route | **VERIFIED LIVE 2026-09-21** | core.submit.elsevier.com/core/v1/journals/ASOC returned 200 live | ASOC, 1568-4946, HYBRID_SUBSCRIPTION, redirectToEm=false |
| Editorial Manager is NOT the live submission route | **VERIFIED LIVE 2026-09-21** | editorialmanager.com/asoc: "Site under development. Do not use for live manuscript submission" | No live form could be inspected |
| Aims and scope | **ARCHIVE 2026-02-03** | journal home capture | Scope gate accepted: DEFENSIBLE FIT WITH EDITORIAL RISK |
| Keywords 1 to 7 | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | guide capture | Done — reduced 8 to 7 |
| Graphical abstract required; min 531 x 1328 px | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | guide capture | Done — 1527 x 610 px, proportional |
| Highlights optional; 3-5 bullets, max 85 characters | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | guide capture | Done — 5 bullets, all within limit |
| Single anonymized review; no blinded file | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | guide capture | Authors visible on the title page; no anonymized manuscript prepared |
| Vitae: 100-word biography and passport photograph per author | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | guide capture | Biographies drafted; photographs are author-supplied |
| Corresponding author contact details incl. phone numbers | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | submission checklist in the guide capture | Email `caio.tertu@hotmail.com` and postal address supplied; phone only if the live portal asks |
| No institutional-email requirement | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | guide capture: the rule is only "the email address of each author" | Any valid address satisfies it |
| Reference formatting flexible at submission | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | guide capture | Author-date, consistent — compliant |
| Research data Option C: deposit, cite and link | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | guide capture | Statement drafted with a DOI placeholder; deposit pending |
| No page limit, no file-size limit, no abstract word limit, no acronym ban | **ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION** | guide capture — searched exhaustively, none stated | 32 pp, 1.4 MB, 257-word abstract all unconstrained |

## Why some rules could not be verified live

ScienceDirect and elsevier.com return HTTP 403 to every automated route, this session's web-search budget is exhausted, and the Internet Archive holds no capture of this journal's Guide for Authors newer than 2024-06-30 — re-checked on 2026-09-21 and still none. The live Editorial Manager site states it is not to be used for submission, so no live form exists to read. Elsevier's own policy pages ARE reachable, which is why the generative-AI rules could be verified live and the journal-specific formatting rules could not.
