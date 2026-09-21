# Data and code availability — statement for submission

Applied Soft Computing applies Elsevier **Option C**, quoted from the Guide for Authors
(ARCHIVE-ONLY, 2024-06-30 — recheck at submission):

> "You are required to: Deposit your research data in a relevant data repository. Cite and
> link to this dataset in your article. If this is not possible, make a statement
> explaining why research data cannot be shared."

## Statement to enter at submission

> The protocol, the frozen factor models, the campaign and analysis source code, and every
> analysis artifact from which the reported numbers are read are openly available at
> `https://github.com/caioribeiro99/doe_nbi_hpo_project` under the MIT licence, archived at
> Zenodo with DOI **[ZENODO_DOI_PLACEHOLDER]**. The four datasets analysed — MAGIC Gamma
> Telescope, Adult / Census Income, Bank Marketing and Spambase — are public and are
> retrieved by checksum-verified loaders; they are not redistributed in the archive. The
> per-unit evaluation caches are large intermediate state rather than evidence and are
> deliberately not versioned; the committed artifacts are sufficient to recompute every
> reported quantity without re-running the campaign.

**The DOI is a placeholder and has not been invented.** It is the one field in this package
that requires an act outside the repository: creating the Zenodo deposit.

## Release procedure

1. Push a GitHub release from the submission commit on `submission/paper2-asoc`.
2. Enable the Zenodo–GitHub integration for the repository, so the release is archived and
   a DOI minted automatically. `.zenodo.json` at the repository root supplies the title,
   the three authors with their ORCIDs, the affiliation, the keywords and the description,
   so the deposit metadata does not have to be retyped in the Zenodo form.
3. Substitute the concept DOI into the statement above and into the manuscript's data and
   code availability paragraph.

## What the archive includes and excludes

| Included | Excluded |
|---|---|
| frozen protocol and configuration | raw datasets (public, checksum-verified loaders) |
| campaign and analysis source code | per-unit evaluation caches |
| 88-run design and frozen factor models | — |
| aggregated per-unit indicator tables and analysis artifacts | — |
| environment and dependency information | — |
| reproducibility instructions and tag/commit references | — |
