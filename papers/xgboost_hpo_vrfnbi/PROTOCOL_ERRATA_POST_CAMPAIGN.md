# Post-campaign errata

Dated 2026-10-11, revision branch `revision/paper2-matheus-review`. This file records statements in frozen records
that are inaccurate, without editing those records: `PROTOCOL_AMENDMENTS.md` (the amendment ledger), the protocol
under `protocol/`, the frozen screening artifacts and the code at the protocol and results tags are unchanged. The
manuscript states the corrected facts (Sections 8.6 and 8.7, Supplement S4.4).

## E1. "No arm had executed" before amendments 20 to 23

Statements that no arm had executed when the panel was screened or amended appear in the ledger's closing
section (`PROTOCOL_AMENDMENTS.md`, line 538, "How 'no arm has been executed' can be checked"), in
`protocol/dataset_selection.md` (line 56), in `scripts/final_panel_screening.py` (lines 33 and 177) and in
`audits/final_panel_screening.json` (line 272). For amendments 20 to 23 they are inaccurate: pre-freeze engineering smoke runs on Spambase had executed every arm, and
one complete unit (commit `66e62bb`, 2026-09-14) computed every arm's indicators against both references. The
accurate statement is that no confirmatory arm had executed and, by the ledger's record, no arm performance had been
inspected. The smoke unit's outputs are not archived.

## E2. The docstring of `tucker_congruence` claims a sign resolution the code does not perform

`src/doe_xgb/campaign/factor_model.py`, lines 185-194: the docstring of `tucker_congruence` says the congruence is
"computed after matching each column of `b` to the column of `a` it best corresponds to and resolving the sign". The
function (lines 196-209) matches columns greedily by absolute congruence and returns the signed value without
orienting the matched column, so a congruence recorded by the runner can be
negative because of the matching alone. The congruences recorded during the campaign are reported as recorded, and
the post hoc optimal-assignment matching with sign orientation is reported beside them, in Supplement S4.4
(`scripts/tucker_congruence_refit.py`, `analysis/tucker_congruence_refit.json`). The frozen code is not edited.
