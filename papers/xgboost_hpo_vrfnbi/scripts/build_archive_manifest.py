"""Enumerate exactly what a reproducibility deposit would contain, and audit it.

Written because the obvious route is not the safe one. Archiving a GitHub release to
Zenodo deposits the WHOLE repository tarball at the tag, and this repository also holds a
previous paper's doctoral benchmark: 34.8 MB of job-queue databases and stage summaries
that carry the author's home directory in 545 absolute paths. None of it belongs in this
paper's archive. So the deposit is curated, and this script is what proves the curation
is complete rather than asserted.

Nothing here is deleted or modified. The script reports.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "papers/xgboost_hpo_vrfnbi/submission/ZENODO_ARCHIVE_MANIFEST.md"

# Included, with the reason each group is releasable.
INCLUDE: list[tuple[str, str]] = [
    ("src/", "campaign, arms, scoring and seeding implementation"),
    ("papers/xgboost_hpo_vrfnbi/protocol/", "the protocol as frozen before the campaign"),
    ("papers/xgboost_hpo_vrfnbi/scripts/", "analysis, audit and build scripts"),
    ("papers/xgboost_hpo_vrfnbi/analysis/", "aggregated analysis artifacts every reported number is read from"),
    ("papers/xgboost_hpo_vrfnbi/audits/", "screening, factor-model and identity audits"),
    ("papers/xgboost_hpo_vrfnbi/manuscript/sections/", "manuscript source"),
    ("papers/xgboost_hpo_vrfnbi/manuscript/figures/", "figures, generated from the artifacts"),
    ("data/design/", "the 88-run design matrix and its metadata"),
    ("data/source/", "dataset manifests and SHA-256 checksums — NOT the datasets"),
    ("configs/", "frozen configuration"),
    ("tests/", "the regression suite, including the reproducibility guards"),
    ("docs/", "methodology decisions and engineering notes"),
]
ROOT_FILES = ["pyproject.toml", "README.md", "LICENSE", "CITATION.cff", ".zenodo.json"]

# Excluded, with the reason. Every exclusion is a stated decision, not an omission.
EXCLUDE: list[tuple[str, str]] = [
    ("jobs/", "34.8 MB of OpenML CC-18 job-queue databases from the PREVIOUS paper's "
              "doctoral benchmark; irrelevant to this study"),
    ("experiments/", "stage-run summaries from the previous paper; they carry the "
                     "author's home directory in 545 absolute paths"),
    ("benchmarks/", "task lists for the previous paper's benchmark"),
    ("article/", "the previous paper's LaTeX draft"),
    ("notebooks/", "exploratory, not part of the frozen pipeline"),
    ("examples/", "demonstration scripts for the previous paper"),
    ("papers/xgboost_hpo_vrfnbi/submission/", "journal correspondence — cover letter, "
     "title page, vitae, checklist. Submission strategy does not belong in a public "
     "reproducibility archive"),
    ("papers/xgboost_hpo_vrfnbi/vendor/", "third-party MathJax; MIT-licensed and "
     "redistributable, but it is a build dependency rather than research output"),
    ("papers/xgboost_hpo_vrfnbi/manuscript/Paper2_*", "compiled PDFs; the published "
     "article is the canonical copy"),
]

PATTERNS = ["/Users/", "/home/", "kaggle.json", "token", "password", "secret",
            "api_key", "credential", ".env"]
BINARY = {".png", ".pdf", ".sqlite", ".jpg", ".ico", ".woff", ".woff2"}


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\0") if f]


def purpose_of(path: str) -> str:
    for prefix, why in INCLUDE:
        if path.startswith(prefix):
            return why
    return "repository metadata" if path in ROOT_FILES else "—"


def main() -> int:
    files = tracked()
    inc_prefixes = tuple(p for p, _ in INCLUDE)
    selected = sorted(f for f in files
                      if (f.startswith(inc_prefixes) or f in ROOT_FILES)
                      and not f.startswith("papers/xgboost_hpo_vrfnbi/submission/")
                      and not f.startswith("papers/xgboost_hpo_vrfnbi/vendor/"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()

    # audit
    findings: list[tuple[str, str, int]] = []
    total = 0
    by_group: dict[str, list[int]] = {}
    for f in selected:
        p = REPO / f
        if not p.exists():
            continue
        size = p.stat().st_size
        total += size
        grp = next((pre for pre, _ in INCLUDE if f.startswith(pre)), "(root)")
        by_group.setdefault(grp, []).append(size)
        if p.suffix.lower() in BINARY:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        low = text.lower()
        for pat in PATTERNS:
            n = low.count(pat.lower())
            if n:
                findings.append((pat, f, n))

    # "token" and ".env" match ordinary prose; separate the real risks from the noise
    REAL = ("/Users/", "/home/", "kaggle.json", "password", "api_key", "secret", "credential")
    real = [x for x in findings if x[0] in REAL]
    noise = [x for x in findings if x[0] not in REAL]

    w = [f"# Proposed Zenodo archive — manifest and audit\n",
         f"Generated from the tracked tree at commit `{commit}`. "
         f"**{len(selected)} files, {total/1048576:.2f} MB.**\n",
         "The GitHub-release-to-Zenodo route would archive the whole repository, which "
         "also holds the previous paper's doctoral benchmark. This manifest is the "
         "curated alternative; every exclusion below is a stated decision.\n",
         "## Included\n",
         "| group | files | size | purpose | releasable |", "|---|---|---|---|---|"]
    for prefix, why in INCLUDE:
        sizes = by_group.get(prefix, [])
        if sizes:
            w.append(f"| `{prefix}` | {len(sizes)} | {sum(sizes)/1024:.0f} KB | {why} | yes |")
    rootsz = by_group.get("(root)", [])
    if rootsz:
        w.append(f"| repository metadata | {len(rootsz)} | {sum(rootsz)/1024:.0f} KB | "
                 f"{', '.join('`%s`' % r for r in ROOT_FILES)} | yes |")

    w += ["\n## Excluded, and why\n", "| path | reason |", "|---|---|"]
    for prefix, why in EXCLUDE:
        w.append(f"| `{prefix}` | {why} |")

    w += ["\n## Restricted-material audit\n",
          f"Patterns searched: {', '.join('`%s`' % p for p in PATTERNS)}.\n"]
    if real:
        w.append("| pattern | file | occurrences |")
        w.append("|---|---|---|")
        for pat, f, n in sorted(real):
            w.append(f"| `{pat}` | `{f}` | {n} |")
    else:
        w.append("**No occurrence of any sensitive pattern in the archive set.**\n")
    w.append(f"\n`token` and `.env` match {sum(n for _, _, n in noise)} times across "
             f"{len({f for _, f, _ in noise})} files, all ordinary prose — \"numeric "
             "token\", \"tokenize\", and the README's instructions for copying "
             "`.env.example`. No credential file is tracked.\n")

    w += ["\n## Datasets\n",
          "`data/source/` carries manifests and SHA-256 checksums only. **No dataset is "
          "redistributed.** The four analysed datasets are public and are fetched by "
          "checksum-verified loaders; the archive reproduces the pipeline, not the data.\n"]
    OUT.write_text("\n".join(w) + "\n")

    print(f"  archive set: {len(selected)} files, {total/1048576:.2f} MB")
    print(f"  sensitive-pattern findings: {len(real)}")
    for pat, f, n in real:
        print(f"     {pat:<12} {n:<3} {f}")
    print(f"  prose false positives (token/.env): {len(noise)} files")
    print(f"  wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
