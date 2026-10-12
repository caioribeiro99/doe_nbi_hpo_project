"""The tag the supplement names must be the tag the package actually carries.

A commit cannot contain a tag pointing at itself, so the supplement names its package
tag as a forward reference. That is honest only if the reference is then true. This
closes the loop after the freeze: the named tag must exist, and it must resolve to a
commit whose own supplement names that same tag. Hardcoding the tag instead is how the
v3 tag survived into the v4 candidate.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
SUP = "papers/xgboost_hpo_vrfnbi/manuscript/SUPPLEMENT.md"
TAG_ROW = re.compile(r"\|\s*final package tag\s*\|\s*`([^`]+)`")
# A review draft names the tag as planned, not as its own (build_supplement.py, RELEASE).
PLANNED_ROW = re.compile(r"\|\s*package tag\s*\|\s*not yet created; planned name\s*`([^`]+)`")


def git(*args: str) -> tuple[int, str]:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def main() -> int:
    working = (REPO / SUP).read_text()
    named = TAG_ROW.search(working)
    planned = PLANNED_ROW.search(working)
    if not named and planned:
        tag = planned.group(1)
        print(f"  working supplement is a review draft; planned tag {tag}")
        code, _ = git("rev-parse", "--verify", "--quiet", tag + "^{commit}")
        if code == 0:
            print(f"FAIL: {tag} exists but the working supplement still calls it planned")
            return 1
        print("PENDING: the tag is not created; run again after the release build and the tag")
        return 0
    if not named:
        print("FAIL: the working supplement states no package tag")
        return 1
    tag = named.group(1)
    print(f"  working supplement names      {tag}")

    code, commit = git("rev-parse", "--verify", "--quiet", tag + "^{commit}")
    if code != 0:
        print(f"  {tag} does not exist yet — the package is not frozen")
        print("PENDING: run again after the tag is created")
        return 0
    print(f"  {tag} resolves to             {commit[:12]}")

    code, at_tag = git("show", f"{tag}:{SUP}")
    if code != 0:
        print(f"FAIL: {tag} carries no supplement at {SUP}")
        return 1
    inside = TAG_ROW.search(at_tag)
    if not inside:
        print(f"FAIL: the supplement at {tag} states no package tag")
        return 1
    print(f"  supplement at that tag names  {inside.group(1)}")
    if inside.group(1) != tag:
        print(f"FAIL: {tag} carries a supplement naming {inside.group(1)} instead")
        return 1

    # and the source commit it names must be an ancestor of the tagged commit
    m = re.search(r"\|\s*manuscript source commit\s*\|\s*`([0-9a-f]{40})`", at_tag)
    if not m:
        print("FAIL: the supplement states no manuscript source commit")
        return 1
    code, _ = git("merge-base", "--is-ancestor", m.group(1), commit)
    print(f"  source commit {m.group(1)[:12]} is an ancestor: {code == 0}")
    if code != 0:
        print("FAIL: the named source commit is not an ancestor of the tagged commit")
        return 1

    print(f"\nPASS: {tag} and the supplement it carries agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
