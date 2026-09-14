"""Stage checkpoints must make a resumed unit indistinguishable from an unbroken one.

The confirmatory campaign is 120 units and roughly 17 hours. It WILL be
interrupted -- a machine sleeps, a process is killed, a disk fills. Resume is not a
convenience here, it is the difference between an interruption costing minutes and
costing the campaign. Two failure modes matter and are tested:

* a stage killed mid-write must not read as complete on resume, or the unit
  continues from a truncated artifact and every downstream stage is silently wrong;
* a completed stage must not re-run, or resume re-spends real evaluations and the
  logical budget the paper reports stops being the budget the campaign spent.
"""
from __future__ import annotations

import json

import pytest

from doe_xgb.campaign.runner import STAGES, Checkpoint


@pytest.fixture()
def ck(tmp_path):
    return Checkpoint(tmp_path)


def test_a_stage_not_yet_run_is_not_done(ck):
    assert not ck.done("factor_model")


def test_a_saved_stage_is_done_and_round_trips(ck):
    ck.save("factor_model", {"weights": [0.5, 0.5]})
    assert ck.done("factor_model")
    assert ck.load("factor_model")["weights"] == [0.5, 0.5]


def test_save_stamps_provenance(ck):
    out = ck.save("nbi_s", {"candidates": []})
    assert out["_complete"] is True
    assert out["_stage"] == "nbi_s"
    assert isinstance(out["_written_at"], float)


def test_a_truncated_checkpoint_is_not_done(ck):
    """A crash mid-write must not present a half artifact as complete."""
    ck.save("surrogates", {"terms": list(range(50))})
    p = ck.path("surrogates")
    text = p.read_text()
    p.write_text(text[: len(text) // 2])          # truncate, as a killed write would
    assert not ck.done("surrogates"), "a truncated checkpoint read as complete"


def test_a_checkpoint_without_the_complete_flag_is_not_done(ck):
    p = ck.path("ws_s")
    ck.root.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"candidates": [], "_stage": "ws_s"}))
    assert not ck.done("ws_s")


def test_an_explicitly_incomplete_checkpoint_is_not_done(ck):
    p = ck.path("ws_s")
    ck.root.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"_complete": False}))
    assert not ck.done("ws_s")


def test_no_tmp_file_survives_a_successful_save(ck):
    ck.save("metrics", {"hv": 1.0})
    assert not ck.path("metrics").with_suffix(".tmp").exists()
    assert list(ck.root.glob("*.tmp")) == []


def test_resume_reports_exactly_the_stages_already_written(ck):
    """The runner's skip decision is `done()` per stage; nothing else."""
    completed = STAGES[:6]
    for s in completed:
        ck.save(s, {"payload": s})
    assert [s for s in STAGES if ck.done(s)] == list(completed)
    assert [s for s in STAGES if not ck.done(s)] == list(STAGES[6:])


def test_a_resumed_checkpoint_returns_identical_content(tmp_path):
    """Reopening the directory in a new Checkpoint yields the same artifacts."""
    a = Checkpoint(tmp_path)
    payloads = {s: {"payload": s, "n": i} for i, s in enumerate(STAGES[:5])}
    for s, p in payloads.items():
        a.save(s, p)
    b = Checkpoint(tmp_path)                       # as a resumed process would
    for s, p in payloads.items():
        assert b.done(s)
        loaded = b.load(s)
        assert {k: v for k, v in loaded.items() if not k.startswith("_")} == p


def test_saving_a_stage_twice_overwrites_rather_than_appends(ck):
    ck.save("design", {"rows": 88})
    ck.save("design", {"rows": 88, "extra": True})
    assert ck.load("design")["extra"] is True
    assert len(list(ck.root.glob("design*"))) == 1


def test_every_declared_stage_has_a_distinct_checkpoint_path(ck):
    paths = {ck.path(s) for s in STAGES}
    assert len(paths) == len(STAGES)
