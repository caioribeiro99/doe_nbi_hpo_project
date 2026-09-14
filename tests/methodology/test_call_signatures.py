"""Every intra-package call must bind against the callee's signature.

A stage that cannot be *called* is indistinguishable from a stage that was never
written, and it fails late: ``_run_historical`` was reached only after the split,
design, factor model, surrogates and external validation had all executed, which
in the confirmatory campaign is hours of compute per unit before the traceback.

The regression this pins down is real: the V0-V9 pass added a keyword-only
``symmetric_grid`` parameter to ``_run_historical`` and forwarded it to
``run_historical_ws``, which never accepted it, while the runner's own call site
never supplied it. Three mutually inconsistent signatures, 235 passing tests, and
a campaign that stopped at stage six.

This walks the AST of every campaign module and binds each call to a function
defined inside the package against ``inspect.signature``. It checks arity and
keyword names only -- never types, which is the type checker's job and not a
methodological claim.
"""
from __future__ import annotations

import ast
import inspect
import importlib
import pathlib

import pytest

PKG = "doe_xgb.campaign"
ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "doe_xgb" / "campaign"
MODULES = sorted(p.stem for p in ROOT.glob("*.py") if p.stem != "__init__")


class _Any:
    """A stand-in argument. Only arity and keyword names are under test."""


def _resolve(mod, name: str):
    """The object a bare ``name`` refers to in ``mod``, or None if not a function."""
    obj = getattr(mod, name, None)
    if obj is None or not (inspect.isfunction(obj) or inspect.isclass(obj)):
        return None
    # Only functions that live in this package: stdlib and third-party callees are
    # somebody else's contract.
    home = getattr(obj, "__module__", "")
    return obj if home.startswith(PKG) else None


def _calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            yield node


@pytest.mark.parametrize("modname", MODULES)
def test_intra_package_calls_bind(modname):
    mod = importlib.import_module(f"{PKG}.{modname}")
    src = (ROOT / f"{modname}.py").read_text()
    tree = ast.parse(src)

    checked = 0
    for node in _calls(tree):
        target = _resolve(mod, node.func.id)
        if target is None:
            continue
        # A splat hides the true arity from static inspection; skip rather than
        # pretend to have checked it.
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue
        if any(kw.arg is None for kw in node.keywords):
            continue

        pos = [_Any()] * len(node.args)
        kws = {kw.arg: _Any() for kw in node.keywords}
        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            # Builtin-derived classes (e.g. an Exception subclass) expose no
            # introspectable signature. Nothing to bind against.
            continue
        try:
            sig.bind(*pos, **kws)
        except TypeError as exc:
            pytest.fail(
                f"{modname}.py:{node.lineno}: call to {node.func.id}("
                f"{len(pos)} positional, {sorted(kws)}) does not bind against "
                f"{node.func.id}{sig}: {exc}")
        checked += 1

    # A test that silently checks nothing is worse than no test.
    if modname in {"runner", "arms", "scoring"}:
        assert checked > 0, f"no intra-package calls found in {modname}.py"


def test_the_detector_catches_a_real_mismatch():
    """The check must fail on the exact shape of the bug it exists to prevent."""
    def callee(a, b, *, flag: bool) -> None:
        ...

    sig = inspect.signature(callee)
    sig.bind(_Any(), _Any(), flag=_Any())            # correct call binds
    with pytest.raises(TypeError):
        sig.bind(_Any(), _Any())                     # the omitted keyword-only arg
    with pytest.raises(TypeError):
        sig.bind(_Any(), _Any(), flag=_Any(), extra=_Any())   # the unknown keyword
