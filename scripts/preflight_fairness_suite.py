#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from doe_xgb.fairness_dataset_utils import (  # noqa: E402
    load_bank_dataset,
    load_credit_card_default_dataset,
    load_generic_fairness_dataset,
)


def _load_cfg(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _min_joint_stratum(y: pd.Series, protected: pd.Series) -> int:
    joint = pd.DataFrame({"y": y.astype(int), "protected": protected.astype(int)})
    return int(joint.value_counts().min()) if not joint.empty else 0


def _load_dataset(cfg: Dict[str, Any]):
    dataset_kind = str(cfg.get("dataset_kind", "generic"))
    path = Path(str(cfg["path"]))
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    if dataset_kind == "bank":
        return load_bank_dataset(path)
    if dataset_kind == "credit_card_default":
        return load_credit_card_default_dataset(
            path,
            protected_attr_mode=str(cfg.get("protected_attr_mode", "sex_male_is_1")),
            target_positive=str(cfg.get("target_positive", "1")),
        )
    if dataset_kind == "generic":
        return load_generic_fairness_dataset(
            path,
            target_col=str(cfg.get("target_col", "y")),
            target_positive=str(cfg.get("target_positive")) if cfg.get("target_positive") is not None else None,
            protected_col=str(cfg.get("protected_col")) if cfg.get("protected_col") is not None else None,
            protected_attr_mode=str(cfg.get("protected_attr_mode", "binary_one_is_privileged")),
            drop_unknown_rows=bool(cfg.get("drop_unknown_rows", False)),
        )
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


def main() -> None:
    p = argparse.ArgumentParser(description="Preflight validation for the 3-base finance fairness suite.")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "fairness_suite_3bases_finance_r30.json"))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config).resolve())
    rows: List[Dict[str, Any]] = []
    failed = 0

    for dcfg in cfg.get("datasets", []):
        name = str(dcfg["name"])
        path = Path(str(dcfg["path"]))
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        row: Dict[str, Any] = {
            "dataset": name,
            "path": str(path),
            "dataset_kind": str(dcfg.get("dataset_kind", "generic")),
        }
        try:
            X, y, protected = _load_dataset(dcfg)
            y_s = pd.Series(y).astype(int)
            p_s = pd.Series(protected).astype(int)
            row.update(
                {
                    "status": "OK",
                    "n_rows": int(len(X)),
                    "n_features": int(X.shape[1]),
                    "y0": int((y_s == 0).sum()),
                    "y1": int((y_s == 1).sum()),
                    "protected0": int((p_s == 0).sum()),
                    "protected1": int((p_s == 1).sum()),
                    "min_joint_stratum": int(_min_joint_stratum(y_s, p_s)),
                    "kfold_safe": bool(_min_joint_stratum(y_s, p_s) >= int(args.n_splits)),
                    "na_features": int(pd.DataFrame(X).isna().sum().sum()),
                }
            )
            if not row["kfold_safe"]:
                row["status"] = "FAIL"
                row["error"] = f"Smallest joint stratum < n_splits ({args.n_splits})"
                failed += 1
        except Exception as e:
            row.update({"status": "FAIL", "error": str(e)})
            failed += 1
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"saved preflight -> {out}")

    if failed:
        raise SystemExit(1)
    print("✅ Preflight passed.")


if __name__ == "__main__":
    main()
