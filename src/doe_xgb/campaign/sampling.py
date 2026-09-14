"""System sampling during a batch: CPU, memory, swap and load.

Separate from the evaluator so that the campaign runner and the Stage B benchmark
observe the machine the same way.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import psutil


class Sampler:
    """Samples system state on a background thread for the duration of a block."""

    def __init__(self, period: float = 0.5) -> None:
        self.period = period
        self.samples: list[dict[str, float]] = []
        self._stop = False
        self._t: threading.Thread | None = None

    def __enter__(self) -> "Sampler":
        psutil.cpu_percent(interval=None)          # prime the delta
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self) -> None:
        proc = psutil.Process()
        while not self._stop:
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                self.samples.append({"cpu": psutil.cpu_percent(interval=None),
                                     "rss": float(rss),
                                     "swap_used": float(psutil.swap_memory().used),
                                     "load1": os.getloadavg()[0]})
            except Exception:
                pass
            time.sleep(self.period)

    def __exit__(self, *exc: Any) -> None:
        self._stop = True
        if self._t is not None:
            self._t.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        s = self.samples
        return {"cpu_mean_pct": round(float(np.mean([x["cpu"] for x in s])), 1),
                "cpu_max_pct": round(float(np.max([x["cpu"] for x in s])), 1),
                "peak_rss_gb": round(float(np.max([x["rss"] for x in s])) / 2**30, 2),
                "swap_used_delta_mb": round(
                    (s[-1]["swap_used"] - s[0]["swap_used"]) / 2**20, 1),
                "load1_max": round(float(np.max([x["load1"] for x in s])), 1),
                "samples": len(s)}


__all__ = ["Sampler"]
