"""Timing helpers for throughput and phase breakdowns."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


def _cuda_sync_if_needed(device: str | None = None) -> None:
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            torch.cuda.synchronize()
    except Exception:
        return


@dataclass
class StepTimer:
    device: str | None = None
    totals: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str):
        _cuda_sync_if_needed(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            _cuda_sync_if_needed(self.device)
            elapsed = time.perf_counter() - start
            self.totals[name] = self.totals.get(name, 0.0) + elapsed

    def snapshot_and_reset(self) -> dict[str, float]:
        values = dict(self.totals)
        self.totals.clear()
        return values
