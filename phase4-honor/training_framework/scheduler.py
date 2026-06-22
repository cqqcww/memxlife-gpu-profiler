"""Learning-rate scheduler construction."""

from __future__ import annotations

import math

from .config import SchedulerConfig


def build_scheduler(optimizer, config: SchedulerConfig, max_steps: int):
    try:
        from torch.optim.lr_scheduler import LambdaLR
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required on the remote training server") from exc

    name = config.name.lower()
    warmup = max(0, config.warmup_steps)
    min_ratio = max(0.0, min(1.0, config.min_lr_ratio))

    if name == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)

    if name != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler: {config.name}")

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-8, float(step + 1) / float(warmup))
        denom = max(1, max_steps - warmup)
        progress = min(1.0, max(0.0, float(step - warmup) / float(denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
