"""Optimizer construction."""

from __future__ import annotations

from .config import OptimizerConfig


def build_optimizer(model, config: OptimizerConfig):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required on the remote training server") from exc

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if name.endswith(".bias") or "norm" in lowered or "ln_" in lowered:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    name = config.name.lower()
    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for optimizer.name=adafactor") from exc
        return Adafactor(
            groups,
            lr=config.lr,
            eps=(1e-30, config.eps),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=config.weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )

    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {config.name}")

    return torch.optim.AdamW(
        groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )
