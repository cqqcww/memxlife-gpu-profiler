"""Preflight checks for profile-composed training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


MIB = 1024 * 1024


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ModuleNotFoundError:
        return False


def inspect_dataset_columns(config: ExperimentConfig) -> dict[str, Any]:
    if not config.data.dataset_name:
        return {
            "available_columns": ["text"],
            "text_field_found": True,
            "inspection_error": None,
        }
    if not _module_available("datasets"):
        return {
            "available_columns": [],
            "text_field_found": False,
            "inspection_error": "datasets package is not installed",
        }

    try:
        from datasets import load_dataset

        kwargs = {}
        if config.data.dataset_config:
            kwargs["name"] = config.data.dataset_config
        sample = load_dataset(
            config.data.dataset_name,
            **kwargs,
            split=f"{config.data.dataset_split}[:1]",
        )
        columns = list(getattr(sample, "column_names", []) or [])
        return {
            "available_columns": columns,
            "text_field_found": config.data.text_field in columns,
            "inspection_error": None,
        }
    except Exception as exc:  # pragma: no cover - depends on remote dataset/network state.
        return {
            "available_columns": [],
            "text_field_found": False,
            "inspection_error": f"{type(exc).__name__}: {exc}",
        }


def oom_recommendations(
    config: ExperimentConfig,
    *,
    parameter_count: int | None,
    cuda: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if not cuda.get("available"):
        return recommendations

    total_gb = cuda.get("memory_total_gb")
    if parameter_count and total_gb:
        # AdamW training memory is dominated by weights, gradients, and optimizer
        # states. This rough fp32 estimate is deliberately conservative.
        adamw_state_gb = parameter_count * 16 / 1_000_000_000
        if adamw_state_gb > 0.55 * float(total_gb):
            recommendations.append(
                "Estimated AdamW state is large relative to GPU memory before activations and temporary optimizer buffers; prefer preflight-only, a low-memory optimizer, offload, smaller seq_len/batch_size, or a smaller model."
            )
        elif adamw_state_gb > 0.35 * float(total_gb):
            recommendations.append(
                "Model state is a meaningful fraction of GPU memory; if CUDA OOM occurs, reduce batch_size before changing the model."
            )

    if config.trainer.batch_size >= 8:
        recommendations.append(
            "If CUDA OOM occurs at this batch size, try halving batch_size and using grad_accum_steps to preserve effective batch."
        )
    if config.trainer.seq_len >= 512:
        recommendations.append(
            "If CUDA OOM occurs at this sequence length, try seq_len=256 or 128 for the smoke run."
        )
    if config.trainer.grad_accum_steps > 1:
        recommendations.append(
            "Gradient accumulation lowers memory pressure but adds extra forward/backward passes, so throughput may trail a true larger batch."
        )
    return recommendations


def estimate_training_memory(
    config: ExperimentConfig,
    *,
    parameter_count: int | None,
    cuda: dict[str, Any],
) -> dict[str, Any]:
    """Estimate training memory with simple, auditable assumptions.

    The goal is not cycle-accurate prediction. It is a calibrated feasibility
    check that explains why AdamW is risky for large models and why a
    low-memory optimizer such as Adafactor changes the decision boundary.
    """

    tokens_per_step = (
        config.trainer.seq_len
        * config.trainer.batch_size
        * config.trainer.grad_accum_steps
    )
    if not parameter_count:
        return {
            "available": False,
            "reason": "parameter_count is unavailable",
            "tokens_per_step": tokens_per_step,
        }

    optimizer_name = config.optimizer.name.lower()
    parameter_mb = parameter_count * 4 / MIB
    gradient_mb = parameter_count * 4 / MIB
    if optimizer_name == "adamw":
        optimizer_state_mb = parameter_count * 8 / MIB
        optimizer_state_note = "AdamW keeps two fp32 moment tensors, roughly 8 bytes/parameter."
    elif optimizer_name == "adafactor":
        optimizer_state_mb = parameter_count * 0.5 / MIB
        optimizer_state_note = (
            "Adafactor uses factored second-moment state for large matrices; "
            "0.5 bytes/parameter is a coarse calibrated proxy."
        )
    else:
        optimizer_state_mb = parameter_count * 4 / MIB
        optimizer_state_note = "Unknown optimizer; assume one fp32 state tensor."

    activation_precision_factor = 2.0 if config.trainer.mixed_precision == "off" else 1.0
    checkpoint_factor = 0.7 if config.model.gradient_checkpointing else 1.0
    activation_mb = (
        tokens_per_step
        * (parameter_count / 1_000_000_000)
        * 1.5
        * activation_precision_factor
        * checkpoint_factor
    )
    predicted_allocated_peak_mb = parameter_mb + gradient_mb + optimizer_state_mb + activation_mb
    reserved_safety_factor = 1.25
    predicted_reserved_peak_mb = predicted_allocated_peak_mb * reserved_safety_factor
    memory_limit_mb = None
    if cuda.get("memory_total_gb"):
        memory_limit_mb = float(cuda["memory_total_gb"]) * 1024
    memory_ratio_reserved = (
        predicted_reserved_peak_mb / memory_limit_mb if memory_limit_mb else None
    )

    return {
        "available": True,
        "tokens_per_step": tokens_per_step,
        "optimizer": optimizer_name,
        "parameter_mb": parameter_mb,
        "gradient_mb": gradient_mb,
        "optimizer_state_mb": optimizer_state_mb,
        "activation_mb": activation_mb,
        "predicted_allocated_peak_mb": predicted_allocated_peak_mb,
        "predicted_reserved_peak_mb": predicted_reserved_peak_mb,
        "reserved_safety_factor": reserved_safety_factor,
        "memory_limit_mb": memory_limit_mb,
        "memory_ratio_reserved": memory_ratio_reserved,
        "assumptions": {
            "parameter_dtype": "fp32 model weights",
            "gradient_dtype": "fp32 gradients",
            "optimizer_state": optimizer_state_note,
            "activation_proxy": (
                "tokens_per_step * parameter_count_billions * 1.5 MB, "
                "scaled for mixed precision and gradient checkpointing"
            ),
            "reserved_safety_factor": (
                "1.25x proxy for allocator reservation, fragmentation, "
                "and temporary optimizer buffers"
            ),
        },
    }


def build_preflight_report(
    config: ExperimentConfig,
    project_root: Path,
    *,
    tokenizer: object | None = None,
    parameter_count: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    data_source = "huggingface_dataset" if config.data.dataset_name else "local_text"
    local_path = None
    local_path_exists = None
    if config.data.local_text_path:
        path = Path(config.data.local_text_path)
        if not path.is_absolute():
            path = project_root / path
        local_path = str(path)
        local_path_exists = path.exists()

    tokenizer_info = {}
    if tokenizer is not None:
        tokenizer_info = {
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
        }

    warnings: list[str] = []
    if data_source == "local_text" and not local_path_exists:
        warnings.append("local_text_path does not exist")
    if data_source == "huggingface_dataset" and not _module_available("datasets"):
        warnings.append("datasets package is not installed")
    if not _module_available("transformers"):
        warnings.append("transformers package is not installed")
    if config.trainer.compile_model:
        warnings.append("compile_model can improve throughput but may add first-step overhead")
    if config.model.trust_remote_code:
        warnings.append("trust_remote_code is enabled; this is acceptable for stretch profiles but should be explicit")

    cuda = {"available": False}
    if _module_available("torch"):
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cuda.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                    "device_count": torch.cuda.device_count(),
                    "memory_total_gb": props.total_memory / (1024**3),
                }
            )
    elif config.trainer.device in {"auto", "cuda"}:
        warnings.append("torch package is not installed")

    dataset_inspection = inspect_dataset_columns(config)
    if dataset_inspection.get("inspection_error"):
        warnings.append(f"dataset column inspection failed: {dataset_inspection['inspection_error']}")
    elif not dataset_inspection.get("text_field_found"):
        warnings.append(
            f"data.text_field={config.data.text_field!r} was not found in inspected dataset columns"
        )

    tokens_per_step = (
        config.trainer.seq_len
        * config.trainer.batch_size
        * config.trainer.grad_accum_steps
    )
    memory_estimate = estimate_training_memory(config, parameter_count=parameter_count, cuda=cuda)
    recommendations = oom_recommendations(config, parameter_count=parameter_count, cuda=cuda)
    memory_ratio = memory_estimate.get("memory_ratio_reserved")
    if isinstance(memory_ratio, (int, float)):
        if memory_ratio >= 1.0:
            recommendations.append(
                "Memory predictor estimates this configuration can exceed available CUDA memory; use preflight-only, reduce token budget, or switch to a lower-memory optimizer."
            )
        elif memory_ratio >= 0.8:
            recommendations.append(
                "Memory predictor estimates high CUDA pressure; run a short probe before longer training and watch reserved memory."
            )
    return {
        "metadata": config.metadata.__dict__,
        "model": {
            "name_or_path": config.model.name_or_path,
            "tokenizer_name": config.model.tokenizer_name or config.model.name_or_path,
            "from_pretrained": config.model.from_pretrained,
            "trust_remote_code": config.model.trust_remote_code,
            "gradient_checkpointing": config.model.gradient_checkpointing,
            "parameter_count": parameter_count,
        },
        "data": {
            "source": data_source,
            "dataset_name": config.data.dataset_name,
            "dataset_config": config.data.dataset_config,
            "dataset_split": config.data.dataset_split,
            "text_field": config.data.text_field,
            "local_text_path": local_path,
            "local_text_path_exists": local_path_exists,
            "max_samples": config.data.max_samples,
            "use_cache": config.data.use_cache,
            "cache_dir": config.data.cache_dir,
            "available_columns": dataset_inspection.get("available_columns", []),
            "text_field_found": dataset_inspection.get("text_field_found"),
            "inspection_error": dataset_inspection.get("inspection_error"),
        },
        "trainer": {
            "seq_len": config.trainer.seq_len,
            "batch_size": config.trainer.batch_size,
            "grad_accum_steps": config.trainer.grad_accum_steps,
            "tokens_per_step": tokens_per_step,
            "max_steps": config.trainer.max_steps,
            "mixed_precision": config.trainer.mixed_precision,
            "device": str(device or config.trainer.device),
            "compile_model": config.trainer.compile_model,
        },
        "optimizer": {
            "name": config.optimizer.name,
            "lr": config.optimizer.lr,
            "weight_decay": config.optimizer.weight_decay,
        },
        "memory_estimate": memory_estimate,
        "tokenizer": tokenizer_info,
        "environment": {
            "torch_available": _module_available("torch"),
            "transformers_available": _module_available("transformers"),
            "datasets_available": _module_available("datasets"),
            "cuda": cuda,
        },
        "warnings": warnings,
        "recommendations": recommendations,
    }


def render_preflight_markdown(report: dict[str, Any]) -> str:
    metadata = report.get("metadata", {})
    model = report.get("model", {})
    data = report.get("data", {})
    trainer = report.get("trainer", {})
    memory = report.get("memory_estimate", {})
    env = report.get("environment", {})
    warnings = report.get("warnings", [])
    recommendations = report.get("recommendations", [])
    lines = [
        "# Preflight Report",
        "",
        "## Profiles",
        "",
        f"- Base config: `{metadata.get('base_config') or 'direct config'}`",
        f"- Model profile: `{metadata.get('model_profile') or 'none'}`",
        f"- Data profile: `{metadata.get('data_profile') or 'none'}`",
        f"- Notes: {metadata.get('notes') or 'none'}",
        "",
        "## Model And Data",
        "",
        f"- Model: `{model.get('name_or_path')}`",
        f"- Parameters: `{model.get('parameter_count')}`",
        f"- Data source: `{data.get('source')}`",
        f"- Dataset/local path: `{data.get('dataset_name') or data.get('local_text_path')}`",
        f"- Available columns: `{data.get('available_columns') or []}`",
        f"- Text field found: `{data.get('text_field_found')}`",
        f"- Cache enabled: `{data.get('use_cache')}`",
        "",
        "## Runtime",
        "",
        f"- Device: `{trainer.get('device')}`",
        f"- Mixed precision: `{trainer.get('mixed_precision')}`",
        f"- Tokens per optimizer step: `{trainer.get('tokens_per_step')}`",
        f"- CUDA available: `{env.get('cuda', {}).get('available')}`",
        "",
        "## Memory Estimate",
        "",
    ]
    if memory.get("available"):
        ratio = memory.get("memory_ratio_reserved")
        ratio_text = f"{ratio:.1%}" if isinstance(ratio, (int, float)) else "n/a"
        lines.extend(
            [
                f"- Parameter memory: `{memory.get('parameter_mb'):.0f} MiB`",
                f"- Gradient memory: `{memory.get('gradient_mb'):.0f} MiB`",
                f"- Optimizer state: `{memory.get('optimizer_state_mb'):.0f} MiB`",
                f"- Activation proxy: `{memory.get('activation_mb'):.0f} MiB`",
                f"- Predicted allocated peak: `{memory.get('predicted_allocated_peak_mb'):.0f} MiB`",
                f"- Predicted reserved peak: `{memory.get('predicted_reserved_peak_mb'):.0f} MiB`",
                f"- Reserved / device memory: `{ratio_text}`",
            ]
        )
    else:
        lines.append(f"- unavailable: {memory.get('reason') or 'unknown'}")
    lines.extend(
        [
            "",
        "## Recommendations",
        "",
        ]
    )
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_preflight_report(
    config: ExperimentConfig,
    project_root: Path,
    run_dir: Path,
    *,
    tokenizer: object | None = None,
    parameter_count: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    report = build_preflight_report(
        config,
        project_root,
        tokenizer=tokenizer,
        parameter_count=parameter_count,
        device=device,
    )
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "preflight.md").write_text(render_preflight_markdown(report), encoding="utf-8")
    return report
