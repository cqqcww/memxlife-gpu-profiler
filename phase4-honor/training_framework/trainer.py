"""Explicit training and validation loop."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .checkpoint import CheckpointManager
from .config import ExperimentConfig, dump_config
from .data import build_dataloaders
from .logger import RunLogger
from .model import build_model_and_tokenizer
from .optim import build_optimizer
from .preflight import write_preflight_report
from .scheduler import build_scheduler
from .timing import StepTimer


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def autocast_context(device, mode: str):
    import contextlib
    import torch

    if device.type != "cuda":
        return contextlib.nullcontext()
    if mode == "off":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def cuda_memory_metrics(device) -> dict[str, float]:
    """Return lightweight CUDA memory metrics in MiB when CUDA is active."""

    if getattr(device, "type", None) != "cuda":
        return {}
    import torch

    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "cuda_reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
    }


class Trainer:
    def __init__(self, config: ExperimentConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.run_dir = project_root / config.trainer.output_dir / f"{config.trainer.run_name}-{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, self.run_dir / "copied_config.yaml")

    def fit(self) -> dict:
        import torch

        cfg = self.config
        device = resolve_device(cfg.trainer.device)
        bundle = build_model_and_tokenizer(cfg.model)
        model = bundle.model.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if cfg.trainer.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)
        write_preflight_report(
            cfg,
            self.project_root,
            self.run_dir,
            tokenizer=bundle.tokenizer,
            parameter_count=bundle.parameter_count,
            device=str(device),
        )

        train_loader, val_loader, data_meta = build_dataloaders(
            cfg.data,
            bundle.tokenizer,
            cfg.trainer.seq_len,
            cfg.trainer.batch_size,
            self.project_root,
        )
        optimizer = build_optimizer(model, cfg.optimizer)
        scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.trainer.max_steps)
        logger = RunLogger(
            self.run_dir,
            cfg.logging.tensorboard,
            cfg.logging.jsonl,
            cfg.logging.console,
        )
        ckpt = CheckpointManager(
            self.run_dir,
            keep_last=cfg.checkpoint.keep_last,
            save_rng=cfg.checkpoint.save_rng,
        )

        global_step = 0
        resume_info = None
        if cfg.trainer.resume_from:
            payload = ckpt.load(cfg.trainer.resume_from, model, optimizer, scheduler, map_location=device)
            global_step = int(payload.get("global_step", 0))
            scheduler_lrs = scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else []
            resume_info = {
                "resumed_from": cfg.trainer.resume_from,
                "loaded_checkpoint_path": payload.get("_loaded_checkpoint_path"),
                "global_step": global_step,
                "optimizer_lr": optimizer.param_groups[0]["lr"],
                "scheduler_lrs": scheduler_lrs,
                "checkpoint_extra": payload.get("extra", {}),
                "rng_restored": bool(payload.get("rng")),
            }
            logger.log(
                "resume",
                global_step,
                {
                    "global_step": global_step,
                    "optimizer_lr": optimizer.param_groups[0]["lr"],
                    "rng_restored": 1 if payload.get("rng") else 0,
                },
            )

        logger.log(
            "setup",
            global_step,
            {
                "parameters": bundle.parameter_count,
                "train_blocks": data_meta["train_blocks"],
                "val_blocks": data_meta["val_blocks"],
                "cache_hit": 1 if data_meta.get("cache") == "hit" else 0,
                **cuda_memory_metrics(device),
            },
        )

        timer = StepTimer(device=str(device))
        model.train()
        train_iter = iter(train_loader)
        last_loss = math.nan
        tokens_per_step = cfg.trainer.batch_size * cfg.trainer.seq_len * cfg.trainer.grad_accum_steps

        while global_step < cfg.trainer.max_steps:
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(cfg.trainer.grad_accum_steps):
                with timer.measure("data_s"):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with timer.measure("forward_s"):
                    with autocast_context(device, cfg.trainer.mixed_precision):
                        outputs = model(**batch)
                        loss = outputs.loss / cfg.trainer.grad_accum_steps
                with timer.measure("backward_s"):
                    loss.backward()
                accum_loss += float(loss.detach().cpu())

            with timer.measure("optimizer_s"):
                if cfg.trainer.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip_norm)
                optimizer.step()
                scheduler.step()

            global_step += 1
            last_loss = accum_loss
            elapsed = time.perf_counter() - step_start
            timings = timer.snapshot_and_reset()
            metrics = {
                "loss": last_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "step_time_s": elapsed,
                "tokens_per_sec": tokens_per_step / max(elapsed, 1e-9),
                **timings,
                **cuda_memory_metrics(device),
            }
            if global_step % cfg.trainer.log_every_steps == 0:
                logger.log("train", global_step, metrics)
            if global_step % cfg.trainer.validate_every_steps == 0:
                val_metrics = self.validate(model, val_loader, device)
                logger.log("validate", global_step, val_metrics)
            if cfg.checkpoint.enabled and global_step % cfg.checkpoint.save_every_steps == 0:
                path = ckpt.save(
                    global_step,
                    model,
                    optimizer,
                    scheduler,
                    extra={"loss": last_loss, "lr": optimizer.param_groups[0]["lr"]},
                )
                logger.log("checkpoint", global_step, {"saved": str(path)})

        final_ckpt = None
        if cfg.checkpoint.enabled:
            final_ckpt = ckpt.save(global_step, model, optimizer, scheduler, extra={"final": True, "loss": last_loss})
        summary = {
            "run_dir": str(self.run_dir),
            "global_step": global_step,
            "final_loss": last_loss,
            "final_lr": optimizer.param_groups[0]["lr"],
            "final_checkpoint": str(final_ckpt) if final_ckpt is not None else None,
            "data": data_meta,
            "parameters": bundle.parameter_count,
            "resume": resume_info,
            "preflight": str(self.run_dir / "preflight.json"),
            "cuda_memory": cuda_memory_metrics(device),
        }
        ckpt.write_metadata(summary)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._write_summary_md(summary)
        logger.log("done", global_step, {"final_loss": last_loss})
        logger.close()
        return summary

    def preflight_only(self) -> dict:
        """Build model/tokenizer and write preflight artifacts without training."""

        import torch

        cfg = self.config
        device = resolve_device(cfg.trainer.device)
        bundle = build_model_and_tokenizer(cfg.model)
        model = bundle.model.to(device)
        if cfg.trainer.compile_model and hasattr(torch, "compile"):
            # Do not compile during preflight; record the warning but avoid first-step cost.
            pass
        report = write_preflight_report(
            cfg,
            self.project_root,
            self.run_dir,
            tokenizer=bundle.tokenizer,
            parameter_count=bundle.parameter_count,
            device=str(device),
        )
        summary = {
            "run_dir": str(self.run_dir),
            "mode": "preflight_only",
            "global_step": 0,
            "parameters": bundle.parameter_count,
            "preflight": str(self.run_dir / "preflight.json"),
            "warnings": report.get("warnings", []),
            "recommendations": report.get("recommendations", []),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._write_preflight_summary_md(summary)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return summary

    def validate(self, model, val_loader, device) -> dict:
        import torch

        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                outputs = model(**batch)
                losses.append(float(outputs.loss.detach().cpu()))
        model.train()
        val_loss = sum(losses) / max(1, len(losses))
        return {
            "loss": val_loss,
            "perplexity": math.exp(min(20.0, val_loss)),
            **cuda_memory_metrics(device),
        }

    def _write_summary_md(self, summary: dict) -> None:
        lines = [
            f"# Run Summary: {self.config.trainer.run_name}",
            "",
            f"- Run directory: `{summary['run_dir']}`",
            f"- Global step: `{summary['global_step']}`",
            f"- Final loss: `{summary['final_loss']:.6f}`",
            f"- Final LR: `{summary['final_lr']:.8g}`",
            f"- Parameters: `{summary['parameters']}`",
            f"- Final checkpoint: `{summary['final_checkpoint']}`",
            f"- Preflight: `{summary['preflight']}`",
            f"- Data cache: `{summary['data'].get('cache')}`",
            f"- Token blocks: `{summary['data'].get('num_blocks')}`",
            "",
            "This summary is generated by the Phase 4 mini training framework.",
        ]
        (self.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_preflight_summary_md(self, summary: dict) -> None:
        lines = [
            f"# Preflight Summary: {self.config.trainer.run_name}",
            "",
            f"- Run directory: `{summary['run_dir']}`",
            "- Mode: `preflight_only`",
            f"- Parameters: `{summary['parameters']}`",
            f"- Preflight: `{summary['preflight']}`",
            "",
            "## Recommendations",
            "",
        ]
        recommendations = summary.get("recommendations") or []
        lines.extend(f"- {item}" for item in recommendations) if recommendations else lines.append("- none")
        lines.extend(["", "## Warnings", ""])
        warnings = summary.get("warnings") or []
        lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- none")
        (self.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
