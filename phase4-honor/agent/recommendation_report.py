"""Generate a compact recommendation section from probe/stability evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_json(path: str | Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return json.loads(resolved.read_text(encoding="utf-8"))


def _resolve_run_dir(run_dir: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(run_dir)
    if path.exists():
        return path
    if path.is_absolute() and len(path.parts) > 1 and path.parts[1] == "workspace":
        candidate = project_root / Path(*path.parts[2:])
        if candidate.exists():
            return candidate
    if not path.is_absolute():
        candidate = project_root / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve run directory: {run_dir}")


def memory_calibration(preflight: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    estimate = preflight.get("memory_estimate") or {}
    actual = summary.get("cuda_memory") or {}
    predicted_reserved = _number(estimate.get("predicted_reserved_peak_mb"))
    predicted_allocated = _number(estimate.get("predicted_allocated_peak_mb"))
    actual_reserved = _number(actual.get("cuda_peak_reserved_mb"))
    actual_allocated = _number(actual.get("cuda_peak_allocated_mb"))
    reserved_error_pct = None
    allocated_error_pct = None
    if predicted_reserved is not None and actual_reserved:
        reserved_error_pct = (predicted_reserved - actual_reserved) / actual_reserved * 100
    if predicted_allocated is not None and actual_allocated:
        allocated_error_pct = (predicted_allocated - actual_allocated) / actual_allocated * 100
    return {
        "predicted_allocated_peak_mb": predicted_allocated,
        "actual_allocated_peak_mb": actual_allocated,
        "allocated_error_pct": allocated_error_pct,
        "predicted_reserved_peak_mb": predicted_reserved,
        "actual_reserved_peak_mb": actual_reserved,
        "reserved_error_pct": reserved_error_pct,
        "memory_ratio_reserved": estimate.get("memory_ratio_reserved"),
        "assumptions": estimate.get("assumptions", {}),
    }


def build_recommendation_section(
    *,
    probe_payload: dict[str, Any],
    stability_payload: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    probe_recommendation = probe_payload.get("recommendation") or {}
    stability = stability_payload.get("stability") or {}
    run_dir = _resolve_run_dir(stability["run_dir"], project_root)
    summary = _load_json(run_dir / "summary.json", project_root)
    preflight = _load_json(run_dir / "preflight.json", project_root)
    model = preflight.get("model", {})
    trainer = preflight.get("trainer", {})
    optimizer = preflight.get("optimizer", {})
    calibration = memory_calibration(preflight, summary)
    status = stability.get("status")
    completed = stability.get("completed_train_steps")
    requested = stability.get("requested_steps")
    reasons = []
    if status != "pass":
        reasons.append("stability runner did not pass")
    if completed != requested:
        reasons.append("not all requested steps completed")
    if not reasons:
        reasons.append("50-step real-data stability passed without OOM or NaN")

    return {
        "title": "Current Phase 4 Recommendation",
        "recommended_config": {
            "model": model.get("name_or_path"),
            "data_profile": preflight.get("metadata", {}).get("data_profile"),
            "optimizer": optimizer.get("name"),
            "tokens_per_step": trainer.get("tokens_per_step"),
            "seq_len": trainer.get("seq_len"),
            "batch_size": trainer.get("batch_size"),
            "grad_accum_steps": trainer.get("grad_accum_steps"),
            "mixed_precision": trainer.get("mixed_precision"),
            "gradient_checkpointing": model.get("gradient_checkpointing"),
            "checkpointing": "disabled for probe/stability runs",
        },
        "evidence": {
            "probe_action": probe_recommendation.get("action"),
            "probe_reason": probe_recommendation.get("reason"),
            "stability_status": status,
            "completed_steps": completed,
            "requested_steps": requested,
            "avg_tokens_per_sec": stability.get("avg_tokens_per_sec"),
            "first_train_loss": stability.get("first_train_loss"),
            "last_train_loss": stability.get("last_train_loss"),
            "last_val_loss": stability.get("last_val_loss"),
            "cuda_peak_allocated_mb": stability.get("cuda_peak_allocated_mb"),
            "cuda_peak_reserved_mb": stability.get("cuda_peak_reserved_mb"),
        },
        "memory_calibration": calibration,
        "decision": {
            "promote_for_longer_probe": status == "pass",
            "reasons": reasons,
            "risks": [
                "50 steps is feasibility evidence, not convergence evidence.",
                "WikiText-2 subset is still small; longer runs are needed for stability confidence.",
                "Reserved memory is high enough that larger token budgets should be gated by probes.",
            ],
            "next_steps": [
                "Run 100-200 step real-data stability before calling this a durable training recipe.",
                "Keep AdamW classified as unsafe on this GPU unless offload/sharding is introduced.",
                "Use the calibrated memory estimate as the first preflight gate for new model profiles.",
            ],
        },
    }


def _fmt(value: Any, digits: int = 2) -> str:
    number = _number(value)
    return f"{number:.{digits}f}" if number is not None else "n/a"


def render_markdown(section: dict[str, Any]) -> str:
    cfg = section["recommended_config"]
    evidence = section["evidence"]
    memory = section["memory_calibration"]
    decision = section["decision"]
    lines = [
        f"# {section['title']}",
        "",
        "## Recommended Configuration",
        "",
        f"- Model: `{cfg.get('model')}`",
        f"- Data profile: `{cfg.get('data_profile')}`",
        f"- Optimizer: `{cfg.get('optimizer')}`",
        f"- Tokens/step: `{cfg.get('tokens_per_step')}`",
        f"- Shape: `seq_len={cfg.get('seq_len')}, batch={cfg.get('batch_size')}, grad_accum={cfg.get('grad_accum_steps')}`",
        f"- Mixed precision: `{cfg.get('mixed_precision')}`",
        f"- Gradient checkpointing: `{cfg.get('gradient_checkpointing')}`",
        "",
        "## Evidence",
        "",
        f"- Stability status: `{evidence.get('stability_status')}`",
        f"- Completed steps: `{evidence.get('completed_steps')}/{evidence.get('requested_steps')}`",
        f"- Average tokens/sec: `{_fmt(evidence.get('avg_tokens_per_sec'))}`",
        f"- Train loss: `{_fmt(evidence.get('first_train_loss'), 4)} -> {_fmt(evidence.get('last_train_loss'), 4)}`",
        f"- Last validation loss: `{_fmt(evidence.get('last_val_loss'), 4)}`",
        f"- Peak CUDA allocated/reserved: `{_fmt(evidence.get('cuda_peak_allocated_mb'), 0)} / {_fmt(evidence.get('cuda_peak_reserved_mb'), 0)} MiB`",
        "",
        "## Memory Calibration",
        "",
        f"- Predicted allocated peak: `{_fmt(memory.get('predicted_allocated_peak_mb'), 0)} MiB`",
        f"- Actual allocated peak: `{_fmt(memory.get('actual_allocated_peak_mb'), 0)} MiB`",
        f"- Allocated prediction error: `{_fmt(memory.get('allocated_error_pct'), 1)}%`",
        f"- Predicted reserved peak: `{_fmt(memory.get('predicted_reserved_peak_mb'), 0)} MiB`",
        f"- Actual reserved peak: `{_fmt(memory.get('actual_reserved_peak_mb'), 0)} MiB`",
        f"- Reserved prediction error: `{_fmt(memory.get('reserved_error_pct'), 1)}%`",
        "",
        "## Decision",
        "",
        f"- Promote for longer probe: `{decision.get('promote_for_longer_probe')}`",
    ]
    lines.extend(f"- Reason: {reason}" for reason in decision.get("reasons", []))
    lines.extend(f"- Risk: {risk}" for risk in decision.get("risks", []))
    lines.extend(f"- Next: {step}" for step in decision.get("next_steps", []))
    return "\n".join(lines) + "\n"


def write_recommendation_report(
    *,
    probe_path: str | Path,
    stability_path: str | Path,
    output_base: str | Path = "runs/recommendations/phase4-current-recommendation",
    project_root: Path = PROJECT_ROOT,
) -> dict[str, str]:
    probe_payload = _load_json(probe_path, project_root)
    stability_payload = _load_json(stability_path, project_root)
    section = build_recommendation_section(
        probe_payload=probe_payload,
        stability_payload=stability_payload,
        project_root=project_root,
    )
    out_base = Path(output_base)
    if not out_base.is_absolute():
        out_base = project_root / out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    json_path.write_text(json.dumps(section, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(section), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        default="runs/recommendations/deepseek_adafactor_wikitext_realdata-latest.json",
    )
    parser.add_argument(
        "--stability",
        default="runs/recommendations/deepseek_adafactor_wikitext_realdata-stability-latest.json",
    )
    parser.add_argument(
        "--output-base",
        default="runs/recommendations/phase4-current-recommendation",
    )
    args = parser.parse_args(argv)
    artifacts = write_recommendation_report(
        probe_path=args.probe,
        stability_path=args.stability,
        output_base=args.output_base,
    )
    print(json.dumps({"artifacts": artifacts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
