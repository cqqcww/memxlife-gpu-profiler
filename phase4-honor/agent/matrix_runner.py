"""Run small, auditable experiment matrices."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.analyzer import render_markdown, summarize_run
from agent.ledger import append_ledger
from agent.runner import latest_run
from training_framework.config import load_yaml_dict
from training_framework.config_merge import resolve_config_dict, set_dotted_value, write_resolved_config


def load_matrix(path: str | Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    matrix_path = Path(path)
    if not matrix_path.is_absolute():
        matrix_path = project_root / matrix_path
    matrix = load_yaml_dict(matrix_path)
    if not matrix.get("name"):
        raise ValueError("matrix.name is required")
    if not matrix.get("base_config"):
        raise ValueError("matrix.base_config is required")
    for key in ("model_profile", "data_profile"):
        value = matrix.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"matrix.{key} must be a path string when provided")
    variants = matrix.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("matrix.variants must be a non-empty list")
    return matrix


def expand_variants(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    expanded = []
    for item in matrix["variants"]:
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("each matrix variant must have a name")
        overrides = item.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"variant {item['name']} overrides must be a mapping")
        expanded.append(
            {
                "name": str(item["name"]),
                "overrides": dict(overrides),
                "preflight_only": bool(item.get("preflight_only", False)),
            }
        )
    return expanded


def materialize_variant_config(
    matrix: dict[str, Any],
    variant: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    raw = resolve_config_dict(
        matrix["base_config"],
        model_profile=matrix.get("model_profile"),
        data_profile=matrix.get("data_profile"),
        project_root=project_root,
    )
    for key, value in variant["overrides"].items():
        set_dotted_value(raw, str(key), value)
    out_dir = project_root / "runs" / "matrix_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{matrix['name']}-{variant['name']}.json"
    return write_resolved_config(raw, out_path)


def classify_failure(returncode: int, output: str) -> dict[str, str] | None:
    if returncode == 0:
        return None
    lowered = output.lower()
    if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        return {
            "failure_kind": "cuda_oom",
            "failure_reason": (
                "CUDA OOM during run. For large models this often means optimizer "
                "state or temporary optimizer buffers exceeded GPU memory."
            ),
        }
    if "modulenotfounderror" in lowered:
        return {
            "failure_kind": "missing_dependency",
            "failure_reason": "A required Python package was missing in the runtime.",
        }
    if "runtimeerror" in lowered:
        return {
            "failure_kind": "runtime_error",
            "failure_reason": "RuntimeError occurred; inspect subprocess log for details.",
        }
    return {
        "failure_kind": "nonzero_exit",
        "failure_reason": f"Training command exited with status {returncode}.",
    }


def run_config(
    config_path: Path,
    project_root: Path = PROJECT_ROOT,
    *,
    log_path: Path | None = None,
    preflight_only: bool = False,
) -> tuple[int, str]:
    cmd = [sys.executable, "train.py", "--config", str(config_path)]
    if preflight_only:
        cmd.append("--preflight-only")
    completed = subprocess.run(
        cmd,
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
    return completed.returncode, output


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def variant_complexity(overrides: dict[str, Any]) -> dict[str, Any]:
    batch_size = int(overrides.get("trainer.batch_size") or 0)
    grad_accum = int(overrides.get("trainer.grad_accum_steps") or 1)
    seq_len = overrides.get("trainer.seq_len")
    gradient_checkpointing = overrides.get("model.gradient_checkpointing")
    effective_batch = batch_size * grad_accum if batch_size else None
    if grad_accum > 1:
        label = f"batch={batch_size}, grad_accum={grad_accum} (extra forward/backward passes)"
        penalty = grad_accum
    else:
        label = f"batch={batch_size}, grad_accum=1 (direct batch)"
        penalty = 1
    if seq_len is not None:
        label = f"{label}, seq_len={seq_len}"
    if gradient_checkpointing is not None:
        label = f"{label}, gradient_checkpointing={gradient_checkpointing}"
        if gradient_checkpointing is True:
            penalty += 1
    return {
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "seq_len": seq_len,
        "gradient_checkpointing": gradient_checkpointing,
        "effective_batch_size": effective_batch,
        "label": label,
        "penalty": penalty,
    }


def choose_best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        record
        for record in records
        if record.get("status") == 0 and _number(record.get("avg_tokens_per_sec")) is not None
    ]
    if not candidates:
        return None

    val_losses = [
        float(record["last_val_loss"])
        for record in candidates
        if _number(record.get("last_val_loss")) is not None
    ]
    best_val = min(val_losses) if val_losses else None
    val_tolerance = best_val * 1.10 if best_val is not None else None

    def val_sane(record: dict[str, Any]) -> bool:
        record_val = _number(record.get("last_val_loss"))
        if val_tolerance is None:
            return True
        if record_val is None:
            return False
        return record_val <= val_tolerance

    sane = [record for record in candidates if val_sane(record)] or candidates
    sane.sort(
        key=lambda record: (
            float(record["avg_tokens_per_sec"]),
            -int(record.get("complexity", {}).get("penalty", 99)),
        ),
        reverse=True,
    )
    best = sane[0]
    val_loss = best.get("last_val_loss")
    val_note = "validation loss was unavailable"
    if _number(val_loss) is not None and best_val is not None:
        delta = (float(val_loss) / best_val - 1.0) * 100.0 if best_val > 0 else 0.0
        val_note = f"validation loss {float(val_loss):.4f} is within {delta:.1f}% of the best observed {best_val:.4f}"
    return {
        "variant": best["variant"],
        "avg_tokens_per_sec": best.get("avg_tokens_per_sec"),
        "last_val_loss": val_loss,
        "complexity": best.get("complexity", {}),
        "reason": (
            f"{best['variant']} is selected because it has the highest average tokens/sec "
            f"among validation-sane candidates; {val_note}; "
            f"complexity={best.get('complexity', {}).get('label', 'unknown')}."
        ),
    }


def render_matrix_markdown(
    matrix: dict[str, Any],
    records: list[dict[str, Any]],
    selection: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"# Matrix Summary: {matrix['name']}",
        "",
        matrix.get("description") or "",
        "",
        "| Variant | Execution | Status | Avg tokens/sec | Last val loss | Peak CUDA MB | Failure | Complexity | Run dir |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for record in records:
        avg = record.get("avg_tokens_per_sec")
        avg_number = _number(avg)
        avg_text = f"{avg_number:.2f}" if avg_number is not None else "n/a"
        val = record.get("last_val_loss")
        val_number = _number(val)
        val_text = f"{val_number:.4f}" if val_number is not None else "n/a"
        peak_alloc = _number(record.get("cuda_peak_allocated_mb"))
        peak_reserved = _number(record.get("cuda_peak_reserved_mb"))
        if peak_alloc is not None and peak_reserved is not None:
            peak_text = f"{peak_alloc:.0f}/{peak_reserved:.0f}"
        elif peak_alloc is not None:
            peak_text = f"{peak_alloc:.0f}/n/a"
        else:
            peak_text = "n/a"
        failure_text = record.get("failure_kind") or ""
        lines.append(
            f"| `{record['variant']}` | {record.get('execution') or 'train'} | "
            f"`{record['status']}` | {avg_text} | {val_text} | {peak_text} | {failure_text} | "
            f"{record.get('complexity', {}).get('label', 'n/a')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    if selection:
        lines.extend(
            [
                "",
                "## Best Config Reasoning",
                "",
                f"- Selected variant: `{selection['variant']}`",
                f"- Average tokens/sec: `{selection.get('avg_tokens_per_sec'):.2f}`",
                f"- Last validation loss: `{selection.get('last_val_loss'):.4f}`"
                if isinstance(selection.get("last_val_loss"), (int, float))
                else "- Last validation loss: `n/a`",
                f"- Reason: {selection['reason']}",
            ]
        )
    return "\n".join(lines) + "\n"


def run_matrix(
    matrix_path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    ledger_path: str | Path = "runs/ledger.jsonl",
    dry_run: bool = False,
) -> int:
    matrix = load_matrix(matrix_path, project_root)
    variants = expand_variants(matrix)
    records: list[dict[str, Any]] = []
    status = 0

    for variant in variants:
        config_path = materialize_variant_config(matrix, variant, project_root=project_root)
        record = {
            "mode": "matrix_runner",
            "matrix": matrix["name"],
            "variant": variant["name"],
            "config": str(config_path),
            "overrides": variant["overrides"],
            "execution": "preflight_only" if variant.get("preflight_only") else "train",
            "complexity": variant_complexity(variant["overrides"]),
            "status": 0,
        }
        if dry_run:
            records.append(record)
            continue

        log_path = (
            project_root
            / "runs"
            / "matrix_logs"
            / f"{matrix['name']}-{variant['name']}-{int(time.time())}.log"
        )
        run_status, run_output = run_config(
            config_path,
            project_root,
            log_path=log_path,
            preflight_only=bool(variant.get("preflight_only")),
        )
        status = status or run_status
        record["status"] = run_status
        record["subprocess_log"] = str(log_path)
        failure = classify_failure(run_status, run_output)
        if failure:
            record.update(failure)
        run_dir = latest_run(project_root)
        if run_dir is not None:
            summary = summarize_run(run_dir)
            (run_dir / "agent_summary.md").write_text(render_markdown(summary), encoding="utf-8")
            record.update(
                {
                    "run_dir": str(run_dir),
                    "avg_tokens_per_sec": summary.get("avg_tokens_per_sec"),
                    "last_train_loss": summary.get("last_train", {}).get("loss"),
                    "last_val_loss": summary.get("last_validate", {}).get("loss"),
                    "cuda_peak_allocated_mb": summary.get("cuda_peak_allocated_mb"),
                    "cuda_peak_reserved_mb": summary.get("cuda_peak_reserved_mb"),
                }
            )
        append_ledger(project_root / ledger_path, record)
        records.append(record)

    selection = None if dry_run else choose_best_record(records)
    payload = {"matrix": matrix["name"], "records": records, "selection": selection}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    summary_dir = project_root / "runs" / "matrix_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_base = summary_dir / f"{matrix['name']}-{stamp}"
    (summary_base.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (summary_base.with_suffix(".md")).write_text(
        render_matrix_markdown(matrix, records, selection),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--ledger", default="runs/ledger.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_matrix(args.matrix, ledger_path=args.ledger, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
