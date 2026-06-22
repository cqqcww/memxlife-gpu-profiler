"""Bounded token-budget probing for early training feasibility decisions."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.analyzer import render_markdown, summarize_run
from agent.ledger import append_ledger
from agent.matrix_runner import (
    classify_failure,
    materialize_variant_config,
    run_config,
    variant_complexity,
)
from agent.runner import latest_run
from training_framework.config import load_yaml_dict


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def load_probe(path: str | Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    probe_path = Path(path)
    if not probe_path.is_absolute():
        probe_path = project_root / probe_path
    probe = load_yaml_dict(probe_path)
    if not probe.get("name"):
        raise ValueError("probe.name is required")
    if not probe.get("base_config"):
        raise ValueError("probe.base_config is required")
    token_budgets = probe.get("token_budgets")
    if not isinstance(token_budgets, list) or not token_budgets:
        raise ValueError("probe.token_budgets must be a non-empty list")
    budgets = [int(item) for item in token_budgets]
    if any(budget <= 0 for budget in budgets):
        raise ValueError("probe.token_budgets must contain positive integers")
    probe["token_budgets"] = budgets
    return probe


def token_budget_overrides(probe: dict[str, Any], token_budget: int) -> dict[str, Any]:
    batch_size = int(probe.get("batch_size", 1))
    grad_accum_steps = int(probe.get("grad_accum_steps", 1))
    if batch_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("batch_size and grad_accum_steps must be positive")
    seq_len = max(1, token_budget // (batch_size * grad_accum_steps))
    overrides = dict(probe.get("base_overrides") or {})
    overrides.update(
        {
            "trainer.seq_len": seq_len,
            "trainer.batch_size": batch_size,
            "trainer.grad_accum_steps": grad_accum_steps,
            "trainer.max_steps": int(probe.get("max_steps", 3)),
            "trainer.log_every_steps": int(probe.get("log_every_steps", 1)),
            "trainer.validate_every_steps": int(probe.get("validate_every_steps", probe.get("max_steps", 3))),
            "trainer.run_name": f"{probe.get('run_name_prefix', probe['name'])}-tok{token_budget}",
        }
    )
    return overrides


def materialize_probe_config(
    probe: dict[str, Any],
    token_budget: int,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    variant = {
        "name": f"tok{token_budget}",
        "overrides": token_budget_overrides(probe, token_budget),
    }
    matrix_like = {
        "name": probe["name"],
        "base_config": probe["base_config"],
        "model_profile": probe.get("model_profile"),
        "data_profile": probe.get("data_profile"),
    }
    return materialize_variant_config(matrix_like, variant, project_root=project_root)


def _new_latest_run(project_root: Path, before: Path | None) -> Path | None:
    after = latest_run(project_root)
    if after is None:
        return None
    if before is None or after != before:
        return after
    return None


def build_recommendation(probe: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [
        record
        for record in records
        if record.get("status") == 0 and _number(record.get("avg_tokens_per_sec")) is not None
    ]
    failures = [record for record in records if record.get("status") != 0]
    memory_limit = _number(probe.get("memory_limit_mb"))
    headroom_ratio = float(probe.get("headroom_ratio_for_expand", 0.75))
    min_speedup_ratio = float(probe.get("min_speedup_ratio", 1.03))

    if not successes:
        first_failure = failures[0] if failures else {}
        return {
            "selected_token_budget": None,
            "action": "fall_back",
            "reason": (
                "No successful training probe completed. Start with preflight-only, "
                "lower token budget, or a lower-memory optimizer before retrying."
            ),
            "failure_kind": first_failure.get("failure_kind"),
        }

    selected = successes[-1]
    last_failure = failures[-1] if failures and failures[-1].get("token_budget", 0) > selected["token_budget"] else None
    peak_reserved = _number(selected.get("cuda_peak_reserved_mb"))
    memory_ratio = peak_reserved / memory_limit if peak_reserved is not None and memory_limit else None
    next_budget = int(selected["token_budget"]) * 2
    speed_ratio = None
    if len(successes) >= 2:
        prev_speed = _number(successes[-2].get("avg_tokens_per_sec"))
        cur_speed = _number(successes[-1].get("avg_tokens_per_sec"))
        if prev_speed and cur_speed:
            speed_ratio = cur_speed / prev_speed

    if last_failure:
        action = "promote_last_safe"
        reason = (
            f"`{selected['variant']}` is the last successful budget before "
            f"`{last_failure['variant']}` failed as `{last_failure.get('failure_kind', 'unknown')}`."
        )
    elif speed_ratio is not None and speed_ratio < min_speedup_ratio:
        action = "stability_run"
        reason = (
            f"`{selected['variant']}` is safe, but throughput improved only "
            f"{speed_ratio:.2f}x over the previous safe point; run a longer stability check."
        )
    elif memory_ratio is not None and memory_ratio < headroom_ratio:
        action = "try_larger_budget"
        reason = (
            f"`{selected['variant']}` is safe and reserved CUDA peak is "
            f"{memory_ratio:.0%} of the configured memory limit; try `{next_budget}` tokens/step next."
        )
    else:
        action = "stability_run"
        reason = (
            f"`{selected['variant']}` is the best bounded safe point. Prefer a longer "
            "50-100 step run before expanding further."
        )

    return {
        "selected_variant": selected["variant"],
        "selected_token_budget": selected["token_budget"],
        "avg_tokens_per_sec": selected.get("avg_tokens_per_sec"),
        "last_val_loss": selected.get("last_val_loss"),
        "cuda_peak_allocated_mb": selected.get("cuda_peak_allocated_mb"),
        "cuda_peak_reserved_mb": selected.get("cuda_peak_reserved_mb"),
        "memory_limit_mb": memory_limit,
        "memory_ratio_reserved": memory_ratio,
        "next_token_budget": next_budget if action == "try_larger_budget" else None,
        "action": action,
        "reason": reason,
    }


def render_recommendation_markdown(
    probe: dict[str, Any],
    records: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> str:
    lines = [
        f"# Auto Probe Recommendation: {probe['name']}",
        "",
        probe.get("description") or "",
        "",
        "| Variant | Status | Tokens/step | Avg tokens/sec | Last val loss | Peak CUDA MB | Failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        avg = _number(record.get("avg_tokens_per_sec"))
        val = _number(record.get("last_val_loss"))
        peak_alloc = _number(record.get("cuda_peak_allocated_mb"))
        peak_reserved = _number(record.get("cuda_peak_reserved_mb"))
        avg_text = f"{avg:.2f}" if avg is not None else "n/a"
        val_text = f"{val:.4f}" if val is not None else "n/a"
        peak = "n/a"
        if peak_alloc is not None and peak_reserved is not None:
            peak = f"{peak_alloc:.0f}/{peak_reserved:.0f}"
        elif peak_alloc is not None:
            peak = f"{peak_alloc:.0f}/n/a"
        lines.append(
            f"| `{record['variant']}` | `{record['status']}` | {record['token_budget']} | "
            f"{avg_text} | {val_text} | {peak} | {record.get('failure_kind') or ''} |"
        )
    selected = recommendation.get("selected_token_budget")
    selected_text = f"`{selected}`" if selected is not None else "`none`"
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Action: `{recommendation['action']}`",
            f"- Selected token budget: {selected_text}",
            f"- Next token budget: `{recommendation.get('next_token_budget')}`"
            if recommendation.get("next_token_budget")
            else "- Next token budget: `n/a`",
            f"- Reason: {recommendation['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_probe_artifacts(
    probe: dict[str, Any],
    records: list[dict[str, Any]],
    recommendation: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, str]:
    payload = {"probe": probe["name"], "records": records, "recommendation": recommendation}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    summary_dir = project_root / "runs" / "auto_probe_summaries"
    recommendation_dir = project_root / "runs" / "recommendations"
    summary_dir.mkdir(parents=True, exist_ok=True)
    recommendation_dir.mkdir(parents=True, exist_ok=True)
    summary_base = summary_dir / f"{probe['name']}-{stamp}"
    latest_base = recommendation_dir / f"{probe['name']}-latest"
    markdown = render_recommendation_markdown(probe, records, recommendation)
    summary_json = summary_base.with_suffix(".json")
    summary_md = summary_base.with_suffix(".md")
    latest_json = latest_base.with_suffix(".json")
    latest_md = latest_base.with_suffix(".md")
    for path in (summary_json, latest_json):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for path in (summary_md, latest_md):
        path.write_text(markdown, encoding="utf-8")
    return {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
    }


def run_auto_probe(
    probe_path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    ledger_path: str | Path = "runs/ledger.jsonl",
    dry_run: bool = False,
) -> int:
    probe = load_probe(probe_path, project_root)
    records: list[dict[str, Any]] = []
    status = 0

    for token_budget in probe["token_budgets"]:
        overrides = token_budget_overrides(probe, token_budget)
        config_path = materialize_probe_config(probe, token_budget, project_root=project_root)
        record = {
            "mode": "auto_probe",
            "probe": probe["name"],
            "variant": f"tok{token_budget}",
            "token_budget": token_budget,
            "config": str(config_path),
            "overrides": overrides,
            "complexity": variant_complexity(overrides),
            "status": 0,
        }
        if dry_run:
            records.append(record)
            continue

        log_path = (
            project_root
            / "runs"
            / "auto_probe_logs"
            / f"{probe['name']}-tok{token_budget}-{int(time.time())}.log"
        )
        before = latest_run(project_root)
        run_status, run_output = run_config(config_path, project_root, log_path=log_path)
        status = status or run_status
        record["status"] = run_status
        record["subprocess_log"] = str(log_path)
        failure = classify_failure(run_status, run_output)
        if failure:
            record.update(failure)
        run_dir = _new_latest_run(project_root, before)
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
        if run_status != 0 and probe.get("stop_on_failure", True):
            break

    if dry_run:
        recommendation = {
            "selected_token_budget": None,
            "next_token_budget": None,
            "action": "dry_run_only",
            "reason": "Dry-run materialized probe configs but did not execute training, so no speed/loss/memory recommendation is available yet.",
        }
    else:
        recommendation = build_recommendation(probe, records)
    artifact_paths = write_probe_artifacts(probe, records, recommendation, project_root=project_root)
    payload = {
        "probe": probe["name"],
        "records": records,
        "recommendation": recommendation,
        "artifacts": artifact_paths,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--ledger", default="runs/ledger.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_auto_probe(args.probe, ledger_path=args.ledger, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
