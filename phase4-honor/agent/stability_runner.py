"""Run longer stability checks from auto-probe recommendations."""

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

from agent.analyzer import load_events, render_markdown, summarize_run
from agent.ledger import append_ledger
from agent.matrix_runner import classify_failure, run_config
from agent.runner import latest_run
from training_framework.config import dump_yaml_dict, load_yaml_dict
from training_framework.config_merge import set_dotted_value


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def load_recommendation(path: str | Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    recommendation_path = Path(path)
    if not recommendation_path.is_absolute():
        recommendation_path = project_root / recommendation_path
    data = json.loads(recommendation_path.read_text(encoding="utf-8"))
    if not isinstance(data.get("records"), list):
        raise ValueError("recommendation records must be a list")
    if not isinstance(data.get("recommendation"), dict):
        raise ValueError("recommendation payload must contain a recommendation object")
    return data


def selected_record(payload: dict[str, Any]) -> dict[str, Any]:
    recommendation = payload["recommendation"]
    selected_variant = recommendation.get("selected_variant")
    selected_budget = recommendation.get("selected_token_budget")
    for record in payload["records"]:
        if selected_variant and record.get("variant") == selected_variant:
            return record
        if selected_budget is not None and record.get("token_budget") == selected_budget:
            return record
    raise ValueError("Could not find selected record in recommendation payload")


def resolve_config_path(config_path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(config_path)
    if path.exists():
        return path
    if path.is_absolute() and len(path.parts) > 1 and path.parts[1] == "workspace":
        local_workspace_relative = project_root / Path(*path.parts[2:])
        if local_workspace_relative.exists():
            return local_workspace_relative
    if not path.is_absolute():
        candidate = project_root / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve source config: {config_path}")


def materialize_stability_config(
    payload: dict[str, Any],
    *,
    steps: int,
    validate_every: int | None = None,
    log_every: int | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if steps <= 0:
        raise ValueError("steps must be positive")
    record = selected_record(payload)
    source_config = resolve_config_path(str(record["config"]), project_root)
    raw = load_yaml_dict(source_config)
    selected_budget = record.get("token_budget") or payload["recommendation"].get("selected_token_budget")
    validate_every = validate_every or max(1, steps // 5)
    log_every = log_every or max(1, min(10, steps // 10))
    base_name = str(raw.get("trainer", {}).get("run_name") or record.get("variant") or "stability")
    run_name = f"{base_name}-stability-{steps}step"
    set_dotted_value(raw, "trainer.run_name", run_name)
    set_dotted_value(raw, "trainer.max_steps", steps)
    set_dotted_value(raw, "trainer.validate_every_steps", validate_every)
    set_dotted_value(raw, "trainer.log_every_steps", log_every)
    raw.setdefault("metadata", {})
    notes = str(raw["metadata"].get("notes") or "")
    stability_note = (
        f"stability_run_from={payload.get('probe', 'unknown')} "
        f"selected_tokens_per_step={selected_budget} steps={steps}"
    )
    raw["metadata"]["notes"] = f"{notes} || {stability_note}" if notes else stability_note
    out_dir = project_root / "runs" / "stability_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{payload.get('probe', 'recommendation')}-{record.get('variant', 'selected')}-{steps}step.json"
    dump_yaml_dict(raw, out_path)
    return out_path


def summarize_stability(run_dir: Path, run_status: int, record: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_run(run_dir)
    run_summary_path = run_dir / "summary.json"
    run_summary = {}
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    events = load_events(run_dir / "events.jsonl")
    train_events = [event for event in events if event.get("event") == "train"]
    validate_events = [event for event in events if event.get("event") == "validate"]
    train_losses = [
        float(event["metrics"]["loss"])
        for event in train_events
        if _number(event.get("metrics", {}).get("loss")) is not None
    ]
    val_losses = [
        float(event["metrics"]["loss"])
        for event in validate_events
        if _number(event.get("metrics", {}).get("loss")) is not None
    ]
    nonfinite_loss = len(train_losses) != len(train_events) or len(val_losses) != len(validate_events)
    expected_steps = int(record.get("requested_steps") or 0)
    completed_steps = int(run_summary.get("global_step") or len(train_events))
    logged_train_events = len(train_events)
    status = "pass"
    reasons = []
    if run_status != 0:
        status = "fail"
        reasons.append(f"training command exited with status {run_status}")
    if expected_steps and completed_steps < expected_steps:
        status = "fail"
        reasons.append(f"completed {completed_steps}/{expected_steps} requested train steps")
    if nonfinite_loss:
        status = "fail"
        reasons.append("non-finite train or validation loss observed")
    if not val_losses:
        status = "warn" if status == "pass" else status
        reasons.append("no validation event was recorded")

    stability = {
        "run_dir": str(run_dir),
        "status": status,
        "reasons": reasons,
        "requested_steps": expected_steps,
        "completed_train_steps": completed_steps,
        "logged_train_events": logged_train_events,
        "validation_events": len(validate_events),
        "first_train_loss": train_losses[0] if train_losses else None,
        "last_train_loss": train_losses[-1] if train_losses else None,
        "last_val_loss": val_losses[-1] if val_losses else None,
        "avg_tokens_per_sec": summary.get("avg_tokens_per_sec"),
        "cuda_peak_allocated_mb": summary.get("cuda_peak_allocated_mb"),
        "cuda_peak_reserved_mb": summary.get("cuda_peak_reserved_mb"),
        "source_recommendation": record.get("source_recommendation"),
        "source_variant": record.get("source_variant"),
        "source_token_budget": record.get("source_token_budget"),
    }
    return stability


def render_stability_markdown(stability: dict[str, Any]) -> str:
    avg = _number(stability.get("avg_tokens_per_sec"))
    peak_alloc = _number(stability.get("cuda_peak_allocated_mb"))
    peak_reserved = _number(stability.get("cuda_peak_reserved_mb"))
    lines = [
        "# Stability Run Summary",
        "",
        f"- Status: `{stability['status']}`",
        f"- Run directory: `{stability['run_dir']}`",
        f"- Source recommendation: `{stability.get('source_recommendation')}`",
        f"- Source variant: `{stability.get('source_variant')}`",
        f"- Source token budget: `{stability.get('source_token_budget')}`",
        f"- Completed train steps: `{stability['completed_train_steps']}/{stability['requested_steps']}`",
        f"- Logged train events: `{stability.get('logged_train_events')}`",
        f"- Validation events: `{stability['validation_events']}`",
        f"- Average tokens/sec: `{avg:.2f}`" if avg is not None else "- Average tokens/sec: `n/a`",
        f"- Last train loss: `{stability.get('last_train_loss')}`",
        f"- Last validation loss: `{stability.get('last_val_loss')}`",
        f"- Peak CUDA MB allocated/reserved: `{peak_alloc:.0f}/{peak_reserved:.0f}`"
        if peak_alloc is not None and peak_reserved is not None
        else "- Peak CUDA MB allocated/reserved: `n/a`",
        "",
        "## Reasons",
        "",
    ]
    reasons = stability.get("reasons") or ["No stability issues detected by the runner."]
    lines.extend(f"- {reason}" for reason in reasons)
    return "\n".join(lines) + "\n"


def write_stability_artifacts(
    payload: dict[str, Any],
    stability: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, str]:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    summary_dir = project_root / "runs" / "stability_summaries"
    recommendation_dir = project_root / "runs" / "recommendations"
    summary_dir.mkdir(parents=True, exist_ok=True)
    recommendation_dir.mkdir(parents=True, exist_ok=True)
    name = str(payload.get("probe", "recommendation"))
    summary_base = summary_dir / f"{name}-stability-{stamp}"
    latest_base = recommendation_dir / f"{name}-stability-latest"
    body = render_stability_markdown(stability)
    package = {"probe": payload.get("probe"), "stability": stability}
    for path in (summary_base.with_suffix(".json"), latest_base.with_suffix(".json")):
        path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    for path in (summary_base.with_suffix(".md"), latest_base.with_suffix(".md")):
        path.write_text(body, encoding="utf-8")
    return {
        "summary_json": str(summary_base.with_suffix(".json")),
        "summary_md": str(summary_base.with_suffix(".md")),
        "latest_json": str(latest_base.with_suffix(".json")),
        "latest_md": str(latest_base.with_suffix(".md")),
    }


def run_stability(
    recommendation_path: str | Path,
    *,
    steps: int = 50,
    validate_every: int | None = None,
    log_every: int | None = None,
    project_root: Path = PROJECT_ROOT,
    ledger_path: str | Path = "runs/ledger.jsonl",
    dry_run: bool = False,
) -> int:
    payload = load_recommendation(recommendation_path, project_root)
    source = selected_record(payload)
    config_path = materialize_stability_config(
        payload,
        steps=steps,
        validate_every=validate_every,
        log_every=log_every,
        project_root=project_root,
    )
    record = {
        "mode": "stability_runner",
        "source_recommendation": str(recommendation_path),
        "source_variant": source.get("variant"),
        "source_token_budget": source.get("token_budget"),
        "requested_steps": steps,
        "config": str(config_path),
        "status": 0,
    }
    if dry_run:
        print(json.dumps({"record": record, "config": str(config_path)}, indent=2, sort_keys=True))
        return 0

    before = latest_run(project_root)
    log_path = (
        project_root
        / "runs"
        / "stability_logs"
        / f"{payload.get('probe', 'recommendation')}-{source.get('variant', 'selected')}-{steps}step-{int(time.time())}.log"
    )
    run_status, run_output = run_config(config_path, project_root, log_path=log_path)
    record["status"] = run_status
    record["subprocess_log"] = str(log_path)
    failure = classify_failure(run_status, run_output)
    if failure:
        record.update(failure)
    run_dir = latest_run(project_root)
    if run_dir is None or run_dir == before:
        append_ledger(project_root / ledger_path, record)
        print(json.dumps({"record": record, "error": "no new run directory detected"}, indent=2, sort_keys=True))
        return run_status or 1

    record["run_dir"] = str(run_dir)
    stability_record = dict(record)
    stability = summarize_stability(run_dir, run_status, stability_record)
    (run_dir / "agent_summary.md").write_text(render_markdown(summarize_run(run_dir)), encoding="utf-8")
    artifact_paths = write_stability_artifacts(payload, stability, project_root=project_root)
    record.update(
        {
            "stability_status": stability["status"],
            "avg_tokens_per_sec": stability.get("avg_tokens_per_sec"),
            "last_train_loss": stability.get("last_train_loss"),
            "last_val_loss": stability.get("last_val_loss"),
            "cuda_peak_allocated_mb": stability.get("cuda_peak_allocated_mb"),
            "cuda_peak_reserved_mb": stability.get("cuda_peak_reserved_mb"),
            "artifacts": artifact_paths,
        }
    )
    append_ledger(project_root / ledger_path, record)
    print(json.dumps({"record": record, "stability": stability, "artifacts": artifact_paths}, indent=2, sort_keys=True))
    return run_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommendation", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--ledger", default="runs/ledger.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_stability(
        args.recommendation,
        steps=args.steps,
        validate_every=args.validate_every,
        log_every=args.log_every,
        ledger_path=args.ledger,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
