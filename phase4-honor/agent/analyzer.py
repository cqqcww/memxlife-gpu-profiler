"""Parse run artifacts and produce compact summaries."""

from __future__ import annotations

import json
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _max_metric(events: list[dict], key: str) -> float | None:
    values = []
    for event in events:
        value = event.get("metrics", {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return max(values) if values else None


def summarize_run(run_dir: str | Path) -> dict:
    run_path = Path(run_dir)
    events = load_events(run_path / "events.jsonl")
    train = [e for e in events if e.get("event") == "train"]
    valid = [e for e in events if e.get("event") == "validate"]
    last_train = train[-1]["metrics"] if train else {}
    last_valid = valid[-1]["metrics"] if valid else {}
    speeds = [e["metrics"].get("tokens_per_sec") for e in train if e["metrics"].get("tokens_per_sec")]
    avg_speed = sum(speeds) / len(speeds) if speeds else None
    summary = {
        "run_dir": str(run_path),
        "train_events": len(train),
        "validate_events": len(valid),
        "last_train": last_train,
        "last_validate": last_valid,
        "avg_tokens_per_sec": avg_speed,
        "cuda_peak_allocated_mb": _max_metric(events, "cuda_peak_allocated_mb"),
        "cuda_peak_reserved_mb": _max_metric(events, "cuda_peak_reserved_mb"),
    }
    (run_path / "agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_markdown(summary: dict) -> str:
    avg = summary.get("avg_tokens_per_sec")
    avg_text = f"{avg:.2f}" if isinstance(avg, (int, float)) else "n/a"
    lines = [
        "# Agent Run Summary",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Train events: `{summary['train_events']}`",
        f"- Validation events: `{summary['validate_events']}`",
        f"- Average tokens/sec: `{avg_text}`",
        "",
        "## Last Train Metrics",
        "",
        "```json",
        json.dumps(summary.get("last_train", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Last Validation Metrics",
        "",
        "```json",
        json.dumps(summary.get("last_validate", {}), indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"
