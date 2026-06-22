#!/usr/bin/env python3
"""Export compact evidence tables from Phase 4 run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


TIMING_KEYS = ["data_s", "forward_s", "backward_s", "optimizer_s"]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_metric(events: list[dict], key: str, skip_first: bool = False) -> float | None:
    values = []
    train_events = [event for event in events if event.get("event") == "train"]
    if skip_first and len(train_events) > 1:
        train_events = train_events[1:]
    for event in train_events:
        value = event.get("metrics", {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return mean(values) if values else None


def last_metric(events: list[dict], event_name: str, key: str) -> float | None:
    selected = [event for event in events if event.get("event") == event_name]
    if not selected:
        return None
    value = selected[-1].get("metrics", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def summarize_run(run_dir: Path) -> dict:
    summary = load_json(run_dir / "summary.json")
    events = load_events(run_dir / "events.jsonl")
    row = {
        "run": run_dir.name,
        "steps": summary.get("global_step"),
        "cache": (summary.get("data") or {}).get("cache"),
        "parameters": summary.get("parameters"),
        "final_train_loss": summary.get("final_loss"),
        "final_val_loss": last_metric(events, "validate", "loss"),
        "mean_tokens_per_sec": mean_metric(events, "tokens_per_sec", skip_first=True),
    }
    for key in TIMING_KEYS:
        value = mean_metric(events, key, skip_first=True)
        row[key.replace("_s", "_ms")] = value * 1000.0 if value is not None else None
    return row


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict], path: Path) -> None:
    columns = [
        "run",
        "steps",
        "cache",
        "parameters",
        "final_train_loss",
        "final_val_loss",
        "mean_tokens_per_sec",
        "data_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
    ]
    lines = [
        "# Phase 4 Event Evidence Table",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(col)) for col in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-md", default="runs/evidence_table.md")
    parser.add_argument("--output-csv", default="runs/evidence_table.csv")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    run_dirs = sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and (path / "summary.json").exists()
    )
    rows = [summarize_run(path) for path in run_dirs]
    write_markdown(rows, Path(args.output_md))
    write_csv(rows, Path(args.output_csv))
    print(json.dumps({"rows": len(rows), "output_md": args.output_md, "output_csv": args.output_csv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
