"""Subprocess runner for agent-controlled experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.analyzer import render_markdown, summarize_run
from agent.ledger import append_ledger


def run_training(config_path: str, project_root: Path) -> int:
    cmd = [sys.executable, "train.py", "--config", config_path]
    return subprocess.call(cmd, cwd=project_root)


def latest_run(project_root: Path) -> Path | None:
    runs = sorted(
        (
            p
            for p in (project_root / "runs").glob("*")
            if p.is_dir() and (p / "copied_config.yaml").exists()
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return runs[-1] if runs else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ledger", default="runs/ledger.jsonl")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    status = run_training(args.config, project_root)
    run_dir = latest_run(project_root)
    record = {"config": args.config, "status": status}
    if run_dir is not None:
        summary = summarize_run(run_dir)
        (run_dir / "agent_summary.md").write_text(render_markdown(summary), encoding="utf-8")
        record.update({"run_dir": str(run_dir), "avg_tokens_per_sec": summary.get("avg_tokens_per_sec")})
    append_ledger(project_root / args.ledger, record)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
