"""Deterministic experiment planner.

The planner intentionally stays simple: it reads prior run summaries and chooses
the next config from a small ladder. This makes the agent auditable for the
report instead of pretending to be a black-box optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.analyzer import render_markdown, summarize_run
from agent.ledger import append_ledger
from agent.runner import latest_run, run_training


EXPERIMENT_LADDER = [
    "configs/debug.yaml",
    "configs/baseline_tinystories.yaml",
    "configs/cached_tinystories.yaml",
    "configs/mixed_precision.yaml",
]


def choose_next_config(completed_configs: set[str]) -> str | None:
    for config in EXPERIMENT_LADDER:
        if config not in completed_configs:
            return config
    return None


def load_ledger(path: str | Path) -> list[dict]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    records = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def completed_configs(records: list[dict]) -> set[str]:
    return {
        str(record.get("config"))
        for record in records
        if record.get("status") == 0 and record.get("config")
    }


def infer_bottleneck(summary: dict) -> tuple[str, str]:
    metrics = summary.get("last_train", {})
    data_s = float(metrics.get("data_s") or 0.0)
    forward_s = float(metrics.get("forward_s") or 0.0)
    backward_s = float(metrics.get("backward_s") or 0.0)
    optimizer_s = float(metrics.get("optimizer_s") or 0.0)
    compute_s = forward_s + backward_s

    if data_s > 0.002 and data_s > 0.2 * max(compute_s, 1e-9):
        return (
            "data loading/tokenization time is visible in the step breakdown",
            "try or keep token-block caching and compare dataloader worker settings",
        )
    if optimizer_s > 0.75 * max(forward_s, 1e-9):
        return (
            "optimizer overhead is large relative to forward time",
            "try a larger batch size or gradient accumulation sweep to amortize per-step overhead",
        )
    if compute_s > 2.0 * max(optimizer_s, 1e-9):
        return (
            "forward/backward compute dominates the measured step time",
            "try mixed precision if CUDA supports it and verify loss stability",
        )
    return (
        "no single phase dominates strongly in the latest logged step",
        "prioritize evidence export, longer stability run, or a small batch-size sweep",
    )


def write_patch_proposal(run_dir: str | Path, bottleneck: str, proposal: str) -> Path:
    path = Path(run_dir) / "agent_patch_proposal.md"
    path.write_text(
        "\n".join(
            [
                "# Agent Patch Proposal",
                "",
                f"- Observed bottleneck: {bottleneck}",
                f"- Proposed change: {proposal}",
                "- Expected effect: improve tokens/sec or reduce variance while keeping checkpoint/resume intact.",
                "- Risk: added complexity may hide bugs in timing or resume behavior.",
                "- Rollback: revert the proposed config/code change and rerun the previous stable config.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def run_next(ledger_path: Path, project_root: Path, goal: str) -> int:
    records = load_ledger(ledger_path)
    next_config = choose_next_config(completed_configs(records))
    if next_config is None:
        print(
            json.dumps(
                {
                    "goal": goal,
                    "next_config": None,
                    "message": "experiment ladder is complete",
                },
                indent=2,
            )
        )
        return 0

    status = run_training(next_config, project_root)
    run_dir = latest_run(project_root)
    record = {
        "mode": "planner_run_next",
        "goal": goal,
        "config": next_config,
        "status": status,
    }
    if run_dir is not None:
        summary = summarize_run(run_dir)
        (run_dir / "agent_summary.md").write_text(render_markdown(summary), encoding="utf-8")
        bottleneck, proposal = infer_bottleneck(summary)
        proposal_path = write_patch_proposal(run_dir, bottleneck, proposal)
        record.update(
            {
                "run_dir": str(run_dir),
                "avg_tokens_per_sec": summary.get("avg_tokens_per_sec"),
                "proposal_path": str(proposal_path),
                "observed_bottleneck": bottleneck,
                "proposed_change": proposal,
            }
        )
    append_ledger(ledger_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default="improve_tokens_per_sec")
    parser.add_argument("--ledger", default="runs/ledger.jsonl")
    parser.add_argument("--run-next", action="store_true")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    ledger_path = project_root / args.ledger
    records = load_ledger(ledger_path)
    next_config = choose_next_config(completed_configs(records))
    if not args.run_next:
        print(
            json.dumps(
                {
                    "goal": args.goal,
                    "completed_configs": sorted(completed_configs(records)),
                    "next_config": next_config,
                    "ledger": str(ledger_path),
                },
                indent=2,
            )
        )
        return 0
    return run_next(ledger_path, project_root, args.goal)


if __name__ == "__main__":
    raise SystemExit(main())
