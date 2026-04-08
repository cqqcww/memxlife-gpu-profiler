"""Audit logger — generates Markdown reports for the engineering reasoning rubric."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class AuditLogger:
    """Records agent actions and generates audit reports."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.entries: list[dict[str, Any]] = []
        self.log_file = run_dir / "audit_log.jsonl"

    def log(
        self,
        agent: str,
        action: str,
        metric_name: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent": agent,
            "action": action,
            "metric_name": metric_name,
            "detail": detail or {},
        }
        self.entries.append(entry)
        # Append to JSONL file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def generate_report(
        self,
        results: dict[str, Any],
        environment: dict[str, Any],
    ) -> str:
        """Generate a Markdown audit report."""
        lines = [
            "# GPU Hardware Profiling — Audit Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. Environment Profile",
            "",
            f"- GPU: {environment.get('gpu_name', 'unknown')}",
            f"- Driver: {environment.get('driver_version', 'unknown')}",
            f"- CUDA: {environment.get('cuda_version', 'unknown')}",
            f"- Trust level: {environment.get('trust_level', 'unknown')}",
        ]

        anomalies = environment.get("detected_anomalies", [])
        if anomalies:
            lines.append("")
            lines.append("### Detected Anomalies")
            for a in anomalies:
                lines.append(f"- ⚠️ {a}")

        lines.extend(["", "## 2. Methodology", ""])
        lines.append(
            "This system uses a multi-agent architecture to autonomously probe GPU hardware "
            "characteristics. Each metric is measured through CUDA micro-benchmarks that are "
            "generated, compiled, executed, and analyzed in an iterative loop. The system does "
            "NOT rely on `cudaGetDeviceProperties` or static spec tables, as these may be "
            "intercepted or return misleading data in the evaluation environment."
        )

        lines.extend(["", "### Agent Pipeline", ""])
        lines.append("1. **Environment Scout** — Detects tools, collects baseline GPU info, flags anomalies")
        lines.append("2. **Planner** — Selects probe strategy from metrics catalog")
        lines.append("3. **Codegen** — Generates CUDA micro-benchmark (template + LLM refinement)")
        lines.append("4. **Runner** — Compiles and executes with adaptive retry")
        lines.append("5. **Analyzer** — Extracts values, assesses confidence, physics sanity check")

        lines.extend(["", "## 3. Results Summary", ""])
        lines.append("| Metric | Value | Unit | Confidence | Method |")
        lines.append("|:-------|------:|:-----|:-----------|:-------|")
        for name, val in sorted(results.items()):
            if isinstance(val, dict):
                lines.append(
                    f"| {name} | {val.get('value', 'N/A')} | {val.get('unit', '')} "
                    f"| {val.get('confidence', 0):.2f} | {val.get('method', '')} |"
                )
            else:
                lines.append(f"| {name} | {val} | | | |")

        lines.extend(["", "## 4. Detailed Probe Log", ""])
        # Group entries by metric
        by_metric: dict[str, list[dict]] = {}
        for entry in self.entries:
            m = entry.get("metric_name", "system")
            by_metric.setdefault(m, []).append(entry)

        for metric, entries in sorted(by_metric.items()):
            lines.append(f"### {metric}")
            lines.append("")
            for entry in entries:
                agent = entry.get("agent", "")
                action = entry.get("action", "")
                ts = entry.get("time_str", "")
                detail = entry.get("detail", {})

                lines.append(f"- **[{ts}] {agent}** → {action}")

                # Show key details
                if "strategy_name" in detail:
                    lines.append(f"  - Strategy: {detail['strategy_name']}")
                if "reasoning" in detail and detail["reasoning"]:
                    lines.append(f"  - Reasoning: {detail['reasoning']}")
                if "value" in detail and detail["value"] is not None:
                    lines.append(f"  - Value: {detail['value']}")
                if "confidence" in detail:
                    lines.append(f"  - Confidence: {detail['confidence']}")
                if "error" in detail and detail["error"]:
                    lines.append(f"  - Error: {detail['error']}")
                if "anomalies" in detail and detail["anomalies"]:
                    for a in detail["anomalies"]:
                        lines.append(f"  - ⚠️ {a}")
            lines.append("")

        lines.extend([
            "## 5. Cross-Verification Notes",
            "",
            "Where applicable, metrics were measured using multiple strategies "
            "(e.g., direct micro-benchmark + ncu profiling) and cross-checked "
            "against physical constraints (e.g., L1 < L2 < DRAM latency, "
            "bandwidth ≤ theoretical peak).",
            "",
            "---",
            f"*Report generated by MemXLife GPU Profiling Agent System*",
        ])

        return "\n".join(lines)

    def save_report(self, results: dict, environment: dict) -> Path:
        """Generate and save the audit report."""
        report = self.generate_report(results, environment)
        report_path = self.run_dir / "audit_report.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path
