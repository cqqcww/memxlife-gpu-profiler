"""Parser for NVIDIA Nsight Compute (ncu) output."""

from __future__ import annotations

import csv
import io
import re
from typing import Any


def parse_ncu_csv(output: str) -> list[dict[str, Any]]:
    """Parse ncu --csv output into a list of metric dicts."""
    # ncu CSV output has header rows starting with "==PROF=="
    lines = []
    in_csv = False
    for line in output.splitlines():
        if line.startswith('"Metric Name"') or line.startswith("Metric Name"):
            in_csv = True
        if in_csv:
            lines.append(line)

    if not lines:
        return _parse_ncu_text(output)

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    results = []
    for row in reader:
        name = row.get("Metric Name", "").strip()
        value_str = row.get("Metric Value", row.get("Avg", "")).strip()
        unit = row.get("Metric Unit", row.get("Unit", "")).strip()
        if name:
            results.append({
                "name": name,
                "value": _parse_numeric(value_str),
                "value_raw": value_str,
                "unit": unit,
            })
    return results


def _parse_ncu_text(output: str) -> list[dict[str, Any]]:
    """Fallback: parse ncu text output for metric lines."""
    results = []
    # Pattern: metric_name    value    unit
    for line in output.splitlines():
        line = line.strip()
        # Look for lines with metric-like names
        m = re.match(
            r"([\w.]+(?:__[\w.]+)+)\s+([\d.,]+(?:\.\d+)?)\s*(%|bytes|cycles|GB/s|.*)?",
            line,
        )
        if m:
            results.append({
                "name": m.group(1),
                "value": _parse_numeric(m.group(2)),
                "value_raw": m.group(2),
                "unit": (m.group(3) or "").strip(),
            })
    return results


def extract_ncu_metric(parsed: list[dict[str, Any]], metric_name: str) -> float | None:
    """Extract a specific metric value from parsed ncu output."""
    for entry in parsed:
        if entry["name"] == metric_name:
            return entry["value"]
    # Partial match
    for entry in parsed:
        if metric_name in entry["name"] or entry["name"] in metric_name:
            return entry["value"]
    return None


def _parse_numeric(s: str) -> float | None:
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None
