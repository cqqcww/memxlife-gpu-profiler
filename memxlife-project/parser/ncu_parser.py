"""Parser for NVIDIA Nsight Compute (ncu) output.

Supports multiple ncu output formats:
  - --csv mode (primary)
  - Text/table mode (fallback)
  - --page raw mode
  - SOL (Speed of Light) section parsing
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any


def parse_ncu_csv(output: str) -> list[dict[str, Any]]:
    """Parse ncu --csv output into a list of metric dicts.

    Handles multiple CSV header formats:
      - "Metric Name","Metric Value","Metric Unit"
      - "Metric Name","Min","Max","Avg","Unit"
    """
    # ncu CSV output has header rows starting with "==PROF=="
    lines = []
    in_csv = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith('"Metric Name"') or stripped.startswith("Metric Name"):
            in_csv = True
            lines = [line]  # Reset: use latest header if multiple kernels
            continue
        if in_csv:
            # Stop at blank lines or new section markers
            if not stripped or stripped.startswith("=="):
                if len(lines) > 1:
                    break
                continue
            lines.append(line)

    if not lines:
        return _parse_ncu_text(output)

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    results = []
    for row in reader:
        name = row.get("Metric Name", "").strip()
        if not name:
            continue

        # Try different value column names
        value_str = ""
        for col in ("Metric Value", "Avg", "Average", "Max", "Value"):
            v = row.get(col, "").strip()
            if v:
                value_str = v
                break

        unit = ""
        for col in ("Metric Unit", "Unit"):
            u = row.get(col, "").strip()
            if u:
                unit = u
                break

        results.append({
            "name": name,
            "value": _parse_numeric(value_str),
            "value_raw": value_str,
            "unit": unit,
        })
    return results


def parse_ncu_raw_page(output: str) -> list[dict[str, Any]]:
    """Parse ncu --page raw output format.

    This format shows each metric on its own line with name, value, and unit
    in a tabular format.
    """
    results = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("-"):
            continue
        # --page raw format: "  metric.name               value  unit"
        # Also handles the "Section: ..." header lines
        if line.startswith("Section:") or line.startswith("Kernel:"):
            continue

        m = re.match(
            r"\s*([\w.]+(?:__[\w.]+)*(?:\.[\w.]+)*)\s+"  # metric name
            r"([\d.,]+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"    # value (inc. scientific)
            r"(%|bytes|byte|KB|MB|GB|cycles?|ns|us|ms|s|GB/s|"
            r"instructions?|warps?|sectors?|requests?)?\s*$",  # unit
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


def parse_ncu_sol_section(output: str) -> dict[str, float]:
    """Parse the SOL (Speed of Light) section from ncu output.

    Returns a dict with keys like:
      - sol_compute_pct: SM throughput as % of peak
      - sol_memory_pct: memory throughput as % of peak
      - sol_roofline: operational intensity
    """
    sol = {}

    # Look for SOL section in ncu output
    for line in output.splitlines():
        line_lower = line.lower().strip()

        # SOL Compute
        m = re.search(r"sol\s*compute[:\s]+(\d+\.?\d*)\s*%", line_lower)
        if m:
            sol["sol_compute_pct"] = float(m.group(1))

        m = re.search(r"sol\s*memory[:\s]+(\d+\.?\d*)\s*%", line_lower)
        if m:
            sol["sol_memory_pct"] = float(m.group(1))

        # SM throughput
        if "sm__throughput" in line and "pct_of_peak" in line:
            val = _extract_last_number(line)
            if val is not None:
                sol["sol_compute_pct"] = val

        # Memory throughput
        if "compute_memory_throughput" in line and "pct_of_peak" in line:
            val = _extract_last_number(line)
            if val is not None:
                sol["sol_memory_pct"] = val

    return sol


def extract_ncu_metric(parsed: list[dict[str, Any]], metric_name: str) -> float | None:
    """Extract a specific metric value from parsed ncu output.

    Uses three-level matching:
      1. Exact match
      2. Suffix match (e.g., "dram__bytes.sum" matches "...dram__bytes.sum")
      3. Fuzzy substring match
    """
    # Level 1: Exact match
    for entry in parsed:
        if entry["name"] == metric_name:
            return entry["value"]

    # Level 2: Suffix match — handles when ncu prefixes with kernel/section info
    for entry in parsed:
        if entry["name"].endswith(metric_name) or metric_name.endswith(entry["name"]):
            return entry["value"]

    # Level 3: Fuzzy substring match
    for entry in parsed:
        if metric_name in entry["name"] or entry["name"] in metric_name:
            return entry["value"]

    # Level 4: Match by key parts (e.g., "dram__throughput" matches
    # "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    parts = metric_name.split(".")
    if len(parts) > 1:
        base = parts[0]
        for entry in parsed:
            if base in entry["name"]:
                return entry["value"]

    return None


def extract_all_matching(
    parsed: list[dict[str, Any]], pattern: str
) -> list[dict[str, Any]]:
    """Extract all metrics matching a regex pattern."""
    regex = re.compile(pattern)
    return [e for e in parsed if regex.search(e["name"])]


def summarize_ncu_metrics(parsed: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a compact summary of parsed ncu metrics.

    Groups metrics by category (compute, memory, occupancy, etc.)
    """
    summary: dict[str, list[dict]] = {
        "compute": [],
        "memory": [],
        "occupancy": [],
        "instructions": [],
        "other": [],
    }

    for entry in parsed:
        name = entry["name"]
        if any(k in name for k in ("sm__throughput", "pipe_fma", "pipe_tensor", "pipe_alu")):
            summary["compute"].append(entry)
        elif any(k in name for k in ("dram__", "l1tex__", "l2__", "lts__", "memory_throughput")):
            summary["memory"].append(entry)
        elif "warp" in name or "occupancy" in name.lower():
            summary["occupancy"].append(entry)
        elif "inst" in name or "sass" in name:
            summary["instructions"].append(entry)
        else:
            summary["other"].append(entry)

    return {
        "total_metrics": len(parsed),
        "categories": {k: len(v) for k, v in summary.items()},
        "grouped": summary,
    }


def _parse_ncu_text(output: str) -> list[dict[str, Any]]:
    """Fallback: parse ncu text/table output for metric lines."""
    results = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue

        # Pattern 1: metric_name    value    unit
        m = re.match(
            r"([\w.]+(?:__[\w.]+)+)\s+([\d.,]+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
            r"(%|bytes|cycles|GB/s|KB|MB|GB|instructions|warps|sectors|requests)?",
            line,
        )
        if m:
            results.append({
                "name": m.group(1),
                "value": _parse_numeric(m.group(2)),
                "value_raw": m.group(2),
                "unit": (m.group(3) or "").strip(),
            })
            continue

        # Pattern 2: "  metric_name    value  unit  description"
        # (ncu default text output with leading spaces)
        m2 = re.match(
            r"\s{2,}([\w.]+(?:__[\w.]+)*(?:\.[\w.]+)*)\s+"
            r"([\d.,]+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            line,
        )
        if m2:
            results.append({
                "name": m2.group(1),
                "value": _parse_numeric(m2.group(2)),
                "value_raw": m2.group(2),
                "unit": "",
            })

    return results


def _parse_numeric(s: str) -> float | None:
    """Parse a numeric string, handling commas and scientific notation."""
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _extract_last_number(line: str) -> float | None:
    """Extract the last number from a line."""
    numbers = re.findall(r"[\d.,]+(?:\.\d+)?", line)
    if numbers:
        return _parse_numeric(numbers[-1])
    return None
