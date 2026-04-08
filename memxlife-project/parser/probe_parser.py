"""Parser for micro-benchmark probe stdout output.

Expected output format from probes:
    RESULT:<metric_name>=<value>
    UNIT:<unit>
    METHOD:<description>
    ITERATIONS:<n>
    WARMUP:<n>
    ERROR:<message>  (optional)
"""

from __future__ import annotations

import re
from typing import Any


def parse_probe_output(stdout: str) -> dict[str, Any]:
    """Parse key=value probe output into a structured dict."""
    result: dict[str, Any] = {
        "values": {},
        "unit": "",
        "method": "",
        "iterations": 0,
        "warmup": 0,
        "errors": [],
        "raw_lines": [],
    }

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        result["raw_lines"].append(line)

        if line.startswith("RESULT:"):
            payload = line[len("RESULT:"):].strip()
            m = re.match(r"([^=]+)=(.+)", payload)
            if m:
                key = m.group(1).strip()
                try:
                    val = float(m.group(2).strip())
                except ValueError:
                    val = m.group(2).strip()
                result["values"][key] = val
        elif line.startswith("UNIT:"):
            result["unit"] = line[len("UNIT:"):].strip()
        elif line.startswith("METHOD:"):
            result["method"] = line[len("METHOD:"):].strip()
        elif line.startswith("ITERATIONS:"):
            try:
                result["iterations"] = int(line[len("ITERATIONS:"):].strip())
            except ValueError:
                pass
        elif line.startswith("WARMUP:"):
            try:
                result["warmup"] = int(line[len("WARMUP:"):].strip())
            except ValueError:
                pass
        elif line.startswith("ERROR:"):
            result["errors"].append(line[len("ERROR:"):].strip())

    return result


def extract_primary_value(parsed: dict[str, Any], metric_name: str) -> float | None:
    """Extract the primary measurement value for a given metric."""
    values = parsed.get("values", {})
    # Direct match
    if metric_name in values:
        v = values[metric_name]
        return float(v) if isinstance(v, (int, float)) else None
    # Try partial match
    for key, val in values.items():
        if metric_name in key or key in metric_name:
            return float(val) if isinstance(val, (int, float)) else None
    # If only one value, return it
    if len(values) == 1:
        v = next(iter(values.values()))
        return float(v) if isinstance(v, (int, float)) else None
    return None
