"""Probe template registry — loads and parameterizes CUDA micro-benchmark templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

TEMPLATES_DIR = Path(__file__).parent / "templates"


def list_templates() -> list[str]:
    """List available probe template names (without .cu extension)."""
    if not TEMPLATES_DIR.exists():
        return []
    return [f.stem for f in TEMPLATES_DIR.glob("*.cu") if f.stem != "common"]


def load_template(name: str) -> str:
    """Load a probe template source code by name."""
    path = TEMPLATES_DIR / f"{name}.cu"
    if not path.exists():
        raise FileNotFoundError(f"Probe template not found: {path}")
    return path.read_text(encoding="utf-8")


def load_common_header() -> str:
    """Load the common.cuh header if it exists."""
    path = TEMPLATES_DIR / "common.cuh"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def parameterize_template(template_code: str, params: dict[str, Any]) -> str:
    """Replace {{param_name}} placeholders in template with actual values."""
    result = template_code
    for key, value in params.items():
        placeholder = "{{" + key + "}}"
        result = result.replace(placeholder, str(value))
    return result


def get_template_info(name: str) -> dict[str, Any]:
    """Get metadata about a template (extractable params, description)."""
    try:
        code = load_template(name)
    except FileNotFoundError:
        return {"name": name, "exists": False}

    # Extract parameter placeholders
    import re
    params = list(set(re.findall(r"\{\{(\w+)\}\}", code)))

    # Extract description from first comment block
    desc = ""
    for line in code.splitlines():
        if line.strip().startswith("//"):
            desc = line.strip().lstrip("/ ").strip()
            break

    return {
        "name": name,
        "exists": True,
        "params": params,
        "description": desc,
        "line_count": len(code.splitlines()),
    }
