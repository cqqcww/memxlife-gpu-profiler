"""Knowledge store — JSON-based storage for probe results and claims."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core.models import ProbeResult


class KnowledgeStore:
    """Persistent knowledge base for the profiling session."""

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self._data: dict[str, Any] = {
            "observations": [],
            "claims": [],
            "environment": {},
            "metric_history": {},  # metric_name -> list of observations
        }
        if store_path.exists():
            try:
                self._data = json.loads(store_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps(self._data, indent=2, default=str), encoding="utf-8"
        )

    def set_environment(self, env_dict: dict[str, Any]) -> None:
        self._data["environment"] = env_dict

    def add_observation(self, result: ProbeResult) -> None:
        entry = {
            **result.to_dict(),
            "recorded_at": time.time(),
        }
        self._data["observations"].append(entry)
        self._data["metric_history"].setdefault(result.metric_name, [])
        self._data["metric_history"][result.metric_name].append(entry)

    def add_claim(self, claim: dict[str, Any]) -> None:
        claim.setdefault("timestamp", time.time())
        claim.setdefault("status", "active")
        self._data["claims"].append(claim)

    def get_metric_history(self, metric_name: str) -> list[dict[str, Any]]:
        return self._data["metric_history"].get(metric_name, [])

    def get_best_for_metric(self, metric_name: str) -> dict[str, Any] | None:
        history = self.get_metric_history(metric_name)
        if not history:
            return None
        return max(history, key=lambda x: x.get("confidence", 0))

    def summary_for_prompt(self) -> str:
        """Compact summary for injection into LLM prompts."""
        lines = []
        env = self._data.get("environment", {})
        if env:
            lines.append(f"GPU: {env.get('gpu_name', 'unknown')}")
            lines.append(f"Trust: {env.get('trust_level', 'unknown')}")

        lines.append(f"Total observations: {len(self._data['observations'])}")
        lines.append(f"Total claims: {len(self._data['claims'])}")

        for metric, history in self._data["metric_history"].items():
            best = max(history, key=lambda x: x.get("confidence", 0))
            val = best.get("value")
            conf = best.get("confidence", 0)
            lines.append(f"  {metric}: value={val}, confidence={conf:.2f}, attempts={len(history)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)
