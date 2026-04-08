"""Run state management — tracks progress across the entire profiling session."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core.models import EnvironmentProfile, ProbeResult


class RunState:
    """Persistent state for a single profiling run."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.state_file = run_dir / "run_state.json"
        self._state: dict[str, Any] = {
            "status": "initialized",
            "created_at": time.time(),
            "environment": {},
            "targets": [],
            "results": {},       # metric_name -> best ProbeResult dict
            "all_attempts": {},   # metric_name -> list of ProbeResult dicts
            "claims": [],
            "audit_log": [],
        }
        if self.state_file.exists():
            self._load()

    def _load(self) -> None:
        try:
            self._state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self._state, indent=2, default=str),
            encoding="utf-8",
        )

    @property
    def status(self) -> str:
        return self._state.get("status", "initialized")

    @status.setter
    def status(self, value: str) -> None:
        self._state["status"] = value
        self._state["updated_at"] = time.time()

    def set_environment(self, env: EnvironmentProfile) -> None:
        self._state["environment"] = env.to_dict()

    def get_environment(self) -> dict[str, Any]:
        return self._state.get("environment", {})

    def set_targets(self, targets: list[str]) -> None:
        self._state["targets"] = targets

    def record_attempt(self, result: ProbeResult) -> None:
        name = result.metric_name
        self._state["all_attempts"].setdefault(name, [])
        self._state["all_attempts"][name].append(result.to_dict())

        # Update best result if this one has higher confidence
        current_best = self._state["results"].get(name)
        if current_best is None or result.confidence > current_best.get("confidence", 0):
            self._state["results"][name] = result.to_dict()

    def get_best_result(self, metric_name: str) -> dict[str, Any] | None:
        return self._state["results"].get(metric_name)

    def get_all_attempts(self, metric_name: str) -> list[dict[str, Any]]:
        return self._state["all_attempts"].get(metric_name, [])

    def get_results(self) -> dict[str, Any]:
        return dict(self._state["results"])

    def add_audit_entry(self, entry: dict[str, Any]) -> None:
        entry.setdefault("timestamp", time.time())
        self._state["audit_log"].append(entry)

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self._state["audit_log"])

    def export_results_json(self) -> dict[str, Any]:
        """Export final results.json for submission."""
        out = {}
        for name, result in self._state["results"].items():
            if result.get("value") is not None:
                out[name] = result["value"]
        return out

    def to_dict(self) -> dict[str, Any]:
        return dict(self._state)
