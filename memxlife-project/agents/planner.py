"""Planner agent — selects probe strategy for each target metric."""

from __future__ import annotations

import json
from typing import Any

from agents.base import BaseAgent
from core.models import AgentContext, Task
from llm.client import LLMClient
from llm.prompts import PLANNER_SYSTEM, PLANNER_USER


class PlannerAgent(BaseAgent):
    name = "planner"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def can_handle(self, task: Task) -> bool:
        return task.kind == "plan_probe"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        metric_name = task.payload["metric_name"]
        metric_spec = task.payload["metric_spec"]
        previous_attempts = task.payload.get("previous_attempts", [])
        kb_summary = task.payload.get("kb_summary", "")

        strategies_json = json.dumps(
            [s.to_dict() for s in metric_spec.strategies], indent=2
        )
        previous_json = json.dumps(previous_attempts, indent=2, default=str) if previous_attempts else "None"

        user_prompt = PLANNER_USER.format(
            metric_name=metric_name,
            metric_description=metric_spec.description,
            strategies_json=strategies_json,
            environment_summary=ctx.environment.summary_for_prompt(),
            previous_attempts=previous_json,
            kb_summary=kb_summary or "No prior measurements.",
        )

        try:
            raw = self.llm.complete_for_agent(
                agent_role="planner",
                system_prompt=PLANNER_SYSTEM,
                user_prompt=user_prompt,
            )
            decision = _extract_json(raw)
        except Exception as e:
            # Fallback: pick highest priority strategy not yet tried
            decision = self._heuristic_fallback(metric_spec, previous_attempts)
            decision["fallback_reason"] = str(e)

        decision["metric_name"] = metric_name
        return decision

    def _heuristic_fallback(
        self, metric_spec: Any, previous_attempts: list[dict]
    ) -> dict[str, Any]:
        """Deterministic fallback when LLM is unavailable."""
        tried = {a.get("strategy_name", "") for a in previous_attempts}
        for strategy in sorted(metric_spec.strategies, key=lambda s: s.priority):
            if strategy.name not in tried:
                return {
                    "selected_strategy": strategy.name,
                    "params_override": {},
                    "needs_ncu": strategy.needs_ncu,
                    "ncu_metrics": strategy.ncu_metrics,
                    "reasoning": f"Heuristic fallback: selecting next untried strategy '{strategy.name}'",
                    "cross_verify": metric_spec.cross_verify,
                }
        # All tried — retry first strategy
        first = metric_spec.strategies[0] if metric_spec.strategies else None
        return {
            "selected_strategy": first.name if first else "unknown",
            "params_override": {},
            "needs_ncu": first.needs_ncu if first else False,
            "ncu_metrics": first.ncu_metrics if first else [],
            "reasoning": "All strategies exhausted, retrying first.",
            "cross_verify": False,
        }


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response."""
    import re
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try code block
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try first { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"error": "Could not parse LLM response", "raw": text[:500]}
