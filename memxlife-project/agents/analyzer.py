"""Analyzer agent — interprets probe results, assesses confidence, updates KB."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import BaseAgent
from core.models import AgentContext, ProbeResult, Task
from llm.client import LLMClient
from llm.prompts import ANALYZER_SYSTEM, ANALYZER_USER
from parser.probe_parser import extract_primary_value, parse_probe_output
from parser.ncu_parser import parse_ncu_csv, extract_ncu_metric

logger = logging.getLogger(__name__)

# Physical sanity bounds for calibration layer
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "dram_latency_cycles": (200, 1500),
    "l1_latency_cycles": (10, 80),
    "l2_latency_cycles": (100, 500),
    "l2_cache_size_kb": (256, 131072),
    "max_global_mem_bandwidth_gb_s": (10, 5000),
    "max_shmem_bandwidth_gb_s": (1, 200000),
    "actual_boost_clock_mhz": (100, 3500),
    "bank_conflict_penalty_cycles": (0, 100),
    "max_shmem_per_block_kb": (16, 228),
}


class AnalyzerAgent(BaseAgent):
    name = "analyzer"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def can_handle(self, task: Task) -> bool:
        return task.kind == "analyze_result"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        metric_name = task.payload["metric_name"]
        strategy_name = task.payload.get("strategy_name", "unknown")
        stdout = task.payload.get("stdout", "")
        stderr = task.payload.get("stderr", "")
        ncu_output = task.payload.get("ncu_output", "")
        previous_results = task.payload.get("previous_results", [])
        run_success = task.payload.get("success", False)

        # Mock mode — deterministic parse only, high confidence
        if ctx.mock_mode:
            parsed = parse_probe_output(stdout)
            value = extract_primary_value(parsed, metric_name)
            calibration = self._calibrate(metric_name, value)
            confidence = 0.9 if (value is not None and calibration["in_bounds"]) else 0.3
            return {
                "metric_name": metric_name,
                "strategy_name": strategy_name,
                "value": value,
                "unit": parsed.get("unit", ""),
                "confidence": confidence,
                "method": parsed.get("method", strategy_name),
                "needs_retry": False,
                "reasoning": "Mock mode — deterministic parse",
                "anomalies": [],
                "calibration": calibration,
                "probe_result": ProbeResult(
                    metric_name=metric_name, value=value,
                    unit=parsed.get("unit", ""), confidence=confidence,
                    method=parsed.get("method", strategy_name),
                    strategy_name=strategy_name,
                ).to_dict(),
            }

        # If execution failed, return immediately
        if not run_success:
            error_msg = task.payload.get("error", "execution failed")
            return {
                "metric_name": metric_name,
                "strategy_name": strategy_name,
                "value": None,
                "confidence": 0.0,
                "needs_retry": True,
                "retry_reason": f"Execution failed: {error_msg}",
                "error": error_msg,
            }

        # Step 1: Deterministic parsing
        parsed = parse_probe_output(stdout)
        deterministic_value = extract_primary_value(parsed, metric_name)

        # Step 2: Parse ncu output if available
        ncu_value = None
        if ncu_output:
            ncu_parsed = parse_ncu_csv(ncu_output)
            ncu_metrics = task.payload.get("ncu_metrics", [])
            for m in ncu_metrics:
                v = extract_ncu_metric(ncu_parsed, m)
                if v is not None:
                    ncu_value = v
                    break

        # Step 3: Calibration layer — physics sanity check
        calibration = self._calibrate(metric_name, deterministic_value)

        # Step 4: LLM analysis for confidence and reasoning
        llm_analysis = self._llm_analyze(
            metric_name=metric_name,
            strategy_name=strategy_name,
            stdout=stdout,
            stderr=stderr,
            ncu_output=ncu_output,
            previous_results=previous_results,
            ctx=ctx,
        )

        # Step 5: Synthesize final result
        final_value = deterministic_value
        confidence = 0.0

        if llm_analysis.get("extracted_value") is not None:
            llm_value = llm_analysis["extracted_value"]
            # If deterministic parse succeeded, prefer it but cross-check
            if deterministic_value is not None:
                # Check agreement
                if deterministic_value != 0 and abs(llm_value - deterministic_value) / abs(deterministic_value) < 0.1:
                    confidence = min(llm_analysis.get("confidence", 0.5) + 0.1, 1.0)
                else:
                    # Disagreement — trust deterministic parse
                    confidence = max(llm_analysis.get("confidence", 0.3) - 0.2, 0.1)
            else:
                final_value = llm_value
                confidence = llm_analysis.get("confidence", 0.4)
        elif deterministic_value is not None:
            confidence = 0.5  # Parsed but no LLM confirmation

        # Apply calibration penalty
        if not calibration["in_bounds"]:
            confidence = min(confidence, 0.3)

        # ncu cross-verification bonus
        if ncu_value is not None and final_value is not None and final_value != 0:
            if abs(ncu_value - final_value) / abs(final_value) < 0.15:
                confidence = min(confidence + 0.15, 1.0)

        needs_retry = llm_analysis.get("needs_retry", confidence < 0.5)

        # Build ProbeResult
        probe_result = ProbeResult(
            metric_name=metric_name,
            value=final_value,
            unit=parsed.get("unit", llm_analysis.get("unit", "")),
            confidence=round(confidence, 3),
            method=parsed.get("method", strategy_name),
            strategy_name=strategy_name,
            raw_stdout=stdout[:5000],
            raw_stderr=stderr[:2000],
            ncu_output=ncu_output[:5000],
        )

        return {
            "metric_name": metric_name,
            "strategy_name": strategy_name,
            "value": final_value,
            "unit": probe_result.unit,
            "confidence": probe_result.confidence,
            "method": probe_result.method,
            "needs_retry": needs_retry,
            "retry_reason": llm_analysis.get("retry_reason", ""),
            "suggested_strategy": llm_analysis.get("suggested_strategy", ""),
            "reasoning": llm_analysis.get("reasoning", ""),
            "anomalies": llm_analysis.get("anomalies", []) + calibration.get("warnings", []),
            "deterministic_value": deterministic_value,
            "ncu_value": ncu_value,
            "calibration": calibration,
            "probe_result": probe_result.to_dict(),
        }

    def _calibrate(self, metric_name: str, value: float | None) -> dict[str, Any]:
        """Physics sanity check against known bounds."""
        if value is None:
            return {"in_bounds": False, "warnings": ["No value to calibrate"]}

        bounds = PHYSICAL_BOUNDS.get(metric_name)
        if bounds is None:
            return {"in_bounds": True, "warnings": []}

        lo, hi = bounds
        warnings = []
        in_bounds = True

        if value < lo:
            warnings.append(f"{metric_name}={value} is below physical minimum ({lo})")
            in_bounds = False
        elif value > hi:
            warnings.append(f"{metric_name}={value} is above physical maximum ({hi})")
            in_bounds = False

        return {"in_bounds": in_bounds, "warnings": warnings, "bounds": [lo, hi]}

    def _llm_analyze(
        self,
        metric_name: str,
        strategy_name: str,
        stdout: str,
        stderr: str,
        ncu_output: str,
        previous_results: list,
        ctx: AgentContext,
    ) -> dict[str, Any]:
        """Use LLM to interpret results and assess confidence."""
        prev_json = ""
        if previous_results:
            prev_json = json.dumps(previous_results[-3:], indent=2, default=str)

        user_prompt = ANALYZER_USER.format(
            metric_name=metric_name,
            strategy_name=strategy_name,
            method_description=strategy_name,
            stdout=stdout[:4000],
            stderr=stderr[:1000],
            ncu_output=ncu_output[:3000] if ncu_output else "Not available",
            environment_summary=ctx.environment.summary_for_prompt(),
            previous_results=prev_json or "None",
        )

        try:
            raw = self.llm.complete(
                system_prompt=ANALYZER_SYSTEM,
                user_prompt=user_prompt,
                model=self.llm.config.analyzer_model,
            )
            return _extract_json(raw)
        except Exception as e:
            logger.warning("Analyzer LLM call failed: %s", e)
            return {
                "extracted_value": None,
                "confidence": 0.3,
                "reasoning": f"LLM analysis unavailable: {e}",
                "anomalies": [],
                "needs_retry": True,
                "retry_reason": "LLM analysis failed",
            }


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response."""
    import re
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}
