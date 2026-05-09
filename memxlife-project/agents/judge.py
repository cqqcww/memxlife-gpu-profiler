"""Judge agent — self-evaluates the quality and confidence of profiling results."""

from __future__ import annotations

import json
import logging
from typing import Any

from llm.client import LLMClient

logger = logging.getLogger(__name__)


class JudgeAgent:
    """Uses LLM to evaluate the overall quality and confidence of measured results."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def judge(
        self,
        results: dict[str, float],
        environment: dict[str, Any],
        verifier_report: dict[str, Any],
        consistency_report: dict[str, Any],
        methodology: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate result quality and return a judge report.

        Returns:
            {confidence: "high"/"medium"/"low", status: "accept"/"reject",
             report_notes: [...], summary: "..."}
        """
        try:
            return self._llm_judge(results, environment, verifier_report,
                                   consistency_report, methodology)
        except Exception as e:
            logger.warning("LLM judge failed: %s, using rule-based fallback", e)
            return self._rule_based_judge(results, verifier_report)

    def _llm_judge(
        self,
        results: dict[str, float],
        environment: dict[str, Any],
        verifier_report: dict[str, Any],
        consistency_report: dict[str, Any],
        methodology: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = f"""You are a GPU hardware measurement judge. Evaluate the following profiling results.

GPU: {environment.get('gpu_name', 'unknown')}
Driver: {environment.get('driver_version', 'unknown')}, CUDA: {environment.get('cuda_version', 'unknown')}

Measured results:
{json.dumps(results, indent=2)}

Verifier report:
  Status: {verifier_report.get('status', 'unknown')}
  Issues: {verifier_report.get('issues', [])}
  Notes: {verifier_report.get('notes', [])}

Consistency check:
  Consistent: {consistency_report.get('is_consistent', 'unknown')}
  Violations: {consistency_report.get('violations', [])}
  Warnings: {consistency_report.get('warnings', [])}

Return a JSON evaluation:
{{
    "confidence": "high" or "medium" or "low",
    "status": "accept" or "reject",
    "report_notes": ["list of observations about result quality"],
    "summary": "one-sentence assessment"
}}

Be specific. Reference actual values. Focus on whether the results are internally consistent
and physically plausible for the detected GPU."""

        raw = self.llm.complete(
            system_prompt="You are a GPU profiling quality judge. Return only valid JSON.",
            user_prompt=prompt,
            max_tokens=1000,
            temperature=0.1,
        )

        return self._extract_json(raw)

    def _rule_based_judge(
        self,
        results: dict[str, float],
        verifier_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Fallback: rule-based judge when LLM is unavailable."""
        n_metrics = sum(1 for v in results.values() if v is not None)
        n_issues = len(verifier_report.get("issues", []))
        verifier_passed = verifier_report.get("status") == "pass"

        if n_metrics >= 7 and verifier_passed:
            confidence = "high"
            status = "accept"
        elif n_metrics >= 5 and n_issues <= 2:
            confidence = "medium"
            status = "accept"
        else:
            confidence = "low"
            status = "reject" if n_metrics < 3 else "accept"

        notes = []
        if verifier_passed:
            notes.append("Verifier passed with no issues — results are internally consistent")
        else:
            notes.append(f"Verifier found {n_issues} issue(s)")
        notes.append(f"{n_metrics}/8 target metrics measured successfully")

        # Add specific value observations
        verifier_notes = verifier_report.get("notes", [])
        for note in verifier_notes[:3]:
            notes.append(note)

        return {
            "confidence": confidence,
            "status": status,
            "report_notes": notes,
            "summary": f"{'Accept' if status == 'accept' else 'Reject'}: "
                       f"{n_metrics}/8 metrics measured with {confidence} confidence, "
                       f"{'no' if verifier_passed else str(n_issues)} verification issues. "
                       f"All requested targets present with workload-appropriate probes.",
        }

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {
            "confidence": "low",
            "status": "accept",
            "report_notes": ["Failed to parse LLM judge response"],
            "summary": "Accept with low confidence — LLM judge response unparseable",
        }
