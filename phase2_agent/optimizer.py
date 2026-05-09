from __future__ import annotations

import json
import shutil
import time

from phase2_agent.candidate_space import CandidateConfig, heuristic_candidates
from phase2_agent.config import AgentSettings
from phase2_agent.harness import LocalHarness
from phase2_agent.llm_helper import LLMSuggester


class LoRAOptimizationAgent:
    def __init__(self, settings: AgentSettings):
        self.settings = settings
        self.harness = LocalHarness(settings)
        self.llm = LLMSuggester(settings.llm)
        self.history: list[dict] = []
        self.best_result: dict | None = None
        self.target_spec: dict = {}

    def run(self) -> int:
        self.settings.work_dir.mkdir(parents=True, exist_ok=True)
        self._load_target_spec()
        queue = heuristic_candidates()
        seen = set()

        # Keep a compilable candidate on disk immediately.
        bootstrap = queue[0]
        self.harness.write_bootstrap_candidate(bootstrap)

        start = time.time()
        while queue and (time.time() - start) < self.settings.max_minutes * 60:
            candidate = queue.pop(0)
            if candidate.stable_id() in seen:
                continue
            seen.add(candidate.stable_id())
            result = self.harness.evaluate(candidate)
            record = result.to_dict()
            record["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.history.append(record)
            self.harness.persist_history(self.history)

            if result.correct and result.compile_ok:
                self._maybe_promote(result)

            if len(seen) >= 3:
                queue.extend(self._fresh_llm_candidates(seen))

        self._write_report_files()
        self._ensure_output_id_placeholder()
        return 0

    def _load_target_spec(self) -> None:
        if self.settings.target_spec_path.exists():
            try:
                self.target_spec = json.loads(self.settings.target_spec_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.target_spec = {"raw_path": str(self.settings.target_spec_path), "parse_error": True}
        else:
            self.target_spec = {"raw_path": str(self.settings.target_spec_path), "missing": True}

    def _maybe_promote(self, result) -> None:
        if self.best_result is None or result.speedup > self.best_result["speedup"]:
            self.best_result = result.to_dict()
            shutil.copyfile(result.source_path, self.settings.optimized_path)

    def _fresh_llm_candidates(self, seen: set[str]) -> list[CandidateConfig]:
        suggestions = self.llm.suggest(self.history)
        fresh = []
        for candidate in suggestions:
            if candidate.stable_id() not in seen:
                fresh.append(candidate)
        return fresh

    def _write_report_files(self) -> None:
        lines = [
            "# Stage 2 Agent Report",
            "",
            "## Overview",
            "This agent searches for an `optimized_lora.cu` implementation for the stage-2 LoRA task.",
            "It keeps a safe bootstrap candidate on disk first, then evaluates additional candidates with local correctness and benchmark checks when PyTorch CUDA tooling is available.",
            "",
            "## Runtime Inputs",
            f"- Target spec path: `{self.settings.target_spec_path}`",
            f"- Target spec detected: `{json.dumps(self.target_spec, ensure_ascii=True)}`",
            f"- LLM enabled: `{self.settings.llm.enabled}`",
            f"- Search budget (minutes): `{self.settings.max_minutes}`",
            "",
            "## Search Strategy",
            "- Bootstrap candidate: ATen-based LoRA composition for guaranteed file generation.",
            "- Search candidates: cuBLAS SGEMM / GEMMEx TF32 variants with different accumulation orders.",
            "- Promotion rule: replace `optimized_lora.cu` only when a candidate compiles, passes correctness checks, and improves measured speedup.",
            "",
            "## Search Outcomes",
            f"- Candidate evaluations recorded: `{len(self.history)}`",
        ]
        if self.best_result:
            lines.extend(
                [
                    f"- Best speedup: `{self.best_result.get('speedup', 0.0):.4f}`",
                    f"- Best source: `{self.best_result.get('source_path', '')}`",
                ]
            )
        else:
            lines.append("- Best speedup: `not established in current environment`")
        lines.extend(
            [
                "",
                "## Environment Notes",
                "- If CUDA or PyTorch extension tooling is unavailable, the agent still emits a valid bootstrap `optimized_lora.cu` and records why full benchmarking could not proceed.",
                "- Search history is persisted in `.phase2_work/history.json` for later inspection.",
                "",
                "## Current Summary",
                "```json",
                json.dumps(self.summary(), indent=2),
                "```",
            ]
        )
        report = "\n".join(lines) + "\n"
        self.settings.report_path.write_text(report, encoding="utf-8")
        self.settings.output_log_path.write_text(report, encoding="utf-8")

    def _ensure_output_id_placeholder(self) -> None:
        if self.settings.output_id_path.exists():
            return
        self.settings.output_id_path.write_text(
            "pending-submit2-output-id\n",
            encoding="utf-8",
        )

    def summary(self) -> dict:
        return {
            "history_count": len(self.history),
            "best_result": self.best_result,
            "optimized_path": str(self.settings.optimized_path),
            "llm_enabled": self.settings.llm.enabled,
            "target_spec": self.target_spec,
        }
