from __future__ import annotations

import shutil
import time
from dataclasses import asdict

from phase2_agent.candidate_space import CandidateConfig, heuristic_candidates
from phase2_agent.codegen import render_candidate
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

    def run(self) -> int:
        self.settings.work_dir.mkdir(parents=True, exist_ok=True)
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

        return 0

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

    def summary(self) -> dict:
        return {
            "history_count": len(self.history),
            "best_result": self.best_result,
            "optimized_path": str(self.settings.optimized_path),
            "llm_enabled": self.settings.llm.enabled,
        }

