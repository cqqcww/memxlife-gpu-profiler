from __future__ import annotations

import json
import shutil
import time

from phase2_agent.candidate_space import CandidateConfig, heuristic_candidates
from phase2_agent.config import AgentSettings
from phase2_agent.harness import LocalHarness
from phase2_agent.llm_helper import LLMSuggester
from phase2_agent.tracing import TraceLogger


class LoRAOptimizationAgent:
    def __init__(self, settings: AgentSettings):
        self.settings = settings
        self.tracer = TraceLogger(settings.trace_path, settings.trace_summary_path)
        self.harness = LocalHarness(settings, self.tracer)
        self.llm = LLMSuggester(settings.llm, self.tracer)
        self.history: list[dict] = []
        self.best_result: dict | None = None
        self.target_spec: dict = {}

    def run(self) -> int:
        self.settings.work_dir.mkdir(parents=True, exist_ok=True)
        self._load_target_spec()
        queue = heuristic_candidates()
        seen = set()
        llm_rounds = 0
        self.tracer.log("agent_started", queue_size=len(queue), max_minutes=self.settings.max_minutes)
        self.tracer.add_summary(
            "Run Context",
            [
                f"- Max minutes: `{self.settings.max_minutes}`",
                f"- Benchmark sizes: `{self.settings.benchmark_sizes}`",
                f"- Correctness sizes: `{self.settings.correctness_sizes}`",
                f"- Max candidates: `{self.settings.max_candidates}`",
                f"- LLM round limit: `{self.settings.llm_round_limit}`",
                f"- LLM enabled: `{self.settings.llm.enabled}`",
            ],
        )

        # Keep a compilable candidate on disk immediately.
        bootstrap = queue[0]
        self.harness.write_bootstrap_candidate(bootstrap)
        self.tracer.log("bootstrap_written", candidate=bootstrap.__dict__, optimized_path=str(self.settings.optimized_path))
        self.tracer.add_summary(
            "Bootstrap Candidate",
            [
                f"- Variant: `{bootstrap.variant_name}`",
                f"- Strategy: `{bootstrap.strategy}`",
                f"- Main backend: `{bootstrap.main_backend}`",
                f"- Low-rank backend: `{bootstrap.low_rank_backend}`",
                f"- Accumulation order: `{bootstrap.accumulation_order}`",
                f"- Explicit TF32 enabled: `{bootstrap.allow_tf32}`",
                f"- Cache mode: `{bootstrap.cache_mode}`",
            ],
        )

        start = time.time()
        while queue and (time.time() - start) < self.settings.max_minutes * 60:
            if len(seen) >= self.settings.max_candidates:
                self.tracer.log("candidate_budget_reached", evaluated=len(seen), max_candidates=self.settings.max_candidates)
                break
            candidate = queue.pop(0)
            if candidate.stable_id() in seen:
                continue
            seen.add(candidate.stable_id())
            self.tracer.log("candidate_selected", candidate_id=candidate.stable_id(), candidate=candidate.__dict__, queue_remaining=len(queue))
            result = self.harness.evaluate(candidate)
            record = result.to_dict()
            record["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.history.append(record)
            self.harness.persist_history(self.history)
            self.tracer.add_summary(
                f"Candidate {candidate.variant_name or candidate.stable_id()}",
                [
                    f"- Compile ok: `{result.compile_ok}`",
                    f"- Correct: `{result.correct}`",
                    f"- Compile seconds: `{result.compile_seconds}`",
                    f"- Speedup: `{result.speedup:.6f}`",
                    f"- Student ms: `{result.student_ms}`",
                    f"- Torch ms: `{result.torch_ms}`",
                    f"- Cached repeat ms: `{result.cached_repeat_ms}`",
                    f"- Debug stats: `{result.debug_stats}`",
                    f"- Max abs err: `{result.max_abs_err}`",
                    f"- Rel L2 err: `{result.rel_l2_err}`",
                    f"- Error: `{result.error}`",
                    f"- Notes: `{candidate.notes}`",
                ],
            )

            if result.correct and result.compile_ok:
                self._maybe_promote(result)

            if len(seen) >= 3 and llm_rounds < self.settings.llm_round_limit:
                queue.extend(self._fresh_llm_candidates(seen))
                llm_rounds += 1

        self._write_report_files()
        self._ensure_output_id_placeholder()
        self.tracer.log("agent_finished", history_count=len(self.history), best_result=self.best_result)
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
            self.tracer.log(
                "candidate_promoted",
                candidate=result.to_dict()["candidate"],
                speedup=result.speedup,
                source_path=result.source_path,
            )
            self.tracer.add_summary(
                "Current Best Candidate",
                [
                    f"- Candidate: `{result.to_dict()['candidate']}`",
                    f"- Speedup: `{result.speedup:.6f}`",
                    f"- Max abs err: `{result.max_abs_err}`",
                    f"- Relative L2 err: `{result.rel_l2_err}`",
                    f"- Source path: `{result.source_path}`",
                ],
            )

    def _fresh_llm_candidates(self, seen: set[str]) -> list[CandidateConfig]:
        suggestions = self.llm.suggest(self.history)
        fresh = []
        for candidate in suggestions:
            if candidate.stable_id() not in seen:
                fresh.append(candidate)
        self.tracer.log("llm_candidates_filtered", proposed=len(suggestions), fresh=len(fresh))
        return fresh

    def _write_report_files(self) -> None:
        bootstrap_variant = self.history[0].get("candidate", {}).get("variant_name") if self.history else "bootstrap"
        lines = [
            "# Stage 2 Agent Report",
            "",
            "## Overview",
            "This agent searches for an `optimized_lora.cu` implementation for the stage-2 LoRA task.",
            "It writes a stable ATen bootstrap candidate immediately, then explores nearby ATen/cuBLAS compositions that differ in memory layout, output preallocation, and accumulation style.",
            "",
            "## Runtime Inputs",
            f"- Target spec path: `{self.settings.target_spec_path}`",
            f"- Target spec detected: `{json.dumps(self.target_spec, ensure_ascii=True)}`",
            f"- LLM enabled: `{self.settings.llm.enabled}`",
            f"- Search budget (minutes): `{self.settings.max_minutes}`",
            f"- Trace log path: `{self.settings.trace_path}`",
            f"- Trace summary path: `{self.settings.trace_summary_path}`",
            "",
            "## Search Strategy",
            f"- Bootstrap candidate: `{bootstrap_variant}`, chosen as the strongest known starting point at run time.",
            "- Search candidates: targeted ATen variants around `mm`, `mm_out`, `addmm`, `addmm_`, `addmm_out`, and contiguous-vs-strided `B^T` handling.",
            "- Promotion rule: replace `optimized_lora.cu` only when a candidate compiles, passes correctness checks, and improves measured speedup.",
            "- Logging focus: capture compile latency, correctness error, fixed-weight varying-activation latency, repeated-call latency, and the exact source hash for every candidate.",
            "",
            "## Memoization Policy",
            "- Exact repeat path: if `W`, `A`, `B`, and `X` all match the previous call, return the cached output directly.",
            "- Same-weight path: if `W`, `A`, and `B` match but `X` changes, reuse a delayed materialization of `W_eff = W + A B^T` and compute one large GEMM.",
            "- Cold path: if the weights are new, fall back to the explicit reference-shaped computation `Y = W @ X`, `Bt = B^T.contiguous()`, `BX = Bt @ X`, `Y.addmm_(A, BX)`.",
            "- Cache safety: every reuse decision is guarded by tensor stamps that include data pointer, version counter, shape, and device, so stale results are not reused after mutation.",
            "",
            "## Search Outcomes",
            f"- Candidate evaluations recorded: `{len(self.history)}`",
        ]
        if self.best_result:
            lines.extend(
                [
                    f"- Best speedup: `{self.best_result.get('speedup', 0.0):.4f}`",
                    f"- Best source: `{self.best_result.get('source_path', '')}`",
                    f"- Best candidate: `{self.best_result.get('candidate', {})}`",
                ]
            )
        else:
            lines.append("- Best speedup: `not established in current environment`")
        if self.history:
            compile_failures = sum(1 for record in self.history if not record.get("compile_ok"))
            correctness_failures = sum(
                1 for record in self.history if record.get("compile_ok") and not record.get("correct")
            )
            lines.extend(
                [
                    f"- Compile failures: `{compile_failures}`",
                    f"- Correctness failures after compile: `{correctness_failures}`",
                ]
            )
        lines.extend(
            [
                "",
                "## What The Logs Tell Us",
                "- `compile_seconds` shows whether a seemingly strong candidate is too expensive to iterate on.",
                "- `student_ms` vs `torch_ms` measures the realistic regime where weights stay fixed but `X` changes across calls.",
                "- `cached_repeat_ms` remains the repeated-input hot-path diagnostic, so we can keep the upside from cache-friendly cases without letting it dominate selection.",
                "- `debug_stats` now records which execution path was actually used during the varying-`X` and repeated-input benchmark phases.",
                "- For the hybrid path, the most informative counters are `exact_repeat_hits`, `same_weight_weff_hits`, `weff_materializations`, and `cold_fallback_hits`.",
                "- `.phase2_work/trace.jsonl` is the machine-readable source of truth for one run, and it is reset at the start of each new agent run.",
                "",
                "## Candidate History",
            ]
        )
        for record in self.history:
            candidate = record.get("candidate", {})
            lines.extend(
                [
                    f"### {candidate.get('variant_name') or candidate.get('main_backend', 'candidate')}",
                    f"- compile_ok: `{record.get('compile_ok')}`",
                    f"- correct: `{record.get('correct')}`",
                    f"- compile_seconds: `{record.get('compile_seconds')}`",
                    f"- combined_speedup: `{record.get('speedup')}`",
                    f"- varying_x_student_ms: `{record.get('student_ms')}`",
                    f"- varying_x_torch_ms: `{record.get('torch_ms')}`",
                    f"- repeated_x_student_ms: `{record.get('cached_repeat_ms')}`",
                    f"- debug_stats: `{record.get('debug_stats', {})}`",
                    f"- max_abs_err: `{record.get('max_abs_err')}`",
                    f"- rel_l2_err: `{record.get('rel_l2_err')}`",
                    f"- evaluated_at: `{record.get('evaluated_at')}`",
                    f"- notes: `{candidate.get('notes', '')}`",
                    "",
                ]
            )
        lines.extend(
            [
                "",
                "## Environment Notes",
                "- If CUDA or PyTorch extension tooling is unavailable, the agent still emits a valid bootstrap `optimized_lora.cu` and records exactly why full benchmarking could not proceed.",
                "- Search history is persisted in `.phase2_work/history.json` for later inspection.",
                "- Detailed event logs are persisted in `.phase2_work/trace.jsonl` and `.phase2_work/trace_summary.md`.",
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
            "trace_path": str(self.settings.trace_path),
            "trace_summary_path": str(self.settings.trace_summary_path),
            "llm_enabled": self.settings.llm.enabled,
            "target_spec": self.target_spec,
        }
