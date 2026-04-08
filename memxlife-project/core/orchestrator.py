"""Orchestrator — main loop that coordinates all agents to probe GPU metrics."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from agents.analyzer import AnalyzerAgent
from agents.codegen import CodegenAgent
from agents.planner import PlannerAgent
from agents.runner import RunnerAgent
from agents.scout import ScoutAgent
from audit.logger import AuditLogger
from config import Config
from core.models import AgentContext, EnvironmentProfile, ProbeResult, Task
from core.state import RunState
from knowledge.metrics_catalog import get_metric_spec, list_supported_metrics
from knowledge.store import KnowledgeStore
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the multi-agent GPU profiling pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config.llm)

        # Agents
        self.scout = ScoutAgent()
        self.planner = PlannerAgent(self.llm)
        self.codegen = CodegenAgent(self.llm)
        self.runner = RunnerAgent()
        self.analyzer = AnalyzerAgent(self.llm)

    def run(self, target_spec_path: str, output_dir: str | None = None) -> dict[str, Any]:
        """Main entry point: read target_spec.json, probe all metrics, output results.json."""

        # Load targets
        spec = json.loads(Path(target_spec_path).read_text(encoding="utf-8"))
        targets = spec.get("targets", [])
        if not targets:
            raise ValueError("target_spec.json contains no targets")

        logger.info("Loaded %d target metrics: %s", len(targets), targets)

        # Set up run context
        run_id = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(output_dir or self.config.run.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        ctx = AgentContext(
            run_id=run_id,
            run_dir=run_dir,
            mock_mode=self.config.run.mock_mode,
        )

        state = RunState(run_dir)
        state.set_targets(targets)
        state.status = "running"

        kb = KnowledgeStore(run_dir / "knowledge.json")
        audit = AuditLogger(run_dir)

        audit.log("orchestrator", "run_started", detail={"targets": targets, "mock_mode": ctx.mock_mode})

        # ── Phase 0: Environment Scout ───────────────────────────
        logger.info("Phase 0: Environment scouting...")
        scout_task = Task(id="scout-0", kind="scout_environment")
        scout_result = self.scout.run(scout_task, ctx)
        state.set_environment(ctx.environment)
        kb.set_environment(ctx.environment.to_dict())
        audit.log("scout", "environment_detected", detail=scout_result)
        state.save()
        kb.save()

        logger.info("Environment: %s", ctx.environment.summary_for_prompt())

        # ── Phase 1: Probe each target metric ────────────────────
        logger.info("Phase 1: Probing %d metrics...", len(targets))

        for target in targets:
            logger.info("─── Probing: %s ───", target)
            audit.log("orchestrator", "metric_started", metric_name=target)

            self._probe_metric(
                metric_name=target,
                ctx=ctx,
                state=state,
                kb=kb,
                audit=audit,
            )

            state.save()
            kb.save()

        # ── Phase 2: Aggregate results ───────────────────────────
        logger.info("Phase 2: Aggregating results...")
        state.status = "completed"

        final_results = state.export_results_json()

        # Write results.json
        results_path = run_dir / "results.json"
        results_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
        logger.info("Results written to %s", results_path)

        # Write detailed results (with confidence etc.)
        detailed_path = run_dir / "results_detailed.json"
        detailed_path.write_text(
            json.dumps(state.get_results(), indent=2, default=str), encoding="utf-8"
        )

        # Generate audit report
        audit.log("orchestrator", "run_completed", detail={"results": final_results})
        report_path = audit.save_report(
            results=state.get_results(),
            environment=ctx.environment.to_dict(),
        )
        logger.info("Audit report written to %s", report_path)

        state.save()
        kb.save()

        return final_results

    def _probe_metric(
        self,
        metric_name: str,
        ctx: AgentContext,
        state: RunState,
        kb: KnowledgeStore,
        audit: AuditLogger,
    ) -> None:
        """Run the Planner → Codegen → Runner → Analyzer loop for one metric."""

        metric_spec = get_metric_spec(metric_name)
        if metric_spec is None:
            logger.warning("Metric '%s' not in catalog — will rely on LLM", metric_name)
            # Create a minimal spec for unknown metrics
            from core.models import MetricSpec
            metric_spec = MetricSpec(name=metric_name, description=f"Unknown metric: {metric_name}")

        max_retries = self.config.run.max_retries_per_metric
        previous_attempts: list[dict[str, Any]] = []
        previous_errors: list[str] = []

        for attempt in range(max_retries):
            logger.info("  Attempt %d/%d for %s", attempt + 1, max_retries, metric_name)

            # ── Planner ──────────────────────────────────────────
            plan_task = Task(
                id=f"plan-{metric_name}-{attempt}",
                kind="plan_probe",
                payload={
                    "metric_name": metric_name,
                    "metric_spec": metric_spec,
                    "previous_attempts": previous_attempts,
                    "kb_summary": kb.summary_for_prompt(),
                },
            )
            plan_result = self.planner.run(plan_task, ctx)
            strategy_name = plan_result.get("selected_strategy", "unknown")

            audit.log("planner", "strategy_selected", metric_name=metric_name, detail={
                "strategy_name": strategy_name,
                "reasoning": plan_result.get("reasoning", ""),
                "attempt": attempt,
            })
            logger.info("  Planner selected: %s", strategy_name)

            # Find the strategy object
            strategy_dict = plan_result.get("strategy")
            if not strategy_dict:
                # Try to find from spec
                for s in metric_spec.strategies:
                    if s.name == strategy_name:
                        strategy_dict = s.to_dict()
                        break
                if not strategy_dict:
                    strategy_dict = {"name": strategy_name, "probe_template": None, "params": {}}

            # ── Codegen ──────────────────────────────────────────
            codegen_task = Task(
                id=f"codegen-{metric_name}-{attempt}",
                kind="generate_probe",
                payload={
                    "metric_name": metric_name,
                    "strategy": strategy_dict,
                    "params_override": plan_result.get("params_override", {}),
                    "previous_errors": previous_errors,
                },
            )
            codegen_result = self.codegen.run(codegen_task, ctx)

            cuda_code = codegen_result.get("cuda_code", "")
            if not cuda_code:
                error_msg = codegen_result.get("error", "No CUDA code generated")
                logger.warning("  Codegen failed: %s", error_msg)
                previous_errors.append(f"Codegen: {error_msg}")
                previous_attempts.append({
                    "strategy_name": strategy_name,
                    "error": error_msg,
                    "phase": "codegen",
                })
                audit.log("codegen", "generation_failed", metric_name=metric_name, detail={
                    "error": error_msg,
                })
                continue

            audit.log("codegen", "code_generated", metric_name=metric_name, detail={
                "strategy_name": strategy_name,
                "code_lines": len(cuda_code.splitlines()),
                "compile_command": codegen_result.get("compile_command", ""),
            })

            # ── Runner ───────────────────────────────────────────
            runner_task = Task(
                id=f"run-{metric_name}-{attempt}",
                kind="run_probe",
                attempts=attempt,
                payload={
                    "metric_name": metric_name,
                    "strategy_name": strategy_name,
                    "cuda_code": cuda_code,
                    "compile_command": codegen_result.get("compile_command", "nvcc -O2 -o probe probe.cu"),
                    "run_command": codegen_result.get("run_command", "./probe"),
                    "needs_ncu": plan_result.get("needs_ncu", False),
                    "ncu_metrics": plan_result.get("ncu_metrics", []),
                },
            )
            runner_result = self.runner.run(runner_task, ctx)

            if not runner_result.get("success", False):
                error_msg = runner_result.get("error", "execution failed")
                logger.warning("  Runner failed: %s", error_msg)
                previous_errors.append(f"Runner ({runner_result.get('phase', 'unknown')}): {error_msg}")
                previous_attempts.append({
                    "strategy_name": strategy_name,
                    "error": error_msg,
                    "phase": runner_result.get("phase", "unknown"),
                })
                audit.log("runner", "execution_failed", metric_name=metric_name, detail={
                    "error": error_msg,
                    "phase": runner_result.get("phase", ""),
                })
                continue

            audit.log("runner", "execution_succeeded", metric_name=metric_name, detail={
                "strategy_name": strategy_name,
                "elapsed_sec": runner_result.get("elapsed_sec", 0),
            })

            # ── Analyzer ─────────────────────────────────────────
            analyzer_task = Task(
                id=f"analyze-{metric_name}-{attempt}",
                kind="analyze_result",
                payload={
                    "metric_name": metric_name,
                    "strategy_name": strategy_name,
                    "stdout": runner_result.get("stdout", ""),
                    "stderr": runner_result.get("stderr", ""),
                    "ncu_output": runner_result.get("ncu_output", ""),
                    "ncu_metrics": plan_result.get("ncu_metrics", []),
                    "success": True,
                    "previous_results": previous_attempts,
                },
            )
            analysis = self.analyzer.run(analyzer_task, ctx)

            value = analysis.get("value")
            confidence = analysis.get("confidence", 0.0)

            audit.log("analyzer", "analysis_complete", metric_name=metric_name, detail={
                "value": value,
                "confidence": confidence,
                "reasoning": analysis.get("reasoning", ""),
                "anomalies": analysis.get("anomalies", []),
                "needs_retry": analysis.get("needs_retry", False),
            })

            # Record attempt
            probe_result = ProbeResult(
                metric_name=metric_name,
                value=value,
                unit=analysis.get("unit", ""),
                confidence=confidence,
                method=analysis.get("method", strategy_name),
                strategy_name=strategy_name,
            )
            state.record_attempt(probe_result)
            kb.add_observation(probe_result)

            previous_attempts.append({
                "strategy_name": strategy_name,
                "value": value,
                "confidence": confidence,
                "error": analysis.get("error"),
            })

            logger.info(
                "  Result: %s = %s (confidence: %.2f)",
                metric_name, value, confidence,
            )

            # Check if we're done
            if confidence >= self.config.run.confidence_threshold:
                logger.info("  ✓ Confidence threshold met for %s", metric_name)
                audit.log("orchestrator", "metric_completed", metric_name=metric_name, detail={
                    "value": value,
                    "confidence": confidence,
                    "attempts": attempt + 1,
                })
                break

            if not analysis.get("needs_retry", True):
                logger.info("  Analyzer says no retry needed for %s", metric_name)
                break

            logger.info("  Confidence %.2f < %.2f, retrying...", confidence, self.config.run.confidence_threshold)

        else:
            # Exhausted all retries
            logger.warning("  ✗ Max retries exhausted for %s", metric_name)
            audit.log("orchestrator", "metric_exhausted", metric_name=metric_name, detail={
                "max_retries": max_retries,
                "best_result": state.get_best_result(metric_name),
            })
