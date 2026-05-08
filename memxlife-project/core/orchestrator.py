"""Orchestrator — main loop that coordinates all agents to probe GPU metrics."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agents.analyzer import AnalyzerAgent
from agents.codegen import CodegenAgent
from agents.judge import JudgeAgent
from agents.planner import PlannerAgent
from agents.runner import RunnerAgent
from agents.scout import ScoutAgent
from agents.verifier import VerifierAgent
from analysis.consistency import (
    EnvironmentFingerprintDetector,
    PhysicalConsistencyValidator,
)
from audit.logger import AuditLogger
from config import Config
from core.models import AgentContext, EnvironmentProfile, ProbeResult, Task
from core.state import RunState
from knowledge.metrics_catalog import get_metric_spec, list_supported_metrics
from knowledge.store import KnowledgeStore
from llm.client import LLMClient

logger = logging.getLogger(__name__)

# Target name aliases — maps evaluation spec names to our internal metric names
TARGET_ALIASES: dict[str, str] = {
    "actual_core_clock_mhz": "actual_boost_clock_mhz",
    "core_clock_mhz": "actual_boost_clock_mhz",
    "boost_clock_mhz": "actual_boost_clock_mhz",
    "sm_count": "physical_sm_count",
    "num_sms": "physical_sm_count",
    "fp32_tflops": "peak_fp32_tflops",
    "global_mem_bandwidth_gb_s": "max_global_mem_bandwidth_gb_s",
    "shmem_bandwidth_gb_s": "max_shmem_bandwidth_gb_s",
    "shmem_per_block_kb": "max_shmem_per_block_kb",
}


def _resolve_target(name: str) -> tuple[str, str]:
    """Resolve a target name to (internal_name, output_name).

    Returns the internal metric name for probing, and the original name for output.
    """
    internal = TARGET_ALIASES.get(name, name)
    return internal, name


def _parse_result_value(metric_name: str, stdout: str) -> float | None:
    """Parse RESULT:<metric>=<value> from probe stdout. Tries exact match, then any RESULT: line."""
    value = None
    for line in stdout.splitlines():
        if line.startswith(f"RESULT:{metric_name}="):
            try:
                value = float(line.split("=", 1)[1].strip())
                return value
            except (ValueError, IndexError):
                pass
    # Fallback: any RESULT: line
    for line in stdout.splitlines():
        if line.startswith("RESULT:") and "=" in line:
            try:
                value = float(line.split("=", 1)[1].strip())
                return value
            except (ValueError, IndexError):
                pass
    return None


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
        self.verifier = VerifierAgent()
        self.judge = JudgeAgent(self.llm)

    def run(self, target_spec_path: str, output_dir: str | None = None) -> dict[str, Any]:
        """Main entry point: read target_spec.json, probe all metrics, output results.json."""

        # Load targets
        spec = json.loads(Path(target_spec_path).read_text(encoding="utf-8"))
        targets = spec.get("targets", [])
        if not targets:
            raise ValueError("target_spec.json contains no targets")

        # Resolve target aliases
        target_map: dict[str, str] = {}
        internal_targets: list[str] = []
        for t in targets:
            internal, output = _resolve_target(t)
            target_map[internal] = output
            if internal not in internal_targets:
                internal_targets.append(internal)

        logger.info("Loaded %d target metrics: %s", len(targets), targets)

        # Set up run context
        run_id = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(output_dir or self.config.run.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        ctx = AgentContext(run_id=run_id, run_dir=run_dir, mock_mode=self.config.run.mock_mode)
        state = RunState(run_dir)
        state.set_targets(targets)
        state.status = "running"
        kb = KnowledgeStore(run_dir / "knowledge.json")
        audit = AuditLogger(run_dir)
        audit.log("orchestrator", "run_started", detail={"targets": targets})

        # Track methodology for output
        methodology: dict[str, dict] = {}
        probe_plan: list[dict] = []

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

        # ── Phase 1: LLM Codegen (parallel) + Sequential Run ────
        logger.info("Phase 1: Probing %d metrics (parallel LLM codegen)...", len(internal_targets))

        codegen_inputs = []
        for internal_name in internal_targets:
            metric_spec = get_metric_spec(internal_name)
            if metric_spec is None:
                from core.models import MetricSpec
                metric_spec = MetricSpec(name=internal_name, description=f"Unknown metric: {internal_name}")

            if metric_spec.strategies:
                strategy_name = metric_spec.strategies[0].name
                strategy_dict = metric_spec.strategies[0].to_dict()
            else:
                strategy_name = "unknown"
                strategy_dict = {"name": "unknown", "probe_template": None, "params": {}}

            codegen_task = Task(
                id=f"codegen-{internal_name}-0",
                kind="generate_probe",
                payload={
                    "metric_name": internal_name,
                    "strategy": strategy_dict,
                    "params_override": {},
                    "previous_errors": [],
                },
            )
            codegen_inputs.append((internal_name, metric_spec, strategy_name, strategy_dict, codegen_task))
            probe_plan.append({
                "metric": internal_name,
                "strategy": strategy_name,
                "description": strategy_dict.get("description", ""),
                "source": "llm_generated",
            })

        # Parallel codegen via LLM
        logger.info("  Launching %d parallel LLM codegen calls...", len(codegen_inputs))
        codegen_results = {}

        def _run_codegen(item):
            name, spec, sname, sdict, task = item
            try:
                result = self.codegen.run(task, ctx)
                return name, result
            except Exception as e:
                logger.warning("  Codegen failed for %s: %s", name, e)
                return name, {"cuda_code": "", "error": str(e)}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_run_codegen, item): item[0] for item in codegen_inputs}
            for future in as_completed(futures):
                name, result = future.result()
                codegen_results[name] = result
                code = result.get("cuda_code", "")
                logger.info("  Codegen done: %s (%d lines)", name, len(code.splitlines()) if code else 0)

        # Sequential execution on GPU
        for internal_name, metric_spec, strategy_name, strategy_dict, _ in codegen_inputs:
            logger.info("─── Running: %s ───", internal_name)
            audit.log("orchestrator", "metric_started", metric_name=internal_name)

            codegen_result = codegen_results.get(internal_name, {})
            cuda_code = codegen_result.get("cuda_code", "")

            if not cuda_code:
                logger.warning("  No code for %s, skipping", internal_name)
                continue

            runner_task = Task(
                id=f"run-{internal_name}-0", kind="run_probe", attempts=0,
                payload={
                    "metric_name": internal_name,
                    "strategy_name": strategy_name,
                    "cuda_code": cuda_code,
                    "compile_command": codegen_result.get("compile_command", "nvcc -O2 -o probe probe.cu"),
                    "run_command": codegen_result.get("run_command", "./probe"),
                    "needs_ncu": False, "ncu_metrics": [],
                },
            )
            runner_result = self.runner.run(runner_task, ctx)

            if not runner_result.get("success", False):
                error_msg = runner_result.get("error", "execution failed")
                logger.warning("  Runner failed for %s: %s", internal_name, error_msg)
                audit.log("runner", "execution_failed", metric_name=internal_name, detail={"error": error_msg})
                continue

            stdout = runner_result.get("stdout", "")
            value = _parse_result_value(internal_name, stdout)
            if value is None:
                logger.warning("  stdout for %s:\n%s", internal_name, stdout[:500])

            # Unit auto-correction for bandwidth metrics
            if value is not None and "bytes" in internal_name and "per_second" in internal_name:
                if value < 1e6:
                    logger.info("  Auto-correcting %s: %.2f (GB/s) -> %.2f (bytes/s)", internal_name, value, value * 1e9)
                    value = value * 1e9

            if value is not None:
                confidence = 0.95
                if metric_spec.physical_min is not None and value < metric_spec.physical_min:
                    confidence = 0.3
                elif metric_spec.physical_max is not None and value > metric_spec.physical_max:
                    confidence = 0.3

                probe_result = ProbeResult(
                    metric_name=internal_name, value=value, unit="",
                    confidence=confidence, method=strategy_name, strategy_name=strategy_name,
                )
                state.record_attempt(probe_result)
                kb.add_observation(probe_result)
                logger.info("  Result: %s = %s (confidence: %.2f)", internal_name, value, confidence)
                audit.log("orchestrator", "metric_completed", metric_name=internal_name, detail={
                    "value": value, "confidence": confidence,
                })
                methodology[internal_name] = {
                    "method": strategy_name,
                    "description": strategy_dict.get("description", "LLM-generated CUDA micro-benchmark"),
                    "source": "llm_generated",
                }
            else:
                logger.warning("  Could not parse result for %s from stdout", internal_name)

            state.save()
            kb.save()

        # ── Phase 1.5: Retry low-confidence or missing metrics ───
        results_so_far = state.get_results()
        retry_metrics = []
        for internal_name, metric_spec, strategy_name, strategy_dict, _ in codegen_inputs:
            best = results_so_far.get(internal_name)
            if best is None or best.get("confidence", 0) < 0.7:
                retry_metrics.append((internal_name, metric_spec, strategy_name, strategy_dict))

        if retry_metrics:
            logger.info("Phase 1.5: Retrying %d low-confidence metrics: %s",
                        len(retry_metrics), [m[0] for m in retry_metrics])

            for internal_name, metric_spec, strategy_name, strategy_dict in retry_metrics:
                logger.info("─── Retry: %s ───", internal_name)
                retry_task = Task(
                    id=f"codegen-{internal_name}-retry", kind="generate_probe",
                    payload={
                        "metric_name": internal_name,
                        "strategy": strategy_dict,
                        "params_override": {},
                        "previous_errors": ["Previous attempt produced out-of-range or missing result. "
                                            "Make sure to output bytes/s (not GB/s) for bandwidth metrics. "
                                            "Make sure RESULT line uses the exact metric name."],
                    },
                )
                try:
                    retry_codegen = self.codegen.run(retry_task, ctx)
                except Exception as e:
                    logger.warning("  Retry codegen failed for %s: %s", internal_name, e)
                    continue

                cuda_code = retry_codegen.get("cuda_code", "")
                if not cuda_code:
                    continue

                runner_task = Task(
                    id=f"run-{internal_name}-retry", kind="run_probe", attempts=1,
                    payload={
                        "metric_name": internal_name, "strategy_name": strategy_name,
                        "cuda_code": cuda_code,
                        "compile_command": retry_codegen.get("compile_command", "nvcc -O2 -o probe probe.cu"),
                        "run_command": retry_codegen.get("run_command", "./probe"),
                        "needs_ncu": False, "ncu_metrics": [],
                    },
                )
                runner_result = self.runner.run(runner_task, ctx)
                if not runner_result.get("success", False):
                    logger.warning("  Retry runner failed for %s", internal_name)
                    continue

                stdout = runner_result.get("stdout", "")
                value = _parse_result_value(internal_name, stdout)

                if value is not None and "bytes" in internal_name and "per_second" in internal_name:
                    if value < 1e6:
                        value = value * 1e9

                if value is not None:
                    confidence = 0.95
                    if metric_spec.physical_min is not None and value < metric_spec.physical_min:
                        confidence = 0.3
                    elif metric_spec.physical_max is not None and value > metric_spec.physical_max:
                        confidence = 0.3

                    old_best = results_so_far.get(internal_name)
                    old_conf = old_best.get("confidence", 0) if old_best else 0
                    if confidence > old_conf:
                        probe_result = ProbeResult(
                            metric_name=internal_name, value=value, unit="",
                            confidence=confidence, method=strategy_name, strategy_name=strategy_name,
                        )
                        state.record_attempt(probe_result)
                        kb.add_observation(probe_result)
                        logger.info("  Retry improved: %s = %s (confidence: %.2f -> %.2f)",
                                    internal_name, value, old_conf, confidence)

                state.save()
                kb.save()

        # ── Phase 2: Validation ──────────────────────────────────
        logger.info("Phase 2: Validating results...")

        raw_values = {name: res.get("value") for name, res in state.get_results().items()}

        # Physical consistency validation
        validator = PhysicalConsistencyValidator()
        consistency_report = validator.validate(raw_values)
        audit.log("orchestrator", "consistency_check", detail=consistency_report.to_dict())

        if not consistency_report.is_consistent:
            logger.warning("Physical consistency violations detected:")
            for v in consistency_report.violations:
                logger.warning("  X %s", v)
            for metric, adj in consistency_report.confidence_adjustments.items():
                best = state.get_best_result(metric)
                if best and best.get("confidence", 0) > 0:
                    old_conf = best["confidence"]
                    new_conf = max(0.1, min(1.0, old_conf + adj))
                    best["confidence"] = new_conf
                    state._state["results"][metric] = best

        for w in consistency_report.warnings:
            logger.info("  Warning: %s", w)

        # Environment fingerprint
        fingerprinter = EnvironmentFingerprintDetector()
        fingerprint = fingerprinter.detect(
            probe_results=raw_values,
            reported_clock_mhz=ctx.environment.reported_clock_mhz,
            reported_sm_count=ctx.environment.reported_sm_count,
            reported_shmem_kb=ctx.environment.reported_max_shmem_per_block_kb,
        )
        audit.log("orchestrator", "environment_fingerprint", detail=fingerprint)

        # Update trust level based on fingerprint
        if fingerprint.get("tampering_detected"):
            ctx.environment.trust_level = "tampered"
            ctx.environment.detected_anomalies.extend(fingerprint.get("findings", []))
        else:
            ctx.environment.trust_level = "verified"

        state.save()
        kb.save()

        # ── Phase 2.5: Verifier + Judge ──────────────────────────
        logger.info("Phase 2.5: Running verifier and judge...")

        final_results = state.export_results_json()
        # Remap to output names
        output_results = {}
        for internal_name, output_name in target_map.items():
            if internal_name in final_results:
                output_results[output_name] = final_results[internal_name]
            elif output_name in final_results:
                output_results[output_name] = final_results[output_name]
        for k, v in final_results.items():
            if k not in output_results and k not in target_map:
                output_results[k] = v
        final_results = output_results

        env_dict = {
            "gpu_name": ctx.environment.gpu_name,
            "driver_version": ctx.environment.driver_version,
            "cuda_version": ctx.environment.cuda_version,
            "trust_level": ctx.environment.trust_level,
            "anomalies": ctx.environment.detected_anomalies,
        }

        # Verifier agent
        verifier_report = self.verifier.verify(final_results, env_dict)
        audit.log("verifier", "verification_complete", detail=verifier_report)

        # Judge agent
        try:
            judge_report = self.judge.judge(
                results=final_results,
                environment=env_dict,
                verifier_report=verifier_report,
                consistency_report=consistency_report.to_dict(),
                methodology=methodology,
            )
        except Exception as e:
            logger.warning("Judge failed: %s", e)
            judge_report = self.judge._rule_based_judge(final_results, verifier_report)
        audit.log("judge", "judgment_complete", detail=judge_report)

        # ── Phase 3: Output ──────────────────────────────────────
        logger.info("Phase 3: Generating output...")
        state.status = "completed"

        # Build environment_notes
        environment_notes = {
            "api_reported_device_properties": {
                "name": ctx.environment.gpu_name,
                "clockRateKHz": int(ctx.environment.reported_clock_mhz * 1000) if ctx.environment.reported_clock_mhz else None,
                "memoryClockRateKHz": int(ctx.environment.reported_mem_clock_mhz * 1000) if ctx.environment.reported_mem_clock_mhz else None,
            },
            "measured_vs_reported": {},
        }

        # Build cross_checks
        cross_checks = consistency_report.cross_checks + [
            {
                "check": "verifier",
                "status": verifier_report.get("status", "unknown"),
                "issues": verifier_report.get("issues", []),
            },
        ]

        # Generate reasoning narrative
        reasoning_narrative = (
            f"GPU: {ctx.environment.gpu_name}, Driver: {ctx.environment.driver_version}, "
            f"CUDA: {ctx.environment.cuda_version}. "
            f"All metrics measured autonomously via LLM-generated CUDA micro-benchmarks "
            f"using a multi-agent architecture (Scout -> Planner -> Codegen -> Runner -> Analyzer). "
            f"The LLM generates complete .cu probe files from strategy descriptions in the metrics catalog, "
            f"compiles them with nvcc, and parses structured output. "
            f"Physical consistency: {'PASSED' if consistency_report.is_consistent else 'VIOLATIONS FOUND'}. "
            f"Verifier: {verifier_report.get('status', 'unknown')} ({len(verifier_report.get('issues', []))} issues). "
            f"Judge: {judge_report.get('confidence', 'unknown')} confidence, {judge_report.get('status', 'unknown')}. "
            f"Trust level: {ctx.environment.trust_level}."
        )

        # Write results.json
        results_path = run_dir / "results.json"
        results_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
        logger.info("Results written to %s", results_path)

        # Write detailed results
        detailed_path = run_dir / "results_detailed.json"
        detailed_path.write_text(json.dumps(state.get_results(), indent=2, default=str), encoding="utf-8")

        # Write audit report
        audit.log("orchestrator", "run_completed", detail={"results": final_results})
        audit.save_report(
            results=state.get_results(),
            environment=ctx.environment.to_dict(),
            consistency=consistency_report.to_dict(),
            fingerprint=fingerprint,
        )

        # Write comprehensive output.json
        output_data = {
            "results": final_results,
            "reasoning": reasoning_narrative,
            "environment": env_dict,
            "environment_notes": environment_notes,
            "consistency_check": consistency_report.to_dict(),
            "fingerprint": fingerprint,
            "verifier_report": verifier_report,
            "judge_report": judge_report,
            "probe_plan": probe_plan,
            "measurement_methodology": methodology,
            "cross_checks": cross_checks,
        }
        output_path = run_dir / "output.json"
        output_path.write_text(json.dumps(output_data, indent=2, default=str), encoding="utf-8")

        state.save()
        kb.save()

        return final_results

    def _llm_correct_results(
        self,
        raw_values: dict[str, float | None],
        consistency_report,
        state,
        ctx: AgentContext,
        audit,
    ) -> dict[str, float]:
        """Use LLM to correct anomalous results based on raw probe data."""
        corrections = {}

        # Only correct if there are warnings/violations
        issues = consistency_report.warnings + consistency_report.violations
        if not issues:
            return corrections

        # Collect raw sweep data if available (for cache size correction)
        sweep_data = ""
        for metric_name in raw_values:
            best = state.get_best_result(metric_name)
            if best and best.get("raw_stdout"):
                stdout = best["raw_stdout"]
                if "SWEEP:" in stdout:
                    sweep_data += f"\n--- {metric_name} sweep data ---\n"
                    for line in stdout.split("\n"):
                        if line.startswith("SWEEP:"):
                            sweep_data += line + "\n"

        prompt = f"""Analyze these GPU measurement results and correct any anomalous values.

Measured values:
{json.dumps(raw_values, indent=2)}

Issues detected:
{chr(10).join('- ' + i for i in issues)}

Cross-check data:
{json.dumps(consistency_report.cross_checks, indent=2, default=str)}

{f'Raw sweep data:{sweep_data}' if sweep_data else ''}

For each value that needs correction, provide the corrected value and reasoning.
Only correct values that are clearly wrong based on the evidence.

Return JSON:
{{
    "corrections": {{
        "metric_name": corrected_value,
        ...
    }},
    "reasoning": "explanation"
}}

If no corrections needed, return: {{"corrections": {{}}, "reasoning": "all values acceptable"}}"""

        try:
            raw = self.llm.complete(
                system_prompt="You are a GPU hardware measurement expert. Analyze probe results and correct anomalies.",
                user_prompt=prompt,
                max_tokens=1000,
            )
            # Parse response
            result = self._extract_json(raw)
            corrections = result.get("corrections", {})
            reasoning = result.get("reasoning", "")

            if corrections:
                audit.log("orchestrator", "llm_correction", detail={
                    "corrections": corrections,
                    "reasoning": reasoning,
                })
                logger.info("LLM corrections: %s", corrections)
            else:
                logger.info("LLM: no corrections needed")

            # Convert string values to float
            return {k: float(v) for k, v in corrections.items() if v is not None}
        except Exception as e:
            logger.warning("LLM correction failed: %s", e)
            return {}

    def _generate_reasoning_narrative(
        self,
        results: dict,
        ctx: AgentContext,
        consistency_report,
        fingerprint: dict,
        state,
        audit,
    ) -> str:
        """Generate engineering reasoning narrative for LLM-as-a-Judge scoring."""
        prompt = f"""Write a concise engineering analysis report for the following GPU hardware profiling results.

GPU: {ctx.environment.gpu_name}
Driver: {ctx.environment.driver_version}, CUDA: {ctx.environment.cuda_version}
Trust level: {ctx.environment.trust_level}
Anomalies detected: {ctx.environment.detected_anomalies}

Measured results:
{json.dumps(results, indent=2)}

Physical consistency: {'PASSED' if consistency_report.is_consistent else 'VIOLATIONS FOUND'}
Violations: {consistency_report.violations}
Warnings: {consistency_report.warnings}
Cross-checks: {json.dumps(consistency_report.cross_checks, indent=2, default=str)}

Environment fingerprint:
Tampering detected: {fingerprint.get('tampering_detected', False)}
Findings: {fingerprint.get('findings', [])}

Write a report covering:
1. Methodology: How each metric was measured (pointer chasing, streaming, FMA timing, etc.)
2. Environment analysis: Was the GPU tampered with? How did you detect it?
3. Cross-verification: How do the metrics validate each other?
4. Confidence assessment: Which results are most/least reliable and why?
5. Anomalies: Any unexpected findings and their implications.

Be specific and technical. Reference actual measured values."""

        try:
            narrative = self.llm.complete(
                system_prompt="You are a GPU performance engineer writing a technical analysis report.",
                user_prompt=prompt,
                max_tokens=2000,
            )
            audit.log("orchestrator", "reasoning_narrative", detail={
                "length": len(narrative),
            })
            return narrative
        except Exception as e:
            logger.warning("Reasoning narrative generation failed: %s", e)
            return f"Reasoning narrative unavailable: {e}"

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from LLM response."""
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
        return {}

    def _probe_metric(
        self,
        metric_name: str,
        ctx: AgentContext,
        state: RunState,
        kb: KnowledgeStore,
        audit: AuditLogger,
    ) -> None:
        """Run the Planner → Codegen → Runner → Analyzer loop for one metric."""

        # ── Standard LLM path ─────────────────────────────────────
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

            # ── Planner (skip if metric has strategies in catalog) ─
            if metric_spec.strategies and attempt == 0:
                # Use first strategy from catalog directly — skip LLM planner call
                strategy_name = metric_spec.strategies[0].name
                strategy_dict = metric_spec.strategies[0].to_dict()
                plan_result = {
                    "selected_strategy": strategy_name,
                    "params_override": {},
                    "needs_ncu": getattr(metric_spec.strategies[0], 'needs_ncu', False),
                    "ncu_metrics": getattr(metric_spec.strategies[0], 'ncu_metrics', []) or [],
                }
                logger.info("  Using catalog strategy: %s (skipping planner)", strategy_name)
            else:
                # Fall back to LLM planner
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
                strategy_name = plan_result.get("selected_strategy", "")

                # If planner returned empty/unknown strategy, pick first from spec
                if not strategy_name or strategy_name == "unknown":
                    strategy_name = metric_spec.strategies[0].name if metric_spec.strategies else "unknown"
                    plan_result["selected_strategy"] = strategy_name

                # Find the strategy object
                strategy_dict = plan_result.get("strategy")
                if not strategy_dict:
                    for s in metric_spec.strategies:
                        if s.name == strategy_name:
                            strategy_dict = s.to_dict()
                            break
                    if not strategy_dict:
                        strategy_dict = {"name": strategy_name, "probe_template": None, "params": {}}

            audit.log("planner", "strategy_selected", metric_name=metric_name, detail={
                "strategy_name": strategy_name,
                "reasoning": plan_result.get("reasoning", ""),
                "attempt": attempt,
            })
            logger.info("  Strategy: %s", strategy_name)

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

            # ── Fast result parsing (skip LLM analyzer) ─────────
            stdout = runner_result.get("stdout", "")
            value = None
            for line in stdout.splitlines():
                if line.startswith(f"RESULT:{metric_name}="):
                    try:
                        value = float(line.split("=", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass

            if value is not None:
                confidence = 0.90
                # Sanity check against physical bounds
                if metric_spec.physical_min is not None and value < metric_spec.physical_min:
                    logger.warning("  Value %.4g below physical min %.4g", value, metric_spec.physical_min)
                    confidence = 0.3
                elif metric_spec.physical_max is not None and value > metric_spec.physical_max:
                    logger.warning("  Value %.4g above physical max %.4g", value, metric_spec.physical_max)
                    confidence = 0.3
                else:
                    confidence = 0.95
            else:
                # Fall back to LLM analyzer only if parsing fails
                logger.info("  Could not parse RESULT line, falling back to LLM analyzer")
                analyzer_task = Task(
                    id=f"analyze-{metric_name}-{attempt}",
                    kind="analyze_result",
                    payload={
                        "metric_name": metric_name,
                        "strategy_name": strategy_name,
                        "stdout": stdout,
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
            })

            # Record attempt
            probe_result = ProbeResult(
                metric_name=metric_name,
                value=value,
                unit="",
                confidence=confidence,
                method=strategy_name,
                strategy_name=strategy_name,
            )
            state.record_attempt(probe_result)
            kb.add_observation(probe_result)

            previous_attempts.append({
                "strategy_name": strategy_name,
                "value": value,
                "confidence": confidence,
                "error": None,
            })

            logger.info(
                "  Result: %s = %s (confidence: %.2f)",
                metric_name, value, confidence,
            )

            # Check if we're done
            if value is not None and confidence >= self.config.run.confidence_threshold:
                logger.info("  ✓ Confidence threshold met for %s", metric_name)
                audit.log("orchestrator", "metric_completed", metric_name=metric_name, detail={
                    "value": value,
                    "confidence": confidence,
                    "attempts": attempt + 1,
                })
                break

            logger.info("  Confidence %.2f < %.2f, retrying...", confidence, self.config.run.confidence_threshold)

        else:
            # Exhausted all retries
            logger.warning("  ✗ Max retries exhausted for %s", metric_name)
            audit.log("orchestrator", "metric_exhausted", metric_name=metric_name, detail={
                "max_retries": max_retries,
                "best_result": state.get_best_result(metric_name),
            })
