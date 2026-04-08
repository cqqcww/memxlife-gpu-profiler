"""Codegen agent — generates CUDA micro-benchmark code from templates + LLM."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import BaseAgent
from core.models import AgentContext, Task
from llm.client import LLMClient
from llm.prompts import CODEGEN_SYSTEM, CODEGEN_USER
from probes.registry import load_template

logger = logging.getLogger(__name__)


def _build_compile_command(params: dict[str, Any]) -> str:
    """Build nvcc compile command with -D flags from params."""
    defines = []
    for key, value in params.items():
        if key.isupper() and isinstance(value, (int, float)):
            defines.append(f"-D{key}={value}")
        elif key.isupper() and isinstance(value, str):
            defines.append(f'-D{key}=\\"{value}\\"')
    defines_str = " ".join(defines)
    return f"nvcc -O2 {defines_str} -o probe probe.cu"


class CodegenAgent(BaseAgent):
    name = "codegen"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def can_handle(self, task: Task) -> bool:
        return task.kind == "generate_probe"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        metric_name = task.payload["metric_name"]
        strategy = task.payload["strategy"]
        params_override = task.payload.get("params_override", {})
        previous_errors = task.payload.get("previous_errors", [])

        strategy_name = strategy.get("name", "unknown")
        template_name = strategy.get("probe_template")
        base_params = {**strategy.get("params", {}), **params_override}

        # Mock mode — return stub code so Runner can produce mock results
        if ctx.mock_mode:
            return {
                "metric_name": metric_name,
                "strategy_name": strategy_name,
                "cuda_code": f"// mock probe for {metric_name}\nint main() {{ return 0; }}",
                "compile_command": "nvcc -O2 -o probe probe.cu",
                "run_command": "./probe",
                "expected_output_format": f"RESULT:{metric_name}=<value>",
                "codegen": "mock",
            }

        # Load template if available
        template_code = ""
        if template_name:
            try:
                template_code = load_template(template_name)
            except FileNotFoundError:
                logger.info("Template %s not found, will generate from scratch", template_name)

        # Build compile command with -D flags for params
        compile_cmd = _build_compile_command(base_params)

        # If we have a template, try LLM to refine it; fall back to template directly
        if template_code:
            try:
                return self._generate_with_llm(
                    metric_name=metric_name,
                    strategy_name=strategy_name,
                    strategy_description=strategy.get("description", ""),
                    template_code=template_code,
                    params=base_params,
                    compile_cmd=compile_cmd,
                    previous_errors=previous_errors,
                    ctx=ctx,
                )
            except Exception as e:
                logger.warning("LLM codegen failed, using template directly: %s", e)
                return {
                    "metric_name": metric_name,
                    "strategy_name": strategy_name,
                    "cuda_code": template_code,
                    "compile_command": compile_cmd,
                    "run_command": "./probe",
                    "expected_output_format": f"RESULT:{metric_name}=<value>",
                    "codegen": "template-fallback",
                }

        # No template — must use LLM
        return self._generate_with_llm(
            metric_name=metric_name,
            strategy_name=strategy_name,
            strategy_description=strategy.get("description", ""),
            template_code="",
            params=base_params,
            compile_cmd=compile_cmd,
            previous_errors=previous_errors,
            ctx=ctx,
        )

    def _generate_with_llm(
        self,
        metric_name: str,
        strategy_name: str,
        strategy_description: str,
        template_code: str,
        params: dict[str, Any],
        compile_cmd: str,
        previous_errors: list,
        ctx: AgentContext,
    ) -> dict[str, Any]:
        errors_text = ""
        if previous_errors:
            errors_text = "\n".join(
                f"Attempt {i+1}: {e}" for i, e in enumerate(previous_errors[-3:])
            )

        user_prompt = CODEGEN_USER.format(
            metric_name=metric_name,
            strategy_name=strategy_name,
            strategy_description=strategy_description,
            template_code=template_code or "No template available — generate from scratch.",
            params_json=json.dumps(params, indent=2),
            environment_summary=ctx.environment.summary_for_prompt(),
            previous_errors=errors_text or "None — first attempt.",
        )

        try:
            raw = self.llm.complete(
                system_prompt=CODEGEN_SYSTEM,
                user_prompt=user_prompt,
                model=self.llm.config.codegen_model,
            )
            decision = _extract_json(raw)
        except Exception as e:
            logger.error("Codegen LLM call failed: %s", e)
            # If we have a template, use it directly
            if template_code:
                return {
                    "metric_name": metric_name,
                    "strategy_name": strategy_name,
                    "cuda_code": template_code,
                    "compile_command": compile_cmd,
                    "run_command": "./probe",
                    "expected_output_format": f"RESULT:{metric_name}=<value>",
                    "codegen": "template-fallback",
                }
            return {
                "metric_name": metric_name,
                "strategy_name": strategy_name,
                "cuda_code": "",
                "compile_command": "",
                "run_command": "",
                "error": f"Codegen failed: {e}",
                "codegen": "failed",
            }

        cuda_code = decision.get("cuda_code", "")
        llm_compile = decision.get("compile_command", "")
        run_cmd = decision.get("run_command", "./probe")

        # Prefer LLM compile command if it looks valid, otherwise use our generated one
        final_compile = llm_compile if llm_compile.startswith("nvcc") else compile_cmd

        return {
            "metric_name": metric_name,
            "strategy_name": strategy_name,
            "cuda_code": cuda_code if cuda_code else template_code,
            "compile_command": final_compile,
            "run_command": run_cmd,
            "expected_output_format": decision.get("expected_output_format", ""),
            "codegen": "llm",
        }


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling code blocks."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for marker in ("```json", "```"):
        if marker in text:
            idx = text.index(marker) + len(marker)
            if marker == "```json":
                nl = text.find("\n", idx)
                if nl != -1:
                    idx = nl + 1
            end = text.find("```", idx)
            if end != -1:
                try:
                    return json.loads(text[idx:end].strip())
                except json.JSONDecodeError:
                    pass
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    return {}
