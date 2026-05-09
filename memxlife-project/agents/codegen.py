"""Codegen agent — generates CUDA micro-benchmarks via LLM with compile-fix loop.

Architecture (Scheme E):
  Round 1: LLM generates complete .cu file
    → nvcc compile test
    → Success? Done!
    → Fail?
  Round 2: Feed compile errors back to LLM → "fix these errors"
    → nvcc compile test
    → Success? Done!
    → Fail?
  Round 3: Discard and regenerate from scratch (different temperature)
    → nvcc compile test

All measurement logic is LLM-generated. No static CUDA templates.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from core.models import AgentContext, Task
from llm.client import LLMClient
from llm.prompts import CODEGEN_SYSTEM, CODEGEN_USER

logger = logging.getLogger(__name__)


def _find_nvcc() -> str:
    """Find nvcc binary path."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    for p in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"]:
        if os.path.isfile(p):
            return p
    return "nvcc"


def _detect_arch() -> str:
    """Auto-detect GPU arch flag."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            cap = r.stdout.strip().split("\n")[0].strip().replace(".", "")
            return f"-arch=sm_{cap}"
    except Exception:
        pass
    return "-arch=sm_80"


def _try_compile(cuda_code: str, nvcc: str, arch: str) -> tuple[bool, str]:
    """Try to compile CUDA code in a temp directory. Returns (success, error_msg)."""
    with tempfile.TemporaryDirectory() as td:
        cu_path = Path(td) / "probe.cu"
        cu_path.write_text(cuda_code, encoding="utf-8")
        try:
            r = subprocess.run(
                [nvcc, "-O2", "-w", arch, "-o", str(Path(td) / "probe"), str(cu_path)],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0:
                return True, ""
            # Extract just the error lines (skip warnings)
            errors = []
            for line in r.stderr.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip())
            error_msg = "\n".join(errors[:15]) if errors else r.stderr[-1000:]
            return False, error_msg
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out (60s)"
        except OSError as e:
            return False, f"nvcc not found: {e}"


# Prompt for compile-fix round
FIX_PROMPT = """The following CUDA code failed to compile. Fix the errors and return the COMPLETE corrected .cu file.

COMPILE ERRORS:
{errors}

ORIGINAL CODE:
```cuda
{code}
```

Return ONLY the complete corrected .cu source code. No explanations, no markdown fences, no JSON.
The code must compile with: nvcc -O2 -o probe probe.cu"""


class CodegenAgent(BaseAgent):
    name = "codegen"

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._nvcc = None
        self._arch = None

    @property
    def nvcc(self) -> str:
        if self._nvcc is None:
            self._nvcc = _find_nvcc()
        return self._nvcc

    @property
    def arch(self) -> str:
        if self._arch is None:
            self._arch = _detect_arch()
        return self._arch

    def can_handle(self, task: Task) -> bool:
        return task.kind == "generate_probe"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        metric_name = task.payload["metric_name"]
        strategy = task.payload["strategy"]
        params_override = task.payload.get("params_override", {})
        previous_errors = task.payload.get("previous_errors", [])

        strategy_name = strategy.get("name", "unknown")
        base_params = {**strategy.get("params", {}), **params_override}

        if ctx.mock_mode:
            return self._mock_result(metric_name, strategy_name)

        # ── Round 1: Generate complete code ───────────────────
        logger.info("Codegen Round 1: generating complete .cu for %s", metric_name)
        code = self._generate_full_code(
            metric_name, strategy_name,
            strategy.get("description", ""),
            base_params, previous_errors, ctx,
            temperature=0.2,
        )

        if not code:
            return self._fail(metric_name, strategy_name, "LLM returned empty code")

        # Try compile
        ok, errors = _try_compile(code, self.nvcc, self.arch)
        if ok:
            logger.info("Codegen Round 1: compile SUCCESS for %s", metric_name)
            return self._success(metric_name, strategy_name, code, "round1")

        # ── Round 2: Fix compile errors ───────────────────────
        logger.info("Codegen Round 2: fixing compile errors for %s", metric_name)
        fixed_code = self._fix_compile_errors(code, errors)

        if fixed_code:
            ok2, errors2 = _try_compile(fixed_code, self.nvcc, self.arch)
            if ok2:
                logger.info("Codegen Round 2: compile SUCCESS for %s", metric_name)
                return self._success(metric_name, strategy_name, fixed_code, "round2-fix")

            # ── Round 2.5: Try fixing again with new errors ───
            logger.info("Codegen Round 2.5: second fix attempt for %s", metric_name)
            fixed_code2 = self._fix_compile_errors(fixed_code, errors2)
            if fixed_code2:
                ok3, _ = _try_compile(fixed_code2, self.nvcc, self.arch)
                if ok3:
                    logger.info("Codegen Round 2.5: compile SUCCESS for %s", metric_name)
                    return self._success(metric_name, strategy_name, fixed_code2, "round2.5-fix")

        # ── Round 3: Full regenerate with higher temperature ──
        logger.info("Codegen Round 3: full regenerate for %s", metric_name)
        all_errors = previous_errors + [errors]
        code3 = self._generate_full_code(
            metric_name, strategy_name,
            strategy.get("description", ""),
            base_params, all_errors, ctx,
            temperature=0.4,  # slightly more creative
        )

        if code3:
            ok4, _ = _try_compile(code3, self.nvcc, self.arch)
            if ok4:
                logger.info("Codegen Round 3: compile SUCCESS for %s", metric_name)
                return self._success(metric_name, strategy_name, code3, "round3-regen")

        # All rounds failed
        logger.warning("Codegen: all rounds failed for %s", metric_name)
        # Return best code we have (even if it doesn't compile — Runner will report error)
        return self._success(metric_name, strategy_name, code, "all-rounds-failed")

    def _generate_full_code(
        self,
        metric_name: str,
        strategy_name: str,
        strategy_description: str,
        params: dict,
        previous_errors: list,
        ctx: AgentContext,
        temperature: float = 0.2,
    ) -> str:
        """Generate complete .cu file via single LLM call."""
        errors_text = ""
        if previous_errors:
            errors_text = "\n".join(
                f"Attempt {i+1}: {e}" for i, e in enumerate(previous_errors[-3:])
            )

        user_prompt = CODEGEN_USER.format(
            metric_name=metric_name,
            strategy_name=strategy_name,
            strategy_description=strategy_description,
            params_json=json.dumps(params, indent=2),
            environment_summary=ctx.environment.summary_for_prompt(),
            previous_errors=errors_text or "None — first attempt.",
        )

        try:
            raw = self.llm.complete(
                system_prompt=CODEGEN_SYSTEM,
                user_prompt=user_prompt,
                max_tokens=4096,
                temperature=temperature,
            )
            return self._extract_code(raw)
        except Exception as e:
            logger.warning("Codegen LLM call failed: %s", e)
            return ""

    def _fix_compile_errors(self, code: str, errors: str) -> str:
        """Ask LLM to fix compile errors. Short, focused call."""
        prompt = FIX_PROMPT.format(errors=errors, code=code)

        try:
            raw = self.llm.complete(
                system_prompt="You are a CUDA compiler error fixer. Return ONLY the corrected complete .cu source code. No markdown, no explanations.",
                user_prompt=prompt,
                max_tokens=4096,
                temperature=0.1,  # conservative for fixes
            )
            fixed = self._extract_code(raw)
            # Sanity check: fixed code should still have main() and RESULT:
            if fixed and "main(" in fixed and len(fixed) > 100:
                return fixed
            return ""
        except Exception as e:
            logger.warning("Fix LLM call failed: %s", e)
            return ""

    def _extract_code(self, text: str) -> str:
        """Extract CUDA code from LLM response (handles JSON wrapping, markdown, raw code)."""
        text = text.strip()
        if not text:
            return ""

        # Try JSON extraction first (LLM might return JSON with cuda_code field)
        if text.startswith("{"):
            try:
                d = json.loads(text)
                code = d.get("cuda_code", "")
                if code and "main(" in code:
                    return code
            except json.JSONDecodeError:
                # Try fixing common JSON issues: the LLM sometimes returns
                # JSON with the cuda_code value containing unescaped newlines
                import re
                m = re.search(r'"cuda_code"\s*:\s*"(.*)"', text, re.DOTALL)
                if m:
                    raw_code = m.group(1)
                    # Unescape common escape sequences
                    code = raw_code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")
                    if code and "main(" in code:
                        return code

        # Try extracting from ```cuda or ```cpp blocks
        for marker in ("```cuda", "```cpp", "```c", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                # Skip to next line after marker
                nl = text.find("\n", start)
                if nl != -1:
                    start = nl + 1
                end = text.find("```", start)
                if end != -1:
                    code = text[start:end].strip()
                    if code and ("main(" in code or "__global__" in code):
                        return code

        # Try JSON extraction from middle of text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                d = json.loads(text[brace_start:brace_end + 1])
                code = d.get("cuda_code", "")
                if code and "main(" in code:
                    return code
            except json.JSONDecodeError:
                # Try regex extraction from the JSON-like block
                import re
                chunk = text[brace_start:brace_end + 1]
                m = re.search(r'"cuda_code"\s*:\s*"(.*)"', chunk, re.DOTALL)
                if m:
                    raw_code = m.group(1)
                    code = raw_code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")
                    if code and "main(" in code:
                        return code

        # Raw code — if it looks like C/CUDA, use it directly
        if "#include" in text or "__global__" in text or "int main(" in text:
            # Strip any leading/trailing non-code text
            lines = text.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if not in_code and (line.strip().startswith("#include") or
                                     line.strip().startswith("__global__") or
                                     line.strip().startswith("//") or
                                     line.strip() == ""):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            return "\n".join(code_lines).strip()

        return ""

    def _success(self, metric_name: str, strategy_name: str, code: str, round_name: str) -> dict:
        return {
            "metric_name": metric_name,
            "strategy_name": strategy_name,
            "cuda_code": code,
            "compile_command": "nvcc -O2 -o probe probe.cu",
            "run_command": "./probe",
            "expected_output_format": f"RESULT:{metric_name}=<value>",
            "codegen": f"llm-{round_name}",
        }

    def _fail(self, metric_name: str, strategy_name: str, error: str) -> dict:
        return {
            "metric_name": metric_name,
            "strategy_name": strategy_name,
            "cuda_code": "",
            "compile_command": "",
            "run_command": "",
            "error": error,
            "codegen": "failed",
        }

    def _mock_result(self, metric_name: str, strategy_name: str) -> dict:
        return {
            "metric_name": metric_name,
            "strategy_name": strategy_name,
            "cuda_code": f"// mock probe for {metric_name}\nint main() {{ return 0; }}",
            "compile_command": "nvcc -O2 -o probe probe.cu",
            "run_command": "./probe",
            "expected_output_format": f"RESULT:{metric_name}=<value>",
            "codegen": "mock",
        }
