"""Runner agent — compiles and executes CUDA probes with adaptive retry."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from core.models import AgentContext, Task

logger = logging.getLogger(__name__)

# Compile flag variants for adaptive retry
COMPILE_FLAG_VARIANTS = [
    [],                          # default
    ["-arch=sm_80"],             # Ampere
    ["-arch=sm_86"],             # Ampere (GA106)
    ["-arch=sm_89"],             # Ada Lovelace
    ["-arch=sm_90"],             # Hopper
    ["-arch=sm_70"],             # Volta
    ["-arch=sm_75"],             # Turing
    ["--std=c++17"],             # C++17 mode
    ["-Xcompiler", "-w"],       # suppress warnings
]


class RunnerAgent(BaseAgent):
    name = "runner"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "run_probe"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        if ctx.mock_mode:
            return self._mock_run(task, ctx)

        metric_name = task.payload["metric_name"]
        cuda_code = task.payload["cuda_code"]
        compile_command = task.payload["compile_command"]
        run_command = task.payload["run_command"]
        strategy_name = task.payload.get("strategy_name", "unknown")
        needs_ncu = task.payload.get("needs_ncu", False)
        ncu_metrics = task.payload.get("ncu_metrics", [])

        # Set up working directory
        work_dir = ctx.metric_dir(metric_name) / f"attempt_{task.attempts:02d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write source file
        cu_file = work_dir / "probe.cu"
        cu_file.write_text(cuda_code, encoding="utf-8")

        # Compile with adaptive retry
        compile_result = self._compile_with_retry(
            compile_command, work_dir, ctx.mock_mode,
            max_retries=3,
            timeout=60,
        )

        if not compile_result["success"]:
            return {
                "metric_name": metric_name,
                "strategy_name": strategy_name,
                "phase": "compile",
                "success": False,
                "error": compile_result["error"],
                "stderr": compile_result["stderr"],
                "work_dir": str(work_dir),
            }

        # Execute probe
        exec_result = self._execute(
            run_command, work_dir, timeout=ctx.run_dir and 120 or 120
        )

        # Optional ncu profiling
        ncu_output = ""
        if needs_ncu and ncu_metrics and ctx.environment.tools.get("ncu"):
            ncu_output = self._run_ncu(
                run_command, ncu_metrics, work_dir, timeout=180
            )

        # Save all artifacts
        (work_dir / "compile_log.txt").write_text(
            compile_result.get("stdout", "") + "\n" + compile_result.get("stderr", ""),
            encoding="utf-8",
        )
        (work_dir / "exec_stdout.txt").write_text(exec_result.get("stdout", ""), encoding="utf-8")
        (work_dir / "exec_stderr.txt").write_text(exec_result.get("stderr", ""), encoding="utf-8")
        if ncu_output:
            (work_dir / "ncu_output.txt").write_text(ncu_output, encoding="utf-8")

        return {
            "metric_name": metric_name,
            "strategy_name": strategy_name,
            "phase": "execute",
            "success": exec_result["success"],
            "returncode": exec_result["returncode"],
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
            "ncu_output": ncu_output,
            "elapsed_sec": exec_result.get("elapsed_sec", 0),
            "work_dir": str(work_dir),
            "compile_flags_used": compile_result.get("flags_used", []),
            "error": exec_result.get("error"),
        }

    def _compile_with_retry(
        self, base_command: str, work_dir: Path, mock: bool,
        max_retries: int = 3, timeout: int = 60,
    ) -> dict[str, Any]:
        """Compile with adaptive flag variants on failure."""
        last_error = ""
        last_stderr = ""

        for attempt, extra_flags in enumerate(COMPILE_FLAG_VARIANTS[:max_retries + 1]):
            cmd = base_command
            if extra_flags:
                # Insert flags after 'nvcc'
                parts = cmd.split(None, 1)
                if len(parts) == 2:
                    cmd = parts[0] + " " + " ".join(extra_flags) + " " + parts[1]

            try:
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode == 0:
                    return {
                        "success": True,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "flags_used": extra_flags,
                        "attempt": attempt,
                    }
                last_error = f"nvcc returned {result.returncode}"
                last_stderr = result.stderr
                logger.info(
                    "Compile attempt %d failed (flags=%s): %s",
                    attempt, extra_flags, result.stderr[:200],
                )
            except subprocess.TimeoutExpired:
                last_error = f"Compile timed out after {timeout}s"
                last_stderr = ""
            except OSError as e:
                last_error = str(e)
                last_stderr = ""

        return {
            "success": False,
            "error": last_error,
            "stderr": last_stderr,
            "stdout": "",
            "flags_used": [],
        }

    def _execute(
        self, run_command: str, work_dir: Path, timeout: int = 120
    ) -> dict[str, Any]:
        """Execute the compiled probe binary."""
        start = time.time()
        try:
            result = subprocess.run(
                ["bash", "-c", run_command],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.time() - start
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_sec": elapsed,
                "error": None if result.returncode == 0 else f"Exit code {result.returncode}",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "elapsed_sec": time.time() - start,
                "error": f"Timeout after {timeout}s",
            }
        except OSError as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "elapsed_sec": time.time() - start,
                "error": str(e),
            }

    def _run_ncu(
        self, run_command: str, metrics: list[str], work_dir: Path, timeout: int = 180
    ) -> str:
        """Run ncu profiling on the probe binary."""
        # Extract binary name from run command
        binary = run_command.strip().split()[0]
        metrics_str = ",".join(metrics)
        ncu_cmd = f"ncu --metrics {metrics_str} --csv {binary}"

        # Try with sudo if needed
        try:
            result = subprocess.run(
                ["bash", "-c", ncu_cmd],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout
            # Check for permission error
            if "ERR_NVGPUCTRPERM" in result.stderr:
                logger.info("ncu permission error, retrying with sudo")
                sudo_cmd = f"sudo -n {ncu_cmd}"
                result2 = subprocess.run(
                    ["bash", "-c", sudo_cmd],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result2.returncode == 0:
                    return result2.stdout
                return f"ncu failed even with sudo:\n{result2.stderr}"
            return f"ncu failed:\n{result.stderr}"
        except (subprocess.TimeoutExpired, OSError) as e:
            return f"ncu error: {e}"

    def _mock_run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        """Return simulated results for local development."""
        metric_name = task.payload["metric_name"]
        strategy_name = task.payload.get("strategy_name", "mock")

        mock_values = {
            "dram_latency_cycles": 442,
            "l1_latency_cycles": 28,
            "l2_latency_cycles": 193,
            "l2_cache_size_kb": 49152,
            "max_global_mem_bandwidth_gb_s": 1008.4,
            "max_shmem_bandwidth_gb_s": 19200.0,
            "actual_boost_clock_mhz": 2520,
            "bank_conflict_penalty_cycles": 23,
            "max_shmem_per_block_kb": 100,
        }
        value = mock_values.get(metric_name, 42.0)

        work_dir = ctx.metric_dir(metric_name) / f"attempt_{task.attempts:02d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        stdout = (
            f"RESULT:{metric_name}={value}\n"
            f"UNIT:mock\n"
            f"METHOD:mock simulation\n"
            f"ITERATIONS:100\n"
            f"WARMUP:10\n"
        )
        (work_dir / "exec_stdout.txt").write_text(stdout, encoding="utf-8")

        return {
            "metric_name": metric_name,
            "strategy_name": strategy_name,
            "phase": "execute",
            "success": True,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "ncu_output": "",
            "elapsed_sec": 0.1,
            "work_dir": str(work_dir),
            "compile_flags_used": [],
            "error": None,
        }
