"""Environment Scout agent — detects GPU environment and available tools."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from agents.base import BaseAgent
from core.models import AgentContext, EnvironmentProfile, Task


class ScoutAgent(BaseAgent):
    name = "scout"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "scout_environment"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        env = EnvironmentProfile()
        env.tools = self._detect_tools()

        # nvidia-smi query
        smi_output = self._run_cmd(["nvidia-smi"])
        env.raw_nvidia_smi = smi_output
        self._parse_nvidia_smi(smi_output, env)

        # nvidia-smi detailed query
        detail = self._run_cmd([
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.current.graphics,clocks.current.memory,count",
            "--format=csv,noheader,nounits",
        ])
        self._parse_nvidia_smi_detail(detail, env)

        # CUDA version from nvcc
        nvcc_out = self._run_cmd(["nvcc", "--version"])
        self._parse_nvcc_version(nvcc_out, env)

        # Device query if available
        if shutil.which("deviceQuery"):
            dq = self._run_cmd(["deviceQuery"])
            env.raw_device_query = dq

        # Anomaly detection
        self._detect_anomalies(env)

        # Save artifacts
        artifact_dir = ctx.run_dir / "scout"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "nvidia_smi.txt").write_text(smi_output, encoding="utf-8")
        if env.raw_device_query:
            (artifact_dir / "device_query.txt").write_text(env.raw_device_query, encoding="utf-8")

        ctx.environment = env

        return {
            "environment": env.to_dict(),
            "summary": env.summary_for_prompt(),
            "artifact_dir": str(artifact_dir),
        }

    def _detect_tools(self) -> dict[str, bool]:
        tools = ["nvidia-smi", "ncu", "nsys", "nvcc", "deviceQuery", "bandwidthTest"]
        return {t: shutil.which(t) is not None for t in tools}

    def _run_cmd(self, cmd: list[str], timeout: int = 30) -> str:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.stdout + result.stderr
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return ""

    def _parse_nvidia_smi(self, output: str, env: EnvironmentProfile) -> None:
        for line in output.splitlines():
            if "Driver Version:" in line:
                parts = line.split("Driver Version:")
                if len(parts) > 1:
                    env.driver_version = parts[1].strip().split()[0]
            if "CUDA Version:" in line:
                parts = line.split("CUDA Version:")
                if len(parts) > 1:
                    env.cuda_version = parts[1].strip().split()[0]

    def _parse_nvidia_smi_detail(self, output: str, env: EnvironmentProfile) -> None:
        line = output.strip().split("\n")[0] if output.strip() else ""
        if not line:
            return
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 1:
            env.gpu_name = parts[0]
        if len(parts) >= 2:
            env.driver_version = env.driver_version or parts[1]
        if len(parts) >= 3:
            try:
                env.reported_clock_mhz = float(parts[2])
            except ValueError:
                pass
        if len(parts) >= 4:
            try:
                env.reported_mem_clock_mhz = float(parts[3])
            except ValueError:
                pass

    def _parse_nvcc_version(self, output: str, env: EnvironmentProfile) -> None:
        for line in output.splitlines():
            if "release" in line.lower():
                # e.g., "Cuda compilation tools, release 12.0, V12.0.140"
                parts = line.split("release")
                if len(parts) > 1:
                    ver = parts[1].strip().split(",")[0].strip()
                    env.cuda_version = env.cuda_version or ver

    def _detect_anomalies(self, env: EnvironmentProfile) -> None:
        anomalies = []
        # Check for suspiciously low clock
        if env.reported_clock_mhz and env.reported_clock_mhz < 500:
            anomalies.append(
                f"GPU clock ({env.reported_clock_mhz} MHz) is unusually low — "
                "possible frequency locking"
            )
        # Check for non-standard clock values (not typical boost clocks)
        if env.reported_clock_mhz and env.reported_clock_mhz % 15 != 0:
            anomalies.append(
                f"GPU clock ({env.reported_clock_mhz} MHz) is not a standard value — "
                "possible frequency locking"
            )
        env.detected_anomalies = anomalies
        if anomalies:
            env.trust_level = "suspicious"
