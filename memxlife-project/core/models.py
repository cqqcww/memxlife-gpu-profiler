"""Core data models for the multi-agent GPU profiling system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    """Unit of work dispatched to an agent."""
    id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    status: TaskStatus = TaskStatus.PENDING
    error: str | None = None
    attempts: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at,
        }


@dataclass
class EnvironmentProfile:
    """GPU environment information collected by the Scout agent."""
    gpu_name: str = "unknown"
    driver_version: str = "unknown"
    cuda_version: str = "unknown"
    tools: dict[str, bool] = field(default_factory=dict)
    # Detected (possibly tampered) values
    reported_clock_mhz: float | None = None
    reported_mem_clock_mhz: float | None = None
    reported_sm_count: int | None = None
    reported_max_shmem_per_block_kb: int | None = None
    # Trust level: "untrusted" (from API), "measured" (from probes)
    trust_level: str = "untrusted"
    raw_nvidia_smi: str = ""
    raw_device_query: str = ""
    detected_anomalies: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "tools": self.tools,
            "reported_clock_mhz": self.reported_clock_mhz,
            "reported_mem_clock_mhz": self.reported_mem_clock_mhz,
            "reported_sm_count": self.reported_sm_count,
            "reported_max_shmem_per_block_kb": self.reported_max_shmem_per_block_kb,
            "trust_level": self.trust_level,
            "detected_anomalies": self.detected_anomalies,
            "timestamp": self.timestamp,
        }

    def summary_for_prompt(self) -> str:
        """Compact summary suitable for injection into LLM prompts."""
        lines = [
            f"GPU: {self.gpu_name}",
            f"Driver: {self.driver_version}, CUDA: {self.cuda_version}",
            f"Tools: {', '.join(k for k, v in self.tools.items() if v) or 'none detected'}",
            f"Reported clock: {self.reported_clock_mhz} MHz, mem clock: {self.reported_mem_clock_mhz} MHz",
            f"Reported SM count: {self.reported_sm_count}",
            f"Trust level: {self.trust_level}",
        ]
        if self.detected_anomalies:
            lines.append(f"ANOMALIES: {'; '.join(self.detected_anomalies)}")
        return "\n".join(lines)


@dataclass
class ProbeResult:
    """Result of a single probe attempt for one metric."""
    metric_name: str
    value: float | None = None
    unit: str = ""
    confidence: float = 0.0  # 0.0 - 1.0
    method: str = ""
    strategy_name: str = ""
    raw_stdout: str = ""
    raw_stderr: str = ""
    ncu_output: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "method": self.method,
            "strategy_name": self.strategy_name,
            "error": self.error,
            "evidence_refs": self.evidence_refs,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentContext:
    """Shared context passed to every agent invocation."""
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run_dir: Path = field(default_factory=lambda: Path("runs"))
    environment: EnvironmentProfile = field(default_factory=EnvironmentProfile)
    mock_mode: bool = False

    def iter_dir(self, iteration: int) -> Path:
        d = self.run_dir / "iterations" / f"iter_{iteration:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def metric_dir(self, metric_name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in metric_name)
        d = self.run_dir / "metrics" / safe
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass
class RetryPolicy:
    max_retries: int = 2
    retry_delay_sec: float = 1.0


@dataclass
class ProbeStrategy:
    """A single strategy for probing a hardware metric."""
    name: str
    probe_template: str | None = None  # template filename (without .cu)
    params: dict[str, Any] = field(default_factory=dict)
    ncu_metrics: list[str] = field(default_factory=list)
    needs_ncu: bool = False
    priority: int = 1  # lower = try first
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "probe_template": self.probe_template,
            "params": self.params,
            "ncu_metrics": self.ncu_metrics,
            "needs_ncu": self.needs_ncu,
            "priority": self.priority,
            "description": self.description,
        }


@dataclass
class MetricSpec:
    """Specification for a target hardware metric."""
    name: str
    description: str = ""
    unit: str = ""
    strategies: list[ProbeStrategy] = field(default_factory=list)
    cross_verify: bool = False
    tolerance_pct: float = 5.0
    physical_min: float | None = None
    physical_max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "strategies": [s.to_dict() for s in self.strategies],
            "cross_verify": self.cross_verify,
            "tolerance_pct": self.tolerance_pct,
            "physical_min": self.physical_min,
            "physical_max": self.physical_max,
        }
