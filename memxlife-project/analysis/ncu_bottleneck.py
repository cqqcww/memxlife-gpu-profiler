"""NCU Bottleneck Analyzer — reads ncu metrics to diagnose GPU kernel performance.

Implements the Phase 1 core functionality:
  1. Roofline Model → Compute-Bound vs Memory-Bound classification
  2. Memory hierarchy analysis (L1/L2/DRAM bottleneck identification)
  3. Compute unit analysis (Tensor Core / FMA utilization)
  4. Occupancy analysis
  5. Common bottleneck detection (bank conflict, warp divergence, uncoalesced access)
  6. Structured diagnostic report generation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from parser.ncu_parser import parse_ncu_csv, extract_ncu_metric

logger = logging.getLogger(__name__)


class BottleneckType(Enum):
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    LATENCY_BOUND = "latency_bound"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


class MemoryLevel(Enum):
    L1 = "l1"
    L2 = "l2"
    DRAM = "dram"
    NONE = "none"


class ComputeUnit(Enum):
    TENSOR_CORE = "tensor_core"
    FMA_FP32 = "fma_fp32"
    FMA_FP64 = "fma_fp64"
    INT = "int"
    NONE = "none"


@dataclass
class BottleneckDiagnosis:
    """Full diagnosis of a CUDA kernel's performance bottleneck."""
    # Primary classification
    primary_bottleneck: BottleneckType = BottleneckType.UNKNOWN
    compute_pct: float = 0.0  # sm__throughput %
    memory_pct: float = 0.0   # gpu__compute_memory_throughput %
    sol_compute: float = 0.0
    sol_memory: float = 0.0

    # Memory hierarchy
    memory_bottleneck_level: MemoryLevel = MemoryLevel.NONE
    l1_throughput_pct: float = 0.0
    l2_throughput_pct: float = 0.0
    dram_throughput_pct: float = 0.0

    # Compute units
    primary_compute_unit: ComputeUnit = ComputeUnit.NONE
    tensor_core_pct: float = 0.0
    fma_fp32_pct: float = 0.0
    fp32_instructions: float = 0.0

    # Occupancy
    theoretical_occupancy_pct: float = 0.0
    achieved_occupancy_pct: float = 0.0
    occupancy_gap: float = 0.0

    # Specific bottlenecks detected
    bottlenecks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    optimization_suggestions: list[str] = field(default_factory=list)

    # Raw metrics used
    metrics_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_bottleneck": self.primary_bottleneck.value,
            "compute_pct": self.compute_pct,
            "memory_pct": self.memory_pct,
            "sol_compute": self.sol_compute,
            "sol_memory": self.sol_memory,
            "memory_bottleneck_level": self.memory_bottleneck_level.value,
            "l1_throughput_pct": self.l1_throughput_pct,
            "l2_throughput_pct": self.l2_throughput_pct,
            "dram_throughput_pct": self.dram_throughput_pct,
            "primary_compute_unit": self.primary_compute_unit.value,
            "tensor_core_pct": self.tensor_core_pct,
            "fma_fp32_pct": self.fma_fp32_pct,
            "theoretical_occupancy_pct": self.theoretical_occupancy_pct,
            "achieved_occupancy_pct": self.achieved_occupancy_pct,
            "occupancy_gap": self.occupancy_gap,
            "bottlenecks": self.bottlenecks,
            "warnings": self.warnings,
            "optimization_suggestions": self.optimization_suggestions,
            "metrics_used": self.metrics_used,
        }


# ── Metric name aliases for robust extraction ─────────────────────

METRIC_ALIASES: dict[str, list[str]] = {
    # Compute utilization
    "compute_throughput": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_active",
        "smsp__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    # Memory throughput
    "memory_throughput": [
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_active",
    ],
    # L1
    "l1_sectors": [
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors.sum",
    ],
    # L2
    "l2_throughput": [
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    ],
    # DRAM
    "dram_throughput": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    # Tensor Cores
    "tensor_core_active": [
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    ],
    # FP32/FMA
    "fma_active": [
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    ],
    "fp32_instructions": [
        "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    ],
    # Occupancy
    "theoretical_occupancy": [
        "sm__maximum_warps_per_active_cycle_pct",
        "sm__maximum_warps_avg_per_active_cycle",
    ],
    "achieved_occupancy": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.per_cycle_active",
    ],
    # Bank conflicts
    "bank_conflicts": [
        "l1tex__data_bank_conflicts_pipe_lsu.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    ],
    # Warp divergence
    "warp_execution_efficiency": [
        "smsp__thread_inst_executed_per_inst_executed.ratio",
        "sm__sass_thread_inst_executed_per_inst_executed.ratio",
    ],
    # SOL (Speed of Light)
    "sol_compute": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "sol_memory": [
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
}


def _extract_with_aliases(
    parsed: list[dict[str, Any]], alias_key: str
) -> float | None:
    """Try to extract a metric value using multiple alias names."""
    aliases = METRIC_ALIASES.get(alias_key, [])
    for name in aliases:
        val = extract_ncu_metric(parsed, name)
        if val is not None:
            return val
    return None


class NcuBottleneckAnalyzer:
    """Analyzes ncu output to diagnose CUDA kernel bottlenecks."""

    def analyze(
        self, ncu_output: str, kernel_name: str = ""
    ) -> BottleneckDiagnosis:
        """Analyze ncu output and produce a full bottleneck diagnosis.

        Args:
            ncu_output: Raw ncu stdout (CSV or text format)
            kernel_name: Optional kernel name for logging

        Returns:
            BottleneckDiagnosis with all analysis results
        """
        parsed = parse_ncu_csv(ncu_output)
        diag = BottleneckDiagnosis()

        if not parsed:
            diag.warnings.append("No metrics parsed from ncu output")
            return diag

        # Step 1: Roofline — Compute vs Memory
        self._analyze_roofline(parsed, diag)

        # Step 2: Memory hierarchy analysis
        self._analyze_memory_hierarchy(parsed, diag)

        # Step 3: Compute unit analysis
        self._analyze_compute_units(parsed, diag)

        # Step 4: Occupancy analysis
        self._analyze_occupancy(parsed, diag)

        # Step 5: Specific bottleneck detection
        self._detect_specific_bottlenecks(parsed, diag)

        # Step 6: Generate optimization suggestions
        self._generate_suggestions(diag)

        logger.info(
            "NCU diagnosis for %s: %s (compute=%.1f%%, memory=%.1f%%)",
            kernel_name or "kernel",
            diag.primary_bottleneck.value,
            diag.compute_pct,
            diag.memory_pct,
        )
        return diag

    def _analyze_roofline(
        self, parsed: list[dict], diag: BottleneckDiagnosis
    ) -> None:
        """Step 1: Determine Compute-Bound vs Memory-Bound."""
        compute = _extract_with_aliases(parsed, "compute_throughput")
        memory = _extract_with_aliases(parsed, "memory_throughput")

        if compute is not None:
            diag.compute_pct = compute
            diag.sol_compute = compute
            diag.metrics_used["compute_throughput"] = compute

        if memory is not None:
            diag.memory_pct = memory
            diag.sol_memory = memory
            diag.metrics_used["memory_throughput"] = memory

        # Classification logic
        if compute is None and memory is None:
            diag.primary_bottleneck = BottleneckType.UNKNOWN
            diag.warnings.append(
                "Neither compute nor memory throughput metrics available"
            )
        elif compute is None:
            diag.primary_bottleneck = BottleneckType.MEMORY_BOUND
        elif memory is None:
            diag.primary_bottleneck = BottleneckType.COMPUTE_BOUND
        elif max(compute, memory) < 20:
            # Both are low — latency bound
            diag.primary_bottleneck = BottleneckType.LATENCY_BOUND
            diag.bottlenecks.append(
                f"Low utilization (compute={compute:.1f}%, memory={memory:.1f}%) "
                "suggests latency-bound kernel"
            )
        elif compute > memory * 1.2:
            diag.primary_bottleneck = BottleneckType.COMPUTE_BOUND
        elif memory > compute * 1.2:
            diag.primary_bottleneck = BottleneckType.MEMORY_BOUND
        else:
            diag.primary_bottleneck = BottleneckType.BALANCED

    def _analyze_memory_hierarchy(
        self, parsed: list[dict], diag: BottleneckDiagnosis
    ) -> None:
        """Step 2: If memory-bound, identify which level is the bottleneck."""
        l2 = _extract_with_aliases(parsed, "l2_throughput")
        dram = _extract_with_aliases(parsed, "dram_throughput")

        if l2 is not None:
            diag.l2_throughput_pct = l2
            diag.metrics_used["l2_throughput"] = l2
        if dram is not None:
            diag.dram_throughput_pct = dram
            diag.metrics_used["dram_throughput"] = dram

        # L1 sectors (not a % — high value indicates L1 pressure)
        l1 = _extract_with_aliases(parsed, "l1_sectors")
        if l1 is not None:
            diag.metrics_used["l1_sectors"] = l1

        # Determine memory bottleneck level
        if diag.primary_bottleneck in (
            BottleneckType.MEMORY_BOUND,
            BottleneckType.BALANCED,
        ):
            if dram is not None and dram > 70:
                diag.memory_bottleneck_level = MemoryLevel.DRAM
                diag.bottlenecks.append(
                    f"DRAM throughput at {dram:.1f}% — VRAM bandwidth saturated"
                )
            elif l2 is not None and l2 > 70:
                diag.memory_bottleneck_level = MemoryLevel.L2
                diag.bottlenecks.append(
                    f"L2 throughput at {l2:.1f}% — massive data exchange between L2 and SM"
                )
            elif l1 is not None and l1 > 1e6:
                diag.memory_bottleneck_level = MemoryLevel.L1
                diag.bottlenecks.append(
                    f"High L1 sector count ({l1:.0f}) — frequent global memory misses"
                )

    def _analyze_compute_units(
        self, parsed: list[dict], diag: BottleneckDiagnosis
    ) -> None:
        """Step 3: If compute-bound, identify which compute units are active."""
        tensor = _extract_with_aliases(parsed, "tensor_core_active")
        fma = _extract_with_aliases(parsed, "fma_active")
        fp32 = _extract_with_aliases(parsed, "fp32_instructions")

        if tensor is not None:
            diag.tensor_core_pct = tensor
            diag.metrics_used["tensor_core_active"] = tensor
        if fma is not None:
            diag.fma_fp32_pct = fma
            diag.metrics_used["fma_active"] = fma
        if fp32 is not None:
            diag.fp32_instructions = fp32
            diag.metrics_used["fp32_instructions"] = fp32

        if diag.primary_bottleneck == BottleneckType.COMPUTE_BOUND:
            if tensor is not None and tensor > 50:
                diag.primary_compute_unit = ComputeUnit.TENSOR_CORE
            elif fma is not None and fma > 30:
                diag.primary_compute_unit = ComputeUnit.FMA_FP32
            elif tensor is not None and tensor < 10 and diag.compute_pct > 50:
                diag.bottlenecks.append(
                    f"Compute-bound but Tensor Core utilization is low ({tensor:.1f}%) "
                    "— kernel may not be using Tensor Cores for matrix ops"
                )
                diag.primary_compute_unit = ComputeUnit.FMA_FP32

    def _analyze_occupancy(
        self, parsed: list[dict], diag: BottleneckDiagnosis
    ) -> None:
        """Step 4: Check occupancy — are we filling the GPU?"""
        theoretical = _extract_with_aliases(parsed, "theoretical_occupancy")
        achieved = _extract_with_aliases(parsed, "achieved_occupancy")

        if theoretical is not None:
            diag.theoretical_occupancy_pct = theoretical
            diag.metrics_used["theoretical_occupancy"] = theoretical
        if achieved is not None:
            diag.achieved_occupancy_pct = achieved
            diag.metrics_used["achieved_occupancy"] = achieved

        if theoretical is not None and achieved is not None:
            diag.occupancy_gap = theoretical - achieved
            if diag.occupancy_gap > 20:
                diag.bottlenecks.append(
                    f"Large occupancy gap: theoretical={theoretical:.1f}% "
                    f"vs achieved={achieved:.1f}% — instruction latency or "
                    "uneven block distribution"
                )
            if achieved < 25:
                diag.warnings.append(
                    f"Very low achieved occupancy ({achieved:.1f}%) — "
                    "kernel has too few active warps"
                )

    def _detect_specific_bottlenecks(
        self, parsed: list[dict], diag: BottleneckDiagnosis
    ) -> None:
        """Step 5: Check for specific common bottlenecks."""
        # Bank conflicts
        conflicts = _extract_with_aliases(parsed, "bank_conflicts")
        if conflicts is not None:
            diag.metrics_used["bank_conflicts"] = conflicts
            if conflicts > 0:
                diag.bottlenecks.append(
                    f"Shared memory bank conflicts detected ({conflicts:.0f} conflicts)"
                )

        # Warp divergence
        warp_eff = _extract_with_aliases(parsed, "warp_execution_efficiency")
        if warp_eff is not None:
            diag.metrics_used["warp_execution_efficiency"] = warp_eff
            if warp_eff < 28:  # ideal is 32 threads/warp
                diag.bottlenecks.append(
                    f"Warp divergence: only {warp_eff:.1f} threads active per "
                    "instruction (ideal=32) — excessive branching within warps"
                )

        # Uncoalesced access pattern detection
        # High L1 sectors relative to what's expected suggests uncoalesced
        l1_sectors = diag.metrics_used.get("l1_sectors")
        if l1_sectors is not None and l1_sectors > 1e7:
            diag.bottlenecks.append(
                f"Very high L1 sector requests ({l1_sectors:.0f}) — "
                "likely uncoalesced global memory access pattern"
            )

    def _generate_suggestions(self, diag: BottleneckDiagnosis) -> None:
        """Step 6: Generate optimization suggestions based on diagnosis."""
        bt = diag.primary_bottleneck
        sugg = diag.optimization_suggestions

        if bt == BottleneckType.MEMORY_BOUND:
            if diag.memory_bottleneck_level == MemoryLevel.DRAM:
                sugg.append("Reduce global memory accesses; increase data reuse")
                sugg.append("Use shared memory to stage frequently accessed data")
                sugg.append("Consider using float4/int4 vectorized loads")
            elif diag.memory_bottleneck_level == MemoryLevel.L2:
                sugg.append("Improve data locality to keep working set in L2")
                sugg.append("Consider tiling/blocking to reduce L2 ↔ DRAM traffic")
            if "uncoalesced" in " ".join(diag.bottlenecks).lower():
                sugg.append(
                    "Fix memory access pattern: ensure adjacent threads access "
                    "adjacent addresses for coalesced access"
                )

        elif bt == BottleneckType.COMPUTE_BOUND:
            if diag.tensor_core_pct < 10 and diag.compute_pct > 50:
                sugg.append(
                    "Enable Tensor Core usage: use FP16/BF16 data types "
                    "and wmma/mma intrinsics"
                )
            if diag.primary_compute_unit == ComputeUnit.FMA_FP32:
                sugg.append(
                    "Consider reducing precision (FP32 → FP16/BF16) for "
                    "higher throughput"
                )
            sugg.append("Consider algorithmic improvements to reduce compute ops")

        elif bt == BottleneckType.LATENCY_BOUND:
            sugg.append("Increase parallelism (more threads/blocks)")
            sugg.append("Use instruction-level parallelism (ILP)")
            sugg.append("Check for excessive synchronization barriers")

        # Occupancy suggestions
        if diag.occupancy_gap > 20:
            sugg.append(
                "Reduce per-thread register usage or shared memory to "
                "increase occupancy"
            )
            sugg.append("Consider using __launch_bounds__ to control register usage")

        # Bank conflict suggestions
        if any("bank conflict" in b.lower() for b in diag.bottlenecks):
            sugg.append(
                "Adjust shared memory indexing: add padding "
                "(e.g., smem[row][col+1]) to avoid bank conflicts"
            )

        # Warp divergence suggestions
        if any("warp divergence" in b.lower() for b in diag.bottlenecks):
            sugg.append(
                "Reduce if/else branches within warps; restructure data "
                "so threads in a warp follow the same path"
            )

    def generate_report(self, diag: BottleneckDiagnosis, kernel_name: str = "") -> str:
        """Generate a human-readable Markdown diagnostic report."""
        lines = [
            f"# NCU Bottleneck Analysis{f' — {kernel_name}' if kernel_name else ''}",
            "",
            "## 1. Roofline Classification",
            "",
            f"- **Primary Bottleneck**: `{diag.primary_bottleneck.value}`",
            f"- Compute Utilization (SOL): **{diag.compute_pct:.1f}%**",
            f"- Memory Throughput (SOL): **{diag.memory_pct:.1f}%**",
            "",
        ]

        if diag.primary_bottleneck == BottleneckType.COMPUTE_BOUND:
            lines.append(
                "> ⚡ This kernel is **Compute-Bound**: the GPU's compute units "
                "are the bottleneck."
            )
        elif diag.primary_bottleneck == BottleneckType.MEMORY_BOUND:
            lines.append(
                "> 💾 This kernel is **Memory-Bound**: data transfer speeds "
                "are limiting performance."
            )
        elif diag.primary_bottleneck == BottleneckType.LATENCY_BOUND:
            lines.append(
                "> ⏳ This kernel is **Latency-Bound**: low utilization across "
                "both compute and memory."
            )
        else:
            lines.append(
                "> ⚖️ This kernel is **Balanced**: similar utilization "
                "for compute and memory."
            )

        # Memory hierarchy
        lines.extend([
            "",
            "## 2. Memory Hierarchy",
            "",
            f"- L2 Throughput: {diag.l2_throughput_pct:.1f}%",
            f"- DRAM Throughput: {diag.dram_throughput_pct:.1f}%",
            f"- Memory Bottleneck Level: `{diag.memory_bottleneck_level.value}`",
            "",
        ])

        # Compute units
        lines.extend([
            "## 3. Compute Units",
            "",
            f"- Tensor Core Utilization: {diag.tensor_core_pct:.1f}%",
            f"- FMA/FP32 Utilization: {diag.fma_fp32_pct:.1f}%",
            f"- Primary Compute Unit: `{diag.primary_compute_unit.value}`",
            "",
        ])

        # Occupancy
        lines.extend([
            "## 4. Occupancy",
            "",
            f"- Theoretical: {diag.theoretical_occupancy_pct:.1f}%",
            f"- Achieved: {diag.achieved_occupancy_pct:.1f}%",
            f"- Gap: {diag.occupancy_gap:.1f}%",
            "",
        ])

        # Bottlenecks
        if diag.bottlenecks:
            lines.extend(["## 5. Detected Bottlenecks", ""])
            for b in diag.bottlenecks:
                lines.append(f"- ⚠️ {b}")
            lines.append("")

        # Suggestions
        if diag.optimization_suggestions:
            lines.extend(["## 6. Optimization Suggestions", ""])
            for i, s in enumerate(diag.optimization_suggestions, 1):
                lines.append(f"{i}. {s}")
            lines.append("")

        return "\n".join(lines)

    def analyze_and_report(
        self, ncu_output: str, kernel_name: str = ""
    ) -> tuple[BottleneckDiagnosis, str]:
        """Convenience: analyze and generate report in one call."""
        diag = self.analyze(ncu_output, kernel_name)
        report = self.generate_report(diag, kernel_name)
        return diag, report
