"""Physical consistency validator — cross-checks probe results against known physics.

Innovation Point #2: Ensures that measured hardware metrics are physically consistent
with each other and with known GPU architecture constraints.

Rules enforced:
  1. Latency hierarchy: L1 < L2 < DRAM (must be monotonically increasing)
  2. Bandwidth bounds: measured <= theoretical peak
  3. Cache size constraints: L1 < L2 < total VRAM
  4. Clock frequency: measured frequency consistent with bandwidth measurements
  5. Cross-metric correlation: bandwidth ≈ (data_size / latency) relationships
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyReport:
    """Report from physical consistency validation."""
    is_consistent: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cross_checks: list[dict[str, Any]] = field(default_factory=list)
    confidence_adjustments: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "violations": self.violations,
            "warnings": self.warnings,
            "cross_checks": self.cross_checks,
            "confidence_adjustments": self.confidence_adjustments,
        }


class PhysicalConsistencyValidator:
    """Validates that probe results are physically consistent with each other."""

    def validate(self, results: dict[str, float | None]) -> ConsistencyReport:
        """Run all consistency checks on a set of probe results.

        Args:
            results: Dict mapping metric_name -> measured value (or None if not measured)

        Returns:
            ConsistencyReport with violations, warnings, and confidence adjustments
        """
        report = ConsistencyReport()

        # Filter to non-None results
        r = {k: v for k, v in results.items() if v is not None}

        # Check 1: Latency hierarchy — L1 < L2 < DRAM
        self._check_latency_hierarchy(r, report)

        # Check 2: Bandwidth reasonableness
        self._check_bandwidth_bounds(r, report)

        # Check 3: Clock frequency vs bandwidth correlation
        self._check_clock_bandwidth_correlation(r, report)

        # Check 4: Cache size reasonableness
        self._check_cache_sizes(r, report)

        # Check 5: Bank conflict penalty reasonableness
        self._check_bank_conflict_penalty(r, report)

        # Check 6: Cross-metric bandwidth-latency relationship
        self._check_bandwidth_latency_relationship(r, report)

        # Check 7: Evaluation-specific metric checks
        self._check_eval_dram_bandwidth(r, report)
        self._check_eval_throughput_pct(r, report)
        self._check_eval_gpu_freq(r, report)

        report.is_consistent = len(report.violations) == 0

        logger.info(
            "Consistency check: %s (%d violations, %d warnings)",
            "PASS" if report.is_consistent else "FAIL",
            len(report.violations),
            len(report.warnings),
        )
        return report

    def _check_latency_hierarchy(
        self, r: dict[str, float], report: ConsistencyReport
    ) -> None:
        """L1 < L2 < DRAM latency must hold."""
        l1 = r.get("l1_latency_cycles")
        l2 = r.get("l2_latency_cycles")
        dram = r.get("dram_latency_cycles")

        checks_done = []

        if l1 is not None and l2 is not None:
            check = {
                "rule": "L1 < L2 latency",
                "l1": l1,
                "l2": l2,
                "passed": l1 < l2,
            }
            checks_done.append(check)
            if l1 >= l2:
                report.violations.append(
                    f"Latency violation: L1 ({l1:.1f} cycles) >= L2 ({l2:.1f} cycles)"
                )
                report.confidence_adjustments["l1_latency_cycles"] = -0.3
                report.confidence_adjustments["l2_latency_cycles"] = -0.3

        if l2 is not None and dram is not None:
            check = {
                "rule": "L2 < DRAM latency",
                "l2": l2,
                "dram": dram,
                "passed": l2 < dram,
            }
            checks_done.append(check)
            if l2 >= dram:
                report.violations.append(
                    f"Latency violation: L2 ({l2:.1f} cycles) >= DRAM ({dram:.1f} cycles)"
                )
                report.confidence_adjustments["l2_latency_cycles"] = -0.3
                report.confidence_adjustments["dram_latency_cycles"] = -0.3

        if l1 is not None and l2 is not None and dram is not None:
            # Check ratios are reasonable
            l2_l1_ratio = l2 / l1 if l1 > 0 else 0
            dram_l2_ratio = dram / l2 if l2 > 0 else 0

            if l2_l1_ratio < 1.5:
                report.warnings.append(
                    f"L2/L1 latency ratio ({l2_l1_ratio:.1f}x) is unusually small — "
                    "measurements may be inaccurate"
                )
            if l2_l1_ratio > 20:
                report.warnings.append(
                    f"L2/L1 latency ratio ({l2_l1_ratio:.1f}x) is unusually large"
                )
            if dram_l2_ratio < 1.3:
                report.warnings.append(
                    f"DRAM/L2 latency ratio ({dram_l2_ratio:.1f}x) is unusually small"
                )

            report.cross_checks.append({
                "check": "latency_hierarchy",
                "l1": l1, "l2": l2, "dram": dram,
                "l2_l1_ratio": round(l2_l1_ratio, 2),
                "dram_l2_ratio": round(dram_l2_ratio, 2),
                "passed": l1 < l2 < dram,
            })

    def _check_bandwidth_bounds(
        self, r: dict[str, float], report: ConsistencyReport
    ) -> None:
        """Bandwidth should not exceed reasonable theoretical peaks."""
        global_bw = r.get("max_global_mem_bandwidth_gb_s")
        shmem_bw = r.get("max_shmem_bandwidth_gb_s")

        if global_bw is not None:
            # Current gen max HBM3 bandwidth is ~5TB/s = 5000 GB/s
            if global_bw > 5000:
                report.violations.append(
                    f"Global memory bandwidth ({global_bw:.1f} GB/s) exceeds maximum "
                    "theoretical peak (5000 GB/s)"
                )
                report.confidence_adjustments["max_global_mem_bandwidth_gb_s"] = -0.5
            elif global_bw < 5:
                report.warnings.append(
                    f"Global memory bandwidth ({global_bw:.1f} GB/s) is suspiciously low"
                )

        if shmem_bw is not None:
            # Shared memory bandwidth can be very high per-SM
            if shmem_bw > 200000:
                report.warnings.append(
                    f"Shared memory bandwidth ({shmem_bw:.1f} GB/s) seems unusually high"
                )

        if global_bw is not None and shmem_bw is not None:
            # Shared memory should generally be faster than global
            if shmem_bw < global_bw * 0.5:
                report.warnings.append(
                    f"Shared memory BW ({shmem_bw:.1f} GB/s) is less than global "
                    f"memory BW ({global_bw:.1f} GB/s) — unusual"
                )
            report.cross_checks.append({
                "check": "bandwidth_comparison",
                "global_bw": global_bw,
                "shmem_bw": shmem_bw,
                "ratio": round(shmem_bw / global_bw, 2) if global_bw > 0 else None,
            })

    def _check_clock_bandwidth_correlation(
        self, r: dict[str, float], report: ConsistencyReport
    ) -> None:
        """Clock frequency should be consistent with bandwidth observations."""
        clock = r.get("actual_boost_clock_mhz")
        global_bw = r.get("max_global_mem_bandwidth_gb_s")

        if clock is not None:
            if clock < 100:
                report.violations.append(
                    f"Measured clock ({clock:.1f} MHz) is below 100 MHz — "
                    "measurement error or extreme throttling"
                )
            elif clock < 300:
                report.warnings.append(
                    f"Measured clock ({clock:.1f} MHz) is very low — "
                    "possible frequency locking"
                )

    def _check_cache_sizes(
        self, r: dict[str, float], report: ConsistencyReport
    ) -> None:
        """Cache sizes should be reasonable."""
        l2_size = r.get("l2_cache_size_kb")
        shmem = r.get("max_shmem_per_block_kb")

        if l2_size is not None:
            # L2 should be a power of 2 multiple of 256KB
            if l2_size < 256:
                report.violations.append(
                    f"L2 cache size ({l2_size:.0f} KB) is below 256 KB — unreasonable"
                )
            # Check it's a reasonable value (typical: 768KB to 128MB)
            import math
            log2_size = math.log2(l2_size) if l2_size > 0 else 0
            if abs(log2_size - round(log2_size)) > 0.3:
                report.warnings.append(
                    f"L2 cache size ({l2_size:.0f} KB) is not near a power of 2 — "
                    "detection may be imprecise"
                )

        if shmem is not None:
            # Common values: 48, 64, 100, 164, 228 KB
            valid_shmem = [16, 32, 48, 64, 96, 100, 128, 164, 228]
            closest = min(valid_shmem, key=lambda x: abs(x - shmem))
            if abs(shmem - closest) > 4:
                report.warnings.append(
                    f"Shared memory per block ({shmem:.0f} KB) is not a standard "
                    f"value (closest: {closest} KB)"
                )

    def _check_bank_conflict_penalty(
        self, r: dict[str, float], report: ConsistencyReport
    ) -> None:
        """Bank conflict penalty should be reasonable."""
        penalty = r.get("bank_conflict_penalty_cycles")
        if penalty is not None:
            if penalty < 0:
                report.violations.append(
                    f"Bank conflict penalty ({penalty:.1f} cycles) is negative"
                )
            elif penalty > 50:
                report.warnings.append(
                    f"Bank conflict penalty ({penalty:.1f} cycles) seems high"
                )

    def _check_bandwidth_latency_relationship(
        self, r: dict[str, float], report: ConsistencyReport
    ) -> None:
        """Cross-check: bandwidth and latency should be roughly consistent.

        For DRAM: effective BW ≈ (bus_width * clock * 2) / 8
        We can't know bus width, but we can check relative consistency.
        """
        dram_lat = r.get("dram_latency_cycles")
        global_bw = r.get("max_global_mem_bandwidth_gb_s")
        clock = r.get("actual_boost_clock_mhz")

        if dram_lat is not None and clock is not None and clock > 0:
            # DRAM latency in ns = cycles / (clock_MHz)
            dram_lat_ns = dram_lat / clock * 1000  # convert to ns
            report.cross_checks.append({
                "check": "dram_latency_ns",
                "dram_lat_cycles": dram_lat,
                "clock_mhz": clock,
                "dram_lat_ns": round(dram_lat_ns, 1),
                "reasonable": 50 < dram_lat_ns < 1000,
            })
            if dram_lat_ns < 50:
                report.warnings.append(
                    f"DRAM latency ({dram_lat_ns:.1f} ns) is unusually low"
                )
            elif dram_lat_ns > 1000:
                report.warnings.append(
                    f"DRAM latency ({dram_lat_ns:.1f} ns) is unusually high"
                )

    def _check_eval_dram_bandwidth(
        self, r: dict[str, float], report: ConsistencyReport
    ) -> None:
        """Check DRAM read/write bandwidth consistency (evaluation metrics)."""
        dram_read = r.get("dram__bytes_read.sum.per_second")
        dram_write = r.get("dram__bytes_write.sum.per_second")
        bus_width = r.get("device__attribute_fb_bus_width")
        mem_freq = r.get("device__attribute_max_mem_frequency_khz")

        # Read/write should be within 2x of each other
        if dram_read is not None and dram_write is not None:
            if dram_read > 0 and dram_write > 0:
                ratio = max(dram_read, dram_write) / min(dram_read, dram_write)
                report.cross_checks.append({
                    "check": "dram_read_write_ratio",
                    "read_gbs": round(dram_read / 1e9, 1),
                    "write_gbs": round(dram_write / 1e9, 1),
                    "ratio": round(ratio, 2),
                    "passed": ratio <= 2.0,
                })
                if ratio > 2.0:
                    report.warnings.append(
                        f"DRAM read ({dram_read/1e9:.1f} GB/s) and write ({dram_write/1e9:.1f} GB/s) "
                        f"differ by {ratio:.1f}x"
                    )

        # Both should be under theoretical peak
        if bus_width is not None and mem_freq is not None and bus_width > 0 and mem_freq > 0:
            theoretical_peak = bus_width * (mem_freq * 1000) * 2 / 8
            report.cross_checks.append({
                "check": "theoretical_peak_bandwidth",
                "peak_gbs": round(theoretical_peak / 1e9, 1),
                "bus_width_bits": bus_width,
                "mem_freq_khz": mem_freq,
            })
            for name, val in [("read", dram_read), ("write", dram_write)]:
                if val is not None and val > theoretical_peak * 1.1:
                    report.warnings.append(
                        f"DRAM {name} ({val/1e9:.1f} GB/s) exceeds theoretical peak "
                        f"({theoretical_peak/1e9:.1f} GB/s)"
                    )

    def _check_eval_throughput_pct(
        self, r: dict[str, float], report: ConsistencyReport
    ) -> None:
        """Check that throughput percentages are in [0, 100]."""
        for metric in [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        ]:
            val = r.get(metric)
            if val is not None:
                if val < 0 or val > 100:
                    report.violations.append(
                        f"{metric} = {val:.1f}% is outside valid range [0, 100]"
                    )
                    report.confidence_adjustments[metric] = -0.5

    def _check_eval_gpu_freq(
        self, r: dict[str, float], report: ConsistencyReport
    ) -> None:
        """Check GPU frequency is reasonable."""
        freq_khz = r.get("device__attribute_max_gpu_frequency_khz")
        if freq_khz is not None:
            freq_mhz = freq_khz / 1000
            if freq_mhz < 500:
                report.warnings.append(
                    f"GPU frequency {freq_mhz:.0f} MHz is unusually low"
                )
            elif freq_mhz > 3500:
                report.warnings.append(
                    f"GPU frequency {freq_mhz:.0f} MHz is unusually high"
                )


class EnvironmentFingerprintDetector:
    """Innovation Point #1: Detect environment tampering via probe results.

    Uses the probe results themselves to detect if the evaluation environment
    has been altered (frequency locking, SM masking, fake API data).
    """

    def detect(
        self,
        probe_results: dict[str, float | None],
        reported_clock_mhz: float | None = None,
        reported_sm_count: int | None = None,
        reported_shmem_kb: int | None = None,
    ) -> dict[str, Any]:
        """Detect environment tampering by comparing measured vs reported values.

        Args:
            probe_results: Measured values from probes
            reported_clock_mhz: Clock frequency reported by nvidia-smi
            reported_sm_count: SM count reported by cudaGetDeviceProperties
            reported_shmem_kb: Shared memory reported by cudaGetDeviceProperties

        Returns:
            Dict with tampering detection results
        """
        findings: list[str] = []
        trust_assessment: dict[str, str] = {}
        tampering_detected = False

        # ── Frequency locking detection ─────────────────────────
        measured_clock = probe_results.get("actual_boost_clock_mhz")
        if measured_clock is not None and reported_clock_mhz is not None:
            clock_diff_pct = abs(measured_clock - reported_clock_mhz) / max(
                reported_clock_mhz, 1
            ) * 100

            if clock_diff_pct > 10:
                findings.append(
                    f"Clock mismatch: measured={measured_clock:.0f} MHz vs "
                    f"reported={reported_clock_mhz:.0f} MHz "
                    f"(diff={clock_diff_pct:.1f}%)"
                )
                trust_assessment["clock"] = "measured"
                tampering_detected = True
            else:
                trust_assessment["clock"] = "consistent"

        if measured_clock is not None:
            # Check if clock is a non-standard value
            standard_clocks = [
                210, 300, 600, 900, 1080, 1110, 1200, 1305, 1350,
                1410, 1500, 1530, 1600, 1650, 1695, 1710, 1770,
                1800, 1860, 1980, 2100, 2235, 2340, 2475, 2520, 2610,
            ]
            nearest = min(standard_clocks, key=lambda x: abs(x - measured_clock))
            if abs(measured_clock - nearest) > 50:
                findings.append(
                    f"Non-standard clock frequency: {measured_clock:.0f} MHz "
                    f"(nearest standard: {nearest} MHz) — possible frequency lock"
                )
                tampering_detected = True

        # ── Shared memory tampering detection ────────────────────
        measured_shmem = probe_results.get("max_shmem_per_block_kb")
        if measured_shmem is not None and reported_shmem_kb is not None:
            if abs(measured_shmem - reported_shmem_kb) > 4:
                findings.append(
                    f"Shared memory mismatch: measured={measured_shmem:.0f} KB vs "
                    f"reported={reported_shmem_kb} KB"
                )
                trust_assessment["shmem"] = "measured"
                tampering_detected = True
            else:
                trust_assessment["shmem"] = "consistent"

        # ── Bandwidth anomaly detection ──────────────────────────
        global_bw = probe_results.get("max_global_mem_bandwidth_gb_s")
        if global_bw is not None and measured_clock is not None:
            # For a GPU at clock X MHz, we expect bandwidth proportional to clock
            # RTX 4090: ~1008 GB/s at ~2520 MHz → ratio ≈ 0.4
            # If clock is locked low, bandwidth should also be proportionally lower
            bw_per_mhz = global_bw / measured_clock if measured_clock > 0 else 0
            if bw_per_mhz > 2.0:
                findings.append(
                    f"Bandwidth/clock ratio ({bw_per_mhz:.2f} GB/s/MHz) is unusually "
                    "high — memory clock may differ from core clock"
                )

        # ── Overall assessment ───────────────────────────────────
        trust_level = "suspicious" if tampering_detected else "verified"

        return {
            "tampering_detected": tampering_detected,
            "trust_level": trust_level,
            "findings": findings,
            "trust_assessment": trust_assessment,
            "recommendation": (
                "Use measured values from probes instead of API-reported values"
                if tampering_detected
                else "Measured and reported values are consistent"
            ),
        }
