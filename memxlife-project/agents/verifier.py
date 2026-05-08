"""Verifier agent — cross-validates probe results for physical consistency."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VerifierAgent:
    """Cross-validates all measured results against each other and known physics."""

    def verify(
        self,
        results: dict[str, float],
        environment: dict[str, Any],
    ) -> dict[str, Any]:
        """Run all verification checks and return a structured report.

        Args:
            results: metric_name -> numeric value
            environment: environment profile dict from scout

        Returns:
            {status: "pass"/"fail", issues: [...], suggested_followups: [...], summary: "..."}
        """
        issues: list[str] = []
        notes: list[str] = []

        sm_count = results.get("launch__sm_count")
        dram_read = results.get("dram__bytes_read.sum.per_second")
        dram_write = results.get("dram__bytes_write.sum.per_second")
        gpu_freq = results.get("device__attribute_max_gpu_frequency_khz")
        mem_freq = results.get("device__attribute_max_mem_frequency_khz")
        bus_width = results.get("device__attribute_fb_bus_width")
        sm_tput = results.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        mem_tput = results.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")

        # Check 1: SM count should be reasonable (1-256)
        if sm_count is not None:
            if sm_count < 1 or sm_count > 256:
                issues.append(f"SM count {sm_count} is out of reasonable range [1, 256]")
            else:
                notes.append(f"SM count {int(sm_count)} is plausible")

        # Check 2: DRAM read/write should be roughly similar
        if dram_read is not None and dram_write is not None:
            if dram_read > 0 and dram_write > 0:
                ratio = max(dram_read, dram_write) / min(dram_read, dram_write)
                if ratio > 2.0:
                    issues.append(
                        f"DRAM read ({dram_read/1e9:.1f} GB/s) and write ({dram_write/1e9:.1f} GB/s) "
                        f"differ by {ratio:.1f}x — expected within 2x"
                    )
                else:
                    notes.append(
                        f"DRAM read/write ratio {ratio:.2f}x is within normal range"
                    )

        # Check 3: DRAM bandwidth vs theoretical peak
        if bus_width is not None and mem_freq is not None and bus_width > 0 and mem_freq > 0:
            # Theoretical peak = bus_width_bits * mem_freq_hz * 2 (DDR) / 8 (bits to bytes)
            # mem_freq is in kHz, so *1000 for Hz
            theoretical_peak = bus_width * (mem_freq * 1000) * 2 / 8
            notes.append(f"Theoretical peak bandwidth: {theoretical_peak/1e9:.1f} GB/s")

            for name, val in [("read", dram_read), ("write", dram_write)]:
                if val is not None:
                    pct_of_peak = val / theoretical_peak * 100 if theoretical_peak > 0 else 0
                    if val > theoretical_peak * 1.1:
                        issues.append(
                            f"DRAM {name} ({val/1e9:.1f} GB/s) exceeds theoretical peak "
                            f"({theoretical_peak/1e9:.1f} GB/s) by {pct_of_peak-100:.0f}%"
                        )
                    elif pct_of_peak < 5:
                        issues.append(
                            f"DRAM {name} ({val/1e9:.1f} GB/s) is only {pct_of_peak:.1f}% of "
                            f"theoretical peak — suspiciously low"
                        )
                    else:
                        notes.append(
                            f"DRAM {name}: {val/1e9:.1f} GB/s = {pct_of_peak:.1f}% of peak"
                        )

        # Check 4: Throughput percentages must be 0-100
        if sm_tput is not None:
            if sm_tput < 0 or sm_tput > 100:
                issues.append(f"SM throughput {sm_tput:.1f}% is outside [0, 100] range")
            else:
                notes.append(f"SM throughput {sm_tput:.1f}% is in valid range")

        if mem_tput is not None:
            if mem_tput < 0 or mem_tput > 100:
                issues.append(f"Memory throughput {mem_tput:.1f}% is outside [0, 100] range")
            else:
                notes.append(f"Memory throughput {mem_tput:.1f}% is in valid range")

        # Check 5: Memory throughput consistency with bandwidth measurement
        if mem_tput is not None and dram_read is not None and bus_width and mem_freq:
            theoretical_peak = bus_width * (mem_freq * 1000) * 2 / 8
            if theoretical_peak > 0:
                implied_pct = dram_read / theoretical_peak * 100
                if abs(implied_pct - mem_tput) > 30:
                    notes.append(
                        f"Memory throughput ({mem_tput:.1f}%) and implied from DRAM read "
                        f"({implied_pct:.1f}%) differ significantly — different measurement methods"
                    )

        # Check 6: GPU frequency reasonableness
        if gpu_freq is not None:
            freq_mhz = gpu_freq / 1000
            if freq_mhz < 500 or freq_mhz > 3500:
                issues.append(f"GPU frequency {freq_mhz:.0f} MHz is outside reasonable range [500, 3500]")
            else:
                notes.append(f"GPU frequency {freq_mhz:.0f} MHz is reasonable")

        # Check 7: Bus width should be standard
        if bus_width is not None:
            standard_widths = [64, 128, 192, 256, 320, 384, 512, 768, 1024, 2048, 4096]
            if bus_width not in standard_widths:
                issues.append(f"Bus width {bus_width} bits is non-standard")
            else:
                notes.append(f"Bus width {int(bus_width)} bits is a standard value")

        # Build summary
        status = "pass" if len(issues) == 0 else "fail"
        n_metrics = sum(1 for v in results.values() if v is not None)
        summary = (
            f"Verified {n_metrics} metrics. "
            + (f"All checks passed. " if not issues else f"Found {len(issues)} issue(s). ")
            + " ".join(notes[:3])
        )

        report = {
            "status": status,
            "issues": issues,
            "notes": notes,
            "suggested_followups": [],
            "summary": summary,
        }

        logger.info("Verifier: %s (%d issues, %d notes)", status, len(issues), len(notes))
        return report
