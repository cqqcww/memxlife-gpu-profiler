"""Metrics catalog — pre-defined probe strategies for each target hardware metric."""

from __future__ import annotations

from core.models import MetricSpec, ProbeStrategy


def build_catalog() -> dict[str, MetricSpec]:
    """Return the full catalog of supported hardware metrics and their probe strategies."""
    return {
        "dram_latency_cycles": MetricSpec(
            name="dram_latency_cycles",
            description="DRAM (global memory) access latency in GPU clock cycles",
            unit="cycles",
            strategies=[
                ProbeStrategy(
                    name="pointer_chase_large",
                    probe_template="latency_pointer_chase",
                    params={"data_size_mb": 256, "stride_mode": "random", "iterations": 1000, "warmup": 100},
                    priority=1,
                    description="Pointer chasing over large array (exceeds L2) to measure DRAM latency",
                ),
                ProbeStrategy(
                    name="pointer_chase_sequential",
                    probe_template="latency_pointer_chase",
                    params={"data_size_mb": 512, "stride_mode": "large_stride", "iterations": 500, "warmup": 50},
                    priority=2,
                    description="Pointer chasing with large stride to force DRAM access",
                ),
            ],
            cross_verify=True,
            tolerance_pct=10,
            physical_min=200,
            physical_max=1500,
        ),
        "l1_latency_cycles": MetricSpec(
            name="l1_latency_cycles",
            description="L1 cache access latency in GPU clock cycles",
            unit="cycles",
            strategies=[
                ProbeStrategy(
                    name="pointer_chase_small",
                    probe_template="latency_pointer_chase",
                    params={"data_size_kb": 8, "stride_mode": "sequential", "iterations": 5000, "warmup": 500},
                    priority=1,
                    description="Pointer chasing over small array (fits in L1) to measure L1 latency",
                ),
            ],
            cross_verify=False,
            tolerance_pct=15,
            physical_min=10,
            physical_max=80,
        ),
        "l2_latency_cycles": MetricSpec(
            name="l2_latency_cycles",
            description="L2 cache access latency in GPU clock cycles",
            unit="cycles",
            strategies=[
                ProbeStrategy(
                    name="pointer_chase_medium",
                    probe_template="latency_pointer_chase",
                    params={"data_size_kb": 512, "stride_mode": "random", "iterations": 2000, "warmup": 200},
                    priority=1,
                    description="Pointer chasing over medium array (exceeds L1, fits in L2)",
                ),
            ],
            cross_verify=False,
            tolerance_pct=10,
            physical_min=100,
            physical_max=500,
        ),
        "l2_cache_size_kb": MetricSpec(
            name="l2_cache_size_kb",
            description="L2 cache capacity in kilobytes",
            unit="KB",
            strategies=[
                ProbeStrategy(
                    name="cache_sweep",
                    probe_template="cache_size_sweep",
                    params={"min_kb": 64, "max_kb": 65536, "steps": 24, "iterations": 500},
                    priority=1,
                    description="Sweep data sizes and detect latency cliff at L2 boundary",
                ),
            ],
            cross_verify=False,
            tolerance_pct=5,
            physical_min=256,
            physical_max=131072,
        ),
        "max_global_mem_bandwidth_gb_s": MetricSpec(
            name="max_global_mem_bandwidth_gb_s",
            description="Peak achievable global memory (VRAM) bandwidth in GB/s",
            unit="GB/s",
            strategies=[
                ProbeStrategy(
                    name="bandwidth_stream",
                    probe_template="bandwidth_global",
                    params={"data_size_mb": 256, "iterations": 20, "warmup": 5},
                    priority=1,
                    description="Streaming read/write to measure peak global memory bandwidth",
                ),
                ProbeStrategy(
                    name="bandwidth_ncu",
                    probe_template=None,
                    ncu_metrics=["dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes.sum"],
                    needs_ncu=True,
                    priority=2,
                    description="Use ncu to measure DRAM throughput directly",
                ),
            ],
            cross_verify=True,
            tolerance_pct=5,
            physical_min=10,
            physical_max=5000,
        ),
        "max_shmem_bandwidth_gb_s": MetricSpec(
            name="max_shmem_bandwidth_gb_s",
            description="Peak achievable shared memory bandwidth in GB/s",
            unit="GB/s",
            strategies=[
                ProbeStrategy(
                    name="bandwidth_shmem",
                    probe_template="bandwidth_shared",
                    params={"block_size": 256, "iterations": 100, "warmup": 20},
                    priority=1,
                    description="Shared memory read/write bandwidth measurement",
                ),
            ],
            cross_verify=False,
            tolerance_pct=10,
            physical_min=1,
            physical_max=50000,
        ),
        "actual_boost_clock_mhz": MetricSpec(
            name="actual_boost_clock_mhz",
            description="Actual sustained GPU core clock frequency under load in MHz",
            unit="MHz",
            strategies=[
                ProbeStrategy(
                    name="fma_clock_probe",
                    probe_template="clock_frequency",
                    params={"fma_iterations": 1000000, "warmup": 100000, "trials": 5},
                    priority=1,
                    description="Measure actual clock via timed FMA loop with known instruction count",
                ),
                ProbeStrategy(
                    name="nvidia_smi_clock",
                    probe_template=None,
                    params={"use_nvidia_smi": True},
                    priority=3,
                    description="Read clock from nvidia-smi (may be inaccurate if locked)",
                ),
            ],
            cross_verify=True,
            tolerance_pct=2,
            physical_min=100,
            physical_max=3500,
        ),
        "bank_conflict_penalty_cycles": MetricSpec(
            name="bank_conflict_penalty_cycles",
            description="Latency penalty of a shared memory bank conflict vs conflict-free access",
            unit="cycles",
            strategies=[
                ProbeStrategy(
                    name="bank_conflict_probe",
                    probe_template="bank_conflict",
                    params={"block_size": 256, "iterations": 1000, "warmup": 100},
                    priority=1,
                    description="Compare conflict-free vs conflicting shared memory access patterns",
                ),
            ],
            cross_verify=False,
            tolerance_pct=20,
            physical_min=0,
            physical_max=100,
        ),
        "max_shmem_per_block_kb": MetricSpec(
            name="max_shmem_per_block_kb",
            description="Maximum shared memory per thread block in KB",
            unit="KB",
            strategies=[
                ProbeStrategy(
                    name="shmem_alloc_probe",
                    probe_template="shmem_capacity",
                    params={"start_kb": 16, "max_kb": 228, "step_kb": 4},
                    priority=1,
                    description="Binary search for max shared memory allocation per block",
                ),
            ],
            cross_verify=False,
            tolerance_pct=0,
            physical_min=16,
            physical_max=228,
        ),
    }


def get_metric_spec(metric_name: str) -> MetricSpec | None:
    catalog = build_catalog()
    return catalog.get(metric_name)


def list_supported_metrics() -> list[str]:
    return list(build_catalog().keys())
