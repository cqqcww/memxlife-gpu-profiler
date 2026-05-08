"""Unit tests for the memxlife framework."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import ProbeResult, ProbeStrategy, MetricSpec, EnvironmentProfile, AgentContext
from knowledge.metrics_catalog import build_catalog, get_metric_spec, list_supported_metrics
from knowledge.store import KnowledgeStore
from parser.probe_parser import parse_probe_output, extract_primary_value
from parser.ncu_parser import parse_ncu_csv, extract_ncu_metric


# ── Probe output parser ─────────────────────────────────────

class TestProbeParser:
    def test_parse_basic_result(self):
        stdout = "RESULT:dram_latency_cycles=442.5\nUNIT:cycles\nMETHOD:pointer_chase\nITERATIONS:1000\nWARMUP:100\n"
        parsed = parse_probe_output(stdout)
        assert parsed["values"]["dram_latency_cycles"] == 442.5
        assert parsed["unit"] == "cycles"
        assert parsed["method"] == "pointer_chase"
        assert parsed["iterations"] == 1000
        assert parsed["warmup"] == 100

    def test_parse_integer_result(self):
        stdout = "RESULT:l2_cache_size_kb=49152\nUNIT:KB\n"
        parsed = parse_probe_output(stdout)
        assert parsed["values"]["l2_cache_size_kb"] == 49152.0

    def test_parse_multiple_results(self):
        stdout = "RESULT:copy_bw=800.5\nRESULT:read_bw=900.2\nUNIT:GB/s\n"
        parsed = parse_probe_output(stdout)
        assert parsed["values"]["copy_bw"] == 800.5
        assert parsed["values"]["read_bw"] == 900.2

    def test_parse_empty_output(self):
        parsed = parse_probe_output("")
        assert parsed["values"] == {}
        assert parsed["unit"] == ""

    def test_parse_noisy_output(self):
        stdout = "Some debug info\nwarning: something\nRESULT:actual_boost_clock_mhz=2520.0\nmore noise\n"
        parsed = parse_probe_output(stdout)
        assert parsed["values"]["actual_boost_clock_mhz"] == 2520.0

    def test_parse_error_line(self):
        stdout = "RESULT:foo=1.0\nERROR:something went wrong\n"
        parsed = parse_probe_output(stdout)
        assert "something went wrong" in parsed["errors"]

    def test_extract_primary_direct_match(self):
        parsed = {"values": {"dram_latency_cycles": 442.0}, "unit": "cycles"}
        val = extract_primary_value(parsed, "dram_latency_cycles")
        assert val == 442.0

    def test_extract_primary_partial_match(self):
        parsed = {"values": {"latency_cycles": 442.0}}
        val = extract_primary_value(parsed, "dram_latency_cycles")
        assert val == 442.0

    def test_extract_primary_single_value(self):
        parsed = {"values": {"some_metric": 99.0}}
        val = extract_primary_value(parsed, "totally_different")
        assert val == 99.0

    def test_extract_primary_no_values(self):
        parsed = {"values": {}}
        val = extract_primary_value(parsed, "dram_latency_cycles")
        assert val is None


# ── NCU parser ───────────────────────────────────────────────

class TestNcuParser:
    def test_parse_csv_basic(self):
        csv_text = '"Metric Name","Metric Value","Metric Unit"\n"dram__bytes.sum","1234567890","bytes"\n'
        result = parse_ncu_csv(csv_text)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "dram__bytes.sum"
        assert result[0]["value"] == 1234567890.0

    def test_parse_empty(self):
        result = parse_ncu_csv("")
        assert result == []

    def test_extract_metric_found(self):
        parsed = [{"name": "dram__bytes.sum", "value": 1234567890.0}]
        val = extract_ncu_metric(parsed, "dram__bytes.sum")
        assert val == 1234567890.0

    def test_extract_metric_missing(self):
        parsed = [{"name": "other__metric", "value": 100.0}]
        val = extract_ncu_metric(parsed, "dram__bytes.sum")
        assert val is None

    def test_extract_metric_empty(self):
        val = extract_ncu_metric([], "dram__bytes.sum")
        assert val is None


# ── Metrics catalog ──────────────────────────────────────────

class TestMetricsCatalog:
    def test_all_9_metrics(self):
        catalog = build_catalog()
        expected = [
            "dram_latency_cycles", "l1_latency_cycles", "l2_latency_cycles",
            "l2_cache_size_kb", "max_global_mem_bandwidth_gb_s",
            "max_shmem_bandwidth_gb_s", "actual_boost_clock_mhz",
            "bank_conflict_penalty_cycles", "max_shmem_per_block_kb",
        ]
        for name in expected:
            assert name in catalog, f"Missing metric: {name}"
        assert len(catalog) == 9

    def test_each_metric_has_strategies(self):
        for name, spec in build_catalog().items():
            assert len(spec.strategies) >= 1, f"{name} has no strategies"

    def test_params_are_uppercase(self):
        for name, spec in build_catalog().items():
            for strat in spec.strategies:
                for key in strat.params:
                    assert key.isupper(), f"{name}/{strat.name}: param '{key}' not uppercase"

    def test_no_static_templates(self):
        """All probe_template fields should be None — code is LLM-generated."""
        for name, spec in build_catalog().items():
            for strat in spec.strategies:
                assert strat.probe_template is None, \
                    f"{name}/{strat.name}: probe_template should be None"

    def test_physical_bounds(self):
        for name, spec in build_catalog().items():
            assert spec.physical_min < spec.physical_max, \
                f"{name}: physical_min >= physical_max"

    def test_get_metric_spec(self):
        spec = get_metric_spec("dram_latency_cycles")
        assert spec is not None
        assert spec.unit == "cycles"

    def test_get_nonexistent(self):
        assert get_metric_spec("nonexistent") is None

    def test_list_supported_metrics(self):
        assert len(list_supported_metrics()) == 9


# ── Knowledge store ──────────────────────────────────────────

class TestKnowledgeStore:
    def test_add_and_get(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb.json")
        store.add_observation(ProbeResult(
            metric_name="test_metric", value=42.0, unit="units",
            confidence=0.9, method="test", strategy_name="test_strat",
        ))
        history = store.get_metric_history("test_metric")
        assert len(history) == 1
        assert history[0]["value"] == 42.0

    def test_get_best(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb.json")
        store.add_observation(ProbeResult("m", 10.0, "u", 0.5, "a", "s1"))
        store.add_observation(ProbeResult("m", 20.0, "u", 0.9, "b", "s2"))
        best = store.get_best_for_metric("m")
        assert best is not None
        assert best["value"] == 20.0
        assert best["confidence"] == 0.9

    def test_get_empty(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb.json")
        assert store.get_metric_history("nonexistent") == []
        assert store.get_best_for_metric("nonexistent") is None

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "kb.json"
        store = KnowledgeStore(path)
        store.add_observation(ProbeResult("m", 42.0, "u", 0.9, "a", "s"))
        store.save()
        store2 = KnowledgeStore(path)
        assert len(store2.get_metric_history("m")) == 1

    def test_summary(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb.json")
        store.add_observation(ProbeResult("m", 42.0, "u", 0.9, "a", "s"))
        summary = store.summary_for_prompt()
        assert "m" in summary
        assert isinstance(summary, str)


# ── Models ───────────────────────────────────────────────────

class TestModels:
    def test_probe_result_to_dict(self):
        pr = ProbeResult("m", 42.0, "u", 0.9, "method", "strat")
        d = pr.to_dict()
        assert d["metric_name"] == "m"
        assert d["value"] == 42.0
        assert d["confidence"] == 0.9

    def test_probe_strategy_to_dict(self):
        ps = ProbeStrategy(name="t", probe_template="tpl", params={"A": 1}, priority=1, description="d")
        d = ps.to_dict()
        assert d["name"] == "t"
        assert d["probe_template"] == "tpl"

    def test_environment_defaults(self):
        env = EnvironmentProfile()
        assert env.gpu_name == "unknown"
        assert env.trust_level == "untrusted"
        summary = env.summary_for_prompt()
        assert "unknown" in summary

    def test_agent_context_mock(self):
        ctx = AgentContext(mock_mode=True)
        assert ctx.mock_mode is True
        assert isinstance(ctx.environment, EnvironmentProfile)


# ── Integration: mock end-to-end ─────────────────────────────

class TestMockIntegration:
    def test_full_mock_all_metrics(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py", "--mock", "tests/full_target_spec.json"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        # Find JSON block in output
        lines = result.stdout.strip().split("\n")
        json_start = json_end = None
        for i, line in enumerate(lines):
            if line.strip() == "{":
                json_start = i
            if line.strip() == "}":
                json_end = i
        assert json_start is not None and json_end is not None
        data = json.loads("\n".join(lines[json_start:json_end + 1]))
        assert len(data) == 9
        for key, val in data.items():
            assert isinstance(val, (int, float)) and val > 0, f"{key}={val}"


# ── NCU Bottleneck Analyzer ──────────────────────────────────

from analysis.ncu_bottleneck import NcuBottleneckAnalyzer, BottleneckType, MemoryLevel


class TestNcuBottleneckAnalyzer:
    def _make_csv(self, metrics: dict[str, float]) -> str:
        """Build a fake ncu CSV output from metric name→value pairs."""
        lines = ['"Metric Name","Metric Value","Metric Unit"']
        for name, val in metrics.items():
            lines.append(f'"{name}","{val}","%"')
        return "\n".join(lines)

    def test_compute_bound(self):
        csv = self._make_csv({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
        })
        analyzer = NcuBottleneckAnalyzer()
        diag = analyzer.analyze(csv)
        assert diag.primary_bottleneck == BottleneckType.COMPUTE_BOUND
        assert diag.compute_pct == 85.0
        assert diag.memory_pct == 30.0

    def test_memory_bound(self):
        csv = self._make_csv({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 75.0,
        })
        analyzer = NcuBottleneckAnalyzer()
        diag = analyzer.analyze(csv)
        assert diag.primary_bottleneck == BottleneckType.MEMORY_BOUND
        assert diag.memory_bottleneck_level == MemoryLevel.DRAM

    def test_latency_bound(self):
        csv = self._make_csv({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 5.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 3.0,
        })
        analyzer = NcuBottleneckAnalyzer()
        diag = analyzer.analyze(csv)
        assert diag.primary_bottleneck == BottleneckType.LATENCY_BOUND

    def test_balanced(self):
        csv = self._make_csv({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 55.0,
        })
        analyzer = NcuBottleneckAnalyzer()
        diag = analyzer.analyze(csv)
        assert diag.primary_bottleneck == BottleneckType.BALANCED

    def test_tensor_core_detection(self):
        csv = self._make_csv({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
            "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active": 70.0,
        })
        analyzer = NcuBottleneckAnalyzer()
        diag = analyzer.analyze(csv)
        from analysis.ncu_bottleneck import ComputeUnit
        assert diag.primary_compute_unit == ComputeUnit.TENSOR_CORE

    def test_bank_conflict_detection(self):
        csv = self._make_csv({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 40.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
            "l1tex__data_bank_conflicts_pipe_lsu.sum": 50000.0,
        })
        analyzer = NcuBottleneckAnalyzer()
        diag = analyzer.analyze(csv)
        assert any("bank conflict" in b.lower() for b in diag.bottlenecks)

    def test_occupancy_gap(self):
        csv = self._make_csv({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 40.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
            "sm__maximum_warps_per_active_cycle_pct": 90.0,
            "sm__warps_active.avg.pct_of_peak_sustained_active": 40.0,
        })
        analyzer = NcuBottleneckAnalyzer()
        diag = analyzer.analyze(csv)
        assert diag.occupancy_gap == 50.0
        assert any("occupancy" in b.lower() for b in diag.bottlenecks)

    def test_report_generation(self):
        csv = self._make_csv({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
        })
        analyzer = NcuBottleneckAnalyzer()
        diag, report = analyzer.analyze_and_report(csv, "matmul_kernel")
        assert "Compute-Bound" in report
        assert "matmul_kernel" in report

    def test_empty_input(self):
        analyzer = NcuBottleneckAnalyzer()
        diag = analyzer.analyze("")
        assert diag.primary_bottleneck == BottleneckType.UNKNOWN


# ── Physical Consistency Validator ────────────────────────────

from analysis.consistency import PhysicalConsistencyValidator, EnvironmentFingerprintDetector


class TestPhysicalConsistency:
    def test_valid_results(self):
        results = {
            "l1_latency_cycles": 28.0,
            "l2_latency_cycles": 193.0,
            "dram_latency_cycles": 442.0,
            "max_global_mem_bandwidth_gb_s": 272.0,
            "actual_boost_clock_mhz": 2520.0,
            "l2_cache_size_kb": 32768.0,
            "max_shmem_per_block_kb": 100.0,
            "bank_conflict_penalty_cycles": 23.0,
        }
        v = PhysicalConsistencyValidator()
        report = v.validate(results)
        assert report.is_consistent
        assert len(report.violations) == 0

    def test_latency_hierarchy_violation(self):
        results = {
            "l1_latency_cycles": 200.0,  # L1 > L2 — violation!
            "l2_latency_cycles": 100.0,
            "dram_latency_cycles": 442.0,
        }
        v = PhysicalConsistencyValidator()
        report = v.validate(results)
        assert not report.is_consistent
        assert any("L1" in v and "L2" in v for v in report.violations)

    def test_bandwidth_too_high(self):
        results = {"max_global_mem_bandwidth_gb_s": 99999.0}
        v = PhysicalConsistencyValidator()
        report = v.validate(results)
        assert not report.is_consistent

    def test_negative_bank_conflict(self):
        results = {"bank_conflict_penalty_cycles": -5.0}
        v = PhysicalConsistencyValidator()
        report = v.validate(results)
        assert not report.is_consistent

    def test_cross_checks_populated(self):
        results = {
            "l1_latency_cycles": 28.0,
            "l2_latency_cycles": 193.0,
            "dram_latency_cycles": 442.0,
            "actual_boost_clock_mhz": 2520.0,
        }
        v = PhysicalConsistencyValidator()
        report = v.validate(results)
        assert len(report.cross_checks) > 0

    def test_none_values_skipped(self):
        results = {"l1_latency_cycles": None, "l2_latency_cycles": 193.0}
        v = PhysicalConsistencyValidator()
        report = v.validate(results)
        assert report.is_consistent  # Can't violate with only one value


class TestEnvironmentFingerprint:
    def test_no_tampering(self):
        fp = EnvironmentFingerprintDetector()
        result = fp.detect(
            probe_results={"actual_boost_clock_mhz": 2520.0},
            reported_clock_mhz=2520.0,
        )
        assert not result["tampering_detected"]

    def test_clock_mismatch(self):
        fp = EnvironmentFingerprintDetector()
        result = fp.detect(
            probe_results={"actual_boost_clock_mhz": 825.0},
            reported_clock_mhz=2520.0,
        )
        assert result["tampering_detected"]
        assert any("mismatch" in f.lower() for f in result["findings"])

    def test_shmem_mismatch(self):
        fp = EnvironmentFingerprintDetector()
        result = fp.detect(
            probe_results={"max_shmem_per_block_kb": 48.0},
            reported_shmem_kb=100,
        )
        assert result["tampering_detected"]

    def test_non_standard_clock(self):
        fp = EnvironmentFingerprintDetector()
        result = fp.detect(
            probe_results={"actual_boost_clock_mhz": 777.0},
        )
        assert result["tampering_detected"]
        assert any("non-standard" in f.lower() for f in result["findings"])


# ── Enhanced NCU Parser ──────────────────────────────────────

from parser.ncu_parser import parse_ncu_raw_page, parse_ncu_sol_section, extract_all_matching, summarize_ncu_metrics


class TestEnhancedNcuParser:
    def test_parse_raw_page(self):
        raw = """Section: GPU Speed Of Light Throughput
  sm__throughput.avg.pct_of_peak_sustained_elapsed    85.5  %
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed    30.2  %
"""
        result = parse_ncu_raw_page(raw)
        assert len(result) == 2
        assert result[0]["name"] == "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        assert result[0]["value"] == 85.5

    def test_parse_sol_section(self):
        output = "SOL Compute: 85.5 %\nSOL Memory: 30.2 %\n"
        sol = parse_ncu_sol_section(output)
        assert sol.get("sol_compute_pct") == 85.5
        assert sol.get("sol_memory_pct") == 30.2

    def test_extract_all_matching(self):
        parsed = [
            {"name": "dram__bytes.sum", "value": 1000.0},
            {"name": "dram__throughput.avg", "value": 80.0},
            {"name": "sm__throughput.avg", "value": 50.0},
        ]
        dram = extract_all_matching(parsed, r"dram__")
        assert len(dram) == 2

    def test_summarize_metrics(self):
        parsed = [
            {"name": "sm__throughput.avg", "value": 50.0},
            {"name": "dram__bytes.sum", "value": 1000.0},
            {"name": "sm__warps_active.avg", "value": 60.0},
        ]
        summary = summarize_ncu_metrics(parsed)
        assert summary["total_metrics"] == 3

    def test_suffix_match(self):
        parsed = [{"name": "kernel_0::dram__bytes.sum", "value": 42.0}]
        val = extract_ncu_metric(parsed, "dram__bytes.sum")
        assert val == 42.0

    def test_csv_with_avg_column(self):
        csv_text = '"Metric Name","Min","Max","Avg","Unit"\n"dram__bytes.sum","100","200","150","bytes"\n'
        result = parse_ncu_csv(csv_text)
        assert len(result) == 1
        assert result[0]["value"] == 150.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
