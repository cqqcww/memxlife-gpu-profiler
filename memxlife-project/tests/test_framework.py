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
from probes.registry import list_templates, load_template
from agents.codegen import _build_compile_command


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
                if strat.probe_template:
                    for key in strat.params:
                        assert key.isupper(), f"{name}/{strat.name}: param '{key}' not uppercase"

    def test_strategies_reference_valid_templates(self):
        templates = list_templates()
        for name, spec in build_catalog().items():
            for strat in spec.strategies:
                if strat.probe_template:
                    assert strat.probe_template in templates, \
                        f"{name}/{strat.name}: template '{strat.probe_template}' not found"

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


# ── Template registry ────────────────────────────────────────

class TestTemplateRegistry:
    def test_list_templates(self):
        templates = list_templates()
        assert len(templates) >= 7
        for name in ["latency_pointer_chase", "cache_size_sweep", "bandwidth_global",
                      "bandwidth_shared", "clock_frequency", "bank_conflict", "shmem_capacity"]:
            assert name in templates, f"Missing template: {name}"

    def test_load_template(self):
        code = load_template("latency_pointer_chase")
        assert "RESULT:" in code
        assert "main(" in code

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_template("nonexistent_xyz")

    def test_all_templates_have_result_and_main(self):
        for name in list_templates():
            code = load_template(name)
            assert "RESULT:" in code, f"{name} missing RESULT:"
            assert "main(" in code, f"{name} missing main()"


# ── Compile command builder ──────────────────────────────────

class TestCompileCommand:
    def test_basic_defines(self):
        cmd = _build_compile_command({"DATA_SIZE_BYTES": 268435456, "ITERATIONS": 1000, "WARMUP": 100})
        assert "-DDATA_SIZE_BYTES=268435456" in cmd
        assert "-DITERATIONS=1000" in cmd
        assert "-DWARMUP=100" in cmd
        assert cmd.startswith("nvcc")

    def test_skips_non_uppercase(self):
        cmd = _build_compile_command({"DATA_SIZE_BYTES": 256, "use_nvidia_smi": True})
        assert "-DDATA_SIZE_BYTES=256" in cmd
        assert "nvidia_smi" not in cmd

    def test_empty_params(self):
        cmd = _build_compile_command({})
        assert "nvcc" in cmd
        assert "-o probe" in cmd


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
            capture_output=True, text=True, timeout=30,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
