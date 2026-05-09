# MemXLife — GPU Hardware Profiling Agent System

Multi-agent system that autonomously measures GPU memory hierarchy characteristics through CUDA micro-benchmarks. Designed for environments where `cudaGetDeviceProperties` and spec tables may be unreliable.

## Quick Start

```bash
# Mock mode (no GPU required)
python3 main.py --mock tests/full_target_spec.json

# Real GPU execution
export ANTHROPIC_API_KEY=sk-...
python3 main.py tests/full_target_spec.json

# Run tests
python3 -m pytest tests/ -v
```

## Supported Metrics

| Metric | Unit | Method |
|--------|------|--------|
| `dram_latency_cycles` | cycles | Pointer chase (large stride) |
| `l1_latency_cycles` | cycles | Pointer chase (small working set) |
| `l2_latency_cycles` | cycles | Pointer chase (medium working set) |
| `l2_cache_size_kb` | KB | Cache sweep (latency knee detection) |
| `max_global_mem_bandwidth_gb_s` | GB/s | Streaming copy benchmark |
| `max_shmem_bandwidth_gb_s` | GB/s | Shared memory throughput test |
| `actual_boost_clock_mhz` | MHz | FMA instruction timing |
| `bank_conflict_penalty_cycles` | cycles | Conflict vs no-conflict delta |
| `max_shmem_per_block_kb` | KB | Binary search allocation probe |

## Architecture

```
main.py → Orchestrator
            ├── Scout Agent      (environment detection)
            ├── Planner Agent    (strategy selection)
            ├── Codegen Agent    (CUDA template + compile command)
            ├── Runner Agent     (compile + execute + ncu profiling)
            └── Analyzer Agent   (parse results, confidence scoring, physics calibration)
```

Each metric goes through the Planner → Codegen → Runner → Analyzer loop with up to 3 retries. The system uses CUDA micro-benchmark templates compiled with `-D` flags for parameterization, falling back to LLM-generated code when templates aren't available.

## Project Structure

```
├── main.py                  # CLI entry point
├── config.py                # Configuration (LLM, runtime)
├── core/
│   ├── orchestrator.py      # Main pipeline loop
│   ├── models.py            # Data classes
│   └── state.py             # Run state persistence
├── agents/
│   ├── scout.py             # GPU environment detection
│   ├── planner.py           # Strategy selection (LLM + heuristic fallback)
│   ├── codegen.py           # CUDA code generation from templates
│   ├── runner.py            # Compile + execute with adaptive retry
│   └── analyzer.py          # Result parsing + confidence scoring
├── probes/
│   ├── registry.py          # Template loader
│   └── templates/           # 7 CUDA micro-benchmark templates
├── parser/
│   ├── probe_parser.py      # RESULT:/UNIT:/METHOD: stdout parser
│   └── ncu_parser.py        # Nsight Compute CSV/text parser
├── knowledge/
│   ├── metrics_catalog.py   # 9 metric definitions + strategies
│   └── store.py             # JSON knowledge base
├── llm/
│   ├── client.py            # Anthropic/OpenAI with fallback
│   └── prompts.py           # Agent system/user prompts
├── audit/
│   └── logger.py            # JSONL log + Markdown report generator
└── tests/
    └── test_framework.py    # 40 unit tests
```

## Output

Each run produces a timestamped directory under `runs/` containing:
- `results.json` — final metric values
- `results_detailed.json` — values with confidence scores
- `knowledge.json` — full observation history
- `audit_report.md` — methodology and reasoning trace
- `audit_log.jsonl` — structured event log
