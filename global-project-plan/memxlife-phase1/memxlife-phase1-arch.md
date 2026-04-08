# MemXLife Phase 1 — Architecture Design Document

## 1. Overview

Multi-agent system for autonomous GPU hardware intrinsic profiling.
Reads `target_spec.json`, generates CUDA micro-benchmarks, executes them, analyzes results, outputs `results.json`.

## 2. Agents

### 2.1 Environment Scout
- Runs once at startup
- Detects: nvcc, ncu, nsys, nvidia-smi availability
- Collects baseline GPU info (marked as "untrusted" due to anti-cheat)
- Detects environment tampering (frequency locking, SM masking)
- Output: `EnvironmentProfile` → injected into all subsequent agent prompts

### 2.2 Planner
- LLM: Claude Sonnet (configurable)
- Input: target metric + KB state + environment profile + previous attempts
- Reads `metrics_catalog.py` for pre-defined probe strategies per metric
- Decides: which probe template, parameters, whether ncu is needed, whether cross-verification needed
- Manages retry logic: if Analyzer reports low confidence, Planner picks alternative strategy

### 2.3 Codegen
- LLM: Claude Opus (configurable)
- Input: probe strategy from Planner + template from registry
- Hybrid approach: pre-built CUDA templates + LLM parameterization/modification
- Output: .cu source, compile command, run command, expected output format
- Can modify template logic if standard approach fails

### 2.4 Runner
- No LLM — deterministic execution
- Steps: write source → nvcc compile → execute binary → optional ncu profiling
- Adaptive retry: on compile failure, analyzes error type, adjusts flags (e.g., lower compute capability)
- Captures: stdout, stderr, return code, ncu output
- Preserves all raw artifacts under runs/<run_id>/iterations/

### 2.5 Analyzer
- LLM: Claude Opus (configurable)
- Input: execution results + ncu output + KB + environment profile
- Extracts numeric values from probe output
- Confidence assessment (0.0-1.0)
- Calibration layer: physics sanity checks (L1 < L2 < DRAM, bandwidth <= theoretical peak)
- Anomaly detection (values outside physically reasonable range)
- Updates Knowledge Base
- Decides: sufficient confidence → finalize, or → request Planner retry

## 3. Core Infrastructure

### 3.1 Orchestrator
- Borrowed pattern from GPUProfiler: Task-based dispatch with retry
- Main loop: Scout → for each target: Planner→Codegen→Runner→Analyzer loop → aggregate results
- Max retries per metric (configurable, default 3)
- Timeout per metric (configurable)

### 3.2 Data Models
- `Task(id, kind, payload, result, status, error, attempts)`
- `AgentContext(run_id, run_dir, environment_profile)`
- `ProbeResult(metric_name, value, unit, confidence, method, evidence_refs)`
- `EnvironmentProfile(gpu_name, driver_version, cuda_version, tools, detected_frequency, detected_sm_count, trust_level)`

### 3.3 Knowledge Store
- JSON-based (inspired by GPUArchitect's JSONL claims)
- Stores: probe results, claims, environment observations
- Each entry: metric, value, confidence, method, timestamp, evidence path

### 3.4 Metrics Catalog
- Pre-defined strategies for each target metric
- Each metric has: description, ordered strategy list, cross-verify flag, tolerance
- Target metrics from spec: dram_latency_cycles, l1_latency_cycles, l2_latency_cycles, l2_cache_size_kb, max_shmem_bandwidth_gb_s, max_global_mem_bandwidth_gb_s, actual_boost_clock_mhz, bank_conflict_penalty_cycles

## 4. Probe Templates

Pre-built CUDA micro-benchmarks:
- `latency_pointer_chase.cu` — L1/L2/DRAM latency via pointer chasing (avoids prefetcher)
- `bandwidth_global.cu` — Global memory bandwidth (streaming read/write)
- `bandwidth_shared.cu` — Shared memory bandwidth
- `cache_size_sweep.cu` — L2 cache capacity detection (latency vs data size curve)
- `clock_frequency.cu` — Actual boost clock via FMA loop timing
- `bank_conflict.cu` — Shared memory bank conflict penalty measurement
- `common.cuh` — Shared utilities (timing, warmup, output formatting)

Output format: key=value pairs on stdout for deterministic parsing.

## 5. Audit Logger

- Generates Markdown report per run
- Records: environment profile, each probe attempt (strategy, code, output, analysis), final results
- Designed for the 30-point "Engineering Reasoning" evaluation rubric
- Shows cross-verification reasoning and anomaly detection

## 6. LLM Integration

- Abstraction layer supporting Claude (primary) and OpenAI (fallback)
- Per-agent model configuration
- Prompt templates in `llm/prompts.py`
- Token budget management
- Resilient backend: auto-fallback on API failure

## 7. Development Strategy

- Phase A: Framework skeleton on Mac (mock Runner)
- Phase B: CUDA probe templates (can write/review on Mac, test on GPU server)
- Phase C: Integration on GPU server
- Phase D: Testing, tuning, audit report polish
