# Report 3

## Overview

This Phase 3 submission focuses on improving decode throughput while preserving correctness in the provided lightweight LLM serving runtime.

The runtime entrypoint is [run.sh](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/run.sh). During execution, `run.sh` materializes `/workspace/engine.py` from the selected local source variant using [scripts/render_phase3_engine.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/scripts/render_phase3_engine.py). The current default rendered variant is `current_best`, sourced from [phase3_engine_sources/current_best_engine.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase3_engine_sources/current_best_engine.py).

The current kept implementation in [engine.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/workspace/engine.py) combines:

- shared-varlen decode K/V updates via batched `index_put_`
- fused `torch.nn.functional.rms_norm` fast path when available
- conservative same-length shared-batch promotion during decode (`batch >= 4`, `position >= 16`)

The strategy is intentionally conservative because several more aggressive decode optimizations improved one public metric while regressing overall decode stability or correctness.

## Experiment Process

Local and remote experiments were run inside the course GPU containers using:

- `bash scripts/run_public_tests.sh`
- `python3 evaluator/benchmark_breakdown.py --engine workspace/engine.py --model-config target/model_config.json --weight-dir target/weights --device auto`

The detailed experiment log is summarized in:

- [stage3_outputs/remote_breakdown_decode_experiments_20260601.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_decode_experiments_20260601.md)

Key findings:

1. Batched `index_put_` for shared-varlen decode materially improved mixed decode throughput.
2. `F.rms_norm` provided a small but consistent speedup and was kept.
3. More aggressive heuristics such as unconditional promotion, gated repack, grouped-query decode rewriting, and pre-expanded shared KV caches were rejected because they either regressed public throughput or failed stress correctness.

## Current Best Local Fallback Result

The current best retained public fallback run is:

- source: [stage3_outputs/remote_public_eval_same_length_shared_promotion_retry_20260602_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_same_length_shared_promotion_retry_20260602_summary.md)

Metrics:

- public throughput: `4341.225941249363 tokens/s`
- public mixed throughput: `5298.911609907223 tokens/s`
- correctness smoke: passed
- stress correctness: passed

For comparison, the earlier stable official-local fallback was:

- source: [stage3_outputs/remote_public_eval_official_26e0fd0e_local_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_official_26e0fd0e_local_summary.md)

Metrics:

- public throughput: `4317.533831028395 tokens/s`
- public mixed throughput: `5227.276692606842 tokens/s`

This means the current kept version improved the local fallback public metrics by roughly:

- throughput: about `+0.55%`
- mixed throughput: about `+1.37%`

## Official Submission Notes

Recent official submission mission ids include:

- `26e0fd0e7af23fd18b33fa614d190232`
- `8ce297580454bada353095e830250317`

The official `outputs3` publishing service was unstable during this work, so local fallback evaluation inside course GPU containers was used as the primary comparison method when public artifacts were unavailable.

## Compliance Notes

This report is placed under `/workspace` as `report3.md`.

The runtime log is emitted to:

- `/workspace/output3.log`

For compatibility with earlier wording and scripts, the same run also writes:

- `/workspace/results.log`

The engine artifact consumed by evaluation is generated into:

- `/workspace/engine.py`
