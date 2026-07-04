# Remote Public Eval Summary

Source log: [remote_public_eval_gated_repack_20260602.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_gated_repack_20260602.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.031698038103058934,
  "tokens": 128,
  "tokens_per_second": 4038.104805850672
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.12976182904094458,
  "tokens": 656,
  "tokens_per_second": 5055.415794062275
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is the local fallback path when the official `outputs3` publishing chain is unstable.
