# Remote Public Eval Summary

Source log: [remote_public_eval_official_26e0fd0e_local.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_official_26e0fd0e_local.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.029646554030478,
  "tokens": 128,
  "tokens_per_second": 4317.533831028395
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.12549555697478354,
  "tokens": 656,
  "tokens_per_second": 5227.276692606842
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is the local fallback path when the official `outputs3` publishing chain is unstable.
