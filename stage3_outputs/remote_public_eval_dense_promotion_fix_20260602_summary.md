# Remote Public Eval Summary

Source log: [remote_public_eval_dense_promotion_fix_20260602.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_dense_promotion_fix_20260602.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.029835852095857263,
  "tokens": 128,
  "tokens_per_second": 4290.140586190027
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.1234217369928956,
  "tokens": 656,
  "tokens_per_second": 5315.109120833072
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is the local fallback path when the official `outputs3` publishing chain is unstable.
