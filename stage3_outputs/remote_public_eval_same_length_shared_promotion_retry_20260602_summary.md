# Remote Public Eval Summary

Source log: [remote_public_eval_same_length_shared_promotion_retry_20260602.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_same_length_shared_promotion_retry_20260602.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.02948475885204971,
  "tokens": 128,
  "tokens_per_second": 4341.225941249363
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.12379900785163045,
  "tokens": 656,
  "tokens_per_second": 5298.911609907223
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is the local fallback path when the official `outputs3` publishing chain is unstable.
