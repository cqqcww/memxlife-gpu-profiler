# Remote Public Eval Summary

Source log: [remote_public_eval_20260601.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_20260601.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.02707965299487114,
  "tokens": 128,
  "tokens_per_second": 4726.796167744213
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.1407301309518516,
  "tokens": 656,
  "tokens_per_second": 4661.4040331166825
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is a reliable local fallback while the official `outputs3` publishing path remains unstable.
