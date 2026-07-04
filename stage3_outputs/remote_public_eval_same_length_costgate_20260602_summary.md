# Remote Public Eval Summary

Source log: [remote_public_eval_same_length_costgate_20260602.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_same_length_costgate_20260602.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.030208203941583633,
  "tokens": 128,
  "tokens_per_second": 4237.259528819565
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.1235444531776011,
  "tokens": 656,
  "tokens_per_second": 5309.829645342057
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is the local fallback path when the official `outputs3` publishing chain is unstable.
